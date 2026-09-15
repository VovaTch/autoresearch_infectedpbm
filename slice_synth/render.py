"""
Headless renders from the synthesizer, without opening a window.

Same service the app drives, so what comes out of here is what comes out of
there. Useful for three things the GUI is bad at: smoke-testing a checkpoint over
ssh, filling a directory to listen through later, and reproducing a saved recipe.

Usage:
  uv run python -m slice_synth.render --tracks 3 7 --styles window random
  uv run python -m slice_synth.render --tracks 0 --seconds 30 --cfg 8 --reference
  uv run python -m slice_synth.render --replay renders_synth/20260908_*.json
  uv run python -m slice_synth.render --tracks 3 14 --together --walk windows --period 512
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from slice_synth.config import resolve_config
from slice_synth.model.library import load_spec, save_render
from slice_synth.model.tokens import stats_line
from slice_synth.model.types import (
    PromptSpec,
    Render,
    RenderSpec,
    StyleSpec,
    build_variants,
)
from slice_synth.worker.client import InProcessRenderProducer
from slice_synth.worker.protocol import RenderProgress, RenderResult, WorkerReady
from slice_synth.worker.service import SynthService

STYLE_KINDS = ("window", "random", "jitter", "interp", "null")


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_synth.yaml")
    parser.add_argument("--checkpoint", default="", help="default: the config's pick")
    parser.add_argument("--tracks", type=int, nargs="+", default=[0])
    parser.add_argument("--styles", nargs="+", default=["window"], choices=STYLE_KINDS)
    parser.add_argument(
        "--style-track", type=int, default=-1, help="default: the render's own track"
    )
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--prompt-sec", type=float, default=None, help="0 = cold start")
    parser.add_argument("--start-frac", type=float, default=0.35)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--noise", type=float, default=0.25, help="jitter strength")
    parser.add_argument("--mix", type=float, default=0.5, help="interp position")
    parser.add_argument(
        "--reference", action="store_true", help="also decode real tokens"
    )
    parser.add_argument(
        "--together",
        action="store_true",
        help="all --tracks as coexisting ids in ONE clip",
    )
    parser.add_argument("--walk", default="none", choices=("none", "windows", "random"))
    parser.add_argument(
        "--period", type=int, default=512, help="frames per style segment"
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--replay", nargs="+", default=[], help="re-render saved .json recipes"
    )
    return parser.parse_args()


def style_for(
    kind: str,
    track_idx: int,
    seed: int,
    args: argparse.Namespace,
    corpus: dict[int, int],
) -> StyleSpec | None:
    """
    Build one style recipe for a track.

    Args:
      kind (str): one of STYLE_KINDS.
      track_idx (int): the track being generated.
      seed (int): seed for the recipe's own draws.
      args (argparse.Namespace): CLI arguments.
      corpus (dict[int, int]): track id -> number of style windows, for picking
        an interp partner that actually exists.

    Returns:
      StyleSpec | None: the recipe, or None to null the style stream.
    """
    if kind == "null":
        return None
    source = args.style_track if args.style_track >= 0 else track_idx
    partner = next((t for t in sorted(corpus) if t != source), source)
    return StyleSpec(
        kind=kind,  # type: ignore[arg-type]
        track_idx=source,
        window=-1,
        track_b=partner,
        window_b=-1,
        mix=args.mix,
        noise=args.noise,
        seed=seed,
        walk=args.walk,
        period=args.period,
    )


def collect(
    producer: InProcessRenderProducer,
) -> tuple[list[RenderResult], WorkerReady | None]:
    """
    Args:
      producer (InProcessRenderProducer): the synchronous producer.

    Returns:
      tuple[list[RenderResult], WorkerReady | None]: results and the last
        readiness notice, with progress printed as it goes.
    """
    results: list[RenderResult] = []
    ready: WorkerReady | None = None
    for message in producer.poll():
        if isinstance(message, RenderResult):
            results.append(message)
        elif isinstance(message, WorkerReady):
            ready = message
        elif isinstance(message, RenderProgress) and message.step % 2048 == 0:
            print(f"    {message.step}/{message.total} positions", flush=True)
    return results, ready


def main() -> int:
    """
    Returns:
      int: process exit code.
    """
    args = parse_args()
    cfg = resolve_config(args.config)
    if args.checkpoint:
        cfg.generator.checkpoint = args.checkpoint
    out_root = Path(args.out).expanduser() if args.out else cfg.output_root

    service = SynthService(cfg)
    producer = InProcessRenderProducer(service)
    producer.start()
    _, ready = collect(producer)
    if ready is None or not ready.ok:
        print(f"worker failed to load: {ready.error if ready else 'no reply'}")
        return 1

    fps = float(ready.meta["frames_per_second"])
    num_tokens = int(ready.meta["num_tokens"])
    by_idx = {t.track_idx: t for t in ready.tracks}
    windows = {t.track_idx: len(t.style_bounds) for t in ready.tracks}
    print(f"checkpoint {ready.checkpoint}  window {ready.window_frames} frames")
    print(f"corpus {len(ready.tracks)} tracks  {fps:.3f} fps\n")

    if args.replay:
        specs = [
            load_spec(Path(p))[0]
            for pattern in args.replay
            for p in sorted(glob.glob(pattern))
        ]
        if not specs:
            print("no recipes matched")
            return 1
    else:
        seconds = args.seconds if args.seconds is not None else cfg.ui.seconds
        prompt_sec = (
            args.prompt_sec if args.prompt_sec is not None else cfg.ui.prompt_seconds
        )
        frames = int(seconds * fps)
        specs = []
        known = [t for t in args.tracks if t in by_idx]
        for missing in (t for t in args.tracks if t not in by_idx):
            print(f"no track {missing} in this corpus; skipping")
        # together: one cell, primed from and labelled by the first track
        cells = [known[:1]] if args.together and known else [[t] for t in known]
        for cell in cells:
            track_idx = cell[0]
            track = by_idx[track_idx]
            start = int(
                max(
                    0,
                    min(
                        track.num_frames - frames - 1,
                        track.num_frames * args.start_frac,
                    ),
                )
            )
            base = RenderSpec(
                n_frames=frames,
                prompt=PromptSpec(
                    kind="corpus" if prompt_sec > 0 else "none",
                    track_idx=track_idx,
                    start_frame=start,
                    seconds=prompt_sec,
                ),
                cfg_strength=args.cfg if args.cfg is not None else cfg.ui.cfg_strength,
                temperature=(
                    args.temperature
                    if args.temperature is not None
                    else cfg.ui.temperature
                ),
                top_k=args.top_k if args.top_k is not None else cfg.ui.top_k,
                top_p=args.top_p if args.top_p is not None else cfg.ui.top_p,
                checkpoint=ready.checkpoint,
            )
            styles = [
                style_for(k, track_idx, args.seeds[0], args, windows)
                for k in args.styles
            ]
            tracks = known if args.together else [track_idx]
            specs += build_variants(
                base, tracks, styles, args.seeds, args.reference, args.together
            )

    print(f"rendering {len(specs)} variants -> {out_root}")
    producer.submit(specs)
    results, _ = collect(producer)

    failed = 0
    for result in results:
        if not result.ok:
            failed += 1
            print(f"  FAILED {result.spec.label()}: {result.error}")
            continue
        assert result.pcm is not None and result.tokens is not None
        saved = save_render(
            Render(
                spec=result.spec,
                tokens=result.tokens,
                pcm=result.pcm,
                sample_rate=result.sample_rate,
                style_used=result.style_used,
                fill=result.fill,
            ),
            out_root,
        )
        peak = float(np.abs(result.pcm).max()) / 32768.0
        print(
            f"  {saved.stem}.wav  {result.pcm.size / result.sample_rate:5.1f}s  "
            f"peak {peak:.3f}  fill {result.fill:.2f}"
        )
        print(f"    {stats_line(result.tokens, num_tokens, fps)}")

    producer.close()
    print(f"\nwrote {len(results) - failed} renders, {failed} failed")
    return 1 if failed and not (len(results) - failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
