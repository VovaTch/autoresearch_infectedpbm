"""
Conditioning ladder: does MORE conditioning make the model sound worse?

The A/B harness cannot answer this. PairSampler draws one Conditioning per pair
and varies only Sampling.seed, so every judgement in the bank holds conditioning
constant and no within-pair contrast on it exists. The only cross-cell signals
are ~22 anchors and ~17 repeats, far too thin to split eight ways.

This probe inverts that: one track, one span, ONE SEED, and the conditioning
swept across it. Every audible difference between two outputs is therefore
caused by conditioning alone.

Deliberately NOT banked as preference data. pair_sampler's own docstring warns
that pairs differing in conditioning "teach a preference model the prompt
distribution rather than quality", so the service is pointed at a scratch bank
root and the real one is left untouched.

Usage:
  uv run python probe_conditioning.py
  uv run python probe_conditioning.py --track deeply_disturbed --seconds 10
  uv run python probe_conditioning.py --cfg 0 2 3 --seed 4242
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ab_harness.config import load_config
from ab_harness.model.audio import measure_lufs
from ab_harness.model.pair_sampler import pick_style_window
from ab_harness.model.types import ClipSpec, Conditioning, Sampling, Tier
from ab_harness.worker.service import GenerationService

REPO = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_ab.yaml")
    parser.add_argument("--checkpoint", default="", help="default: config's auto pick")
    parser.add_argument("--track", default=None, help="substring; default = track 0")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--start-frac", type=float, default=0.35)
    parser.add_argument("--prompt-sec", type=float, default=3.0)
    parser.add_argument(
        "--cfg",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 3.0],
        help="guidance strengths; 0.0 is plain conditional sampling",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1234],
        help="every cell is generated at each of these seeds",
    )
    parser.add_argument("--out", default="renders_probe_cond")
    return parser.parse_args()


def build_grid(
    track_idx: int,
    start: int,
    frames: int,
    style_window: int,
    prompt_frames: int,
    strengths: list[float],
    seeds: list[int],
    checkpoint: str,
) -> list[ClipSpec]:
    """
    Build the full conditioning grid at one fixed seed and span.

    Args:
      track_idx (int): corpus index of the track.
      start (int): span start frame.
      frames (int): span length in tokenizer frames.
      style_window (int): style window index, disjoint from the span.
      prompt_frames (int): prompt length for the prompted half of the grid.
      strengths (list[float]): cfg strengths to sweep.
      seeds (list[int]): seeds every cell is generated at.
      checkpoint (str): checkpoint tag to record on each spec.

    Returns:
      list[ClipSpec]: one generated spec per cell, plus the real-token
        reference as the ceiling.
    """
    specs: list[ClipSpec] = []
    combos = [
        (use_id, use_style, prompted, cfg)
        for seed in seeds
        for use_id in (False, True)
        for use_style in (False, True)
        for prompted in (False, True)
        for cfg in strengths
    ]
    seed_of = [
        seed
        for seed in seeds
        for _ in range(2 * 2 * 2 * len(strengths))
    ]
    for (use_id, use_style, prompted, cfg), seed in zip(combos, seed_of):
        cond = Conditioning(
            track_idx=track_idx,
            start_frame=start,
            use_track_id=use_id,
            use_style=use_style,
            style_window=style_window if use_style else -1,
            prompt_frames=prompt_frames if prompted else 0,
            cfg_strength=cfg,
        )
        tag = (
            f"id{int(use_id)}_st{int(use_style)}"
            f"_pr{int(prompted)}_cfg{cfg:g}_s{seed}"
        )
        specs.append(
            ClipSpec(
                item_id=f"probe_{tag}",
                tier=Tier.BULK,
                group_id=f"probe_{tag}",
                n_frames=frames,
                conditioning=cond,
                sampling=Sampling(
                    seed=seed, temperature=1.0, top_k=250, top_p=0.0
                ),
                generator="ar",
                checkpoint=checkpoint,
            )
        )
    specs.append(
        ClipSpec(
            item_id="probe_REFERENCE",
            tier=Tier.BULK,
            group_id="probe_REFERENCE",
            n_frames=frames,
            conditioning=Conditioning(track_idx=track_idx, start_frame=start),
            sampling=Sampling(seed=0, temperature=0.0),
            generator="reference",
            checkpoint="",
        )
    )
    return specs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.checkpoint:
        cfg.generator.checkpoint = args.checkpoint

    # Scratch bank: produce_many banks every clip it makes, and these must never
    # become ratable preference material.
    scratch = Path(tempfile.mkdtemp(prefix="probe_cond_"))
    cfg.bank.root = str(scratch)

    service = GenerationService(cfg)
    service.load()
    corpus = service.corpus()

    track = corpus[0]
    if args.track:
        hits = [t for t in corpus if args.track.lower() in t.track_name.lower()]
        if not hits:
            raise SystemExit(f"no track matching {args.track!r}")
        track = hits[0]

    fps = cfg.sampler.fps
    frames = int(args.seconds * fps)
    start = int(
        max(0, min(track.num_frames - frames - 1, track.num_frames * args.start_frac))
    )
    prompt_frames = min(int(args.prompt_sec * fps), max(0, frames - 1))
    # Section 11.3: the style window must be DISJOINT from the span, or the
    # conditioning carries the answer and flatters itself.
    style_window = pick_style_window(
        track.style_bounds, start, start + frames, random.Random(args.seeds[0])
    )

    print(f"checkpoint  {cfg.generator.checkpoint}")
    print(f"track       [{track.track_idx}] {track.track_name}")
    print(
        f"span        frames {start}-{start + frames} "
        f"({args.seconds:.0f}s @ {fps:.1f} fps)"
    )
    print(f"style win   {style_window}   prompt {prompt_frames} frames")
    print(f"seeds       {args.seeds}  (every cell run at each)")
    print(f"scratch     {scratch}\n")

    specs = build_grid(
        track.track_idx,
        start,
        frames,
        style_window,
        prompt_frames,
        list(args.cfg),
        list(args.seeds),
        cfg.generator.checkpoint,
    )
    print(f"producing {len(specs)} clips ({len(specs) - 1} cells + 1 reference)\n")

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    # Chunked like bake_ab_bank: produce_many groups every same-length spec into
    # ONE sampling batch, and its except-clause fails the whole group together,
    # so an oversized grid loses every clip rather than the last few.
    size = max(1, cfg.generator.max_batch)
    results = []
    for start_i in range(0, len(specs), size):
        results += service.produce_many(specs[start_i : start_i + size])

    print(f"{'cell':<26} {'fill':>6} {'LUFS':>7} {'peak':>6}")
    for res in results:
        tag = res.spec.item_id.replace("probe_", "")
        if not res.ok or res.pcm is None:
            print(f"{tag:<26}  FAILED: {res.error}")
            continue
        pcm = res.pcm.astype(np.float32) / 32768.0
        torchaudio.save(
            str(out / f"{tag}.wav"),
            torch.from_numpy(pcm)[None],
            res.sample_rate,
        )
        lufs = measure_lufs(pcm, res.sample_rate)
        print(f"{tag:<26} {res.fill:6.3f} {lufs:7.1f} {np.abs(pcm).max():6.3f}")

    service.close()
    shutil.rmtree(scratch, ignore_errors=True)
    ok = sum(1 for r in results if r.ok and r.pcm is not None)
    print(f"\nwrote {ok} wavs ({len(results) - ok} failed) -> {out}")
    print("REFERENCE.wav is the tokenizer ceiling; everything else shares its seed.")


if __name__ == "__main__":
    main()
