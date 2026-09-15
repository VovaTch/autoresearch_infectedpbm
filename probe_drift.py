"""
Does the generation drift away from the track as it runs?

Ear verdict 2026-09-08 on the copy-synthesis clips: the round trip (B) is a bit
distorted but acceptable, while the AR generation (C) "sounds like it's losing
its grip, slowly diverging from the original track". That is the signature of
exposure bias -- the model trains on ground-truth history and at inference feeds
on its own samples, so errors compound along the time axis.

Log-spectral RMSE could not see it: averaged over a whole clip, a generation
that starts right and ends wrong scores the same as one uniformly mediocre. This
measures the thing directly -- distance from the track's own timbre as a
function of ELAPSED TIME within the clip.

Three series per track, all from the same span and LUFS-matched:

  real        the source tokens decoded            expected flat, the floor
  roundtrip   identical to real here, kept for symmetry with the A/B/C probe
  generated   prompted AR sample                   rising == drift

A rising generated curve against a flat real curve is drift, and the slope is
the number to optimise. Clips are long on purpose: 12 s was barely enough to
hear it.

Usage:
  uv run python probe_drift.py
  uv run python probe_drift.py --seconds 60 --window 16384
"""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ab_harness.config import load_config
from ab_harness.model.audio import normalize_lufs, to_int16
from ab_harness.worker.generator import SampleRequest
from ab_harness.worker.service import GenerationService
from probe_cross_track import profile_distance, timbre_profile

REPO = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_ab.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tracks", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--bin-seconds", type=float, default=5.0)
    parser.add_argument("--prompt-sec", type=float, default=3.0)
    parser.add_argument("--start-frac", type=float, default=0.35)
    parser.add_argument("--window", type=int, default=16384)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--out", default="renders_drift")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.bank.root = tempfile.mkdtemp(prefix="probe_drift_")
    if args.checkpoint:
        cfg.generator.checkpoint = args.checkpoint
    cfg.generator.window_frames = args.window

    service = GenerationService(cfg)
    service.load()
    corpus, tracks = service.corpus(), service._tracks
    gen, dec = service._generator, service._decoder
    assert gen is not None and dec is not None
    sr = dec.sample_rate

    frames = int(args.seconds * cfg.sampler.fps)
    n_prompt = int(args.prompt_sec * cfg.sampler.fps)
    rng = random.Random(0)
    chosen = rng.sample([t for t in corpus if t.num_frames > frames * 2 + 1], args.tracks)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    def render(tag: str, toks: np.ndarray) -> np.ndarray:
        """Decode, LUFS-match, write, return the waveform."""
        audio = normalize_lufs(dec.decode(toks), sr, cfg.session.target_lufs)
        pcm = to_int16(audio).astype(np.float32) / 32768.0
        torchaudio.save(str(out / f"{tag}.wav"), torch.from_numpy(pcm)[None], sr)
        return pcm

    def curve(pcm: np.ndarray, ref: np.ndarray) -> list[float]:
        """
        Timbre distance to `ref` in successive bins.

        Args:
          pcm (np.ndarray): (N,) waveform to analyse.
          ref (np.ndarray): (bands,) reference timbre profile.

        Returns:
          list[float]: one distance per bin of --bin-seconds.
        """
        step = int(args.bin_seconds * sr)
        return [
            profile_distance(timbre_profile(pcm[i : i + step], sr), ref)
            for i in range(0, len(pcm) - step + 1, step)
        ]

    print(f"checkpoint {cfg.generator.checkpoint}")
    print(f"{args.seconds:.0f} s clips, {args.bin_seconds:.0f} s bins, "
          f"prompt {args.prompt_sec:.0f} s, cfg {args.cfg}, seeds {args.seeds}\n")

    real_curves: list[list[float]] = []
    gen_curves: list[list[float]] = []
    for t in chosen:
        idx = t.track_idx
        start = int(max(0, min(t.num_frames - frames - 1, t.num_frames * args.start_frac)))
        span = tracks[idx].tokens[start : start + frames].numpy().astype(np.int16)

        # The track's own timbre, taken from a DIFFERENT span so the reference is
        # the track's sound in general, not this particular passage.
        other = max(0, start - frames - 1) if start > frames else start + frames
        other = min(other, t.num_frames - frames - 1)
        ref_pcm = render(f"t{idx}_REF", tracks[idx].tokens[other : other + frames].numpy().astype(np.int16))
        ref = timbre_profile(ref_pcm, sr)

        real_pcm = render(f"t{idx}_real", span)
        real_curves.append(curve(real_pcm, ref))

        prompt = torch.from_numpy(span[:n_prompt].astype(np.int64)).long()
        reqs = [
            SampleRequest(
                track_idx=idx,
                style=tracks[idx].style[rng.randrange(tracks[idx].style.shape[0])],
                frames=frames,
                prompt=prompt,
                temperature=1.0,
                top_k=250,
                top_p=0.0,
                cfg_strength=args.cfg,
                seed=s,
            )
            for s in args.seeds
        ]
        for req, code in zip(reqs, gen.sample_batch(reqs)):
            pcm = render(f"t{idx}_gen_s{req.seed}", code.numpy().astype(np.int16))
            gen_curves.append(curve(pcm, ref))
        print(f"done t{idx}  {t.track_name[:44]}", flush=True)

    n = min(min(len(c) for c in real_curves), min(len(c) for c in gen_curves))
    real = np.array([c[:n] for c in real_curves])
    gene = np.array([c[:n] for c in gen_curves])

    print(f"\n{'t (s)':>7} {'real':>16} {'generated':>18}")
    for b in range(n):
        lo = b * args.bin_seconds
        print(
            f"{lo:>4.0f}-{lo + args.bin_seconds:<3.0f} "
            f"{real[:, b].mean():7.4f} +-{real[:, b].std(ddof=1):6.4f} "
            f"{gene[:, b].mean():8.4f} +-{gene[:, b].std(ddof=1):6.4f}"
        )

    x = np.arange(n) * args.bin_seconds
    for label, arr in (("real", real), ("generated", gene)):
        slopes = [np.polyfit(x, row, 1)[0] for row in arr]
        m = float(np.mean(slopes))
        se = float(np.std(slopes, ddof=1) / np.sqrt(len(slopes))) if len(slopes) > 1 else float("nan")
        print(f"\n{label:<10} slope {m:+.5f} per s   t={m / se if se else float('nan'):.2f}  "
              f"(first bin {arr[:, 0].mean():.4f} -> last {arr[:, -1].mean():.4f})")
    print("\nA positive generated slope against a flat real slope is drift.")

    service.close()


if __name__ == "__main__":
    main()
