"""Codebook usage statistics for the lvl1_vqgan residual quantizer.

Dead-code count (the only usage metric train.py logs) misses skew: a level
where 200 of 2048 codes absorb 90% of assignments reports zero dead codes but
is functionally a 200-entry table. This script runs encoder+quantizer only
(no decoder, no losses, CPU) over cached slices and reports the assignment
distribution per RQ level.

The module's own `usage` buffer cannot answer this: it is `persistent=False`
(absent from the checkpoint) and is a halving EMA (`usage /= 2` every call),
so it reflects roughly the last batch -- enough for dead-code revival, useless
as a distribution estimate.

Usage:
    CKPT=saved_20260822_batch64/lvl1_vqgan_last.ckpt uv run python codebook_usage.py
"""

from __future__ import annotations

import glob
import math
import os
import time

import torch

from render_samples import CACHE, SLICE, load_module
from train import (
    build_learning_params,
    build_loss_aggregator,
    build_optimizer_cfg,
    build_scheduler_cfg,
)


def collect_indices(
    module: torch.nn.Module,
    slices_per_track: int = 40,
    batch: int = 8,
    max_tracks: int | None = None,
) -> torch.Tensor:
    """Tokenize evenly-spaced slices from every cached track.

    Args:
      module: loaded LightningModule exposing `.model.tokenize`.
      slices_per_track (int): slices sampled per track file.
      batch (int): slices per forward pass.
      max_tracks (int | None): cap on track files; None = all.

    Returns:
      torch.Tensor: (N, T, R) int64 code indices.
    """
    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))[:max_tracks]
    chunks: list[torch.Tensor] = []
    for n_done, path in enumerate(files, 1):
        pool = torch.load(path, map_location="cpu")
        step = max(1, pool.shape[0] // slices_per_track)
        picked = pool[::step][:slices_per_track].reshape(-1, 1, SLICE).float()
        for i in range(0, picked.shape[0], batch):
            with torch.no_grad():
                chunks.append(module.model.tokenize(picked[i : i + batch]))
        print(f"  [{n_done}/{len(files)}] {os.path.basename(path)[:48]}", flush=True)
    return torch.cat(chunks, dim=0)


def level_stats(counts: torch.Tensor) -> dict[str, float]:
    """Distribution statistics for one level's assignment histogram.

    Args:
      counts (torch.Tensor): (num_tokens,) assignment counts.

    Returns:
      dict[str, float]: dead count, entropy in bits, normalized entropy,
        perplexity, and the assignment mass held by the top 10% of codes.
    """
    n = counts.numel()
    total = float(counts.sum())
    p = counts.double() / total
    nz = p[p > 0]
    entropy = float(-(nz * nz.log2()).sum())
    top = int(math.ceil(0.1 * n))
    return {
        "dead": float((counts == 0).sum()),
        "entropy_bits": entropy,
        "norm_entropy": entropy / math.log2(n),
        "perplexity": 2.0**entropy,
        "top10pct_mass": float(counts.sort(descending=True).values[:top].sum() / total),
    }


def main() -> None:
    """Load a checkpoint, tokenize cached audio, print per-level usage stats."""
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    ckpt = os.environ.get("CKPT", "saved_20260822_batch64/lvl1_vqgan_last.ckpt")
    num_tokens = int(os.environ.get("NUM_TOKENS", "2048"))

    module = load_module(
        ckpt,
        build_learning_params(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        build_loss_aggregator(),
        num_tokens=num_tokens,
        per_level_codebooks=True,
    )

    t0 = time.time()
    idx = collect_indices(
        module,
        slices_per_track=int(os.environ.get("SLICES_PER_TRACK", "40")),
        max_tracks=int(os.environ["MAX_TRACKS"]) if "MAX_TRACKS" in os.environ else None,
    )
    n_slices, n_frames, n_levels = idx.shape
    print(
        f"\n{ckpt}\n{n_slices} slices x {n_frames} frames = "
        f"{n_slices * n_frames} draws/level  ({time.time() - t0:.0f}s)\n"
    )

    hdr = f"{'lvl':>4} {'dead':>6} {'H_bits':>8} {'H_norm':>8} {'perplex':>9} {'top10%':>8}"
    print(hdr)
    total_perplex = 0.0
    for lvl in range(n_levels):
        counts = torch.bincount(idx[..., lvl].flatten(), minlength=num_tokens)
        s = level_stats(counts)
        total_perplex += s["perplexity"]
        print(
            f"{lvl:>4} {s['dead']:6.0f} {s['entropy_bits']:8.3f} "
            f"{s['norm_entropy']:8.3f} {s['perplexity']:9.1f} {s['top10pct_mass']:8.3f}"
        )
    print(
        f"\nmax H per level = {math.log2(num_tokens):.1f} bits "
        f"(perplexity {num_tokens})\n"
        f"effective codes {total_perplex:.0f} / {n_levels * num_tokens} nominal"
    )


if __name__ == "__main__":
    main()
