"""Rate-distortion diagnostics for the lvl1_vqgan residual quantizer.

Answers whether alignment/commitment loss is capacity-bound, and if so which
lever pays: more RQ levels (depth) or a wider codebook (width).

Two measurements, both encoder-only on CPU:

  1. Residual decay. Mean square of the RQ residual after each level, relative
     to the encoder output. A steep level 2->3 drop means a 4th level still has
     structure to quantize; a flat one means depth is exhausted.

  2. Intrinsic dimension of z_e, via the participation ratio of its covariance
     spectrum, D_eff = (sum lambda)^2 / sum lambda^2. VQ distortion falls as
     N^(-2/D_eff), so this converts "double the codebook" into a predicted MSE
     reduction. token_dim is 1024, but z_e occupies far fewer directions, and
     the payoff of width is set by the smaller number.

Usage:
    CKPT=saved_20260822_batch64/lvl1_vqgan_last.ckpt uv run python rq_residual.py
"""

from __future__ import annotations

import glob
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


def measure(
    module: torch.nn.Module,
    slices_per_track: int = 16,
    batch: int = 8,
    max_tracks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate RQ residual energy per level and the z_e covariance.

    Args:
      module: loaded LightningModule exposing `.model.encode` and the RQ codebook.
      slices_per_track (int): slices sampled per cached track file.
      batch (int): slices per forward pass.
      max_tracks (int | None): cap on track files; None = all.

    Returns:
      tuple: (R+1,) float64 summed square error per stage (index 0 = z_e energy),
        one (token_dim, token_dim) float64 scatter matrix per level holding the
        input that level quantizes, and the frame count.
    """
    cb = module.model.vq_module.vq_codebook
    n_levels = cb._num_rq_steps
    err = torch.zeros(n_levels + 1, dtype=torch.float64)
    scatter: list[torch.Tensor] | None = None
    n_frames = 0

    files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))[:max_tracks]
    for n_done, path in enumerate(files, 1):
        pool = torch.load(path, map_location="cpu")
        step = max(1, pool.shape[0] // slices_per_track)
        picked = pool[::step][:slices_per_track].reshape(-1, 1, SLICE).float()
        for i in range(0, picked.shape[0], batch):
            with torch.no_grad():
                z_e = module.model.encode(picked[i : i + batch])  # (B, C, T)
                if scatter is None:
                    d = z_e.shape[1]
                    scatter = [torch.zeros(d, d, dtype=torch.float64)] * 0 + [
                        torch.zeros(d, d, dtype=torch.float64) for _ in range(n_levels)
                    ]
                n_frames += z_e.shape[0] * z_e.shape[2]

                err[0] += float((z_e.double() ** 2).sum())
                x_res = z_e.clone()
                for level in range(n_levels):
                    # scatter[level] is what level `level` is asked to quantize
                    flat = x_res.transpose(1, 2).reshape(-1, x_res.shape[1]).double()
                    scatter[level] += flat.T @ flat
                    z_q, _ = cb._quantize_level(x_res, level, True)
                    x_res = x_res - z_q.squeeze(1)
                    err[level + 1] += float((x_res.double() ** 2).sum())
        print(f"  [{n_done}/{len(files)}] {os.path.basename(path)[:48]}", flush=True)
    assert scatter is not None
    return err, scatter, n_frames


def main() -> None:
    """Load a checkpoint, measure RQ decay and z_e intrinsic dimension, print both."""
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    ckpt = os.environ.get("CKPT", "saved_20260822_batch64/lvl1_vqgan_last.ckpt")

    module = load_module(
        ckpt,
        build_learning_params(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        build_loss_aggregator(),
        per_level_codebooks=True,
    )

    t0 = time.time()
    err, scatter, n_frames = measure(
        module,
        slices_per_track=int(os.environ.get("SLICES_PER_TRACK", "16")),
        max_tracks=int(os.environ["MAX_TRACKS"]) if "MAX_TRACKS" in os.environ else None,
    )
    print(f"\n{ckpt}\n{n_frames} frames  ({time.time() - t0:.0f}s)\n")

    print("RQ residual decay (energy relative to z_e):")
    print(f"{'stage':>18} {'rel_energy':>12} {'vs prev':>9} {'cum dB':>8}")
    rel = err / err[0]
    for i in range(len(err)):
        name = "z_e (no quant)" if i == 0 else f"after level {i}"
        ratio = float(err[i] / err[i - 1]) if i else float("nan")
        db = 10.0 * torch.log10(rel[i]).item() if rel[i] > 0 else float("-inf")
        rstr = f"{ratio:9.3f}" if i else " " * 9
        print(f"{name:>18} {float(rel[i]):12.4f} {rstr} {db:8.2f}")

    last = float(err[-1] / err[-2])
    print(
        f"\nExtrapolated level 4 (ratio {last:.3f} held): "
        f"rel {float(rel[-1]) * last:.4f} "
        f"({10.0 * torch.log10(rel[-1] * last).item():.2f} dB), "
        f"{(1 - last) * 100:.1f}% residual cut"
    )

    print(
        "\nIntrinsic dimension of each level's input, and the MSE cut that\n"
        "doubling THAT level's table would buy (N^-2/D). D_pr is the\n"
        "participation ratio (optimistic); d90 is dims holding 90% of variance\n"
        "(pessimistic). The truth is bracketed by the two gain columns.\n"
    )
    print(
        f"{'level':>6} {'D_pr':>7} {'d90':>6} {'d99':>6} "
        f"{'gain@D_pr':>10} {'gain@d90':>9}"
    )
    for level, sc in enumerate(scatter):
        lam = torch.linalg.eigvalsh(sc / n_frames).clamp(min=0).flip(0)
        d_pr = float(lam.sum() ** 2 / (lam**2).sum())
        cum = lam.cumsum(0) / lam.sum()
        d90 = int((cum < 0.90).sum()) + 1
        d99 = int((cum < 0.99).sum()) + 1
        g_pr = (1.0 - 2.0 ** (-2.0 / d_pr)) * 100
        g_90 = (1.0 - 2.0 ** (-2.0 / d90)) * 100
        print(
            f"{level + 1:>6} {d_pr:7.1f} {d90:6d} {d99:6d} "
            f"{g_pr:9.1f}% {g_90:8.1f}%"
        )


if __name__ == "__main__":
    main()
