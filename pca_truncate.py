"""Does the low-variance tail of z_e carry signal the decoder uses?

`rq_residual.py` found z_e is extremely anisotropic: participation ratio 5.3
against token_dim 1024, with 90% of variance in 34 dims and 99% in 325. Euclidean
VQ resolves only the high-variance head, so whatever lives in the tail is
quantized away. That is correct behaviour if the tail is noise, and a real loss
if it is signal.

This script decides which. It builds the global PCA basis of z_e, truncates z_e
to its top-k principal directions, and decodes with the quantizer bypassed
(`bypass_vq`), so the only variable is how many latent directions the decoder
gets. Reconstruction is scored against the source audio with the same
multi-resolution STFT distance used at eval.

Read it as: if k=34 already matches k=1024, the tail is noise and the anisotropy
is the model compressing correctly -- leave token_dim alone. If truncation costs
real mrstft, the tail is signal the quantizer is currently discarding, and a
low-dim bottleneck before lookup (DAC/EnCodec factorized codes) has a target.

Usage:
    CKPT=saved_20260827_cont9h/lvl1_vqgan_last.ckpt uv run python pca_truncate.py
"""

from __future__ import annotations

import glob
import os
import time

import torch

from render_samples import CACHE, SLICE, load_module, multi_res_stft_dist
from train import (
    build_learning_params,
    build_loss_aggregator,
    build_optimizer_cfg,
    build_scheduler_cfg,
)

KS = (2, 5, 16, 34, 128, 325, 1024)


def _mrstft(rec: torch.Tensor, ref: torch.Tensor) -> float:
    """Multi-resolution STFT distance on (B, 1, L) tensors.

    The decoder's ISTFT returns more samples than it was given, so both sides
    are trimmed to the slice length before scoring.

    Args:
      rec (torch.Tensor): (B, 1, L') reconstruction.
      ref (torch.Tensor): (B, 1, SLICE) source audio.

    Returns:
      float: mean multi-resolution STFT distance over the batch.
    """
    b = ref.shape[0]
    return multi_res_stft_dist(
        rec[..., :SLICE].reshape(b, -1), ref[..., :SLICE].reshape(b, -1)
    )


def iter_slices(
    slices_per_track: int, batch: int, max_tracks: int | None
) -> list[torch.Tensor]:
    """Evenly-spaced slice batches from the cached track files.

    Args:
      slices_per_track (int): slices sampled per track file.
      batch (int): slices per batch.
      max_tracks (int | None): cap on track files; None = all.

    Returns:
      list[torch.Tensor]: (B, 1, SLICE) float32 batches.
    """
    out: list[torch.Tensor] = []
    for path in sorted(glob.glob(os.path.join(CACHE, "*.pt")))[:max_tracks]:
        pool = torch.load(path, map_location="cpu")
        step = max(1, pool.shape[0] // slices_per_track)
        picked = pool[::step][:slices_per_track].reshape(-1, 1, SLICE).float()
        out += [picked[i : i + batch] for i in range(0, picked.shape[0], batch)]
    return out


def pca_basis(
    module: torch.nn.Module, batches: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global PCA of z_e over the given batches.

    Args:
      module: loaded LightningModule exposing `.model.encode`.
      batches (list[torch.Tensor]): (B, 1, SLICE) input batches.

    Returns:
      tuple: (token_dim,) float32 mean and (token_dim, token_dim) float32
        eigenvectors as columns, ordered by descending variance.
    """
    total: torch.Tensor | None = None
    scatter: torch.Tensor | None = None
    n = 0
    for x in batches:
        with torch.no_grad():
            z = module.model.encode(x)
        flat = z.transpose(1, 2).reshape(-1, z.shape[1]).double()
        total = flat.sum(0) if total is None else total + flat.sum(0)
        scatter = flat.T @ flat if scatter is None else scatter + flat.T @ flat
        n += flat.shape[0]
    assert total is not None and scatter is not None
    mean = total / n
    cov = scatter / n - torch.outer(mean, mean)
    _, vecs = torch.linalg.eigh(cov)
    return mean.float(), vecs.flip(1).float()


def main() -> None:
    """Load a checkpoint, PCA-truncate z_e at several ranks, score each decode."""
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    ckpt = os.environ.get("CKPT", "saved_20260827_cont9h/lvl1_vqgan_last.ckpt")

    module = load_module(
        ckpt,
        build_learning_params(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        build_loss_aggregator(),
        per_level_codebooks=True,
    )

    batches = iter_slices(
        slices_per_track=int(os.environ.get("SLICES_PER_TRACK", "24")),
        batch=8,
        max_tracks=int(os.environ.get("MAX_TRACKS", "8")),
    )
    n_slices = sum(b.shape[0] for b in batches)
    print(f"{n_slices} slices\nBuilding PCA basis...", flush=True)
    t0 = time.time()
    mean, vecs = pca_basis(module, batches)

    print("Decoding truncated latents (quantizer bypassed)...", flush=True)
    was_bypass = module.model.bypass_vq
    module.model.bypass_vq = True
    scores = {k: 0.0 for k in KS}
    quant_score = 0.0
    try:
        for x in batches:
            with torch.no_grad():
                z = module.model.encode(x)
                centered = (z.transpose(1, 2) - mean).transpose(1, 2)
                for k in KS:
                    basis = vecs[:, :k]
                    proj = torch.einsum("dk,bdt->bkt", basis, centered)
                    z_k = torch.einsum("dk,bkt->bdt", basis, proj)
                    z_k = (z_k.transpose(1, 2) + mean).transpose(1, 2)
                    rec, _ = module.model.decode(z_k, (x.shape[0], 1, -1))
                    scores[k] += _mrstft(rec, x) * x.shape[0]
                module.model.bypass_vq = False
                rec_q, _ = module.model.decode(z, (x.shape[0], 1, -1))
                quant_score += _mrstft(rec_q, x) * x.shape[0]
                module.model.bypass_vq = True
    finally:
        module.model.bypass_vq = was_bypass

    print(f"\n{ckpt}  ({time.time() - t0:.0f}s)\n")
    full = scores[KS[-1]] / n_slices
    print(f"{'top-k dims':>11} {'mrstft':>8} {'vs full':>9}")
    for k in KS:
        v = scores[k] / n_slices
        print(f"{k:>11} {v:8.4f} {v - full:+9.4f}")
    print(f"\n{'full + RQ':>11} {quant_score / n_slices:8.4f} "
          f"{quant_score / n_slices - full:+9.4f}   <- cost of quantization")


if __name__ == "__main__":
    main()
