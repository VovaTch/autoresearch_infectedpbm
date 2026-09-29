"""Is the tokenizer's TRAINING error a capacity floor or a rate floor?

2026-09-23. The question this answers: train cdpam sits at 0.0277 and will not
go to zero, which looks like "the model is too small to even overfit 6.3 h of
audio". But a VQ-VAE is not a free autoencoder -- everything has to pass
through 3 x 11 bits per latent frame at 172.27 frames/s = 5.68 kbit/s, a 124x
compression of 44.1 kHz/16-bit PCM. No decoder, at any size, reconstructs music
transparently at 5.68 kbit/s.

So this measures the rate-distortion curve on TRAINING slices, where the model
has already seen every sample many times:

  rq1      levels 0 only     1.89 kbit/s
  rq2      levels 0-1        3.79 kbit/s
  rq3      levels 0-2        5.68 kbit/s  (the production path)
  bypass   z_e, no quantizer  continuous  (the capacity-only floor)

Read it like this. If bypass error collapses toward zero while rq3 sits at the
training floor, the floor is the BOTTLENECK and a bigger decoder buys nothing.
If bypass stays near rq3, the encoder/decoder pair genuinely cannot fit the
data and capacity is the limiter.

Runs on CPU by default so it does not perturb training on either GPU.

Env: N_SLICES (default 8), DEVICE (default cpu), CKPT.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from prepare import build_data_module
from render_samples import (
    SR,
    build_learning_params,
    build_loss_aggregator,
    build_optimizer_cfg,
    build_scheduler_cfg,
    load_module,
    multi_res_stft_dist,
)

CKPT = os.environ.get("CKPT", "saved_20260914_dsteps2_24h/lvl1_vqgan_last.ckpt")
DEVICE = os.environ.get("DEVICE", "cpu")
N_SLICES = int(os.environ.get("N_SLICES", "8"))
FPS = 44100 / 256  # one latent frame per STFT hop


@torch.no_grad()
def decode_levels(net, z_e: torch.Tensor, levels: int | None) -> torch.Tensor:
    """
    Decode a latent using only the first `levels` RQ stages.

    Args:
      net: the MultiLvlVQVariationalAutoEncoder.
      z_e (torch.Tensor): (B, C, T) continuous encoder output.
      levels (int | None): how many RQ levels to keep; None bypasses the
        quantizer entirely and decodes z_e.

    Returns:
      torch.Tensor: (B, 1, L) reconstruction.
    """
    if levels is None:
        return net.decoder(z_e)
    indices = net.vq_module(z_e)["indices"]  # (B, T, R)
    books = net.vq_module.vq_codebook.code_embedding  # (R, N, C)
    z_q = torch.zeros(
        indices.shape[0], indices.shape[1], books.shape[-1], dtype=z_e.dtype
    )
    for level in range(levels):
        z_q = z_q + books[level][indices[..., level]]
    return net.decoder(z_q.transpose(1, 2).contiguous())


def main() -> None:
    lp, oc, sc, la = (
        build_learning_params(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        build_loss_aggregator(),
    )
    dm = build_data_module(lp)
    dm.setup("fit")
    train_items = dm.train_dataset
    print(f"train split: {len(train_items)} slices; probing {N_SLICES} of them")

    module = load_module(CKPT, lp, oc, sc, la, per_level_codebooks=True)
    net = module.model.to(DEVICE).eval()

    import cdpam
    import torchaudio

    evaluator = cdpam.CDPAM(dev="cpu")
    resample = torchaudio.transforms.Resample(SR, 22050)

    def cdpam_score(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(
            evaluator.forward(
                resample(a.float()) * 32768.0, resample(b.float()) * 32768.0
            )
            .mean()
            .item()
        )

    arms: dict[str, int | None] = {"rq1": 1, "rq2": 2, "rq3": 3, "bypass": None}
    rates = {"rq1": 11 * FPS, "rq2": 22 * FPS, "rq3": 33 * FPS, "bypass": float("nan")}
    acc: dict[str, list[tuple[float, float]]] = {a: [] for a in arms}

    step = max(1, len(train_items) // N_SLICES)
    for n in range(N_SLICES):
        x = train_items[n * step]["slice"].reshape(1, 1, -1).to(DEVICE).float()
        z_e = net.encode(x)
        for arm, levels in arms.items():
            rec = decode_levels(net, z_e, levels)
            L = min(x.shape[-1], rec.shape[-1])
            a, b = x.reshape(1, -1)[:, :L].cpu(), rec.reshape(1, -1)[:, :L].cpu()
            acc[arm].append((cdpam_score(a, b), multi_res_stft_dist(a, b)))
        print(f"  slice {n + 1}/{N_SLICES} done")

    print(f"\n{'arm':<10}{'kbit/s':>9}{'cdpam':>10}{'mrstft':>10}")
    print("-" * 39)
    for arm in arms:
        m = np.mean(acc[arm], axis=0)
        rate = rates[arm] / 1000.0
        rs = "  cont." if np.isnan(rate) else f"{rate:9.2f}"
        print(f"{arm:<10}{rs}{m[0]:>10.4f}{m[1]:>10.4f}")
    print("\nTraining-set reference from the 24 h run: cdpam 0.0277, mrstft 1.287")


if __name__ == "__main__":
    main()
