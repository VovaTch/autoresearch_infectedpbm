"""
Per-critic real/fake separation of a train.py checkpoint's discriminator ensemble.

A GAN is only as good as the critics that actually discriminate. Measured
2026-09-12: the whole ensemble was carried by disc4 (AUC 0.93) while the five
HiFi-GAN period critics sat at AUC 0.52-0.62 after 9.5 h -- spectral norm on a
plain conv stack caps its gain so the logits cannot separate. This prints, for
every sub-discriminator, mean D(real), mean D(fake), their gap, the hinge loss
it alone would report, and a rank AUC (0.5 = coin flip).

Usage:
    uv run python probe_critics.py saved_x/last.ckpt --periods 2 3 5 7 11 \\
        --period-norm weight --device cuda:1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from render_samples import _apply_ema
from train import (
    build_learning_params,
    build_loss_aggregator,
    build_module,
    build_optimizer_cfg,
    build_scheduler_cfg,
)
from train_ar import load_track_audio

SAMPLE_RATE = 44100
HOP = 256
SLICE = 32768
DEFAULT_TRACKS = ["deeply_disturbed", "Cookie From Space", "Savant On", "Bark_Original"]
LEGACY_NAMES = ["mel1k", "mel2k", "mel4k", "wave"]


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt", help="train.py Lightning checkpoint")
    parser.add_argument("--periods", nargs="*", type=int, default=[], help="gan.disc_periods")
    parser.add_argument("--period-norm", default="spectral", help="gan.period_norm")
    parser.add_argument("--freq-pool", type=int, default=2, help="gan.disc_freq_pool")
    parser.add_argument("--disc-width", type=int, default=1, help="gan.disc_width")
    parser.add_argument(
        "--stft", nargs="*", default=[], help="gan.disc_stft_resolutions as n_fft:hop"
    )
    parser.add_argument("--tracks", nargs="+", default=DEFAULT_TRACKS)
    parser.add_argument("--slices-per-track", type=int, default=12)
    parser.add_argument("--tracks-dir", default="~/.cache/infected_pbm/tracks")
    parser.add_argument("--slices-dir", default="~/.cache/infected_pbm/slices")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_real_slices(args: argparse.Namespace) -> torch.Tensor:
    """
    Args:
      args (argparse.Namespace): CLI arguments (tracks, dirs, slices_per_track).

    Returns:
      torch.Tensor: (N, 1, SLICE) real audio slices from one third into each track.
    """
    tracks_dir = Path(args.tracks_dir).expanduser()
    slices_dir = Path(args.slices_dir).expanduser()
    out: list[torch.Tensor] = []
    for sub in args.tracks:
        path = sorted(p for p in tracks_dir.glob("*.mp3") if sub in p.name)[0]
        wav = load_track_audio(path, SAMPLE_RATE, HOP, slices_dir).float()
        start = wav.shape[-1] // 3
        span = args.slices_per_track * SLICE
        out.append(wav[:, start : start + span].reshape(-1, 1, SLICE))
    return torch.cat(out)


def rank_auc(pos: torch.Tensor, neg: torch.Tensor, cap: int = 4000) -> float:
    """
    Args:
      pos (torch.Tensor): (P,) logits on real data.
      neg (torch.Tensor): (Q,) logits on fake data.
      cap (int): subsample size per side to bound the pairwise comparison.

    Returns:
      float: P(D(real) > D(fake)) over random pairs.
    """
    a = pos[torch.randperm(len(pos))[:cap]]
    b = neg[torch.randperm(len(neg))[:cap]]
    return float((a[:, None] > b[None, :]).float().mean())


def main() -> None:
    """Load the checkpoint, reconstruct real slices, print per-critic separation."""
    args = parse_args()
    module = build_module(
        build_learning_params(),
        build_loss_aggregator(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        disc_freq_pool=args.freq_pool,
        disc_width=args.disc_width,
        disc_periods=tuple(args.periods),
        period_norm=args.period_norm,
        disc_stft_resolutions=tuple(tuple(map(int, r.split(":"))) for r in args.stft),
        per_level_codebooks=True,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(module.load_state_dict(ckpt["state_dict"], strict=True))
    _apply_ema(module, ckpt)
    module = module.to(args.device).eval()

    real = load_real_slices(args).to(args.device)
    with torch.no_grad():
        fake = torch.cat(
            [module.forward({"slice": real[i : i + 8]})["slice"] for i in range(0, len(real), 8)]
        )
        names = (
            LEGACY_NAMES
            + [f"p{p}" for p in args.periods]
            + [f"s{r.split(':')[0]}" for r in args.stft]
        )
        print(f"\n{'critic':<6} {'D(real)':>8} {'D(fake)':>8} {'gap':>6} {'hinge_d':>8} {'AUC':>5}")
        for name, disc in zip(names, module.discriminator._discriminators):
            d_real = disc(real)["logits"][..., 0].flatten()
            d_fake = disc(fake)["logits"][..., 0].flatten()
            hinge = 0.5 * (torch.relu(1 - d_real).mean() + torch.relu(1 + d_fake).mean())
            print(
                f"{name:<6} {d_real.mean():8.3f} {d_fake.mean():8.3f} "
                f"{d_real.mean() - d_fake.mean():6.3f} {hinge:8.3f} {rank_auc(d_real, d_fake):5.2f}"
            )
    print("\nAUC 0.5 = coin flip; a critic below ~0.65 contributes no useful gradient.")


if __name__ == "__main__":
    main()
