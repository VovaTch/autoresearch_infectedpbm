"""Per-band relative reconstruction error, straight from checkpoints.

Why this exists: composite/cdpam/mrstft are scalars that cannot say WHERE a
model is losing. This reports relative magnitude error per frequency band, on
the held-out split, so "the melody sounds smeared" becomes a number attached to
1-4 kHz.

Two things it deliberately does NOT do, both of which invalidate the older
render-based tables (see spectral_table.py):
  * no peak normalization -- render_samples.py peak-normalizes per file, which
    inflates measured error on high-crest/GAN models by up to 12x;
  * no slice concatenation -- each slice is scored on its own, so the render
    seam click train (1.35 Hz) never enters the measurement.

The split is reproduced with L.seed_everything(42) exactly as train.py does, so
the eval slices are the same ones the run never trained on.

Usage:
  uv run python band_error.py saved_A/lvl1_vqgan_best.ckpt saved_B/...ckpt
  uv run python band_error.py --split test --slices 512 ckpt1 ckpt2
"""

from __future__ import annotations

import argparse
import os

import lightning as L
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from prepare import build_data_module
from train import (
    build_learning_params,
    build_loss_aggregator,
    build_module,
    build_optimizer_cfg,
    build_scheduler_cfg,
)

SR = 44100
N_FFT = 2048
HOP = 512
BANDS: dict[str, tuple[float, float]] = {
    "0-250": (0.0, 250.0),
    "250-1k": (250.0, 1000.0),
    "1k-4k": (1000.0, 4000.0),
    "4k-10k": (4000.0, 10000.0),
    "10k+": (10000.0, SR / 2),
}


def band_slices() -> dict[str, slice]:
    """
    Map each band to its rfft bin range at N_FFT.

    Returns:
      dict[str, slice]: band name -> slice over the (N_FFT // 2 + 1) bins.
    """
    freqs = torch.fft.rfftfreq(N_FFT, d=1.0 / SR)
    out: dict[str, slice] = {}
    for name, (lo, hi) in BANDS.items():
        idx = torch.nonzero((freqs >= lo) & (freqs < hi)).flatten()
        out[name] = slice(int(idx[0]), int(idx[-1]) + 1)
    return out


def band_error(
    orig: torch.Tensor, recon: torch.Tensor, bands: dict[str, slice]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Relative magnitude error per band, per item.

    Args:
      orig (torch.Tensor): (B, 1, L) reference waveform.
      recon (torch.Tensor): (B, 1, L) reconstruction, same scale as orig.
      bands (dict[str, slice]): band name -> rfft bin slice.

    Returns:
      dict[str, tuple[torch.Tensor, torch.Tensor]]: band -> (err, ref_energy),
        both (B,). err is ||dX|| / ||X|| over the band's bins on magnitude
        spectrograms; ref_energy is the ||X|| denominator, returned so callers
        can drop degenerate slices (see evaluate).
    """
    win = torch.hann_window(N_FFT, device=orig.device)

    def mag(x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x.squeeze(1), N_FFT, HOP, N_FFT, win, return_complex=True, center=True
        )
        return spec.abs()

    mo, mr = mag(orig), mag(recon)
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, sl in bands.items():
        num = (mr[:, sl] - mo[:, sl]).pow(2).sum(dim=(1, 2)).sqrt()
        den = mo[:, sl].pow(2).sum(dim=(1, 2)).sqrt()
        out[name] = (num / den.clamp_min(1e-12), den)
    return out


def reproduce_split(cfg: dict, batch_size: int):
    """
    Rebuild train.py's train/val/test split bit-exactly.

    random_split() draws from the GLOBAL torch RNG, so the split depends on
    every RNG draw made since seed_everything(42) -- including the generator
    AND discriminator weight init inside build_module(). Skipping that init
    shifts the split so far that only ~6% of the val slices still match, i.e.
    most of the "held-out" set would be slices the run trained on. So this
    replays train.py's prelude in order rather than seeding and calling setup().

    Args:
      cfg (dict): parsed training config.
      batch_size (int): eval batch size (does not affect the split).

    Returns:
      tuple: (SplitDatasetModule with setup() done, the built LightningModule).
    """
    m, vq, gan, tr = cfg["model"], cfg["vq"], cfg["gan"], cfg["train"]
    L.seed_everything(42, workers=True)

    lp = build_learning_params()
    lp.save_path = tr.get("save_path", ".")
    lp.devices = tr.get("devices", "auto")
    lp.learning_rate = tr["lr"]
    if tr.get("batch_size") is not None:
        lp.batch_size = tr["batch_size"]
    optimizer_cfg = build_optimizer_cfg()
    optimizer_cfg["lr"] = tr["lr"]
    scheduler_cfg = build_scheduler_cfg(
        total_minutes=tr["minutes"],
        max_lr=tr["lr"],
        pct_start=tr["lr_pct_start"],
        div_factor=tr.get("lr_div_factor", 25.0),
    )
    loss_aggregator = build_loss_aggregator(commit_weight=vq["commit_weight"])
    dm = build_data_module(lp)
    module = build_module(
        lp,
        loss_aggregator,
        optimizer_cfg,
        scheduler_cfg,
        gss=gan["gan_start_step"],
        token_dim=m["token_dim"],
        num_rq_steps=m["num_rq"],
        num_tokens=m["num_tokens"],
        time_downsample=m["time_downsample"],
        disc_width=gan["disc_width"],
        disc_freq_pool=gan.get("disc_freq_pool", 5),
        d_weight_cap=gan["d_weight_cap"],
        disc_warmup=gan.get("disc_warmup", 0),
        hidden=m["hidden"],
        per_level_codebooks=m.get("per_level_codebooks", False),
        ze_norm=m.get("ze_norm", "none"),
    )
    dm.setup("fit")
    return dm, module


def load_generator(ckpt_path: str, module, device: torch.device):
    """
    Load a checkpoint's generator weights into an already-built module.

    Args:
      ckpt_path (str): path to a lightning .ckpt.
      module: LightningModule from reproduce_split().
      device (torch.device): where to place the model.

    Returns:
      MultiLvlVQVariationalAutoEncoder: eval-mode generator.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    result = module.load_state_dict(sd, strict=False)
    bad = [k for k in result.missing_keys + result.unexpected_keys]
    if bad:
        print(f"  WARN key mismatch: {bad[:3]}")
    return module.model.to(device).eval()


# A slice with no energy in a band makes ||dX|| / ||X|| meaningless -- the
# dataset contains at least one pure-digital-silence slice, which alone drove
# the mean to 2.5e10. Drop slices whose reference band energy is below this
# fraction of that band's median across the eval set.
DEGENERATE_FRAC = 1e-3


def evaluate(gen, loader: DataLoader, device: torch.device) -> dict[str, tuple]:
    """
    Accumulate per-band error over a dataloader.

    Args:
      gen: generator whose forward returns a dict holding "slice".
      loader (DataLoader): batches with a "slice" key, (B, 1, L).
      device (torch.device): compute device.

    Returns:
      dict[str, tuple[float, float, float, int]]: band -> (median, mean,
        stderr, n_dropped). Median leads because the error distribution is
        right-tailed; mean and stderr are kept for significance tests.
    """
    bands = band_slices()
    errs: dict[str, list[torch.Tensor]] = {k: [] for k in BANDS}
    dens: dict[str, list[torch.Tensor]] = {k: [] for k in BANDS}
    with torch.no_grad():
        for batch in loader:
            orig = batch["slice"].to(device)
            recon = gen(orig)["slice"]
            for name, (err, den) in band_error(orig, recon, bands).items():
                errs[name].append(err.cpu())
                dens[name].append(den.cpu())
    out: dict[str, tuple] = {}
    for name in BANDS:
        e, d = torch.cat(errs[name]), torch.cat(dens[name])
        keep = d > DEGENERATE_FRAC * d.median()
        v = e[keep]
        out[name] = (
            v.median().item(),
            v.mean().item(),
            (v.std() / v.numel() ** 0.5).item(),
            int((~keep).sum()),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpts", nargs="+", help="checkpoints to compare")
    ap.add_argument("--config", default="config_20260827_cont9h.yaml")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--slices", type=int, default=512, help="eval slices (0 = all)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dm, module = reproduce_split(cfg, args.batch_size)
    ds = dm.val_dataset if args.split == "val" else dm.test_dataset
    if args.slices and args.slices < len(ds):
        ds = Subset(ds, list(range(args.slices)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    device = torch.device(args.device)
    print(f"{args.split} split: {len(ds)} slices, {args.config}\n")

    results: dict[str, dict[str, tuple]] = {}
    for ck in args.ckpts:
        print(f"scoring {ck}")
        gen = load_generator(ck, module, device)
        results[ck] = evaluate(gen, loader, device)

    names = [os.path.basename(os.path.dirname(c)) or c for c in args.ckpts]
    w = max(20, max(len(n) for n in names) + 2)
    print("\nrelative magnitude error per band, median (mean +- stderr)")
    print("lower is better; both models scored on the SAME held-out slices\n")
    print("  band    " + "".join(f"{n:>{w}}" for n in names))
    for band in BANDS:
        row = ""
        for c in args.ckpts:
            med, mean, se, _ = results[c][band]
            row += f"{med:>{w - 18}.4f}  ({mean:.4f}+-{se:.4f})"
        print(f"  {band:<8}" + row)
    dropped = {b: results[args.ckpts[0]][b][3] for b in BANDS}
    if any(dropped.values()):
        print(f"\n  degenerate slices dropped per band: {dropped}")


if __name__ == "__main__":
    main()
