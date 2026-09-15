"""
Validation-only sweep: which config change broke generalisation?

The 16384-frame run warm-started from a checkpoint whose best val loss was
6.59 and reported 26.79 at its very first validation, 500 optimiser steps in,
then worsened every epoch. An instant jump like that is a configuration effect,
not training dynamics, and three things changed at once: crop_frames
4096 -> 16384, rope_theta 10000 -> 40000, and p_drop_cond 0.1 -> 0.5.

p_drop_cond cannot be the cause -- _run forces prob = 0.0 outside "train" -- so
this scores the untouched base checkpoint over the same held-out windows under
each combination of the other two. No training, no weight updates.

  crop  4096  theta 10000   the trained geometry; should reproduce ~6.59
  crop  4096  theta 40000   theta alone, at the trained length
  crop 16384  theta 10000   length alone
  crop 16384  theta 40000   both, the configuration that was running

Usage:
  uv run python eval_val.py
  uv run python eval_val.py --checkpoint saved_ar_20260906_16k/resume_base.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_ar import (
    ArLightningModule,
    ValCropDataset,
    build_model,
    build_token_cache,
    load_config,
    load_token_cache,
)

REPO = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_ar_16k_cont.yaml")
    parser.add_argument(
        "--checkpoint",
        default="saved_ar_20260829_24h/ar_latest.ckpt",
        help="weights to score; the pre-16k base by default",
    )
    parser.add_argument("--crops", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--thetas", type=float, nargs="+", default=[10000.0, 40000.0])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0, help="0 scores all")
    return parser.parse_args()


def score(cfg, crop: int, theta: float, ckpt: Path, batch_size: int, cap: int) -> tuple[float, int]:
    """
    Mean validation cross-entropy for one (crop, theta) pair.

    Args:
      cfg (ArConfig): base config, copied before mutation.
      crop (int): crop_frames to evaluate at.
      theta (float): rope_theta to evaluate at.
      ckpt (Path): checkpoint whose weights are scored.
      batch_size (int): val batch size.
      cap (int): stop after this many batches; 0 scores all.

    Returns:
      tuple[float, int]: mean loss over scored frames and the batch count.
    """
    cfg.data.crop_frames = crop
    cfg.model.rope_theta = theta

    cache_dir = build_token_cache(cfg, force=False)
    tracks, manifest = load_token_cache(cache_dir, cfg.data)
    val_set = ValCropDataset(tracks, cfg.data)
    loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)

    model = build_model(cfg, tracks, manifest)
    module = ArLightningModule(model, cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device).eval()

    total, seen = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if cap and i >= cap:
                break
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                total += float(module._run(batch, "val"))
            seen += 1
    del module, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return total / max(seen, 1), seen


def main() -> None:
    args = parse_args()
    ckpt = REPO / args.checkpoint
    print(f"checkpoint {args.checkpoint}\n")
    print(f"{'crop':>6} {'theta':>8} {'val loss':>10} {'batches':>8}")
    for crop in args.crops:
        for theta in args.thetas:
            cfg = load_config(args.config)
            loss, seen = score(cfg, crop, theta, ckpt, args.batch_size, args.max_batches)
            print(f"{crop:>6} {theta:>8.0f} {loss:>10.4f} {seen:>8}", flush=True)
    print("\nuniform-random CE over 2048 codes = 7.62")


if __name__ == "__main__":
    main()
