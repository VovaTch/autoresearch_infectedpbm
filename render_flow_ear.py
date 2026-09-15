"""
Render the flow enhancer for listening: one folder, A/B and several step counts.

mrstft and cdpam are proxies, and this repo has already caught them disagreeing
with the ear (the GAN sounded better while scoring worse). So this writes audio
first and prints the numbers underneath, rather than gating on them: every
variant of the same span sits side by side, LUFS-matched, so the only difference
you hear is the model.

Usage:
    uv run python render_flow_ear.py --checkpoint saved_flow_20260910/flow_latest.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ab_harness.model.audio import normalize_lufs
from enhance_flow import enhance
from prepare import multi_res_stft_distance
from probe_copy_synthesis import spectral_rmse
from score_renders import features
from train_flow import build_geom, load_config, load_flow_module, load_pair_cache, pair_cache_dir

REPO = Path(__file__).resolve().parent
TARGET_LUFS = -23.0


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "config_flow.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tracks", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--start-frac", type=float, default=0.35)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=[None],
        help="bridge-noise scales to render; every steps x sigma pair is written",
    )
    parser.add_argument("--no-project", action="store_true")
    parser.add_argument("--out", default="renders_flow_ear")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Write every variant of each held-out span, then print the proxy metrics."""
    args = parse_args()
    cfg = load_config(args.config)
    module = load_flow_module(args.checkpoint, args.device)
    tracks, manifest = load_pair_cache(pair_cache_dir(cfg), cfg.data, held_out=True)
    geom = build_geom(cfg, manifest)
    rate = geom.sample_rate
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    span = int(args.seconds * rate)

    rows: list[tuple[str, str, dict[str, float], float, float]] = []
    for track in tracks[: args.tracks]:
        start = int(args.start_frac * track.num_frames) * geom.hop
        start = min(start, max(0, track.clean.numel() - span))
        clean = track.clean[start : start + span].float().reshape(1, 1, -1)
        rt = track.rt[start : start + span].float().reshape(1, 1, -1)

        variants: dict[str, torch.Tensor] = {}
        for sigma in args.sigma:
            tag = "sdef" if sigma is None else f"s{sigma:.2f}"
            for steps in args.steps:
                variants[f"C_{steps}step_{tag}"] = enhance(
                    module,
                    rt.to(args.device),
                    steps=steps,
                    project_steps=False if args.no_project else None,
                    sigma=sigma,
                ).cpu()
        usable = min(v.shape[-1] for v in variants.values())
        variants["A_original"] = clean[..., :usable]
        variants["B_roundtrip"] = rt[..., :usable]

        matched = {
            name: normalize_lufs(sig[..., :usable].reshape(-1).numpy(), rate, TARGET_LUFS)
            for name, sig in variants.items()
        }
        stem = f"{track.track_idx:03d}_{track.track_name[:24].strip().replace(' ', '_')}"
        for name in sorted(matched):
            torchaudio.save(
                str(out_dir / f"{stem}__{name}.wav"),
                torch.from_numpy(matched[name]).reshape(1, -1),
                rate,
            )
        for name in sorted(matched):
            pred = torch.from_numpy(matched[name]).reshape(1, 1, -1)
            ref = torch.from_numpy(matched["A_original"]).reshape(1, 1, -1)
            rows.append(
                (
                    track.track_name[:20],
                    name,
                    features(matched[name], rate),
                    spectral_rmse(matched["A_original"], matched[name], rate),
                    float(multi_res_stft_distance(pred, ref)),
                )
            )

    print(f"\n{'track':<20} {'variant':<18} {'crest':>7} {'beat':>6} {'cent':>6} {'hf':>6} {'rmse':>7} {'mrstft':>7}")
    print("-" * 82)
    for name, variant, feat, rmse, mrs in rows:
        print(
            f"{name:<20} {variant:<18} {feat['crest']:7.2f} {feat['beat']:6.3f} "
            f"{feat['cent']:6.0f} {feat['hf']:6.3f} {rmse:7.4f} {mrs:7.4f}"
        )
    print(f"\naudio -> {out_dir}   (same span, LUFS {TARGET_LUFS}; compare A vs B vs each C)")


if __name__ == "__main__":
    main()
