"""
Score the flow enhancer on the held-out tracks: A original / B round trip / C enhanced.

Mirrors probe_copy_synthesis.py's A/B/C framing, which is where the target came
from: the round trip alone owns 55% of the log-spectral error and 81% of the
crest error, so C has to close that gap, not merely differ from B. Everything is
LUFS-matched to the session target first -- per-clip peak normalisation is the
bias that invalidated earlier A/Bs.

The verdict lines are pre-registered in config_flow.yaml's header and repeated in
the code below, so the outcome cannot be re-argued after seeing the numbers.

Usage:
    uv run python eval_flow.py --checkpoint saved_flow/flow_latest.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ab_harness.model.audio import normalize_lufs
from enhance_flow import enhance
from prepare import cdpam_distance, make_cdpam_evaluator, multi_res_stft_distance, stft_consistency
from probe_copy_synthesis import spectral_rmse
from score_renders import features
from train_flow import (
    build_geom,
    load_config,
    load_flow_module,
    load_pair_cache,
    pair_cache_dir,
    stft_c,
    uncompress,
)

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
    parser.add_argument("--tracks", type=int, default=0, help="0 = every held-out track")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--start-frac", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--win-frames", type=int, default=None)
    parser.add_argument("--no-project", action="store_true")
    parser.add_argument("--out", default="renders_flow")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-cdpam", action="store_true", help="skip the cdpam column")
    return parser.parse_args()


def save(path: Path, pcm: np.ndarray, rate: int) -> None:
    """
    Args:
      path (Path): destination wav.
      pcm (np.ndarray): (L,) float32 waveform.
      rate (int): sample rate.
    """
    torchaudio.save(str(path), torch.from_numpy(pcm).reshape(1, -1), rate)


def predicted_consistency(module, rt: torch.Tensor) -> float:
    """
    How realizable is the spectrogram the model emits, before any ISTFT?

    The decoder this model cleans up measures 0.5136 here against a real-audio
    floor of 0.0005 (probe_consistency.py); the whole design bets on that number
    coming down. Read at t=0, i.e. the model's one-shot answer.

    Read on one crop-sized window, not the whole span: the bottleneck attention
    is quadratic in the frequency-time grid, so a 12 s clip would cost hundreds
    of times a crop.

    Args:
      module (FlowModule): trained model.
      rt (torch.Tensor): (1, 1, L) round-trip waveform on the model's device.

    Returns:
      float: relative L1 gap between the prediction and the STFT of its ISTFT.
    """
    from train_flow import istft_c

    geom = module.geom
    rt = rt[..., : geom.crop_samples(module.cfg.data.crop_frames)]
    with torch.no_grad():
        x0 = stft_c(rt, geom)
        x1_hat = module.net(x0, x0, torch.zeros(1, device=rt.device))
        wav = istft_c(x1_hat, geom, rt.shape[-1])
        return float(
            stft_consistency(
                uncompress(x1_hat, geom), wav, n_fft=geom.n_fft, hop=geom.hop, shift=0
            )
        )


def main() -> None:
    """Render and score A/B/C on every held-out track, then print the verdict."""
    args = parse_args()
    cfg = load_config(args.config)
    module = load_flow_module(args.checkpoint, args.device)
    tracks, manifest = load_pair_cache(pair_cache_dir(cfg), cfg.data, held_out=True)
    geom = build_geom(cfg, manifest)
    rate = geom.sample_rate
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.tracks:
        tracks = tracks[: args.tracks]

    evaluator = None if args.no_cdpam else make_cdpam_evaluator("cpu")
    span = int(args.seconds * rate)
    rows: list[dict[str, float]] = []

    header = (
        f"{'track':<28} {'rmse A>B':>9} {'rmse A>C':>9} "
        f"{'crestA':>7} {'crestB':>7} {'crestC':>7} "
        f"{'beatB':>6} {'beatC':>6} {'mrsB':>6} {'mrsC':>6} {'cdpB':>6} {'cdpC':>6} {'cons':>6}"
    )
    print(header)
    print("-" * len(header))

    for track in tracks:
        start = int(args.start_frac * track.num_frames) * geom.hop
        start = min(start, max(0, track.clean.numel() - span))
        clean = track.clean[start : start + span].float().reshape(1, 1, -1)
        rt = track.rt[start : start + span].float().reshape(1, 1, -1)

        enhanced = enhance(
            module,
            rt.to(args.device),
            steps=args.steps,
            project_steps=False if args.no_project else None,
            win_frames=args.win_frames,
        ).cpu()
        usable = enhanced.shape[-1]
        clean, rt = clean[..., :usable], rt[..., :usable]

        matched = {
            name: normalize_lufs(sig.reshape(-1).numpy(), rate, TARGET_LUFS)
            for name, sig in (("A", clean), ("B", rt), ("C", enhanced))
        }
        tensors = {
            k: torch.from_numpy(v).reshape(1, 1, -1) for k, v in matched.items()
        }

        row = {
            "rmse_b": spectral_rmse(matched["A"], matched["B"], rate),
            "rmse_c": spectral_rmse(matched["A"], matched["C"], rate),
            "mrstft_b": float(multi_res_stft_distance(tensors["B"], tensors["A"])),
            "mrstft_c": float(multi_res_stft_distance(tensors["C"], tensors["A"])),
            "cons": predicted_consistency(module, rt.to(args.device)),
        }
        for name in ("A", "B", "C"):
            feat = features(matched[name], rate)
            row[f"crest_{name.lower()}"] = feat["crest"]
            row[f"beat_{name.lower()}"] = feat["beat"]
            row[f"cent_{name.lower()}"] = feat["cent"]
            row[f"hf_{name.lower()}"] = feat["hf"]
        if evaluator is not None:
            row["cdpam_b"] = float(cdpam_distance(evaluator, tensors["B"], tensors["A"], rate))
            row["cdpam_c"] = float(cdpam_distance(evaluator, tensors["C"], tensors["A"], rate))
        else:
            row["cdpam_b"] = row["cdpam_c"] = float("nan")
        rows.append(row)

        stem = f"{track.track_idx:03d}_{track.track_name[:28].replace(' ', '_')}"
        for name in ("A", "B", "C"):
            save(out_dir / f"{stem}_{name}.wav", matched[name], rate)

        print(
            f"{track.track_name[:28]:<28} {row['rmse_b']:9.4f} {row['rmse_c']:9.4f} "
            f"{row['crest_a']:7.2f} {row['crest_b']:7.2f} {row['crest_c']:7.2f} "
            f"{row['beat_b']:6.3f} {row['beat_c']:6.3f} "
            f"{row['mrstft_b']:6.3f} {row['mrstft_c']:6.3f} "
            f"{row['cdpam_b']:6.4f} {row['cdpam_c']:6.4f} {row['cons']:6.4f}"
        )

    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("-" * len(header))
    print(
        f"{'MEAN':<28} {mean['rmse_b']:9.4f} {mean['rmse_c']:9.4f} "
        f"{mean['crest_a']:7.2f} {mean['crest_b']:7.2f} {mean['crest_c']:7.2f} "
        f"{mean['beat_b']:6.3f} {mean['beat_c']:6.3f} "
        f"{mean['mrstft_b']:6.3f} {mean['mrstft_c']:6.3f} "
        f"{mean['cdpam_b']:6.4f} {mean['cdpam_c']:6.4f} {mean['cons']:6.4f}"
    )
    print(
        f"\ncentroid A/B/C {mean['cent_a']:.0f} / {mean['cent_b']:.0f} / {mean['cent_c']:.0f} Hz"
        f"    hf>5k {mean['hf_a']:.3f} / {mean['hf_b']:.3f} / {mean['hf_c']:.3f}"
    )
    print(verdict(mean))
    print(f"renders -> {out_dir}")


def verdict(mean: dict[str, float]) -> str:
    """
    Apply the pre-registered outcome lines from config_flow.yaml's header.

    FAIL is tested before PASS: a model that closes the crest gap by smoothing
    the mix would score well on crest alone, so the spectral and rhythmic terms
    hold a veto.

    Args:
      mean (dict[str, float]): per-metric means over the held-out tracks.

    Returns:
      str: the verdict line.
    """
    crest_gap = abs(mean["crest_c"] - mean["crest_a"])
    was_gap = abs(mean["crest_b"] - mean["crest_a"])
    beat_drop = mean["beat_b"] - mean["beat_c"]
    cdpam_worse = mean["cdpam_c"] > mean["cdpam_b"]

    if mean["mrstft_c"] > mean["mrstft_b"] or cdpam_worse or beat_drop > 0.05:
        why = []
        if mean["mrstft_c"] > mean["mrstft_b"]:
            why.append(f"mrstft {mean['mrstft_b']:.3f} -> {mean['mrstft_c']:.3f}")
        if cdpam_worse:
            why.append(f"cdpam {mean['cdpam_b']:.4f} -> {mean['cdpam_c']:.4f}")
        if beat_drop > 0.05:
            why.append(f"beat -{beat_drop:.3f}")
        return "FAIL: " + ", ".join(why)

    if (
        crest_gap <= 1.0
        and mean["rmse_c"] <= 0.8 * mean["rmse_b"]
        and abs(mean["beat_c"] - mean["beat_b"]) <= 0.02
    ):
        return (
            f"PASS: crest gap {was_gap:.2f} -> {crest_gap:.2f} dB, "
            f"rmse {mean['rmse_b']:.4f} -> {mean['rmse_c']:.4f}"
        )

    return (
        f"WASH: crest gap {was_gap:.2f} -> {crest_gap:.2f} dB "
        f"(needs <=1.0), rmse {mean['rmse_b']:.4f} -> {mean['rmse_c']:.4f} "
        f"(needs <={0.8 * mean['rmse_b']:.4f})"
    )


if __name__ == "__main__":
    main()
