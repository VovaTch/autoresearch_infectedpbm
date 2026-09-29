"""Three-way verdict for the tanh fine-tune (config_20260921_tanh_ft.yaml).

Arms, all on the same clips, all rms-matched to the peak-normed source:

  onnx_guard  production onnx/ graphs (dsteps2_24h, EMA) through the OLD peak
              guard (whole clip scaled so its max sample sits at 1.0) -- what
              the harness played before 2026-09-21.
  onnx_knee   same graphs through the new soft knee -- what it plays now.
  adhoc_tanh  dsteps2_24h torch ckpt, tanh applied to the raw output, no
              training; then knee (rms matching can push the body back over 1).
  ft_tanh     the fine-tuned ckpt with its trained tanh; then knee.

Missing arms are skipped, so this runs before the fine-tune finishes.
Env: N_CLIPS (default 3 per track), FT_CKPT, ONNX_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from render_fixed import decode_full, peak_norm, rms_match, soft_knee, stats
from render_samples import (
    SR,
    build_learning_params,
    build_loss_aggregator,
    build_optimizer_cfg,
    build_scheduler_cfg,
    load_module,
    multi_res_stft_dist,
    pick_clips,
)
from roundtrip_external import round_trip

OUT = Path(__file__).resolve().parent / "renders_tanh_ft"
BASE_CKPT = "saved_20260914_dsteps2_24h/lvl1_vqgan_last.ckpt"
FT_CKPT = os.environ.get("FT_CKPT", "saved_20260921_tanh_ft/lvl1_vqgan_last.ckpt")
ONNX_DIR = Path(os.environ.get("ONNX_DIR", "onnx"))
CHUNK_FRAMES, MARGIN = 4096, 256


def main() -> None:
    OUT.mkdir(exist_ok=True)
    lp, oc, sc, la = (
        build_learning_params(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        build_loss_aggregator(),
    )
    import cdpam

    evaluator = cdpam.CDPAM(dev="cpu")
    resample = torchaudio.transforms.Resample(SR, 22050)

    def cdpam_score(orig: torch.Tensor, rec: torch.Tensor) -> float:
        return float(
            evaluator.forward(
                resample(orig.float()) * 32768.0, resample(rec.float()) * 32768.0
            )
            .mean()
            .item()
        )

    clips = pick_clips(int(os.environ.get("N_CLIPS", "3")))
    print(f"Picked {len(clips)} clips.")
    base = load_module(BASE_CKPT, lp, oc, sc, la, per_level_codebooks=True).cuda()
    ft = None
    if Path(FT_CKPT).exists():
        ft = load_module(
            FT_CKPT, lp, oc, sc, la, per_level_codebooks=True, output_act="tanh"
        ).cuda()
    else:
        print(f"SKIP ft_tanh: {FT_CKPT} missing")

    cols = ("cdpam", "mrstft", "crest_dB", "rms", "peak", "over_ppm")
    hdr = f"{'clip':<24}{'arm':<12}" + "".join(f"{c:>10}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    acc: dict[str, list[tuple[float, ...]]] = {}

    for i, (name, clip) in enumerate(clips):
        clip = peak_norm(clip)
        n = clip.shape[1] // 256 * 256
        clip = clip[:, :n]
        onnx_out, _ = round_trip(ONNX_DIR, clip, CHUNK_FRAMES, MARGIN)
        raw_base = decode_full(base, clip)
        arms = {"ORIG": clip, "onnx": onnx_out, "adhoc_tanh": torch.tanh(raw_base)}
        if ft is not None:
            arms["ft_tanh"] = decode_full(ft, clip)
        arms = rms_match(arms, clip)
        onnx_rms = arms.pop("onnx")
        group = {
            "ORIG": arms["ORIG"],
            "onnx_guard": onnx_rms / (onnx_rms.abs().max() + 1e-8),
            "onnx_knee": soft_knee(onnx_rms),
            "adhoc_tanh": soft_knee(arms["adhoc_tanh"]),
        }
        if "ft_tanh" in arms:
            group["ft_tanh"] = soft_knee(arms["ft_tanh"])
        orig = group["ORIG"]

        for arm, rec in group.items():
            torchaudio.save(str(OUT / f"clip{i}_{name}_{arm}.wav"), rec, SR)
            L = min(orig.shape[1], rec.shape[1])
            cd = 0.0 if arm == "ORIG" else cdpam_score(orig[:, :L], rec[:, :L])
            mr = 0.0 if arm == "ORIG" else multi_res_stft_dist(orig[:, :L], rec[:, :L])
            row = (cd, mr, *stats(rec))
            acc.setdefault(arm, []).append(row)
            print(
                f"{('clip' + str(i) + ' ' + name)[:23]:<24}{arm:<12}"
                + "".join(f"{v:>10.4f}" for v in row)
            )

    print("\n" + "=" * len(hdr))
    print(f"{'MEAN':<24}{'arm':<12}" + "".join(f"{c:>10}" for c in cols))
    for arm, rows in acc.items():
        m = np.mean(rows, axis=0)
        print(f"{'':<24}{arm:<12}" + "".join(f"{v:>10.4f}" for v in m))
    print(f"\nWavs in: {OUT}")


if __name__ == "__main__":
    main()
