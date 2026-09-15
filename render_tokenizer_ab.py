"""
Render tokenizer round trips from several train.py checkpoints, side by side.

Every render_samples.py clip is peak-normalised per file, which inflated error
up to 12x on high-crest models and biased earlier ear tests. This writes the
original and each checkpoint's encode->decode of the SAME span, LUFS-matched,
so the only difference you hear is the checkpoint. Numbers print underneath as
proxies, not verdicts -- the ear has overruled them three times in this repo.

Usage:
    uv run python render_tokenizer_ab.py \\
        cont9h=saved_20260827_cont9h/lvl1_vqgan_last.ckpt \\
        smoke=saved_20260911_fp2cons_smoke/lvl1_vqgan_last.ckpt

A value of `onnx` (e.g. `onnx=onnx`) renders through onnx/encoder.onnx ->
onnx/decoder.onnx -- the exported tokenizer the AR stage was trained on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ab_harness.model.audio import normalize_lufs
from prepare import multi_res_stft_distance
from probe_copy_synthesis import spectral_rmse
from render_samples import load_module, reconstruct
from score_renders import features
from train import (
    build_learning_params,
    build_loss_aggregator,
    build_optimizer_cfg,
    build_scheduler_cfg,
)
from train_ar import load_track_audio

REPO = Path(__file__).resolve().parent
SAMPLE_RATE = 44100
HOP = 256
TARGET_LUFS = -23.0
DEFAULT_TRACKS = ["deeply_disturbed", "Cookie From Space", "Savant On", "Bark_Original"]


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpts", nargs="+", help="name=path pairs, one per checkpoint")
    parser.add_argument("--tracks", nargs="+", default=DEFAULT_TRACKS, help="mp3 substrings")
    parser.add_argument("--tracks-dir", default="~/.cache/infected_pbm/tracks")
    parser.add_argument("--slices-dir", default="~/.cache/infected_pbm/slices")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--start-frac", type=float, default=0.35)
    parser.add_argument(
        "--start-sec", type=float, default=None, help="absolute start; overrides --start-frac"
    )
    parser.add_argument("--onnx-dir", default="onnx")
    parser.add_argument("--out", default="renders_tokenizer_ab")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def find_track(tracks_dir: Path, substr: str) -> Path:
    """
    Args:
      tracks_dir (Path): folder of source mp3s.
      substr (str): substring to match against the file name.

    Returns:
      Path: the first matching mp3, sorted by name.
    """
    hits = sorted(p for p in tracks_dir.glob("*.mp3") if substr in p.name)
    if not hits:
        raise ValueError(f"no track matched '{substr}' in {tracks_dir}")
    return hits[0]


def load_generator(path: str, device: str) -> torch.nn.Module:
    """
    Args:
      path (str): train.py Lightning checkpoint (EMA applied by load_module).
      device (str): torch device string.

    Returns:
      torch.nn.Module: the Lightning module in eval mode on `device`.
    """
    module = load_module(
        path,
        build_learning_params(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        build_loss_aggregator(),
        per_level_codebooks=True,
    )
    return module.to(device).eval()


class OnnxRoundTrip:
    """
    encoder.onnx -> indices -> decoder.onnx: the tokenizer exactly as exported
    (and as the AR stage consumed it).

    Args:
      onnx_dir (Path): folder holding encoder.onnx and decoder.onnx.
    """

    def __init__(self, onnx_dir: Path) -> None:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), providers=providers)
        self._dec = ort.InferenceSession(str(onnx_dir / "decoder.onnx"), providers=providers)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
          clip (torch.Tensor): (1, L) waveform, L a multiple of HOP.

        Returns:
          torch.Tensor: (1, L) reconstruction.
        """
        wav = clip.reshape(1, 1, -1).cpu().numpy().astype(np.float32)
        idx = self._enc.run(None, {"waveform": wav})[0]
        out = self._dec.run(None, {"indices": idx})[0]
        return torch.from_numpy(out).reshape(1, -1)[:, : clip.shape[-1]]


def main() -> None:
    """Write A plus one B per checkpoint for every span, then print the proxies."""
    args = parse_args()
    named = dict(item.split("=", 1) for item in args.ckpts)
    modules = {
        name: (OnnxRoundTrip(Path(args.onnx_dir)) if path == "onnx" else load_generator(path, args.device))
        for name, path in named.items()
    }
    tracks_dir = Path(args.tracks_dir).expanduser()
    slices_dir = Path(args.slices_dir).expanduser()
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    span = int(args.seconds * SAMPLE_RATE) // HOP * HOP

    rows: list[tuple[str, str, dict[str, float], float, float]] = []
    for substr in args.tracks:
        path = find_track(tracks_dir, substr)
        wav = load_track_audio(path, SAMPLE_RATE, HOP, slices_dir)
        start_smp = (
            int(args.start_sec * SAMPLE_RATE)
            if args.start_sec is not None
            else int(args.start_frac * wav.shape[-1])
        )
        start = min(start_smp // HOP * HOP, wav.shape[-1] - span)
        clean = wav[:, start : start + span].float()

        variants = {"A_original": clean}
        with torch.no_grad():
            for name, module in modules.items():
                if isinstance(module, OnnxRoundTrip):
                    variants[f"B_{name}"] = module(clean)
                else:
                    variants[f"B_{name}"] = reconstruct(module, clean.to(args.device)).cpu()
        matched = {
            name: normalize_lufs(sig.reshape(-1).numpy(), SAMPLE_RATE, TARGET_LUFS)
            for name, sig in variants.items()
        }
        stem = path.stem[:28].strip().replace(" ", "_")
        ref = torch.from_numpy(matched["A_original"]).reshape(1, 1, -1)
        for name in sorted(matched):
            torchaudio.save(
                str(out_dir / f"{stem}__{name}.wav"),
                torch.from_numpy(matched[name]).reshape(1, -1),
                SAMPLE_RATE,
            )
            pred = torch.from_numpy(matched[name]).reshape(1, 1, -1)
            rows.append(
                (
                    stem[:20],
                    name,
                    features(matched[name], SAMPLE_RATE),
                    spectral_rmse(matched["A_original"], matched[name], SAMPLE_RATE),
                    float(multi_res_stft_distance(pred, ref)),
                )
            )

    print(f"\n{'track':<20} {'variant':<14} {'crest':>7} {'beat':>6} {'cent':>6} {'hf':>6} {'rmse':>7} {'mrstft':>7}")
    print("-" * 78)
    for name, variant, feat, rmse, mrs in rows:
        print(
            f"{name:<20} {variant:<14} {feat['crest']:7.2f} {feat['beat']:6.3f} "
            f"{feat['cent']:6.0f} {feat['hf']:6.3f} {rmse:7.4f} {mrs:7.4f}"
        )
    print(f"\naudio -> {out_dir}   (same span, LUFS {TARGET_LUFS}; compare A vs each B)")


if __name__ == "__main__":
    main()
