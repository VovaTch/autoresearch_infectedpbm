"""
Whole-track encode -> decode round trip on arbitrary mp3s through the exported
ONNX tokenizer, for listening to out-of-distribution material.

Unlike render_tokenizer_ab.py this takes explicit file paths (tracks outside
the training cache), runs the ENTIRE track in overlap-trimmed chunks, and
writes the reconstruction at its native level (no LUFS / peak matching, so a
gain error shows up instead of being hidden). Metrics are computed on
non-overlapping 32768-sample slices, the same unit the trainer scores, so the
numbers sit on the same scale as results.tsv / test/*.

Usage:
  uv run python roundtrip_external.py ~/Downloads/a.mp3 ~/Downloads/b.mp3
  uv run python roundtrip_external.py a.mp3 --onnx-dir onnx --out renders_external
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

from generate_ar import decode_tokens
from prepare import cdpam_distance, make_cdpam_evaluator, multi_res_stft_distance
from probe_copy_synthesis import spectral_rmse
from score_renders import features
from train_ar import encode_chunked, load_track_audio

REPO = Path(__file__).resolve().parent
SAMPLE_RATE = 44100
HOP = 256
SLICE = 32768
SILENCE_DBFS = -60.0


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", nargs="+", help="mp3 paths (outside the training cache)")
    parser.add_argument("--onnx-dir", default="onnx")
    parser.add_argument("--out", default="renders_external")
    parser.add_argument("--chunk-frames", type=int, default=4096)
    parser.add_argument("--margin", type=int, default=256)
    parser.add_argument("--cdpam-batch", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def round_trip(
    onnx_dir: Path, wav: torch.Tensor, chunk_frames: int, margin: int
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Args:
      onnx_dir (Path): folder holding encoder.onnx and decoder.onnx.
      wav (torch.Tensor): (1, L) mono waveform, L a multiple of HOP.
      chunk_frames (int): frames per encode/decode window.
      margin (int): context frames discarded on each side of a window.

    Returns:
      tuple[torch.Tensor, np.ndarray]: (1, L) reconstruction and (T, R) indices.
    """
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), providers=providers)
    idx = encode_chunked(enc, wav, HOP, chunk_frames, margin)
    del enc
    dec = ort.InferenceSession(str(onnx_dir / "decoder.onnx"), providers=providers)
    out = decode_tokens(dec, torch.from_numpy(idx).unsqueeze(0), HOP, chunk_frames, margin)
    return torch.from_numpy(out).reshape(1, -1)[:, : wav.shape[-1]], idx


def slice_metrics(
    ref: torch.Tensor, pred: torch.Tensor, evaluator, batch: int
) -> dict[str, float]:
    """
    Score on non-overlapping SLICE-sample windows, averaged over the track,
    ignoring slices quieter than SILENCE_DBFS.

    Args:
      ref (torch.Tensor): (1, L) original.
      pred (torch.Tensor): (1, L) reconstruction.
      evaluator: cdpam.CDPAM instance.
      batch (int): slices per cdpam forward.

    Returns:
      dict[str, float]: cdpam, mrstft, n_slices.
    """
    n = ref.shape[-1] // SLICE
    r = ref[0, : n * SLICE].reshape(n, 1, SLICE)
    p = pred[0, : n * SLICE].reshape(n, 1, SLICE)
    # drop digital-silence slices: spectral convergence divides by ~0 there
    keep = r.pow(2).mean(dim=(1, 2)).sqrt() > 10 ** (SILENCE_DBFS / 20)
    r, p = r[keep], p[keep]
    n = int(keep.sum())
    cd, mr = [], []
    for i in range(0, n, batch):
        rb, pb = r[i : i + batch], p[i : i + batch]
        cd.append(float(cdpam_distance(evaluator, pb, rb)) * rb.shape[0])
        mr.append(float(multi_res_stft_distance(pb, rb)) * rb.shape[0])
    return {"cdpam": sum(cd) / n, "mrstft": sum(mr) / n, "n_slices": float(n)}


def main() -> None:
    """Round-trip every track, write A/B wavs, print per-track scores."""
    args = parse_args()
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator = make_cdpam_evaluator(args.device)
    onnx_dir = Path(args.onnx_dir)

    rows: list[tuple[str, dict[str, float], float, float, dict[str, float], dict[str, float]]] = []
    for track in args.tracks:
        path = Path(track).expanduser()
        wav = load_track_audio(path, SAMPLE_RATE, HOP)
        print(f"{path.name}: {wav.shape[-1] / SAMPLE_RATE:.1f} s, {wav.shape[-1] // HOP} frames")
        with torch.no_grad():
            rec, idx = round_trip(onnx_dir, wav, args.chunk_frames, args.margin)
        stem = path.stem[:40].strip().replace(" ", "_")
        torchaudio.save(str(out_dir / f"{stem}__A_original.wav"), wav, SAMPLE_RATE)
        torchaudio.save(str(out_dir / f"{stem}__B_roundtrip.wav"), rec, SAMPLE_RATE)
        np.save(out_dir / f"{stem}__tokens.npy", idx)

        a, b = wav.reshape(-1).numpy(), rec.reshape(-1).numpy()
        gain_db = 20 * np.log10(np.sqrt(np.mean(b**2)) / np.sqrt(np.mean(a**2)))
        usage = {f"L{k}": len(np.unique(idx[:, k])) / 2048 for k in range(idx.shape[1])}
        metrics = slice_metrics(wav, rec, evaluator, args.cdpam_batch)
        rows.append(
            (stem[:30], metrics, spectral_rmse(a, b, SAMPLE_RATE), gain_db, features(a, SAMPLE_RATE), features(b, SAMPLE_RATE))
        )
        rows[-1][1].update(usage)

    print(f"\n{'track':<30} {'cdpam':>7} {'mrstft':>7} {'rmse':>7} {'gain_dB':>8} {'crestA':>7} {'crestB':>7} {'hfA':>6} {'hfB':>6} {'codebook use L0/L1/L2':>22}")
    print("-" * 122)
    for name, m, rmse, gain, fa, fb in rows:
        use = "/".join(f"{m[f'L{k}']:.2f}" for k in range(3) if f"L{k}" in m)
        print(
            f"{name:<30} {m['cdpam']:7.4f} {m['mrstft']:7.4f} {rmse:7.4f} {gain:8.2f} "
            f"{fa['crest']:7.2f} {fb['crest']:7.2f} {fa['hf']:6.3f} {fb['hf']:6.3f} {use:>22}"
        )
    print(f"\naudio -> {out_dir}   (native level; compare A vs B)")


if __name__ == "__main__":
    main()
