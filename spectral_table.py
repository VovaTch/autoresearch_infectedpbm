"""Spectral/dynamics table over rendered wavs, grouped by model tag.

The GAN legs in this lineage move things the recon metrics cannot see: high
frequency tilt (the win) and crest factor (the damage). This computes both from
the wavs in ./renders/, so the verdict on an adversarial run is reproducible
instead of ad-hoc.

Tag convention from render_samples.py: clip<N>_<name>_<TAG>.wav, TAG=ORIG for
the reference. All wavs are already peak-normalized by the renderer.

NOTE: all arms are recomputed here with one method (Hann window), so numbers are
internally comparable. Historic tables in results.tsv used slightly different
windowing and their absolute values will not line up; compare within a table.
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict

import torch
import torchaudio

SR = 44100
N_FFT = 2048
HOP = 512
RENDERS = os.path.join(os.path.dirname(__file__), "renders")
BANDS: dict[str, tuple[float, float]] = {
    "low<250": (0.0, 250.0),
    "mid.25-5k": (250.0, 5000.0),
    "hf>5k": (5000.0, SR / 2),
}


def analyze(wav: torch.Tensor) -> dict[str, float]:
    """Per-clip spectral + dynamics stats.

    Args:
      wav (torch.Tensor): [1, L] peak-normalized mono waveform.

    Returns:
      dict[str, float]: band mean magnitudes, centroid (kHz), crest (dB), rms.
    """
    x = wav.squeeze(0)
    window = torch.hann_window(N_FFT)
    mag = torch.stft(
        x, N_FFT, HOP, window=window, return_complex=True
    ).abs()  # [F, T]
    freqs = torch.linspace(0, SR / 2, mag.shape[0])

    out: dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        out[name] = float(mag[sel].mean())

    power = mag.pow(2)
    centroid = (power * freqs[:, None]).sum() / (power.sum() + 1e-12)
    out["centroid_kHz"] = float(centroid) / 1000.0

    rms = float(x.pow(2).mean().sqrt())
    out["rms"] = rms
    out["crest_dB"] = 20.0 * torch.log10(torch.tensor(float(x.abs().max()) / (rms + 1e-12))).item()
    return out


def main() -> None:
    by_tag: dict[str, list[dict[str, float]]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RENDERS, "*.wav"))):
        tag = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.transforms.Resample(sr, SR)(wav)
        by_tag[tag].append(analyze(wav))

    if not by_tag:
        print(f"No wavs in {RENDERS}")
        return

    cols = ["low<250", "mid.25-5k", "hf>5k", "centroid_kHz", "crest_dB", "rms"]
    tags = ["ORIG"] + sorted(t for t in by_tag if t != "ORIG")
    n = len(next(iter(by_tag.values())))
    print(f"Spectral table over {n} clips per tag (Hann, n_fft={N_FFT}, hop={HOP})\n")
    print(f"{'tag':<16}" + "".join(f"{c:>14}" for c in cols))
    print("-" * (16 + 14 * len(cols)))
    for tag in tags:
        if tag not in by_tag:
            continue
        rows = by_tag[tag]
        means = {c: sum(r[c] for r in rows) / len(rows) for c in cols}
        print(f"{tag:<16}" + "".join(f"{means[c]:>14.5f}" for c in cols))
    print(
        "\nhf>5k toward ORIG = tilt restored (the GAN's only proven win).\n"
        "crest_dB above ORIG = transient artifacts; rms below = the audible"
        " loudness drop.\nBoth are the GAN's known damage mode."
    )


if __name__ == "__main__":
    main()
