"""
Objective quality read over a render directory, grouped by filename cell.

Identification says the conditioning is being followed; it says nothing about
whether the result is listenable. High guidance is known to make output
spectrally conservative, so quality has to be measured on the same clips.

  beat    onset-envelope autocorrelation peak over 60-200 BPM, 0-1. The
          reference tracks sit near 0.8; cold generations near 0.4.
  crest   peak-to-RMS in dB. Collapsed dynamics read low.
  cent    spectral centroid in Hz.
  hf      fraction of energy above 5 kHz.

Usage:
  uv run python score_renders.py renders_cfg_temp
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torchaudio


def onset_env(pcm: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """
    Half-wave-rectified spectral flux.

    Args:
      pcm (np.ndarray): (N,) mono waveform.
      sr (int): sample rate.

    Returns:
      tuple[np.ndarray, float]: (F,) onset envelope and its frame rate in Hz.
    """
    n, hop = 1024, 256
    win = np.hanning(n)
    mag = np.array(
        [
            np.abs(np.fft.rfft(pcm[i : i + n] * win))
            for i in range(0, max(1, len(pcm) - n), hop)
        ]
    )
    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    return flux - flux.mean(), sr / hop


def beat_strength(pcm: np.ndarray, sr: int) -> float:
    """
    Args:
      pcm (np.ndarray): (N,) mono waveform.
      sr (int): sample rate.

    Returns:
      float: strongest onset autocorrelation peak in the 60-200 BPM lag band,
        normalised by lag 0. Near 0 is arrhythmic, near 1 is strictly periodic.
    """
    env, fps = onset_env(pcm, sr)
    if env.size < 8 or not np.any(env):
        return 0.0
    ac = np.correlate(env, env, mode="full")[env.size - 1 :]
    if ac[0] <= 0:
        return 0.0
    lo, hi = int(fps * 60.0 / 200.0), int(fps * 60.0 / 60.0)
    band = ac[lo : hi + 1]
    return float(band.max() / ac[0]) if band.size else 0.0


def features(pcm: np.ndarray, sr: int) -> dict[str, float]:
    """
    Args:
      pcm (np.ndarray): (N,) mono waveform.
      sr (int): sample rate.

    Returns:
      dict[str, float]: beat, crest (dB), centroid (Hz), hf fraction.
    """
    rms = float(np.sqrt(np.mean(pcm**2)))
    crest = 20 * np.log10(np.abs(pcm).max() / rms) if rms > 0 else 0.0
    n, hop = 2048, 512
    spec = np.mean(
        np.array(
            [
                np.abs(np.fft.rfft(pcm[i : i + n] * np.hanning(n)))
                for i in range(0, max(1, len(pcm) - n), hop)
            ]
        ),
        axis=0,
    )
    freqs = np.fft.rfftfreq(n, 1 / sr)
    tot = spec.sum()
    return {
        "beat": beat_strength(pcm, sr),
        "crest": float(crest),
        "cent": float((spec * freqs).sum() / tot) if tot > 0 else 0.0,
        "hf": float(spec[freqs > 5000].sum() / tot) if tot > 0 else 0.0,
    }


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "renders_cfg_temp")
    rows: dict[str, list[dict[str, float]]] = {}
    for wav in sorted(root.glob("*.wav")):
        audio, sr = torchaudio.load(str(wav))
        pcm = audio.mean(0).numpy()
        stem = wav.stem
        if stem.startswith("REAL"):
            key = "REAL"
        elif stem.startswith("null"):
            key = re.sub(r"_s\d+$", "", stem)
        else:
            m = re.search(r"_(cfg[\d.]+_T[\d.]+)", stem)
            key = m.group(1) if m else stem
        rows.setdefault(key, []).append(features(pcm, sr))

    def sort_key(k: str) -> tuple[float, float, float]:
        if k == "REAL":
            return (-1.0, 0.0, 0.0)
        m = re.search(r"T([\d.]+)", k)
        t = float(m.group(1)) if m else 0.0
        c = re.search(r"cfg([\d.]+)", k)
        return (1.0 / t if t else 0.0, float(c.group(1)) if c else -1.0, 0.0)

    print(f"{'group':<16} {'n':>3} {'beat':>15} {'crest dB':>13} "
          f"{'centroid':>10} {'hf>5k':>7}")
    for key in sorted(rows, key=sort_key):
        vals = rows[key]
        g = lambda f: np.array([v[f] for v in vals])
        b, c = g("beat"), g("crest")
        print(f"{key:<16} {len(vals):>3} {b.mean():7.3f} ±{b.std(ddof=1) if len(vals)>1 else 0:.3f} "
              f"{c.mean():7.2f} ±{c.std(ddof=1) if len(vals)>1 else 0:.2f} "
              f"{g('cent').mean():10.0f} {g('hf').mean():7.3f}")


if __name__ == "__main__":
    main()
