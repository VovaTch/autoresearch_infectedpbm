"""
Mel spectrogram in numpy, for the UI process.

The repo already computes mels three ways -- torchaudio in prepare.py, the
training loss stack in train.py, matplotlib in listen_compare.ipynb -- and not
one of them can be used here: the UI process must not import torch, and librosa
drags numba in for what amounts to twenty lines of triangle arithmetic. So this
is the fourth, and it is the small one.

Resolution is chosen from the width being drawn rather than fixed. A 90 s clip
at hop 512 is 7757 frames of 1025 bins, most of which land on the same pixel
column; picking the hop from the column count instead keeps the cost flat in
clip length and the picture identical. The FFT size then follows the hop so
windows always overlap -- a hop wider than the window would step over transients
entirely, which on this material is most of what there is to see.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_MELS = 128
MIN_FFT = 2048
MIN_HOP = 256
TOP_DB = 80.0


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """
    Args:
      hz (np.ndarray | float): frequency in Hz.

    Returns:
      np.ndarray | float: the HTK mel scale value.
    """
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """
    Args:
      mel (np.ndarray | float): HTK mel scale value.

    Returns:
      np.ndarray | float: frequency in Hz.
    """
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = DEFAULT_MELS,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Triangular mel filters over the rfft bins.

    Deliberately unnormalized: dividing each filter by its bandwidth (the Slaney
    convention) makes the narrow low bands as loud as the wide high ones, which
    for a *display* means amplifying the quietest part of the picture into the
    brightest. Summing raw magnitude keeps the image reading like the spectrum
    it came from.

    Cached: the bank depends only on its arguments, and it is rebuilt on every
    resize otherwise.

    Args:
      sample_rate (int): samples per second.
      n_fft (int): FFT size the bins came from.
      n_mels (int): number of mel bands.
      fmin (float): lowest band edge in Hz.
      fmax (float | None): highest band edge in Hz; Nyquist when None.

    Returns:
      np.ndarray: (n_mels, n_fft // 2 + 1) float32 weights.
    """
    top = float(fmax) if fmax is not None else sample_rate / 2.0
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    # n_mels + 2 edges give n_mels overlapping triangles
    edges = np.asarray(
        mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(top), n_mels + 2))
    )
    weights = np.zeros((n_mels, bins.size), dtype=np.float32)
    for band in range(n_mels):
        lo, mid, hi = edges[band], edges[band + 1], edges[band + 2]
        rising = (bins - lo) / max(mid - lo, 1e-9)
        falling = (hi - bins) / max(hi - mid, 1e-9)
        weights[band] = np.clip(np.minimum(rising, falling), 0.0, None)
    return weights


def plan_geometry(samples: int, columns: int) -> tuple[int, int]:
    """
    Choose an FFT size and hop for drawing `samples` into `columns` pixels.

    Args:
      samples (int): waveform length.
      columns (int): pixel columns available.

    Returns:
      tuple[int, int]: (n_fft, hop). The hop never drops below MIN_HOP, and the
        window is always at least twice the hop so frames overlap.
    """
    hop = max(MIN_HOP, -(-samples // max(1, columns)))
    n_fft = MIN_FFT
    while n_fft < 2 * hop:
        n_fft *= 2
    return n_fft, hop


def mel_spectrogram(
    pcm: np.ndarray,
    sample_rate: int,
    columns: int = 1024,
    n_mels: int = DEFAULT_MELS,
    top_db: float = TOP_DB,
) -> np.ndarray:
    """
    Log-magnitude mel spectrogram, scaled to [0, 1] for drawing.

    Frames are cut in blocks rather than all at once: a 90 s clip is four million
    samples, and one rfft over every frame at once is a 130 MB complex array for
    a picture 800 pixels wide.

    Args:
      pcm (np.ndarray): (N,) int16 or float waveform.
      sample_rate (int): samples per second.
      columns (int): target number of time columns.
      n_mels (int): number of mel bands.
      top_db (float): dynamic range below the peak that maps to 0.0.

    Returns:
      np.ndarray: (n_mels, T) float32 in [0, 1], low bands first. (n_mels, 0)
        for an empty or too-short buffer, so callers need no special case.
    """
    if pcm.size == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)
    scale = 32768.0 if np.issubdtype(pcm.dtype, np.integer) else 1.0
    signal = np.asarray(pcm, dtype=np.float32) / scale

    n_fft, hop = plan_geometry(signal.size, columns)
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    frames = 1 + (signal.size - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    bank = mel_filterbank(sample_rate, n_fft, n_mels)

    # A strided view costs nothing; the rfft over it is what has to be blocked.
    view = np.lib.stride_tricks.as_strided(
        signal,
        shape=(frames, n_fft),
        strides=(signal.strides[0] * hop, signal.strides[0]),
        writeable=False,
    )
    out = np.empty((n_mels, frames), dtype=np.float32)
    block = 512
    for start in range(0, frames, block):
        stop = min(start + block, frames)
        mag = np.abs(np.fft.rfft(view[start:stop] * window, axis=-1)).astype(np.float32)
        out[:, start:stop] = bank @ mag.T

    db = 20.0 * np.log10(np.maximum(out, 1e-10))
    ceiling = float(db.max())
    return np.clip((db - (ceiling - top_db)) / top_db, 0.0, 1.0).astype(np.float32)


__all__ = [
    "DEFAULT_MELS",
    "hz_to_mel",
    "mel_filterbank",
    "mel_spectrogram",
    "mel_to_hz",
    "plan_geometry",
]
