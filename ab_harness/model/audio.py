"""
Loudness matching for A/B playback (PLAN section 7.7).

A reward model trained on raw audio pairs learns loudness and brightness,
because louder reliably wins a blind A/B; the generator then learns to be a
compressor. This is preattentive rather than a matter of taste, which is why
every mastering comparison tool auto level-matches. Normalizing here is
insurance, not a judgement about anyone's ear.

render_samples.loudness_match solves the neighbouring problem by RMS-matching to
a reference clip and applying one shared headroom scale; its own comment records
identical audio scoring 0.2153 peak-normalized against 0.0176 RMS-matched. The
argument carries over, and gated K-weighted LUFS against an absolute target is
the stricter form of it: it does not depend on whichever clip happened to be
measured first, and it weights the bands the ear actually judges loudness by.
"""

from __future__ import annotations

import numpy as np
import pyloudnorm as pyln

DEFAULT_TARGET_LUFS = -23.0
PEAK_CEILING = 0.99
# BS.1770 needs 400 ms for one gating block; below that the meter cannot run.
MIN_SECONDS = 0.4


def measure_lufs(pcm: np.ndarray, sample_rate: int) -> float:
    """
    Measure integrated loudness.

    Args:
      pcm (np.ndarray): (N,) float32 mono waveform.
      sample_rate (int): samples per second.

    Returns:
      float: integrated loudness in LUFS, or -inf for silence and for buffers
        too short to gate.
    """
    if pcm.size < int(MIN_SECONDS * sample_rate) or not np.any(pcm):
        return float("-inf")
    meter = pyln.Meter(sample_rate)
    return float(meter.integrated_loudness(pcm.astype(np.float64)))


def normalize_lufs(
    pcm: np.ndarray,
    sample_rate: int,
    target: float = DEFAULT_TARGET_LUFS,
    ceiling: float = PEAK_CEILING,
) -> np.ndarray:
    """
    Scale a waveform to a target loudness without letting it clip.

    The peak guard can leave a clip quieter than the target. That is the correct
    trade: a clipped clip loses the A/B for a reason unrelated to the model,
    which is exactly the confound this module exists to remove. Pick a target
    low enough that the guard rarely engages.

    Args:
      pcm (np.ndarray): (N,) float32 mono waveform.
      sample_rate (int): samples per second.
      target (float): desired integrated loudness in LUFS.
      ceiling (float): maximum absolute sample value after scaling.

    Returns:
      np.ndarray: (N,) float32 scaled waveform. Silence is returned untouched.
    """
    loudness = measure_lufs(pcm, sample_rate)
    if not np.isfinite(loudness):
        return pcm.astype(np.float32, copy=True)
    gain = float(10.0 ** ((target - loudness) / 20.0))
    peak = float(np.abs(pcm).max())
    if peak * gain > ceiling:
        gain = ceiling / peak
    return (pcm * gain).astype(np.float32)


def to_int16(pcm: np.ndarray) -> np.ndarray:
    """
    Convert a float waveform to the int16 PCM QAudioSink consumes.

    Args:
      pcm (np.ndarray): (N,) float32 waveform in [-1, 1].

    Returns:
      np.ndarray: (N,) int16 samples.
    """
    clipped = np.clip(pcm, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


QUIET_FLOOR_DBFS = -45.0


def envelope(pcm: np.ndarray, buckets: int) -> np.ndarray:
    """
    Peak envelope of a waveform, for drawing and for emptiness tests.

    Peak rather than RMS: a clip that is silent apart from a few transients is
    exactly what the rater wants to spot, and RMS averages that away.

    Args:
      pcm (np.ndarray): (N,) int16 or float waveform.
      buckets (int): number of columns to reduce to.

    Returns:
      np.ndarray: (buckets,) float32 peaks in [0, 1]. All zeros for an empty
        buffer or a non-positive bucket count.
    """
    buckets = max(0, int(buckets))
    if buckets == 0:
        return np.zeros(0, dtype=np.float32)
    if pcm.size == 0:
        return np.zeros(buckets, dtype=np.float32)
    scale = 32768.0 if np.issubdtype(pcm.dtype, np.integer) else 1.0
    edges = np.linspace(0, pcm.size, buckets + 1).astype(np.int64)
    mag = np.abs(pcm.astype(np.float32)) / scale
    # A bucket narrower than one sample would be empty; max.reduceat needs at
    # least one element per slice, so widen the last edge instead.
    edges[-1] = pcm.size
    starts = np.minimum(edges[:-1], pcm.size - 1)
    peaks = np.maximum.reduceat(mag, starts)
    return np.clip(peaks, 0.0, 1.0).astype(np.float32)


def fill_fraction(pcm: np.ndarray, floor_dbfs: float = QUIET_FLOOR_DBFS) -> float:
    """
    How much of a clip is above the near-silence floor.

    This is the "mostly empty" detector: a generation that trails off into
    silence, or never starts, scores near 0 and can be filtered out of the
    worklist before it costs a listen.

    Args:
      pcm (np.ndarray): (N,) int16 or float waveform.
      floor_dbfs (float): level below which a window counts as empty.

    Returns:
      float: fraction of 50 ms windows peaking above the floor, in [0, 1].
    """
    if pcm.size == 0:
        return 0.0
    # 50 ms windows at any plausible rate; the count, not the rate, is what
    # matters, and this keeps a 10 s and a 90 s clip on the same scale.
    buckets = max(1, int(round(pcm.size / (0.05 * 44100))))
    peaks = envelope(pcm, buckets)
    return float(np.mean(peaks > 10.0 ** (floor_dbfs / 20.0)))
