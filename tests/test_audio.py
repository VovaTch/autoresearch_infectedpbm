"""Loudness normalization: the one guard against the loudest clip winning."""

from __future__ import annotations

import numpy as np
import pytest

from ab_harness.model.audio import (
    PEAK_CEILING,
    envelope,
    fill_fraction,
    measure_lufs,
    normalize_lufs,
    to_int16,
)

SR = 44100


def _noise(seconds: float, scale: float, seed: int = 0) -> np.ndarray:
    """
    Args:
      seconds (float): duration.
      scale (float): amplitude.
      seed (int): RNG seed.

    Returns:
      np.ndarray: (N,) float32 noise.
    """
    rng = np.random.default_rng(seed)
    return (rng.normal(0, scale, int(seconds * SR))).astype(np.float32)


@pytest.mark.parametrize("scale", [0.01, 0.05, 0.2])
def test_normalize_hits_target(scale: float) -> None:
    out = normalize_lufs(_noise(3.0, scale), SR, target=-23.0)
    assert measure_lufs(out, SR) == pytest.approx(-23.0, abs=0.1)


def test_two_clips_of_different_loudness_end_up_matched() -> None:
    quiet = normalize_lufs(_noise(3.0, 0.01, seed=1), SR, target=-23.0)
    loud = normalize_lufs(_noise(3.0, 0.4, seed=2), SR, target=-23.0)
    assert measure_lufs(quiet, SR) == pytest.approx(measure_lufs(loud, SR), abs=0.2)


def test_peak_guard_prevents_clipping() -> None:
    # a very quiet clip needs a large gain, which would clip its own peaks
    out = normalize_lufs(_noise(3.0, 1e-4), SR, target=0.0)
    assert np.abs(out).max() <= PEAK_CEILING + 1e-6


def test_silence_is_returned_untouched() -> None:
    silence = np.zeros(SR, dtype=np.float32)
    out = normalize_lufs(silence, SR)
    assert out.shape == silence.shape
    assert not np.any(out)
    assert measure_lufs(silence, SR) == float("-inf")


def test_buffer_shorter_than_a_gating_block_is_passed_through() -> None:
    tiny = _noise(0.1, 0.1)
    out = normalize_lufs(tiny, SR)
    assert np.allclose(out, tiny)


def test_to_int16_clamps_out_of_range_input() -> None:
    out = to_int16(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))
    assert out.dtype == np.int16
    assert out.min() >= -32767 and out.max() <= 32767


# -- envelope and emptiness --------------------------------------------------


def test_envelope_reduces_to_the_requested_width() -> None:
    peaks = envelope(to_int16(_noise(2.0, 0.3)), 64)
    assert peaks.shape == (64,)
    assert peaks.dtype == np.float32
    assert np.all((peaks >= 0.0) & (peaks <= 1.0))


def test_envelope_survives_more_buckets_than_samples() -> None:
    # a 90 s clip drawn into a 200 px strip is the normal case; the reverse is
    # what a resize during startup produces, and it must not raise
    assert envelope(np.zeros(3, dtype=np.int16), 32).shape == (32,)
    assert envelope(np.zeros(0, dtype=np.int16), 8).tolist() == [0.0] * 8


def test_fill_fraction_separates_a_full_clip_from_a_mostly_empty_one() -> None:
    """
    The whole point of the worklist's filter: a generation that never starts or
    trails off into silence should be visible without listening to it.
    """
    full = to_int16(_noise(4.0, 0.2))
    mostly_empty = np.zeros(4 * SR, dtype=np.int16)
    mostly_empty[: SR // 4] = full[: SR // 4]
    assert fill_fraction(full) > 0.95
    assert fill_fraction(mostly_empty) < 0.1
    assert fill_fraction(np.zeros(SR, dtype=np.int16)) == 0.0
