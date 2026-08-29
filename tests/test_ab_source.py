"""
The A/B toggle. QtCore only: no widgets, no audio device.

The property everything else rests on is that toggling does not move the
playhead. If it did, the fidelity tier would be judged on two clips heard at
different moments, which is not the comparison anyone thinks they are making.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QIODevice

from ab_harness.viewmodel.ab_source import BYTES_PER_FRAME, CHANNELS, AbSource

RATE = 1000


def _read(source: AbSource, frames: int) -> np.ndarray:
    """
    Read and de-interleave, so the tests read in mono source frames.

    Args:
      source (AbSource): device under test.
      frames (int): how many source frames to pull.

    Returns:
      np.ndarray: (n,) int16 left-channel samples actually served.
    """
    return _read_stereo(source, frames)[0::CHANNELS]


def _read_stereo(source: AbSource, frames: int) -> np.ndarray:
    """
    Args:
      source (AbSource): device under test.
      frames (int): how many source frames to pull.

    Returns:
      np.ndarray: (n * CHANNELS,) interleaved int16 samples.
    """
    return np.frombuffer(source.readData(frames * BYTES_PER_FRAME), dtype=np.int16)


@pytest.fixture
def source(qapp) -> AbSource:
    """
    Args:
      qapp: the session QCoreApplication.

    Returns:
      AbSource: an open device holding two constant, distinguishable buffers.
    """
    device = AbSource(RATE, crossfade_ms=0.0)
    device.set_pair(
        np.full(100, 1000, dtype=np.int16), np.full(100, -1000, dtype=np.int16)
    )
    device.open(QIODevice.OpenModeFlag.ReadOnly)
    return device


def test_serves_the_active_buffer(source: AbSource) -> None:
    assert np.all(_read(source, 10) == 1000)
    source.toggle()
    assert np.all(_read(source, 10) == -1000)


def test_toggling_does_not_move_the_playhead(source: AbSource) -> None:
    _read(source, 30)
    assert source.position == 30
    source.toggle()
    assert source.position == 30
    _read(source, 10)
    assert source.position == 40


def test_toggling_to_the_active_side_is_a_no_op(source: AbSource) -> None:
    emitted: list[int] = []
    source.toggled.connect(emitted.append)
    source.toggle(0)
    assert emitted == []
    source.toggle(1)
    assert emitted == [1]


def test_crossfade_spans_the_configured_window(qapp) -> None:
    device = AbSource(RATE, crossfade_ms=10.0)  # 10 samples at 1000 Hz
    device.set_pair(
        np.full(60, 1000, dtype=np.int16), np.full(60, -1000, dtype=np.int16)
    )
    device.open(QIODevice.OpenModeFlag.ReadOnly)
    _read(device, 10)
    device.toggle()
    data = _read(device, 30)
    # first sample still weighted towards the outgoing buffer, last fully arrived
    assert data[0] > 0
    assert data[9] < 0
    assert np.all(data[10:] == -1000)
    # monotone descent through the fade, so there is no click
    assert np.all(np.diff(data[:10]) < 0)


def test_eof_returns_empty_and_emits_finished(source: AbSource) -> None:
    finished: list[bool] = []
    source.finished.connect(lambda: finished.append(True))
    _read(source, 100)
    assert source.readData(200) == b""
    assert finished == [True]
    # the signal fires once, not on every subsequent read
    source.readData(200)
    assert finished == [True]


def test_restart_rewinds_without_changing_sides(source: AbSource) -> None:
    source.toggle()
    _read(source, 50)
    source.restart()
    assert source.position == 0
    assert source.active == 1
    assert np.all(_read(source, 5) == -1000)


def test_seek_clamps_to_the_clip(source: AbSource) -> None:
    source.seek_seconds(-5.0)
    assert source.position == 0
    source.seek_seconds(999.0)
    assert source.position == 100
    assert source.readData(10) == b""


def test_buffers_of_different_lengths_are_padded_to_match(qapp) -> None:
    device = AbSource(RATE)
    device.set_pair(np.full(50, 500, dtype=np.int16), np.full(20, -500, dtype=np.int16))
    device.open(QIODevice.OpenModeFlag.ReadOnly)
    assert device.duration == pytest.approx(50 / RATE)
    device.toggle()
    data = _read(device, 50)
    assert data.size == 50
    assert np.all(data[20:] == 0)


def test_a_zero_length_read_returns_nothing(source: AbSource) -> None:
    assert source.readData(0) == b""
    # a request too small for one stereo frame cannot be served either
    assert source.readData(BYTES_PER_FRAME - 1) == b""
    assert source.position == 0


def test_output_is_stereo_with_both_channels_carrying_the_signal(
    source: AbSource,
) -> None:
    """
    Regression: the format was mono, and most backends route a mono stream to
    the first channel only -- the clip played in the left ear alone.
    """
    data = _read_stereo(source, 10)
    assert data.size == 10 * CHANNELS
    left, right = data[0::CHANNELS], data[1::CHANNELS]
    assert np.array_equal(left, right), "channels differ; output is not centred"
    assert np.all(left == 1000)


def test_a_frames_worth_of_bytes_is_reported_for_both_channels(
    source: AbSource,
) -> None:
    assert source.bytesAvailable() == 100 * BYTES_PER_FRAME
    _read(source, 40)
    assert source.bytesAvailable() == 60 * BYTES_PER_FRAME
