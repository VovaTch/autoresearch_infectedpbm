"""
Playback transport.

Skipped wholesale when the machine has no audio output, so the suite still runs
on a headless box; where a device does exist these pin the format, which is
where the one-ear bug lived.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtMultimedia")

from ab_harness.viewmodel.ab_source import CHANNELS
from ab_harness.viewmodel.player_vm import PlayerViewModel


@pytest.fixture
def player(qapp) -> PlayerViewModel:
    """
    Args:
      qapp: the session QApplication.

    Returns:
      PlayerViewModel: a transport, or a skip when no output device exists.
    """
    from PySide6.QtMultimedia import QMediaDevices

    if QMediaDevices.defaultAudioOutput().isNull():
        pytest.skip("no audio output device on this machine")
    return PlayerViewModel(sample_rate=44100, crossfade_ms=5.0)


def test_the_sink_asks_for_stereo(player: PlayerViewModel) -> None:
    """
    Regression: the format was mono, and the output devices here are stereo
    ("Built-in Audio Analog Stereo"), so the clip played in the left ear alone.
    """
    fmt = player._sink.format()
    assert fmt.channelCount() == CHANNELS
    assert fmt.sampleRate() == 44100


def test_loading_a_pair_resets_the_playhead_and_the_side(
    player: PlayerViewModel,
) -> None:
    left = np.full(44100, 500, dtype=np.int16)
    right = np.full(44100, -500, dtype=np.int16)
    player.load(left, right, autoplay=False)
    assert player.source.position == 0
    assert player.source.active == 0
    assert player.source.duration == pytest.approx(1.0)
    assert not player.playing


def test_selecting_a_side_leaves_the_playhead_alone(player: PlayerViewModel) -> None:
    player.load(
        np.zeros(44100, dtype=np.int16), np.zeros(44100, dtype=np.int16), autoplay=False
    )
    player.source.seek_seconds(0.5)
    before = player.source.position
    player.select(1)
    assert player.source.active == 1
    assert player.source.position == before


def test_playing_state_is_published(player: PlayerViewModel) -> None:
    states: list[bool] = []
    player.playing_changed.connect(states.append)
    player.load(
        np.zeros(4410, dtype=np.int16), np.zeros(4410, dtype=np.int16), autoplay=False
    )
    player.play()
    player.stop()
    assert states[-2:] == [True, False]


def test_restart_rewinds(player: PlayerViewModel) -> None:
    player.load(
        np.zeros(44100, dtype=np.int16), np.zeros(44100, dtype=np.int16), autoplay=False
    )
    player.source.seek_seconds(0.7)
    player.restart()
    assert player.source.position == 0
