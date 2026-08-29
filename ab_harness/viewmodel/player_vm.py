"""
Playback transport over an AbSource.

QAudioSink pushing raw PCM rather than QMediaPlayer over files, for two reasons
that both come from the same place: the clips never exist as files (only tokens
are persisted), and a file-backed player cannot switch sources at a shared
playhead, which is the one interaction the fidelity tier depends on.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QIODevice, QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

from ab_harness.viewmodel.ab_source import CHANNELS, AbSource

TICK_MS = 50


class PlayerViewModel(QObject):
    """
    Transport state for one comparison.

    Args:
      sample_rate (int): playback rate.
      crossfade_ms (float): fade applied when switching sides.
      parent (QObject | None): Qt parent.
    """

    position_changed = Signal(float, float)
    playing_changed = Signal(bool)
    side_changed = Signal(int)
    finished = Signal()

    def __init__(
        self, sample_rate: int = 44100, crossfade_ms: float = 5.0, parent=None
    ) -> None:
        super().__init__(parent)
        self.source = AbSource(sample_rate, crossfade_ms, self)
        self.source.finished.connect(self._on_finished)
        self.source.toggled.connect(self.side_changed)

        # Stereo, even though the clips are mono: a mono format is routed to the
        # first channel alone by most backends, putting the clip in one ear.
        # AbSource duplicates the mono source across both channels.
        fmt = QAudioFormat()
        fmt.setSampleRate(sample_rate)
        fmt.setChannelCount(CHANNELS)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self._sink = QAudioSink(QMediaDevices.defaultAudioOutput(), fmt, self)

        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._playing = False

    # -- content -------------------------------------------------------------

    def load(self, left: np.ndarray, right: np.ndarray, autoplay: bool = True) -> None:
        """
        Load a new pair, resetting the playhead and the active side.

        Args:
          left (np.ndarray): (N,) int16 samples for the left clip.
          right (np.ndarray): (M,) int16 samples for the right clip.
          autoplay (bool): start playing immediately.
        """
        self.stop()
        self.source.set_pair(left, right)
        self.source.toggle(0)
        self._emit_position()
        if autoplay:
            self.play()

    # -- transport -----------------------------------------------------------

    @property
    def playing(self) -> bool:
        """
        Returns:
          bool: True while audio is being pulled.
        """
        return self._playing

    def play(self) -> None:
        """Start or resume playback from the current playhead."""
        if self._playing:
            return
        if not self.source.isOpen():
            self.source.open(QIODevice.OpenModeFlag.ReadOnly)
        self._sink.start(self.source)
        self._timer.start()
        self._playing = True
        self.playing_changed.emit(True)

    def stop(self) -> None:
        """Halt playback, leaving the playhead where it is."""
        self._timer.stop()
        self._sink.stop()
        if self._playing:
            self._playing = False
            self.playing_changed.emit(False)

    def toggle_play(self) -> None:
        """Play if stopped, stop if playing."""
        self.stop() if self._playing else self.play()

    def restart(self) -> None:
        """Rewind to the start and keep playing if already playing."""
        was_playing = self._playing
        self.stop()
        self.source.restart()
        self._emit_position()
        if was_playing:
            self.play()

    def select(self, side: int) -> None:
        """
        Args:
          side (int): 0 for left, 1 for right. Crossfades at the playhead.
        """
        self.source.toggle(side)

    def flip(self) -> None:
        """Switch to the other side at the playhead."""
        self.source.toggle(None)

    def seek(self, seconds: float) -> None:
        """
        Args:
          seconds (float): new playhead position.
        """
        was_playing = self._playing
        self.stop()
        self.source.seek_seconds(seconds)
        self._emit_position()
        if was_playing:
            self.play()

    # -- internals -----------------------------------------------------------

    def _tick(self) -> None:
        """Emit the playhead for the progress display."""
        self._emit_position()

    def _emit_position(self) -> None:
        """Publish the current playhead and clip length."""
        self.position_changed.emit(self.source.seconds, self.source.duration)

    def _on_finished(self) -> None:
        """Stop the transport when the source runs out."""
        self.stop()
        self._emit_position()
        self.finished.emit()
