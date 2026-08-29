"""
The instant A/B toggle (PLAN section 9.2).

Switching between two candidates at a shared playhead, mid-playback, the way
mastering comparison tools do it, is far more discriminating than playing one
clip and then the other -- and it is the only practical way to judge the
fidelity tier at all. Restarting playback on every switch destroys the
comparison, because by the time the second clip reaches the moment in question
the first one is no longer in echoic memory.

So both clips are resident in RAM as int16 and this device serves whichever one
is active from a single cursor. Toggling flips an index and leaves the cursor
alone. A short equal-power crossfade covers the discontinuity; without it the
switch clicks, and a click is itself an audible difference between the two
candidates, which is exactly the confound the harness is built to avoid.

The tokenizer decodes to mono -- waveform_layout is (batch, 1, frames * hop) --
but a mono QAudioFormat is routed to the first channel alone by most backends,
so the clip arrives in the left ear only. Buffers are therefore held as mono and
duplicated to interleaved stereo on the way out. Playing the same signal in both
ears is also the right call for rating: a difference in stereo placement between
two candidates is one more thing that wins an A/B for reasons unrelated to
quality.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QIODevice, Signal

CHANNELS = 2
BYTES_PER_SAMPLE = 2
BYTES_PER_FRAME = BYTES_PER_SAMPLE * CHANNELS


class AbSource(QIODevice):
    """
    Pull-mode audio source over two aligned int16 buffers.

    Args:
      sample_rate (int): samples per second, for cursor-to-seconds conversion.
      crossfade_ms (float): equal-power fade length applied on a toggle.
      parent (QObject | None): Qt parent.
    """

    finished = Signal()
    toggled = Signal(int)

    def __init__(
        self, sample_rate: int = 44100, crossfade_ms: float = 5.0, parent=None
    ) -> None:
        super().__init__(parent)
        self.sample_rate = sample_rate
        # 0 disables the fade outright; the read path skips it when empty
        self.fade_samples = max(0, int(crossfade_ms * sample_rate / 1000.0))
        self._buffers: list[np.ndarray] = [
            np.zeros(0, dtype=np.int16),
            np.zeros(0, dtype=np.int16),
        ]
        self._length = 0
        self._pos = 0
        self._active = 0
        self._fade_from = 0
        self._fade_left = 0
        self._ended = False

    # -- content -------------------------------------------------------------

    def set_pair(self, left: np.ndarray, right: np.ndarray) -> None:
        """
        Load a new comparison and rewind.

        Buffers are zero-padded to a common length so the cursor means the same
        moment in both; a reference clip clipped short by a track boundary would
        otherwise make the shorter side end early and give itself away.

        Args:
          left (np.ndarray): (N,) int16 samples for side A.
          right (np.ndarray): (M,) int16 samples for side B.
        """
        length = max(left.size, right.size)
        self._buffers = [
            np.pad(np.asarray(left, dtype=np.int16), (0, length - left.size)),
            np.pad(np.asarray(right, dtype=np.int16), (0, length - right.size)),
        ]
        self._length = length
        self.restart()

    def restart(self) -> None:
        """Rewind to the start without changing which side is active."""
        self._pos = 0
        self._fade_left = 0
        self._ended = False

    # -- transport -----------------------------------------------------------

    @property
    def active(self) -> int:
        """
        Returns:
          int: 0 for the left clip, 1 for the right.
        """
        return self._active

    @property
    def position(self) -> int:
        """
        Returns:
          int: the shared playhead, in samples.
        """
        return self._pos

    @property
    def seconds(self) -> float:
        """
        Returns:
          float: the shared playhead, in seconds.
        """
        return self._pos / self.sample_rate

    @property
    def duration(self) -> float:
        """
        Returns:
          float: clip length in seconds.
        """
        return self._length / self.sample_rate

    def seek_seconds(self, seconds: float) -> None:
        """
        Args:
          seconds (float): new playhead position, clamped to the clip.
        """
        self._pos = int(max(0.0, min(seconds, self.duration)) * self.sample_rate)
        self._fade_left = 0
        self._ended = self._pos >= self._length

    def toggle(self, index: int | None = None) -> None:
        """
        Switch sides at the current playhead.

        Args:
          index (int | None): 0 or 1, or None to flip. Selecting the side that
            is already active is a no-op, so holding a key does not stack fades.
        """
        target = (1 - self._active) if index is None else int(index)
        if target == self._active:
            return
        self._fade_from = self._active
        self._fade_left = self.fade_samples
        self._active = target
        self.toggled.emit(target)

    # -- QIODevice -----------------------------------------------------------

    def isSequential(self) -> bool:
        """
        Returns:
          bool: True; QAudioSink pulls this device forward only.
        """
        return True

    def bytesAvailable(self) -> int:
        """
        Returns:
          int: bytes remaining in the clip, counting both output channels.
        """
        return max(0, self._length - self._pos) * BYTES_PER_FRAME

    def readData(self, maxlen: int) -> bytes:
        """
        Serve the active buffer, mixing in the outgoing one while a fade runs.

        Args:
          maxlen (int): bytes Qt is willing to take, across both channels.

        Returns:
          bytes: little-endian interleaved stereo int16 PCM, the mono source
            duplicated to both channels. Empty once the clip has ended.
        """
        wanted = maxlen // BYTES_PER_FRAME
        if wanted <= 0:
            return b""
        remaining = self._length - self._pos
        if remaining <= 0:
            if not self._ended:
                self._ended = True
                self.finished.emit()
            return b""

        count = min(wanted, remaining)
        chunk = self._buffers[self._active][self._pos : self._pos + count].astype(
            np.float32
        )
        if self._fade_left > 0:
            faded = min(count, self._fade_left)
            done = self.fade_samples - self._fade_left
            # phase runs over (0, 1]: the first sample after a toggle already
            # carries some of the incoming clip, and the last is fully arrived
            phase = (done + 1 + np.arange(faded, dtype=np.float32)) / self.fade_samples
            outgoing = self._buffers[self._fade_from][
                self._pos : self._pos + faded
            ].astype(np.float32)
            chunk[:faded] = outgoing * np.cos(phase * np.pi / 2) + chunk[
                :faded
            ] * np.sin(phase * np.pi / 2)
            self._fade_left -= faded
        self._pos += count
        mono = np.clip(chunk, -32768, 32767).astype(np.int16)
        return np.repeat(mono, CHANNELS).tobytes()

    def writeData(self, data: bytes) -> int:
        """
        Args:
          data (bytes): ignored; this device is read-only.

        Returns:
          int: always 0.
        """
        return 0
