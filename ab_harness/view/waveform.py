"""
Peak-envelope waveform strip.

Drawn per candidate, under each side panel. A generation that trails off into
silence or never starts is instantly visible as a flat strip, which is the one
defect that costs a full listen to find by ear and none at all to find by eye.

It shows the audio the rater is about to hear and nothing else -- no id, no
seed, no checkpoint -- so it cannot break the blind. Both sides are drawn with
the same scale and the same colour for the same reason.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QResizeEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from ab_harness.model.audio import envelope

WAVE = QColor("#5b6672")
WAVE_ACTIVE = QColor("#4aa3ff")
PLAYHEAD = QColor("#e8eef5")
BACKGROUND = QColor("#22262b")
MIN_BAR_PX = 2


class WaveformView(QWidget):
    """
    A fixed-height envelope strip with a playhead.

    Args:
      height (int): widget height in pixels.
      parent (QWidget | None): Qt parent.
    """

    def __init__(self, height: int = 64, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._pcm: np.ndarray = np.zeros(0, dtype=np.int16)
        self._peaks: np.ndarray = np.zeros(0, dtype=np.float32)
        self._position = 0.0
        self._active = False

    # -- content -------------------------------------------------------------

    def set_pcm(self, pcm: np.ndarray) -> None:
        """
        Args:
          pcm (np.ndarray): (N,) int16 samples for this side.
        """
        self._pcm = pcm
        self._position = 0.0
        self._rebuild()

    def set_position(self, fraction: float) -> None:
        """
        Args:
          fraction (float): playhead position in [0, 1].
        """
        self._position = min(1.0, max(0.0, fraction))
        self.update()

    def set_active(self, active: bool) -> None:
        """
        Args:
          active (bool): True when this side is the one sounding.
        """
        if active != self._active:
            self._active = active
            self.update()

    # -- painting ------------------------------------------------------------

    def _rebuild(self) -> None:
        """Recompute the envelope for the current width, then repaint."""
        # Once per load and per resize, never per frame: a 90 s clip is four
        # million samples and the playhead moves twenty times a second.
        self._peaks = envelope(self._pcm, max(1, self.width() // MIN_BAR_PX))
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Args:
          event (QResizeEvent): the resize; the envelope is width-dependent.
        """
        super().resizeEvent(event)
        self._rebuild()

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Args:
          event (QPaintEvent): the repaint request.
        """
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if self._peaks.size:
            width = self.width() / self._peaks.size
            mid = self.height() / 2.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(WAVE_ACTIVE if self._active else WAVE)
            for index, peak in enumerate(self._peaks):
                half = max(0.5, float(peak) * mid)
                painter.drawRect(
                    QRectF(index * width, mid - half, max(1.0, width - 0.5), 2 * half)
                )
        if self._position > 0.0:
            painter.setPen(PLAYHEAD)
            x = self._position * self.width()
            painter.drawLine(int(x), 0, int(x), self.height())
        painter.end()
