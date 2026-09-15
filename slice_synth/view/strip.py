"""
The shared behaviour of the three strips under a render.

Waveform, mel and tokens are three views of one clip on one time axis. What they
have in common is not what they draw but how they behave: a fixed height, an
image rebuilt on load and resize rather than per frame, a playhead, and a click
or drag anywhere on them that moves it.

That last part is the reason there are three strips at all. The interesting
moment in a generation is rarely at the top, and the strips are how you find it
by eye -- a flat waveform, a dark band in the mel, a run of one colour in the
tokens -- and then jump straight to it.

ab_harness.view.waveform.WaveformView already implements the same contract for
the waveform; it is used unchanged rather than being refactored onto this base,
because the rating harness has its own reasons for every line of it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

BACKGROUND = QColor("#22262b")
PLAYHEAD = QColor("#e8eef5")
MARKER = QColor("#4aa3ff")
EMPTY_TEXT = QColor("#5b6672")


class ScrubStrip(QWidget):
    """
    A fixed-height image strip with a playhead, scrubbable by mouse.

    Args:
      height (int): widget height in pixels, or its minimum when expanding.
      parent (QWidget | None): Qt parent.
      expanding (bool): take vertical space when there is any going spare.
    """

    scrubbed = Signal(float)

    def __init__(
        self, height: int = 64, parent: QWidget | None = None, expanding: bool = False
    ) -> None:
        super().__init__(parent)
        if expanding:
            # The mel is the strip worth giving spare room to: it is the only
            # one whose vertical axis carries information.
            self.setMinimumHeight(height)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
        else:
            self.setFixedHeight(height)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._image: QImage | None = None
        self._position = 0.0
        self._marker = 0.0
        self._placeholder = ""

    # -- content -------------------------------------------------------------

    def set_position(self, fraction: float) -> None:
        """
        Args:
          fraction (float): playhead position in [0, 1].
        """
        self._position = min(1.0, max(0.0, fraction))
        self.update()

    def set_marker(self, fraction: float) -> None:
        """
        Args:
          fraction (float): where the prompted region ends, in [0, 1]. Everything
            left of it was forced to real audio, so it is not the model's work
            and should not be judged as if it were.
        """
        self._marker = min(1.0, max(0.0, fraction))
        self.update()

    def set_placeholder(self, text: str) -> None:
        """
        Args:
          text (str): message shown when there is nothing to draw.
        """
        self._placeholder = text
        self.update()

    def set_image(self, rgb: np.ndarray | None) -> None:
        """
        Args:
          rgb (np.ndarray | None): (H, W, 3) uint8 image, or None to clear.

        QImage does not take ownership of the buffer it is constructed over, so
        the bytes are deep-copied into it. Without that, the temporary this was
        built from is collected the moment set_image returns and the strip
        paints whatever the allocator put there instead -- intermittently, which
        is the worst way for it to fail.
        """
        if rgb is None or rgb.size == 0:
            self._image = None
        else:
            data = np.ascontiguousarray(rgb, dtype=np.uint8)
            height, width, _ = data.shape
            self._image = QImage(
                data.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888
            ).copy()
        self.update()

    # -- input ---------------------------------------------------------------

    def _emit_scrub(self, event: QMouseEvent) -> None:
        """
        Args:
          event (QMouseEvent): press or drag whose x maps to a position.
        """
        if self.width() <= 0:
            return
        self.scrubbed.emit(min(1.0, max(0.0, event.position().x() / self.width())))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Args:
          event (QMouseEvent): the click. Left button only; a middle-click paste
            on X11 would otherwise seek blindly.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._emit_scrub(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Args:
          event (QMouseEvent): the drag; held-button scrubbing is how you find a
            moment you heard once and want again.
        """
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._emit_scrub(event)
        else:
            super().mouseMoveEvent(event)

    # -- painting ------------------------------------------------------------

    def rebuild(self) -> None:
        """Recompute the image for the current width. Overridden by subclasses."""

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Args:
          event (QResizeEvent): the resize; the image is width-dependent.
        """
        super().resizeEvent(event)
        self.rebuild()

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Args:
          event (QPaintEvent): the repaint request.
        """
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if self._image is not None:
            painter.drawImage(QRect(0, 0, self.width(), self.height()), self._image)
        elif self._placeholder:
            painter.setPen(EMPTY_TEXT)
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder
            )
        if self._marker > 0.0:
            painter.setPen(MARKER)
            x = int(self._marker * self.width())
            painter.drawLine(x, 0, x, self.height())
        if self._position > 0.0:
            painter.setPen(PLAYHEAD)
            x = int(self._position * self.width())
            painter.drawLine(x, 0, x, self.height())
        painter.end()


__all__ = ["BACKGROUND", "MARKER", "PLAYHEAD", "ScrubStrip"]
