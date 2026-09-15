"""
Waveform scrubbing.

The strip is the only place in the harness where a mouse position turns into a
playhead, so what is pinned here is the mapping: which events seek, which do
not, and that the fraction is the fraction of the widget the click landed at.
Dead air runs 30% of a 90 s generation, and a rater who cannot skip it pays for
that in listening time on every structure pair.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from ab_harness.view.waveform import WaveformView

WIDTH = 200


def _press(x: float, buttons: Qt.MouseButton | None = None) -> QMouseEvent:
    """
    Args:
      x (float): horizontal position inside the widget.
      buttons (Qt.MouseButton | None): buttons held, for move events.

    Returns:
      QMouseEvent: a left-button event at (x, 8).
    """
    held = Qt.MouseButton.LeftButton if buttons is None else buttons
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(x, 8.0),
        QPointF(x, 8.0),
        Qt.MouseButton.LeftButton,
        held,
        Qt.KeyboardModifier.NoModifier,
    )


@pytest.fixture
def wave(qapp) -> WaveformView:
    """
    Returns:
      WaveformView: a loaded strip of known width.
    """
    view = WaveformView()
    view.resize(WIDTH, 64)
    view.set_pcm((np.random.default_rng(0).normal(0, 3000, 44100)).astype(np.int16))
    return view


def test_a_click_emits_its_fraction_of_the_width(wave: WaveformView) -> None:
    seen: list[float] = []
    wave.scrubbed.connect(seen.append)
    wave.mousePressEvent(_press(0.25 * WIDTH))
    assert seen == pytest.approx([0.25])


def test_a_click_past_the_edge_is_clamped(wave: WaveformView) -> None:
    seen: list[float] = []
    wave.scrubbed.connect(seen.append)
    wave.mousePressEvent(_press(-10.0))
    wave.mousePressEvent(_press(WIDTH + 40.0))
    assert seen == pytest.approx([0.0, 1.0])


def test_dragging_scrubs_and_hovering_does_not(wave: WaveformView) -> None:
    seen: list[float] = []
    wave.scrubbed.connect(seen.append)
    wave.mouseMoveEvent(_press(0.5 * WIDTH))
    wave.mouseMoveEvent(_press(0.75 * WIDTH, buttons=Qt.MouseButton.NoButton))
    assert seen == pytest.approx([0.5])


def test_an_empty_strip_does_not_seek(qapp) -> None:
    """No clip loaded means no position to seek to, and the pair is not shown."""
    view = WaveformView()
    view.resize(WIDTH, 64)
    seen: list[float] = []
    view.scrubbed.connect(seen.append)
    view.mousePressEvent(_press(0.5 * WIDTH))
    assert seen == []


def test_a_middle_click_does_not_seek(wave: WaveformView) -> None:
    seen: list[float] = []
    wave.scrubbed.connect(seen.append)
    wave.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(0.5 * WIDTH, 8.0),
            QPointF(0.5 * WIDTH, 8.0),
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert seen == []
