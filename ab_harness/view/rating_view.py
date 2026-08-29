"""
The rating widget: keyboard-only, blind, one comparison at a time.

Nothing on screen identifies either candidate. The two side panels are labelled
A and B and differ only in which is currently sounding; checkpoint, seed and
conditioning stay in the log, never in the window.

Every action has a button as well as a key. The keys are what make the target
of a few seconds per pair reachable (PLAN section 7.5), but a mouse path matters
for the first sessions, before the map is in muscle memory, and for the long
structure-tier pairs where there is time to spare. Buttons take no focus, so
they never swallow the next keystroke.

Beside the comparison sits the worklist: what is queued, what is still being
generated, and two buttons that say which tier the GPU should make more of. The
tier mix is a coin flip per pair, so a 20% share can go a dozen pairs without
showing up; picking from a list is how the rater takes that back. Each side also
carries its waveform, which is what makes a mostly-silent generation visible
without listening to it.

Key map, chosen so the whole loop is reachable without moving a hand:

  space   play / pause              1  A is better
  a / <-  listen to A               2  tie / can't tell
  d / ->  listen to B               3  B is better
  s       flip sides                x  skip (unrateable, not a tie)
  r       restart from the top      shift+a / shift+d  play that side from the top
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ab_harness.model.pipeline import WorkItem
from ab_harness.model.types import Pair, Tier
from ab_harness.view.waveform import WaveformView
from ab_harness.view.worklist import DEFAULT_QUIET_FILL, WorklistPanel

# palette(mid) resolves to near-invisible grey on a dark theme, and the hint and
# key legend are exactly the text a first-time rater needs to read.
MUTED = "#9aa3ad"

TIER_HINT: dict[Tier, str] = {
    Tier.BULK: "Toggle A/B at the same moment. Judge artifacts, roughness, smearing.",
    Tier.STRUCTURE: "Play each one through. Judge development, contrast, whether it goes anywhere.",
}


class SidePanel(QFrame):
    """
    One candidate's panel: a letter, its waveform, and whether it is sounding.

    The waveform shows the audio about to be played and nothing else, so it
    cannot break the blind -- and a clip that is mostly silence is visible
    before it costs a listen.

    Args:
      letter (str): "A" or "B".
      parent (QWidget | None): Qt parent.
    """

    def __init__(self, letter: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._letter = QLabel(letter, self)
        self._letter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._letter.setStyleSheet("font-size: 64px; font-weight: 600; border: none;")
        self._wave = WaveformView(parent=self)
        self._state = QLabel("", self)
        self._state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state.setFixedHeight(24)
        self._state.setStyleSheet(f"color: {MUTED}; border: none;")
        layout = QVBoxLayout(self)
        layout.addWidget(self._letter, stretch=1)
        layout.addWidget(self._wave)
        layout.addWidget(self._state)
        self.set_active(False)

    def set_pcm(self, pcm: np.ndarray) -> None:
        """
        Args:
          pcm (np.ndarray): (N,) int16 samples for this side.
        """
        self._wave.set_pcm(pcm)

    def set_position(self, fraction: float) -> None:
        """
        Args:
          fraction (float): playhead position in [0, 1].
        """
        self._wave.set_position(fraction)

    def set_active(self, active: bool) -> None:
        """
        Args:
          active (bool): True when this side is the one being heard.
        """
        self._state.setText("sounding" if active else "")
        self._wave.set_active(active)
        # Scoped to the object name: QLabel derives from QFrame, so a bare
        # "QFrame { border }" rule would draw a box around each child label too.
        colour = "#4aa3ff" if active else "#555b63"
        self.setStyleSheet(
            f"QFrame#sidePanel {{ border: 2px solid {colour}; border-radius: 8px; }}"
        )


class RatingView(QWidget):
    """
    The full rating surface.

    Emits intent only; every decision is made by the view-model.

    Args:
      parent (QWidget | None): Qt parent.
      quiet_fill (float): fill fraction below which a queued pair counts as
        mostly empty in the worklist.
    """

    chose = Signal(str)
    skipped = Signal()
    listen = Signal(int)
    flip = Signal()
    play_pause = Signal()
    restart = Signal()
    listen_from_start = Signal(int)
    picked = Signal(str)
    generate = Signal(str, int)

    def __init__(
        self,
        parent: QWidget | None = None,
        quiet_fill: float = DEFAULT_QUIET_FILL,
    ) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._question = QLabel("preparing...", self)
        self._question.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._question.setStyleSheet("font-size: 22px; font-weight: 600;")
        self._hint = QLabel("", self)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {MUTED};")

        self._panels = (SidePanel("A", self), SidePanel("B", self))
        panels = QHBoxLayout()
        panels.addWidget(self._panels[0])
        panels.addWidget(self._panels[1])

        self._progress = QProgressBar(self)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 1000)
        self._time = QLabel("0.0 / 0.0 s", self)
        self._time.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._legend = QLabel(
            "space play   a/d listen   s flip   r restart   "
            "1 A better   2 tie   3 B better   x skip",
            self,
        )
        self._legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._legend.setStyleSheet(f"color: {MUTED}; font-family: monospace;")

        self._play = self._button("Play", self.play_pause.emit)
        transport = QHBoxLayout()
        transport.addWidget(self._button("Listen A", lambda: self.listen.emit(0)))
        transport.addWidget(self._play)
        transport.addWidget(self._button("Restart", self.restart.emit))
        transport.addWidget(self._button("Listen B", lambda: self.listen.emit(1)))

        decide = QHBoxLayout()
        decide.addWidget(self._button("A is better", lambda: self.chose.emit("left")))
        decide.addWidget(self._button("Tie", lambda: self.chose.emit("tie")))
        decide.addWidget(self._button("B is better", lambda: self.chose.emit("right")))
        decide.addWidget(self._button("Skip", self.skipped.emit))

        self.worklist = WorklistPanel(quiet_fill, self)
        self.worklist.setFixedWidth(240)
        self.worklist.selected.connect(self.picked)
        self.worklist.generate.connect(self.generate)

        rating = QVBoxLayout()
        rating.addWidget(self._question)
        rating.addWidget(self._hint)
        rating.addLayout(panels, stretch=1)
        rating.addWidget(self._progress)
        rating.addWidget(self._time)
        rating.addLayout(transport)
        rating.addLayout(decide)
        rating.addWidget(self._legend)

        layout = QHBoxLayout(self)
        layout.addLayout(rating, stretch=1)
        layout.addWidget(self.worklist)

    def _button(self, text: str, slot) -> QPushButton:
        """
        Build a button that cannot steal keyboard focus.

        Without NoFocus the first click would move focus off the rating surface
        and every subsequent keystroke would go nowhere, which reads as the
        harness having frozen.

        Args:
          text (str): button label.
          slot: callable invoked on click.

        Returns:
          QPushButton: the wired button.
        """
        button = QPushButton(text, self)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(slot)
        return button

    # -- display -------------------------------------------------------------

    def show_pair(self, pair: Pair) -> None:
        """
        Args:
          pair (Pair): the comparison to display. Only its question, tier and
            waveforms reach the screen.
        """
        self._question.setText(pair.spec.question)
        self._hint.setText(TIER_HINT[pair.spec.tier])
        self._panels[0].set_pcm(np.asarray(pair.left.pcm))
        self._panels[1].set_pcm(np.asarray(pair.right.pcm))
        self.set_active_side(0)

    def set_worklist(self, items: list[WorkItem]) -> None:
        """
        Args:
          items (list[WorkItem]): the queue to display, ready entries first.
        """
        self.worklist.set_items(items)

    def set_waiting(self, waiting: bool) -> None:
        """
        Args:
          waiting (bool): True while the pipeline is still producing.
        """
        if waiting:
            self._question.setText("preparing the next pair...")
            self._hint.setText("")

    def set_playing(self, playing: bool) -> None:
        """
        Args:
          playing (bool): whether audio is running, for the transport label.
        """
        self._play.setText("Pause" if playing else "Play")

    def set_active_side(self, side: int) -> None:
        """
        Args:
          side (int): 0 for A, 1 for B.
        """
        for index, panel in enumerate(self._panels):
            panel.set_active(index == side)

    def set_position(self, seconds: float, duration: float) -> None:
        """
        Args:
          seconds (float): playhead position.
          duration (float): clip length.
        """
        fraction = seconds / duration if duration else 0.0
        self._progress.setValue(int(1000 * fraction))
        self._time.setText(f"{seconds:.1f} / {duration:.1f} s")
        for panel in self._panels:
            panel.set_position(fraction)

    # -- input ---------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Translate keys to intent signals.

        Args:
          event (QKeyEvent): the key press.
        """
        key = event.key()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_Space:
            self.play_pause.emit()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.listen_from_start.emit(0) if shift else self.listen.emit(0)
        elif key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.listen_from_start.emit(1) if shift else self.listen.emit(1)
        elif key == Qt.Key.Key_S:
            self.flip.emit()
        elif key == Qt.Key.Key_R:
            self.restart.emit()
        elif key == Qt.Key.Key_1:
            self.chose.emit("left")
        elif key == Qt.Key.Key_2:
            self.chose.emit("tie")
        elif key == Qt.Key.Key_3:
            self.chose.emit("right")
        elif key == Qt.Key.Key_X:
            self.skipped.emit()
        else:
            super().keyPressEvent(event)
