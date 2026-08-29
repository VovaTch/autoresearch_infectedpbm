"""
The queue, as a list the rater works through.

A random tier mix is a coin flip per pair, not a schedule: at the configured 20%
share the 90 s tier turned up once in thirteen pairs. Showing what is queued and
letting the rater pick from it turns the mix into a choice, and the two generate
buttons say what the GPU should spend the next two minutes on.

Rows carry the tier, the length, and how full of audio the clip is -- never an
id, a seed or a checkpoint, so picking from the list cannot break the blind.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ab_harness.model.audio import fill_fraction
from ab_harness.model.pipeline import WorkItem
from ab_harness.model.types import Tier

PAIR_ID_ROLE = Qt.ItemDataRole.UserRole
MUTED = "#9aa3ad"
DEFAULT_QUIET_FILL = 0.25


class WorklistPanel(QWidget):
    """
    Queue list plus the on-demand generate controls.

    Args:
      quiet_fill (float): a ready pair whose quieter side is filled below this
        fraction counts as mostly empty and can be hidden.
      parent (QWidget | None): Qt parent.
    """

    selected = Signal(str)
    generate = Signal(str, int)

    def __init__(
        self, quiet_fill: float = DEFAULT_QUIET_FILL, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.quiet_fill = quiet_fill
        self._fill: dict[str, float] = {}
        self._items: list[WorkItem] = []
        self._rows: list[tuple[str, str, bool]] = []

        self._list = QListWidget(self)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.itemActivated.connect(self._emit_selected)
        self._list.itemClicked.connect(self._emit_selected)

        self._hide_quiet = QCheckBox("hide mostly empty", self)
        self._hide_quiet.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._hide_quiet.toggled.connect(lambda _: self._render())

        self._count = QSpinBox(self)
        self._count.setRange(1, 8)
        self._count.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._status = QLabel("", self)
        self._status.setStyleSheet(f"color: {MUTED};")
        self._status.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self._count)
        buttons.addWidget(self._button("more 10 s", Tier.BULK))
        buttons.addWidget(self._button("more 90 s", Tier.STRUCTURE))

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("queue", self))
        layout.addWidget(self._list, stretch=1)
        layout.addWidget(self._hide_quiet)
        layout.addLayout(buttons)
        layout.addWidget(self._status)

    def _button(self, text: str, tier: Tier) -> QPushButton:
        """
        Args:
          text (str): button label.
          tier (Tier): tier the button queues.

        Returns:
          QPushButton: a button that cannot steal keyboard focus.
        """
        button = QPushButton(text, self)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(
            lambda: self.generate.emit(str(tier), self._count.value())
        )
        return button

    # -- content -------------------------------------------------------------

    def set_items(self, items: list[WorkItem]) -> None:
        """
        Args:
          items (list[WorkItem]): the queue, ready entries first.
        """
        self._items = list(items)
        live = {item.pair_id for item in self._items}
        self._fill = {k: v for k, v in self._fill.items() if k in live}
        self._render()

    def _fill_of(self, item: WorkItem) -> float:
        """
        Args:
          item (WorkItem): a queue entry.

        Returns:
          float: the quieter side's fill fraction, or -1.0 while it is still
            being produced. Cached: a 90 s clip is four million samples and the
            list is rebuilt several times a second.
        """
        if item.pair is None:
            return -1.0
        if item.pair_id not in self._fill:
            self._fill[item.pair_id] = min(
                fill_fraction(np.asarray(item.pair.left.pcm)),
                fill_fraction(np.asarray(item.pair.right.pcm)),
            )
        return self._fill[item.pair_id]

    def _label(self, item: WorkItem, fill: float) -> str:
        """
        Args:
          item (WorkItem): a queue entry.
          fill (float): its fill fraction, or -1.0 when unknown.

        Returns:
          str: the row text. Blind by construction: length, fullness, status.
        """
        length = f"{item.seconds:.0f} s"
        if fill < 0.0:
            return f"{length}   generating..."
        return f"{length}   fill {fill * 100:3.0f}%"

    def _render(self) -> None:
        """
        Rebuild the rows, applying the mostly-empty filter.

        The queue is republished several times a second; rebuilding rows that
        did not change would clear the selection under the rater's cursor, so
        an unchanged list is left alone.
        """
        rows = []
        hidden = 0
        for item in self._items:
            fill = self._fill_of(item)
            quiet = 0.0 <= fill < self.quiet_fill
            if quiet and self._hide_quiet.isChecked():
                hidden += 1
                continue
            rows.append((item.pair_id, self._label(item, fill), fill < 0.0))
        if rows != self._rows:
            self._rows = rows
            self._list.clear()
            for pair_id, label, pending in rows:
                row = QListWidgetItem(label)
                row.setData(PAIR_ID_ROLE, pair_id)
                if pending:
                    row.setForeground(Qt.GlobalColor.gray)
                self._list.addItem(row)
        ready = sum(1 for item in self._items if item.ready)
        parts = [f"{ready} ready", f"{len(self._items) - ready} generating"]
        if hidden:
            parts.append(f"{hidden} quiet hidden")
        self._status.setText(" - ".join(parts))

    def _emit_selected(self, row: QListWidgetItem) -> None:
        """
        Args:
          row (QListWidgetItem): the clicked row.
        """
        pair_id = row.data(PAIR_ID_ROLE)
        if pair_id:
            self.selected.emit(str(pair_id))
