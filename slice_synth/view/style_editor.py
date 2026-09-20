"""
Picking and shaping the style embedding.

The style descriptor is the app's main lever, and unlike the track id it is a
continuous 1024-d thing, so there is more than one useful way to choose one. All
four live here:

  window   an existing 10 s window of a real track -- a mood the model has heard
  random   a fresh unit vector -- somewhere on the sphere it has never been
  jitter   a real window nudged off itself, with a strength slider
  interp   slerp between two windows, for morphing one track's feel into another

Plus a null entry, which drops the stream and lets the model fall back to its
learned null. All five are list entries rather than a mode switch, because the
point of the app is hearing them against each other in one batch.

A style entry defaults to SAME_TRACK: it follows whichever track each render is
conditioned on. Without that, ticking five tracks and one style would condition
all five on the first track's mood, which is a mistake that looks like a result.

Any entry can also walk: every `period` frames the descriptor is swapped in
place for a random corpus window of the render's own tracks, for a fresh unit
vector, or for the style model's continuation (train_style_ar.py, primed with
the base track's real windows). The entry as written is always the opening
segment, except for a cold style-model walk.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ab_harness.model.pair_sampler import TrackInfo
from slice_synth.model.types import DEFAULT_PERIOD, SAME_TRACK, StyleSpec

SPEC_ROLE = Qt.ItemDataRole.UserRole
KIND_LABELS = {
    "window": "real window",
    "random": "random vector",
    "jitter": "jittered window",
    "interp": "interpolation",
}
WALK_LABELS = {
    "none": "hold for the whole clip",
    "windows": "random corpus windows",
    "random": "random vectors",
    "ar": "style model (AR)",
}
AR_ROWS = ("ar prefix", "ar temperature", "ar guidance")


class StyleListEditor(QWidget):
    """
    An editable list of style recipes.

    Args:
      parent (QWidget | None): Qt parent.
      default (StyleSpec | None): template every new entry starts from -- its
        walk and style-model settings carry over, the kind and window do not.
        None starts entries un-walked on the render's own track.
    """

    changed = Signal()

    def __init__(
        self, parent: QWidget | None = None, default: StyleSpec | None = None
    ) -> None:
        super().__init__(parent)
        self._tracks: list[TrackInfo] = []
        self._loading = False
        self._default = default or StyleSpec(track_idx=SAME_TRACK, track_b=SAME_TRACK)

        self._list = QListWidget(self)
        self._list.setMaximumHeight(110)
        self._list.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._list.currentRowChanged.connect(lambda _: self._load_current())

        buttons = QHBoxLayout()
        for kind, text in (
            ("window", "+ window"),
            ("random", "+ random"),
            ("jitter", "+ jitter"),
            ("interp", "+ interp"),
        ):
            buttons.addWidget(self._add_button(text, kind))
        buttons.addWidget(self._add_button("+ null", ""))
        remove = QPushButton("remove", self)
        remove.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        remove.clicked.connect(self._remove_current)
        buttons.addWidget(remove)
        buttons.addStretch(1)

        self._track = QComboBox(self)
        self._track_b = QComboBox(self)
        self._window = self._spin(
            -1, 9999, "-1 = auto: a window disjoint from the generated span"
        )
        self._window_b = self._spin(-1, 9999, "-1 = auto")
        self._mix = self._dspin(0.0, 1.0, 0.05, "0 = first window, 1 = second")
        self._noise = self._dspin(
            0.0, 2.0, 0.05, "how far off the real window to push it"
        )
        self._seed = self._spin(
            0, 2**31 - 1, "seeds every random draw this entry makes"
        )
        self._walk = QComboBox(self)
        for walk, text in WALK_LABELS.items():
            self._walk.addItem(text, walk)
        self._walk.setToolTip(
            "swap the style in place every period, starting from this entry; "
            "'windows' draws from the render's own tracks (all of them, when ids coexist)"
        )
        self._period = self._spin(
            1, 65536, "frames per style segment; 512 is the style model's own slice"
        )
        self._period.setValue(DEFAULT_PERIOD)
        self._ar_prefix = self._spin(
            0,
            64,
            "real windows of the base track fed to the style model, ending at this "
            "entry's window; 0 = cold start, every segment is sampled",
        )
        self._ar_temperature = self._dspin(
            0.0, 3.0, 0.05, "style model sampling temperature; 0 = greedy"
        )
        self._ar_cfg = self._dspin(
            0.0,
            8.0,
            0.25,
            "guidance on the track id: 0 = null id, 1 = plain conditional, "
            "above 1 pushes toward the track",
        )

        self._form = QFormLayout()
        self._form.addRow("track", self._track)
        self._form.addRow("window", self._window)
        self._form.addRow("track B", self._track_b)
        self._form.addRow("window B", self._window_b)
        self._form.addRow("mix", self._mix)
        self._form.addRow("noise", self._noise)
        self._form.addRow("seed", self._seed)
        self._form.addRow("walk", self._walk)
        self._form.addRow("period", self._period)
        self._form.addRow("ar prefix", self._ar_prefix)
        self._form.addRow("ar temperature", self._ar_temperature)
        self._form.addRow("ar guidance", self._ar_cfg)

        for widget in (self._track, self._track_b, self._walk):
            widget.currentIndexChanged.connect(self._apply)
        for widget in (
            self._window,
            self._window_b,
            self._seed,
            self._period,
            self._ar_prefix,
        ):
            widget.valueChanged.connect(self._apply)
        for widget in (self._mix, self._noise, self._ar_temperature, self._ar_cfg):
            widget.valueChanged.connect(self._apply)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)
        layout.addLayout(buttons)
        layout.addLayout(self._form)

        self.add(replace(self._default, kind="window", window=-1))

    # -- construction helpers ------------------------------------------------

    def _add_button(self, text: str, kind: str) -> QPushButton:
        """
        Args:
          text (str): button label.
          kind (str): style kind to append, or "" for a null entry.

        Returns:
          QPushButton: the wired button.
        """
        button = QPushButton(text, self)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(
            lambda: self.add(
                None
                if not kind
                else replace(self._default, kind=kind)  # type: ignore[arg-type]
            )
        )
        return button

    def _spin(self, low: int, high: int, tip: str) -> QSpinBox:
        """
        Args:
          low (int): minimum.
          high (int): maximum.
          tip (str): tooltip.

        Returns:
          QSpinBox: a configured spin box.
        """
        box = QSpinBox(self)
        box.setRange(low, high)
        box.setToolTip(tip)
        return box

    def _dspin(self, low: float, high: float, step: float, tip: str) -> QDoubleSpinBox:
        """
        Args:
          low (float): minimum.
          high (float): maximum.
          step (float): single step.
          tip (str): tooltip.

        Returns:
          QDoubleSpinBox: a configured spin box.
        """
        box = QDoubleSpinBox(self)
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setDecimals(2)
        box.setToolTip(tip)
        return box

    # -- content -------------------------------------------------------------

    def set_corpus(self, tracks: list[TrackInfo]) -> None:
        """
        Args:
          tracks (list[TrackInfo]): the corpus, for the track selectors.
        """
        self._tracks = list(tracks)
        self._loading = True
        for combo in (self._track, self._track_b):
            current = combo.currentData()
            combo.clear()
            combo.addItem("own (this render)", SAME_TRACK)
            for track in tracks:
                combo.addItem(
                    f"{track.track_idx:02d} {track.track_name[:38]}", track.track_idx
                )
            index = combo.findData(current if current is not None else SAME_TRACK)
            combo.setCurrentIndex(max(0, index))
        self._loading = False
        self._load_current()

    def styles(self) -> list[StyleSpec | None]:
        """
        Returns:
          list[StyleSpec | None]: every entry, in list order. None is the null
            stream. Empty when the list is empty, which the caller must treat as
            "generate nothing" rather than "generate unconditioned".
        """
        return [
            self._list.item(row).data(SPEC_ROLE) for row in range(self._list.count())
        ]

    def add(self, spec: StyleSpec | None) -> None:
        """
        Args:
          spec (StyleSpec | None): the entry to append; None adds a null entry.
        """
        item = QListWidgetItem(self._text(spec))
        item.setData(SPEC_ROLE, spec)
        self._list.addItem(item)
        self._list.setCurrentItem(item)
        self.changed.emit()

    def _remove_current(self) -> None:
        """Drop the selected entry."""
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)
            self.changed.emit()

    def _text(self, spec: StyleSpec | None) -> str:
        """
        Args:
          spec (StyleSpec | None): the entry.

        Returns:
          str: its row label.
        """
        if spec is None:
            return "null  (learned null, no style)"
        return f"{KIND_LABELS[spec.kind]}  {spec.label()}"

    # -- editing -------------------------------------------------------------

    def _load_current(self) -> None:
        """Show the selected entry's parameters and hide the irrelevant ones."""
        spec = (
            self._list.currentItem().data(SPEC_ROLE)
            if self._list.currentItem()
            else None
        )
        enabled = {
            "window": {"track", "window", "seed"},
            "random": {"seed"},
            "jitter": {"track", "window", "noise", "seed"},
            "interp": {"track", "window", "track B", "window B", "mix", "seed"},
        }.get(spec.kind if spec else "", set())
        if spec is not None:
            enabled = enabled | {"walk"}
            if spec.walk != "none":
                enabled = enabled | {"period"}
            if spec.walk == "ar":
                enabled = enabled | set(AR_ROWS)

        rows = {
            "track": self._track,
            "window": self._window,
            "track B": self._track_b,
            "window B": self._window_b,
            "mix": self._mix,
            "noise": self._noise,
            "seed": self._seed,
            "walk": self._walk,
            "period": self._period,
            "ar prefix": self._ar_prefix,
            "ar temperature": self._ar_temperature,
            "ar guidance": self._ar_cfg,
        }
        for name, widget in rows.items():
            visible = name in enabled
            widget.setVisible(visible)
            label = self._form.labelForField(widget)
            if isinstance(label, QLabel):
                label.setVisible(visible)

        if spec is None:
            return
        self._loading = True
        self._track.setCurrentIndex(max(0, self._track.findData(spec.track_idx)))
        self._track_b.setCurrentIndex(max(0, self._track_b.findData(spec.track_b)))
        self._window.setValue(spec.window)
        self._window_b.setValue(spec.window_b)
        self._mix.setValue(spec.mix)
        self._noise.setValue(spec.noise)
        self._seed.setValue(spec.seed)
        self._walk.setCurrentIndex(max(0, self._walk.findData(spec.walk)))
        self._period.setValue(max(1, spec.period))
        self._ar_prefix.setValue(max(0, spec.ar_prefix))
        self._ar_temperature.setValue(spec.ar_temperature)
        self._ar_cfg.setValue(spec.ar_cfg)
        self._loading = False

    def _apply(self) -> None:
        """Write the editor's values back into the selected entry."""
        item = self._list.currentItem()
        if self._loading or item is None:
            return
        spec = item.data(SPEC_ROLE)
        if spec is None:
            return
        updated = replace(
            spec,
            track_idx=(
                int(self._track.currentData() or SAME_TRACK)
                if self._track.currentData() is not None
                else SAME_TRACK
            ),
            track_b=(
                int(self._track_b.currentData())
                if self._track_b.currentData() is not None
                else SAME_TRACK
            ),
            window=self._window.value(),
            window_b=self._window_b.value(),
            mix=self._mix.value(),
            noise=self._noise.value(),
            seed=self._seed.value(),
            walk=self._walk.currentData() or "none",
            period=self._period.value(),
            ar_prefix=self._ar_prefix.value(),
            ar_temperature=self._ar_temperature.value(),
            ar_cfg=self._ar_cfg.value(),
        )
        item.setData(SPEC_ROLE, updated)
        item.setText(self._text(updated))
        if updated.walk != spec.walk:
            self._load_current()  # the period / ar rows appear or go with the walk
        self.changed.emit()


__all__ = ["SPEC_ROLE", "StyleListEditor"]
