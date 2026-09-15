"""
The results half: pick a variant, hear it, look at it, keep it.

Three strips share one time axis and one playhead -- envelope, mel, tokens --
because the three questions a bad render raises are answered in different ones.
Is it there at all (waveform), does it still have a top end (mel), did the model
collapse or start looping (tokens). Clicking any of them moves the playhead in
all of them, which is how you get to the interesting eight seconds of a ninety
second clip without listening to the other eighty-two.

Nothing here is blind. The rating harness hides everything about a clip on
purpose; this panel does the opposite and prints the recipe next to the audio,
because the whole point is learning which settings produce what.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ab_harness.view.waveform import WaveformView
from slice_synth.model.tokens import stats_line
from slice_synth.model.types import Render
from slice_synth.view.mel import MelView
from slice_synth.view.tokens import TokenView

ITEM_ROLE = Qt.ItemDataRole.UserRole
STATUS_MARK = {"generating": "…", "ready": "●", "failed": "✗"}


class ResultsPanel(QWidget):
    """
    Variant list, the three strips, transport and save.

    Args:
      mel_columns (int): mel time resolution cap.
      parent (QWidget | None): Qt parent.
    """

    selected = Signal(str)
    save_requested = Signal()
    clear_requested = Signal()
    scrubbed = Signal(float)
    play_pause = Signal()
    restart = Signal()
    seek_by = Signal(float)

    def __init__(self, mel_columns: int = 1024, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fps = 172.265625
        self._num_tokens = 2048
        self._render: Render | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.variants = QListWidget(self)
        self.variants.setMinimumHeight(110)
        self.variants.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.variants.currentItemChanged.connect(self._on_current)

        self.wave = WaveformView(72, self)
        self.mel = MelView(160, mel_columns, self)
        self.tokens = TokenView(54, self._num_tokens, self)
        for strip in (self.wave, self.mel, self.tokens):
            strip.scrubbed.connect(self.scrubbed)

        self.stats = QLabel("", self)
        self.stats.setTextFormat(Qt.TextFormat.PlainText)
        self.recipe = QLabel("", self)
        self.recipe.setWordWrap(True)
        self.position = QLabel("0.0 / 0.0 s", self)

        self.play = QPushButton("▶ play", self)
        self.play.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play.clicked.connect(self.play_pause)
        self.rewind = QPushButton("↺", self)
        self.rewind.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.rewind.setMaximumWidth(40)
        self.rewind.clicked.connect(self.restart)
        self.save = QPushButton("Save", self)
        self.save.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.save.setEnabled(False)
        self.save.clicked.connect(self.save_requested)
        self.clear = QPushButton("Clear list", self)
        self.clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clear.clicked.connect(self.clear_requested)

        transport = QHBoxLayout()
        transport.addWidget(self.play)
        transport.addWidget(self.rewind)
        transport.addWidget(self.position)
        transport.addStretch(1)
        transport.addWidget(self.save)
        transport.addWidget(self.clear)

        detail = QWidget(self)
        inner = QVBoxLayout(detail)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(self.wave)
        inner.addWidget(self.mel, 1)
        inner.addWidget(self.tokens)
        inner.addWidget(self.stats)
        inner.addLayout(transport)
        inner.addWidget(self.recipe)

        # A splitter rather than a stretch factor: a sixteen-variant batch wants
        # a tall list and a single render wants none of one, and only the person
        # listening knows which they are doing.
        self.split = QSplitter(Qt.Orientation.Vertical, self)
        self.split.addWidget(self.variants)
        self.split.addWidget(detail)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([180, 620])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.split)

    # -- content -------------------------------------------------------------

    def set_geometry_meta(self, fps: float, num_tokens: int) -> None:
        """
        Args:
          fps (float): tokenizer frames per second, for reporting loop periods.
          num_tokens (int): codebook size, for the palette and the entropy
            ceiling.
        """
        self._fps = fps
        self._num_tokens = num_tokens
        self.tokens.set_num_tokens(num_tokens)

    def set_variants(self, variants: list) -> None:
        """
        Refresh the list, keeping the current selection where possible.

        Args:
          variants (list): Variant rows from the view-model.
        """
        current = self.current_item_id()
        self.variants.blockSignals(True)
        self.variants.clear()
        for variant in variants:
            item = QListWidgetItem(self._row_text(variant))
            item.setData(ITEM_ROLE, variant.item_id)
            if variant.status == "failed":
                item.setToolTip(variant.error)
            self.variants.addItem(item)
            if variant.item_id == current:
                self.variants.setCurrentItem(item)
        self.variants.blockSignals(False)
        if self.variants.currentItem() is None and self.variants.count():
            # Nothing was selected, or the selection was cancelled away.
            self.variants.setCurrentRow(0)

    def show_render(self, render: Render | None) -> None:
        """
        Args:
          render (Render | None): the clip to display, or None to clear.
        """
        self._render = render
        self.save.setEnabled(render is not None)
        if render is None:
            self.wave.set_pcm(np.zeros(0, dtype=np.int16))
            self.mel.set_pcm(np.zeros(0, dtype=np.int16))
            self.tokens.set_tokens(np.zeros((0, 0), dtype=np.int64))
            self.stats.setText("")
            self.recipe.setText("")
            self.position.setText("0.0 / 0.0 s")
            return

        self.wave.set_pcm(render.pcm)
        self.mel.set_pcm(render.pcm, render.sample_rate)
        self.tokens.set_tokens(render.tokens)

        # Everything left of the marker was forced to real audio, so it is not
        # the model's work and should not be read as if it were.
        # Clamped: a saved recipe from another session, or a config edit, can
        # carry a prompt longer than the clip. The stats then have to describe
        # something rather than an empty slice.
        prompt_frames = (
            0
            if render.spec.is_reference
            else min(
                render.spec.prompt.frames(self._fps), max(0, render.spec.n_frames - 1)
            )
        )
        marker = min(1.0, prompt_frames / max(1, render.spec.n_frames))
        for strip in (self.mel, self.tokens):
            strip.set_marker(marker)

        generated = render.tokens[prompt_frames:] if prompt_frames else render.tokens
        self.stats.setText(stats_line(generated, self._num_tokens, self._fps))
        self.recipe.setText(self._recipe_text(render))

    def set_position(self, seconds: float, duration: float) -> None:
        """
        Args:
          seconds (float): playhead position.
          duration (float): clip length.
        """
        fraction = seconds / duration if duration > 0 else 0.0
        self.wave.set_position(fraction)
        self.mel.set_position(fraction)
        self.tokens.set_position(fraction)
        self.position.setText(f"{seconds:.1f} / {duration:.1f} s")

    def set_playing(self, playing: bool) -> None:
        """
        Args:
          playing (bool): transport state.
        """
        self.play.setText("❚❚ pause" if playing else "▶ play")

    def current_item_id(self) -> str:
        """
        Returns:
          str: the selected variant's id, or "" when nothing is selected.
        """
        item = self.variants.currentItem()
        return str(item.data(ITEM_ROLE)) if item else ""

    # -- rendering helpers ---------------------------------------------------

    def _row_text(self, variant) -> str:
        """
        Args:
          variant: a Variant row.

        Returns:
          str: its one-line description.
        """
        mark = STATUS_MARK.get(variant.status, "?")
        kept = "  saved" if variant.saved is not None else ""
        if variant.status == "ready" and variant.render is not None:
            fill = variant.render.fill
            extra = f"  {variant.render.seconds:.0f}s" + (
                f"  fill {fill:.2f}" if fill >= 0 else ""
            )
        elif variant.status == "failed":
            extra = f"  {variant.error[:40]}"
        else:
            extra = "  generating"
        return f"{mark}  {variant.spec.label()}{extra}{kept}"

    def _recipe_text(self, render: Render) -> str:
        """
        Args:
          render (Render): the displayed clip.

        Returns:
          str: the settings that produced it, spelled out.
        """
        spec = render.spec
        prompt = spec.prompt
        if spec.is_reference:
            source = f"real tokens, track {prompt.track_idx} @ {prompt.start_frame}"
            return f"{source}  ·  {render.seconds:.1f}s  ·  the tokenizer's ceiling"
        if prompt.kind == "none":
            primed = "cold start"
        elif prompt.kind == "file":
            primed = f"file {prompt.path.rsplit('/', 1)[-1]} @ {prompt.start_sec:.0f}s, {prompt.seconds:g}s"
        else:
            primed = (
                f"track {prompt.track_idx} @ frame {prompt.start_frame} "
                f"({prompt.start_frame / self._fps:.0f}s), {prompt.seconds:g}s"
            )
        style = (
            "null" if spec.style is None else f"{spec.style.kind} {spec.style.label()}"
        )
        if spec.style is not None and spec.style.walking:
            style += f"  ·  walk {spec.style.walk}/{spec.style.period}"
        track = "null" if spec.track_idx is None else "+".join(map(str, spec.tracks))
        peak = float(np.abs(render.pcm).max()) / 32768.0 if render.pcm.size else 0.0
        return (
            f"id {track}  ·  style {style}  ·  prompt {primed}\n"
            f"cfg {spec.cfg_strength:g}  T {spec.temperature:g}  "
            f"top-k {spec.top_k}  top-p {spec.top_p:g}  seed {spec.seed}  ·  "
            f"peak {peak:.2f}  ·  {spec.checkpoint}"
        )

    # -- input ---------------------------------------------------------------

    def _on_current(self, item: QListWidgetItem | None, _previous=None) -> None:
        """
        Args:
          item (QListWidgetItem | None): the newly selected row.
        """
        self.selected.emit(str(item.data(ITEM_ROLE)) if item else "")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Args:
          event (QKeyEvent): a key press. Space plays, r rewinds, the arrows nudge
            the playhead, s saves -- the transport bindings a listening pass
            needs without reaching for the mouse.
        """
        key = event.key()
        step = 5.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        if key == Qt.Key.Key_Space:
            self.play_pause.emit()
        elif key == Qt.Key.Key_R:
            self.restart.emit()
        elif key == Qt.Key.Key_Left:
            self.seek_by.emit(-step)
        elif key == Qt.Key.Key_Right:
            self.seek_by.emit(step)
        elif key == Qt.Key.Key_S:
            self.save_requested.emit()
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            row = self.variants.currentRow() + (1 if key == Qt.Key.Key_Down else -1)
            if 0 <= row < self.variants.count():
                self.variants.setCurrentRow(row)
        else:
            super().keyPressEvent(event)


__all__ = ["ITEM_ROLE", "ResultsPanel"]
