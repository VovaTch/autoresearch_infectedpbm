"""
The control panel: everything that decides what gets generated.

Laid out top to bottom in the order the decisions actually get made -- which
model, which tracks, which mood, what to start from, how to sample -- and every
axis that can hold more than one value does, because the cross product is the
feature. Sampling is bound by kernel launch latency rather than arithmetic, so
sixteen variants cost about what one does; a panel that only let you ask for one
would be throwing that away.

Specs are built here, fully resolved, rather than as a template the worker
expands. That is deliberate: the panel is the only layer that knows each track's
length, so "prompt from 35% in" becomes a concrete start frame per track before
anything is queued, and every spec that reaches the worker is exactly
reproducible on its own.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ab_harness.checkpoints import backend_of
from ab_harness.model.pair_sampler import TrackInfo
from slice_synth.config import UiCfg
from slice_synth.model.types import (
    SAME_TRACK,
    PromptSpec,
    RenderSpec,
    StyleSpec,
    build_variants,
)
from slice_synth.view.style_editor import StyleListEditor
from slice_synth.worker.protocol import WorkerReady

TRACK_ROLE = Qt.ItemDataRole.UserRole
FOLLOW = -1  # prompt from whichever track the render is conditioned on
SLIDER_STEPS = 1000


class ControlPanel(QWidget):
    """
    Conditioning and sampling controls.

    Args:
      ui (UiCfg): opening values for every knob.
      checkpoints (list[str] | dict[str, list[str]] | None): models offered
        in the selector, current first -- a flat list, or one list per backend
        ("ar", "zflow", "mdm"), which also shows a backend picker when more
        than one backend has something trained. None hides the selectors,
        which is what tests get.
      parent (QWidget | None): Qt parent.
    """

    generate_requested = Signal(object)
    cancel_requested = Signal()
    checkpoint_picked = Signal(str)

    def __init__(
        self,
        ui: UiCfg,
        checkpoints: list[str] | dict[str, list[str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.ui = ui
        self._tracks: list[TrackInfo] = []
        self._fps = 172.265625
        self._menus = _menus(checkpoints)
        offered = [b for b, paths in self._menus.items() if paths]
        self._checkpoint = self._menus[offered[0]][0] if offered else ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.backends = QComboBox(self)
        self.backends.setObjectName("backends")
        self.backends.setToolTip(
            "generation backend: ar = token model, zflow = latent flow, "
            "mdm = masked diffusion. Picking one loads its newest checkpoint."
        )
        for backend in offered:
            self.backends.addItem(backend, backend)
        self.backends.setVisible(len(offered) > 1)
        self.backends.activated.connect(self._on_backend)
        layout.addWidget(self.backends)

        self.checkpoints = QComboBox(self)
        self.checkpoints.setObjectName("checkpoints")
        self.checkpoints.setToolTip("model every new variant is sampled from")
        self.checkpoints.setVisible(bool(offered))
        self.checkpoints.activated.connect(self._on_checkpoint)
        layout.addWidget(self.checkpoints)

        layout.addWidget(self._track_box())
        layout.addWidget(self._style_box())
        layout.addWidget(self._prompt_box())
        layout.addWidget(self._sampling_box())
        if offered:
            self._show_backend(offered[0])

        self.generate = QPushButton("Generate", self)
        self.generate.setDefault(True)
        self.generate.clicked.connect(self._on_generate)
        self.cancel = QPushButton("Cancel", self)
        self.cancel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_requested)
        row = QHBoxLayout()
        row.addWidget(self.generate, 2)
        row.addWidget(self.cancel, 1)
        layout.addLayout(row)

        self.progress = QProgressBar(self)
        self.progress.setRange(0, SLIDER_STEPS)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("idle")
        layout.addWidget(self.progress)

        self.count = QLabel("", self)
        self.count.setWordWrap(True)
        layout.addWidget(self.count)
        layout.addStretch(1)

        self.set_enabled(False)
        self._update_count()

    # -- sections ------------------------------------------------------------

    def _track_box(self) -> QGroupBox:
        """
        Returns:
          QGroupBox: the track picker, with a filter and a null-id row.
        """
        box = QGroupBox("tracks", self)
        inner = QVBoxLayout(box)
        self.filter = QLineEdit(box)
        self.filter.setPlaceholderText("filter...")
        self.filter.textChanged.connect(self._apply_filter)
        inner.addWidget(self.filter)

        self.track_list = QListWidget(box)
        self.track_list.setMinimumHeight(150)
        self.track_list.itemChanged.connect(lambda _: self._update_count())
        inner.addWidget(self.track_list)

        row = QHBoxLayout()
        for text, state in (("all", True), ("none", False)):
            button = QPushButton(text, box)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _=False, s=state: self._check_visible(s))
            row.addWidget(button)
        row.addStretch(1)
        inner.addLayout(row)

        # separately = one clip per ticked track (the cross product);
        # together = every ticked track as a coexisting id token in ONE clip
        self.combine = QButtonGroup(box)
        mode = QHBoxLayout()
        for index, (text, tip) in enumerate(
            (
                ("separately", "one clip per ticked track"),
                (
                    "together",
                    "every ticked track as its own id token in one clip, all at "
                    "the same position -- a mishmash. The null row stays separate.",
                ),
            )
        ):
            button = QRadioButton(text, box)
            button.setToolTip(tip)
            button.setChecked(index == 0)
            self.combine.addButton(button, index)
            mode.addWidget(button)
        mode.addStretch(1)
        inner.addLayout(mode)
        self.combine.idToggled.connect(lambda _i, _on: self._update_count())
        return box

    def _style_box(self) -> QGroupBox:
        """
        Returns:
          QGroupBox: the style recipe list.
        """
        box = QGroupBox("style embeddings", self)
        inner = QVBoxLayout(box)
        self.styles = StyleListEditor(
            box,
            StyleSpec(
                track_idx=SAME_TRACK,
                track_b=SAME_TRACK,
                walk=self.ui.style_walk,  # type: ignore[arg-type]
                ar_temperature=self.ui.style_ar_temperature,
                ar_cfg=self.ui.style_ar_cfg,
            ),
        )
        self.styles.changed.connect(self._update_count)
        inner.addWidget(self.styles)
        return box

    def _prompt_box(self) -> QGroupBox:
        """
        Returns:
          QGroupBox: the prompt source picker.
        """
        box = QGroupBox("prompt", self)
        inner = QVBoxLayout(box)

        self.prompt_kind = QButtonGroup(box)
        row = QHBoxLayout()
        for index, (text, tip) in enumerate(
            (
                (
                    "none",
                    "cold start: measurably the worst setting -- beat 0.41 vs 0.55, tempo off-grid 19 times in 32",
                ),
                ("corpus", "continue a span of a cached track"),
                ("file", "continue any audio file, encoded on the fly"),
            )
        ):
            button = QRadioButton(text, box)
            button.setToolTip(tip)
            button.setChecked(index == 1)
            self.prompt_kind.addButton(button, index)
            row.addWidget(button)
        row.addStretch(1)
        inner.addLayout(row)
        self.prompt_kind.idToggled.connect(lambda _i, _on: self._sync_prompt())

        form = QFormLayout()
        self.prompt_track = QComboBox(box)
        self.prompt_track.setToolTip("which track the priming audio comes from")
        form.addRow("track", self.prompt_track)

        self.start = QSlider(Qt.Orientation.Horizontal, box)
        self.start.setRange(0, SLIDER_STEPS)
        self.start.setValue(350)
        self.start.valueChanged.connect(self._sync_start_label)
        self.start_label = QLabel("35%", box)
        self._start_row = QHBoxLayout()
        self._start_row.addWidget(self.start, 3)
        self._start_row.addWidget(self.start_label, 1)
        form.addRow("start", self._start_row)

        self.prompt_path = QLineEdit(box)
        self.prompt_path.setPlaceholderText("audio file...")
        browse = QPushButton("...", box)
        browse.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        browse.setMaximumWidth(32)
        browse.clicked.connect(self._browse)
        self._path_row = QHBoxLayout()
        self._path_row.addWidget(self.prompt_path, 4)
        self._path_row.addWidget(browse, 1)
        self._browse_button = browse
        form.addRow("file", self._path_row)

        self.file_start = QDoubleSpinBox(box)
        self.file_start.setRange(0.0, 36000.0)
        self.file_start.setSingleStep(1.0)
        self.file_start.setSuffix(" s")
        form.addRow("file at", self.file_start)

        self.prompt_seconds = QDoubleSpinBox(box)
        self.prompt_seconds.setRange(0.1, 60.0)
        self.prompt_seconds.setSingleStep(0.5)
        self.prompt_seconds.setMaximum(max(0.1, self.ui.seconds / 2.0))
        self.prompt_seconds.setValue(self.ui.prompt_seconds)
        self.prompt_seconds.setSuffix(" s")
        form.addRow("length", self.prompt_seconds)

        inner.addLayout(form)
        self._prompt_form = form
        return box

    def _sampling_box(self) -> QGroupBox:
        """
        Returns:
          QGroupBox: clip length and decoder settings.
        """
        box = QGroupBox("sampling", self)
        form = QFormLayout(box)

        self.seconds = QDoubleSpinBox(box)
        self.seconds.setRange(1.0, 180.0)
        self.seconds.setSingleStep(1.0)
        self.seconds.setValue(self.ui.seconds)
        self.seconds.setSuffix(" s")
        self.seconds.valueChanged.connect(self._cap_prompt)
        form.addRow("length", self.seconds)

        self.temperature = QDoubleSpinBox(box)
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setDecimals(2)
        self.temperature.setValue(self.ui.temperature)
        self.temperature.setToolTip("0 is argmax")
        form.addRow("temperature", self.temperature)

        self.top_k = QSpinBox(box)
        self.top_k.setRange(0, 2048)
        self.top_k.setValue(self.ui.top_k)
        self.top_k.setToolTip("0 disables")
        form.addRow("top-k", self.top_k)

        self.top_p = QDoubleSpinBox(box)
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setDecimals(2)
        self.top_p.setValue(self.ui.top_p)
        self.top_p.setToolTip("0 disables")
        form.addRow("top-p", self.top_p)

        self.cfg = QDoubleSpinBox(box)
        self.cfg.setRange(0.0, self.ui.max_cfg)
        self.cfg.setSingleStep(0.5)
        self.cfg.setDecimals(2)
        self.cfg.setValue(self.ui.cfg_strength)
        self.cfg.setToolTip(
            "guidance. 0 and 1 are both plain conditional sampling; track "
            "identity is at chance until roughly 8. Each guided variant costs "
            "two KV rows, so it halves how many fit in one pass."
        )
        form.addRow("cfg", self.cfg)

        self.seed = QSpinBox(box)
        self.seed.setRange(0, 2**31 - 1)
        form.addRow("seed", self.seed)

        self.seeds = QSpinBox(box)
        self.seeds.setRange(1, 32)
        self.seeds.setValue(self.ui.seeds)
        self.seeds.setToolTip("consecutive seeds per conditioning cell")
        self.seeds.valueChanged.connect(self._update_count)
        form.addRow("seeds", self.seeds)

        self.reference = QCheckBox("decode real tokens too", box)
        self.reference.setToolTip(
            "one extra clip per track, decoded from the real codes for the same "
            "span. The tokenizer alone accounts for 55% of the spectral error, "
            "so this is the ceiling a generation is being judged against."
        )
        self.reference.stateChanged.connect(lambda _: self._update_count())
        form.addRow("", self.reference)
        return box

    # -- content -------------------------------------------------------------

    def set_corpus(self, ready: WorkerReady) -> None:
        """
        Args:
          ready (WorkerReady): the worker's corpus publication.
        """
        self._tracks = list(ready.tracks)
        self._fps = float(ready.meta.get("frames_per_second", self._fps))
        self._checkpoint = ready.checkpoint

        self.track_list.blockSignals(True)
        self.track_list.clear()
        null = QListWidgetItem("null  (unconditional id)")
        null.setToolTip("drop the track-id stream and use the model's learned null")
        null.setFlags(null.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        null.setCheckState(Qt.CheckState.Unchecked)
        null.setData(TRACK_ROLE, None)
        self.track_list.addItem(null)
        for track in self._tracks:
            item = QListWidgetItem(f"{track.track_idx:02d}  {track.track_name[:44]}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if track.track_idx == 0
                else Qt.CheckState.Unchecked
            )
            item.setData(TRACK_ROLE, track.track_idx)
            self.track_list.addItem(item)
        self.track_list.blockSignals(False)

        self.prompt_track.clear()
        self.prompt_track.addItem("follow (this render)", FOLLOW)
        for track in self._tracks:
            self.prompt_track.addItem(
                f"{track.track_idx:02d} {track.track_name[:38]}", track.track_idx
            )
        self.styles.set_corpus(self._tracks)

        index = self.checkpoints.findData(ready.checkpoint)
        if index >= 0:
            self.checkpoints.setCurrentIndex(index)
        self.set_enabled(True)
        self._sync_prompt()
        self._update_count()

    def set_enabled(self, enabled: bool) -> None:
        """
        Args:
          enabled (bool): whether the panel accepts input. Off until the worker
            has published a corpus -- there is nothing to pick from before that.
        """
        self.generate.setEnabled(enabled)
        self.track_list.setEnabled(enabled)
        for button in self.combine.buttons():
            button.setEnabled(enabled)
        self.styles.setEnabled(enabled)

    def set_busy(self, busy: bool) -> None:
        """
        Args:
          busy (bool): whether a batch is in flight.
        """
        self.cancel.setEnabled(busy)
        self.checkpoints.setEnabled(not busy)
        self.backends.setEnabled(not busy)
        if not busy:
            self.progress.setValue(0)
            self.progress.setFormat("idle")

    def set_progress(self, fraction: float, step: int, total: int) -> None:
        """
        Args:
          fraction (float): progress in [0, 1].
          step (int): positions produced.
          total (int): positions in the batch.
        """
        self.progress.setValue(int(fraction * SLIDER_STEPS))
        self.progress.setFormat(f"{step}/{total} positions" if total else "idle")

    def set_checkpoint(self, checkpoint: str) -> None:
        """
        Args:
          checkpoint (str): the model now loaded.
        """
        self._checkpoint = checkpoint
        if checkpoint and self.checkpoints.findData(checkpoint) < 0:
            self._show_backend(backend_of(checkpoint))
        index = self.checkpoints.findData(checkpoint)
        if index >= 0:
            self.checkpoints.setCurrentIndex(index)
        self.checkpoints.setEnabled(True)
        self.backends.setEnabled(True)
        self._knobs_for(backend_of(checkpoint))

    def _show_backend(self, backend: str) -> None:
        """
        Args:
          backend (str): whose checkpoints the checkpoint selector lists.
        """
        row = self.backends.findData(backend)
        if row >= 0:
            self.backends.setCurrentIndex(row)
        self.checkpoints.clear()
        for path in self._menus.get(backend, []):
            self.checkpoints.addItem(_short(path), path)
        self._knobs_for(backend)

    def _knobs_for(self, backend: str) -> None:
        """
        Grey out the sampling knobs a backend ignores.

        Args:
          backend (str): "ar" (all live), "mdm" (top-p unused) or "zflow"
            (temperature / top-k / top-p unused: it integrates an ODE, and its
            own knobs are the config's flow_steps / flow_churn).
        """
        self.temperature.setEnabled(backend != "zflow")
        self.top_k.setEnabled(backend != "zflow")
        self.top_p.setEnabled(backend == "ar")

    # -- spec building -------------------------------------------------------

    def selected_tracks(self) -> list[int | None]:
        """
        Returns:
          list[int | None]: ticked track ids, with None for the null-id row.
        """
        out: list[int | None] = []
        for row in range(self.track_list.count()):
            item = self.track_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(TRACK_ROLE))
        return out

    @property
    def together(self) -> bool:
        """
        Returns:
          bool: True when ticked tracks share one clip rather than each its own.
        """
        return self.combine.checkedId() == 1

    def _prompt_for(self, track_idx: int | None, frames: int) -> PromptSpec:
        """
        Build the prompt for one track, resolving "follow" and the start slider.

        Args:
          track_idx (int | None): the track this render is conditioned on.
          frames (int): the clip length, so the start never lands past the point
            where a full clip still fits.

        Returns:
          PromptSpec: a fully concrete prompt.
        """
        kind = ("none", "corpus", "file")[self.prompt_kind.checkedId()]
        seconds = self.prompt_seconds.value()
        if kind == "file":
            return PromptSpec(
                kind="file",
                seconds=seconds,
                path=self.prompt_path.text().strip(),
                start_sec=self.file_start.value(),
            )
        chosen = self.prompt_track.currentData()
        source = track_idx if chosen == FOLLOW else chosen
        if source is None:
            # A null-id render still has to prime from somewhere; the first
            # ticked real track is the least surprising choice.
            source = next(
                (t for t in self.selected_tracks() if t is not None),
                self._tracks[0].track_idx if self._tracks else 0,
            )
        info = next((t for t in self._tracks if t.track_idx == source), None)
        length = info.num_frames if info else frames + 1
        fraction = self.start.value() / SLIDER_STEPS
        start = int(max(0, min(length - frames - 1, length * fraction)))
        return PromptSpec(
            kind=kind, track_idx=int(source), start_frame=start, seconds=seconds
        )

    def build_specs(self) -> list[RenderSpec]:
        """
        Expand the panel into one spec per combination.

        Returns:
          list[RenderSpec]: the batch to queue. Empty when nothing is ticked.
        """
        tracks = self.selected_tracks()
        styles = self.styles.styles()
        if not tracks or not styles:
            return []
        frames = int(self.seconds.value() * self._fps)
        base_seed = self.seed.value()
        seeds = [base_seed + i for i in range(self.seeds.value())]

        # Each cell gets its own base because the prompt "follows" its track.
        # Together, the blend follows its primary (first ticked real) track and
        # a ticked null still stands alone.
        cells: list[list[int | None]] = [[t] for t in tracks]
        if self.together:
            real: list[int | None] = [t for t in tracks if t is not None]
            cells = [real] if real else []
            if None in tracks:
                cells.append([None])

        specs: list[RenderSpec] = []
        for cell in cells:
            base = RenderSpec(
                n_frames=frames,
                prompt=self._prompt_for(cell[0], frames),
                cfg_strength=self.cfg.value(),
                temperature=self.temperature.value(),
                top_k=self.top_k.value(),
                top_p=self.top_p.value(),
                checkpoint=self._checkpoint,
            )
            specs += build_variants(
                base, cell, styles, seeds, self.reference.isChecked(), self.together
            )
        return specs

    # -- wiring --------------------------------------------------------------

    def _on_generate(self) -> None:
        """Emit the built batch, or say why there isn't one."""
        specs = self.build_specs()
        if not specs:
            self.count.setText("tick at least one track and one style entry")
            return
        self.generate_requested.emit(specs)

    def _on_checkpoint(self, index: int) -> None:
        """
        Args:
          index (int): row picked in the selector.
        """
        path = self.checkpoints.itemData(index)
        if path and path != self._checkpoint:
            self.checkpoints.setEnabled(False)
            self.backends.setEnabled(False)
            self.checkpoint_picked.emit(path)

    def _on_backend(self, index: int) -> None:
        """
        Args:
          index (int): backend row picked; its newest checkpoint is loaded.
        """
        backend = self.backends.itemData(index)
        if not backend or not self._menus.get(backend):
            return
        self._show_backend(backend)
        self._on_checkpoint(0)

    def _browse(self) -> None:
        """Pick a prompt file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "prompt audio", "", "Audio (*.wav *.mp3 *.flac *.ogg *.m4a);;All (*)"
        )
        if path:
            self.prompt_path.setText(path)
            self.prompt_kind.button(2).setChecked(True)

    def _apply_filter(self, text: str) -> None:
        """
        Args:
          text (str): substring to match against track names.
        """
        needle = text.strip().lower()
        for row in range(self.track_list.count()):
            item = self.track_list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _check_visible(self, checked: bool) -> None:
        """
        Args:
          checked (bool): "all" ticks every row the filter is showing, which is
            what makes "everything matching 'infected'" one gesture. "none"
            clears every row including the hidden ones -- a filtered-away track
            that stayed ticked would generate without appearing anywhere.
        """
        self.track_list.blockSignals(True)
        for row in range(self.track_list.count()):
            item = self.track_list.item(row)
            if checked and item.isHidden():
                continue
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        self.track_list.blockSignals(False)
        self._update_count()

    def _cap_prompt(self, seconds: float) -> None:
        """
        Keep the prompt shorter than the clip.

        A prompt as long as the render leaves nothing generated: the whole clip
        would be forced real audio wearing a generation's label, which is the
        one result that could not be told apart by ear. Half the length is the
        cap, so there is always as much model as prompt.

        Args:
          seconds (float): the new clip length.
        """
        self.prompt_seconds.setMaximum(max(0.1, seconds / 2.0))
        self._update_count()

    def _sync_start_label(self, value: int) -> None:
        """
        Args:
          value (int): slider position.
        """
        self.start_label.setText(f"{100 * value // SLIDER_STEPS}%")

    def _sync_prompt(self) -> None:
        """
        Show only the fields the chosen prompt source uses.

        Rows built from a layout rather than a bare widget have to be looked up
        by that layout: labelForField(widget) returns nothing for them, which
        leaves a stranded label beside empty space.
        """
        kind = self.prompt_kind.checkedId()
        rows: list[tuple[object, list[QWidget], set[int]]] = [
            (self.prompt_track, [self.prompt_track], {1}),
            (self._start_row, [self.start, self.start_label], {1}),
            (self._path_row, [self.prompt_path, self._browse_button], {2}),
            (self.file_start, [self.file_start], {2}),
            (self.prompt_seconds, [self.prompt_seconds], {1, 2}),
        ]
        for field, widgets, kinds in rows:
            visible = kind in kinds
            for widget in widgets:
                widget.setVisible(visible)
            label = self._prompt_form.labelForField(field)  # type: ignore[arg-type]
            if isinstance(label, QLabel):
                label.setVisible(visible)

    def _update_count(self) -> None:
        """Say how many clips the current selection would produce."""
        specs = self.build_specs()
        if not specs:
            self.count.setText("nothing selected")
            return
        guided = sum(1 for s in specs if s.cfg_strength > 0 and s.style is not None)
        note = f" ({guided} guided, 2 KV rows each)" if guided else ""
        blended = next((s for s in specs if s.co_tracks), None)
        if blended is not None:
            note += f", ids {'+'.join(map(str, blended.tracks))} in one clip"
        self.count.setText(f"{len(specs)} variants{note}")


def _menus(
    checkpoints: list[str] | dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    """
    Args:
      checkpoints (list[str] | dict[str, list[str]] | None): selector input.

    Returns:
      dict[str, list[str]]: backend -> checkpoints, a flat list split by the
        family of each path.
    """
    if checkpoints is None:
        return {}
    if isinstance(checkpoints, dict):
        return {b: list(paths) for b, paths in checkpoints.items()}
    menus: dict[str, list[str]] = {}
    for path in checkpoints:
        menus.setdefault(backend_of(path), []).append(path)
    return menus


def _short(checkpoint: str) -> str:
    """
    Args:
      checkpoint (str): repo-relative checkpoint path.

    Returns:
      str: "<run dir>/<file stem>", enough to tell two runs apart in a combo box.
    """
    parts = checkpoint.replace("\\", "/").split("/")
    stem = parts[-1].removesuffix(".ckpt")
    return f"{parts[-2]}/{stem}" if len(parts) > 1 else stem


__all__ = ["FOLLOW", "TRACK_ROLE", "ControlPanel"]
