"""
Window shell: wires the two panels to the view-model and the transport.

All the logic lives one layer down. This file connects signals and renders
status, which is what keeps the synthesizer testable without a display.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QStatusBar,
)

from ab_harness.viewmodel.player_vm import PlayerViewModel
from slice_synth.config import UiCfg
from slice_synth.model.types import Render
from slice_synth.view.controls import ControlPanel, _short
from slice_synth.view.results import ResultsPanel
from slice_synth.viewmodel.synth_vm import SynthViewModel
from slice_synth.worker.protocol import WorkerReady

MESSAGE_MS = 8000


class MainWindow(QMainWindow):
    """
    The synthesizer window.

    Args:
      vm (SynthViewModel): session state.
      player (PlayerViewModel): playback transport, borrowed from the rating
        harness. It is built for A/B, so the same buffer is loaded into both
        sides and the flip degenerates to a no-op -- cheaper than a second
        QAudioSink implementation that would drift from the first.
      ui (UiCfg): opening values for the controls.
      checkpoints (list[str] | None): models offered in the selector.
    """

    def __init__(
        self,
        vm: SynthViewModel,
        player: PlayerViewModel,
        ui: UiCfg,
        checkpoints: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.vm = vm
        self.player = player
        self.controls = ControlPanel(ui, checkpoints, self)
        self.results = ResultsPanel(ui.mel_columns, self)
        self.setWindowTitle("slice synthesizer")
        self.resize(1320, 860)

        scroll = QScrollArea(self)
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(520)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(scroll)
        splitter.addWidget(self.results)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._checkpoint = QLabel("loading...", self)
        self._counts = QLabel("", self)
        status = QStatusBar(self)
        status.addPermanentWidget(self._counts)
        status.addWidget(self._checkpoint)
        self.setStatusBar(status)

        self.controls.generate_requested.connect(vm.generate)
        self.controls.cancel_requested.connect(vm.cancel)
        self.controls.checkpoint_picked.connect(vm.switch_checkpoint)

        self.results.selected.connect(vm.select)
        self.results.save_requested.connect(vm.save_selected)
        self.results.clear_requested.connect(vm.clear)
        self.results.scrubbed.connect(player.seek_fraction)
        self.results.play_pause.connect(player.toggle_play)
        self.results.restart.connect(player.restart)
        self.results.seek_by.connect(self._seek_by)

        vm.corpus_changed.connect(self._on_corpus)
        vm.variants_changed.connect(self._on_variants)
        vm.selected_changed.connect(self._on_render)
        vm.progress_changed.connect(self.controls.set_progress)
        vm.busy_changed.connect(self.controls.set_busy)
        vm.message.connect(self._on_message)
        vm.checkpoint_changed.connect(self._on_checkpoint)

        player.position_changed.connect(self.results.set_position)
        player.playing_changed.connect(self.results.set_playing)

        self.results.setFocus()

    # -- slots ---------------------------------------------------------------

    def _on_corpus(self, ready: WorkerReady) -> None:
        """
        Args:
          ready (WorkerReady): the worker's corpus publication.
        """
        self.controls.set_corpus(ready)
        self.results.set_geometry_meta(
            float(ready.meta.get("frames_per_second", 172.265625)),
            int(ready.meta.get("num_tokens", 2048)),
        )
        style_ar = Path(ready.style_ar_checkpoint).parent.name
        self._counts.setText(
            f"{len(ready.tracks)} tracks · window {ready.window_frames}"
            + (f" · style-ar {style_ar}" if style_ar else "")
        )

    def _on_variants(self, variants: list) -> None:
        """
        Args:
          variants (list): the current variant rows.
        """
        self.results.set_variants(variants)
        ready = sum(1 for v in variants if v.status == "ready")
        self._counts.setText(f"{ready}/{len(variants)} ready")

    def _on_render(self, render: Render | None) -> None:
        """
        Args:
          render (Render | None): the clip now selected. Loaded without autoplay:
            a batch finishing while you are listening to an earlier variant
            should not talk over it.
        """
        self.results.show_render(render)
        if render is None:
            self.player.stop()
            return
        self.player.load(render.pcm, render.pcm, autoplay=False)
        self.results.set_position(0.0, render.seconds)

    def _on_message(self, text: str) -> None:
        """
        Args:
          text (str): a transient status message.
        """
        if (bar := self.statusBar()) is not None:
            bar.showMessage(text, MESSAGE_MS)

    def _on_checkpoint(self, checkpoint: str, error: str) -> None:
        """
        Args:
          checkpoint (str): the model now loaded.
          error (str): empty on success.
        """
        self.controls.set_checkpoint(checkpoint)
        self._checkpoint.setText(
            f"{_short(checkpoint)}{'  (load failed)' if error else ''}"
        )

    def _seek_by(self, seconds: float) -> None:
        """
        Args:
          seconds (float): offset to move the playhead by.
        """
        self.player.seek(self.player.source.seconds + seconds)

    def closeEvent(self, event) -> None:
        """
        Args:
          event (QCloseEvent): the close request; stops audio and the pump.
        """
        self.player.stop()
        self.vm.stop()
        super().closeEvent(event)


__all__ = ["MainWindow"]
