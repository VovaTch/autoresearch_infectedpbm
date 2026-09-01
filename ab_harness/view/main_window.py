"""
Window shell: wires the rating view to the two view-models.

All the logic lives one layer down. This file only connects signals and renders
status, which is what keeps the rating loop testable without a display.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMainWindow,
    QStatusBar,
)

from ab_harness.model.types import Pair, Tier
from ab_harness.view.rating_view import RatingView
from ab_harness.view.worklist import DEFAULT_QUIET_FILL
from ab_harness.viewmodel.player_vm import PlayerViewModel
from ab_harness.viewmodel.session_vm import SessionViewModel


def _short(checkpoint: str) -> str:
    """
    Args:
      checkpoint (str): repo-relative checkpoint path.

    Returns:
      str: "<run dir>/<file stem>", which is what tells two runs apart in a
        status bar without spending it all on one label.
    """
    parts = checkpoint.replace("\\", "/").split("/")
    stem = parts[-1].removesuffix(".ckpt")
    return f"{parts[-2]}/{stem}" if len(parts) > 1 else stem


class MainWindow(QMainWindow):
    """
    The harness window.

    Args:
      session (SessionViewModel): rating loop.
      player (PlayerViewModel): playback transport.
      quiet_fill (float): fill fraction below which a queued pair counts as
        mostly empty and can be hidden from the worklist.
      checkpoints (list[str] | None): models offered in the selector, current
        one first. None hides the selector entirely, which is what tests and a
        repo with nothing trained get.
    """

    def __init__(
        self,
        session: SessionViewModel,
        player: PlayerViewModel,
        quiet_fill: float = DEFAULT_QUIET_FILL,
        checkpoints: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.player = player
        self.view = RatingView(self, quiet_fill=quiet_fill)
        self.setCentralWidget(self.view)
        self.setWindowTitle("A/B harness")
        self.resize(880, 520)

        self._checkpoints = QComboBox(self)
        # No focus: the rating view owns the keyboard, and a combo that steals
        # space or the arrow keys mid-session breaks the whole transport.
        self._checkpoints.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._checkpoints.setToolTip("model new pairs are drawn from")
        for path in checkpoints or []:
            self._checkpoints.addItem(_short(path), path)
        self._checkpoints.setVisible(bool(checkpoints))
        self._checkpoints.activated.connect(self._on_checkpoint_picked)

        self._rated = QLabel("0 rated", self)
        self._queue = QLabel("queue 0/0", self)
        self._structure = QLabel("90 s: 0 banked", self)
        self._rejected = QLabel("", self)
        status = QStatusBar(self)
        status.addPermanentWidget(self._rejected)
        status.addPermanentWidget(self._structure)
        status.addPermanentWidget(self._queue)
        status.addWidget(self._rated)
        status.addWidget(self._checkpoints)
        self.setStatusBar(status)

        self.view.chose.connect(self._on_chose)
        self.view.skipped.connect(session.skip)
        self.view.listen.connect(player.select)
        self.view.flip.connect(player.flip)
        self.view.play_pause.connect(player.toggle_play)
        self.view.restart.connect(player.restart)
        self.view.listen_from_start.connect(self._listen_from_start)
        self.view.picked.connect(self._on_picked)
        self.view.generate.connect(self._on_generate)

        session.pair_changed.connect(self._on_pair)
        session.waiting.connect(self.view.set_waiting)
        session.queue_changed.connect(self._on_queue)
        session.structure_changed.connect(self._on_structure)
        session.rejected_changed.connect(self._on_rejected)
        session.counts_changed.connect(lambda n: self._rated.setText(f"{n} rated"))
        session.worklist_changed.connect(self.view.set_worklist)
        session.checkpoint_changed.connect(self._on_checkpoint_changed)

        player.position_changed.connect(self.view.set_position)
        player.side_changed.connect(self.view.set_active_side)
        player.playing_changed.connect(self.view.set_playing)

        self.view.setFocus()

    def _on_checkpoint_picked(self, index: int) -> None:
        """
        Args:
          index (int): row the rater chose. The selector is disabled until the
            worker answers, because a second switch queued behind the first
            would drop a queue that was already being refilled.
        """
        path = self._checkpoints.itemData(index)
        if not path or path == self.session.checkpoint:
            return
        self.player.stop()
        self._checkpoints.setEnabled(False)
        self.session.switch_checkpoint(path)

    def _on_checkpoint_changed(self, checkpoint: str, error: str) -> None:
        """
        Args:
          checkpoint (str): the model now loaded, which on failure is the one
            that was already there.
          error (str): empty on success.
        """
        self._checkpoints.setEnabled(True)
        index = self._checkpoints.findData(checkpoint)
        if index >= 0:
            self._checkpoints.setCurrentIndex(index)
        if error and (bar := self.statusBar()) is not None:
            bar.showMessage(f"checkpoint load failed: {error}", 10000)

    def _on_pair(self, pair: Pair) -> None:
        """
        Args:
          pair (Pair): the comparison to show and start playing.
        """
        self.view.show_pair(pair)
        self.player.load(pair.left.pcm, pair.right.pcm)
        self.view.setFocus()

    def _on_chose(self, choice: str) -> None:
        """
        Args:
          choice (str): "left", "right" or "tie".
        """
        self.player.stop()
        self.session.choose(choice)  # type: ignore[arg-type]

    def _on_picked(self, pair_id: str) -> None:
        """
        Args:
          pair_id (str): worklist entry the rater chose; ignored if it is not
            ready yet, so a click on a generating row does nothing rather than
            clearing the pair on screen.
        """
        if self.session.select(pair_id):
            self.view.setFocus()

    def _on_generate(self, tier: str, count: int) -> None:
        """
        Args:
          tier (str): tier value from the worklist buttons.
          count (int): how many pairs to queue.
        """
        self.session.request(Tier(tier), count)
        self.view.setFocus()

    def _listen_from_start(self, side: int) -> None:
        """
        Args:
          side (int): 0 or 1; select that side and rewind, for the structure
            tier where each candidate is judged whole.
        """
        self.player.select(side)
        self.player.restart()

    def _on_queue(self, ready: int, inflight: int) -> None:
        """
        Args:
          ready (int): pairs decoded and waiting.
          inflight (int): pairs still being produced.
        """
        self._queue.setText(f"queue {ready}/{ready + inflight}")

    def _on_rejected(self, count: int) -> None:
        """
        Args:
          count (int): pairs dropped for being mostly dead air.
        """
        self._rejected.setText(f"{count} dropped (empty)" if count else "")

    def _on_structure(self, banked: int, building: int, skipped: int) -> None:
        """
        Args:
          banked (int): structure comparisons available to draw.
          building (int): structure pairs generating in the background.
          skipped (int): structure draws declined so far for want of material.
        """
        parts = [f"90 s: {banked} banked"]
        if building:
            parts.append("building")
        if skipped:
            parts.append(f"{skipped} skipped")
        self._structure.setText(" · ".join(parts))

    def closeEvent(self, event) -> None:
        """
        Args:
          event (QCloseEvent): the close request; stops audio and the pump.
        """
        self.player.stop()
        self.session.stop()
        super().closeEvent(event)
