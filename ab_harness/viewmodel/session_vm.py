"""
The rating loop.

Owns the current pair, times the decision, and writes exactly one judgement per
comparison. The view knows only "left" and "right"; which checkpoint or seed sits
behind each side never reaches it, so nothing on screen can leak identity.

The pipeline is pumped on a timer rather than blocked on, so a slow generation
shows up as "preparing" in the status bar instead of a frozen window.
"""

from __future__ import annotations

import time
import uuid

from PySide6.QtCore import QObject, QTimer, Signal

from ab_harness.model.judgement import utc_now
from ab_harness.model.pipeline import PairPipeline, WorkItem
from ab_harness.model.protocols import JudgementSink
from ab_harness.model.types import Choice, Judgement, Pair, Side, Tier

PUMP_MS = 250


class SessionViewModel(QObject):
    """
    One rating session.

    Args:
      pipeline (PairPipeline): supplies ready pairs.
      sink (JudgementSink): where decisions are persisted.
      target_lufs (float): loudness both clips were matched to; logged so a
        later change to the target is visible in the data.
      session_id (str | None): identifier, generated when omitted.
      parent (QObject | None): Qt parent.
    """

    pair_changed = Signal(object)
    waiting = Signal(bool)
    queue_changed = Signal(int, int)
    structure_changed = Signal(int, int, int)
    rejected_changed = Signal(int)
    counts_changed = Signal(int)
    worklist_changed = Signal(list)

    def __init__(
        self,
        pipeline: PairPipeline,
        sink: JudgementSink,
        target_lufs: float = -23.0,
        session_id: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.sink = sink
        self.target_lufs = target_lufs
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._pair: Pair | None = None
        self._shown_at = 0.0
        self._rated = 0
        self._waiting = False

        self._timer = QTimer(self)
        self._timer.setInterval(PUMP_MS)
        self._timer.timeout.connect(self._pump)

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Begin the session and try to show the first pair."""
        self._timer.start()
        self.advance()

    def stop(self) -> None:
        """Stop pumping the pipeline."""
        self._timer.stop()

    @property
    def pair(self) -> Pair | None:
        """
        Returns:
          Pair | None: the comparison currently on screen.
        """
        return self._pair

    @property
    def rated(self) -> int:
        """
        Returns:
          int: judgements recorded this session.
        """
        return self._rated

    # -- advancing -----------------------------------------------------------

    def advance(self) -> bool:
        """
        Show the next ready pair.

        Returns:
          bool: True if a pair was shown; False while the queue is still filling,
            in which case the timer retries.
        """
        pair = self.pipeline.next_pair()
        self._emit_queue()
        if pair is None:
            self._set_waiting(True)
            return False
        self._set_waiting(False)
        self._pair = pair
        self._shown_at = time.monotonic()
        self.pair_changed.emit(pair)
        return True

    def _pump(self) -> None:
        """Keep the queue moving, and pick up a pair if one was being waited on."""
        if self._pair is None or self._waiting:
            self.advance()
            return
        self.pipeline.pump()
        self._emit_queue()

    def _set_waiting(self, value: bool) -> None:
        """
        Args:
          value (bool): whether the session is stalled on the producer.
        """
        if value != self._waiting:
            self._waiting = value
            self.waiting.emit(value)

    def worklist(self) -> list[WorkItem]:
        """
        Returns:
          list[WorkItem]: everything queued, ready first.
        """
        return self.pipeline.worklist()

    def select(self, pair_id: str) -> bool:
        """
        Show a specific queued comparison instead of the next one.

        Args:
          pair_id (str): the entry the rater picked.

        Returns:
          bool: True if it was ready and is now on screen.
        """
        pair = self.pipeline.take(pair_id)
        if pair is None:
            self._emit_queue()
            return False
        # the pair being left goes back in the queue rather than being lost
        if self._pair is not None:
            self.pipeline.give_back(self._pair)
        self._emit_queue()
        self._set_waiting(False)
        self._pair = pair
        self._shown_at = time.monotonic()
        self.pair_changed.emit(pair)
        return True

    def request(self, tier: Tier, count: int = 1) -> int:
        """
        Queue more pairs of one tier on demand.

        Args:
          tier (Tier): which tier to produce.
          count (int): how many pairs.

        Returns:
          int: pairs queued.
        """
        queued = self.pipeline.request(tier, count)
        self._emit_queue()
        return queued

    def _emit_queue(self) -> None:
        """Publish queue depth and structure availability for the status bar."""
        self.queue_changed.emit(self.pipeline.ready, self.pipeline.inflight)
        self.worklist_changed.emit(self.pipeline.worklist())
        # Structure draws are declined while nothing is banked. Showing that
        # count is what stops a whole session going by without the tier ever
        # appearing and no way to tell why.
        # Rejections are a silent discard otherwise, which is exactly how the
        # structure tier went missing for thirty comparisons.
        self.rejected_changed.emit(self.pipeline.rejected_quiet)
        self.structure_changed.emit(
            self.pipeline.banked_structure_pairs(),
            self.pipeline.backfilling,
            self.pipeline.skipped_structure,
        )

    # -- rating --------------------------------------------------------------

    def choose(self, choice: Choice) -> None:
        """
        Record a decision and move on.

        Args:
          choice (Choice): "left", "right" or "tie". A tie is a real answer, not
            a skip -- PLAN section 7.1 asks for it to be offered and logged.
        """
        pair = self._pair
        if pair is None:
            return
        spec = pair.spec
        chosen = (
            ""
            if choice == "tie"
            else spec.left.item_id if choice == "left" else spec.right.item_id
        )
        self.sink.append(
            Judgement(
                pair_id=spec.pair_id,
                session_id=self.session_id,
                tier=spec.tier,
                question=spec.question,
                item_left=spec.left.item_id,
                item_right=spec.right.item_id,
                choice=choice,
                chosen_item_id=chosen,
                response_ms=int((time.monotonic() - self._shown_at) * 1000),
                is_repeat=spec.is_repeat,
                is_anchor=spec.is_anchor,
                lufs_target=self.target_lufs,
                was_live=pair.left.was_live or pair.right.was_live,
                ts=utc_now(),
            )
        )
        self.pipeline.mark_rated(spec)
        self._rated += 1
        self.counts_changed.emit(self._rated)
        self._pair = None
        self.advance()

    def skip(self) -> None:
        """
        Drop the current pair without logging anything.

        Distinct from a tie: a skip means the pair was unrateable (a decode
        glitch, a distraction), and pretending it was a tie would poison the
        self-agreement statistic.
        """
        if self._pair is None:
            return
        self._pair = None
        self.advance()

    def side_for(self, side: Side) -> str:
        """
        Args:
          side (Side): "left" or "right".

        Returns:
          str: the true item id behind that side. For diagnostics only; the view
            must not call this while a pair is being rated.
        """
        if self._pair is None:
            return ""
        return (
            self._pair.spec.left.item_id
            if side == "left"
            else self._pair.spec.right.item_id
        )
