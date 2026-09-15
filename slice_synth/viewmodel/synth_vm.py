"""
Session state for the synthesizer.

Holds the variant list, pumps the worker, and owns the one thing the view is not
allowed to know: which render is selected and what it takes to save it.

Pumping on a QTimer rather than a thread is the same choice ab_harness made and
for the same reason -- everything expensive already happens in another process,
so the UI thread only ever moves finished buffers around. There is no lock in
this file because there is nothing to lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PySide6.QtCore import QObject, QTimer, Signal

from slice_synth.model.library import SavedRender, save_render
from slice_synth.model.types import Render, RenderSpec
from slice_synth.worker.protocol import (
    RenderProgress,
    RenderResult,
    WorkerReady,
)

PUMP_MS = 200


@dataclass
class Variant:
    """
    One row of the variant list.

    Args:
      spec (RenderSpec): what was asked for.
      render (Render | None): the finished clip, once it arrives.
      error (str): empty unless the worker refused this one.
      saved (SavedRender | None): where it was written, if it was kept.
    """

    spec: RenderSpec
    render: Render | None = None
    error: str = ""
    saved: SavedRender | None = None

    @property
    def item_id(self) -> str:
        """
        Returns:
          str: the spec's content-addressed id.
        """
        return self.spec.item_id

    @property
    def status(self) -> str:
        """
        Returns:
          str: "generating", "failed" or "ready".
        """
        if self.error:
            return "failed"
        return "ready" if self.render is not None else "generating"


@dataclass
class _Batch:
    """
    Args:
      batch_id (str): id the worker echoes on progress notices.
      specs (list[RenderSpec]): what went out in it.
    """

    batch_id: str
    specs: list[RenderSpec] = field(default_factory=list)


class SynthViewModel(QObject):
    """
    The synthesizer session.

    Args:
      producer (Any): a RenderProducer -- the subprocess one in the app, the
        in-process one in the CLI, a fake in the tests.
      output_root (Path): where saved renders are written.
      parent (QObject | None): Qt parent.
    """

    corpus_changed = Signal(object)
    variants_changed = Signal(list)
    selected_changed = Signal(object)
    progress_changed = Signal(float, int, int)
    busy_changed = Signal(bool)
    message = Signal(str)
    checkpoint_changed = Signal(str, str)

    def __init__(
        self, producer: Any, output_root: Path, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.producer = producer
        self.output_root = Path(output_root)
        self.ready: WorkerReady | None = None
        self._variants: list[Variant] = []
        self._selected: str = ""
        self._inflight = 0
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setInterval(PUMP_MS)
        self._timer.timeout.connect(self._pump)

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Begin pumping the worker."""
        self._timer.start()

    def stop(self) -> None:
        """Stop pumping. The producer is closed by the caller."""
        self._timer.stop()

    # -- state ---------------------------------------------------------------

    @property
    def variants(self) -> list[Variant]:
        """
        Returns:
          list[Variant]: the current list, newest batch last.
        """
        return list(self._variants)

    @property
    def selected(self) -> Variant | None:
        """
        Returns:
          Variant | None: the row on screen, if any.
        """
        return next((v for v in self._variants if v.item_id == self._selected), None)

    @property
    def busy(self) -> bool:
        """
        Returns:
          bool: True while the worker still owes results.
        """
        return self._busy

    @property
    def loaded(self) -> bool:
        """
        Returns:
          bool: True once the worker has published a corpus.
        """
        return self.ready is not None and self.ready.ok

    # -- commands ------------------------------------------------------------

    def generate(self, specs: Sequence[RenderSpec]) -> int:
        """
        Queue a batch and add its rows to the list immediately.

        The rows appear before the audio does, on purpose: a cross product of
        sixteen variants takes a while, and a list that fills in one row at a
        time is the only honest progress display for it.

        Args:
          specs (Sequence[RenderSpec]): the clips to produce.

        Returns:
          int: how many were queued, after dropping ones already in the list.
        """
        if not specs:
            return 0
        known = {v.item_id for v in self._variants}
        fresh = [s for s in specs if s.item_id not in known]
        if not fresh:
            self.message.emit("every variant in that selection is already listed")
            return 0
        self._variants += [Variant(spec=s) for s in fresh]
        self._inflight += len(fresh)
        self.producer.submit(fresh)
        self._set_busy(True)
        self.variants_changed.emit(self.variants)
        return len(fresh)

    def cancel(self) -> None:
        """
        Drop what has not started, and forget the rows that will never arrive.

        The pass already in the sampler still finishes -- interrupting mid-grid
        would leave the KV cache unrecoverable -- so its rows stay.
        """
        self.producer.cancel()
        stale = [v for v in self._variants if v.status == "generating"]
        if not stale:
            return
        keep = {v.item_id for v in self._variants if v.status != "generating"}
        self._variants = [v for v in self._variants if v.item_id in keep]
        self._inflight = 0
        self._set_busy(False)
        self.progress_changed.emit(0.0, 0, 0)
        self.message.emit(f"cancelled {len(stale)} pending")
        self.variants_changed.emit(self.variants)
        if self._selected not in keep:
            self.select("")

    def clear(self) -> None:
        """Empty the variant list, keeping anything still generating."""
        self._variants = [v for v in self._variants if v.status == "generating"]
        self.variants_changed.emit(self.variants)
        self.select("")

    def select(self, item_id: str) -> bool:
        """
        Args:
          item_id (str): the row to show; "" clears the selection.

        Returns:
          bool: True when a finished render is now on screen.
        """
        self._selected = item_id
        variant = self.selected
        render = variant.render if variant is not None else None
        self.selected_changed.emit(render)
        return render is not None

    def save_selected(self) -> SavedRender | None:
        """
        Write the selected render's wav, tokens and recipe.

        Returns:
          SavedRender | None: the paths written, or None when there is nothing
            finished selected.
        """
        variant = self.selected
        if variant is None or variant.render is None:
            self.message.emit("nothing to save")
            return None
        saved = save_render(variant.render, self.output_root)
        variant.saved = saved
        self.message.emit(f"saved {saved.wav.name}")
        self.variants_changed.emit(self.variants)
        return saved

    def switch_checkpoint(self, checkpoint: str) -> None:
        """
        Args:
          checkpoint (str): repo-relative checkpoint to sample from next.
        """
        self.producer.switch_checkpoint(checkpoint)
        self.message.emit(f"loading {checkpoint}...")

    # -- pump ----------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        """
        Args:
          busy (bool): whether the worker still owes results.
        """
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def _pump(self) -> None:
        """Collect whatever the worker has produced since the last tick."""
        arrived = False
        for msg in self.producer.poll():
            if isinstance(msg, WorkerReady):
                self._on_ready(msg)
            elif isinstance(msg, RenderProgress):
                self.progress_changed.emit(msg.fraction, msg.step, msg.total)
            elif isinstance(msg, RenderResult):
                arrived |= self._on_result(msg)
        if arrived:
            self.variants_changed.emit(self.variants)
            if self._inflight <= 0:
                self._set_busy(False)
                self.progress_changed.emit(1.0, 1, 1)

    def _on_ready(self, msg: WorkerReady) -> None:
        """
        Args:
          msg (WorkerReady): the worker's corpus publication.
        """
        first = self.ready is None
        if msg.ok:
            self.ready = msg
            self.corpus_changed.emit(msg)
        self.checkpoint_changed.emit(msg.checkpoint, msg.error)
        if msg.error:
            self.message.emit(f"checkpoint load failed: {msg.error}")
        elif not first:
            self.message.emit(f"now sampling from {msg.checkpoint}")

    def _on_result(self, msg: RenderResult) -> bool:
        """
        Args:
          msg (RenderResult): one finished or failed clip.

        Returns:
          bool: True when the list changed.
        """
        self._inflight = max(0, self._inflight - 1)
        variant = next(
            (v for v in self._variants if v.item_id == msg.spec.item_id), None
        )
        if variant is None:
            # A result for a row that was cancelled away; nothing owns it now.
            return False
        if not msg.ok:
            variant.error = msg.error or "worker returned nothing"
            self.message.emit(f"{msg.spec.label()}: {variant.error}")
        else:
            assert msg.tokens is not None and msg.pcm is not None
            variant.render = Render(
                spec=msg.spec,
                tokens=msg.tokens,
                pcm=msg.pcm,
                sample_rate=msg.sample_rate,
                style_used=msg.style_used,
                fill=msg.fill,
            )
            # Land on the first thing that finishes, so a batch is audible
            # without hunting for the row that is ready. The reference rows
            # complete before any sampling starts, so "whatever is selected" is
            # usually a row that will not be playable for another minute.
            selected = self.selected
            if selected is None or selected.render is None:
                self.select(variant.item_id)
            elif selected.item_id == variant.item_id:
                self.selected_changed.emit(variant.render)
        return True


__all__ = ["PUMP_MS", "SynthViewModel", "Variant"]
