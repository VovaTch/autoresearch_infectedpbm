"""
Producer front-ends: how the UI and the CLI ask for renders.

Two implementations. The in-process one is what slice_synth.render uses and what
makes a failure readable in a traceback; the subprocess one is what the app uses,
so a CUDA OOM or a driver hiccup kills a worker rather than the window.

Unlike the rating harness, `poll` hands back raw protocol messages rather than
finished clips only. The synthesizer has a progress bar and a corpus that arrives
after startup, and both of those are messages the view-model has to see.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import TYPE_CHECKING, Any, Sequence

from slice_synth.config import SynthConfig
from slice_synth.model.types import RenderSpec, digest
from slice_synth.worker.protocol import (
    Cancel,
    RenderProgress,
    RenderRequest,
    RenderResult,
    Shutdown,
    SwitchCheckpoint,
    WorkerReady,
)

if TYPE_CHECKING:
    from slice_synth.worker.service import SynthService


def _child_entry(
    cfg: SynthConfig, requests: "mp.Queue[Any]", results: "mp.Queue[Any]"
) -> None:
    """
    Spawn target that imports the service only inside the child.

    Importing slice_synth.worker.service at module scope would drag torch,
    lightning and onnxruntime into the UI process, which is exactly what the
    process split exists to avoid. spawn pickles this target by module path, so
    the deferred import costs nothing.

    Args:
      cfg (SynthConfig): synthesizer config.
      requests (mp.Queue[Any]): inbound messages.
      results (mp.Queue[Any]): outbound messages.
    """
    from slice_synth.worker.service import run_service

    run_service(cfg, requests, results)


def batch_id(specs: Sequence[RenderSpec]) -> str:
    """
    Args:
      specs (Sequence[RenderSpec]): the clips going out together.

    Returns:
      str: an id for the batch, so a progress notice can be matched to the
        request that is still in flight and ignored when it is stale.
    """
    return digest([s.item_id for s in specs], "b")


class InProcessRenderProducer:
    """
    Synchronous producer that runs the service in the calling process.

    Args:
      service (SynthService): the loaded service, built by the caller so this
        module never imports torch itself.
    """

    def __init__(self, service: "SynthService") -> None:
        self.service = service
        self._out: list[Any] = []

    @property
    def checkpoint(self) -> str:
        """
        Returns:
          str: the checkpoint currently being sampled from.
        """
        return self.service.checkpoint

    def start(self) -> None:
        """Load the model and publish the corpus."""
        self.service.load()
        self._out.append(self.service.ready())

    def submit(self, specs: Sequence[RenderSpec]) -> None:
        """
        Render immediately, in one batch; poll then returns the results.

        Args:
          specs (Sequence[RenderSpec]): clips to produce.
        """
        if not specs:
            return
        batch = batch_id(specs)

        def report(step: int, total: int) -> None:
            self._out.append(RenderProgress(batch_id=batch, step=step, total=total))

        self._out.extend(self.service.render_many(specs, report))

    def poll(self, timeout: float = 0.0) -> list[Any]:
        """
        Args:
          timeout (float): ignored; production already happened in submit.

        Returns:
          list[Any]: everything produced since the last poll.
        """
        out, self._out = self._out, []
        return out

    def cancel(self) -> None:
        """No-op: work is synchronous, so nothing is ever queued to drop."""

    def switch_checkpoint(self, checkpoint: str) -> None:
        """
        Args:
          checkpoint (str): repo-relative checkpoint to sample from next.
        """
        try:
            self.service.switch_checkpoint(checkpoint)
            self._out.append(self.service.ready())
        except Exception as exc:  # noqa: BLE001 - mirrors the subprocess path
            ready = self.service.ready()
            ready.error = f"{type(exc).__name__}: {exc}"
            self._out.append(ready)

    def close(self) -> None:
        """Release the service's model, decoder and encoder."""
        self.service.close()


class ProcessRenderProducer:
    """
    Asynchronous producer backed by a child process.

    The child uses the spawn start method: CUDA contexts do not survive a fork,
    and the UI process must stay free of torch anyway.

    Args:
      cfg (SynthConfig): synthesizer config, pickled to the child.
    """

    def __init__(self, cfg: SynthConfig) -> None:
        self.cfg = cfg
        ctx = mp.get_context("spawn")
        self._requests: "mp.Queue[Any]" = ctx.Queue()
        self._results: "mp.Queue[Any]" = ctx.Queue()
        self._process = ctx.Process(
            target=_child_entry, args=(cfg, self._requests, self._results), daemon=True
        )
        self.checkpoint = cfg.generator.checkpoint
        self._pending = 0

    def start(self) -> None:
        """Launch the worker. Loading the checkpoint takes a few seconds."""
        self._process.start()

    @property
    def alive(self) -> bool:
        """
        Returns:
          bool: True while the worker process is running.
        """
        return self._process.is_alive()

    @property
    def pending(self) -> int:
        """
        Returns:
          int: submitted clips not yet collected.
        """
        return self._pending

    def submit(self, specs: Sequence[RenderSpec]) -> None:
        """
        Queue a batch for the worker.

        The whole batch goes as one message: the sampler is launch-bound, so
        clips sampled together are nearly free, and splitting them would let the
        worker start on part of the grid.

        Args:
          specs (Sequence[RenderSpec]): clips to produce.
        """
        if not specs:
            return
        self._requests.put(RenderRequest(batch_id=batch_id(specs), specs=list(specs)))
        self._pending += len(specs)

    def cancel(self) -> None:
        """
        Drop everything queued but not started.

        The batch already in the sampler still finishes: interrupting mid-grid
        would leave the KV cache in a state nothing else knows how to recover.
        """
        self._requests.put(Cancel())

    def poll(self, timeout: float = 0.0) -> list[Any]:
        """
        Collect worker messages without blocking the UI.

        Args:
          timeout (float): seconds to wait for the first message.

        Returns:
          list[Any]: WorkerReady, RenderProgress and RenderResult messages,
            possibly empty.
        """
        out: list[Any] = []
        first = True
        while True:
            try:
                message = (
                    self._results.get(timeout=timeout)
                    if first and timeout > 0
                    else self._results.get_nowait()
                )
            except queue.Empty:
                return out
            first = False
            if isinstance(message, WorkerReady):
                self.checkpoint = message.checkpoint
            elif isinstance(message, RenderResult):
                self._pending = max(0, self._pending - 1)
            out.append(message)

    def switch_checkpoint(self, checkpoint: str) -> None:
        """
        Ask the worker to sample from another model.

        Queued behind whatever is already in flight, so clips drawn against the
        old model still come back tagged with it.

        Args:
          checkpoint (str): repo-relative checkpoint path.
        """
        self._requests.put(SwitchCheckpoint(checkpoint))

    def close(self) -> None:
        """Ask the worker to stop, then join it with a short grace period."""
        if not self._process.is_alive():
            return
        self._requests.put(Shutdown())
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()


__all__ = ["InProcessRenderProducer", "ProcessRenderProducer", "batch_id"]
