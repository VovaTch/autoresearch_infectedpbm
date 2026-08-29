"""
Producer front-ends: how the rest of the harness asks for clips.

Two implementations of the ClipProducer protocol. The in-process one is what
bake_ab_bank.py uses and what makes a failure easy to read in a traceback; the
subprocess one is what the app uses, so a CUDA OOM or a driver hiccup kills a
worker rather than the rating session.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import TYPE_CHECKING, Any, Sequence

from ab_harness.config import AbConfig
from ab_harness.model.types import Clip, ClipSpec
from ab_harness.worker.protocol import ClipRequest, ClipResult, Shutdown

if TYPE_CHECKING:
    from ab_harness.worker.service import GenerationService


def _child_entry(
    cfg: AbConfig, requests: "mp.Queue[Any]", results: "mp.Queue[Any]"
) -> None:
    """
    Spawn target that imports the service only inside the child.

    Importing ab_harness.worker.service at module scope would drag torch,
    lightning and onnxruntime into the UI process, which is exactly what the
    process split exists to avoid. spawn pickles this target by module path, so
    the deferred import costs nothing.

    Args:
      cfg (AbConfig): harness config.
      requests (mp.Queue[Any]): inbound messages.
      results (mp.Queue[Any]): outbound results.
    """
    from ab_harness.worker.service import run_service

    run_service(cfg, requests, results)


def _to_clip(result: ClipResult) -> Clip | None:
    """
    Args:
      result (ClipResult): a worker reply.

    Returns:
      Clip | None: the playable clip, or None if the request failed.
    """
    if not result.ok:
        return None
    assert result.tokens is not None and result.pcm is not None
    return Clip(
        spec=result.spec,
        tokens=result.tokens,
        pcm=result.pcm,
        was_live=result.was_live,
        fill=result.fill,
    )


class InProcessClipProducer:
    """
    Synchronous producer that runs the service in the calling process.

    Args:
      service (GenerationService): the loaded service, built by the caller so
        this module never imports torch itself.
    """

    def __init__(self, service: "GenerationService") -> None:
        self.service = service
        self._done: list[Clip] = []
        self.errors: list[str] = []

    def submit(self, specs: Sequence[ClipSpec]) -> None:
        """
        Produce clips immediately, in one batch; poll then returns them.

        Args:
          specs (Sequence[ClipSpec]): clips to produce.
        """
        for result in self.service.produce_many(specs):
            clip = _to_clip(result)
            if clip is None:
                self.errors.append(result.error)
            else:
                self._done.append(clip)

    def poll(self, timeout: float = 0.0) -> list[Clip]:
        """
        Args:
          timeout (float): ignored; production already happened in submit.

        Returns:
          list[Clip]: everything produced since the last poll.
        """
        done, self._done = self._done, []
        return done

    def close(self) -> None:
        """Release the service's model and decoder."""
        self.service.close()


class ProcessClipProducer:
    """
    Asynchronous producer backed by a child process.

    The child uses the spawn start method: CUDA contexts do not survive a fork,
    and the UI process must stay free of torch anyway.

    Args:
      cfg (AbConfig): harness config, pickled to the child.
    """

    def __init__(self, cfg: AbConfig) -> None:
        self.cfg = cfg
        ctx = mp.get_context("spawn")
        self._requests: "mp.Queue[Any]" = ctx.Queue()
        self._results: "mp.Queue[Any]" = ctx.Queue()
        self._process = ctx.Process(
            target=_child_entry, args=(cfg, self._requests, self._results), daemon=True
        )
        self._pending = 0
        self.errors: list[str] = []

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

    def submit(self, specs: Sequence[ClipSpec]) -> None:
        """
        Args:
          specs (Sequence[ClipSpec]): clips to queue for the worker.
        """
        for spec in specs:
            self._requests.put(ClipRequest(spec))
            self._pending += 1

    def poll(self, timeout: float = 0.0) -> list[Clip]:
        """
        Collect finished clips without blocking the rating loop.

        A worker-side failure is recorded and dropped rather than raised: the
        pipeline's job is to keep serving pairs, and a spec that cannot be
        produced is simply one the rater never sees.

        Args:
          timeout (float): seconds to wait for the first result.

        Returns:
          list[Clip]: finished clips, possibly empty.
        """
        out: list[Clip] = []
        first = True
        while True:
            try:
                result = (
                    self._results.get(timeout=timeout)
                    if first and timeout > 0
                    else self._results.get_nowait()
                )
            except queue.Empty:
                return out
            first = False
            self._pending = max(0, self._pending - 1)
            clip = _to_clip(result)
            if clip is None:
                self.errors.append(result.error)
            else:
                out.append(clip)

    def close(self) -> None:
        """Ask the worker to stop, then join it with a short grace period."""
        if not self._process.is_alive():
            return
        self._requests.put(Shutdown())
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
