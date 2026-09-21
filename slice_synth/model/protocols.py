"""
The seams.

Two interfaces are worth naming now, because both have a second implementation
already on the roadmap and neither should require touching the UI when it lands.

`RenderProducer` is what the view-model talks to. It has an in-process
implementation for the CLI and a subprocess one for the app, exactly as the
rating harness does, and a fake in the tests.

`SampleSource` is what the worker calls to turn recipes into codes. It lives in
ab_harness.model.protocols (re-exported here): ArGenerator, ZFlowGenerator and
MdmGenerator all satisfy it, and the checkpoint's family picks which one the
worker builds, so nothing above the worker knows which sampler ran.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ab_harness.model.protocols import SampleSource
from slice_synth.model.types import Render, RenderSpec


@runtime_checkable
class RenderProducer(Protocol):
    """Anything that can turn RenderSpecs into playable Renders."""

    checkpoint: str

    def submit(self, specs: Sequence[RenderSpec]) -> None:
        """
        Args:
          specs (Sequence[RenderSpec]): clips to produce, as one batch.
        """
        ...

    def poll(self, timeout: float = 0.0) -> list[Any]:
        """
        Args:
          timeout (float): seconds to wait for the first message.

        Returns:
          list[Any]: whatever has arrived -- Renders, progress notices and
            errors -- since the last call. Never blocks the UI beyond `timeout`.
        """
        ...

    def switch_checkpoint(self, checkpoint: str) -> None:
        """
        Args:
          checkpoint (str): repo-relative checkpoint to sample from next.
        """
        ...

    def close(self) -> None:
        """Release the worker."""
        ...


__all__ = ["Render", "RenderProducer", "RenderSpec", "SampleSource"]
