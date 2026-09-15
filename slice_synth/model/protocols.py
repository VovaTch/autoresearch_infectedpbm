"""
The seams.

Two interfaces are worth naming now, because both have a second implementation
already on the roadmap and neither should require touching the UI when it lands.

`RenderProducer` is what the view-model talks to. It has an in-process
implementation for the CLI and a subprocess one for the app, exactly as the
rating harness does, and a fake in the tests.

`SampleSource` is what the worker calls to turn recipes into codes. ArGenerator
already satisfies it structurally; a latent-DiT sampler satisfying the same
signature drops in behind RenderSpec.kind without the layers above knowing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, runtime_checkable

from slice_synth.model.types import Render, RenderSpec

if TYPE_CHECKING:
    # Only for the SampleSource signature. Importing torch here for real would
    # drag it into the UI process, which is the one thing the model layer is
    # supposed to guarantee it never does.
    import torch


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


@runtime_checkable
class SampleSource(Protocol):
    """A generative model over RVQ token grids."""

    def sample_batch(
        self,
        requests: Sequence[Any],
        progress: Callable[[int, int], None] | None = None,
    ) -> list["torch.Tensor"]:
        """
        Args:
          requests (Sequence[Any]): one lane per clip.
          progress (Callable[[int, int], None] | None): called with (step, total).

        Returns:
          list[torch.Tensor]: (T, R) int64 aligned codes per request, on the CPU.
        """
        ...


__all__ = ["Render", "RenderProducer", "RenderSpec", "SampleSource"]
