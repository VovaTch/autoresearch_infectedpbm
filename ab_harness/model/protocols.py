"""
Collaborator interfaces for the harness.

Every consumer depends on one of these rather than on a concrete class, so tests
inject fakes and never touch a GPU, an audio device or the filesystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from ab_harness.model.types import Clip, ClipSpec, Judgement, Pair, Tier

if TYPE_CHECKING:
    # Only for the SampleSource signature. Importing torch here for real would
    # drag it into the UI process, which the process split exists to prevent.
    import torch


@runtime_checkable
class SampleSource(Protocol):
    """
    A generative model over RVQ token grids, whatever its sampling scheme.

    ArGenerator, ZFlowGenerator and MdmGenerator all satisfy this; the worker
    services pick one by the checkpoint's family (ab_harness.checkpoints) and
    the layers above never learn which. The four attributes are the geometry
    the services need for request budgeting and prompt handling.
    """

    depth: int
    window_frames: int
    num_tracks: int
    pad_id: int

    def sample_batch(
        self,
        requests: Sequence[Any],
        progress: Callable[[int, int], None] | None = None,
    ) -> list["torch.Tensor"]:
        """
        Args:
          requests (Sequence[Any]): one SampleRequest per clip.
          progress (Callable[[int, int], None] | None): called with (step, total).

        Returns:
          list[torch.Tensor]: (T, R) int64 aligned codes per request, on the
            CPU, prompt frames included.
        """
        ...


class ClipProducer(Protocol):
    """Turns ClipSpecs into decoded Clips, asynchronously."""

    def submit(self, specs: Sequence[ClipSpec]) -> None:
        """
        Queue clips for production.

        Args:
          specs (Sequence[ClipSpec]): clips to generate or decode.
        """
        ...

    def poll(self, timeout: float = 0.0) -> list[Clip]:
        """
        Collect whatever has finished.

        Args:
          timeout (float): seconds to wait for the first result.

        Returns:
          list[Clip]: finished clips, possibly empty. Never raises on a
            producer-side failure; failed specs are simply never returned.
        """
        ...

    def close(self) -> None:
        """Shut the producer down and release its resources."""
        ...


class CheckpointSwitchable(Protocol):
    """
    A producer whose model can be changed without restarting the session.

    Kept apart from ClipProducer so the fakes in tests -- and bake_ab_bank.py,
    which loads one model and exits -- stay complete without it.
    """

    @property
    def checkpoint(self) -> str:
        """
        Returns:
          str: the checkpoint currently being sampled from.
        """
        ...

    def switch_checkpoint(self, checkpoint: str) -> None:
        """
        Args:
          checkpoint (str): repo-relative checkpoint to sample from next.
        """
        ...

    def take_checkpoint_change(self) -> object | None:
        """
        Returns:
          object | None: a CheckpointChanged once the switch has resolved, then
            None. Typed loosely so this module stays free of worker imports.
        """
        ...


class TokenStore(Protocol):
    """Persistent store of generated token streams and their specs."""

    def add(self, spec: ClipSpec, tokens: np.ndarray) -> None:
        """
        Args:
          spec (ClipSpec): the clip's recipe.
          tokens (np.ndarray): (T, R) int16 codes.
        """
        ...

    def has(self, item_id: str) -> bool:
        """
        Args:
          item_id (str): the clip id.

        Returns:
          bool: True when the store already holds this clip's tokens.
        """
        ...

    def tokens(self, item_id: str) -> np.ndarray:
        """
        Args:
          item_id (str): the clip id.

        Returns:
          np.ndarray: (T, R) int16 codes.
        """
        ...

    def specs(self, tier: Tier | None = None) -> list[ClipSpec]:
        """
        Args:
          tier (Tier | None): restrict to one tier, or None for all.

        Returns:
          list[ClipSpec]: every stored spec, in insertion order.
        """
        ...

    def refresh(self) -> int:
        """
        Pick up items another process wrote since this store was built.

        Returns:
          int: number of newly indexed specs.
        """
        ...


class PairSource(Protocol):
    """Supplies ready-to-play pairs to the session view-model."""

    def next_pair(self) -> Pair | None:
        """
        Returns:
          Pair | None: the next comparison, or None if nothing is ready yet.
        """
        ...


class JudgementSink(Protocol):
    """Append-only destination for rater decisions."""

    def append(self, judgement: Judgement) -> None:
        """
        Args:
          judgement (Judgement): the decision to persist. Flushed immediately.
        """
        ...
