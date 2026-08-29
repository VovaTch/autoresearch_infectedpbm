"""
Collaborator interfaces for the harness.

Every consumer depends on one of these rather than on a concrete class, so tests
inject fakes and never touch a GPU, an audio device or the filesystem.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ab_harness.model.types import Clip, ClipSpec, Judgement, Pair, Tier


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
