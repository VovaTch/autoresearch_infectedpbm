"""
Messages between the UI process and the generation worker.

Audio crosses as int16 PCM on an ordinary multiprocessing.Queue. Shared memory
would avoid the copy, but a 90 s mono clip is 7.9 MB and pickles in well under
the time it takes to sample it -- not worth the resource_tracker unlink dance,
where a buffer created in one process and unlinked in another leaks or warns
depending on who dies first. int16 is also what QAudioSink consumes, so the
conversion is free rather than an extra step.

This module imports numpy and nothing else, so it is testable without torch,
onnxruntime or a live subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ab_harness.model.types import ClipSpec


@dataclass
class ClipRequest:
    """
    Ask the worker for one clip.

    Args:
      spec (ClipSpec): what to generate, or to decode when generator is
        "reference" or the tokens are already banked.
    """

    spec: ClipSpec


@dataclass
class Shutdown:
    """Sentinel telling the service loop to exit."""


@dataclass
class ClipResult:
    """
    One finished clip, or the reason it failed.

    Args:
      spec (ClipSpec): echo of the request.
      tokens (np.ndarray | None): (T, R) int16 codes, None on failure.
      pcm (np.ndarray | None): (N,) int16 mono, loudness-normalized, None on
        failure.
      sample_rate (int): samples per second of pcm.
      was_live (bool): True if the tokens were sampled rather than loaded.
      fill (float): share of the clip above the near-silence floor; -1.0 when
        unmeasured. See ab_harness.model.audio.fill_fraction.
      error (str): empty on success, otherwise a short description.
    """

    spec: ClipSpec
    tokens: np.ndarray | None = None
    pcm: np.ndarray | None = None
    sample_rate: int = 44100
    was_live: bool = False
    fill: float = -1.0
    error: str = ""

    @property
    def ok(self) -> bool:
        """
        Returns:
          bool: True when both tokens and audio came back.
        """
        return not self.error and self.tokens is not None and self.pcm is not None
