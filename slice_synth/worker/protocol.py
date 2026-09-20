"""
Messages between the UI process and the synthesis worker.

Audio crosses as int16 PCM on an ordinary multiprocessing.Queue, for the reasons
ab_harness.worker.protocol records: a 90 s mono clip is 7.9 MB and pickles in far
less time than it takes to sample, and int16 is what QAudioSink consumes anyway.

This module imports numpy and the model layer and nothing else, so the whole
protocol is testable without torch, onnxruntime or a live subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ab_harness.model.pair_sampler import TrackInfo
from slice_synth.model.types import RenderSpec


@dataclass
class RenderRequest:
    """
    Ask the worker for a batch of clips.

    The whole batch travels as one message rather than one message per clip: the
    sampler is launch-bound, so clips sampled together are very nearly free, and
    splitting them up would let the worker start on a partial batch.

    Args:
      batch_id (str): id echoed on every progress notice from this batch.
      specs (list[RenderSpec]): the clips to produce.
    """

    batch_id: str
    specs: list[RenderSpec] = field(default_factory=list)


@dataclass
class SwitchCheckpoint:
    """
    Ask the worker to sample from a different model from here on.

    Args:
      checkpoint (str): repo-relative checkpoint path.
    """

    checkpoint: str


@dataclass
class Cancel:
    """
    Drop everything queued but not started.

    A 90 s batch is minutes of sampling, and the answer to "wrong settings" has
    to be better than waiting it out. The batch already in the sampler still
    finishes -- interrupting mid-grid would leave the KV cache in a state nothing
    else knows how to recover.
    """


@dataclass
class Shutdown:
    """Sentinel telling the service loop to exit."""


@dataclass
class WorkerReady:
    """
    Sent once the model is loaded, and again after every checkpoint switch.

    This is how the corpus reaches a UI process that holds no torch: the track
    list, their lengths and their style-window bounds all live in the token
    cache, which only the worker can read.

    Args:
      checkpoint (str): the model now loaded. On failure this is the one that was
        already there, since a failed switch keeps the old model rather than
        leaving the worker serving nothing.
      tracks (list[TrackInfo]): the corpus, torch-free.
      meta (dict[str, Any]): tokenizer meta -- fps, hop, sample rate, codebook
        size, RVQ depth.
      window_frames (int): the sliding window actually in use, after clamping to
        the checkpoint's crop_frames.
      error (str): empty on success.
      style_ar_checkpoint (str): the style model an "ar" walk will load, or the
        unresolved "auto" when no saved_style_ar_* run exists yet.
    """

    checkpoint: str
    tracks: list[TrackInfo] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    window_frames: int = 0
    error: str = ""
    style_ar_checkpoint: str = ""

    @property
    def ok(self) -> bool:
        """
        Returns:
          bool: True when the requested model is the one now loaded.
        """
        return not self.error


@dataclass
class RenderProgress:
    """
    How far the current batch has got.

    Args:
      batch_id (str): the batch being sampled.
      step (int): positions produced.
      total (int): positions in the longest lane.
    """

    batch_id: str
    step: int
    total: int

    @property
    def fraction(self) -> float:
        """
        Returns:
          float: progress in [0, 1]; 0.0 when the total is unknown.
        """
        return min(1.0, self.step / self.total) if self.total > 0 else 0.0


@dataclass
class RenderResult:
    """
    One finished clip, or the reason it failed.

    Args:
      spec (RenderSpec): echo of the request.
      tokens (np.ndarray | None): (T, R) int16 codes, None on failure.
      pcm (np.ndarray | None): (N,) int16 mono, loudness-normalized.
      sample_rate (int): samples per second of pcm.
      style_used (np.ndarray | None): (style_dim,) float32 descriptor actually
        resolved. For a random or jittered recipe this is the only record of
        what was heard.
      fill (float): share of the clip above the near-silence floor; -1.0 when
        unmeasured. See ab_harness.model.audio.fill_fraction.
      error (str): empty on success, otherwise a short description.
    """

    spec: RenderSpec
    tokens: np.ndarray | None = None
    pcm: np.ndarray | None = None
    sample_rate: int = 44100
    style_used: np.ndarray | None = None
    fill: float = -1.0
    error: str = ""

    @property
    def ok(self) -> bool:
        """
        Returns:
          bool: True when both tokens and audio came back.
        """
        return not self.error and self.tokens is not None and self.pcm is not None


__all__ = [
    "Cancel",
    "RenderProgress",
    "RenderRequest",
    "RenderResult",
    "Shutdown",
    "SwitchCheckpoint",
    "WorkerReady",
]
