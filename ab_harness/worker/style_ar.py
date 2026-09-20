"""
The style model behind the "ar" style walk, wrapped for both workers.

train_style_ar.py's model is ~1 M parameters and walks at most a few hundred
positions, so it runs on the CPU: sampling is then reproducible from a CPU
torch.Generator regardless of which GPU the token model holds, and it never
competes with the KV cache for VRAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from ab_harness.config import REPO
from train_style_ar import StyleArModule, load_style_ar, sample_sequence


class StyleWalker(Protocol):
    """Anything that continues a descriptor sequence, for walk_from_window."""

    def sample(
        self,
        prefix: torch.Tensor | None,
        track_idx: int | None,
        steps: int,
        temperature: float,
        cfg: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Args:
          prefix (torch.Tensor | None): (P, C) real descriptors to continue from.
          track_idx (int | None): id row; None means null.
          steps (int): descriptors to sample.
          temperature (float): sampling temperature.
          cfg (float): guidance strength on the id.
          generator (torch.Generator): CPU RNG.

        Returns:
          torch.Tensor: (steps, C) unit-norm descriptors.
        """
        ...


class StyleArSampler:
    """
    One loaded style-AR checkpoint.

    Args:
      checkpoint (str): repo-relative checkpoint path.

    Raises:
      FileNotFoundError: when the checkpoint does not exist.
    """

    def __init__(self, checkpoint: str) -> None:
        path = REPO / checkpoint
        if not path.exists():
            raise FileNotFoundError(f"no style-AR checkpoint at {path}")
        self.checkpoint = checkpoint
        self._module: StyleArModule = load_style_ar(Path(path))
        self._held_out = set(self._module.held_out)
        self._warned: set[int] = set()

    @property
    def num_tracks(self) -> int:
        """
        Returns:
          int: id rows the model was built with.
        """
        return self._module.model.num_tracks

    def _usable_id(self, track_idx: int | None) -> int | None:
        """
        Args:
          track_idx (int | None): requested id.

        Returns:
          int | None: the id, or None when its embedding row was never trained
            (a held-out track, or a track this model has no row for).
        """
        if track_idx is None:
            return None
        untrained = track_idx in self._held_out or track_idx >= self.num_tracks
        if untrained and track_idx not in self._warned:
            self._warned.add(track_idx)
            print(
                f"  WARN style-AR {self.checkpoint} has no trained id for track "
                f"{track_idx}; walking with the null id"
            )
        return None if untrained else track_idx

    def sample(
        self,
        prefix: torch.Tensor | None,
        track_idx: int | None,
        steps: int,
        temperature: float,
        cfg: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Args:
          prefix (torch.Tensor | None): (P, C) real descriptors to continue from.
          track_idx (int | None): id row; None or an untrained id means null.
          steps (int): descriptors to sample.
          temperature (float): sampling temperature.
          cfg (float): guidance strength on the id.
          generator (torch.Generator): CPU RNG seeded by the recipe.

        Returns:
          torch.Tensor: (steps, C) unit-norm descriptors.
        """
        return sample_sequence(
            self._module.model,
            self._module.stats,
            prefix,
            self._usable_id(track_idx),
            steps,
            temperature,
            cfg,
            generator,
        )


def walk_from_window(
    sampler: StyleWalker,
    style: torch.Tensor,
    window: int,
    track_idx: int | None,
    segments: int,
    prefix: int,
    temperature: float,
    cfg: float,
    seed: int,
) -> torch.Tensor:
    """
    Continue a real style window with the style model, one row per segment.

    Args:
      sampler (StyleWalker): the loaded style model.
      style (torch.Tensor): (W, D) the track's precomputed unit descriptors.
      window (int): index of the window heard in segment 0.
      track_idx (int | None): id row for the walk; None means the null id.
      segments (int): descriptors the clip needs, the real window included.
      prefix (int): real windows ending at `window` fed to the model; 0 feeds
        the window alone.
      temperature (float): style-model sampling temperature.
      cfg (float): style-model guidance on the id.
      seed (int): seeds the CPU draw.

    Returns:
      torch.Tensor: (segments, D) unit descriptors; row 0 is the real window.
    """
    window = min(max(window, 0), style.shape[0] - 1)
    first = style[window][None].float()
    if segments <= 1:
        return first
    context = style[max(0, window - max(prefix, 1) + 1) : window + 1].float()
    rows = sampler.sample(
        context,
        track_idx,
        segments - 1,
        temperature,
        cfg,
        torch.Generator().manual_seed(int(seed)),
    )
    return torch.cat([first, rows])


__all__ = ["StyleArSampler", "StyleWalker", "walk_from_window"]
