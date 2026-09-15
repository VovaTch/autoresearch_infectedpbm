"""
The token strip: what the model actually chose.

Everything else on screen is downstream of the tokenizer, so when a render sounds
wrong this is the only view that says whether the transformer or the decoder is
responsible. Three signatures are visible at a glance and hard to hear apart:

  collapse   one colour dominates a level -- a handful of codes, low entropy
  looping    a pattern repeats at a fixed pitch across the strip
  seam       the boundary where the forced prompt stops and the model starts

The palette is a hash of the code id, not a ramp over it: codebook indices have
no ordering the decoder respects, so a gradient would invent structure. Adjacent
ids get unrelated colours, which is exactly what makes a repeat look like a
repeat.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from slice_synth.model.tokens import colour_lut, token_image
from slice_synth.view.strip import ScrubStrip

ROW_GAP = 1


class TokenView(ScrubStrip):
    """
    One horizontal band per RVQ level, coarsest at the top.

    Args:
      height (int): widget height in pixels.
      num_tokens (int): codebook size, for the palette.
      parent (QWidget | None): Qt parent.
    """

    def __init__(
        self, height: int = 54, num_tokens: int = 2048, parent: QWidget | None = None
    ) -> None:
        super().__init__(height, parent)
        self._tokens: np.ndarray = np.zeros((0, 0), dtype=np.int64)
        self._lut = colour_lut(num_tokens)
        self.set_placeholder("tokens")

    def set_num_tokens(self, num_tokens: int) -> None:
        """
        Args:
          num_tokens (int): codebook size of the loaded checkpoint.
        """
        if num_tokens != self._lut.shape[0]:
            self._lut = colour_lut(num_tokens)
            self.rebuild()

    def set_tokens(self, tokens: np.ndarray) -> None:
        """
        Args:
          tokens (np.ndarray): (T, R) integer codes.
        """
        self._tokens = tokens
        self._position = 0.0
        self.rebuild()

    def rebuild(self) -> None:
        """Rebuild the per-level image, then repaint."""
        if self._tokens.size == 0:
            self.set_image(None)
            return
        rows = token_image(self._tokens, self._lut)
        levels, frames, _ = rows.shape
        # A thin dark rule between levels; without it three bands of similar
        # density read as one noisy block.
        band = max(1, (self.height() - ROW_GAP * (levels - 1)) // max(1, levels))
        stacked = np.zeros(
            (levels * band + ROW_GAP * (levels - 1), frames, 3), dtype=np.uint8
        )
        for level in range(levels):
            top = level * (band + ROW_GAP)
            stacked[top : top + band] = rows[level][None]
        self.set_image(stacked)


__all__ = ["TokenView"]
