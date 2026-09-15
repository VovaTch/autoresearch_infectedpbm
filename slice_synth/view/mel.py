"""
Mel spectrogram strip.

The waveform says how loud, and almost nothing else. Two failures this model
actually has are invisible in an envelope and obvious here: a generation that
keeps its level but loses the top two octaves, and one that settles into a
smeared, band-limited wash. Both read as a normal-looking waveform.

Resolution follows the widget width (spectrum.plan_geometry), so the cost is flat
in clip length -- a 90 s render redraws as fast as a 10 s one.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from slice_synth.model.spectrum import mel_spectrogram
from slice_synth.view.strip import ScrubStrip

# Dark blue through orange to white: monotone in lightness, so a band that reads
# as brighter really is louder. A rainbow map would not have that property.
STOPS = np.array(
    [
        [8, 10, 30],
        [40, 25, 100],
        [120, 40, 120],
        [200, 70, 80],
        [245, 140, 45],
        [255, 220, 130],
        [255, 255, 245],
    ],
    dtype=np.float32,
)


def heat_lut(levels: int = 256) -> np.ndarray:
    """
    Args:
      levels (int): entries in the table.

    Returns:
      np.ndarray: (levels, 3) uint8 colour ramp, dark at 0 and bright at 1.
    """
    position = np.linspace(0.0, STOPS.shape[0] - 1, levels)
    low = np.clip(np.floor(position).astype(np.int64), 0, STOPS.shape[0] - 1)
    high = np.clip(low + 1, 0, STOPS.shape[0] - 1)
    frac = (position - low)[:, None]
    return (STOPS[low] * (1.0 - frac) + STOPS[high] * frac).astype(np.uint8)


class MelView(ScrubStrip):
    """
    Mel spectrogram over the same time axis as the waveform.

    Args:
      height (int): minimum height in pixels.
      columns (int): time resolution cap; the real count follows the width.
      parent (QWidget | None): Qt parent.
      expanding (bool): grow into whatever vertical space is going spare.
    """

    def __init__(
        self,
        height: int = 96,
        columns: int = 1024,
        parent: QWidget | None = None,
        expanding: bool = True,
    ) -> None:
        super().__init__(height, parent, expanding)
        self._pcm: np.ndarray = np.zeros(0, dtype=np.int16)
        self._sample_rate = 44100
        self._columns = columns
        self._lut = heat_lut()
        self.set_placeholder("mel")

    def set_pcm(self, pcm: np.ndarray, sample_rate: int = 44100) -> None:
        """
        Args:
          pcm (np.ndarray): (N,) int16 samples.
          sample_rate (int): samples per second.
        """
        self._pcm = pcm
        self._sample_rate = sample_rate
        self._position = 0.0
        self.rebuild()

    def rebuild(self) -> None:
        """Recompute the spectrogram at the current width, then repaint."""
        if self._pcm.size == 0 or self.width() <= 0:
            self.set_image(None)
            return
        columns = max(64, min(self._columns, self.width() * 2))
        mel = mel_spectrogram(self._pcm, self._sample_rate, columns=columns)
        if mel.size == 0:
            self.set_image(None)
            return
        # low frequencies at the bottom, which is how every other tool draws it
        indexed = np.clip((mel * 255.0).astype(np.int64), 0, 255)[::-1]
        self.set_image(self._lut[indexed])


__all__ = ["MelView", "heat_lut"]
