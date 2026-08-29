"""
Tokens-to-audio for the harness.

Thin wrapper over the ONNX decoder helpers already in generate_ar, so the
harness and generate_ar cannot drift into decoding the same tokens differently
-- including the chunked overlap trimming, which is what removed the 1.35 Hz
click train from slice-concatenated renders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from generate_ar import decode_tokens, make_decoder


class TokenDecoder:
    """
    ONNX decoder session plus the frame geometry it needs.

    Args:
      path (Path): decoder.onnx location.
      hop (int): samples per frame.
      sample_rate (int): output sample rate.
      use_gpu (bool): try the CUDA execution provider first.
      chunk (int): frames decoded per window.
      margin (int): context frames discarded per side.
    """

    def __init__(
        self,
        path: Path,
        hop: int,
        sample_rate: int,
        use_gpu: bool = True,
        chunk: int = 4096,
        margin: int = 256,
    ) -> None:
        self.session = make_decoder(Path(path), use_gpu)
        self.hop = hop
        self.sample_rate = sample_rate
        self.chunk = chunk
        self.margin = margin

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """
        Decode one clip's codes to mono audio.

        Args:
          tokens (np.ndarray): (T, R) integer codes.

        Returns:
          np.ndarray: (T * hop,) float32 mono waveform.
        """
        codes = torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.int64))[None]
        wav = decode_tokens(self.session, codes, self.hop, self.chunk, self.margin)
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    @property
    def providers(self) -> list[str]:
        """
        Returns:
          list[str]: execution providers the session actually opened.
        """
        return list(self.session.get_providers())
