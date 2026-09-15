"""
Audio file -> RVQ tokens, so a generation can be primed from outside the corpus.

The 53 cached tracks are already tokenized, which covers most prompting. This
covers the rest: hum something, drop in a loop, continue a track the model has
never seen.

Two things about the encoder are worth knowing before trusting the output.
Its tokens depend on the input LENGTH -- cuDNN picks different convolution
algorithms for different shapes, and levels 1-2 flip on roughly 0.05% of frames
between two encodings of the same audio at different lengths. Level 0 is exact.
So the same chunk_frames/margin geometry the token cache was built with is used
here, not because it is faster but because it is the only way the prompt lands
in the same distribution as the training data. And some mp3s are simply
undecodable by the installed backend; that surfaces as an error on the one
render that asked for it.

Encoding a whole track takes seconds, and a prompt is three of them, so results
are cached by (path, mtime, size). Re-prompting from a file you are iterating on
costs one encode, not one per generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from train_ar import encode_chunked, make_session


def load_audio(path: Path, sample_rate: int, hop_length: int) -> torch.Tensor:
    """
    Load any audio file as mono at the tokenizer's rate, trimmed to whole frames.

    train_ar.load_track_audio does this for the corpus but hard-codes
    format="mp3", which is right for the corpus and wrong for a file the user
    picked. The format hint is kept as a fallback: some of the corpus mp3s only
    decode when it is supplied.

    Args:
      path (Path): the audio file.
      sample_rate (int): target rate, 44100 for this tokenizer.
      hop_length (int): STFT hop, 256.

    Returns:
      torch.Tensor: (1, L) float32 mono, L a multiple of hop_length.

    Raises:
      RuntimeError: when the file cannot be decoded.
    """
    try:
        wav, src_rate = torchaudio.load(str(path))
    except Exception:
        try:
            wav, src_rate = torchaudio.load(str(path), format="mp3")
        except Exception as exc:
            raise RuntimeError(f"cannot decode {path.name}: {exc}") from exc
    if src_rate != sample_rate:
        wav = torchaudio.transforms.Resample(src_rate, sample_rate)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    usable = (wav.shape[-1] // hop_length) * hop_length
    if usable <= 0:
        raise RuntimeError(f"{path.name} is shorter than one frame")
    return wav[..., :usable].contiguous().float()


class FileEncoder:
    """
    Lazily-opened ONNX encoder with an on-disk token cache.

    The session is not opened until the first file prompt: most sessions never
    use one, and opening an ONNX graph on the GPU costs seconds and VRAM.

    Args:
      onnx_path (Path): encoder.onnx location.
      meta (dict[str, Any]): tokenizer meta -- sample_rate, hop_length, num_rq.
      chunk_frames (int): frames emitted per encode window; must match the value
        the token cache was built with.
      margin (int): context frames discarded per side, likewise.
      cache_dir (Path | None): where encoded tracks are kept. None disables it.
    """

    def __init__(
        self,
        onnx_path: Path,
        meta: dict[str, Any],
        chunk_frames: int = 4096,
        margin: int = 256,
        cache_dir: Path | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.sample_rate = int(meta["sample_rate"])
        self.hop = int(meta["hop_length"])
        self.num_rq = int(meta["num_rq"])
        self.chunk_frames = chunk_frames
        self.margin = margin
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._session: Any | None = None

    def _key(self, path: Path) -> str:
        """
        Args:
          path (Path): the audio file.

        Returns:
          str: cache key covering identity and content -- a file edited in place
            keeps its name, so mtime and size have to be in the key.
        """
        stat = path.stat()
        blob = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{self.chunk_frames}|{self.margin}"
        return hashlib.blake2b(blob.encode(), digest_size=12).hexdigest()

    def _session_or_open(self):
        """
        Returns:
          onnxruntime.InferenceSession: the encoder graph, opened on first use.

        Raises:
          FileNotFoundError: when the graph is missing.
        """
        if self._session is None:
            if not self.onnx_path.exists():
                raise FileNotFoundError(f"no encoder graph at {self.onnx_path}")
            self._session = make_session(self.onnx_path, self.num_rq)
        return self._session

    def tokens(self, path: str | Path) -> np.ndarray:
        """
        Encode a whole file, using the cache when it is warm.

        Args:
          path (str | Path): the audio file.

        Returns:
          np.ndarray: (T, R) int16 codes for the whole file.

        Raises:
          RuntimeError: when the file cannot be decoded.
          FileNotFoundError: when the file or the encoder graph is missing.
        """
        source = Path(path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"no audio file at {source}")

        cached = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached = self.cache_dir / f"{self._key(source)}.npy"
            if cached.exists():
                return np.load(cached)

        wav = load_audio(source, self.sample_rate, self.hop)
        codes = encode_chunked(
            self._session_or_open(), wav, self.hop, self.chunk_frames, self.margin
        ).astype(np.int16)
        if cached is not None:
            np.save(cached, codes)
            cached.with_suffix(".json").write_text(
                json.dumps({"source": str(source), "frames": int(codes.shape[0])})
            )
        return codes

    def slice(self, path: str | Path, start_sec: float, frames: int) -> np.ndarray:
        """
        Encode a file and cut the prompt out of it.

        Args:
          path (str | Path): the audio file.
          start_sec (float): offset into the file.
          frames (int): prompt length in tokenizer frames.

        Returns:
          np.ndarray: (P, R) int16 codes, P <= frames -- a start near the end of
            a short file yields a short prompt rather than an error, since a
            shorter prompt is still a usable one.
        """
        codes = self.tokens(path)
        fps = self.sample_rate / self.hop
        start = max(0, min(int(start_sec * fps), max(0, codes.shape[0] - 1)))
        return codes[start : start + max(0, frames)]

    def close(self) -> None:
        """Release the ONNX session."""
        self._session = None


__all__ = ["FileEncoder", "load_audio"]
