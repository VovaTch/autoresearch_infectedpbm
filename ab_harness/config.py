"""
Harness configuration.

Follows the repo convention rather than introducing a new one: plain
yaml.safe_load onto dataclasses, with unknown keys rejected. Silent key drops
are how a config typo turns into a session that looks fine and collects the
wrong pairs.

_section mirrors train_ar._build_section rather than importing it. Importing it
would pull torch, lightning and onnxruntime into the UI process, which is the
one thing the process split exists to prevent -- and would add seconds to
startup for fifteen lines of dataclass construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from ab_harness.model.pair_sampler import SamplerCfg

REPO = Path(__file__).resolve().parent.parent

T = TypeVar("T")


def _section(cls: type[T], raw: dict[str, Any] | None, name: str) -> T:
    """
    Instantiate one config dataclass, rejecting unknown keys.

    Args:
      cls (type[T]): the dataclass to build.
      raw (dict[str, Any] | None): the YAML section, or None if absent.
      name (str): section name, for the error message.

    Returns:
      T: an instance of cls.
    """
    raw = raw or {}
    if unknown := set(raw) - {f.name for f in fields(cls)}:  # type: ignore[arg-type]
        raise ValueError(f"unknown key(s) in '{name}': {sorted(unknown)}")
    return cls(**raw)


@dataclass
class BankCfg:
    """
    Where collected tokens and judgements live.

    Args:
      root (str): bank directory, next to the token cache it references.
    """

    root: str = "~/.cache/infected_pbm/ab"


@dataclass
class GeneratorCfg:
    """
    The worker's model and decoding settings.

    Args:
      checkpoint (str): AR checkpoint. Defaults to ar_latest rather than
        ar_best: config_20260829_ar24h.yaml records that val/loss bottoms near
        step 1500 and is a memorization thermometer, not a model selector, so
        ar_best is pinned to a barely-trained model.
      decoder_onnx (str): tokens-to-audio graph.
      device (str): torch device for sampling.
      window_frames (int): context retained by the sliding window; keep at the
        checkpoint's crop_frames so sampling stays in-distribution.
      reprime_frac (float): fraction of the window dropped per re-prime.
      use_gpu_decoder (bool): try the CUDA execution provider for the decoder.
      max_batch (int): clips sampled together. The loop is launch-bound, so a
        step costs 7.4 ms at batch 1 and 7.6 ms at batch 16 -- raising this is
        very nearly free throughput, up to the point where the KV cache stops
        fitting in VRAM.
      batch_wait_s (float): how long the worker waits for more requests before
        sampling what it already has, so a lone request is not held up.
    """

    checkpoint: str = "saved_ar_20260829_24h/ar_latest.ckpt"
    decoder_onnx: str = "onnx/decoder.onnx"
    device: str = "cuda:0"
    window_frames: int = 4096
    reprime_frac: float = 0.25
    use_gpu_decoder: bool = True
    max_batch: int = 8
    batch_wait_s: float = 0.5


@dataclass
class SessionCfg:
    """
    Rating-loop settings.

    Args:
      target_lufs (float): loudness every clip is matched to before playback.
      prefetch_depth (int): pairs kept ready ahead of the rater. Four pairs
        is eight clips, which fills a sampling batch exactly at the default
        max_batch and so buys the most throughput per pair queued.
      seed (int | None): sampler seed; None draws a fresh one per session.
      crossfade_ms (float): equal-power fade applied when toggling A/B.
      structure_live (bool): generate structure-tier pairs on demand. Off by
        default: a 90 s pair is around two minutes of sampling against roughly
        ten seconds of rating, so drawing one inline stalls the queue.
      structure_backfill (int): structure pairs generated in the background at
        once, submitted only when the rating queue is already full. This is how
        the structure tier fills in without a manual bake -- the clips land in
        the bank and are served by later draws and later sessions. 0 disables it.
      min_fill (float): pairs whose quieter side is emptier than this never
        reach the worklist. The 90 s tier averages 30% near-silent against 0%
        for real tokens, and a dead clip costs a listen while teaching a reward
        model only that energy wins. 0 disables the gate.
    """

    target_lufs: float = -23.0
    prefetch_depth: int = 4
    seed: int | None = None
    crossfade_ms: float = 5.0
    structure_live: bool = False
    structure_backfill: int = 1
    min_fill: float = 0.25
    quiet_fill: float = 0.25


@dataclass
class AbConfig:
    """
    Top-level config, one block per YAML section.

    Args:
      bank (BankCfg): storage locations.
      generator (GeneratorCfg): worker settings.
      sampler (SamplerCfg): tier mix and conditioning odds.
      session (SessionCfg): rating-loop settings.
    """

    bank: BankCfg = field(default_factory=BankCfg)
    generator: GeneratorCfg = field(default_factory=GeneratorCfg)
    sampler: SamplerCfg = field(default_factory=SamplerCfg)
    session: SessionCfg = field(default_factory=SessionCfg)

    @property
    def bank_root(self) -> Path:
        """
        Returns:
          Path: the bank directory with ~ expanded.
        """
        return Path(self.bank.root).expanduser()


def load_config(path: str | Path) -> AbConfig:
    """
    Read a YAML config, rejecting unknown keys.

    Args:
      path (str | Path): config file.

    Returns:
      AbConfig: the parsed config.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    if unknown := set(raw) - {"bank", "generator", "sampler", "session"}:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    sampler = _section(SamplerCfg, raw.get("sampler"), "sampler")
    # YAML gives a list where the dataclass wants a tuple
    sampler.cfg_strengths = tuple(sampler.cfg_strengths)
    return AbConfig(
        bank=_section(BankCfg, raw.get("bank"), "bank"),
        generator=_section(GeneratorCfg, raw.get("generator"), "generator"),
        sampler=sampler,
        session=_section(SessionCfg, raw.get("session"), "session"),
    )
