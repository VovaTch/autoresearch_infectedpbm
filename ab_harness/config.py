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

from ab_harness.checkpoints import (
    BACKENDS,
    STYLE_AR_FAMILIES,
    backend_of,
    discover_all,
    resolve_checkpoint,
)
from ab_harness.model.auto_train import AutoTrainCfg
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
      checkpoint (str): AR or DPO checkpoint to sample from, or "auto" for the
        newest one ab_harness.checkpoints finds -- the last DPO run if there is
        one. Resolved to a concrete path at load, and switchable in the app
        without a restart. Prefer a *_latest over a *_best: val/loss bottoms
        near step 1500 (config_20260829_ar24h.yaml) and DPO val/acc is a random
        walk on 38 pairs, so "best" is a memorization thermometer pinned to a
        barely-trained model, not a model selector.
      decoder_onnx (str): tokens-to-audio graph.
      encoder_onnx (str): audio-to-tokens graph. Unused by the rating harness,
        which only ever decodes cached tokens; the slice synthesizer needs it to
        prime a generation from an audio file that is not in the corpus.
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
      style_ar_checkpoint (str): style-level model (train_style_ar.py) behind
        the "ar" style walk -- the slice synthesizer's walk entries and the
        harness's walking conditioning cell (sampler.p_style_walk) -- or "auto"
        for the newest saved_style_ar_* run. Only loaded when a walking clip
        is rendered.
      flow_steps (int): Euler steps per window for a latent-flow checkpoint
        (saved_zflow_*; ab_harness.worker.zflow_gen).
      flow_churn (float): noise re-draw fraction per flow step, 0 = plain ODE.
      requantize_beam (int): beam width turning a flow latent back into RVQ
        tokens (train_zflow.requantize); 8 agrees with the source on 99% of
        tokens, greedy (1) flips ~15% of level 0.
      mdm_steps (list[int]): MaskGIT rounds per RVQ level for a masked
        diffusion checkpoint (saved_mdm_*; ab_harness.worker.mdm_gen).
      mdm_choice_temperature (float): Gumbel scale on the confidence ranking
        that decides which draws survive a round, annealed to 0 per level.
    """

    checkpoint: str = "auto"
    decoder_onnx: str = "onnx/decoder.onnx"
    encoder_onnx: str = "onnx/encoder.onnx"
    device: str = "cuda:0"
    window_frames: int = 4096
    reprime_frac: float = 0.25
    use_gpu_decoder: bool = True
    max_batch: int = 8
    batch_wait_s: float = 0.5
    style_ar_checkpoint: str = "auto"
    flow_steps: int = 32
    flow_churn: float = 0.0
    requantize_beam: int = 8
    mdm_steps: list[int] = field(default_factory=lambda: [16, 8, 8])
    mdm_choice_temperature: float = 4.5


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
      auto_train (AutoTrainCfg): automatic DPO rounds every N usable pairs.
    """

    bank: BankCfg = field(default_factory=BankCfg)
    generator: GeneratorCfg = field(default_factory=GeneratorCfg)
    sampler: SamplerCfg = field(default_factory=SamplerCfg)
    session: SessionCfg = field(default_factory=SessionCfg)
    auto_train: AutoTrainCfg = field(default_factory=AutoTrainCfg)

    def __post_init__(self) -> None:
        """Resolve an "auto" checkpoint once, so no caller sees the sentinel."""
        self.generator.checkpoint = resolve_checkpoint(self.generator.checkpoint, REPO)
        self.generator.style_ar_checkpoint = resolve_checkpoint(
            self.generator.style_ar_checkpoint, REPO, STYLE_AR_FAMILIES
        )

    @property
    def checkpoints(self) -> list[str]:
        """
        Returns:
          list[str]: loadable checkpoints of every backend, AR family first and
            newest first within each, with the configured one in front so the
            selector opens on what is actually running.
        """
        return checkpoint_menu(self.generator.checkpoint)

    @property
    def checkpoints_by_backend(self) -> dict[str, list[str]]:
        """
        Returns:
          dict[str, list[str]]: backend -> its loadable checkpoints, newest
            first, the configured one in front of its own backend's list.
        """
        return checkpoint_menus(self.generator.checkpoint)

    @property
    def bank_root(self) -> Path:
        """
        Returns:
          Path: the bank directory with ~ expanded.
        """
        return Path(self.bank.root).expanduser()


def checkpoint_menus(current: str) -> dict[str, list[str]]:
    """
    Args:
      current (str): the checkpoint in use.

    Returns:
      dict[str, list[str]]: backend -> loadable checkpoints, newest first,
        with `current` moved to the front of its backend's list.
    """
    menus = discover_all(REPO)
    own = backend_of(current)
    menus[own] = [current] + [c for c in menus[own] if c != current]
    return menus


def checkpoint_menu(current: str) -> list[str]:
    """
    Args:
      current (str): the checkpoint in use.

    Returns:
      list[str]: every backend's checkpoints in one list, `current` first.
    """
    menus = checkpoint_menus(current)
    rest = [c for b in BACKENDS for c in menus[b] if c != current]
    return [current] + rest


def load_config(path: str | Path) -> AbConfig:
    """
    Read a YAML config, rejecting unknown keys.

    Args:
      path (str | Path): config file.

    Returns:
      AbConfig: the parsed config.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    if unknown := set(raw) - {
        "bank",
        "generator",
        "sampler",
        "session",
        "auto_train",
    }:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    sampler = _section(SamplerCfg, raw.get("sampler"), "sampler")
    # YAML gives a list where the dataclass wants a tuple
    sampler.cfg_strengths = tuple(sampler.cfg_strengths)
    return AbConfig(
        bank=_section(BankCfg, raw.get("bank"), "bank"),
        generator=_section(GeneratorCfg, raw.get("generator"), "generator"),
        sampler=sampler,
        session=_section(SessionCfg, raw.get("session"), "session"),
        auto_train=_section(AutoTrainCfg, raw.get("auto_train"), "auto_train"),
    )
