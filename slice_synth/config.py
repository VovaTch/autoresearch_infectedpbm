"""
Synthesizer configuration.

Same convention as everything else in the repo -- plain yaml.safe_load onto
dataclasses, unknown keys rejected -- and the generator section is literally
ab_harness's GeneratorCfg rather than a copy of it, so the two apps cannot end
up sampling with different window geometry from the same checkpoint.

Only the sections that describe *this* app are new: where keepers are written,
and what the knobs start at when the window opens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ab_harness.config import REPO, GeneratorCfg, _section

SECTIONS = ("generator", "output", "ui")


@dataclass
class OutputCfg:
    """
    Where saved renders go.

    Args:
      root (str): directory for the wav / tokens / spec triples.
      target_lufs (float): loudness every clip is matched to before playback.
        Kept even though nothing here is a blind comparison: a variant list where
        the loud ones sound better is a variant list that picks itself.
    """

    root: str = "renders_synth"
    target_lufs: float = -23.0


@dataclass
class UiCfg:
    """
    Opening values for the control panel.

    Args:
      seconds (float): clip length.
      prompt_seconds (float): how much real audio primes a generation.
      temperature (float): sampling temperature.
      top_k (int): top-k cutoff, 0 disables.
      top_p (float): nucleus cutoff, 0 disables.
      cfg_strength (float): guidance strength. 0.0 and 1.0 are both plain
        conditional sampling; track identity has been measured absent at 1 and
        present only around 8, so the slider runs to max_cfg rather than the 1-3
        band image diffusion uses.
      max_cfg (float): upper end of the guidance slider.
      crossfade_ms (float): fade applied by the playback transport.
      mel_columns (int): time resolution of the mel strip, in columns.
      seeds (int): how many seeds each conditioning cell is expanded into.
    """

    seconds: float = 10.0
    prompt_seconds: float = 3.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    cfg_strength: float = 2.0
    max_cfg: float = 8.0
    crossfade_ms: float = 5.0
    mel_columns: int = 1024
    seeds: int = 1


@dataclass
class SynthConfig:
    """
    Top-level config, one block per YAML section.

    Args:
      generator (GeneratorCfg): checkpoint, device and sampling geometry.
      output (OutputCfg): where keepers land.
      ui (UiCfg): opening values for the controls.
    """

    generator: GeneratorCfg = field(default_factory=GeneratorCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    ui: UiCfg = field(default_factory=UiCfg)

    def __post_init__(self) -> None:
        """Resolve an 'auto' checkpoint once, so no caller sees the sentinel."""
        from ab_harness.checkpoints import resolve_checkpoint

        self.generator.checkpoint = resolve_checkpoint(self.generator.checkpoint, REPO)

    @property
    def checkpoints(self) -> list[str]:
        """
        Returns:
          list[str]: loadable checkpoints, the configured one first so the
            selector opens on what is actually running.
        """
        from ab_harness.checkpoints import discover_checkpoints

        found = discover_checkpoints(REPO)
        current = self.generator.checkpoint
        return [current] + [c for c in found if c != current]

    @property
    def output_root(self) -> Path:
        """
        Returns:
          Path: the save directory, ~ expanded and repo-anchored when relative.
        """
        root = Path(self.output.root).expanduser()
        return root if root.is_absolute() else REPO / root


def load_config(path: str | Path) -> SynthConfig:
    """
    Read a YAML config, rejecting unknown keys.

    Args:
      path (str | Path): config file.

    Returns:
      SynthConfig: the parsed config.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    if unknown := set(raw) - set(SECTIONS):
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return SynthConfig(
        generator=_section(GeneratorCfg, raw.get("generator"), "generator"),
        output=_section(OutputCfg, raw.get("output"), "output"),
        ui=_section(UiCfg, raw.get("ui"), "ui"),
    )


def resolve_config(name: str) -> SynthConfig:
    """
    Load a config from the working directory or the repo root, or fall back to
    the defaults so the app still starts without one.

    Args:
      name (str): config filename or path.

    Returns:
      SynthConfig: the config to run with.
    """
    for candidate in (Path(name), REPO / name):
        if candidate.exists():
            return load_config(candidate)
    return SynthConfig()


__all__ = ["OutputCfg", "SynthConfig", "UiCfg", "load_config", "resolve_config"]
