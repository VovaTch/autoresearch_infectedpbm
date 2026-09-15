"""
Keeping the good ones.

A render exists only in RAM until it is saved, and what is worth persisting is
not the audio alone. Three files go out together:

  <stem>.wav         the audio, int16, already loudness-matched
  <stem>.tokens.npy  (T, R) int16 codes -- the thing the decoder actually ate
  <stem>.json        the RenderSpec, plus the style vector that was resolved

The json is the part that matters later. Two of the four style modes are random,
so without the resolved vector a keeper made from `random` could never be heard
again except by replaying its wav; with it, the same clip can be re-generated at
a different length, against a different checkpoint, or fed to a trainer as a
conditioning target.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from slice_synth.model.types import Render, RenderSpec

UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_stem(label: str, item_id: str, when: datetime | None = None) -> str:
    """
    Args:
      label (str): human-readable render label.
      item_id (str): content-addressed id, appended so two renders that differ
        only in something the label omits cannot collide.
      when (datetime | None): timestamp to lead with; now when omitted.

    Returns:
      str: a filesystem-safe stem.
    """
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    body = UNSAFE.sub("_", label).strip("_")
    return f"{stamp}_{body}_{item_id}"


@dataclass(frozen=True)
class SavedRender:
    """
    Where one saved render landed.

    Args:
      stem (str): shared filename stem.
      wav (Path): the audio.
      tokens (Path): the codes.
      spec (Path): the recipe.
    """

    stem: str
    wav: Path
    tokens: Path
    spec: Path


def save_render(
    render: Render, root: Path, when: datetime | None = None
) -> SavedRender:
    """
    Write one render's three files.

    Args:
      render (Render): the clip to keep.
      root (Path): output directory, created if missing.
      when (datetime | None): timestamp for the stem.

    Returns:
      SavedRender: the paths written.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(render.spec.label(), render.spec.item_id, when)

    wav = root / f"{stem}.wav"
    sf.write(
        str(wav), render.pcm.astype(np.int16), render.sample_rate, subtype="PCM_16"
    )

    tokens = root / f"{stem}.tokens.npy"
    np.save(tokens, render.tokens.astype(np.int16))

    spec = root / f"{stem}.json"
    spec.write_text(
        json.dumps(
            {
                "spec": render.spec.to_json(),
                "item_id": render.spec.item_id,
                "sample_rate": render.sample_rate,
                "seconds": round(render.seconds, 3),
                "fill": round(float(render.fill), 4),
                "style_used": (
                    None
                    if render.style_used is None
                    else np.round(render.style_used, 6).tolist()
                ),
            },
            indent=2,
        )
    )
    return SavedRender(stem=stem, wav=wav, tokens=tokens, spec=spec)


def load_spec(path: Path) -> tuple[RenderSpec, np.ndarray | None]:
    """
    Read a saved recipe back.

    Args:
      path (Path): a .json written by save_render.

    Returns:
      tuple[RenderSpec, np.ndarray | None]: the spec and the style descriptor
        that was used -- (D,) or (S, D) for a walking style -- when recorded.
    """
    raw = json.loads(Path(path).read_text())
    style = raw.get("style_used")
    return (
        RenderSpec.from_json(raw["spec"]),
        None if style is None else np.asarray(style, dtype=np.float32),
    )


__all__ = ["SavedRender", "load_spec", "safe_stem", "save_render"]
