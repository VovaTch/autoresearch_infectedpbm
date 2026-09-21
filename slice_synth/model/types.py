"""
Value types for the synthesizer.

Plain frozen dataclasses with no Qt, torch or filesystem dependency: the same
objects are built by the UI, pickled to the worker, and written beside a saved
wav as its recipe.

The split that matters is between a *recipe* and a *tensor*. Style descriptors
are 1024-wide floats that only exist worker-side, inside TrackTokens.style, and
two of the four ways of choosing one are random. So the UI never handles a
vector: it sends a StyleSpec, the worker resolves it deterministically from the
spec's seed, and the resolved vector comes back with the audio. That is what
makes "the one that sounded good" reproducible three weeks later from a JSON
file, instead of only replayable from a wav.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Sequence

import numpy as np

from ab_harness.checkpoints import backend_of

StyleKind = Literal["window", "random", "jitter", "interp"]
# How the style moves once the clip is under way: fixed, a new corpus window
# per period, a fresh unit vector per period, or the style model's own
# continuation (train_style_ar.py) sampled one descriptor per period.
StyleWalk = Literal["none", "windows", "random", "ar"]
DEFAULT_PERIOD = 512
# A style entry that follows whichever track the render itself is conditioned on.
SAME_TRACK = -1
PromptKind = Literal["none", "corpus", "file"]
RenderKind = Literal["ar", "zflow", "mdm", "reference"]

# The id embedding's last row is the learned null; the UI spells that as None.
NULL_TRACK: int | None = None


def digest(payload: object, prefix: str) -> str:
    """
    Content-address a recipe.

    Args:
      payload (object): JSON-serializable content to hash.
      prefix (str): human-readable id prefix.

    Returns:
      str: prefix plus 12 hex characters, so the same recipe always resolves to
        the same id and a repeated request is recognisable as one.
    """
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return f"{prefix}_{hashlib.blake2b(blob, digest_size=6).hexdigest()}"


@dataclass(frozen=True)
class StyleSpec:
    """
    How to obtain one 1024-d style descriptor.

    Every mode resolves to a unit vector, matching how the corpus descriptors are
    built (train_ar.compute_style_windows L2-normalizes the windowed mean of z_q).
    A mode that did not would be off-distribution by magnitude alone, before any
    question about direction.

    Args:
      kind (StyleKind): "window" takes a precomputed corpus descriptor;
        "random" draws a fresh unit vector; "jitter" perturbs a real one;
        "interp" slerps between two real ones.
      track_idx (int): track owning the base window (window / jitter / interp A).
        SAME_TRACK (-1) means "whichever track this render is conditioned on",
        which is what makes one style entry usable across a whole selection --
        otherwise ticking five tracks would condition all five on track 0's mood.
      window (int): index into that track's style windows; -1 means "choose one
        disjoint from the span being generated", the rule training enforces.
      track_b (int): track owning the second window, for interp; SAME_TRACK is
        accepted here too.
      window_b (int): second window index, -1 for auto.
      mix (float): interp position; 0.0 is A, 1.0 is B.
      noise (float): jitter strength, in units of the base vector's own scale.
      seed (int): RNG seed for every draw this spec makes, so "random" is still
        reproducible.
      walk (StyleWalk): "none" holds this descriptor for the whole clip;
        "windows" replaces it every `period` frames with a random corpus window
        of the render's own tracks; "random" with a fresh unit vector; "ar"
        continues it with the style model: the next descriptor is sampled from
        the real windows fed as a prefix and the base track's id. Segment 0 is
        this spec as written, except for a cold "ar" walk (ar_prefix 0), where
        the model samples the opening descriptor too.
      period (int): frames per style segment when walking; 512 is the style
        model's own slice, other values stretch or squeeze its walk in time.
      ar_prefix (int): real windows of the base track fed to the style model,
        ending at `window` (so the walk continues from what is heard); 0 starts
        cold. Kinds without a real base window feed just the resolved vector.
      ar_temperature (float): style-model sampling temperature; 0 is greedy.
      ar_cfg (float): guidance on the track id; 1 is plain conditional, 0 uses
        the null id (a corpus-generic walk), above 1 pushes toward the track.
    """

    kind: StyleKind = "window"
    track_idx: int = 0
    window: int = -1
    track_b: int = 0
    window_b: int = -1
    mix: float = 0.5
    noise: float = 0.25
    seed: int = 0
    walk: StyleWalk = "none"
    period: int = DEFAULT_PERIOD
    ar_prefix: int = 4
    ar_temperature: float = 1.0
    ar_cfg: float = 1.0

    @property
    def walking(self) -> bool:
        """
        Returns:
          bool: True when the style changes over the clip.
        """
        return self.walk != "none" and self.period > 0

    def segments(self, positions: int) -> int:
        """
        Args:
          positions (int): sampling steps in the clip.

        Returns:
          int: how many style descriptors the clip needs; 1 when not walking.
        """
        if not self.walking:
            return 1
        return max(1, -(-positions // self.period))

    @staticmethod
    def _track_label(track_idx: int) -> str:
        """
        Args:
          track_idx (int): a track id, possibly SAME_TRACK.

        Returns:
          str: "own" for SAME_TRACK, "t<n>" otherwise.
        """
        return "own" if track_idx == SAME_TRACK else f"t{track_idx}"

    def label(self) -> str:
        """
        Returns:
          str: a short human-readable form for list rows and filenames.
        """
        track = self._track_label(self.track_idx)
        window = f"w{self.window}" if self.window >= 0 else "auto"
        if self.kind == "window":
            base = f"{track}{window}"
        elif self.kind == "random":
            base = f"rand{self.seed}"
        elif self.kind == "jitter":
            base = f"{track}{window}+n{self.noise:g}"
        else:
            base = f"{track}~{self._track_label(self.track_b)}@{self.mix:g}"
        if not self.walking:
            return base
        walk = {"windows": "win", "random": "rnd", "ar": "ar"}[self.walk]
        tail = f"{base}~{walk}{self.period}"
        if self.walk == "ar":
            tail += f"p{self.ar_prefix}" if self.ar_prefix != 4 else ""
            tail += f"c{self.ar_cfg:g}" if self.ar_cfg != 1.0 else ""
            tail += f"T{self.ar_temperature:g}" if self.ar_temperature != 1.0 else ""
        return tail

    def to_json(self) -> dict[str, Any]:
        """
        Returns:
          dict[str, Any]: JSON-safe form.
        """
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> StyleSpec:
        """
        Args:
          raw (dict[str, Any]): a dict written by to_json.

        Returns:
          StyleSpec: the rebuilt spec.
        """
        return cls(**raw)


@dataclass(frozen=True)
class PromptSpec:
    """
    What real audio, if any, the generation continues from.

    Prompting is not a nicety. Measured on the conditioning ladder, prompted
    clips beat cold starts on beat strength 0.548 against 0.409 and land on a
    tempo grid 32/32 times against 13/32 (config_ab.yaml). Training crops always
    begin mid-track, so a cold start is genuinely unseen input.

    Args:
      kind (PromptKind): "none" is a cold start; "corpus" slices cached tokens;
        "file" encodes an arbitrary audio file.
      track_idx (int): corpus track to slice.
      start_frame (int): first frame of the slice, in tokenizer frames.
      seconds (float): how much real audio to force.
      path (str): audio file, for kind "file".
      start_sec (float): offset into that file.
    """

    kind: PromptKind = "corpus"
    track_idx: int = 0
    start_frame: int = 0
    seconds: float = 3.0
    path: str = ""
    start_sec: float = 0.0

    def frames(self, fps: float) -> int:
        """
        Args:
          fps (float): tokenizer frames per second.

        Returns:
          int: prompt length in frames; 0 when this is a cold start.
        """
        return 0 if self.kind == "none" else max(0, int(self.seconds * fps))

    def to_json(self) -> dict[str, Any]:
        """
        Returns:
          dict[str, Any]: JSON-safe form.
        """
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> PromptSpec:
        """
        Args:
          raw (dict[str, Any]): a dict written by to_json.

        Returns:
          PromptSpec: the rebuilt spec.
        """
        return cls(**raw)


@dataclass(frozen=True)
class RenderSpec:
    """
    Everything needed to reproduce one clip exactly.

    Args:
      track_idx (int | None): conditioning track id; None nulls the id stream.
      co_tracks (tuple[int, ...]): further track ids that coexist with track_idx
        as id tokens at the same position -- a mishmash of several tracks in
        one slice. Empty for an ordinary render; ignored when track_idx is None.
      style (StyleSpec | None): style recipe; None nulls the style stream.
      n_frames (int): clip length in tokenizer frames.
      prompt (PromptSpec): priming audio.
      cfg_strength (float): guidance strength. 0.0 and 1.0 are both plain
        conditional sampling -- null + 1.0 * (cond - null) reduces to cond -- and
        identity has been measured absent at 1 and present only by 8, so the
        useful range here runs much higher than image diffusion's.
      temperature (float): softmax temperature; <= 0 is argmax.
      top_k (int): top-k cutoff, 0 disables.
      top_p (float): nucleus cutoff, 0 disables.
      seed (int): sampling RNG seed.
      kind (RenderKind): the backend that sampled it ("ar", "zflow", "mdm",
        from the checkpoint's family); "reference" decodes the real tokens of
        the same span instead, the tokenizer's own ceiling for comparison.
      checkpoint (str): model the clip came from.
    """

    track_idx: int | None = 0
    co_tracks: tuple[int, ...] = ()
    style: StyleSpec | None = None
    n_frames: int = 1722
    prompt: PromptSpec = PromptSpec()
    cfg_strength: float = 2.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    seed: int = 0
    kind: RenderKind = "ar"
    checkpoint: str = ""

    @property
    def item_id(self) -> str:
        """
        Returns:
          str: content-addressed id. Two specs that differ anywhere get different
            ids, so a result can never be matched to the wrong request.
        """
        return digest(self.to_json(), "r")

    @property
    def is_reference(self) -> bool:
        """
        Returns:
          bool: True when this decodes real tokens rather than sampling.
        """
        return self.kind == "reference"

    @property
    def tracks(self) -> tuple[int, ...]:
        """
        Returns:
          tuple[int, ...]: every id token the render carries, primary first;
            empty when the id stream is nulled.
        """
        if self.track_idx is None:
            return ()
        return (self.track_idx,) + tuple(self.co_tracks)

    def label(self) -> str:
        """
        Returns:
          str: a short human-readable form for list rows and filenames.
        """
        track = (
            "null" if self.track_idx is None else "t" + "+".join(map(str, self.tracks))
        )
        if self.is_reference:
            return f"{track} reference"
        style = self.style.label() if self.style is not None else "nullstyle"
        return f"{track} {style} s{self.seed}"

    def to_json(self) -> dict[str, Any]:
        """
        Returns:
          dict[str, Any]: JSON-safe form.
        """
        return {
            "track_idx": self.track_idx,
            "co_tracks": list(self.co_tracks),
            "style": None if self.style is None else self.style.to_json(),
            "n_frames": self.n_frames,
            "prompt": self.prompt.to_json(),
            "cfg_strength": self.cfg_strength,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
            "kind": self.kind,
            "checkpoint": self.checkpoint,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> RenderSpec:
        """
        Args:
          raw (dict[str, Any]): a dict written by to_json.

        Returns:
          RenderSpec: the rebuilt spec.
        """
        style = raw.get("style")
        return cls(
            **raw
            | {
                "co_tracks": tuple(int(t) for t in raw.get("co_tracks", ())),
                "style": None if style is None else StyleSpec.from_json(style),
                "prompt": PromptSpec.from_json(raw["prompt"]),
            }
        )


@dataclass
class Render:
    """
    One finished clip: what was asked for, and what came back.

    Args:
      spec (RenderSpec): the recipe.
      tokens (np.ndarray): (T, R) int16 codes.
      pcm (np.ndarray): (N,) int16 mono, loudness-normalized. int16 is what
        QAudioSink consumes and what crosses the process boundary.
      sample_rate (int): samples per second of pcm.
      style_used (np.ndarray): (style_dim,) float32 descriptor the worker
        actually resolved, or (segments, style_dim) for a walking style; for a
        random or jittered recipe this is the only record of what was heard.
      fill (float): share of the clip above the near-silence floor.
    """

    spec: RenderSpec
    tokens: np.ndarray
    pcm: np.ndarray
    sample_rate: int = 44100
    style_used: np.ndarray | None = None
    fill: float = -1.0

    @property
    def seconds(self) -> float:
        """
        Returns:
          float: clip length in seconds.
        """
        return self.pcm.size / self.sample_rate if self.sample_rate else 0.0


def build_variants(
    base: RenderSpec,
    tracks: Sequence[int | None],
    styles: Sequence[StyleSpec | None],
    seeds: Sequence[int],
    reference: bool = False,
    together: bool = False,
) -> list[RenderSpec]:
    """
    Expand a selection into one spec per combination.

    The cross product is the whole point of the app: sampling is bound by kernel
    launch latency rather than arithmetic (0.48 ms per step per clip at batch 16
    against 7.4 ms at batch 1), so hearing sixteen variants costs what hearing
    one does. Anything the user can tick, they should be able to tick several of.

    A reference row is added per track rather than per cell: it decodes real
    tokens, so conditioning and seed do not apply to it, and one copy per track
    is all the ceiling there is to hear.

    Args:
      base (RenderSpec): the settings shared by every variant.
      tracks (Sequence[int | None]): track ids; None nulls the id stream.
      styles (Sequence[StyleSpec | None]): style recipes; None nulls the stream.
      seeds (Sequence[int]): sampling seeds.
      reference (bool): also decode each track's real tokens for the same span.
      together (bool): fold every real track into ONE cell whose id tokens
        coexist (the first as track_idx, the rest as co_tracks) instead of one
        cell per track. A ticked null still gets its own cell either way, since
        a null cannot join a blend.

    Returns:
      list[RenderSpec]: one spec per (track, style, seed), reference rows last.
        Empty when any axis is empty, since a cross product with nothing in it
        has no members.
    """
    if not tracks or not styles or not seeds:
        return []
    cells: list[tuple[int | None, tuple[int, ...]]] = [(t, ()) for t in tracks]
    if together:
        real = [t for t in tracks if t is not None]
        cells = [(real[0], tuple(real[1:]))] if real else []
        cells += [(None, ())] if None in tracks else []
    out = [
        replace(
            base,
            track_idx=track,
            co_tracks=extra,
            style=style,
            seed=seed,
            kind=backend_of(base.checkpoint),  # type: ignore[arg-type]
        )
        for track, extra in cells
        for style in styles
        for seed in seeds
    ]
    if reference:
        out += [
            replace(
                base,
                track_idx=track,
                co_tracks=(),
                style=None,
                seed=0,
                kind="reference",
            )
            for track in tracks
            if track is not None
        ]
    return out


__all__ = [
    "DEFAULT_PERIOD",
    "NULL_TRACK",
    "SAME_TRACK",
    "PromptKind",
    "PromptSpec",
    "Render",
    "RenderKind",
    "RenderSpec",
    "StyleKind",
    "StyleSpec",
    "StyleWalk",
    "build_variants",
    "digest",
]
