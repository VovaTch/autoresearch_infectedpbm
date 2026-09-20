"""
Value types shared by every layer of the harness.

These are deliberately plain dataclasses with no Qt, torch or filesystem
dependency: the same objects cross the process boundary to the worker, get
persisted to the bank, and drive the view-models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np

Choice = Literal["left", "right", "tie"]
Side = Literal["left", "right"]


class Tier(StrEnum):
    """
    The two rating tiers of PLAN section 7.2.

    They exist because a single "which is better?" question collapses fidelity
    and interest onto whichever is easier to hear, and in short clips that is
    always fidelity.
    """

    BULK = "bulk"
    STRUCTURE = "structure"


QUESTIONS: dict[Tier, str] = {
    Tier.BULK: "Which sounds cleaner?",
    Tier.STRUCTURE: "Which is more interesting?",
}


# Conditioning fields that only mean something when the style walks.
WALK_FIELDS = (
    "style_walk",
    "style_period",
    "style_ar_prefix",
    "style_ar_temperature",
    "style_ar_cfg",
    "style_seed",
)

# Conditioning fields that only mean something when a stream crosses tracks.
CROSS_FIELDS = ("id_track_idx", "style_track_idx")


@dataclass(frozen=True)
class Conditioning:
    """
    The conditioning recipe for one clip.

    Each stream is independently nullable, which is what makes a fully
    unconditional draw one of the eight reachable combinations. The id and
    style streams can also each come from a track other than the span's
    (probe_cross_track.py): the prompt is always the span's own real tokens,
    so crossing a stream is the only way to ask it to do work the prompt does
    not already do.

    Args:
      track_idx (int): index into the sorted corpus: the span's track, whose
        real tokens are the prompt and the anchor's reference.
      start_frame (int): first frame of the span in the track's token stream.
      use_track_id (bool): False nulls the id stream via drop_id.
      use_style (bool): False nulls the style stream via drop_style.
      style_window (int): index into the style track's TrackTokens.style; -1
        when unused.
      id_track_idx (int): track whose id embedding row is fed; -1 means the
        span's own track.
      style_track_idx (int): track style_window indexes into; -1 means the
        span's own track.
      prompt_frames (int): real frames forced before sampling begins.
      cfg_strength (float): guidance strength; 0.0 is plain conditional sampling.
      style_walk (str): "none" holds style_window's descriptor for the whole
        clip; "ar" keeps it for the first segment and lets the style model
        (train_style_ar.py) continue it, one descriptor per style_period
        positions, primed with the real windows ending at style_window.
      style_period (int): positions per style segment when walking; 0 is the
        same as not walking.
      style_ar_prefix (int): real windows fed to the style model as its prefix.
      style_ar_temperature (float): style-model sampling temperature.
      style_ar_cfg (float): style-model guidance on the track id; 1 is plain.
      style_seed (int): seeds the style model's draw. Part of the conditioning,
        not of Sampling, so both clips of a pair hear the same walk and differ
        in token sampling only.
    """

    track_idx: int
    start_frame: int
    use_track_id: bool = True
    use_style: bool = True
    style_window: int = -1
    prompt_frames: int = 0
    cfg_strength: float = 0.0
    style_walk: str = "none"
    style_period: int = 0
    style_ar_prefix: int = 4
    style_ar_temperature: float = 1.0
    style_ar_cfg: float = 1.0
    style_seed: int = 0
    id_track_idx: int = -1
    style_track_idx: int = -1

    @property
    def walking(self) -> bool:
        """
        Returns:
          bool: True when the style changes over the clip.
        """
        return self.use_style and self.style_walk != "none" and self.style_period > 0

    @property
    def id_track(self) -> int:
        """
        Returns:
          int: the track whose id row the id stream carries.
        """
        return self.track_idx if self.id_track_idx < 0 else self.id_track_idx

    @property
    def style_track(self) -> int:
        """
        Returns:
          int: the track style_window indexes into.
        """
        return self.track_idx if self.style_track_idx < 0 else self.style_track_idx

    @property
    def crossing(self) -> bool:
        """
        Returns:
          bool: True when a kept stream comes from a track other than the span's.
        """
        return (self.use_track_id and self.id_track != self.track_idx) or (
            self.use_style and self.style_track != self.track_idx
        )

    def digest_payload(self) -> dict[str, Any]:
        """
        Returns:
          dict[str, Any]: the fields that identify the recipe. The walk fields
            are left out of a non-walking recipe, and the cross-track fields of
            a same-track one, so their digests are the ones every clip banked
            before those cells existed was filed under.
        """
        payload = asdict(self)
        if not self.walking:
            for key in WALK_FIELDS:
                del payload[key]
        if not self.crossing:
            for key in CROSS_FIELDS:
                del payload[key]
        return payload


@dataclass(frozen=True)
class Sampling:
    """
    Decoder settings for one clip.

    Args:
      seed (float): RNG seed; the ONLY field allowed to differ within a pair.
      temperature (float): softmax temperature; <= 0 is argmax.
      top_k (int): top-k cutoff, 0 disables.
      top_p (float): nucleus cutoff, 0 disables.
    """

    seed: int
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0


@dataclass(frozen=True)
class ClipSpec:
    """
    Everything needed to reproduce one clip's tokens exactly.

    Args:
      item_id (str): stable id, also the bank filename stem.
      tier (Tier): which question this clip will be rated under.
      group_id (str): clips sharing a group share conditioning and differ only
        in Sampling.seed (PLAN section 7.4).
      n_frames (int): clip length in tokenizer frames.
      conditioning (Conditioning): the recipe.
      sampling (Sampling): decoder settings.
      generator (str): "ar", "reference", or a future "dit".
      checkpoint (str): checkpoint tag the clip came from.
    """

    item_id: str
    tier: Tier
    group_id: str
    n_frames: int
    conditioning: Conditioning
    sampling: Sampling
    generator: str = "ar"
    checkpoint: str = ""

    @property
    def is_reference(self) -> bool:
        """
        Returns:
          bool: True when this clip is real cached tokens, not a generation.
        """
        return self.generator == "reference"

    def to_json(self) -> dict[str, Any]:
        """
        Returns:
          dict[str, Any]: JSON-safe form for the bank.
        """
        return asdict(self) | {"tier": str(self.tier)}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> ClipSpec:
        """
        Args:
          raw (dict[str, Any]): a dict written by to_json.

        Returns:
          ClipSpec: the rebuilt spec.
        """
        return cls(
            **raw
            | {
                "tier": Tier(raw["tier"]),
                "conditioning": Conditioning(**raw["conditioning"]),
                "sampling": Sampling(**raw["sampling"]),
            }
        )


@dataclass
class Clip:
    """
    A clip's tokens plus the audio the rater will actually hear.

    Args:
      spec (ClipSpec): what produced it.
      tokens (np.ndarray): (T, R) int16 codes. The only thing persisted.
      pcm (np.ndarray): (N,) int16 mono at the tokenizer's sample rate, already
        loudness-normalized. int16 is what QAudioSink consumes and what crosses
        the process boundary. Never written to disk.
      was_live (bool): True if generated during this session rather than loaded.
      fill (float): share of the clip above the near-silence floor, from
        audio.fill_fraction. Measured once by the producer, which already holds
        the waveform, so nothing downstream has to re-scan four million samples.
        -1.0 when it was never measured.
    """

    spec: ClipSpec
    tokens: np.ndarray
    pcm: np.ndarray
    was_live: bool = False
    fill: float = -1.0


@dataclass(frozen=True)
class PairSpec:
    """
    One blind comparison.

    `left` and `right` are already randomized, so no downstream layer can infer
    identity from position.

    Args:
      pair_id (str): stable id for this comparison.
      tier (Tier): rating tier.
      left (ClipSpec): clip shown on the left.
      right (ClipSpec): clip shown on the right.
      is_repeat (bool): a replay of an earlier pair, for self-agreement.
      is_anchor (bool): has an obvious right answer, for fatigue detection.
    """

    pair_id: str
    tier: Tier
    left: ClipSpec
    right: ClipSpec
    is_repeat: bool = False
    is_anchor: bool = False

    @property
    def question(self) -> str:
        """
        Returns:
          str: the question this tier asks.
        """
        return QUESTIONS[self.tier]


@dataclass
class Pair:
    """
    A PairSpec with both clips decoded and loudness-matched, ready to play.

    Args:
      spec (PairSpec): the comparison.
      left (Clip): left clip.
      right (Clip): right clip.
    """

    spec: PairSpec
    left: Clip
    right: Clip

    def clip(self, side: Side) -> Clip:
        """
        Args:
          side (Side): "left" or "right".

        Returns:
          Clip: the clip on that side.
        """
        return self.left if side == "left" else self.right


@dataclass(frozen=True)
class Judgement:
    """
    One logged decision.

    response_ms is logged because it flags the pairs the rater found hard, and
    hard pairs are the informative ones (PLAN section 9.2).

    Args:
      pair_id (str): the comparison.
      session_id (str): rating session, for detecting drift across sessions.
      tier (Tier): rating tier.
      question (str): the exact wording shown.
      item_left (str): true item id shown on the left.
      item_right (str): true item id shown on the right.
      choice (Choice): "left", "right" or "tie".
      chosen_item_id (str): resolved winner, empty on a tie.
      response_ms (int): time from pair shown to key pressed.
      is_repeat (bool): copied from the pair.
      is_anchor (bool): copied from the pair.
      lufs_target (float): loudness both clips were matched to.
      was_live (bool): True if either clip was generated this session.
      ts (str): ISO-8601 UTC timestamp.
    """

    pair_id: str
    session_id: str
    tier: Tier
    question: str
    item_left: str
    item_right: str
    choice: Choice
    chosen_item_id: str
    response_ms: int
    is_repeat: bool
    is_anchor: bool
    lufs_target: float
    was_live: bool
    ts: str

    def to_json(self) -> dict[str, Any]:
        """
        Returns:
          dict[str, Any]: JSON-safe form for the session log.
        """
        return asdict(self) | {"tier": str(self.tier)}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Judgement:
        """
        Args:
          raw (dict[str, Any]): one line of a session log.

        Returns:
          Judgement: the rebuilt judgement.
        """
        return cls(**raw | {"tier": Tier(raw["tier"])})


__all__ = [
    "Choice",
    "Clip",
    "ClipSpec",
    "Conditioning",
    "Judgement",
    "Pair",
    "PairSpec",
    "QUESTIONS",
    "Sampling",
    "Side",
    "Tier",
]
