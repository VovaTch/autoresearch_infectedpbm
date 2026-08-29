"""
Draws the comparisons the rater is shown (PLAN sections 7.2, 7.4, 7.6).

Every design choice here is one that cannot be retrofitted without discarding
the pairs already collected:

  * Two tiers, two questions, from the first pair onwards. A harness built only
    on 10 s clips cannot see the interest axis at all.
  * Both clips in a pair share their conditioning exactly and differ only in
    Sampling.seed. Pairs that differ in conditioning teach a preference model
    the prompt distribution rather than quality.
  * Repeats and anchors are injected here, not in the UI, and carry nothing that
    distinguishes them from an ordinary pair.
  * Sides are randomized per showing, so position never encodes identity.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from ab_harness.model.types import ClipSpec, Conditioning, PairSpec, Sampling, Tier


@dataclass(frozen=True)
class TrackInfo:
    """
    The corpus facts the sampler needs, with no torch dependency.

    Args:
      track_idx (int): index into the sorted corpus and the id embedding.
      track_name (str): display name, never shown during rating.
      num_frames (int): length of the track's token stream.
      style_bounds (tuple[tuple[int, int], ...]): half-open frame ranges of the
        precomputed style windows.
    """

    track_idx: int
    track_name: str
    num_frames: int
    style_bounds: tuple[tuple[int, int], ...]


@dataclass
class SamplerCfg:
    """
    Tier mix, conditioning odds and injection rates.

    Args:
      fps (float): tokenizer frames per second.
      bulk_seconds (float): bulk-tier clip length.
      structure_seconds (float): structure-tier clip length.
      bulk_share (float): fraction of pairs drawn from the bulk tier.
      repeat_rate (float): fraction of showings that replay an earlier pair.
      anchor_rate (float): fraction that put a generation against real tokens.
      prompt_seconds (float): audio prefix length when a prefix is used.
      p_track_id (float): probability the track-id stream is kept.
      p_style (float): probability the style stream is kept.
      p_prompt (float): probability an audio prefix is used.
      cfg_strengths (tuple[float, ...]): guidance strengths to draw from; 0.0 is
        plain conditional sampling (PLAN section 11.4 puts the useful range at
        1-3, far below image-diffusion values).
      temperature (float): sampling temperature.
      top_k (int): top-k cutoff.
      top_p (float): nucleus cutoff.
    """

    fps: float = 172.265625
    bulk_seconds: float = 10.0
    structure_seconds: float = 90.0
    bulk_share: float = 0.8
    repeat_rate: float = 0.1
    anchor_rate: float = 0.05
    prompt_seconds: float = 3.0
    p_track_id: float = 0.5
    p_style: float = 0.5
    p_prompt: float = 0.5
    cfg_strengths: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0

    def frames(self, tier: Tier) -> int:
        """
        Args:
          tier (Tier): the rating tier.

        Returns:
          int: clip length in frames for that tier.
        """
        seconds = self.bulk_seconds if tier is Tier.BULK else self.structure_seconds
        return int(seconds * self.fps)


def pick_style_window(
    bounds: Sequence[tuple[int, int]], start: int, end: int, rng: random.Random
) -> int:
    """
    Choose a style window disjoint from the span being generated.

    Mirrors the rule train_ar._pick_style enforces during training (PLAN section
    11.3): a descriptor computed from the same window as the target is a
    compressed copy of the answer, so sampling must honour the same disjointness
    or it runs off-distribution and flatters itself. Re-stated here in plain
    Python to keep the model layer free of torch.

    Args:
      bounds (Sequence[tuple[int, int]]): half-open window ranges.
      start (int): span start frame.
      end (int): span end frame, exclusive.
      rng (random.Random): sampler.

    Returns:
      int: index into bounds, or -1 when there are no windows at all. Falls back
        to the farthest window when the track is too short to hold a disjoint one.
    """
    if not bounds:
        return -1
    free = [i for i, (lo, hi) in enumerate(bounds) if not (start < hi and lo < end)]
    if free:
        return rng.choice(free)
    centre = (start + end) / 2.0
    return min(
        range(len(bounds)),
        key=lambda i: -abs((bounds[i][0] + bounds[i][1]) / 2.0 - centre),
    )


def save_corpus(path: Path, tracks: Sequence[TrackInfo]) -> None:
    """
    Publish the corpus view for processes that must not import torch.

    The sampler needs track lengths and style-window bounds, which only the
    worker can read out of the token cache. Writing them here lets the UI build
    a sampler without ever loading a checkpoint.

    Args:
      path (Path): destination JSON file.
      tracks (Sequence[TrackInfo]): the corpus.
    """
    Path(path).write_text(json.dumps([asdict(t) for t in tracks], indent=2))


def load_corpus(path: Path) -> list[TrackInfo]:
    """
    Args:
      path (Path): a file written by save_corpus.

    Returns:
      list[TrackInfo]: the corpus, empty if the file does not exist yet.
    """
    if not Path(path).exists():
        return []
    return [
        TrackInfo(
            track_idx=raw["track_idx"],
            track_name=raw["track_name"],
            num_frames=raw["num_frames"],
            style_bounds=tuple(tuple(b) for b in raw["style_bounds"]),
        )
        for raw in json.loads(Path(path).read_text())
    ]


def _digest(payload: object, prefix: str) -> str:
    """
    Args:
      payload (object): JSON-serializable content to hash.
      prefix (str): human-readable id prefix.

    Returns:
      str: prefix plus 12 hex characters. Content-addressed, so the same recipe
        always resolves to the same id and re-baking never duplicates an item.
    """
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return f"{prefix}_{hashlib.blake2b(blob, digest_size=6).hexdigest()}"


class PairSampler:
    """
    Stateful source of PairSpecs.

    Args:
      tracks (Sequence[TrackInfo]): the corpus to draw spans from.
      cfg (SamplerCfg): tier mix and conditioning odds.
      checkpoint (str): tag of the checkpoint new clips come from.
      rng (random.Random | None): seeded sampler; a fresh one if omitted.
    """

    def __init__(
        self,
        tracks: Sequence[TrackInfo],
        cfg: SamplerCfg | None = None,
        checkpoint: str = "",
        rng: random.Random | None = None,
    ) -> None:
        if not tracks:
            raise ValueError("sampler needs at least one track")
        self.tracks = list(tracks)
        self.cfg = cfg or SamplerCfg()
        self.checkpoint = checkpoint
        self.rng = rng or random.Random()
        self._history: list[PairSpec] = []

    # -- drawing -------------------------------------------------------------

    def pick_tier(self) -> Tier:
        """
        Returns:
          Tier: BULK with probability cfg.bulk_share, else STRUCTURE.
        """
        return Tier.BULK if self.rng.random() < self.cfg.bulk_share else Tier.STRUCTURE

    def _pick_span(self, tier: Tier) -> tuple[TrackInfo, int, int]:
        """
        Args:
          tier (Tier): the rating tier, which fixes the clip length.

        Returns:
          tuple[TrackInfo, int, int]: track, start frame and frame count. Tracks
            too short for the tier are skipped; the clip is truncated only if no
            track in the corpus is long enough.
        """
        frames = self.cfg.frames(tier)
        eligible = [t for t in self.tracks if t.num_frames > frames + 1]
        if not eligible:
            longest = max(self.tracks, key=lambda t: t.num_frames)
            return longest, 0, min(frames, longest.num_frames)
        track = self.rng.choice(eligible)
        start = self.rng.randrange(0, track.num_frames - frames)
        return track, start, frames

    def _draw_conditioning(
        self, track: TrackInfo, start: int, frames: int
    ) -> Conditioning:
        """
        Draw one conditioning recipe; all eight stream combinations are reachable.

        Args:
          track (TrackInfo): the track the span comes from.
          start (int): span start frame.
          frames (int): span length.

        Returns:
          Conditioning: the recipe shared by both clips of the pair.
        """
        use_style = self.rng.random() < self.cfg.p_style
        window = (
            pick_style_window(track.style_bounds, start, start + frames, self.rng)
            if use_style
            else -1
        )
        prompt = (
            int(self.cfg.prompt_seconds * self.cfg.fps)
            if self.rng.random() < self.cfg.p_prompt
            else 0
        )
        return Conditioning(
            track_idx=track.track_idx,
            start_frame=start,
            use_track_id=self.rng.random() < self.cfg.p_track_id,
            use_style=use_style and window >= 0,
            style_window=window,
            prompt_frames=min(prompt, max(0, frames - 1)),
            cfg_strength=self.rng.choice(self.cfg.cfg_strengths),
        )

    def _clip(self, tier: Tier, frames: int, cond: Conditioning, seed: int) -> ClipSpec:
        """
        Args:
          tier (Tier): rating tier.
          frames (int): clip length.
          cond (Conditioning): shared recipe.
          seed (int): the one field that varies within a pair.

        Returns:
          ClipSpec: a content-addressed generated clip.
        """
        group = _digest(
            {
                "tier": str(tier),
                "frames": frames,
                "cond": asdict(cond),
                "ckpt": self.checkpoint,
            },
            "grp",
        )
        sampling = Sampling(
            seed=seed,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
        )
        item = _digest({"group": group, "sampling": asdict(sampling)}, "gen")
        return ClipSpec(
            item_id=item,
            tier=tier,
            group_id=group,
            n_frames=frames,
            conditioning=cond,
            sampling=sampling,
            generator="ar",
            checkpoint=self.checkpoint,
        )

    def _reference(self, tier: Tier, frames: int, cond: Conditioning) -> ClipSpec:
        """
        Build the real-token clip for a span, used as the anchor's known winner.

        Args:
          tier (Tier): rating tier.
          frames (int): clip length.
          cond (Conditioning): the span; conditioning fields are inert here.

        Returns:
          ClipSpec: a reference clip, decoded from the token cache rather than
            generated. Its id carries the "ref_" prefix stats.anchor_accuracy
            keys on.
        """
        span = {"track": cond.track_idx, "start": cond.start_frame, "frames": frames}
        item = _digest(span, "ref")
        return ClipSpec(
            item_id=item,
            tier=tier,
            group_id=_digest(span, "grp"),
            n_frames=frames,
            conditioning=cond,
            sampling=Sampling(seed=0, temperature=0.0),
            generator="reference",
            checkpoint="",
        )

    def blind(
        self,
        tier: Tier,
        a: ClipSpec,
        b: ClipSpec,
        is_repeat: bool = False,
        is_anchor: bool = False,
    ) -> PairSpec:
        """
        Randomize sides for a pair assembled elsewhere, e.g. from banked clips.

        Args:
          tier (Tier): rating tier.
          a (ClipSpec): one clip.
          b (ClipSpec): the other.
          is_repeat (bool): replay of an earlier comparison.
          is_anchor (bool): has an obvious right answer.

        Returns:
          PairSpec: the blinded comparison.
        """
        return self._blind(tier, a, b, is_repeat, is_anchor)

    def register(self, spec: PairSpec) -> None:
        """
        Add a comparison to the repeat pool.

        Pairs the pipeline assembles from banked clips must land here too, or
        they could never come round again as a repeat and the self-agreement
        statistic would only ever cover freshly drawn material.

        Args:
          spec (PairSpec): the comparison that was shown.
        """
        self._history.append(spec)

    def _blind(
        self, tier: Tier, a: ClipSpec, b: ClipSpec, is_repeat: bool, is_anchor: bool
    ) -> PairSpec:
        """
        Randomize sides and mint a per-showing pair id.

        The pair id is fresh even for a repeat, so the two showings are separate
        rows in the log; stats.self_agreement matches them on the unordered item
        pair instead.

        Args:
          tier (Tier): rating tier.
          a (ClipSpec): one clip.
          b (ClipSpec): the other.
          is_repeat (bool): replay of an earlier comparison.
          is_anchor (bool): has an obvious right answer.

        Returns:
          PairSpec: the blinded comparison.
        """
        left, right = (a, b) if self.rng.random() < 0.5 else (b, a)
        pair_id = _digest(
            {"l": left.item_id, "r": right.item_id, "n": self.rng.getrandbits(64)},
            "pair",
        )
        return PairSpec(
            pair_id=pair_id,
            tier=tier,
            left=left,
            right=right,
            is_repeat=is_repeat,
            is_anchor=is_anchor,
        )

    def next_spec(self, tier: Tier | None = None) -> PairSpec:
        """
        Draw the next comparison to show.

        Args:
          tier (Tier | None): force a tier instead of drawing one. Used by the
            pipeline's backfill lane, which generates structure-tier material
            ahead of time; None keeps the configured mix.

        Returns:
          PairSpec: a repeat, an anchor, or a fresh gen-vs-gen pair. Repeats are
            only possible once something has been shown; until then the roll
            falls through to a fresh pair. Forcing a tier also skips repeats,
            since a repeat needs no new tokens.
        """
        roll = self.rng.random()
        if tier is None and roll < self.cfg.repeat_rate and self._history:
            earlier = self.rng.choice(self._history)
            return self._blind(
                earlier.tier, earlier.left, earlier.right, True, earlier.is_anchor
            )

        tier = tier if tier is not None else self.pick_tier()
        track, start, frames = self._pick_span(tier)
        cond = self._draw_conditioning(track, start, frames)

        if roll < self.cfg.repeat_rate + self.cfg.anchor_rate:
            spec = self._blind(
                tier,
                self._clip(tier, frames, cond, self.rng.randrange(2**31)),
                self._reference(tier, frames, cond),
                False,
                True,
            )
        else:
            seeds = self.rng.sample(range(2**31), 2)
            spec = self._blind(
                tier,
                self._clip(tier, frames, cond, seeds[0]),
                self._clip(tier, frames, cond, seeds[1]),
                False,
                False,
            )
        self._history.append(spec)
        return spec
