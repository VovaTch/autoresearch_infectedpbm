"""
Direct preference optimization of the AR model on A/B judgements.

Implements PLAN_generative_stage.md section 7.4 -- rung 5 of the preference
ladder. Rung 1 (the A/B harness) already emits exactly what this needs: every
generated clip's tokens are banked as (T, R) int16 next to a reproducible
ClipSpec, and both sides of a comparison provably share their Conditioning
(same group_id) and differ only in Sampling.seed. That shared-conditioning
guarantee is not a nicety -- pairs that differ in conditioning teach the model
the prompt distribution rather than quality.

    uv run python train_dpo.py --config config_dpo.yaml --dry-run
    uv run python train_dpo.py --config config_dpo.yaml --smoke --min-pairs 1
    uv run python train_dpo.py --config config_dpo.yaml

Three things are deliberate and easy to get wrong:

1. Section 7.3 says preference training drifts toward boring, by construction
   and not by rater error. DiversityMonitor measures spread across samples drawn
   from one conditioning and halts the run when it falls below a fraction of its
   step-0 baseline. Diversity is invisible to a pairwise objective, so nothing
   else in this file would notice.

2. Forced prompt frames are masked out of the loss. They are identical on both
   sides of a pair, so scoring them teaches a preference for a prefix the model
   did not choose.

3. cfg_strength is a sampling-time quantity. Clips in the bank were drawn with
   guidance (2.0 or 3.0), but the policy is scored plain-conditional, because
   that is the distribution DPO updates. The mismatch is real and known.

Output checkpoints carry the ArLightningModule layout, so generate_ar.py,
ab_harness and export_onnx.py load them unchanged -- which is what lets the
result be rated against its own base in the harness that produced the data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from ab_harness.model.bank import ClipBank
from ab_harness.model.judgement import read_all
from ab_harness.model.stats import anchor_accuracy, self_agreement
from ab_harness.model.types import ClipSpec, Judgement, Tier
from ab_harness.worker.generator import ArGenerator, SampleRequest
from ab_harness.worker.service import style_vector
from generate_ar import config_from_ckpt
from train_ar import (
    ArConfig,
    ArTransformer,
    DataCfg,
    TrackTokens,
    _build_section,
    build_model,
    load_token_cache,
    prepare_grid,
)

REPO = Path(__file__).resolve().parent


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class PairsCfg:
    """
    Where the preference pairs come from and which ones survive filtering.

    Args:
      bank_root (str): the A/B bank written by ab_harness.
      reference_checkpoint (str): the AR checkpoint that is both the policy
        initialization and the frozen DPO reference.
      restrict_to_reference_checkpoint (bool): keep only pairs sampled from
        reference_checkpoint. DPO tolerates off-policy pairs and today this
        would discard a third of the data, so it defaults off.
      min_anchor_acc (float): sessions scoring below this on anchor pairs are
        discarded as fatigued (section 7.6). Sessions with no anchors pass.
      min_pairs (int): refuse to train below this many usable pairs. Section
        7.5 puts DPO signal at 200-500 pairs; --min-pairs overrides for smoke
        runs.
      val_frac (float): share of *sessions* held out. Splitting by session, not
        by row, is what makes the held-out number mean "agrees with a listening
        session it never trained on".
      tiers (list[str] | None): restrict to some tiers, or None for all.
      split_seed (int): seed for the session split.
    """

    bank_root: str = "~/.cache/infected_pbm/ab"
    reference_checkpoint: str = "saved_ar_20260829_24h/ar_frozen_0829.ckpt"
    restrict_to_reference_checkpoint: bool = False
    min_anchor_acc: float = 0.6
    min_pairs: int = 200
    val_frac: float = 0.25
    tiers: list[str] | None = None
    split_seed: int = 1234


@dataclass
class DpoCfg:
    """
    The objective.

    Args:
      beta (float): DPO temperature; section 7.4 says around 0.1.
      label_smoothing (float): conservative-DPO mixing, 0.0 is plain DPO. A
        non-zero value assumes that share of the labels are flipped, which is
        one way to spend the self-agreement number on the objective.
      sft_weight (float): weight of an added NLL term on the winner. Anchors
        the policy against drift at the cost of pulling toward the winners'
        own distribution; 0.0 is plain DPO.
      length_normalize (bool): divide sequence log-probabilities by their
        supervised token count. Pairs are equal-length by construction, so this
        is off by default and exists for pairs that stop being so.
      trainable_blocks (int): top transformer blocks left trainable, along with
        norm_out and the output heads; everything below is frozen. At a few
        hundred pairs, full-model DPO is the overfitting risk. Set it to
        n_layers for the full-model recipe.
      cache_ref (bool): memoize reference log-probabilities per (item, window).
        The bulk tier fits in one window, so its references are computed once
        and reused for the rest of the run.
    """

    beta: float = 0.1
    label_smoothing: float = 0.0
    sft_weight: float = 0.0
    length_normalize: bool = False
    trainable_blocks: int = 4
    cache_ref: bool = True


@dataclass
class DpoTrainCfg:
    """
    Optimizer, schedule and the diversity gate.

    Args:
      epochs (int): passes over the pair set; a ceiling, not the schedule,
        whenever max_hours is set.
      max_hours (float): wall-clock budget. 0 disables, and epochs alone stops
        the run. At a few hundred pairs an epoch is seconds, so a run of any
        length is many passes over the same pairs -- the budget is a stopping
        rule, not a substitute for early stopping.
      batch_pairs (int): pairs per step; each costs four forwards (two sides,
        policy and reference) so this is four times an AR batch of the same
        number.
      accumulate (int): gradient accumulation steps.
      lr (float): constant learning rate. DPO on a few hundred pairs is a
        nudge, not a training run; no one-cycle schedule.
      weight_decay (float): AdamW decay on matmul weights.
      grad_clip (float): gradient norm clip.
      precision (str): Lightning precision. Log-probabilities are always summed
        in float32 regardless, because a margin is a difference of large sums.
      num_workers (int): dataloader workers; 0 keeps the token corpus in one
        process instead of forking a copy per worker.
      seed (int): global seed.
      device (str): torch device for the model and the diversity sampler.
      save_path (str): checkpoint directory; <date> expands to today.
      diversity_every (int): steps between diversity probes; 0 disables the
        gate entirely, which section 7.3 advises against.
      diversity_clips (int): candidates drawn per probe from one conditioning.
      diversity_seconds (float): probe clip length.
      diversity_gate (float): halt when diversity falls below this fraction of
        its step-0 baseline.
    """

    epochs: int = 8
    max_hours: float = 0.0
    batch_pairs: int = 2
    accumulate: int = 4
    lr: float = 2.0e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    precision: str = "bf16-mixed"
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda:0"
    save_path: str = "saved_dpo_<date>/"
    diversity_every: int = 25
    diversity_clips: int = 6
    diversity_seconds: float = 10.0
    diversity_gate: float = 0.8


@dataclass
class DpoConfig:
    """
    Top-level config, one block per YAML section.

    Args:
      pairs (PairsCfg): data selection.
      dpo (DpoCfg): the objective.
      train (DpoTrainCfg): optimizer and gate.
    """

    pairs: PairsCfg = field(default_factory=PairsCfg)
    dpo: DpoCfg = field(default_factory=DpoCfg)
    train: DpoTrainCfg = field(default_factory=DpoTrainCfg)


def load_config(path: str | Path) -> DpoConfig:
    """
    Read a DPO config from YAML, rejecting unknown keys.

    Args:
      path (str | Path): path to the .yaml file.

    Returns:
      DpoConfig: fully populated config with defaults filled in.
    """
    with open(path, "r") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    if unknown := set(raw) - {"pairs", "dpo", "train"}:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")
    return DpoConfig(
        pairs=_build_section(PairsCfg, raw.get("pairs"), "pairs"),
        dpo=_build_section(DpoCfg, raw.get("dpo"), "dpo"),
        train=_build_section(DpoTrainCfg, raw.get("train"), "train"),
    )


# ===========================================================================
# Pair extraction
# ===========================================================================


@dataclass(frozen=True)
class PrefPair:
    """
    One human preference over two clips that share their conditioning.

    Args:
      pair_id (str): the comparison this came from.
      session_id (str): rating session, the unit the train/val split uses.
      tier (Tier): which question was asked.
      group_id (str): shared conditioning group; equal for both sides by
        construction (section 7.4).
      winner (ClipSpec): the chosen clip.
      loser (ClipSpec): the rejected clip.
    """

    pair_id: str
    session_id: str
    tier: Tier
    group_id: str
    winner: ClipSpec
    loser: ClipSpec

    @property
    def n_frames(self) -> int:
        """
        Returns:
          int: clip length in tokenizer frames, equal on both sides.
        """
        return self.winner.n_frames


@dataclass
class FilterReport:
    """
    Why each judgement did or did not become a training pair.

    Args:
      total (int): judgements read.
      kept (int): pairs surviving every stage.
      dropped (Counter[str]): count per drop reason.
      sessions (dict[str, int]): kept pairs per session.
      tiers (Counter[str]): kept pairs per tier.
      failed_sessions (list[str]): sessions discarded on anchor accuracy.
    """

    total: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    sessions: dict[str, int] = field(default_factory=dict)
    tiers: Counter = field(default_factory=Counter)
    failed_sessions: list[str] = field(default_factory=list)

    def render(self) -> str:
        """
        Returns:
          str: a human-readable summary block.
        """
        lines = [f"judgements read      {self.total}"]
        for reason, count in sorted(self.dropped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  dropped {reason:<28s} {count}")
        lines.append(f"usable pairs         {self.kept}")
        for tier, count in sorted(self.tiers.items()):
            lines.append(f"  tier {tier:<24s} {count}")
        for session, count in sorted(self.sessions.items()):
            lines.append(f"  session {session:<21s} {count}")
        if self.failed_sessions:
            lines.append(f"  anchor-failed sessions   {self.failed_sessions}")
        return "\n".join(lines)


def _session_anchor_pass(
    judgements: Sequence[Judgement], min_acc: float
) -> tuple[set[str], list[str]]:
    """
    Split sessions on anchor accuracy (section 7.6).

    An anchor puts a generation against real tokens, so the reference is the
    expected winner. A session that fails them was a fatigued session and its
    other rows are suspect too. Sessions with no anchors cannot be judged and
    are kept.

    Args:
      judgements (Sequence[Judgement]): every decision.
      min_acc (float): minimum share of anchors answered as expected.

    Returns:
      tuple[set[str], list[str]]: sessions to keep, and those discarded.
    """
    by_session: dict[str, list[Judgement]] = defaultdict(list)
    for judgement in judgements:
        by_session[judgement.session_id].append(judgement)
    keep, failed = set(), []
    for session, rows in by_session.items():
        correct, total = anchor_accuracy(rows)
        if total and correct / total < min_acc:
            failed.append(session)
        else:
            keep.add(session)
    return keep, sorted(failed)


def _unordered(judgement: Judgement) -> tuple[str, str]:
    """
    Args:
      judgement (Judgement): a decision.

    Returns:
      tuple[str, str]: the two item ids, order-independent, so a repeat shown
        with the sides swapped still matches its original.
    """
    return tuple(sorted((judgement.item_left, judgement.item_right)))  # type: ignore[return-value]


def load_preference_pairs(
    bank: ClipBank, cfg: PairsCfg
) -> tuple[list[PrefPair], FilterReport]:
    """
    Turn banked judgements into DPO training pairs.

    Args:
      bank (ClipBank): the A/B bank.
      cfg (PairsCfg): filtering settings.

    Returns:
      tuple[list[PrefPair], FilterReport]: surviving pairs and the audit trail.
    """
    judgements = read_all(bank.sessions_dir)
    report = FilterReport(total=len(judgements))
    live_sessions, failed = _session_anchor_pass(judgements, cfg.min_anchor_acc)
    report.failed_sessions = failed

    wanted = {Tier(t) for t in cfg.tiers} if cfg.tiers else None

    # A repeat the rater answered both ways carries no signal, so a comparison
    # is only usable when every showing of it agreed (section 7.6).
    verdicts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for judgement in judgements:
        if judgement.session_id in live_sessions and judgement.chosen_item_id:
            verdicts[_unordered(judgement)].add(judgement.chosen_item_id)

    pairs: dict[tuple[str, str], PrefPair] = {}
    for judgement in judgements:
        key = _unordered(judgement)
        if judgement.session_id not in live_sessions:
            report.dropped["anchor-failed session"] += 1
            continue
        if judgement.choice == "tie":
            report.dropped["tie"] += 1
            continue
        if judgement.is_anchor:
            # The loser of an anchor is real audio, not a policy sample;
            # training on it is SFT-on-real wearing a DPO costume.
            report.dropped["anchor pair"] += 1
            continue
        if len(verdicts[key]) > 1:
            report.dropped["contradictory repeat"] += 1
            continue
        if key in pairs:
            report.dropped["consistent repeat"] += 1
            continue
        if not (bank.has(judgement.item_left) and bank.has(judgement.item_right)):
            report.dropped["missing tokens"] += 1
            continue
        left, right = bank.spec(judgement.item_left), bank.spec(judgement.item_right)
        if left.group_id != right.group_id:
            report.dropped["conditioning differs"] += 1
            continue
        if left.n_frames != right.n_frames:
            report.dropped["length differs"] += 1
            continue
        if left.is_reference or right.is_reference:
            report.dropped["reference clip"] += 1
            continue
        if wanted is not None and judgement.tier not in wanted:
            report.dropped["tier excluded"] += 1
            continue
        if cfg.restrict_to_reference_checkpoint and (
            left.checkpoint != cfg.reference_checkpoint
            or right.checkpoint != cfg.reference_checkpoint
        ):
            report.dropped["off-policy checkpoint"] += 1
            continue
        winner = left if judgement.chosen_item_id == left.item_id else right
        loser = right if winner is left else left
        pairs[key] = PrefPair(
            pair_id=judgement.pair_id,
            session_id=judgement.session_id,
            tier=judgement.tier,
            group_id=left.group_id,
            winner=winner,
            loser=loser,
        )

    kept = list(pairs.values())
    report.kept = len(kept)
    report.tiers = Counter(str(p.tier) for p in kept)
    report.sessions = dict(Counter(p.session_id for p in kept))
    return kept, report


def split_by_session(
    pairs: Sequence[PrefPair], val_frac: float, seed: int
) -> tuple[list[PrefPair], list[PrefPair]]:
    """
    Hold out whole sessions, never individual rows.

    A row-level split leaks: both showings of a repeated comparison, and every
    pair drawn during one sitting, share the rater's state at that moment.

    Args:
      pairs (Sequence[PrefPair]): every usable pair.
      val_frac (float): target share of sessions held out.
      seed (int): split seed.

    Returns:
      tuple[list[PrefPair], list[PrefPair]]: train and val pairs. Val is empty
        when there is only one session, since holding it out leaves nothing to
        train on.
    """
    sessions = sorted({p.session_id for p in pairs})
    if len(sessions) < 2 or val_frac <= 0:
        return list(pairs), []
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    n_val = max(1, min(len(sessions) - 1, round(val_frac * len(sessions))))
    held = set(shuffled[:n_val])
    train = [p for p in pairs if p.session_id not in held]
    val = [p for p in pairs if p.session_id in held]
    return train, val


# ===========================================================================
# Dataset
# ===========================================================================


class PreferenceDataset(Dataset):
    """
    Winner/loser token windows with the conditioning that produced them.

    Both sides of a pair are scored on the same window of the same length, so
    the log-probability difference the loss consumes is a like-for-like one.
    Clips longer than the model's training context (the 90 s structure tier is
    four times it) are scored on one randomly drawn crop_frames window per
    epoch: scoring them whole would extrapolate rotary positions far past
    anything training saw, which is the same reason the sampler re-primes.

    Args:
      pairs (Sequence[PrefPair]): the comparisons.
      tracks (dict[int, TrackTokens]): corpus by track index, for style vectors.
      bank (ClipBank): token store.
      crop_frames (int): the checkpoint's training context in frames.
      seed (int): base seed for window draws.
      fixed_window (bool): always take the first window instead of a random
        one. Validation uses it so the held-out number does not wobble.
    """

    def __init__(
        self,
        pairs: Sequence[PrefPair],
        tracks: dict[int, TrackTokens],
        bank: ClipBank,
        crop_frames: int,
        seed: int = 0,
        fixed_window: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        self.tracks = tracks
        self.bank = bank
        self.crop_frames = crop_frames
        self.seed = seed
        self.fixed_window = fixed_window
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        Args:
          epoch (int): redraws the windows so long clips are seen in full over
            several epochs rather than through one fixed keyhole.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """
        Returns:
          int: number of pairs.
        """
        return len(self.pairs)

    def scored_length(self, index: int) -> int:
        """
        Args:
          index (int): pair index.

        Returns:
          int: frames actually scored, which the batch sampler buckets on.
        """
        return min(self.pairs[index].n_frames, self.crop_frames)

    def _window(self, pair: PrefPair, index: int) -> int:
        """
        Args:
          pair (PrefPair): the comparison.
          index (int): pair index, part of the window seed.

        Returns:
          int: first frame of the scored window, shared by both sides.
        """
        slack = pair.n_frames - self.crop_frames
        if slack <= 0 or self.fixed_window:
            return 0
        return random.Random(f"{self.seed}:{self.epoch}:{index}").randrange(slack + 1)

    def _side(self, spec: ClipSpec, start: int, length: int) -> dict[str, Any]:
        """
        Args:
          spec (ClipSpec): the clip.
          start (int): first frame of the window.
          length (int): frames to score.

        Returns:
          dict[str, Any]: tokens, the score mask and the conditioning tensors.
        """
        tokens = torch.from_numpy(
            np.asarray(self.bank.tokens(spec.item_id), dtype=np.int64)
        )
        window = tokens[start : start + length]
        if window.shape[0] < length:  # a clip banked shorter than its spec
            pad = length - window.shape[0]
            window = torch.cat([window, window[-1:].expand(pad, -1)], dim=0)
        # Forced prompt frames are identical on both sides; scoring them would
        # teach a preference for a prefix the model did not choose.
        absolute = torch.arange(start, start + length)
        score = absolute >= spec.conditioning.prompt_frames
        cond = spec.conditioning
        return {
            "item_id": spec.item_id,
            "tokens": window,
            "score": score,
            "track_idx": torch.tensor(cond.track_idx, dtype=torch.long),
            "style": style_vector(self.tracks[cond.track_idx], spec).float(),
            "drop_id": torch.tensor(not cond.use_track_id),
            "drop_style": torch.tensor(not cond.use_style),
            "start": torch.tensor(start, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Args:
          index (int): pair index.

        Returns:
          dict[str, Any]: {"win": side, "lose": side, "pair_id", "tier"}.
        """
        pair = self.pairs[index]
        length = self.scored_length(index)
        start = self._window(pair, index)
        return {
            "pair_id": pair.pair_id,
            "tier": str(pair.tier),
            "win": self._side(pair.winner, start, length),
            "lose": self._side(pair.loser, start, length),
        }


class LengthBucketSampler(Sampler[list[int]]):
    """
    Batch only pairs of equal scored length.

    The two tiers differ by an order of magnitude in length, so mixing them in
    one batch would mean padding a 10 s clip out to a 90 s window and masking
    almost all of it -- paying for context nobody scores.

    Args:
      dataset (PreferenceDataset): the pairs.
      batch_size (int): pairs per batch.
      shuffle (bool): reshuffle within buckets each epoch.
      seed (int): shuffle seed.
    """

    def __init__(
        self,
        dataset: PreferenceDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        Args:
          epoch (int): epoch number, mixed into the shuffle seed.
        """
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        """
        Returns:
          list[list[int]]: index batches, none mixing two lengths.
        """
        buckets: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            buckets[self.dataset.scored_length(index)].append(index)
        batches: list[list[int]] = []
        rng = random.Random(f"{self.seed}:{self.epoch}")
        for _, indices in sorted(buckets.items()):
            order = indices[:]
            if self.shuffle:
                rng.shuffle(order)
            batches += [
                order[i : i + self.batch_size]
                for i in range(0, len(order), self.batch_size)
            ]
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        """
        Returns:
          Iterator[list[int]]: one list of pair indices per batch.
        """
        return iter(self._batches())

    def __len__(self) -> int:
        """
        Returns:
          int: number of batches in an epoch.
        """
        return len(self._batches())


def collate_pairs(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Stack a bucket-aligned batch of pairs.

    Args:
      items (Sequence[dict[str, Any]]): dataset items of equal scored length.

    Returns:
      dict[str, Any]: {"win": stacked side, "lose": stacked side, "pair_id",
        "tier"}.
    """
    out: dict[str, Any] = {
        "pair_id": [i["pair_id"] for i in items],
        "tier": [i["tier"] for i in items],
    }
    for side in ("win", "lose"):
        keys = [k for k in items[0][side] if k != "item_id"]
        out[side] = {k: torch.stack([i[side][k] for i in items]) for k in keys}
        out[side]["item_id"] = [i[side]["item_id"] for i in items]
    return out


# ===========================================================================
# Objective
# ===========================================================================


def sequence_logprob(
    model: ArTransformer, side: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sum log p(token) over every scored position of a clip.

    Scored positions are gathered before the softmax so only the supervised
    rows of the (B, L, R, V) logit tensor are ever materialized at float32.

    Args:
      model (ArTransformer): policy or reference.
      side (dict[str, Any]): one collated side, with tokens (B, T, R), score
        (B, T), track_idx (B,), style (B, C), drop_id (B,), drop_style (B,).

    Returns:
      tuple[torch.Tensor, torch.Tensor]: (B,) float32 sequence log-probability
        and (B,) float32 count of supervised positions.
    """
    tokens = side["tokens"].long()
    inputs, targets, mask = prepare_grid(
        tokens, model.pad_id, model.frames_per_pos, side["score"]
    )
    logits = model(
        inputs,
        side["track_idx"].long(),
        side["style"].float(),
        side["drop_id"].bool(),
        side["drop_style"].bool(),
    )
    batch = tokens.shape[0]
    rows = torch.arange(batch, device=tokens.device)
    total = torch.zeros(batch, dtype=torch.float32, device=tokens.device)
    counts = torch.zeros(batch, dtype=torch.float32, device=tokens.device)
    for depth in range(model.num_rq):
        keep = mask[:, :, depth]
        picked = logits[:, :, depth][keep]
        wanted = targets[:, :, depth][keep]
        if picked.numel() == 0:
            continue
        logprob = -F.cross_entropy(picked.float(), wanted, reduction="none")
        where = rows[:, None].expand_as(keep)[keep]
        total = total.index_add(0, where, logprob)
        counts = counts.index_add(0, where, torch.ones_like(logprob))
    return total, counts


def dpo_loss(
    lp_win: torch.Tensor,
    lp_lose: torch.Tensor,
    ref_win: torch.Tensor,
    ref_lose: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    The DPO objective over sequence log-probabilities (section 7.4).

    Args:
      lp_win (torch.Tensor): (B,) policy log-prob of the chosen clip.
      lp_lose (torch.Tensor): (B,) policy log-prob of the rejected clip.
      ref_win (torch.Tensor): (B,) reference log-prob of the chosen clip.
      ref_lose (torch.Tensor): (B,) reference log-prob of the rejected clip.
      beta (float): DPO temperature.
      label_smoothing (float): assumed share of flipped labels; 0.0 is plain
        DPO.

    Returns:
      tuple[torch.Tensor, dict[str, torch.Tensor]]: scalar loss and metrics
        (margin, reward_win, reward_lose, acc).
    """
    reward_win = beta * (lp_win - ref_win)
    reward_lose = beta * (lp_lose - ref_lose)
    margin = reward_win - reward_lose
    loss = -(
        (1.0 - label_smoothing) * F.logsigmoid(margin)
        + label_smoothing * F.logsigmoid(-margin)
    ).mean()
    metrics = {
        "margin": margin.mean().detach(),
        "reward_win": reward_win.mean().detach(),
        "reward_lose": reward_lose.mean().detach(),
        "acc": (margin > 0).float().mean().detach(),
    }
    return loss, metrics


def freeze_trunk(model: ArTransformer, trainable_blocks: int) -> tuple[int, int]:
    """
    Leave only the top N blocks, the output norm and the heads trainable.

    Args:
      model (ArTransformer): the policy.
      trainable_blocks (int): blocks to keep trainable, counted from the top.
        Values at or above the depth train everything.

    Returns:
      tuple[int, int]: trainable and total parameter counts.
    """
    depth = len(model.blocks)
    if trainable_blocks < depth:
        for param in model.parameters():
            param.requires_grad_(False)
        for block in model.blocks[depth - max(0, trainable_blocks) :]:
            for param in block.parameters():
                param.requires_grad_(True)
        for module in (model.norm_out, model.heads):
            for param in module.parameters():
                param.requires_grad_(True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ===========================================================================
# Lightning module
# ===========================================================================


class DpoLightningModule(L.LightningModule):
    """
    Policy and frozen reference, scored side by side.

    Args:
      policy (ArTransformer): the model being updated.
      reference (ArTransformer): a frozen copy of the same checkpoint.
      cfg (DpoConfig): full config.
    """

    def __init__(
        self, policy: ArTransformer, reference: ArTransformer, cfg: DpoConfig
    ) -> None:
        super().__init__()
        self.policy = policy
        self.reference = reference.requires_grad_(False).eval()
        self.cfg = cfg
        self._ref_cache: dict[tuple[str, int], float] = {}

    def train(self, mode: bool = True):  # type: ignore[override]
        """
        Keep the reference in eval no matter what Lightning does to the module.

        Args:
          mode (bool): training mode for the policy.

        Returns:
          DpoLightningModule: self.
        """
        super().train(mode)
        self.reference.eval()
        return self

    def _normalize(self, logprob: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
          logprob (torch.Tensor): (B,) summed log-probability.
          counts (torch.Tensor): (B,) supervised position count.

        Returns:
          torch.Tensor: (B,) per-token log-probability when length_normalize is
            set, otherwise the sum unchanged.
        """
        if not self.cfg.dpo.length_normalize:
            return logprob
        return logprob / counts.clamp(min=1.0)

    @torch.no_grad()
    def _reference_logprob(self, side: dict[str, Any]) -> torch.Tensor:
        """
        Reference log-probabilities, memoized per (item, window) when enabled.

        Args:
          side (dict[str, Any]): one collated side.

        Returns:
          torch.Tensor: (B,) reference log-probability, already normalized.
        """
        keys = [
            (item, int(start))
            for item, start in zip(side["item_id"], side["start"].tolist())
        ]
        if self.cfg.dpo.cache_ref and all(k in self._ref_cache for k in keys):
            return torch.tensor(
                [self._ref_cache[k] for k in keys],
                dtype=torch.float32,
                device=side["tokens"].device,
            )
        logprob, counts = sequence_logprob(self.reference, side)
        value = self._normalize(logprob, counts)
        if self.cfg.dpo.cache_ref:
            for key, item in zip(keys, value.tolist()):
                self._ref_cache[key] = item
        return value

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): collated pair batch.
          stage (str): "train" or "val".

        Returns:
          torch.Tensor: scalar DPO loss.
        """
        lp_win, count_win = sequence_logprob(self.policy, batch["win"])
        lp_lose, count_lose = sequence_logprob(self.policy, batch["lose"])
        lp_win = self._normalize(lp_win, count_win)
        lp_lose = self._normalize(lp_lose, count_lose)
        ref_win = self._reference_logprob(batch["win"])
        ref_lose = self._reference_logprob(batch["lose"])

        loss, metrics = dpo_loss(
            lp_win,
            lp_lose,
            ref_win,
            ref_lose,
            self.cfg.dpo.beta,
            self.cfg.dpo.label_smoothing,
        )
        if self.cfg.dpo.sft_weight:
            # Anchors the policy to the winners' own likelihood, the standard
            # guard against DPO walking away from both sides of every pair.
            loss = (
                loss
                - self.cfg.dpo.sft_weight * (lp_win / count_win.clamp(min=1.0)).mean()
            )

        batch_size = lp_win.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=name == "acc",
                batch_size=batch_size,
            )
        # How far the policy has moved from the reference, in nats per token.
        drift = ((lp_win - ref_win).abs() / count_win.clamp(min=1.0)).mean()
        self.log(f"{stage}/ref_drift", drift.detach(), batch_size=batch_size)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): collated pair batch.
          batch_idx (int): index within the epoch.

        Returns:
          torch.Tensor: training loss.
        """
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): collated pair batch from held-out sessions.
          batch_idx (int): index within the epoch.

        Returns:
          torch.Tensor: validation loss. val/acc is the number that matters:
            the share of unseen human calls the policy now agrees with.
        """
        return self._step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Returns:
          torch.optim.Optimizer: AdamW over the unfrozen parameters only,
            decaying matmul weights and nothing else.
        """
        decay, no_decay = [], []
        for param in self.policy.parameters():
            if param.requires_grad:
                (decay if param.ndim >= 2 else no_decay).append(param)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.train.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.train.lr,
            betas=(0.9, 0.95),
        )


# ===========================================================================
# Diversity gate (section 7.3)
# ===========================================================================


@dataclass
class DiversityReading:
    """
    One probe of how much the policy still varies.

    Args:
      disagreement (float): mean share of frames where two candidates drawn
        from the same conditioning chose different codes.
      entropy (float): unigram code entropy over the drawn tokens, normalized
        by log(num_tokens).
      used (float): mean distinct codes used per depth.
    """

    disagreement: float
    entropy: float
    used: float


def measure_diversity(
    samples: Sequence[torch.Tensor], num_tokens: int
) -> DiversityReading:
    """
    Score a set of candidates drawn from one conditioning.

    Section 7.3 asks for mean pairwise distance in encoder feature space; token
    disagreement is the proxy that needs no decoder inside the training loop,
    and it moves for the same reason -- a policy narrowing its distribution
    stops disagreeing with itself. Swapping in encoder features later changes
    only this function.

    Args:
      samples (Sequence[torch.Tensor]): (T, R) int64 code grids, same length.
      num_tokens (int): codebook size per level, for entropy normalization.

    Returns:
      DiversityReading: the three numbers, all higher-is-more-varied.
    """
    if len(samples) < 2:
        return DiversityReading(float("nan"), float("nan"), float("nan"))
    stacked = torch.stack([s.long() for s in samples])  # (K, T, R)
    pairs = [
        (stacked[i] != stacked[j]).float().mean().item()
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
    ]
    entropies, used = [], []
    for depth in range(stacked.shape[-1]):
        counts = torch.bincount(stacked[:, :, depth].reshape(-1), minlength=num_tokens)
        probs = counts.float() / counts.sum().clamp(min=1)
        nonzero = probs[probs > 0]
        entropies.append(float(-(nonzero * nonzero.log()).sum() / math.log(num_tokens)))
        used.append(float((counts > 0).sum()))
    return DiversityReading(
        disagreement=float(np.mean(pairs)),
        entropy=float(np.mean(entropies)),
        used=float(np.mean(used)),
    )


class DiversityMonitor(L.Callback):
    """
    Sample from one conditioning periodically and halt if variety collapses.

    Pairwise preference data cannot see diversity: variety is a property across
    samples and every comparison sees exactly two, so DPO's log-probability
    push is free to narrow the distribution while every individual sample still
    wins its pair. This callback is the only thing in the run that would notice.

    Args:
      generator (ArGenerator): sampler wrapping the policy being trained.
      request (SampleRequest): the conditioning to draw from; seeds are varied.
      clips (int): candidates per probe.
      every (int): steps between probes; 0 disables.
      gate (float): halt when disagreement drops below this fraction of the
        step-0 baseline.
      num_tokens (int): codebook size, for entropy normalization.
    """

    def __init__(
        self,
        generator: ArGenerator,
        request: SampleRequest,
        clips: int = 6,
        every: int = 25,
        gate: float = 0.8,
        num_tokens: int = 2048,
    ) -> None:
        self.generator = generator
        self.request = request
        self.clips = clips
        self.every = every
        self.gate = gate
        self.num_tokens = num_tokens
        self.baseline: DiversityReading | None = None

    def _probe(self, module: L.LightningModule) -> DiversityReading:
        """
        Args:
          module (L.LightningModule): the DPO module, put in eval for the draw.

        Returns:
          DiversityReading: this probe's numbers.
        """
        was_training = module.training
        module.eval()
        requests = [
            SampleRequest(**{**asdict_shallow(self.request), "seed": index})
            for index in range(self.clips)
        ]
        with torch.no_grad():
            samples = self.generator.sample_batch(requests)
        if was_training:
            module.train()
        return measure_diversity(samples, self.num_tokens)

    def _record(self, trainer: L.Trainer, module: L.LightningModule, tag: str) -> None:
        """
        Args:
          trainer (L.Trainer): the running trainer.
          module (L.LightningModule): the DPO module.
          tag (str): log prefix qualifier.
        """
        reading = self._probe(module)
        # Logged through the logger rather than module.log: probes fire from
        # on_train_start too, where the loop's result collection is not open.
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "diversity/disagreement": reading.disagreement,
                    "diversity/entropy": reading.entropy,
                    "diversity/codes_used": reading.used,
                },
                step=trainer.global_step,
            )
        if self.baseline is None:
            self.baseline = reading
            print(
                f"  diversity baseline: disagreement {reading.disagreement:.4f} "
                f"entropy {reading.entropy:.4f} codes {reading.used:.1f}"
            )
            return
        floor = self.gate * self.baseline.disagreement
        if reading.disagreement < floor:
            print(
                f"  DIVERSITY GATE ({tag}): disagreement {reading.disagreement:.4f} "
                f"< {floor:.4f} ({self.gate:g} x baseline "
                f"{self.baseline.disagreement:.4f}) -- stopping"
            )
            trainer.should_stop = True

    def on_train_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """
        Args:
          trainer (L.Trainer): the running trainer.
          module (L.LightningModule): the DPO module.
        """
        if self.every:
            self._record(trainer, module, "baseline")

    def on_train_batch_end(
        self, trainer: L.Trainer, module: L.LightningModule, *args: Any
    ) -> None:
        """
        Args:
          trainer (L.Trainer): the running trainer.
          module (L.LightningModule): the DPO module.
          *args (Any): outputs, batch and index, unused.
        """
        step = trainer.global_step
        if self.every and step and step % self.every == 0:
            self._record(trainer, module, f"step {step}")


def asdict_shallow(request: SampleRequest) -> dict[str, Any]:
    """
    Copy a SampleRequest's fields without recursing into its tensors.

    dataclasses.asdict deep-copies, which would clone the style tensor on every
    probe; this keeps the same tensor and only rebinds the seed.

    Args:
      request (SampleRequest): the template.

    Returns:
      dict[str, Any]: field name to value.
    """
    return {f: getattr(request, f) for f in SampleRequest.__dataclass_fields__}


# ===========================================================================
# Checkpointing
# ===========================================================================


class ArCheckpointWriter(L.Callback):
    """
    Write checkpoints in the ArLightningModule layout, not the DPO one.

    A plain Lightning save would carry "policy." and "reference." prefixes and a
    second copy of the weights, and nothing downstream could load it. Writing
    the AR layout instead is what lets the result be sampled by generate_ar.py
    and rated in the harness that produced the pairs.

    Args:
      save_dir (Path): destination directory.
      ar_cfg (ArConfig): the config the checkpoint was trained under.
      monitor (str): validation metric to select the best epoch on.
    """

    def __init__(
        self, save_dir: Path, ar_cfg: ArConfig, monitor: str = "val/acc"
    ) -> None:
        self.save_dir = Path(save_dir)
        self.ar_cfg = ar_cfg
        self.monitor = monitor
        self.best = -float("inf")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, module: DpoLightningModule, trainer: L.Trainer, name: str) -> Path:
        """
        Args:
          module (DpoLightningModule): the module holding the policy.
          trainer (L.Trainer): the running trainer, for epoch and step.
          name (str): file name.

        Returns:
          Path: the written checkpoint.
        """
        path = self.save_dir / name
        torch.save(
            {
                "state_dict": {
                    f"model.{k}": v.detach().cpu()
                    for k, v in module.policy.state_dict().items()
                },
                "hyper_parameters": {
                    "config": asdict(self.ar_cfg),
                    "shuffle_cond": False,
                },
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "dpo_config": asdict(module.cfg),
            },
            path,
        )
        return path

    def on_validation_epoch_end(
        self, trainer: L.Trainer, module: L.LightningModule
    ) -> None:
        """
        Args:
          trainer (L.Trainer): the running trainer.
          module (L.LightningModule): the DPO module.
        """
        assert isinstance(module, DpoLightningModule)
        if trainer.sanity_checking:
            return
        score = trainer.callback_metrics.get(self.monitor)
        if score is not None and float(score) > self.best:
            self.best = float(score)
            self._write(module, trainer, "dpo_best.ckpt")

    def on_train_epoch_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """
        Args:
          trainer (L.Trainer): the running trainer.
          module (L.LightningModule): the DPO module.
        """
        assert isinstance(module, DpoLightningModule)
        self._write(module, trainer, "dpo_latest.ckpt")


# ===========================================================================
# Entry points
# ===========================================================================


def load_reference(
    cfg: DpoConfig,
) -> tuple[
    ArTransformer, ArTransformer, ArConfig, dict[int, TrackTokens], dict[str, Any], Path
]:
    """
    Build the policy, the frozen reference and the corpus they condition on.

    Args:
      cfg (DpoConfig): full config.

    Returns:
      tuple: policy, reference, the checkpoint's ArConfig, tracks by index,
        the token-cache manifest and the cache directory.
    """
    ckpt_path = REPO / cfg.pairs.reference_checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ar_cfg = config_from_ckpt(ckpt)

    cache_root = Path(ar_cfg.tokenizer.cache_root).expanduser()
    caches = sorted(cache_root.glob("tokens_*"))
    if not caches:
        raise FileNotFoundError(f"no token cache under {cache_root}")
    cache_dir = caches[-1]
    # single_track would hide most of the corpus, and specs reference all of it
    data_cfg = DataCfg(**{**vars(ar_cfg.data), "single_track": None})
    tracks, manifest = load_token_cache(cache_dir, data_cfg)

    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    models = []
    for _ in range(2):
        model = build_model(ar_cfg, tracks, manifest)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"  WARN missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}"
            )
        models.append(model)
    return (
        models[0],
        models[1],
        ar_cfg,
        {t.track_idx: t for t in tracks},
        manifest,
        cache_dir,
    )


def diversity_request(
    pairs: Sequence[PrefPair],
    tracks: dict[int, TrackTokens],
    seconds: float,
    fps: float,
) -> SampleRequest:
    """
    Fix one conditioning to probe diversity from, for the whole run.

    Drawn without a prompt: section 7.4 asks that unconditional samples keep
    being listened to, and a shared forced prefix would inflate agreement
    between candidates for reasons that have nothing to do with the policy.

    Args:
      pairs (Sequence[PrefPair]): pairs to borrow a conditioning from.
      tracks (dict[int, TrackTokens]): corpus by track index.
      seconds (float): probe clip length.
      fps (float): tokenizer frames per second.

    Returns:
      SampleRequest: the probe template; only its seed varies.
    """
    spec = pairs[0].winner
    cond = spec.conditioning
    return SampleRequest(
        track_idx=cond.track_idx,
        style=style_vector(tracks[cond.track_idx], spec).float(),
        use_track_id=cond.use_track_id,
        use_style=cond.use_style,
        frames=int(seconds * fps),
        prompt=None,
        temperature=spec.sampling.temperature,
        top_k=spec.sampling.top_k,
        top_p=spec.sampling.top_p,
        cfg_strength=cond.cfg_strength,
        seed=0,
    )


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_dpo.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the pair yield and the rater diagnostics, then exit",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="one epoch, tiny batch, frequent probes"
    )
    parser.add_argument(
        "--min-pairs", type=int, default=None, help="override pairs.min_pairs"
    )
    parser.add_argument("--device", default=None, help="override train.device")
    return parser.parse_args()


def main() -> None:
    """Run the pair report, then DPO training unless --dry-run."""
    args = parse_args()
    cfg = load_config(args.config)
    if args.min_pairs is not None:
        cfg.pairs.min_pairs = args.min_pairs
    if args.device is not None:
        cfg.train.device = args.device
    if args.smoke:
        cfg.train.epochs = 1
        cfg.train.batch_pairs = 1
        cfg.train.accumulate = 1
        cfg.train.diversity_every = max(1, cfg.train.diversity_every // 5)
        cfg.train.diversity_clips = min(cfg.train.diversity_clips, 3)

    L.seed_everything(cfg.train.seed, workers=True)
    bank = ClipBank(Path(cfg.pairs.bank_root).expanduser())
    pairs, report = load_preference_pairs(bank, cfg.pairs)

    judgements = read_all(bank.sessions_dir)
    print(report.render())
    for tier in Tier:
        agreement = self_agreement(judgements, tier)
        print(
            f"self-agreement {str(tier):<10s} {agreement.rate:.3f} "
            f"over {agreement.compared} repeats ({agreement.ties} ties excluded)"
        )
    correct, total = anchor_accuracy(judgements)
    print(f"anchor accuracy          {correct}/{total}")

    train_pairs, val_pairs = split_by_session(
        pairs, cfg.pairs.val_frac, cfg.pairs.split_seed
    )
    print(f"train pairs {len(train_pairs)}   val pairs {len(val_pairs)}")
    if args.dry_run:
        return
    if not train_pairs:
        raise SystemExit("no usable pairs; nothing to train on")
    if len(pairs) < cfg.pairs.min_pairs:
        raise SystemExit(
            f"only {len(pairs)} usable pairs against min_pairs={cfg.pairs.min_pairs}. "
            "Section 7.5 puts DPO signal at 200-500 pairs; rate more, or pass "
            "--min-pairs to override deliberately."
        )

    policy, reference, ar_cfg, tracks, manifest, cache_dir = load_reference(cfg)
    bank.require_tokenizer(cache_dir.name)
    trainable, total_params = freeze_trunk(policy, cfg.dpo.trainable_blocks)
    print(
        f"policy {total_params/1e6:.1f}M params, "
        f"{trainable/1e6:.1f}M trainable "
        f"(top {cfg.dpo.trainable_blocks} of {len(policy.blocks)} blocks)"
    )

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    reference.to(device)

    crop = ar_cfg.data.crop_frames
    train_set = PreferenceDataset(train_pairs, tracks, bank, crop, seed=cfg.train.seed)
    train_sampler = LengthBucketSampler(
        train_set, cfg.train.batch_pairs, shuffle=True, seed=cfg.train.seed
    )
    loaders = {
        "train_dataloaders": DataLoader(
            train_set,
            batch_sampler=train_sampler,
            collate_fn=collate_pairs,
            num_workers=cfg.train.num_workers,
        )
    }
    if val_pairs:
        val_set = PreferenceDataset(
            val_pairs, tracks, bank, crop, seed=cfg.train.seed, fixed_window=True
        )
        loaders["val_dataloaders"] = DataLoader(
            val_set,
            batch_sampler=LengthBucketSampler(
                val_set, cfg.train.batch_pairs, shuffle=False
            ),
            collate_fn=collate_pairs,
            num_workers=cfg.train.num_workers,
        )

    module = DpoLightningModule(policy, reference, cfg)
    save_dir = REPO / cfg.train.save_path.replace(
        "<date>", date.today().strftime("%Y%m%d")
    )
    callbacks: list[L.Callback] = [
        ArCheckpointWriter(save_dir, ar_cfg, "val/acc" if val_pairs else "train/acc")
    ]
    if cfg.train.diversity_every:
        callbacks.append(
            DiversityMonitor(
                ArGenerator(policy, device, window_frames=crop),
                diversity_request(
                    train_pairs,
                    tracks,
                    cfg.train.diversity_seconds,
                    float(manifest["tokenizer_meta"]["frames_per_second"]),
                ),
                clips=cfg.train.diversity_clips,
                every=cfg.train.diversity_every,
                gate=cfg.train.diversity_gate,
                num_tokens=int(manifest["tokenizer_meta"]["num_tokens"]),
            )
        )

    # A smoke run proves the wiring, not the objective: a handful of steps, and
    # the diversity probe fires often enough to be seen doing it.
    limits = {"limit_train_batches": 4, "limit_val_batches": 2} if args.smoke else {}
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        max_time=(
            timedelta(hours=cfg.train.max_hours)
            if cfg.train.max_hours and not args.smoke
            else None
        ),
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=[device.index or 0] if device.type == "cuda" else 1,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        accumulate_grad_batches=cfg.train.accumulate,
        enable_checkpointing=False,
        logger=L.pytorch.loggers.TensorBoardLogger(str(save_dir), name="dpo"),
        callbacks=callbacks,
        log_every_n_steps=1,
        **limits,
    )
    start = time.monotonic()
    trainer.fit(module, **loaders)
    print(
        f"done in {(time.monotonic() - start)/60:.1f} min; "
        f"checkpoints in {save_dir}"
    )
    (save_dir / "pairs.json").write_text(
        json.dumps(
            {
                "reference_checkpoint": cfg.pairs.reference_checkpoint,
                "train": [p.pair_id for p in train_pairs],
                "val": [p.pair_id for p in val_pairs],
                "report": {
                    "total": report.total,
                    "kept": report.kept,
                    "dropped": dict(report.dropped),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
