"""
Turn banked A/B judgements into preference pairs.

Torch-free on purpose: the rating UI counts usable pairs with exactly the
filter DPO trains on (ab_harness.model.auto_train), and the UI process must
never import torch. train_dpo.py re-exports everything here.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from ab_harness.model.bank import ClipBank
from ab_harness.model.judgement import read_all
from ab_harness.model.stats import anchor_accuracy
from ab_harness.model.types import ClipSpec, Judgement, Tier


@dataclass
class PairsCfg:
    """
    Where the preference pairs come from and which ones survive filtering.

    Args:
      bank_root (str): the A/B bank written by ab_harness.
      reference_checkpoint (str): the checkpoint (AR, zflow or MDM family)
        that is both the policy initialization and the frozen DPO reference.
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
        if left.conditioning.walking:
            # The bank keeps a walking clip's tokens but not the descriptors its
            # later segments were sampled under; scoring it against the first
            # segment's style alone would train on the wrong conditioning.
            report.dropped["style walk"] += 1
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
