"""
Diagnostics over collected judgements (PLAN sections 7.6, 7.1 rung 2).

Self-agreement is the gate: below ~80% on the bulk tier, label noise dominates
and no amount of reward-model capacity will help -- the comparison itself needs
simplifying before more data is worth collecting. Expect the structure tier to
be noisier; that is normal, not a reason to abandon it.

Anchor accuracy detects fatigue sessions. Sessions that fail it should be
discarded before anything is trained on them.

Usage:
  uv run python -m ab_harness.model.stats ~/.cache/infected_pbm/ab
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ab_harness.model.judgement import read_all
from ab_harness.model.types import Judgement, Tier


def _unordered(judgement: Judgement) -> tuple[str, str]:
    """
    Args:
      judgement (Judgement): a decision.

    Returns:
      tuple[str, str]: the two item ids, order-independent, so a repeat shown
        with the sides swapped still matches its original.
    """
    return tuple(sorted((judgement.item_left, judgement.item_right)))  # type: ignore[return-value]


@dataclass(frozen=True)
class Agreement:
    """
    Self-agreement over repeated comparisons.

    Args:
      compared (int): repeat showings that had an earlier showing to match.
      agreed (int): showings whose winner matched the earlier one.
      ties (int): showings where either result was a tie; excluded from both.
    """

    compared: int
    agreed: int
    ties: int

    @property
    def rate(self) -> float:
        """
        Returns:
          float: agreed / compared, or nan when nothing was comparable.
        """
        return self.agreed / self.compared if self.compared else float("nan")


def self_agreement(
    judgements: Sequence[Judgement], tier: Tier | None = None
) -> Agreement:
    """
    Measure how often the rater made the same call on the same comparison twice.

    Ties are counted but excluded from the rate: "can't tell" twice is not a
    contradiction, and treating it as agreement would flatter the number.

    Args:
      judgements (Sequence[Judgement]): decisions in time order.
      tier (Tier | None): restrict to one tier, or None for all.

    Returns:
      Agreement: the tally.
    """
    first: dict[tuple[str, str], str] = {}
    compared = agreed = ties = 0
    for judgement in judgements:
        if tier is not None and judgement.tier != tier:
            continue
        key = _unordered(judgement)
        if key not in first:
            first[key] = judgement.chosen_item_id
            continue
        if not first[key] or not judgement.chosen_item_id:
            ties += 1
            continue
        compared += 1
        agreed += int(first[key] == judgement.chosen_item_id)
    return Agreement(compared=compared, agreed=agreed, ties=ties)


def anchor_accuracy(judgements: Sequence[Judgement]) -> tuple[int, int]:
    """
    Count anchor pairs answered the expected way.

    An anchor puts a generation against the tokenizer's own reconstruction of
    real tokens, so the reference is the expected winner. Reference item ids
    carry the "ref_" prefix assigned by the pair sampler.

    Args:
      judgements (Sequence[Judgement]): decisions.

    Returns:
      tuple[int, int]: (correct, total) over non-tie anchor showings.
    """
    correct = total = 0
    for judgement in judgements:
        if not judgement.is_anchor or not judgement.chosen_item_id:
            continue
        total += 1
        correct += int(judgement.chosen_item_id.startswith("ref_"))
    return correct, total


def win_counts(judgements: Iterable[Judgement]) -> dict[str, Counter[str]]:
    """
    Tally wins, losses and ties per item.

    This is the input a Bradley-Terry or ELO fit consumes; the fit itself is
    rung 2 and deliberately not implemented here.

    Args:
      judgements (Iterable[Judgement]): decisions.

    Returns:
      dict[str, Counter[str]]: item id -> counts keyed "win"/"loss"/"tie".
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for judgement in judgements:
        left, right = judgement.item_left, judgement.item_right
        if judgement.choice == "tie":
            counts[left]["tie"] += 1
            counts[right]["tie"] += 1
            continue
        winner = judgement.chosen_item_id
        loser = right if winner == left else left
        counts[winner]["win"] += 1
        counts[loser]["loss"] += 1
    return dict(counts)


def report(bank_root: Path) -> str:
    """
    Render the human-readable summary the harness prints after a session.

    Args:
      bank_root (Path): bank directory holding sessions/.

    Returns:
      str: the report.
    """
    judgements = read_all(Path(bank_root).expanduser() / "sessions")
    if not judgements:
        return f"no judgements under {bank_root}"
    lines = [f"{len(judgements)} judgements from {bank_root}"]
    for tier in (None, Tier.BULK, Tier.STRUCTURE):
        agreement = self_agreement(judgements, tier)
        label = "all" if tier is None else str(tier)
        if not agreement.compared:
            lines.append(f"  self-agreement {label:<10} n/a (no repeats yet)")
            continue
        flag = (
            ""
            if agreement.rate >= 0.8 or tier is Tier.STRUCTURE
            else "  <-- below 0.80"
        )
        lines.append(
            f"  self-agreement {label:<10} {agreement.rate:.2%} "
            f"({agreement.agreed}/{agreement.compared}, {agreement.ties} ties){flag}"
        )
    correct, total = anchor_accuracy(judgements)
    if total:
        lines.append(f"  anchors        {correct}/{total} answered as expected")
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("~/.cache/infected_pbm/ab")
    print(report(root))
