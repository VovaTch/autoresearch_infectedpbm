"""Self-agreement, anchors and win tallies over synthetic judgements."""

from __future__ import annotations

from ab_harness.model.stats import anchor_accuracy, report, self_agreement, win_counts
from ab_harness.model.types import Judgement, Tier


def _row(
    left: str,
    right: str,
    choice: str,
    tier: Tier = Tier.BULK,
    anchor: bool = False,
    repeat: bool = False,
    ts: str = "2026-08-29T00:00:00+00:00",
) -> Judgement:
    """
    Args:
      left (str): left item id.
      right (str): right item id.
      choice (str): "left", "right" or "tie".
      tier (Tier): rating tier.
      anchor (bool): anchor flag.
      repeat (bool): repeat flag.
      ts (str): timestamp.

    Returns:
      Judgement: a synthetic row.
    """
    chosen = "" if choice == "tie" else (left if choice == "left" else right)
    return Judgement(
        pair_id=f"{left}:{right}:{ts}",
        session_id="s",
        tier=tier,
        question="q",
        item_left=left,
        item_right=right,
        choice=choice,  # type: ignore[arg-type]
        chosen_item_id=chosen,
        response_ms=100,
        is_repeat=repeat,
        is_anchor=anchor,
        lufs_target=-23.0,
        was_live=False,
        ts=ts,
    )


def test_perfect_self_agreement() -> None:
    rows = [_row("a", "b", "left"), _row("a", "b", "left", repeat=True)]
    agreement = self_agreement(rows)
    assert agreement.compared == 1 and agreement.rate == 1.0


def test_a_contradiction_is_counted() -> None:
    rows = [_row("a", "b", "left"), _row("a", "b", "right", repeat=True)]
    assert self_agreement(rows).rate == 0.0


def test_a_repeat_shown_with_the_sides_swapped_still_matches() -> None:
    rows = [_row("a", "b", "left"), _row("b", "a", "right", repeat=True)]
    assert self_agreement(rows).rate == 1.0


def test_ties_are_excluded_rather_than_counted_as_agreement() -> None:
    rows = [_row("a", "b", "left"), _row("a", "b", "tie", repeat=True)]
    agreement = self_agreement(rows)
    assert agreement.compared == 0 and agreement.ties == 1


def test_agreement_can_be_read_per_tier() -> None:
    rows = [
        _row("a", "b", "left"),
        _row("a", "b", "left", repeat=True),
        _row("c", "d", "left", tier=Tier.STRUCTURE),
        _row("c", "d", "right", tier=Tier.STRUCTURE, repeat=True),
    ]
    assert self_agreement(rows, Tier.BULK).rate == 1.0
    assert self_agreement(rows, Tier.STRUCTURE).rate == 0.0


def test_no_repeats_gives_nothing_to_compare() -> None:
    agreement = self_agreement([_row("a", "b", "left")])
    assert agreement.compared == 0
    assert agreement.rate != agreement.rate  # nan


def test_anchor_accuracy_expects_the_reference_to_win() -> None:
    rows = [
        _row("gen_1", "ref_1", "right", anchor=True),
        _row("gen_2", "ref_2", "left", anchor=True),
        _row("gen_3", "ref_3", "tie", anchor=True),
        _row("gen_4", "gen_5", "left"),
    ]
    assert anchor_accuracy(rows) == (1, 2)


def test_win_counts_tally_all_three_outcomes() -> None:
    rows = [_row("a", "b", "left"), _row("a", "b", "right"), _row("a", "b", "tie")]
    counts = win_counts(rows)
    assert (
        counts["a"]["win"] == 1 and counts["a"]["loss"] == 1 and counts["a"]["tie"] == 1
    )
    assert counts["b"]["win"] == 1 and counts["b"]["loss"] == 1


def test_report_handles_an_empty_bank(tmp_path) -> None:
    assert "no judgements" in report(tmp_path)


def test_report_names_the_tiers(tmp_path) -> None:
    from ab_harness.model.judgement import JudgementLog

    log = JudgementLog(tmp_path / "sessions" / "s0.jsonl")
    log.append(_row("a", "b", "left"))
    log.append(_row("a", "b", "left", repeat=True))
    text = report(tmp_path)
    assert "self-agreement" in text and "bulk" in text
