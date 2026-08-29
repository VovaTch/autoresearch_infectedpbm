"""The judgement log: append-only, flushed, and mirrored."""

from __future__ import annotations

from pathlib import Path

from ab_harness.model.judgement import (
    TSV_COLUMNS,
    JudgementLog,
    read_all,
    read_session,
    rewrite_tsv,
    utc_now,
)
from ab_harness.model.types import Judgement, Tier


def _judgement(pair: str, choice: str = "left", session: str = "s0") -> Judgement:
    """
    Args:
      pair (str): pair id.
      choice (str): the decision.
      session (str): session id.

    Returns:
      Judgement: a filled-in row.
    """
    return Judgement(
        pair_id=pair,
        session_id=session,
        tier=Tier.BULK,
        question="Which sounds cleaner?",
        item_left="gen_a",
        item_right="gen_b",
        choice=choice,  # type: ignore[arg-type]
        chosen_item_id="" if choice == "tie" else f"gen_{choice[0]}",
        response_ms=1234,
        is_repeat=False,
        is_anchor=False,
        lufs_target=-23.0,
        was_live=True,
        ts=utc_now(),
    )


def test_append_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "s0.jsonl"
    log = JudgementLog(path, tmp_path / "judgements.tsv")
    rows = [_judgement("p0"), _judgement("p1", "tie"), _judgement("p2", "right")]
    for row in rows:
        log.append(row)
    assert log.count == 3
    assert read_session(path) == rows


def test_each_row_is_flushed_immediately(tmp_path: Path) -> None:
    path = tmp_path / "s0.jsonl"
    log = JudgementLog(path)
    log.append(_judgement("p0"))
    # readable without closing anything: a crash now would still keep the row
    assert len(read_session(path)) == 1


def test_tsv_mirror_gets_a_header_and_one_row_each(tmp_path: Path) -> None:
    tsv = tmp_path / "judgements.tsv"
    log = JudgementLog(tmp_path / "s0.jsonl", tsv)
    for i in range(3):
        log.append(_judgement(f"p{i}"))
    lines = tsv.read_text().strip().splitlines()
    assert lines[0].split("\t") == list(TSV_COLUMNS)
    assert len(lines) == 4


def test_read_all_merges_sessions_in_time_order(tmp_path: Path) -> None:
    JudgementLog(tmp_path / "s0.jsonl").append(_judgement("p0", session="s0"))
    JudgementLog(tmp_path / "s1.jsonl").append(_judgement("p1", session="s1"))
    merged = read_all(tmp_path)
    assert {j.session_id for j in merged} == {"s0", "s1"}
    assert [j.ts for j in merged] == sorted(j.ts for j in merged)


def test_missing_session_reads_as_empty(tmp_path: Path) -> None:
    assert read_session(tmp_path / "nope.jsonl") == []
    assert read_all(tmp_path) == []


def test_tsv_can_be_regenerated_from_the_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "s0.jsonl"
    log = JudgementLog(path)
    rows = [_judgement("p0"), _judgement("p1")]
    for row in rows:
        log.append(row)
    tsv = tmp_path / "rebuilt.tsv"
    rewrite_tsv(tsv, read_session(path))
    assert len(tsv.read_text().strip().splitlines()) == 3
