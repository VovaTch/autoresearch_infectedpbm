"""
Append-only judgement log (PLAN section 9.2).

One JSON object per line per decision, flushed immediately, so a crash costs at
most the pair in progress. A TSV mirror sits alongside for eyeballing next to
results.tsv; the JSONL is the source of truth and the TSV is regenerable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from ab_harness.model.types import Judgement

TSV_COLUMNS = (
    "ts",
    "session_id",
    "pair_id",
    "tier",
    "choice",
    "chosen_item_id",
    "item_left",
    "item_right",
    "response_ms",
    "is_repeat",
    "is_anchor",
    "was_live",
)


def utc_now() -> str:
    """
    Returns:
      str: current time as an ISO-8601 UTC string.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


class JudgementLog:
    """
    Append-only sink for one rating session.

    Args:
      session_path (Path): the session's .jsonl file; created if absent.
      tsv_path (Path | None): flat mirror to append to, or None to skip it.
    """

    def __init__(self, session_path: Path, tsv_path: Path | None = None) -> None:
        self.session_path = Path(session_path)
        self.tsv_path = Path(tsv_path) if tsv_path is not None else None
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def append(self, judgement: Judgement) -> None:
        """
        Persist one decision to both the session log and the mirror.

        Args:
          judgement (Judgement): the decision.
        """
        with self.session_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(judgement.to_json()) + "\n")
            handle.flush()
        if self.tsv_path is not None:
            self._append_tsv(judgement)
        self._count += 1

    def _append_tsv(self, judgement: Judgement) -> None:
        """
        Args:
          judgement (Judgement): the decision to mirror.
        """
        assert self.tsv_path is not None
        row = judgement.to_json()
        new_file = not self.tsv_path.exists()
        with self.tsv_path.open("a", encoding="utf-8") as handle:
            if new_file:
                handle.write("\t".join(TSV_COLUMNS) + "\n")
            handle.write("\t".join(str(row[c]) for c in TSV_COLUMNS) + "\n")
            handle.flush()

    @property
    def count(self) -> int:
        """
        Returns:
          int: judgements written by this instance.
        """
        return self._count


def read_session(path: Path) -> list[Judgement]:
    """
    Args:
      path (Path): a session .jsonl file.

    Returns:
      list[Judgement]: every judgement in file order; blank lines skipped.
    """
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Judgement.from_json(json.loads(line)) for line in lines if line.strip()]


def read_all(sessions_dir: Path) -> list[Judgement]:
    """
    Args:
      sessions_dir (Path): directory of session logs.

    Returns:
      list[Judgement]: every judgement from every session, sorted by timestamp.
    """
    out: list[Judgement] = []
    for path in sorted(Path(sessions_dir).glob("*.jsonl")):
        out.extend(read_session(path))
    return sorted(out, key=lambda j: j.ts)


def rewrite_tsv(path: Path, judgements: Iterable[Judgement]) -> None:
    """
    Regenerate the flat mirror from the JSONL source of truth.

    Args:
      path (Path): TSV to write.
      judgements (Iterable[Judgement]): rows to write.
    """
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("\t".join(TSV_COLUMNS) + "\n")
        for judgement in judgements:
            row = judgement.to_json()
            handle.write("\t".join(str(row[c]) for c in TSV_COLUMNS) + "\n")
