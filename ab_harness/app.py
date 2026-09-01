"""
Entry point for the A/B preference harness.

Usage:
  uv run python -m ab_harness.app
  uv run python -m ab_harness.app --config config_ab.yaml --seed 7

The UI process deliberately never imports torch: generation happens in a child
process (see ab_harness.worker.client). On a cold bank the window comes up
immediately and shows "preparing" while the worker loads the checkpoint; the
corpus it publishes is what the sampler needs before it can draw anything.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ab_harness.config import REPO, AbConfig, load_config
from ab_harness.model.bank import ClipBank
from ab_harness.model.judgement import JudgementLog, read_all
from ab_harness.model.pair_sampler import PairSampler, TrackInfo, load_corpus
from ab_harness.model.pipeline import PairPipeline
from ab_harness.model.stats import report
from ab_harness.view.main_window import MainWindow
from ab_harness.viewmodel.player_vm import PlayerViewModel
from ab_harness.viewmodel.session_vm import SessionViewModel
from ab_harness.worker.client import ProcessClipProducer

CORPUS_TIMEOUT_S = 300.0


def wait_for_corpus(path: Path, timeout: float = CORPUS_TIMEOUT_S) -> list[TrackInfo]:
    """
    Block until the worker publishes the corpus, or give up.

    Only the first run on a fresh bank pays this; afterwards corpus.json is
    already there and the wait returns immediately.

    Args:
      path (Path): corpus.json location.
      timeout (float): seconds to wait.

    Returns:
      list[TrackInfo]: the corpus, empty if it never appeared.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        tracks = load_corpus(path)
        if tracks:
            return tracks
        time.sleep(0.5)
    return []


def resolve_config(name: str) -> AbConfig:
    """
    Load a config from the working directory or the repo root, or fall back to
    the defaults so the harness still starts without one.

    Args:
      name (str): config filename or path.

    Returns:
      AbConfig: the config to run with.
    """
    for candidate in (Path(name), REPO / name):
        if candidate.exists():
            return load_config(candidate)
    return AbConfig()


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_ab.yaml")
    parser.add_argument("--seed", type=int, default=None, help="override sampler seed")
    parser.add_argument("--session-id", default=None)
    return parser.parse_args()


def main() -> int:
    """
    Returns:
      int: process exit code.
    """
    args = parse_args()
    cfg = resolve_config(args.config)

    bank = ClipBank(cfg.bank_root)
    producer = ProcessClipProducer(cfg)
    producer.start()

    tracks = wait_for_corpus(bank.root / "corpus.json")
    if not tracks:
        producer.close()
        print(
            f"worker never published {bank.root / 'corpus.json'}; "
            "check the checkpoint and token cache paths in the config",
            file=sys.stderr,
        )
        return 1

    seed = args.seed if args.seed is not None else cfg.session.seed
    sampler = PairSampler(
        tracks,
        cfg.sampler,
        checkpoint=cfg.generator.checkpoint,
        rng=random.Random(seed),
    )
    # What has already been judged, so banked material the rater has never
    # heard is offered before anything is replayed.
    rated = {
        item
        for judgement in read_all(bank.sessions_dir)
        for item in (judgement.item_left, judgement.item_right)
    }
    pipeline = PairPipeline(
        sampler,
        producer,
        bank,
        depth=cfg.session.prefetch_depth,
        structure_live=cfg.session.structure_live,
        structure_backfill=cfg.session.structure_backfill,
        min_fill=cfg.session.min_fill,
        rated_items=rated,
    )

    session_id = args.session_id or uuid.uuid4().hex[:12]
    log = JudgementLog(bank.sessions_dir / f"{session_id}.jsonl", bank.judgements_tsv)

    app = QApplication(sys.argv)
    sample_rate = int(
        bank.manifest().get("tokenizer_meta", {}).get("sample_rate", 44100)
    )
    player = PlayerViewModel(sample_rate, cfg.session.crossfade_ms)
    session = SessionViewModel(pipeline, log, cfg.session.target_lufs, session_id)
    window = MainWindow(
        session,
        player,
        quiet_fill=cfg.session.quiet_fill,
        checkpoints=cfg.checkpoints,
    )
    window.show()
    session.start()

    code = app.exec()
    producer.close()
    print(report(bank.root))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
