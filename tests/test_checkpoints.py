"""
Checkpoint discovery.

The selector's contents come from directory names, not from opening files: the
UI process has no torch, and loading a VQ-GAN tokenizer checkpoint into
build_model kills the worker. These pin which families are offered and which one
"auto" lands on, because getting that wrong means a whole rating session spent
on the wrong model.
"""

from __future__ import annotations

import os
from pathlib import Path

from ab_harness.checkpoints import discover_checkpoints, resolve_checkpoint


def _ckpt(repo: Path, rel: str, mtime: float) -> Path:
    """
    Args:
      repo (Path): fake repo root.
      rel (str): path relative to it.
      mtime (float): modification time to stamp, so ordering is deterministic.

    Returns:
      Path: the created file.
    """
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))
    return path


def test_only_ar_and_dpo_families_are_offered(tmp_path: Path) -> None:
    _ckpt(tmp_path, "saved_ar_20260829_24h/ar_latest.ckpt", 100.0)
    _ckpt(tmp_path, "saved_dpo_20260830_6h/dpo_latest.ckpt", 200.0)
    # a VQ-GAN run directory: same suffix, unloadable by the AR worker
    _ckpt(tmp_path, "saved_20260827_cont9h/last.ckpt", 300.0)

    found = discover_checkpoints(tmp_path)
    assert found == [
        "saved_dpo_20260830_6h/dpo_latest.ckpt",
        "saved_ar_20260829_24h/ar_latest.ckpt",
    ]


def test_newest_first_within_a_family(tmp_path: Path) -> None:
    _ckpt(tmp_path, "saved_dpo_a/dpo_best.ckpt", 100.0)
    _ckpt(tmp_path, "saved_dpo_a/dpo_latest.ckpt", 200.0)
    assert discover_checkpoints(tmp_path)[0] == "saved_dpo_a/dpo_latest.ckpt"


def test_auto_prefers_the_last_dpo_run_over_a_newer_ar_one(tmp_path: Path) -> None:
    # DPO is the fine-tune of the AR model, so it wins even when the AR file
    # was written later -- family order, not mtime, decides across families.
    _ckpt(tmp_path, "saved_dpo_20260830_6h/dpo_latest.ckpt", 100.0)
    _ckpt(tmp_path, "saved_ar_20260829_24h/ar_latest.ckpt", 999.0)
    assert (
        resolve_checkpoint("auto", tmp_path) == "saved_dpo_20260830_6h/dpo_latest.ckpt"
    )


def test_auto_falls_back_to_ar_when_no_dpo_run_exists(tmp_path: Path) -> None:
    _ckpt(tmp_path, "saved_ar_20260829_24h/ar_latest.ckpt", 100.0)
    assert (
        resolve_checkpoint("auto", tmp_path) == "saved_ar_20260829_24h/ar_latest.ckpt"
    )


def test_an_explicit_path_is_never_rewritten(tmp_path: Path) -> None:
    _ckpt(tmp_path, "saved_dpo_a/dpo_latest.ckpt", 100.0)
    pinned = "saved_ar_20260829_24h/ar_frozen_0829.ckpt"
    assert resolve_checkpoint(pinned, tmp_path) == pinned


def test_unresolvable_auto_is_left_for_the_worker_to_report(tmp_path: Path) -> None:
    # Returning the sentinel keeps the one clear error in the worker, rather
    # than raising at config-load time in every process that reads a config.
    assert resolve_checkpoint("auto", tmp_path) == "auto"
