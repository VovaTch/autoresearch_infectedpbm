"""
Which trained checkpoints the harness can sample from.

Discovery is by directory prefix, not by opening files: the UI process holds no
torch, and a VQ-GAN tokenizer checkpoint under saved_20260827_cont9h/ looks
exactly like an AR one from the outside while crashing the worker on load.
saved_ar_* and saved_dpo_* are the two families build_model can consume.

The default is the newest DPO checkpoint. Preference fine-tuning starts from an
AR checkpoint and is meant to replace it at the rating stage, so a fresh DPO run
becomes what the harness serves without a config edit -- and because ratings
collected against a stale generator are wasted listening. The AR family is the
fallback for a repo that has never run DPO, and an explicit path in the config
always wins.

This module imports nothing heavier than pathlib, so the UI process, the worker
and bake_ab_bank.py all resolve the same way.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

# Config values meaning "pick for me"; anything else is taken as a path.
AUTO = ("", "auto", "latest")
# Ordered: the family the selector should offer first comes first.
FAMILIES = ("saved_dpo_*", "saved_ar_*")
# The style-level model (train_style_ar.py) is a different family: it is never
# sampled for tokens, only asked for the next style descriptor.
STYLE_AR_FAMILIES = ("saved_style_ar_*",)
# The two non-autoregressive token generators: the latent-flow DiT
# (train_zflow.py, sampled by ab_harness.worker.zflow_gen) and the masked
# discrete diffusion model (train_mdm.py, ab_harness.worker.mdm_gen).
ZFLOW_FAMILIES = ("saved_zflow_*",)
MDM_FAMILIES = ("saved_mdm_*",)
# Backend name -> the directory globs its checkpoints live in. The backend is
# implied by the checkpoint path, so no message or bank record needs a second
# field to say which sampler produced a clip.
BACKENDS: dict[str, tuple[str, ...]] = {
    "ar": FAMILIES,
    "zflow": ZFLOW_FAMILIES,
    "mdm": MDM_FAMILIES,
}
ALL_FAMILIES: tuple[str, ...] = FAMILIES + ZFLOW_FAMILIES + MDM_FAMILIES


def backend_of(checkpoint: str) -> str:
    """
    Which sampler a checkpoint path belongs to.

    Args:
      checkpoint (str): repo-relative checkpoint path.

    Returns:
      str: a key of BACKENDS; "ar" for anything outside the other families,
        which is also what an explicit AR path resolves to.
    """
    top = Path(checkpoint).parts[0] if Path(checkpoint).parts else ""
    for backend, families in BACKENDS.items():
        if backend == "ar":
            continue
        if any(fnmatch(top, family) for family in families):
            return backend
    return "ar"


def discover_all(repo: Path) -> dict[str, list[str]]:
    """
    Args:
      repo (Path): repository root.

    Returns:
      dict[str, list[str]]: backend -> its checkpoints, newest first; every
        backend is present, possibly with an empty list.
    """
    return {b: discover_checkpoints(repo, fams) for b, fams in BACKENDS.items()}


def discover_checkpoints(repo: Path, families: tuple[str, ...] = FAMILIES) -> list[str]:
    """
    List the checkpoints the worker is willing to load, newest first.

    Args:
      repo (Path): repository root holding the saved_* run directories.
      families (tuple[str, ...]): directory globs to search, in offer order.

    Returns:
      list[str]: repo-relative paths, DPO family before AR family and newest
        first inside each. Empty when nothing is trained yet.
    """
    found: list[str] = []
    for family in families:
        paths = [p for d in sorted(repo.glob(family)) for p in d.glob("*.ckpt")]
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        found += [str(p.relative_to(repo)) for p in paths]
    return found


def resolve_checkpoint(
    spec: str, repo: Path, families: tuple[str, ...] = FAMILIES
) -> str:
    """
    Turn a config value into a concrete repo-relative checkpoint path.

    Resolution happens once, at config load, so no "auto" ever reaches a
    ClipSpec: the checkpoint tag on a banked clip has to name the model that
    actually produced it or the pair data cannot be split by generator later.

    Args:
      spec (str): config value; one of AUTO, or a path.
      repo (Path): repository root.
      families (tuple[str, ...]): directory globs to search.

    Returns:
      str: the checkpoint to load. An unresolvable AUTO is returned unchanged so
        the worker raises one clear error naming the path it looked for, rather
        than this call failing at import time in every process.
    """
    if spec not in AUTO:
        return spec
    found = discover_checkpoints(repo, families)
    return found[0] if found else spec
