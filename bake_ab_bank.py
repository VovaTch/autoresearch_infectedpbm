"""
Pre-fill the A/B clip bank so the rating harness never waits on the GPU.

The app generates 10 s bulk-tier clips live, but a 90 s structure-tier pair is
roughly forty seconds of sampling and cannot keep up with anyone rating at a
human pace. Those get baked ahead of time here. Running this on the bulk tier
too is still worth it before a long session: a warm queue from the first pair
beats one that has to fill while the rater sits there.

Only tokens are written. Audio is decoded on demand in RAM, because tokens are
what a reward model and DPO consume -- the wavs were only ever playback
material, and a bank of them costs a thousand times more disk.

This runs the same GenerationService the app's worker runs, so what is baked and
what is streamed cannot drift apart.

Usage:
  uv run python bake_ab_bank.py --config config_ab.yaml --smoke
  uv run python bake_ab_bank.py --config config_ab.yaml --pairs 60 --tier structure

  # long bakes detached, per the power-loss finding
  setsid nohup uv run python -u bake_ab_bank.py --config config_ab.yaml \
      --pairs 200 > logs/bake_ab.log 2>&1 &
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from ab_harness.config import REPO, AbConfig, load_config
from ab_harness.model.pair_sampler import PairSampler
from ab_harness.model.types import ClipSpec, PairSpec, Tier
from ab_harness.worker.service import GenerationService
from generate_ar import token_stats


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_ab.yaml")
    parser.add_argument("--pairs", type=int, default=50, help="pairs to bake")
    parser.add_argument(
        "--tier",
        choices=["both", "bulk", "structure"],
        default="both",
        help="restrict which tier is baked",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="bake 4 bulk pairs and 1 structure pair, with token diagnostics",
    )
    parser.add_argument("--quiet", action="store_true", help="skip token diagnostics")
    return parser.parse_args()


def load_cfg(name: str) -> AbConfig:
    """
    Args:
      name (str): config filename or path.

    Returns:
      AbConfig: the config, or defaults when no file is found.
    """
    for candidate in (Path(name), REPO / name):
        if candidate.exists():
            return load_config(candidate)
    print(f"no config at {name}; using defaults")
    return AbConfig()


def draw(sampler: PairSampler, count: int, tier: str) -> list[PairSpec]:
    """
    Draw the comparisons to bake, honouring a tier restriction.

    Args:
      sampler (PairSampler): the draw source.
      count (int): how many pairs to collect.
      tier (str): "both", "bulk" or "structure".

    Returns:
      list[PairSpec]: the drawn specs. Repeats and anchors are skipped: a repeat
        adds no new tokens, and an anchor's reference side decodes in
        milliseconds, so neither is worth GPU time here.
    """
    wanted = None if tier == "both" else Tier(tier)
    specs: list[PairSpec] = []
    attempts = 0
    while len(specs) < count and attempts < count * 200:
        attempts += 1
        spec = sampler.next_spec()
        if spec.is_repeat or spec.is_anchor:
            continue
        if wanted is not None and spec.tier is not wanted:
            continue
        specs.append(spec)
    return specs


def batches(specs: list[ClipSpec], size: int) -> Iterator[list[ClipSpec]]:
    """
    Chunk clips into sampling batches of one tier at a time.

    Lanes run to the longest in their batch, so a 10 s clip batched with a 90 s
    one would be charged 90 s of steps. Grouping by length first keeps every
    batch honest.

    Args:
      specs (list[ClipSpec]): clips to bake.
      size (int): maximum lanes per batch.

    Returns:
      Iterator[list[ClipSpec]]: batches of same-length clips.
    """
    by_length: dict[int, list[ClipSpec]] = {}
    for spec in specs:
        by_length.setdefault(spec.n_frames, []).append(spec)
    for group in by_length.values():
        for start in range(0, len(group), max(1, size)):
            yield group[start : start + size]


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    torch.manual_seed(args.seed)

    service = GenerationService(cfg)
    started = time.monotonic()
    service.load()
    print(f"service ready in {time.monotonic() - started:.1f}s -> {cfg.bank_root}")

    sampler = PairSampler(
        service.corpus(),
        cfg.sampler,
        checkpoint=cfg.generator.checkpoint,
        rng=random.Random(args.seed),
    )
    if args.smoke:
        specs = draw(sampler, 4, "bulk") + draw(sampler, 1, "structure")
    else:
        specs = draw(sampler, args.pairs, args.tier)
    print(f"baking {len(specs)} pairs ({sum(2 for _ in specs)} clips)")

    num_tokens = int(service.meta["num_tokens"])
    clips = [side for spec in specs for side in (spec.left, spec.right)]
    skipped = sum(service.bank.has(c.item_id) for c in clips)
    todo = [c for c in clips if not service.bank.has(c.item_id)]
    print(f"{skipped} already banked, {len(todo)} to generate")

    made = failed = 0
    for start, batch in enumerate(batches(todo, cfg.generator.max_batch)):
        head = batch[0].conditioning
        print(
            f"\nbatch of {len(batch)}  {batch[0].tier} {batch[0].n_frames} frames  "
            f"first: track {head.track_idx} @ {head.start_frame} "
            f"id={int(head.use_track_id)} style={int(head.use_style)} "
            f"prompt={head.prompt_frames} cfg={head.cfg_strength}",
            flush=True,
        )
        batch_started = time.monotonic()
        results = service.produce_many(batch)
        elapsed = time.monotonic() - batch_started
        for result in results:
            if not result.ok:
                failed += 1
                print(f"  {result.spec.item_id} FAILED: {result.error}")
                continue
            made += 1
            print(f"  {result.spec.item_id} seed {result.spec.sampling.seed}")
            if not args.quiet:
                assert result.tokens is not None
                token_stats(
                    f"seed{result.spec.sampling.seed % 10000}",
                    torch.from_numpy(result.tokens.astype(np.int64)),
                    num_tokens,
                )
        print(
            f"  {elapsed:.1f}s for {len(batch)} clips ({elapsed / len(batch):.1f}s each)"
        )

    total = time.monotonic() - started
    print(
        f"\ndone in {total / 60:.1f} min: {made} generated, {skipped} already banked, "
        f"{failed} failed -> {cfg.bank_root}"
    )
    service.close()


if __name__ == "__main__":
    main()
