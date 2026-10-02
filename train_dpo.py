"""
Direct preference optimization of a token generator on A/B judgements.

Implements PLAN_generative_stage.md section 7.4 -- rung 5 of the preference
ladder -- for all three backends, picked by the reference checkpoint's family
(ab_harness.checkpoints.backend_of):

- ar: exact sequence log-probability (the original recipe).
- zflow: Diffusion-DPO. log p is replaced by the flow-matching ELBO proxy
  -0.5 ||v_hat - (x1 - eps)||^2 on one (t, eps) draw, shared by both sides of
  a pair and by policy and reference, so the noise cancels in the margin.
- mdm: the same with the SoundStorm masked-token CE on one shared corruption.

DPO only needs the likelihood of a banked token grid under its conditioning,
not the sampler that produced it, so every pair trains every backend (pairs
drawn from another backend are off-policy, which DPO tolerates). zflow and
MDM see 512-frame windows: each is scored with the frames before it held
clean as a prefix, exactly how their samplers chained windows.

Rung 1 (the A/B harness) already emits exactly what this needs: every
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
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from ab_harness.config import GeneratorCfg
from ab_harness.model.bank import ClipBank
from ab_harness.model.judgement import read_all
from ab_harness.model.preferences import (
    PairsCfg,
    PrefPair,
    load_preference_pairs,
    split_by_session,
)
from ab_harness.model.protocols import SampleSource
from ab_harness.model.stats import anchor_accuracy, self_agreement
from ab_harness.model.types import ClipSpec, Tier
from ab_harness.worker.generator import ArGenerator, SampleRequest
from ab_harness.worker.loading import LoadedModel, load_checkpoint
from ab_harness.worker.mdm_gen import MdmGenerator
from ab_harness.worker.service import style_vector
from ab_harness.worker.zflow_gen import ZFlowGenerator
from train_ar import ArTransformer, TrackTokens, _build_section, prepare_grid
from train_mdm import MaskedDenoiser, mask_ratio
from train_zflow import FlowCfg, ZFlowModule, clamp_prefix, embed_zq

REPO = Path(__file__).resolve().parent


# ===========================================================================
# Configuration
# ===========================================================================


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
        and reused for the rest of the run. Ignored by the stochastic
        (zflow, MDM) scorers, whose reference changes with every noise draw.
      context_frac (float): share of the window held as clean context before
        the scored frames, capped at the longest prefix the model was trained
        with. Without it a 512-frame window over a 516-frame prompt scores
        nothing, and a later window is scored with no history at all.
      disable_dropout (bool): keep the policy in eval mode. Dropout makes the
        policy/reference ratio noisy even at step 0.
    """

    beta: float = 0.1
    label_smoothing: float = 0.0
    sft_weight: float = 0.0
    length_normalize: bool = False
    trainable_blocks: int = 4
    cache_ref: bool = True
    context_frac: float = 0.5
    disable_dropout: bool = True


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

    A window starting after the clip's first frame opens with `context_frames`
    of the clip's own history, flagged in "prefix" and never scored: that is
    how the samplers chained windows, and it keeps the scored frames
    conditioned on what actually preceded them.

    Args:
      pairs (Sequence[PrefPair]): the comparisons.
      tracks (dict[int, TrackTokens]): corpus by track index, for style vectors.
      bank (ClipBank): token store.
      crop_frames (int): the checkpoint's training context in frames.
      seed (int): base seed for window draws.
      fixed_window (bool): always take the first window instead of a random
        one. Validation uses it so the held-out number does not wobble.
      context_frames (int): clean history held at the head of a window.
      patch (int): window starts and prefix lengths are multiples of this.
    """

    def __init__(
        self,
        pairs: Sequence[PrefPair],
        tracks: dict[int, TrackTokens],
        bank: ClipBank,
        crop_frames: int,
        seed: int = 0,
        fixed_window: bool = False,
        context_frames: int = 0,
        patch: int = 1,
    ) -> None:
        self.pairs = list(pairs)
        self.tracks = tracks
        self.bank = bank
        self.crop_frames = crop_frames
        self.seed = seed
        self.fixed_window = fixed_window
        self.patch = max(1, patch)
        self.context_frames = context_frames - context_frames % self.patch
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
          int: first frame of the scored window, shared by both sides. The
            earliest start is the one whose context just covers the prompt,
            so no window is spent on prompt frames nobody scores.
        """
        slack = pair.n_frames - self.crop_frames
        if slack <= 0:
            return 0
        top = slack - slack % self.patch
        prompt = pair.winner.conditioning.prompt_frames
        low = max(0, prompt - self.context_frames)
        low = min(top, low + (-low) % self.patch)
        if self.fixed_window:
            return low
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        return low + rng.randrange((top - low) // self.patch + 1) * self.patch

    def _side(self, spec: ClipSpec, start: int, length: int) -> dict[str, Any]:
        """
        Args:
          spec (ClipSpec): the clip.
          start (int): first frame of the window.
          length (int): frames to score.

        Returns:
          dict[str, Any]: tokens, the score and prefix masks and the
            conditioning tensors.
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
        cond = spec.conditioning
        held = (
            self.context_frames
            if start > 0
            else min(cond.prompt_frames, self.context_frames)
        )
        held -= held % self.patch
        prefix = torch.arange(length) < held
        absolute = torch.arange(start, start + length)
        score = (absolute >= cond.prompt_frames) & ~prefix
        return {
            "item_id": spec.item_id,
            "tokens": window,
            "score": score,
            "prefix": prefix,
            "track_idx": torch.tensor(cond.id_track, dtype=torch.long),
            "style": style_vector(self.tracks[cond.style_track], spec).float(),
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


class PairScorer(ABC):
    """
    Per-backend likelihood of a side under a model: the only thing DPO needs.

    A stochastic scorer estimates log p from one noise draw. `draw` makes it
    once per batch and the same draw scores winner and loser under policy and
    reference, so the draw's own variance cancels in the DPO margin (the
    Diffusion-DPO recipe).

    Attributes:
      stochastic (bool): scores depend on the draw, so references are never
        cached.
      head_names (tuple[str, ...]): output modules of `net` left trainable by
        freeze_trunk.
      state_prefix (str): key prefix that makes the policy's state_dict load
        as the backend's own checkpoint.
    """

    stochastic: bool = False
    head_names: tuple[str, ...] = ("norm_out", "heads")
    state_prefix: str = "model."

    def draw(
        self, side: dict[str, Any], generator: torch.Generator | None
    ) -> dict[str, torch.Tensor]:
        """
        Args:
          side (dict[str, Any]): one collated side, for shapes and device.
          generator (torch.Generator | None): RNG; None = global.

        Returns:
          dict[str, torch.Tensor]: the noise shared by every score this batch.
        """
        return {}

    def net(self, model: nn.Module) -> nn.Module:
        """
        Args:
          model (nn.Module): policy or reference.

        Returns:
          nn.Module: the transformer holding `blocks` and the heads.
        """
        return model

    @abstractmethod
    def score(
        self, model: nn.Module, side: dict[str, Any], noise: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          model (nn.Module): policy or reference.
          side (dict[str, Any]): one collated side (see PreferenceDataset).
          noise (dict[str, torch.Tensor]): this batch's draw.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: (B,) float32 log-likelihood (or
            its proxy) over scored frames and (B,) float32 scored-unit count.
        """


class ArScorer(PairScorer):
    """Exact autoregressive log-probability; deterministic."""

    def score(
        self, model: nn.Module, side: dict[str, Any], noise: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          model (nn.Module): an ArTransformer.
          side (dict[str, Any]): one collated side.
          noise (dict[str, torch.Tensor]): unused.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: see sequence_logprob.
        """
        assert isinstance(model, ArTransformer)
        return sequence_logprob(model, side)


class ZFlowScorer(PairScorer):
    """
    Diffusion-DPO proxy for a latent flow: -0.5 x velocity error.

    Rectified-flow training minimizes ||v_hat - (x1 - eps)||^2, a weighted ELBO
    up to constants that cancel between policy and reference; summing over
    scored frames and latent dims makes it a sequence-level quantity like the
    AR's summed log-probability.

    Args:
      flow (FlowCfg): the checkpoint's t distribution.
      dims (int): whitened latent width.
    """

    stochastic = True
    head_names = ("norm_out", "ada_out", "out_proj")
    state_prefix = ""

    def __init__(self, flow: FlowCfg, dims: int) -> None:
        self.flow = flow
        self.dims = dims

    def draw(
        self, side: dict[str, Any], generator: torch.Generator | None
    ) -> dict[str, torch.Tensor]:
        """
        Args:
          side (dict[str, Any]): one collated side.
          generator (torch.Generator | None): RNG.

        Returns:
          dict[str, torch.Tensor]: "t" (B,) flow times, "eps" (B, dims, T).
        """
        batch, frames = side["tokens"].shape[:2]
        device = side["tokens"].device
        flow = self.flow
        if flow.t_sampling == "logit_normal":
            normal = torch.randn(batch, generator=generator, device=device)
            t = torch.sigmoid(normal * flow.t_std + flow.t_mean)
        else:
            t = torch.rand(batch, generator=generator, device=device)
        eps = torch.randn(batch, self.dims, frames, generator=generator, device=device)
        return {"t": t.clamp(flow.t_eps, 1.0 - flow.t_eps), "eps": eps}

    def net(self, model: nn.Module) -> nn.Module:
        """
        Args:
          model (nn.Module): a ZFlowModule.

        Returns:
          nn.Module: its LatentDiT.
        """
        assert isinstance(model, ZFlowModule)
        return model.net

    def score(
        self, model: nn.Module, side: dict[str, Any], noise: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          model (nn.Module): a ZFlowModule.
          side (dict[str, Any]): one collated side.
          noise (dict[str, torch.Tensor]): "t" and "eps" from draw.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: (B,) -0.5 x summed squared
            velocity error over scored frames, and (B,) scored elements.
        """
        assert isinstance(model, ZFlowModule)
        z = embed_zq(side["tokens"].long(), model.codebooks)  # type: ignore[arg-type]
        x1 = model.stats.whiten(z.transpose(1, 2))
        eps, t = noise["eps"], noise["t"]
        prefix = side["prefix"].bool()
        x_t = clamp_prefix(model.interpolate(eps, x1, t), x1, prefix)
        v_hat = model.net(
            x_t,
            t,
            side["style"].float(),
            side["drop_style"].bool(),
            side["track_idx"].long(),
            side["drop_id"].bool(),
            prefix if model.net.prefix else None,
        )
        weight = side["score"].float()
        err = (v_hat.float() - (x1 - eps)).pow(2).sum(dim=1)
        return -0.5 * (err * weight).sum(dim=1), weight.sum(dim=1) * x1.shape[1]


class MdmScorer(PairScorer):
    """
    Masked-diffusion proxy: log-probability of one level's masked cells.

    One SoundStorm corruption per batch (level l, ratio, cell pattern) is
    applied identically to both sides; levels above l are fully masked and the
    prefix is never masked, as in train_mdm.MdmModule.corrupt.

    Args:
      schedule (str): the checkpoint's mask schedule.
    """

    stochastic = True
    state_prefix = "net."

    def __init__(self, schedule: str) -> None:
        self.schedule = schedule

    def draw(
        self, side: dict[str, Any], generator: torch.Generator | None
    ) -> dict[str, torch.Tensor]:
        """
        Args:
          side (dict[str, Any]): one collated side.
          generator (torch.Generator | None): RNG.

        Returns:
          dict[str, torch.Tensor]: "level" (B,), "ratio" (B,), "cells" (B, T)
            uniforms deciding which cells of the level are masked.
        """
        batch, frames, levels = side["tokens"].shape
        device = side["tokens"].device
        level = torch.randint(0, levels, (batch,), generator=generator, device=device)
        u = torch.rand(batch, generator=generator, device=device)
        cells = torch.rand(batch, frames, generator=generator, device=device)
        return {"level": level, "ratio": mask_ratio(u, self.schedule), "cells": cells}

    def score(
        self, model: nn.Module, side: dict[str, Any], noise: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          model (nn.Module): a MaskedDenoiser.
          side (dict[str, Any]): one collated side.
          noise (dict[str, torch.Tensor]): the corruption from draw.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: (B,) summed log-probability of the
            masked, scored cells of level l, and (B,) their count.
        """
        assert isinstance(model, MaskedDenoiser)
        tokens = side["tokens"].long()
        batch, frames, levels = tokens.shape
        level = noise["level"].reshape(-1, 1, 1)
        at_level = noise["cells"] < noise["ratio"].unsqueeze(1)
        idx = torch.arange(levels, device=tokens.device).reshape(1, 1, -1)
        masked = (idx > level) | ((idx == level) & at_level.unsqueeze(-1))
        masked = masked & ~side["prefix"].bool().unsqueeze(-1)
        logits = model(
            torch.where(masked, model.mask_id, tokens),
            side["track_idx"].long(),
            side["style"].float(),
            side["drop_id"].bool(),
            side["drop_style"].bool(),
        )
        pick = level.unsqueeze(-1).expand(-1, frames, 1, logits.shape[-1])
        chosen = logits.gather(2, pick).squeeze(2).float()
        truth = tokens.gather(2, level.expand(-1, frames, 1)).squeeze(2)
        logprob = -F.cross_entropy(
            chosen.reshape(-1, chosen.shape[-1]), truth.reshape(-1), reduction="none"
        ).reshape(batch, frames)
        weight = (at_level & side["score"].bool()).float()
        return (logprob * weight).sum(dim=1), weight.sum(dim=1)


def freeze_trunk(
    model: nn.Module,
    trainable_blocks: int,
    heads: Sequence[str] = ("norm_out", "heads"),
) -> tuple[int, int]:
    """
    Leave only the top N blocks and the output modules trainable.

    Args:
      model (nn.Module): the policy's transformer, with a `blocks` list.
      trainable_blocks (int): blocks to keep trainable, counted from the top.
        Values at or above the depth train everything.
      heads (Sequence[str]): attribute names of the output modules.

    Returns:
      tuple[int, int]: trainable and total parameter counts.
    """
    blocks = model.blocks
    assert isinstance(blocks, nn.ModuleList)
    depth = len(blocks)
    if trainable_blocks < depth:
        for param in model.parameters():
            param.requires_grad_(False)
        for block in blocks[depth - max(0, trainable_blocks) :]:
            for param in block.parameters():
                param.requires_grad_(True)
        for name in heads:
            for param in getattr(model, name).parameters():
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
      policy (nn.Module): the model being updated.
      reference (nn.Module): a frozen copy of the same checkpoint.
      cfg (DpoConfig): full config.
      scorer (PairScorer | None): the backend's likelihood; None = AR.
    """

    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        cfg: DpoConfig,
        scorer: PairScorer | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.reference = reference.requires_grad_(False).eval()
        self.cfg = cfg
        self.scorer = scorer if scorer is not None else ArScorer()
        self.cache_ref = cfg.dpo.cache_ref and not self.scorer.stochastic
        self._ref_cache: dict[tuple[str, int], float] = {}

    def train(self, mode: bool = True):  # type: ignore[override]
        """
        Keep the reference in eval no matter what Lightning does to the module,
        and the policy too when dropout is disabled.

        Args:
          mode (bool): training mode for the policy.

        Returns:
          DpoLightningModule: self.
        """
        super().train(mode)
        self.reference.eval()
        if self.cfg.dpo.disable_dropout:
            self.policy.eval()
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
    def _reference_logprob(
        self, side: dict[str, Any], noise: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Reference log-probabilities, memoized per (item, window) when enabled.

        Args:
          side (dict[str, Any]): one collated side.
          noise (dict[str, torch.Tensor]): this batch's scorer draw.

        Returns:
          torch.Tensor: (B,) reference log-probability, already normalized.
        """
        keys = [
            (item, int(start))
            for item, start in zip(side["item_id"], side["start"].tolist())
        ]
        if self.cache_ref and all(k in self._ref_cache for k in keys):
            return torch.tensor(
                [self._ref_cache[k] for k in keys],
                dtype=torch.float32,
                device=side["tokens"].device,
            )
        logprob, counts = self.scorer.score(self.reference, side, noise)
        value = self._normalize(logprob, counts)
        if self.cache_ref:
            for key, item in zip(keys, value.tolist()):
                self._ref_cache[key] = item
        return value

    def _draw(
        self, batch: dict[str, Any], stage: str, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Args:
          batch (dict[str, Any]): collated pair batch.
          stage (str): "train" or "val".
          batch_idx (int): index within the epoch; seeds the val draw so the
            held-out number compares like with like across epochs.

        Returns:
          dict[str, torch.Tensor]: the scorer's noise for both sides.
        """
        generator = None
        if stage == "val":
            device = batch["win"]["tokens"].device
            generator = torch.Generator(device=device)
            generator.manual_seed(self.cfg.train.seed * 7919 + batch_idx)
        return self.scorer.draw(batch["win"], generator)

    def _step(self, batch: dict[str, Any], stage: str, batch_idx: int) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): collated pair batch.
          stage (str): "train" or "val".
          batch_idx (int): index within the epoch.

        Returns:
          torch.Tensor: scalar DPO loss.
        """
        noise = self._draw(batch, stage, batch_idx)
        lp_win, count_win = self.scorer.score(self.policy, batch["win"], noise)
        lp_lose, count_lose = self.scorer.score(self.policy, batch["lose"], noise)
        lp_win = self._normalize(lp_win, count_win)
        lp_lose = self._normalize(lp_lose, count_lose)
        ref_win = self._reference_logprob(batch["win"], noise)
        ref_lose = self._reference_logprob(batch["lose"], noise)

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
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): collated pair batch from held-out sessions.
          batch_idx (int): index within the epoch.

        Returns:
          torch.Tensor: validation loss. val/acc is the number that matters:
            the share of unseen human calls the policy now agrees with.
        """
        return self._step(batch, "val", batch_idx)

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
      generator (SampleSource): sampler wrapping the policy being trained.
      request (SampleRequest): the conditioning to draw from; seeds are varied.
      clips (int): candidates per probe.
      every (int): steps between probes; 0 disables.
      gate (float): halt when disagreement drops below this fraction of the
        step-0 baseline.
      num_tokens (int): codebook size, for entropy normalization.
    """

    def __init__(
        self,
        generator: SampleSource,
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


class PolicyCheckpointWriter(L.Callback):
    """
    Write checkpoints in the backend's own layout, not the DPO one.

    A plain Lightning save would carry "policy." and "reference." prefixes and a
    second copy of the weights, and nothing downstream could load it. Writing
    the reference checkpoint's layout instead -- its hyper_parameters and the
    policy's state under the backend's key prefix -- is what lets the result be
    sampled by the harness that produced the pairs. No optimizer state is
    written, so loaders that prefer EMA weights fall back to these.

    Args:
      save_dir (Path): destination directory; its name must stay in the
        backend's checkpoint family (saved_dpo_*, saved_zflow_*, saved_mdm_*).
      hyper_parameters (dict[str, Any]): the reference checkpoint's.
      state_prefix (str): key prefix of the backend's state_dict.
      monitor (str): validation metric to select the best epoch on.
    """

    def __init__(
        self,
        save_dir: Path,
        hyper_parameters: dict[str, Any],
        state_prefix: str = "model.",
        monitor: str = "val/acc",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.hyper_parameters = hyper_parameters
        self.state_prefix = state_prefix
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
                    f"{self.state_prefix}{k}": v.detach().cpu()
                    for k, v in module.policy.state_dict().items()
                },
                "hyper_parameters": self.hyper_parameters,
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


@dataclass
class PolicyBundle:
    """
    Policy, frozen reference and everything scoring them needs.

    Args:
      policy (nn.Module): the model being trained, wrapped by loaded.generator.
      reference (nn.Module): a frozen copy of the same checkpoint.
      scorer (PairScorer): the backend's likelihood.
      loaded (LoadedModel): the policy's sampler, corpus and manifest.
      window (int): frames per scored window (the training crop).
      context (int): clean history frames held at a window's head.
      patch (int): frame granularity of windows and prefixes.
      hyper_parameters (dict[str, Any]): the reference checkpoint's.
    """

    policy: nn.Module
    reference: nn.Module
    scorer: PairScorer
    loaded: LoadedModel
    window: int
    context: int
    patch: int
    hyper_parameters: dict[str, Any]


def load_policy(cfg: DpoConfig) -> PolicyBundle:
    """
    Load the reference checkpoint twice -- policy and frozen reference -- with
    the scorer and window geometry of its backend.

    Args:
      cfg (DpoConfig): full config.

    Returns:
      PolicyBundle: models, scorer, corpus and geometry.
    """
    checkpoint = cfg.pairs.reference_checkpoint
    gen_cfg = GeneratorCfg(device=cfg.train.device, window_frames=1 << 20)
    loaded = load_checkpoint(checkpoint, gen_cfg)
    frozen = load_checkpoint(checkpoint, gen_cfg, previous=loaded).generator
    hyper_parameters = torch.load(
        REPO / checkpoint, map_location="cpu", mmap=True, weights_only=False
    )["hyper_parameters"]
    generator = loaded.generator
    patch = 1
    scorer: PairScorer
    if isinstance(generator, ZFlowGenerator) and isinstance(frozen, ZFlowGenerator):
        policy, reference = generator.module, frozen.module
        window = generator.window
        model = policy.cfg.model
        scorer = ZFlowScorer(policy.cfg.flow, policy.stats.dims)
        patch = model.patch
        longest = int(model.prefix_max_frac * window) if policy.net.prefix else 0
    elif isinstance(generator, MdmGenerator) and isinstance(frozen, MdmGenerator):
        policy, reference = generator.net, frozen.net
        window = generator.window
        scorer = MdmScorer(generator.mask_cfg.schedule)
        frac = float(hyper_parameters["cfg"]["model"]["prefix_max_frac"])
        longest = int(frac * window)
    elif isinstance(generator, ArGenerator) and isinstance(frozen, ArGenerator):
        policy, reference = generator.model, frozen.model
        window = generator.window_frames
        scorer = ArScorer()
        longest = window - 1
    else:
        raise TypeError(f"no DPO scorer for {type(generator).__name__}")
    context = min(longest, int(cfg.dpo.context_frac * window))
    return PolicyBundle(
        policy=policy,
        reference=reference,
        scorer=scorer,
        loaded=loaded,
        window=window,
        context=context - context % patch,
        patch=patch,
        hyper_parameters=hyper_parameters,
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
        track_idx=cond.id_track,
        style=style_vector(tracks[cond.style_track], spec).float(),
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


def run_metrics(trainer: L.Trainer, writer: PolicyCheckpointWriter) -> dict[str, float]:
    """
    Summarize a finished run for pairs.json, which dpo_round.py and the harness
    read instead of the tensorboard logs.

    Args:
      trainer (L.Trainer): the trainer after fit.
      writer (PolicyCheckpointWriter): holds the best monitored score.

    Returns:
      dict[str, float]: best/final monitored score, final val loss and epochs run.
    """
    final = trainer.callback_metrics
    out = {"epochs": float(trainer.current_epoch)}
    if writer.best > -float("inf"):
        out["best_" + writer.monitor.replace("/", "_")] = writer.best
    for key in (writer.monitor, "val/loss"):
        if key in final:
            out["final_" + key.replace("/", "_")] = float(final[key])
    return out


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
    parser.add_argument("--bank-root", default=None, help="override pairs.bank_root")
    parser.add_argument(
        "--save-path", default=None, help="override train.save_path (<date> expands)"
    )
    return parser.parse_args()


def main() -> None:
    """Run the pair report, then DPO training unless --dry-run."""
    args = parse_args()
    cfg = load_config(args.config)
    if args.min_pairs is not None:
        cfg.pairs.min_pairs = args.min_pairs
    if args.device is not None:
        cfg.train.device = args.device
    if args.bank_root is not None:
        cfg.pairs.bank_root = args.bank_root
    if args.save_path is not None:
        cfg.train.save_path = args.save_path
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

    bundle = load_policy(cfg)
    loaded, scorer = bundle.loaded, bundle.scorer
    bank.require_tokenizer(loaded.cache_dir.name)
    net = scorer.net(bundle.policy)
    trainable, total_params = freeze_trunk(
        net, cfg.dpo.trainable_blocks, scorer.head_names
    )
    print(
        f"{loaded.backend} policy {total_params/1e6:.1f}M params, "
        f"{trainable/1e6:.1f}M trainable "
        f"(top {cfg.dpo.trainable_blocks} of "
        f"{len(list(net.get_submodule('blocks').children()))} blocks); "
        f"window {bundle.window}, context {bundle.context}"
    )

    device = loaded.device
    tracks = loaded.by_idx

    def dataset(pairs: list[PrefPair], fixed: bool) -> PreferenceDataset:
        return PreferenceDataset(
            pairs,
            tracks,
            bank,
            bundle.window,
            seed=cfg.train.seed,
            fixed_window=fixed,
            context_frames=bundle.context,
            patch=bundle.patch,
        )

    train_set = dataset(train_pairs, fixed=False)
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
        val_set = dataset(val_pairs, fixed=True)
        loaders["val_dataloaders"] = DataLoader(
            val_set,
            batch_sampler=LengthBucketSampler(
                val_set, cfg.train.batch_pairs, shuffle=False
            ),
            collate_fn=collate_pairs,
            num_workers=cfg.train.num_workers,
        )

    module = DpoLightningModule(bundle.policy, bundle.reference, cfg, scorer)
    save_dir = REPO / cfg.train.save_path.replace(
        "<date>", date.today().strftime("%Y%m%d")
    )
    writer = PolicyCheckpointWriter(
        save_dir,
        bundle.hyper_parameters,
        scorer.state_prefix,
        "val/acc" if val_pairs else "train/acc",
    )
    callbacks: list[L.Callback] = [writer]
    if cfg.train.diversity_every:
        callbacks.append(
            DiversityMonitor(
                loaded.generator,
                diversity_request(
                    train_pairs,
                    tracks,
                    cfg.train.diversity_seconds,
                    loaded.fps,
                ),
                clips=cfg.train.diversity_clips,
                every=cfg.train.diversity_every,
                gate=cfg.train.diversity_gate,
                num_tokens=int(loaded.meta["num_tokens"]),
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
                "metrics": run_metrics(trainer, writer),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
