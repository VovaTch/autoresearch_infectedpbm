"""
Masked discrete diffusion over the RVQ token grid -- the non-autoregressive
alternative to train_ar.py, trained on the very same data and conditionals.

A bidirectional transformer sees a (T, R) grid where some cells are the MASK
token and predicts every masked cell at once (MaskGIT); levels are handled
coarse-to-fine as in SoundStorm: to train, one level l is drawn per sample,
levels below it are given clean, level l is masked at a random cosine ratio
and every level above it is fully masked. Sampling then fills level 0 in a
few confidence-ranked rounds, then level 1, then level 2.

Conditioning is the AR stage's: the track id and the aligned 512-frame style
descriptor, here as two memory tokens every block cross-attends to
(train_zflow.CondMemory), each dropped independently in training so
classifier-free guidance has a real null branch. A random clean prefix is never
masked and never scored, so a prompt continues natively; the apps chain
windows through ab_harness.worker.mdm_gen.MdmGenerator.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python train_mdm.py --config config_mdm_512.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from train import EMA, TimeOneCycleLR  # noqa: E402
from train_ar import ArConfig, SwiGLU, TrackTokens, load_token_cache  # noqa: E402
from train_ar import load_config as load_ar_config  # noqa: E402
from train_ar import token_cache_dir  # noqa: E402
from train_zflow import (  # noqa: E402
    Attention,
    CondMemory,
    CrossAttention,
    apply_ema,
    ar_dataloaders,
    param_groups,
    rope_cache,
)

# ===========================================================================
# Config
# ===========================================================================


@dataclass
class DataCfg:
    """Where the crops come from: the AR stage's config, verbatim."""

    ar_config: str = "config_ar_512_aligned.yaml"


@dataclass
class ModelCfg:
    """Transformer geometry and conditioning."""

    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    mlp_hidden: int = 1408
    dropout: float = 0.1
    style_bottleneck: int = 128
    rope_theta: float = 10000.0
    # Track-id vocabulary; filled from the token-cache manifest at build time.
    num_tracks: int = 0
    # Independent drop probability of the id token and the style token.
    p_drop_cond: float = 0.5
    # A random leading fraction of each crop, up to this, is never masked and
    # never scored: the prefix the sampler will later be given as a prompt.
    prefix_max_frac: float = 0.5


@dataclass
class MaskCfg:
    """Corruption schedule and the sampler's defaults / validation read-out."""

    # Masking ratio r = cos(pi/2 * u), u ~ U(0, 1): the MaskGIT schedule.
    schedule: str = "cosine"
    # Refinement rounds per RVQ level at sampling time.
    steps: list[int] = field(default_factory=lambda: [16, 8, 8])
    # Gumbel noise scale on the confidence ranking, annealed to 0 by the last
    # round of each level (MaskGIT's choice temperature).
    choice_temperature: float = 4.5
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    cfg_scale: float = 1.0
    val_samples: int = 8


@dataclass
class TrainCfg:
    """Optimiser, schedule and checkpointing, matching the repo's conventions."""

    devices: int | list[int] = 1
    minutes: float = 480.0
    lr: float = 3.0e-4
    lr_pct_start: float = 0.05
    lr_div_factor: float = 25.0
    batch_size: int = 32
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 4
    ema_decay: float | None = 0.999
    seed: int = 42
    save_path: str = "saved_mdm/"
    checkpoint: str | None = None


@dataclass
class MdmConfig:
    """Top-level config, one block per YAML section."""

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


SECTIONS: dict[str, type] = {
    "data": DataCfg,
    "model": ModelCfg,
    "mask": MaskCfg,
    "train": TrainCfg,
}


def _build_section(cls: type, raw: dict[str, Any] | None, name: str):
    """
    Instantiate one config dataclass, rejecting unknown keys.

    Args:
      cls (type): the dataclass to build.
      raw (dict[str, Any] | None): the YAML section, or None if absent.
      name (str): section name, for the error message.

    Returns:
      object: an instance of cls.
    """
    raw = raw or {}
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown key(s) in '{name}': {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> MdmConfig:
    """
    Args:
      path (str | Path): path to the .yaml file.

    Returns:
      MdmConfig: fully populated config with defaults filled in.
    """
    with open(path, "r") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")
    return MdmConfig(
        **{
            name: _build_section(cls, raw.get(name), name)
            for name, cls in SECTIONS.items()
        }
    )


def config_from_dict(raw: dict[str, Any]) -> MdmConfig:
    """
    Args:
      raw (dict[str, Any]): the asdict() form written at training time.

    Returns:
      MdmConfig: reconstructed config.
    """
    return MdmConfig(**{name: cls(**raw[name]) for name, cls in SECTIONS.items()})


def ar_config(cfg: MdmConfig) -> ArConfig:
    """
    Args:
      cfg (MdmConfig): full config.

    Returns:
      ArConfig: the AR-stage config the data comes from.
    """
    path = Path(cfg.data.ar_config).expanduser()
    return load_ar_config(path if path.is_absolute() else REPO / path)


# ===========================================================================
# Masking
# ===========================================================================


def mask_ratio(u: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    """
    Fraction of cells masked at progress u.

    Args:
      u (torch.Tensor): (...) values in [0, 1]; 0 = everything masked.
      schedule (str): "cosine" (MaskGIT) or "linear".

    Returns:
      torch.Tensor: (...) ratios in [0, 1].
    """
    if schedule == "cosine":
        return torch.cos(u * math.pi / 2)
    if schedule == "linear":
        return 1.0 - u
    raise ValueError(f"unknown mask schedule {schedule!r}")


def prefix_mask(lengths: torch.Tensor, frames: int) -> torch.Tensor:
    """
    Args:
      lengths (torch.Tensor): (B,) prefix lengths in frames.
      frames (int): crop length.

    Returns:
      torch.Tensor: (B, frames) bool, True on the first `lengths[b]` frames.
    """
    return torch.arange(frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(
        1
    )


Rng = torch.Generator | Sequence[torch.Generator] | None


def uniform(shape: tuple[int, ...], rng: Rng, device: torch.device) -> torch.Tensor:
    """
    Args:
      shape (tuple[int, ...]): (B, ...) output shape.
      rng (Rng): one generator for the batch, one per lane (len B), or None.
      device (torch.device): target device.

    Returns:
      torch.Tensor: shape of U(0, 1) draws; with per-lane generators lane b's
        draws depend only on generator b, whatever else shares the batch.
    """
    if rng is None or isinstance(rng, torch.Generator):
        return torch.rand(shape, generator=rng, device=device)
    if len(rng) != shape[0]:
        raise ValueError(f"{len(rng)} generators for a batch of {shape[0]}")
    return torch.stack([torch.rand(shape[1:], generator=g, device=device) for g in rng])


def sample_tokens(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Rng = None,
) -> torch.Tensor:
    """
    Draw one index per row of logits; generate_ar.sample_step over any batch.

    Args:
      logits (torch.Tensor): (B, ..., V) logits.
      temperature (float): softmax temperature; <= 0 selects argmax.
      top_k (int): keep only the k highest logits (0 disables).
      top_p (float): nucleus threshold (0 disables).
      generator (Rng): RNG for reproducible draws; a sequence gives lane b its
        own generator (draws by inverse CDF so the batch never interleaves).

    Returns:
      torch.Tensor: (B, ...) int64 sampled indices.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p > 0:
        ordered, order = logits.sort(dim=-1, descending=True)
        probs = ordered.softmax(dim=-1)
        drop = probs.cumsum(dim=-1) - probs > top_p
        ordered = ordered.masked_fill(drop, float("-inf"))
        logits = ordered.gather(-1, order.argsort(dim=-1))
    probs = logits.softmax(dim=-1)
    if generator is None or isinstance(generator, torch.Generator):
        flat = probs.reshape(-1, probs.shape[-1])
        draw = torch.multinomial(flat, num_samples=1, generator=generator)
        return draw.reshape(logits.shape[:-1])
    # inverse-CDF sampling from per-lane uniforms
    u = uniform(tuple(probs.shape[:-1]) + (1,), generator, probs.device)
    cdf = probs.cumsum(dim=-1)
    draw = (cdf < u).sum(dim=-1)
    return draw.clamp_max(probs.shape[-1] - 1)


# ===========================================================================
# Model
# ===========================================================================


class MdmBlock(nn.Module):
    """
    Pre-norm block: bidirectional self-attention, cross-attention to the
    conditioning memory, SwiGLU. No adaLN -- the mask pattern carries the
    "time" of the diffusion, so nothing global needs injecting.

    Args:
      d_model (int): model width.
      n_heads (int): attention heads.
      mlp_hidden (int): SwiGLU inner width.
      dropout (float): dropout on attention weights and residual branches.
    """

    def __init__(
        self, d_model: int, n_heads: int, mlp_hidden: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.norm_x = nn.LayerNorm(d_model)
        self.xattn = CrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, T, D) tokens.
          cos (torch.Tensor): (T, Dh // 2) rotary cosines.
          sin (torch.Tensor): (T, Dh // 2) rotary sines.
          memory (torch.Tensor): (B, M, D) conditioning tokens.

        Returns:
          torch.Tensor: (B, T, D) tokens.
        """
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop(self.xattn(self.norm_x(x), memory))
        return x + self.drop(self.mlp(self.norm2(x)))


class MaskedDenoiser(nn.Module):
    """
    Bidirectional transformer predicting every cell of a partly masked grid.

    Args:
      cfg (ModelCfg): geometry.
      num_tokens (int): codebook size per level; index num_tokens is MASK.
      num_rq (int): RVQ levels R.
      style_dim (int): style descriptor width.
    """

    def __init__(
        self, cfg: ModelCfg, num_tokens: int, num_rq: int, style_dim: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_tokens = num_tokens
        self.num_rq = num_rq
        self.mask_id = num_tokens
        d = cfg.d_model
        self.embed = nn.ModuleList(
            [nn.Embedding(num_tokens + 1, d) for _ in range(num_rq)]
        )
        self.memory = CondMemory(d, cfg.num_tracks, style_dim, cfg.style_bottleneck)
        self.blocks = nn.ModuleList(
            [
                MdmBlock(d, cfg.n_heads, cfg.mlp_hidden, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, num_tokens) for _ in range(num_rq)])
        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None

    def rope(
        self, length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          length (int): frames in the sequence.
          device (torch.device): target device.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: cos and sin, each (length, Dh // 2).
        """
        cached = self._rope
        if cached is None or cached[0].shape[0] < length or cached[0].device != device:
            head_dim = self.cfg.d_model // self.cfg.n_heads
            cached = rope_cache(
                length, head_dim, self.cfg.rope_theta, device, torch.float32
            )
            self._rope = cached
        return cached[0][:length], cached[1][:length]

    def forward(
        self,
        tokens: torch.Tensor,
        track_idx: torch.Tensor | None,
        style: torch.Tensor | None,
        drop_id: torch.Tensor | None = None,
        drop_style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
          tokens (torch.Tensor): (B, T, R) int64 grid with MASK ids.
          track_idx (torch.Tensor | None): (B,) ids; None = null id.
          style (torch.Tensor | None): (B, style_dim) descriptors; None = null.
          drop_id (torch.Tensor | None): (B,) bool id-drop mask.
          drop_style (torch.Tensor | None): (B,) bool style-drop mask.

        Returns:
          torch.Tensor: (B, T, R, N) logits per cell.
        """
        batch, frames, levels = tokens.shape
        h = self.embed[0](tokens[:, :, 0])
        for level in range(1, levels):
            h = h + self.embed[level](tokens[:, :, level])
        memory = self.memory(track_idx, style, drop_id, drop_style, batch=batch)
        cos, sin = self.rope(frames, h.device)
        for block in self.blocks:
            h = block(h, cos, sin, memory)
        h = self.norm_out(h)
        return torch.stack([head(h) for head in self.heads], dim=2)

    def guided_logits(
        self,
        tokens: torch.Tensor,
        track_idx: torch.Tensor | None,
        style: torch.Tensor | None,
        cfg_scale: float | torch.Tensor,
        drop_id: torch.Tensor | None = None,
        drop_style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Logits with classifier-free guidance, l_null + w (l_cond - l_null).

        Args:
          tokens (torch.Tensor): (B, T, R) grid with MASK ids.
          track_idx (torch.Tensor | None): (B,) ids.
          style (torch.Tensor | None): (B, style_dim) descriptors.
          cfg_scale (float | torch.Tensor): 1 = conditional only; the null
            branch drops both streams. A (B,) tensor guides per lane.
          drop_id (torch.Tensor | None): (B,) lanes whose id is nulled even in
            the conditional branch.
          drop_style (torch.Tensor | None): (B,) same for the style.

        Returns:
          torch.Tensor: (B, T, R, N) logits.
        """
        plain = isinstance(cfg_scale, float) and cfg_scale == 1.0
        if plain or (track_idx is None and style is None):
            return self(tokens, track_idx, style, drop_id, drop_style)
        batch = tokens.shape[0]
        zeros = torch.zeros(batch, dtype=torch.bool, device=tokens.device)
        keep_id = zeros if drop_id is None else drop_id
        keep_st = zeros if drop_style is None else drop_style
        drop = torch.ones_like(zeros)
        both = self(
            torch.cat([tokens, tokens]),
            None if track_idx is None else torch.cat([track_idx, track_idx]),
            None if style is None else torch.cat([style, style]),
            torch.cat([keep_id, drop]),
            torch.cat([keep_st, drop]),
        )
        cond, null = both.float().chunk(2)
        w = (
            cfg_scale.reshape(-1, 1, 1, 1)
            if isinstance(cfg_scale, torch.Tensor)
            else cfg_scale
        )
        return null + w * (cond - null)

    @torch.no_grad()
    def sample(
        self,
        frames: int,
        track_idx: torch.Tensor | None,
        style: torch.Tensor | None,
        mask_cfg: MaskCfg,
        prefix: torch.Tensor | None = None,
        prefix_frames: torch.Tensor | None = None,
        cfg_scale: float | torch.Tensor | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        steps: list[int] | None = None,
        generator: Rng = None,
        batch: int | None = None,
        progress: Callable[[int, int], None] | None = None,
        drop_id: torch.Tensor | None = None,
        drop_style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fill a grid level by level with MaskGIT rounds.

        Each round predicts every still-masked cell of the current level,
        samples them, ranks the draws by log-probability plus annealed Gumbel
        noise, keeps the most confident and re-masks the rest according to the
        cosine schedule; the last round keeps everything. Prefix frames stay
        as given at every level.

        Args:
          frames (int): grid length T.
          track_idx (torch.Tensor | None): (B,) ids.
          style (torch.Tensor | None): (B, style_dim) descriptors.
          mask_cfg (MaskCfg): sampler defaults.
          prefix (torch.Tensor | None): (B, T, R) tokens whose first
            `prefix_frames[b]` frames are held fixed.
          prefix_frames (torch.Tensor | None): (B,) prefix lengths.
          cfg_scale (float | torch.Tensor | None): guidance, per lane when a
            (B,) tensor; config default when None.
          temperature (float | None): sampling temperature; default when None.
          top_k (int | None): top-k truncation; default when None.
          top_p (float | None): nucleus threshold; default when None.
          steps (list[int] | None): rounds per level; default when None.
          generator (Rng): RNG for reproducible draws, one per lane or shared.
          batch (int | None): batch size when nothing else fixes it.
          progress (Callable[[int, int], None] | None): (round, total) calls.
          drop_id (torch.Tensor | None): (B,) lanes sampling with a null id.
          drop_style (torch.Tensor | None): (B,) lanes with a null style.

        Returns:
          torch.Tensor: (B, T, R) int64 grid, fully unmasked.
        """
        cfg_scale = mask_cfg.cfg_scale if cfg_scale is None else cfg_scale
        temperature = mask_cfg.temperature if temperature is None else temperature
        top_k = mask_cfg.top_k if top_k is None else top_k
        top_p = mask_cfg.top_p if top_p is None else top_p
        steps = list(mask_cfg.steps) if steps is None else list(steps)
        if len(steps) != self.num_rq:
            raise ValueError(f"steps must list {self.num_rq} rounds, got {steps}")
        ref = next((v for v in (track_idx, style, prefix) if v is not None), None)
        if ref is None:
            if batch is None:
                raise ValueError("batch is required without conditioning or prefix")
            size = batch
        else:
            size = ref.shape[0]
        device = self.mask_id_device()
        tokens = torch.full(
            (size, frames, self.num_rq), self.mask_id, dtype=torch.int64, device=device
        )
        held = torch.zeros(size, frames, dtype=torch.bool, device=device)
        if prefix is not None and prefix_frames is not None:
            held = prefix_mask(prefix_frames.to(device), frames)
            tokens = torch.where(held.unsqueeze(-1), prefix.to(device), tokens)
        total = sum(steps)
        done = 0
        for level in range(self.num_rq):
            unknown = ~held
            for k in range(steps[level]):
                logits = self.guided_logits(
                    tokens, track_idx, style, cfg_scale, drop_id, drop_style
                )[:, :, level]
                drawn = sample_tokens(logits, temperature, top_k, top_p, generator)
                logp = (
                    logits.float()
                    .log_softmax(dim=-1)
                    .gather(-1, drawn.unsqueeze(-1))
                    .squeeze(-1)
                )
                anneal = 1.0 - (k + 1) / steps[level]
                noise = uniform(tuple(logp.shape), generator, device).clamp_min(1e-20)
                conf = logp - mask_cfg.choice_temperature * anneal * torch.log(
                    -torch.log(noise)
                )
                conf = conf.masked_fill(~unknown, float("inf"))
                tokens[:, :, level] = torch.where(unknown, drawn, tokens[:, :, level])
                done += 1
                if progress is not None:
                    progress(done, total)
                if k + 1 == steps[level]:
                    break
                ratio = mask_ratio(
                    torch.tensor((k + 1) / steps[level]), mask_cfg.schedule
                )
                n_mask = (unknown.sum(dim=1).float() * ratio).floor().long()
                rank = conf.argsort(dim=1).argsort(dim=1)
                remask = rank < n_mask.unsqueeze(1)
                tokens[:, :, level] = torch.where(
                    remask, self.mask_id, tokens[:, :, level]
                )
                unknown = remask
        return tokens

    def mask_id_device(self) -> torch.device:
        """
        Returns:
          torch.device: where the parameters live.
        """
        return self.embed[0].weight.device


# ===========================================================================
# Lightning module
# ===========================================================================


class MdmModule(L.LightningModule):
    """
    Masked-diffusion training over AR-stage crops.

    Args:
      cfg (MdmConfig): full config.
      num_tokens (int): codebook size per level.
      num_rq (int): RVQ levels.
      style_dim (int): style descriptor width.
    """

    def __init__(
        self, cfg: MdmConfig, num_tokens: int, num_rq: int, style_dim: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = MaskedDenoiser(cfg.model, num_tokens, num_rq, style_dim)
        self.save_hyperparameters(
            {
                "cfg": asdict(cfg),
                "num_tokens": num_tokens,
                "num_rq": num_rq,
                "style_dim": style_dim,
            }
        )

    def corrupt(
        self, tokens: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SoundStorm-style corruption of a clean grid.

        Args:
          tokens (torch.Tensor): (B, T, R) clean grid.
          generator (torch.Generator | None): RNG for reproducible draws.

        Returns:
          tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the corrupted grid
            (B, T, R), the (B, T, R) bool mask of cells to score (level l's
            masked cells), and the (B,) level drawn per sample.
        """
        batch, frames, levels = tokens.shape
        device = tokens.device

        def rand(*shape: int) -> torch.Tensor:
            return torch.rand(*shape, generator=generator, device=device)

        level = torch.randint(0, levels, (batch,), generator=generator, device=device)
        longest = int(self.cfg.model.prefix_max_frac * frames)
        lengths = torch.randint(
            0, longest + 1, (batch,), generator=generator, device=device
        )
        held = prefix_mask(lengths, frames)
        ratio = mask_ratio(rand(batch), self.cfg.mask.schedule)
        at_level = rand(batch, frames) < ratio.unsqueeze(1)
        idx = torch.arange(levels, device=device).reshape(1, 1, -1)
        lv = level.reshape(-1, 1, 1)
        masked = (idx > lv) | ((idx == lv) & at_level.unsqueeze(-1))
        masked = masked & ~held.unsqueeze(-1)
        corrupted = torch.where(masked, self.net.mask_id, tokens)
        target = masked & (idx == lv)
        return corrupted, target, level

    def _run(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): AR-stage loader batch.
          stage (str): "train" or "val".

        Returns:
          torch.Tensor: () cross-entropy over the scored cells.
        """
        tokens = batch["tokens"].long()
        style = batch["style"].float()
        track_idx = batch["track_idx"].long()
        corrupted, target, level = self.corrupt(tokens)
        target = target & batch["score_mask"].bool().unsqueeze(-1)
        drop_id = drop_style = None
        if stage == "train":
            p = self.cfg.model.p_drop_cond
            drop_id = torch.rand(tokens.shape[0], device=tokens.device) < p
            drop_style = torch.rand(tokens.shape[0], device=tokens.device) < p
        logits = self.net(corrupted, track_idx, style, drop_id, drop_style)
        # only the drawn level is scored: gather it before the (B, T, N) float
        # copy cross-entropy makes, a third of the full grid's memory
        pick = level.reshape(-1, 1, 1, 1).expand(
            -1, tokens.shape[1], 1, logits.shape[-1]
        )
        chosen = logits.gather(2, pick).squeeze(2)
        truth = tokens.gather(2, level.reshape(-1, 1, 1).expand(-1, tokens.shape[1], 1))
        truth = truth.squeeze(2)
        ce = F.cross_entropy(
            chosen.reshape(-1, chosen.shape[-1]), truth.reshape(-1), reduction="none"
        ).reshape(truth.shape)
        scored = target.any(dim=-1).float()  # (B, T): masked cells of level l
        loss = (ce * scored).sum() / scored.sum().clamp_min(1.0)
        hit = (chosen.argmax(dim=-1) == truth).float()
        on_step = stage == "train"
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
        )
        for d in range(tokens.shape[-1]):
            w = scored * (level == d).float().unsqueeze(1)
            count = w.sum().clamp_min(1.0)
            self.log(
                f"{stage}/ce_l{d}",
                (ce * w).sum() / count,
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}/acc_l{d}",
                (hit * w).sum() / count,
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def training_step(self, batch: dict[str, Any], index: int) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): loader batch.
          index (int): batch index, unused.

        Returns:
          torch.Tensor: () loss.
        """
        return self._run(batch, "train")

    def validation_step(self, batch: dict[str, Any], index: int) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): loader batch.
          index (int): batch index; the sampler read-out runs on the first only.

        Returns:
          torch.Tensor: () loss.
        """
        loss = self._run(batch, "val")
        if index == 0:
            self._log_sampler_metrics(batch)
        return loss

    @torch.no_grad()
    def _log_sampler_metrics(self, batch: dict[str, Any]) -> None:
        """
        Continue held-out crops from a clean quarter and log the per-level
        token agreement on the generated part -- weak (many continuations are
        valid) but free and monotone in training.

        Args:
          batch (dict[str, Any]): loader batch.
        """
        n = min(self.cfg.mask.val_samples, batch["tokens"].shape[0])
        tokens = batch["tokens"][:n].long()
        frames = tokens.shape[1]
        gen = torch.Generator(device=tokens.device).manual_seed(self.cfg.train.seed)
        lengths = torch.full((n,), frames // 4, device=tokens.device)
        out = self.net.sample(
            frames,
            batch["track_idx"][:n].long(),
            batch["style"][:n].float(),
            self.cfg.mask,
            prefix=tokens,
            prefix_frames=lengths,
            generator=gen,
        )
        scored = ~prefix_mask(lengths, frames)
        agree = ((out == tokens) & scored.unsqueeze(-1)).float().sum(dim=(0, 1))
        agree = agree / scored.sum().clamp_min(1)
        for d in range(agree.shape[0]):
            self.log(
                f"val/prefix_tok_agree_l{d}", agree[d], on_epoch=True, sync_dist=True
            )

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Returns:
          dict[str, Any]: AdamW (decay on matmul weights only) + wall-clock
            one-cycle schedule.
        """
        decay, no_decay = param_groups(self.net)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.train.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.train.lr,
            betas=(0.9, 0.95),
        )
        scheduler = TimeOneCycleLR(
            optimizer,
            total_minutes=self.cfg.train.minutes,
            max_lr=self.cfg.train.lr,
            pct_start=self.cfg.train.lr_pct_start,
            div_factor=self.cfg.train.lr_div_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ===========================================================================
# Entry points
# ===========================================================================


def load_mdm_module(
    ckpt_path: Path, ema: bool = True, device: str = "cpu"
) -> MdmModule:
    """
    Rebuild a trained module from its checkpoint.

    Args:
      ckpt_path (Path): Lightning checkpoint.
      ema (bool): apply the EMA weights when present.
      device (str): target device.

    Returns:
      MdmModule: eval-mode module on `device`.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    module = MdmModule(
        config_from_dict(hp["cfg"]),
        int(hp["num_tokens"]),
        int(hp["num_rq"]),
        int(hp["style_dim"]),
    )
    module.load_state_dict(ckpt["state_dict"], strict=False)
    if ema:
        print(f"EMA weights applied: {apply_ema(module.net, ckpt)}")
    return module.to(device).eval()


def build_trainer(cfg: MdmConfig) -> L.Trainer:
    """
    Args:
      cfg (MdmConfig): full config.

    Returns:
      L.Trainer: configured trainer.
    """
    save_path = REPO / cfg.train.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.Timer(duration={"minutes": cfg.train.minutes}),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="mdm_best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="mdm_latest",
            monitor=None,
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
            save_on_exception=True,
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    if cfg.train.ema_decay:
        callbacks.append(EMA(decay=cfg.train.ema_decay))
    return L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        max_epochs=-1,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.grad_clip,
        callbacks=callbacks,
        logger=L.pytorch.loggers.TensorBoardLogger(str(save_path), name="mdm"),
        log_every_n_steps=10,
    )


def load_corpus(cfg: MdmConfig) -> tuple[ArConfig, list[TrackTokens], dict[str, Any]]:
    """
    Args:
      cfg (MdmConfig): full config.

    Returns:
      tuple[ArConfig, list[TrackTokens], dict[str, Any]]: the AR config, the
        token cache and its manifest.
    """
    ar_cfg = ar_config(cfg)
    tracks, manifest = load_token_cache(token_cache_dir(ar_cfg), ar_cfg.data)
    return ar_cfg, tracks, manifest


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "config_mdm_512.yaml"))
    return parser.parse_args()


def main() -> None:
    """Train on the AR stage's crops."""
    args = parse_args()
    cfg = load_config(args.config)
    L.seed_everything(cfg.train.seed, workers=True)
    ar_cfg, tracks, manifest = load_corpus(cfg)
    meta = manifest["tokenizer_meta"]
    cfg.model.num_tracks = len(manifest["tracks"])
    train_loader, val_loader = ar_dataloaders(
        ar_cfg, tracks, cfg.train.batch_size, cfg.train.num_workers
    )
    module = MdmModule(
        cfg,
        int(meta["num_tokens"]),
        int(meta["num_rq"]),
        int(tracks[0].style.shape[-1]),
    )
    params = sum(p.numel() for p in module.net.parameters())
    crop = ar_cfg.data.crop_frames
    print(
        f"tracks {len(tracks)}  crop {crop}f = {crop / tracks[0].fps:.1f} s  "
        f"ids {cfg.model.num_tracks}  prefix<= {cfg.model.prefix_max_frac}  "
        f"params {params / 1e6:.1f}M  val crops {len(val_loader.dataset)}"  # type: ignore[arg-type]
    )
    if cfg.train.checkpoint:
        state = torch.load(
            REPO / cfg.train.checkpoint, map_location="cpu", weights_only=False
        )
        missing = module.load_state_dict(state["state_dict"], strict=False)
        print(f"warm start from {cfg.train.checkpoint}: {missing}")
    build_trainer(cfg).fit(module, train_loader, val_loader)
    print(f"done -> {REPO / cfg.train.save_path}")


if __name__ == "__main__":
    main()
