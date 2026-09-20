"""
Flow-matching prior over the tokenizer's quantized latent, for IM-flavored
style transfer by SDEdit.

The tokenizer is near-identity on foreign material (roundtrip_external.py), so
it adds no style of its own. This trains an unconditional / style-conditioned
rectified-flow DiT on the corpus latent z_q (the sum of the three RVQ codebook
vectors per frame, read straight from the AR token cache), whitened by a PCA
basis measured on the training tracks. Stylising a foreign clip is then SDEdit:
noise its latent part-way, integrate the learned flow back to t=1, requantize
to tokens and decode with the frozen ONNX decoder (stylize_zflow.py).

Latent geometry: 172.27 frames/s, token_dim 1024. A crop of 2048 frames is
11.9 s; with temporal patch 2 the transformer sees 1024 tokens.

Validation runs on HELD-OUT TRACKS only (never in-track windows): with ~6 h of
audio a generative model memorises timbre before it generalises, and in-track
validation cannot see that (finding_flow_phase_fail).

Usage:
    uv run python train_zflow.py --build-stats            # PCA stats only
    uv run python train_zflow.py --build-stats --probe-pca  # + k sweep
    CUDA_VISIBLE_DEVICES=1 uv run python train_zflow.py --config config_zflow.yaml
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from train import EMA, TimeOneCycleLR  # noqa: E402
from train_ar import RMSNorm, SwiGLU, TrackTokens, apply_rope, rope_cache  # noqa: E402
from train_ar import DataCfg as ArDataCfg  # noqa: E402
from train_ar import load_token_cache  # noqa: E402
from train_flow import plan_held_out, timestep_embedding  # noqa: E402

# ===========================================================================
# Config
# ===========================================================================


@dataclass
class TokenizerCfg:
    """Frozen tokenizer artefacts and the token cache the latents come from."""

    token_cache: str = "~/.cache/infected_pbm/tokens_88cacecb1e81"
    checkpoint: str = "saved_20260914_dsteps2_24h/last.ckpt"
    meta: str = "onnx/tokenizer_meta.json"
    encoder_onnx: str = "onnx/encoder.onnx"
    decoder_onnx: str = "onnx/decoder.onnx"
    tracks_dir: str = "~/.cache/infected_pbm/tracks"
    slices_dir: str = "~/.cache/infected_pbm/slices"
    chunk_frames: int = 4096
    margin: int = 256


@dataclass
class DataCfg:
    """Cropping, the latent whitening and the held-out track split."""

    crop_frames: int = 2048
    held_out_tracks: int = 5
    split_seed: int = 1234
    steps_per_epoch: int = 500
    val_crops_per_track: int = 8
    # PCA rank kept. Measured 2026-09-15 on z_q: 90% variance in 6 dims, 99% in
    # 23, 99.97% in 128; token agreement after whiten/unwhiten/requantize(beam 8)
    # is 98.7/98.5/97.5% at k=128 vs 99.2/98.8/98.9% at the full 1024.
    pca_dims: int = 128
    # Eigenvalues below floor * max are clamped before whitening so the
    # near-null tail is not amplified into noise (dim 45 ~ 1e-4 of dim 0).
    pca_eig_floor: float = 1.0e-4
    style_context_frames: int = 256
    single_track: str | None = None


@dataclass
class ModelCfg:
    """DiT geometry."""

    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    patch: int = 2
    mlp_hidden: int = 1408
    dropout: float = 0.1
    t_embed_dim: int = 256
    style_bottleneck: int = 128
    # Fraction of training samples whose style vector is replaced by the learned
    # null embedding, so the unconditional branch CFG needs is actually trained.
    p_drop_style: float = 0.2
    rope_theta: float = 10000.0


@dataclass
class FlowCfg:
    """The interpolant, the time distribution and the sampler."""

    t_sampling: str = "logit_normal"
    t_mean: float = 0.0
    t_std: float = 1.0
    t_eps: float = 1.0e-3
    steps: int = 32
    # 0 = deterministic Euler ODE; 1 = every step re-draws the noise part of the
    # iterate (stochastic re-anchoring); between = partial re-draw.
    churn: float = 0.0
    cfg_scale: float = 1.0
    # SDEdit strength used for the validation read-out.
    sdedit_strength: float = 0.5
    val_samples: int = 8


@dataclass
class TrainCfg:
    """Optimiser, schedule and checkpointing, matching the repo's conventions."""

    devices: int | list[int] = 1
    minutes: float = 480.0
    lr: float = 3.0e-4
    lr_pct_start: float = 0.05
    lr_div_factor: float = 25.0
    batch_size: int = 16
    accumulate_grad_batches: int = 1
    precision: str = "32-true"
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 4
    ema_decay: float | None = 0.999
    seed: int = 42
    save_path: str = "saved_zflow/"
    checkpoint: str | None = None


@dataclass
class ZFlowConfig:
    """Top-level config, one block per YAML section."""

    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    flow: FlowCfg = field(default_factory=FlowCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


SECTIONS: dict[str, type] = {
    "tokenizer": TokenizerCfg,
    "data": DataCfg,
    "model": ModelCfg,
    "flow": FlowCfg,
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
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown key(s) in '{name}': {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> ZFlowConfig:
    """
    Read a latent-flow config from YAML.

    Args:
      path (str | Path): path to the .yaml file.

    Returns:
      ZFlowConfig: fully populated config with defaults filled in.
    """
    with open(path, "r") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")
    return ZFlowConfig(
        **{
            name: _build_section(cls, raw.get(name), name)
            for name, cls in SECTIONS.items()
        }
    )


def config_from_dict(raw: dict[str, Any]) -> ZFlowConfig:
    """
    Rebuild a config from the dict stashed in a checkpoint's hyperparameters.

    Args:
      raw (dict[str, Any]): the asdict() form written at training time.

    Returns:
      ZFlowConfig: reconstructed config.
    """
    return ZFlowConfig(**{name: cls(**raw[name]) for name, cls in SECTIONS.items()})


# ===========================================================================
# Codebook ops
# ===========================================================================


def embed_zq(tokens: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """
    Sum the per-level codebook vectors at the given indices.

    Args:
      tokens (torch.Tensor): (..., T, R) int64 indices.
      codebooks (torch.Tensor): (R, N, C) per-level codebooks.

    Returns:
      torch.Tensor: (..., T, C) float32 quantized latent z_q.
    """
    out = torch.zeros(*tokens.shape[:-1], codebooks.shape[-1], device=tokens.device)
    for level in range(codebooks.shape[0]):
        out = out + codebooks[level][tokens[..., level]]
    return out


def requantize(
    z: torch.Tensor, codebooks: torch.Tensor, chunk: int = 8192, beam: int = 8
) -> torch.Tensor:
    """
    Assign a latent to RVQ indices whose codes sum closest to it.

    beam = 1 is train.RQCodeBook.apply_codebook: each level takes the nearest
    code to the residual left by the previous levels. That greedy rule is not
    idempotent on a SUM of codes -- on exact corpus z_q it flips ~15% of level-0
    tokens (measured 2026-09-15). beam > 1 keeps that many partial assignments
    per frame and returns the one with the smallest final residual: beam 8
    agrees on 99% of tokens at 1.6% latent error, 3x the cost of greedy.

    Args:
      z (torch.Tensor): (B, C, T) latent.
      codebooks (torch.Tensor): (R, N, C) per-level codebooks.
      chunk (int): frames scored per cdist call, to bound memory.
      beam (int): partial assignments kept per frame.

    Returns:
      torch.Tensor: (B, T, R) int64 indices.
    """
    batch, dim, frames = z.shape
    levels, codes, _ = codebooks.shape
    flat = z.transpose(1, 2).reshape(-1, dim).float()
    out = torch.empty(flat.shape[0], levels, dtype=torch.int64, device=z.device)
    chunk = max(1, chunk // beam)
    for start in range(0, flat.shape[0], chunk):
        res = flat[start : start + chunk].unsqueeze(1)  # (M, beams, C)
        idx = torch.empty(res.shape[0], 1, 0, dtype=torch.int64, device=z.device)
        for level in range(levels):
            m, width, _ = res.shape
            dist = torch.cdist(res.reshape(m * width, dim), codebooks[level])
            dist = dist.reshape(m, width * codes)
            keep = beam if level < levels - 1 else 1
            best = dist.topk(min(keep, width * codes), dim=1, largest=False).indices
            src, code = best // codes, best % codes  # (M, keep)
            idx = torch.cat(
                [
                    idx.gather(1, src.unsqueeze(-1).expand(-1, -1, level)),
                    code.unsqueeze(-1),
                ],
                dim=-1,
            )
            res = (
                res.gather(1, src.unsqueeze(-1).expand(-1, -1, dim))
                - codebooks[level][code]
            )
        out[start : start + chunk] = idx[:, 0]
    return out.reshape(batch, frames, -1)


def style_vector(z_q: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """
    Mean latent over a span, L2-normalized -- the AR stage's style descriptor.

    Args:
      z_q (torch.Tensor): (..., T, C) latent.
      valid (torch.Tensor | None): (..., T) bool mask of frames to include.

    Returns:
      torch.Tensor: (..., C) unit-norm descriptor.
    """
    if valid is None:
        return F.normalize(z_q.mean(dim=-2), dim=-1)
    weights = valid.to(z_q.dtype).unsqueeze(-1)
    mean = (z_q * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
    return F.normalize(mean, dim=-1)


# ===========================================================================
# Latent whitening
# ===========================================================================


@dataclass
class LatentStats:
    """
    PCA whitening of z_q, measured on the training tracks.

    whiten: x = ((z - mean) @ basis[:, :k]) * scale ; unwhiten inverts it. With
    k = C this is a rotation plus per-direction scaling and loses nothing; the
    eigenvalue floor keeps the near-null tail from being blown up to unit
    variance.

    Args:
      mean (torch.Tensor): (C,) latent mean.
      basis (torch.Tensor): (C, C) eigenvectors as columns, variance descending.
      eigvals (torch.Tensor): (C,) matching variances.
      dims (int): rank kept.
      eig_floor (float): relative floor on eigenvalues before whitening.
    """

    mean: torch.Tensor
    basis: torch.Tensor
    eigvals: torch.Tensor
    dims: int
    eig_floor: float

    @property
    def scale(self) -> torch.Tensor:
        """
        Returns:
          torch.Tensor: (dims,) per-direction whitening factors.
        """
        eig = self.eigvals[: self.dims].clamp_min(
            self.eig_floor * float(self.eigvals[0])
        )
        return eig.rsqrt()

    def to(self, device: torch.device | str) -> "LatentStats":
        """
        Args:
          device (torch.device | str): target device.

        Returns:
          LatentStats: a copy with every tensor moved.
        """
        return LatentStats(
            self.mean.to(device),
            self.basis.to(device),
            self.eigvals.to(device),
            self.dims,
            self.eig_floor,
        )

    def whiten(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
          z (torch.Tensor): (B, C, T) latent.

        Returns:
          torch.Tensor: (B, dims, T) whitened coordinates.
        """
        proj = (z.transpose(1, 2) - self.mean) @ self.basis[:, : self.dims]
        return (proj * self.scale).transpose(1, 2)

    def unwhiten(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, dims, T) whitened coordinates.

        Returns:
          torch.Tensor: (B, C, T) latent.
        """
        proj = x.transpose(1, 2) / self.scale
        return (proj @ self.basis[:, : self.dims].T + self.mean).transpose(1, 2)

    def save(self, path: Path) -> None:
        """
        Args:
          path (Path): destination .pt file (dims / floor are NOT stored; they
            come from config at load time).
        """
        torch.save(
            {"mean": self.mean, "basis": self.basis, "eigvals": self.eigvals}, path
        )

    @classmethod
    def load(cls, path: Path, dims: int, eig_floor: float) -> "LatentStats":
        """
        Args:
          path (Path): file written by save().
          dims (int): rank to keep.
          eig_floor (float): relative eigenvalue floor.

        Returns:
          LatentStats: loaded stats.
        """
        blob = torch.load(path, map_location="cpu", weights_only=True)
        return cls(blob["mean"], blob["basis"], blob["eigvals"], dims, eig_floor)


def cache_dir(cfg: ZFlowConfig) -> Path:
    """
    Args:
      cfg (ZFlowConfig): full config.

    Returns:
      Path: the token cache directory.
    """
    return Path(cfg.tokenizer.token_cache).expanduser()


def stats_path(cfg: ZFlowConfig) -> Path:
    """
    Args:
      cfg (ZFlowConfig): full config.

    Returns:
      Path: where the PCA stats for this split live, inside the token cache.
    """
    tag = f"seed{cfg.data.split_seed}_ho{cfg.data.held_out_tracks}"
    if cfg.data.single_track:
        tag += "_single"
    return cache_dir(cfg) / f"zflow_pca_{tag}.pt"


def load_corpus(cfg: ZFlowConfig) -> tuple[list[TrackTokens], list[int], torch.Tensor]:
    """
    Load the token cache, the held-out split and the codebooks.

    Args:
      cfg (ZFlowConfig): full config.

    Returns:
      tuple[list[TrackTokens], list[int], torch.Tensor]: all tracks, held-out
        track indices, and (R, N, C) codebooks.
    """
    root = cache_dir(cfg)
    tracks, manifest = load_token_cache(
        root, ArDataCfg(single_track=cfg.data.single_track)
    )
    held = plan_held_out(manifest["tracks"], cfg.data)  # type: ignore[arg-type]
    if cfg.data.single_track:
        held = []
    codebooks = torch.load(root / "codebooks.pt", map_location="cpu", weights_only=True)
    return tracks, held, codebooks.float()


def build_latent_stats(cfg: ZFlowConfig, force: bool = False) -> Path:
    """
    Measure the PCA basis of z_q over the TRAINING tracks and cache it.

    Mean and scatter accumulate in float64 over every frame of every training
    track (the same estimator as pca_truncate.pca_basis); held-out tracks are
    excluded so the whitening carries nothing about them.

    Args:
      cfg (ZFlowConfig): full config.
      force (bool): recompute even if the file exists.

    Returns:
      Path: the stats file.
    """
    path = stats_path(cfg)
    if path.exists() and not force:
        print(f"latent stats present: {path}")
        return path
    tracks, held, codebooks = load_corpus(cfg)
    dim = codebooks.shape[-1]
    total = torch.zeros(dim, dtype=torch.float64)
    scatter = torch.zeros(dim, dim, dtype=torch.float64)
    count = 0
    for track in tracks:
        if track.track_idx in held:
            continue
        z = embed_zq(track.tokens, codebooks).double()
        total += z.sum(0)
        scatter += z.T @ z
        count += z.shape[0]
    mean = total / count
    cov = scatter / count - torch.outer(mean, mean)
    eig, vecs = torch.linalg.eigh(cov)
    stats = LatentStats(
        mean.float(),
        vecs.flip(1).float(),
        eig.flip(0).clamp_min(0.0).float(),
        cfg.data.pca_dims,
        cfg.data.pca_eig_floor,
    )
    stats.save(path)
    cum = stats.eigvals.cumsum(0) / stats.eigvals.sum()
    d90 = int((cum < 0.90).sum()) + 1
    d99 = int((cum < 0.99).sum()) + 1
    print(
        f"latent stats -> {path}: {count} frames from {len(tracks) - len(held)} tracks "
        f"(held out {held}); 90% variance in {d90} dims, 99% in {d99}"
    )
    return path


def probe_pca(
    cfg: ZFlowConfig, ks: Sequence[int] = (64, 128, 256, 512, 1024), crops: int = 200
) -> None:
    """
    Token agreement after whiten -> unwhiten -> requantize at several ranks.

    Answers which `pca_dims` is safe: a rank that flips level-0 codes has thrown
    away information the decoder needs, before any model is trained.

    Args:
      cfg (ZFlowConfig): full config.
      ks (Sequence[int]): ranks to test.
      crops (int): random 512-frame crops scored (held-out tracks included).
    """
    tracks, _, codebooks = load_corpus(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codebooks = codebooks.to(device)
    rng = random.Random(cfg.data.split_seed)
    frames = 512
    picks = []
    for _ in range(crops):
        track = rng.choice(tracks)
        start = rng.randint(0, max(0, track.num_frames - frames))
        picks.append(track.tokens[start : start + frames])
    tokens = torch.stack(picks).to(device)  # (N, T, R)
    z = embed_zq(tokens, codebooks).transpose(1, 2)  # (N, C, T)
    num_rq = codebooks.shape[0]
    print(
        f"{'k':>6} "
        + " ".join(f"{'L' + str(r) + ' agree':>9}" for r in range(num_rq))
        + f" {'latent rel err':>15}"
    )
    for k in ks:
        stats = LatentStats.load(stats_path(cfg), k, cfg.data.pca_eig_floor).to(device)
        back = stats.unwhiten(stats.whiten(z))
        rel = float((back - z).norm() / z.norm())
        idx = requantize(back, codebooks)
        agree = (idx == tokens).float().mean(dim=(0, 1))
        print(
            f"{k:>6} "
            + " ".join(f"{100 * float(a):>8.2f}%" for a in agree)
            + f" {rel:>15.5f}"
        )


# ===========================================================================
# Datasets
# ===========================================================================


def _span_tokens(
    track: TrackTokens, start: int, crop: int, ctx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokens for [start - ctx, start + crop + ctx), zero-filled outside the track.

    Args:
      track (TrackTokens): source stream.
      start (int): first frame of the crop.
      crop (int): crop length in frames.
      ctx (int): extra frames on each side for the style descriptor.

    Returns:
      tuple[torch.Tensor, torch.Tensor]: (crop + 2*ctx, R) int64 tokens and a
        (crop + 2*ctx,) bool mask of frames that exist.
    """
    lo, hi = start - ctx, start + crop + ctx
    length = hi - lo
    tokens = torch.zeros(length, track.tokens.shape[1], dtype=torch.int64)
    valid = torch.zeros(length, dtype=torch.bool)
    src_lo, src_hi = max(0, lo), min(track.num_frames, hi)
    tokens[src_lo - lo : src_hi - lo] = track.tokens[src_lo:src_hi]
    valid[src_lo - lo : src_hi - lo] = True
    return tokens, valid


class LatentCropDataset(Dataset):
    """
    Random crops from the training tracks, length-weighted, patch-aligned.

    Items carry TOKENS, not latents: the codebook gather and whitening run on
    the GPU in ZFlowModule.prepare, so the loader moves ~30 KB per item.

    Args:
      tracks (list[TrackTokens]): training tracks.
      cfg (DataCfg): crop settings.
      patch (int): crop offsets are multiples of this.
      seed (int): base seed; each epoch index derives its own stream.
    """

    def __init__(
        self, tracks: list[TrackTokens], cfg: DataCfg, patch: int, seed: int
    ) -> None:
        self.tracks = [t for t in tracks if t.num_frames >= cfg.crop_frames]
        if not self.tracks:
            raise ValueError("no track is long enough for crop_frames")
        self.cfg = cfg
        self.patch = patch
        self.seed = seed
        self.weights = [float(t.num_frames) for t in self.tracks]
        self.length = cfg.steps_per_epoch

    def set_length(self, length: int) -> None:
        """
        Args:
          length (int): items per epoch (steps x batch).
        """
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(f"{self.seed}:{index}:{torch.initial_seed()}")
        track = rng.choices(self.tracks, weights=self.weights, k=1)[0]
        limit = (track.num_frames - self.cfg.crop_frames) // self.patch
        start = rng.randint(0, limit) * self.patch
        tokens, valid = _span_tokens(
            track, start, self.cfg.crop_frames, self.cfg.style_context_frames
        )
        return {"tokens": tokens, "valid": valid}


class ValLatentDataset(Dataset):
    """
    Evenly spaced crops on the held-out tracks, fixed across epochs.

    Args:
      tracks (list[TrackTokens]): held-out tracks.
      cfg (DataCfg): crop settings.
      patch (int): crop offsets are multiples of this.
    """

    def __init__(self, tracks: list[TrackTokens], cfg: DataCfg, patch: int) -> None:
        self.cfg = cfg
        self.items: list[tuple[TrackTokens, int]] = []
        for track in tracks:
            span = track.num_frames - cfg.crop_frames
            if span < 0:
                continue
            for i in range(cfg.val_crops_per_track):
                start = int(span * (i + 0.5) / cfg.val_crops_per_track)
                self.items.append((track, start - start % patch))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track, start = self.items[index]
        tokens, valid = _span_tokens(
            track, start, self.cfg.crop_frames, self.cfg.style_context_frames
        )
        return {"tokens": tokens, "valid": valid}


def build_dataloaders(
    cfg: ZFlowConfig, tracks: list[TrackTokens], held: Sequence[int]
) -> tuple[DataLoader, DataLoader]:
    """
    Args:
      cfg (ZFlowConfig): full config.
      tracks (list[TrackTokens]): every cached track.
      held (Sequence[int]): held-out track indices.

    Returns:
      tuple[DataLoader, DataLoader]: train and validation loaders.
    """
    train_tracks = [t for t in tracks if t.track_idx not in held]
    val_tracks = [t for t in tracks if t.track_idx in held]
    if not val_tracks:
        val_tracks = train_tracks[:1]
        print("WARNING: no held-out tracks; validation reuses a training track")
    train_ds = LatentCropDataset(
        train_tracks, cfg.data, cfg.model.patch, cfg.train.seed
    )
    train_ds.set_length(cfg.data.steps_per_epoch * cfg.train.batch_size)
    val_ds = ValLatentDataset(val_tracks, cfg.data, cfg.model.patch)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


# ===========================================================================
# Model
# ===========================================================================


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    adaLN modulation.

    Args:
      x (torch.Tensor): (B, L, D) normalized activations.
      shift (torch.Tensor): (B, D) additive term.
      scale (torch.Tensor): (B, D) multiplicative term around 1.

    Returns:
      torch.Tensor: (B, L, D) modulated activations.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """
    Bidirectional multi-head self-attention with qk-RMSNorm and RoPE.

    Args:
      d_model (int): model width.
      n_heads (int): heads; d_model must divide evenly.
      dropout (float): attention dropout during training.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, L, D) input.
          cos (torch.Tensor): (L, Dh // 2) rotary cosines.
          sin (torch.Tensor): (L, Dh // 2) rotary sines.

        Returns:
          torch.Tensor: (B, L, D) output.
        """
        batch, length, _ = x.shape
        q, k, v = (
            self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        )
        q = apply_rope(self.q_norm(q).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_norm(k).transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(batch, length, -1))


class DiTBlock(nn.Module):
    """
    Pre-norm transformer block with adaLN-Zero conditioning.

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
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = Attention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = SwiGLU(d_model, mlp_hidden)
        self.drop = nn.Dropout(dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, L, D) tokens.
          c (torch.Tensor): (B, D) conditioning vector.
          cos (torch.Tensor): (L, Dh // 2) rotary cosines.
          sin (torch.Tensor): (L, Dh // 2) rotary sines.

        Returns:
          torch.Tensor: (B, L, D) tokens.
        """
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.drop(
            self.attn(modulate(self.norm1(x), sh1, sc1), cos, sin)
        )
        x = x + g2.unsqueeze(1) * self.drop(self.mlp(modulate(self.norm2(x), sh2, sc2)))
        return x


class LatentDiT(nn.Module):
    """
    1-D DiT predicting the flow velocity of a whitened latent sequence.

    Frames are grouped `patch` at a time into one token; the time and style
    conditioning enter every block through adaLN-Zero, so an untrained network
    outputs exactly zero.

    Args:
      cfg (ModelCfg): geometry.
      in_dim (int): whitened latent width per frame.
      style_dim (int): style-vector width (token_dim).
    """

    def __init__(self, cfg: ModelCfg, in_dim: int, style_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_dim = in_dim
        d = cfg.d_model
        self.in_proj = nn.Linear(in_dim * cfg.patch, d)
        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.t_embed_dim, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, cfg.style_bottleneck),
            nn.SiLU(),
            nn.Linear(cfg.style_bottleneck, d),
        )
        self.null_style = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(d, cfg.n_heads, cfg.mlp_hidden, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out_proj = nn.Linear(d, in_dim * cfg.patch)
        nn.init.zeros_(self.ada_out[1].weight)
        nn.init.zeros_(self.ada_out[1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None

    def rope(
        self, length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          length (int): tokens in the sequence.
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

    def condition(
        self, t: torch.Tensor, style: torch.Tensor | None, drop: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Args:
          t (torch.Tensor): (B,) flow times.
          style (torch.Tensor | None): (B, style_dim) descriptors; None = all null.
          drop (torch.Tensor | None): (B,) bool, True replaces that sample's
            style with the null embedding.

        Returns:
          torch.Tensor: (B, D) conditioning vector.
        """
        c = self.t_mlp(timestep_embedding(t, self.cfg.t_embed_dim))
        null = self.null_style.unsqueeze(0).expand(c.shape[0], -1)
        if style is None:
            return c + null
        s = self.style_mlp(style)
        if drop is not None:
            s = torch.where(drop.unsqueeze(1), null, s)
        return c + s

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor | None = None,
        drop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, in_dim, T) noisy latent, T divisible by patch.
          t (torch.Tensor): (B,) flow times in [0, 1].
          style (torch.Tensor | None): (B, style_dim) descriptors.
          drop (torch.Tensor | None): (B,) bool style-drop mask.

        Returns:
          torch.Tensor: (B, in_dim, T) predicted velocity.
        """
        batch, dim, frames = x.shape
        patch = self.cfg.patch
        if frames % patch:
            raise ValueError(f"frames {frames} not divisible by patch {patch}")
        h = self.in_proj(x.transpose(1, 2).reshape(batch, frames // patch, dim * patch))
        c = self.condition(t, style, drop)
        cos, sin = self.rope(h.shape[1], h.device)
        for block in self.blocks:
            h = block(h, c, cos, sin)
        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        h = self.out_proj(modulate(self.norm_out(h), shift, scale))
        return h.reshape(batch, frames, dim).transpose(1, 2)


# ===========================================================================
# Flow module
# ===========================================================================


def param_groups(net: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Split parameters the way the optimiser (and therefore the EMA state) sees them.

    Args:
      net (nn.Module): the network.

    Returns:
      tuple[list[nn.Parameter], list[nn.Parameter]]: (decay, no_decay) lists,
        each in named_parameters order.
    """
    decay, no_decay = [], []
    for _, param in net.named_parameters():
        if param.requires_grad:
            (no_decay if param.ndim < 2 else decay).append(param)
    return decay, no_decay


def apply_ema(net: nn.Module, ckpt: dict[str, Any]) -> bool:
    """
    Copy EMA weights from EMAOptimizer state over the raw weights.

    Lightning saves raw weights in state_dict; the EMA lives in
    optimizer_states[0]["ema"], ordered as the optimiser's param groups.

    Args:
      net (nn.Module): network whose parameters are overwritten in place.
      ckpt (dict[str, Any]): the loaded checkpoint.

    Returns:
      bool: True if EMA weights were found and applied.
    """
    decay, no_decay = param_groups(net)
    params = decay + no_decay
    for state in ckpt.get("optimizer_states") or []:
        ema = state.get("ema") if isinstance(state, dict) else None
        if ema is not None and len(ema) == len(params):
            with torch.no_grad():
                for p, e in zip(params, ema):
                    p.copy_(e.to(dtype=p.dtype))
            return True
    return False


class ZFlowModule(L.LightningModule):
    """
    Rectified flow between Gaussian noise and the whitened latent.

        x_t = (1 - t) * eps + t * x1,   target v = x1 - eps

    The network predicts v; Euler steps of the ODE dx/dt = v move noise (t=0)
    to data (t=1). SDEdit starts the walk at t0 < 1 from a noised source latent.

    Args:
      cfg (ZFlowConfig): full config.
      stats (LatentStats): whitening of z_q.
      codebooks (torch.Tensor): (R, N, C) codebooks, for the validation read-out.
    """

    def __init__(
        self, cfg: ZFlowConfig, stats: LatentStats, codebooks: torch.Tensor
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = LatentDiT(cfg.model, stats.dims, int(codebooks.shape[-1]))
        self.register_buffer("codebooks", codebooks.float(), persistent=False)
        self.register_buffer("pca_mean", stats.mean, persistent=True)
        self.register_buffer("pca_basis", stats.basis, persistent=True)
        self.register_buffer("pca_eigvals", stats.eigvals, persistent=True)
        self.save_hyperparameters({"cfg": asdict(cfg)})

    @property
    def stats(self) -> LatentStats:
        """
        Returns:
          LatentStats: whitening on the module's current device.
        """
        return LatentStats(
            self.pca_mean,
            self.pca_basis,
            self.pca_eigvals,  # type: ignore[arg-type]
            self.cfg.data.pca_dims,
            self.cfg.data.pca_eig_floor,
        )

    # ------------------------------------------------------------------ data

    def prepare(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokens -> whitened crop latent, style descriptor and source tokens.

        Args:
          batch (dict[str, torch.Tensor]): {"tokens": (B, crop + 2*ctx, R),
            "valid": (B, crop + 2*ctx)}.

        Returns:
          tuple[torch.Tensor, torch.Tensor, torch.Tensor]: x1 (B, dims, crop),
            style (B, C), crop tokens (B, crop, R).
        """
        tokens, valid = batch["tokens"], batch["valid"]
        ctx = self.cfg.data.style_context_frames
        crop = tokens.shape[1] - 2 * ctx
        z = embed_zq(tokens, self.codebooks)  # type: ignore[arg-type]
        style = style_vector(z, valid)
        crop_tokens = tokens[:, ctx : ctx + crop]
        x1 = self.stats.whiten(z[:, ctx : ctx + crop].transpose(1, 2))
        return x1, style, crop_tokens

    # ------------------------------------------------------------------ flow

    def sample_t(self, batch: int, device: torch.device) -> torch.Tensor:
        """
        Args:
          batch (int): draws.
          device (torch.device): target device.

        Returns:
          torch.Tensor: (B,) times in [t_eps, 1 - t_eps].
        """
        flow = self.cfg.flow
        if flow.t_sampling == "logit_normal":
            t = torch.sigmoid(
                torch.randn(batch, device=device) * flow.t_std + flow.t_mean
            )
        elif flow.t_sampling == "uniform":
            t = torch.rand(batch, device=device)
        else:
            raise ValueError(f"unknown t_sampling '{flow.t_sampling}'")
        return t.clamp(flow.t_eps, 1.0 - flow.t_eps)

    @staticmethod
    def interpolate(
        eps: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          eps (torch.Tensor): (B, D, T) noise.
          x1 (torch.Tensor): (B, D, T) data.
          t (torch.Tensor): (B,) times.

        Returns:
          torch.Tensor: (B, D, T) the iterate x_t.
        """
        tt = t.reshape(-1, 1, 1)
        return (1.0 - tt) * eps + tt * x1

    def velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor | None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Predicted velocity, with classifier-free guidance on the style vector.

        Args:
          x (torch.Tensor): (B, D, T) iterate.
          t (torch.Tensor): (B,) times.
          style (torch.Tensor | None): (B, C) descriptors; None = unconditional.
          cfg_scale (float): 1 = conditional only; w > 1 extrapolates
            v_u + w * (v_c - v_u).

        Returns:
          torch.Tensor: (B, D, T) velocity.
        """
        if style is None or cfg_scale == 1.0:
            return self.net(x, t, style)
        both = self.net(
            torch.cat([x, x]),
            torch.cat([t, t]),
            torch.cat([style, style]),
            torch.cat(
                [
                    torch.zeros_like(t, dtype=torch.bool),
                    torch.ones_like(t, dtype=torch.bool),
                ]
            ),
        )
        v_c, v_u = both.chunk(2)
        return v_u + cfg_scale * (v_c - v_u)

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        t_start: float = 0.0,
        style: torch.Tensor | None = None,
        steps: int | None = None,
        churn: float | None = None,
        cfg_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Integrate the flow from t_start to 1.

        Each step estimates the endpoints from the velocity, x1_hat = x + (1-t) v
        and eps_hat = x - t v, then re-forms the iterate at the next time with
        eps_mix = sqrt(1 - churn^2) * eps_hat + churn * eps_new. churn = 0 is
        exactly the Euler ODE step; churn = 1 re-draws the noise entirely
        (stochastic re-anchoring, every iterate on the training interpolant).

        Args:
          x (torch.Tensor): (B, D, T) iterate at t_start (pure noise at 0).
          t_start (float): time of x.
          style (torch.Tensor | None): (B, C) descriptors.
          steps (int | None): Euler steps; config default when None.
          churn (float | None): noise re-draw fraction; config default when None.
          cfg_scale (float | None): guidance; config default when None.
          generator (torch.Generator | None): noise source for churn > 0.

        Returns:
          torch.Tensor: (B, D, T) sample at t=1.
        """
        flow = self.cfg.flow
        steps = flow.steps if steps is None else steps
        churn = flow.churn if churn is None else churn
        cfg_scale = flow.cfg_scale if cfg_scale is None else cfg_scale
        if t_start >= 1.0:
            return x
        grid = torch.linspace(t_start, 1.0, steps + 1, device=x.device)
        batch = x.shape[0]
        for i in range(steps):
            t, t_next = grid[i], grid[i + 1]
            v = self.velocity(x, t.expand(batch), style, cfg_scale)
            x1_hat = x + (1.0 - t) * v
            eps_hat = x - t * v
            if churn > 0.0:
                fresh = torch.randn(
                    x.shape, generator=generator, device=x.device, dtype=x.dtype
                )
                eps_hat = math.sqrt(1.0 - churn * churn) * eps_hat + churn * fresh
            x = (1.0 - t_next) * eps_hat + t_next * x1_hat
        return x

    @torch.no_grad()
    def sdedit(
        self,
        x_src: torch.Tensor,
        strength: float,
        style: torch.Tensor | None = None,
        steps: int | None = None,
        churn: float | None = None,
        cfg_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Noise a source latent to t0 = 1 - strength and integrate back to 1.

        Args:
          x_src (torch.Tensor): (B, D, T) whitened source latent.
          strength (float): 0 returns the source; 1 is a free sample.
          style (torch.Tensor | None): (B, C) descriptors.
          steps (int | None): Euler steps over [t0, 1].
          churn (float | None): see sample().
          cfg_scale (float | None): see sample().
          generator (torch.Generator | None): noise source.

        Returns:
          torch.Tensor: (B, D, T) edited latent.
        """
        t0 = 1.0 - strength
        if t0 >= 1.0:
            return x_src
        eps = torch.randn(
            x_src.shape, generator=generator, device=x_src.device, dtype=x_src.dtype
        )
        x = self.interpolate(
            eps, x_src, torch.full((x_src.shape[0],), t0, device=x_src.device)
        )
        return self.sample(x, t0, style, steps, churn, cfg_scale, generator)

    # ------------------------------------------------------------- training

    def _run(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """
        Args:
          batch (dict[str, torch.Tensor]): loader batch.
          stage (str): "train" or "val".

        Returns:
          torch.Tensor: () MSE between predicted and true velocity.
        """
        x1, style, _ = self.prepare(batch)
        eps = torch.randn_like(x1)
        t = self.sample_t(x1.shape[0], x1.device)
        x_t = self.interpolate(eps, x1, t)
        drop = None
        if stage == "train":
            drop = (
                torch.rand(x1.shape[0], device=x1.device) < self.cfg.model.p_drop_style
            )
        v_hat = self.net(x_t, t, style, drop)
        loss = F.mse_loss(v_hat, x1 - eps)
        on_step = stage == "train"
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], index: int) -> torch.Tensor:
        """
        Args:
          batch (dict[str, torch.Tensor]): loader batch.
          index (int): batch index, unused.

        Returns:
          torch.Tensor: () loss.
        """
        return self._run(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], index: int
    ) -> torch.Tensor:
        """
        Args:
          batch (dict[str, torch.Tensor]): loader batch.
          index (int): batch index; the sampler read-out runs on the first only.

        Returns:
          torch.Tensor: () loss.
        """
        loss = self._run(batch, "val")
        if index == 0:
            self._log_sampler_metrics(batch)
        return loss

    @torch.no_grad()
    def _log_sampler_metrics(self, batch: dict[str, torch.Tensor]) -> None:
        """
        What the sampler delivers on held-out material, decoder-free.

        SDEdit at the config strength: how many source tokens survive per level
        (content retention) and the whitened-latent MSE. An unconditional draw:
        per-dimension std against the data's ~1, a collapse guard.

        Args:
          batch (dict[str, torch.Tensor]): loader batch.
        """
        n = min(self.cfg.flow.val_samples, batch["tokens"].shape[0])
        sub = {k: v[:n] for k, v in batch.items()}
        x1, style, tokens = self.prepare(sub)
        gen = torch.Generator(device=x1.device).manual_seed(self.cfg.train.seed)
        edited = self.sdedit(x1, self.cfg.flow.sdedit_strength, style, generator=gen)
        idx = requantize(self.stats.unwhiten(edited), self.codebooks)  # type: ignore[arg-type]
        agree = (idx == tokens).float().mean(dim=(0, 1))
        for level in range(agree.shape[0]):
            self.log(
                f"val/sdedit_tok_agree_l{level}",
                agree[level],
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            "val/sdedit_latent_mse",
            F.mse_loss(edited, x1),
            on_epoch=True,
            sync_dist=True,
        )
        noise = torch.randn(x1.shape, generator=gen, device=x1.device)
        free = self.sample(noise, 0.0, style, generator=gen)
        self.log(
            "val/sample_std", free.std(dim=(0, 2)).mean(), on_epoch=True, sync_dist=True
        )
        self.log(
            "val/data_std", x1.std(dim=(0, 2)).mean(), on_epoch=True, sync_dist=True
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """
        AdamW with decay only on matmul weights, plus the repo's wall-clock cycle.

        Returns:
          dict[str, Any]: Lightning optimizer/scheduler bundle.
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


def load_zflow_module(
    ckpt_path: Path, ema: bool = True, device: str = "cpu"
) -> ZFlowModule:
    """
    Rebuild a trained module from its checkpoint.

    Args:
      ckpt_path (Path): Lightning checkpoint.
      ema (bool): apply the EMA weights when present.
      device (str): target device.

    Returns:
      ZFlowModule: eval-mode module on `device`.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ckpt["hyper_parameters"]["cfg"])
    state = ckpt["state_dict"]
    stats = LatentStats(
        state["pca_mean"],
        state["pca_basis"],
        state["pca_eigvals"],
        cfg.data.pca_dims,
        cfg.data.pca_eig_floor,
    )
    codebooks = torch.load(
        cache_dir(cfg) / "codebooks.pt", map_location="cpu", weights_only=True
    )
    module = ZFlowModule(cfg, stats, codebooks.float())
    module.load_state_dict(state, strict=False)
    if ema:
        print(f"EMA weights applied: {apply_ema(module.net, ckpt)}")
    return module.to(device).eval()


def build_trainer(cfg: ZFlowConfig) -> L.Trainer:
    """
    Args:
      cfg (ZFlowConfig): full config.

    Returns:
      L.Trainer: configured trainer.
    """
    save_path = REPO / cfg.train.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.Timer(duration={"minutes": cfg.train.minutes}),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="zflow_best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="zflow_latest",
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
        logger=L.pytorch.loggers.TensorBoardLogger(str(save_path), name="zflow"),
        log_every_n_steps=10,
    )


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "config_zflow.yaml"))
    parser.add_argument(
        "--build-stats", action="store_true", help="measure PCA stats and exit"
    )
    parser.add_argument(
        "--probe-pca", action="store_true", help="token agreement per PCA rank"
    )
    parser.add_argument(
        "--force", action="store_true", help="recompute the stats if present"
    )
    return parser.parse_args()


def main() -> None:
    """Measure the latent stats if needed, then train."""
    args = parse_args()
    cfg = load_config(args.config)
    path = build_latent_stats(cfg, force=args.force)
    if args.probe_pca:
        probe_pca(cfg)
    if args.build_stats or args.probe_pca:
        return

    L.seed_everything(cfg.train.seed, workers=True)
    tracks, held, codebooks = load_corpus(cfg)
    stats = LatentStats.load(path, cfg.data.pca_dims, cfg.data.pca_eig_floor)
    train_loader, val_loader = build_dataloaders(cfg, tracks, held)
    module = ZFlowModule(cfg, stats, codebooks)
    params = sum(p.numel() for p in module.net.parameters())
    print(
        f"tracks {len(tracks)} (held out {held})  crop {cfg.data.crop_frames}f "
        f"= {cfg.data.crop_frames / 172.265625:.1f} s, {cfg.data.crop_frames // cfg.model.patch} tokens  "
        f"latent dims {cfg.data.pca_dims}  params {params / 1e6:.1f}M  val crops {len(val_loader.dataset)}"  # type: ignore[arg-type]
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
