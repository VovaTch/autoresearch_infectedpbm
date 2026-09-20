"""
Autoregressive model over STYLE descriptors: the coarse level of the coarse-to-
fine generator.

The 512-context token-AR (config_ar_512_aligned.yaml) is conditioned on the
descriptor of its own grid slice -- the unit-norm mean of z_q over the 512-frame
slice plus 256 frames each side. It only has local coherence; the arc of a
track is meant to come from a model that emits one descriptor per slice. This
file trains that model on the descriptor sequences of the token cache and
exposes `sample_sequence` for the slice synthesizer's "ar" style walk.

Measured on tokens_88cacecb1e81 (53 tracks, 7548 slices of 2.97 s): the
1024-d descriptor is ~32-d effective (PCA-32 keeps 99.7% of the variance);
adjacent slices have cosine 0.91, lag-4 0.61, random pairs 0.39. So the model
works in PCA-whitened coordinates and its head is a mixture of diagonal
Gaussians -- a regression head would collapse onto the corpus mean, the one
thing a random pair already shares.

Conditioning is a track-id prefix token dropped to a learned null with
probability p_drop_id (classifier-free guidance at sampling), plus whatever
real descriptors are fed as a prefix. Validation is on HELD-OUT TRACKS with the
null id (their id rows are never trained): with 7.5k positions the model
memorises tracks long before it generalises, and in-track validation cannot see
that (finding_flow_phase_fail).

Usage:
    uv run python train_style_ar.py --config config_style_ar.yaml --build-phases
    CUDA_VISIBLE_DEVICES=1 uv run python train_style_ar.py --config config_style_ar.yaml
    uv run python train_style_ar.py --config config_style_ar.yaml --sample
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict, dataclass, field
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
from train_ar import Block, RMSNorm, TrackTokens, rope_cache  # noqa: E402
from train_ar import DataCfg as ArDataCfg  # noqa: E402
from train_ar import compute_style_windows, load_token_cache  # noqa: E402
from train_flow import plan_held_out  # noqa: E402
from train_zflow import (
    LatentStats,
    _build_section,
    apply_ema,
    param_groups,
)  # noqa: E402

# ===========================================================================
# Config
# ===========================================================================


@dataclass
class DataCfg:
    """Token cache, held-out split, whitening rank and the crop geometry."""

    token_cache: str = "~/.cache/infected_pbm/tokens_88cacecb1e81"
    held_out_tracks: int = 5
    split_seed: int = 1234
    pca_dims: int = 32
    pca_eig_floor: float = 1.0e-4
    # Descriptors are recomputed on `phases` shifted grids (offset k*slice/phases)
    # so the corpus is not just the cache's 7.5k rows.
    phases: int = 4
    crop_positions: int = 128
    min_crop: int = 8
    steps_per_epoch: int = 200
    single_track: str | None = None


@dataclass
class ModelCfg:
    """Transformer width and the mixture head."""

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    mog_components: int = 8
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    p_drop_id: float = 0.3
    # Train-time Gaussian noise on the whitened inputs (targets untouched): the
    # corpus is 30k rows, so the model memorises exact continuations otherwise.
    input_noise: float = 0.0
    rope_theta: float = 10000.0


@dataclass
class TrainCfg:
    """Optimiser, schedule and checkpointing, matching the repo's conventions."""

    devices: int | list[int] = 1
    minutes: float = 30.0
    lr: float = 3.0e-4
    lr_pct_start: float = 0.05
    lr_div_factor: float = 25.0
    batch_size: int = 64
    accumulate_grad_batches: int = 1
    precision: str = "32-true"
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    num_workers: int = 2
    ema_decay: float | None = 0.999
    seed: int = 42
    save_path: str = "saved_style_ar/"
    checkpoint: str | None = None


@dataclass
class StyleArConfig:
    """Top-level config, one block per YAML section."""

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


SECTIONS: dict[str, type] = {"data": DataCfg, "model": ModelCfg, "train": TrainCfg}


def load_config(path: str | Path) -> StyleArConfig:
    """
    Read a style-AR config from YAML.

    Args:
      path (str | Path): path to the .yaml file.

    Returns:
      StyleArConfig: fully populated config with defaults filled in.
    """
    with open(path, "r") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")
    return StyleArConfig(
        **{
            name: _build_section(cls, raw.get(name), name)
            for name, cls in SECTIONS.items()
        }
    )


def config_from_dict(raw: dict[str, Any]) -> StyleArConfig:
    """
    Rebuild a config from the dict stashed in a checkpoint's hyperparameters.

    Args:
      raw (dict[str, Any]): the asdict() form written at training time.

    Returns:
      StyleArConfig: reconstructed config.
    """
    return StyleArConfig(**{name: cls(**raw[name]) for name, cls in SECTIONS.items()})


# ===========================================================================
# Corpus: descriptors on shifted grids
# ===========================================================================


def cache_dir(cfg: StyleArConfig) -> Path:
    """
    Args:
      cfg (StyleArConfig): full config.

    Returns:
      Path: the token cache directory.
    """
    return Path(cfg.data.token_cache).expanduser()


def load_corpus(
    cfg: StyleArConfig,
) -> tuple[list[TrackTokens], list[int], torch.Tensor, dict[str, Any]]:
    """
    Load the token cache, the held-out split, the codebooks and the manifest.

    Args:
      cfg (StyleArConfig): full config.

    Returns:
      tuple[list[TrackTokens], list[int], torch.Tensor, dict[str, Any]]: all
        tracks, held-out track indices, (R, N, C) codebooks, manifest.
    """
    root = cache_dir(cfg)
    tracks, manifest = load_token_cache(
        root, ArDataCfg(single_track=cfg.data.single_track)
    )
    held = plan_held_out(manifest["tracks"], cfg.data)  # type: ignore[arg-type]
    if cfg.data.single_track:
        held = []
    codebooks = torch.load(root / "codebooks.pt", map_location="cpu", weights_only=True)
    return tracks, held, codebooks.float(), manifest


def phase_windows(
    tokens: torch.Tensor,
    codebooks: torch.Tensor,
    window_frames: int,
    context_frames: int,
    phases: int,
) -> list[torch.Tensor]:
    """
    Grid descriptors of one track on `phases` shifted grids.

    Phase k drops the first k * window_frames // phases frames, so its slices sit
    between phase 0's; phase 0 reproduces the cache's own style_windows.

    Args:
      tokens (torch.Tensor): (T, R) int64 indices.
      codebooks (torch.Tensor): (R, N, C) per-level codebooks.
      window_frames (int): slice length.
      context_frames (int): frames averaged on each side of the slice.
      phases (int): number of grid offsets.

    Returns:
      list[torch.Tensor]: per phase, (W_k, C) float32 unit descriptors.
    """
    out: list[torch.Tensor] = []
    for k in range(phases):
        offset = k * window_frames // phases
        vectors, _ = compute_style_windows(
            tokens[offset:], codebooks, window_frames, context_frames, grid=True
        )
        out.append(vectors)
    return out


def phases_path(cfg: StyleArConfig, manifest: dict[str, Any]) -> Path:
    """
    Args:
      cfg (StyleArConfig): full config.
      manifest (dict[str, Any]): the token cache manifest (style geometry).

    Returns:
      Path: the phase cache file for this geometry, inside the token cache.
    """
    tag = (
        f"w{manifest['style_window_frames']}c{manifest['style_context_frames']}"
        f"p{cfg.data.phases}"
    )
    return cache_dir(cfg) / f"style_ar_phases_{tag}.pt"


def build_phase_cache(
    cfg: StyleArConfig, force: bool = False
) -> dict[int, list[torch.Tensor]]:
    """
    Compute (or load) the shifted-grid descriptors for every track.

    Args:
      cfg (StyleArConfig): full config.
      force (bool): recompute even if the file exists.

    Returns:
      dict[int, list[torch.Tensor]]: track_idx -> per-phase (W_k, C) float32.
    """
    tracks, _, codebooks, manifest = load_corpus(cfg)
    path = phases_path(cfg, manifest)
    if path.exists() and not force:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        return {int(k): [p.float() for p in v] for k, v in blob.items()}
    if not manifest.get("style_grid"):
        raise ValueError(f"{cache_dir(cfg)} is not a grid-style cache")
    window = int(manifest["style_window_frames"])
    context = int(manifest["style_context_frames"])
    out: dict[int, list[torch.Tensor]] = {}
    for track in tracks:
        phases = phase_windows(
            track.tokens, codebooks, window, context, cfg.data.phases
        )
        if not torch.allclose(phases[0], track.style, atol=1e-3):
            raise RuntimeError(f"phase 0 differs from the cache for {track.track_name}")
        out[track.track_idx] = phases
    torch.save({k: [p.half() for p in v] for k, v in out.items()}, path)
    total = sum(p.shape[0] for v in out.values() for p in v)
    print(f"phase cache -> {path}: {total} descriptors from {len(out)} tracks")
    return {k: [p.half().float() for p in v] for k, v in out.items()}


# ===========================================================================
# Whitening
# ===========================================================================


def fit_style_stats(rows: torch.Tensor, dims: int, eig_floor: float) -> LatentStats:
    """
    PCA of the descriptors, truncated to `dims` directions.

    Args:
      rows (torch.Tensor): (N, C) descriptors of the TRAINING tracks.
      dims (int): rank kept.
      eig_floor (float): relative eigenvalue floor before whitening.

    Returns:
      LatentStats: mean (C,), basis (C, dims), eigvals (dims,).
    """
    x = rows.double()
    mean = x.mean(0)
    cov = (x.T @ x) / x.shape[0] - torch.outer(mean, mean)
    eig, vecs = torch.linalg.eigh(cov)
    eig, vecs = eig.flip(0).clamp_min(0.0), vecs.flip(1)
    return LatentStats(
        mean.float(), vecs[:, :dims].float(), eig[:dims].float(), dims, eig_floor
    )


def whiten_rows(stats: LatentStats, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      stats (LatentStats): whitening.
      x (torch.Tensor): (..., C) descriptors.

    Returns:
      torch.Tensor: (..., dims) whitened coordinates.
    """
    return ((x - stats.mean) @ stats.basis[:, : stats.dims]) * stats.scale


def unwhiten_rows(stats: LatentStats, y: torch.Tensor) -> torch.Tensor:
    """
    Args:
      stats (LatentStats): whitening.
      y (torch.Tensor): (..., dims) whitened coordinates.

    Returns:
      torch.Tensor: (..., C) reconstructed descriptors, not normalised.
    """
    return (y / stats.scale) @ stats.basis[:, : stats.dims].T + stats.mean


def decode_rows(stats: LatentStats, y: torch.Tensor) -> torch.Tensor:
    """
    Args:
      stats (LatentStats): whitening.
      y (torch.Tensor): (..., dims) whitened coordinates.

    Returns:
      torch.Tensor: (..., C) unit-norm descriptors, what the token-AR expects.
    """
    return F.normalize(unwhiten_rows(stats, y), dim=-1)


# ===========================================================================
# Datasets
# ===========================================================================


@dataclass
class StyleSeq:
    """One track's descriptor sequences, one per grid phase."""

    track_idx: int
    whitened: list[torch.Tensor]
    raw: list[torch.Tensor]


def build_sequences(
    phase_cache: dict[int, list[torch.Tensor]],
    stats: LatentStats,
    track_ids: Sequence[int],
) -> list[StyleSeq]:
    """
    Args:
      phase_cache (dict[int, list[torch.Tensor]]): per-track phase descriptors.
      stats (LatentStats): whitening.
      track_ids (Sequence[int]): tracks to include.

    Returns:
      list[StyleSeq]: sequences in track order.
    """
    return [
        StyleSeq(
            track_idx=idx,
            whitened=[whiten_rows(stats, p) for p in phase_cache[idx]],
            raw=list(phase_cache[idx]),
        )
        for idx in sorted(track_ids)
    ]


class StyleCropDataset(Dataset):
    """
    Random crops of the training sequences, weighted by track length.

    Args:
      seqs (list[StyleSeq]): training sequences.
      cfg (DataCfg): crop settings.
      length (int): items per epoch.
      seed (int): base seed; each index derives its own stream.
    """

    def __init__(
        self, seqs: list[StyleSeq], cfg: DataCfg, length: int, seed: int
    ) -> None:
        self.seqs = [s for s in seqs if s.whitened[0].shape[0] >= cfg.min_crop]
        if not self.seqs:
            raise ValueError("no sequence is at least min_crop long")
        self.cfg = cfg
        self.length = length
        self.seed = seed
        self.weights = [float(s.whitened[0].shape[0]) for s in self.seqs]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(f"{self.seed}:{index}:{torch.initial_seed()}")
        seq = rng.choices(self.seqs, weights=self.weights, k=1)[0]
        phase = rng.randrange(len(seq.whitened))
        total = seq.whitened[phase].shape[0]
        length = min(self.cfg.crop_positions, total)
        start = rng.randint(0, total - length)
        return {
            "x": seq.whitened[phase][start : start + length],
            "raw": seq.raw[phase][start : start + length],
            "track_idx": seq.track_idx,
        }


class StyleChunkDataset(Dataset):
    """
    Held-out sequences, phase 0, in non-overlapping chunks of crop_positions.

    Args:
      seqs (list[StyleSeq]): held-out sequences.
      cfg (DataCfg): crop settings.
    """

    def __init__(self, seqs: list[StyleSeq], cfg: DataCfg) -> None:
        self.items: list[tuple[StyleSeq, int, int]] = []
        for seq in seqs:
            total = seq.whitened[0].shape[0]
            for start in range(0, total, cfg.crop_positions):
                end = min(total, start + cfg.crop_positions)
                if end - start >= 2:
                    self.items.append((seq, start, end))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        seq, start, end = self.items[index]
        return {
            "x": seq.whitened[0][start:end],
            "raw": seq.raw[0][start:end],
            "track_idx": seq.track_idx,
        }


def collate_pad(items: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Right-pad variable-length crops.

    Args:
      items (Sequence[dict[str, Any]]): dataset items.

    Returns:
      dict[str, torch.Tensor]: x (B, L, dims), raw (B, L, C), mask (B, L) bool,
        track_idx (B,) int64.
    """
    longest = max(int(item["x"].shape[0]) for item in items)
    batch = len(items)
    x = torch.zeros(batch, longest, items[0]["x"].shape[-1])
    raw = torch.zeros(batch, longest, items[0]["raw"].shape[-1])
    mask = torch.zeros(batch, longest, dtype=torch.bool)
    for row, item in enumerate(items):
        n = int(item["x"].shape[0])
        x[row, :n], raw[row, :n], mask[row, :n] = item["x"], item["raw"], True
    return {
        "x": x,
        "raw": raw,
        "mask": mask,
        "track_idx": torch.tensor([int(item["track_idx"]) for item in items]),
    }


def build_dataloaders(
    cfg: StyleArConfig,
    phase_cache: dict[int, list[torch.Tensor]],
    stats: LatentStats,
    held: Sequence[int],
) -> tuple[DataLoader, DataLoader]:
    """
    Args:
      cfg (StyleArConfig): full config.
      phase_cache (dict[int, list[torch.Tensor]]): per-track phase descriptors.
      stats (LatentStats): whitening.
      held (Sequence[int]): held-out track indices.

    Returns:
      tuple[DataLoader, DataLoader]: train and validation loaders.
    """
    train_ids = [idx for idx in phase_cache if idx not in held]
    val_ids = [idx for idx in phase_cache if idx in held]
    if not val_ids:
        val_ids = train_ids[:1]
        print("WARNING: no held-out tracks; validation reuses a training track")
    train_ds = StyleCropDataset(
        build_sequences(phase_cache, stats, train_ids),
        cfg.data,
        cfg.data.steps_per_epoch * cfg.train.batch_size,
        cfg.train.seed,
    )
    val_ds = StyleChunkDataset(build_sequences(phase_cache, stats, val_ids), cfg.data)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_pad,
        drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pad,
    )
    return train_loader, val_loader


# ===========================================================================
# Model
# ===========================================================================


@dataclass
class MogParams:
    """
    Mixture-of-diagonal-Gaussians parameters, one mixture per position.

    Args:
      logits (torch.Tensor): (..., K) unnormalised component weights.
      means (torch.Tensor): (..., K, D) component means.
      log_std (torch.Tensor): (..., K, D) component log standard deviations.
    """

    logits: torch.Tensor
    means: torch.Tensor
    log_std: torch.Tensor

    def select(self, index: int) -> MogParams:
        """
        Args:
          index (int): position along the sequence axis (second to last of logits).

        Returns:
          MogParams: the mixture at that position, sequence axis removed.
        """
        return MogParams(
            self.logits[..., index, :],
            self.means[..., index, :, :],
            self.log_std[..., index, :, :],
        )


class StyleAr(nn.Module):
    """
    Causal transformer over whitened style descriptors with a mixture head.

    Sequence = [track id | null, BOS, x_0, x_1, ...]; the head is read from BOS
    onwards, so output i is the mixture predicting x_i given x_<i.

    Args:
      cfg (ModelCfg): width and head settings.
      dims (int): whitened descriptor width.
      num_tracks (int): id rows; row num_tracks is the learned null.
      max_positions (int): rotary cache size (crop + 2 prefix slots).
    """

    def __init__(
        self, cfg: ModelCfg, dims: int, num_tracks: int, max_positions: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dims = dims
        self.num_tracks = num_tracks
        self.max_positions = max_positions
        d = cfg.d_model
        self.in_proj = nn.Linear(dims, d)
        self.bos = nn.Parameter(torch.zeros(d))
        self.track_emb = nn.Embedding(num_tracks + 1, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm_out = RMSNorm(d)
        k = cfg.mog_components
        self.head = nn.Linear(d, k + 2 * k * dims)
        self.apply(self._init_weights)
        nn.init.normal_(self.bos, std=0.02)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    @property
    def null_id(self) -> int:
        """
        Returns:
          int: the embedding row used when the id is dropped.
        """
        return self.num_tracks

    @property
    def context(self) -> int:
        """
        Returns:
          int: descriptors the model can attend to (prefix slots excluded).
        """
        return self.max_positions - 2

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """
        Args:
          module (nn.Module): module being visited by apply().
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _rope(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          length (int): sequence length needed.
          device (torch.device): target device.
          dtype (torch.dtype): target dtype.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: cos and sin for [0, length).
        """
        if (
            self._cos is None
            or self._cos.shape[0] < length
            or self._cos.device != device
            or self._cos.dtype != dtype
        ):
            self._cos, self._sin = rope_cache(
                max(length, self.max_positions),
                self.cfg.d_model // self.cfg.n_heads,
                self.cfg.rope_theta,
                device,
                dtype,
            )
        assert self._sin is not None
        return self._cos[:length], self._sin[:length]

    def forward(
        self, x: torch.Tensor, track_idx: torch.Tensor, drop_id: torch.Tensor
    ) -> MogParams:
        """
        Args:
          x (torch.Tensor): (B, L, dims) whitened descriptors; L may be 0.
          track_idx (torch.Tensor): (B,) int64 track ids.
          drop_id (torch.Tensor): (B,) bool; True swaps the id for the null row.

        Returns:
          MogParams: mixtures at L + 1 positions; index i predicts x_i.
        """
        batch, length, _ = x.shape
        ids = torch.where(drop_id, torch.full_like(track_idx, self.null_id), track_idx)
        prefix = torch.stack([self.track_emb(ids), self.bos.expand(batch, -1)], dim=1)
        h = torch.cat([prefix, self.in_proj(x)], dim=1)
        h = self.drop(h)
        cos, sin = self._rope(length + 2, h.device, h.dtype)
        for layer, block in enumerate(self.blocks):
            h = block(h, cos, sin, None, layer)
        out = self.head(self.norm_out(h[:, 1:]))
        k = self.cfg.mog_components
        logits = out[..., :k]
        rest = out[..., k:].reshape(batch, length + 1, 2, k, self.dims)
        log_std = rest[:, :, 1].clamp(self.cfg.log_std_min, self.cfg.log_std_max)
        return MogParams(logits, rest[:, :, 0], log_std)


def mog_nll(params: MogParams, target: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of `target` under each position's mixture.

    Args:
      params (MogParams): (..., K) / (..., K, D) mixture parameters.
      target (torch.Tensor): (..., D) observed vectors.

    Returns:
      torch.Tensor: (...) NLL per position.
    """
    diff = (target.unsqueeze(-2) - params.means) * torch.exp(-params.log_std)
    log_comp = (
        -0.5 * diff.pow(2).sum(-1)
        - params.log_std.sum(-1)
        - 0.5 * target.shape[-1] * math.log(2 * math.pi)
    )
    return -torch.logsumexp(F.log_softmax(params.logits, dim=-1) + log_comp, dim=-1)


def mog_sample(
    params: MogParams, generator: torch.Generator | None, temperature: float
) -> torch.Tensor:
    """
    Draw one vector per mixture.

    Args:
      params (MogParams): (B, K) / (B, K, D) mixture parameters.
      generator (torch.Generator | None): RNG; None uses the global one.
      temperature (float): scales the component logits and the noise; 0 is the
        mean of the most likely component.

    Returns:
      torch.Tensor: (B, D) samples.
    """
    if temperature <= 0.0:
        component = params.logits.argmax(-1)
        return params.means[torch.arange(params.means.shape[0]), component]
    probs = F.softmax(params.logits / temperature, dim=-1)
    component = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    rows = torch.arange(params.means.shape[0])
    mean, log_std = params.means[rows, component], params.log_std[rows, component]
    noise = torch.randn(mean.shape, generator=generator, device=mean.device)
    return mean + temperature * torch.exp(log_std) * noise


def guide(cond: MogParams, null: MogParams, scale: float) -> MogParams:
    """
    Classifier-free guidance on mixture parameters -- a HEURISTIC.

    Component k of both passes comes from the same head unit, so pairing them
    is meaningful, but interpolating logits and means is not guidance on the
    density. scale 1 returns `cond` exactly; 0 returns `null` with cond's std.

    Args:
      cond (MogParams): parameters with the id present.
      null (MogParams): parameters with the null id.
      scale (float): guidance strength.

    Returns:
      MogParams: guided parameters.
    """
    if scale == 1.0:
        return cond
    return MogParams(
        null.logits + scale * (cond.logits - null.logits),
        null.means + scale * (cond.means - null.means),
        cond.log_std,
    )


@torch.no_grad()
def sample_sequence(
    model: StyleAr,
    stats: LatentStats,
    prefix: torch.Tensor | None,
    track_idx: int | None,
    steps: int,
    temperature: float = 1.0,
    cfg: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Continue (or start) a descriptor sequence.

    Args:
      model (StyleAr): the model, in eval mode.
      stats (LatentStats): its whitening.
      prefix (torch.Tensor | None): (P, C) real unit descriptors to continue
        from; None starts cold at BOS.
      track_idx (int | None): id row; None means the null id.
      steps (int): descriptors to generate.
      temperature (float): sampling temperature.
      cfg (float): guidance strength; 1 is plain conditional, 0 null only.
      generator (torch.Generator | None): RNG for reproducible draws.

    Returns:
      torch.Tensor: (steps, C) unit-norm descriptors.
    """
    device = next(model.parameters()).device
    rows: list[torch.Tensor] = []
    if prefix is not None and prefix.shape[0]:
        rows = list(whiten_rows(stats, prefix.to(device).float()))
    guided = track_idx is not None and cfg != 1.0
    use_null = track_idx is None or cfg == 0.0
    ids = torch.tensor([model.null_id if use_null else int(track_idx)], device=device)
    out: list[torch.Tensor] = []
    for _ in range(steps):
        context = rows[-model.context :]
        x = (
            torch.stack(context)[None]
            if context
            else torch.zeros(1, 0, model.dims, device=device)
        )
        if guided and not use_null:
            params = model(
                x.expand(2, -1, -1),
                ids.expand(2),
                torch.tensor([False, True], device=device),
            ).select(-1)
            params = guide(
                MogParams(params.logits[:1], params.means[:1], params.log_std[:1]),
                MogParams(params.logits[1:], params.means[1:], params.log_std[1:]),
                cfg,
            )
        else:
            params = model(x, ids, torch.tensor([use_null], device=device)).select(-1)
        nxt = mog_sample(params, generator, temperature)[0]
        rows.append(nxt)
        out.append(nxt)
    if not out:
        return torch.zeros(0, stats.mean.shape[0])
    return decode_rows(stats, torch.stack(out)).cpu()


# ===========================================================================
# Lightning module
# ===========================================================================


def _pair_cos(rows: torch.Tensor, limit: int = 256) -> torch.Tensor:
    """
    Args:
      rows (torch.Tensor): (N, C) unit vectors.
      limit (int): rows scored, from the front.

    Returns:
      torch.Tensor: scalar mean off-diagonal cosine; 0 with fewer than 2 rows.
    """
    rows = F.normalize(rows[:limit], dim=-1)
    n = rows.shape[0]
    if n < 2:
        return torch.zeros((), device=rows.device)
    gram = rows @ rows.T
    return (gram.sum() - gram.diagonal().sum()) / (n * (n - 1))


class StyleArModule(L.LightningModule):
    """
    Trains StyleAr with the id dropped at random; validates with the null id.

    Args:
      cfg (StyleArConfig): full config.
      stats (LatentStats): whitening of the descriptors.
      num_tracks (int): id rows.
      held_out (list[int]): track ids never seen with their own id.
    """

    def __init__(
        self,
        cfg: StyleArConfig,
        stats: LatentStats,
        num_tracks: int,
        held_out: list[int],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = StyleAr(
            cfg.model, stats.dims, num_tracks, cfg.data.crop_positions + 2
        )
        self.register_buffer("pca_mean", stats.mean, persistent=True)
        self.register_buffer("pca_basis", stats.basis, persistent=True)
        self.register_buffer("pca_eigvals", stats.eigvals, persistent=True)
        self.save_hyperparameters(
            {"config": asdict(cfg), "num_tracks": num_tracks, "held_out": held_out}
        )

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

    @property
    def held_out(self) -> list[int]:
        """
        Returns:
          list[int]: track ids whose embedding rows were never trained.
        """
        return [int(i) for i in self.hparams["held_out"]]

    def _run(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """
        Args:
          batch (dict[str, torch.Tensor]): collate_pad output.
          stage (str): "train" or "val".

        Returns:
          torch.Tensor: masked mean NLL per position.
        """
        x, mask, track_idx = batch["x"], batch["mask"], batch["track_idx"]
        batch_size = x.shape[0]
        if stage == "train":
            drop_id = torch.rand(batch_size, device=x.device) < self.cfg.model.p_drop_id
        else:
            drop_id = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        x_in = x[:, :-1]
        if stage == "train" and self.cfg.model.input_noise > 0.0:
            x_in = x_in + self.cfg.model.input_noise * torch.randn_like(x_in)
        params = self.model(x_in, track_idx, drop_id)
        nll = mog_nll(params, x)
        weights = mask.float()
        loss = (nll * weights).sum() / weights.sum().clamp_min(1.0)
        self.log(f"{stage}/nll", loss, prog_bar=True, batch_size=batch_size)
        if stage == "train":
            for name, rows in (("id", ~drop_id), ("null", drop_id)):
                if rows.any():
                    sub = weights[rows]
                    self.log(
                        f"train/nll_{name}",
                        (nll[rows] * sub).sum() / sub.sum().clamp_min(1.0),
                        batch_size=int(rows.sum()),
                    )
        else:
            self._val_diagnostics(batch, params)
        return loss

    def _val_diagnostics(
        self, batch: dict[str, torch.Tensor], params: MogParams
    ) -> None:
        """
        Teacher-forced one-step samples against the real next descriptor.

        Args:
          batch (dict[str, torch.Tensor]): collate_pad output.
          params (MogParams): the mixtures scored in _run.
        """
        raw, mask = batch["raw"], batch["mask"]
        generator = torch.Generator(device=raw.device).manual_seed(0)
        flat = MogParams(
            params.logits.reshape(-1, params.logits.shape[-1]),
            params.means.reshape(-1, *params.means.shape[-2:]),
            params.log_std.reshape(-1, *params.log_std.shape[-2:]),
        )
        sampled = decode_rows(self.stats, mog_sample(flat, generator, 1.0))
        sampled = sampled.reshape(raw.shape)
        valid = mask.reshape(-1)
        next_cos = (sampled * raw).sum(-1).reshape(-1)[valid].mean()
        copy_mask = (mask[:, 1:] & mask[:, :-1]).reshape(-1)
        copy_cos = (raw[:, 1:] * raw[:, :-1]).sum(-1).reshape(-1)[copy_mask].mean()
        batch_size = raw.shape[0]
        self.log("val/next_cos", next_cos, batch_size=batch_size)
        self.log("val/copy_cos", copy_cos, batch_size=batch_size)
        self.log(
            "val/sample_pair_cos",
            _pair_cos(sampled.reshape(-1, raw.shape[-1])[valid]),
            batch_size=batch_size,
        )
        self.log(
            "val/data_pair_cos",
            _pair_cos(raw.reshape(-1, raw.shape[-1])[valid]),
            batch_size=batch_size,
        )

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._run(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._run(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        """
        AdamW with decay only on matmul weights, plus the repo's wall-clock cycle.

        Returns:
          dict[str, Any]: Lightning optimizer/scheduler bundle.
        """
        decay, no_decay = param_groups(self.model)
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


def load_style_ar(ckpt_path: Path, ema: bool = True) -> StyleArModule:
    """
    Rebuild a trained module from its checkpoint, on the CPU.

    Args:
      ckpt_path (Path): Lightning checkpoint.
      ema (bool): apply the EMA weights when present.

    Returns:
      StyleArModule: eval-mode module.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    cfg = config_from_dict(hp["config"])
    state = ckpt["state_dict"]
    stats = LatentStats(
        state["pca_mean"],
        state["pca_basis"],
        state["pca_eigvals"],
        cfg.data.pca_dims,
        cfg.data.pca_eig_floor,
    )
    module = StyleArModule(cfg, stats, int(hp["num_tracks"]), list(hp["held_out"]))
    module.load_state_dict(state, strict=True)
    if ema:
        apply_ema(module.model, ckpt)
    return module.eval()


def build_trainer(cfg: StyleArConfig) -> L.Trainer:
    """
    Args:
      cfg (StyleArConfig): full config.

    Returns:
      L.Trainer: configured trainer.
    """
    save_path = REPO / cfg.train.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.Timer(duration={"minutes": cfg.train.minutes}),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="style_ar_best",
            monitor="val/nll",
            mode="min",
            save_top_k=1,
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="style_ar_latest",
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
        logger=L.pytorch.loggers.TensorBoardLogger(str(save_path), name="style_ar"),
        log_every_n_steps=10,
    )


def _seq_stats(
    rows: torch.Tensor, mean_dir: torch.Tensor
) -> tuple[float, float, float]:
    """
    Args:
      rows (torch.Tensor): (N, C) unit descriptors.
      mean_dir (torch.Tensor): (C,) unit corpus-mean direction.

    Returns:
      tuple[float, float, float]: adjacent cosine, cosine to the corpus mean,
        mean pairwise cosine.
    """
    adjacent = float((rows[1:] * rows[:-1]).sum(-1).mean()) if len(rows) > 1 else 0.0
    return adjacent, float((rows @ mean_dir).mean()), float(_pair_cos(rows))


def sample_report(
    module: StyleArModule,
    phase_cache: dict[int, list[torch.Tensor]],
    names: dict[int, str],
    steps: int,
    prefix: int,
    temperature: float,
    cfg: float,
) -> None:
    """
    Compare sampled continuations with the real ones, track by track.

    Args:
      module (StyleArModule): trained module.
      phase_cache (dict[int, list[torch.Tensor]]): per-track descriptors.
      names (dict[int, str]): track names.
      steps (int): descriptors sampled per track.
      prefix (int): real descriptors fed before sampling.
      temperature (float): sampling temperature.
      cfg (float): guidance for tracks with a trained id.
    """
    stats = module.stats
    held = set(module.held_out)
    train_ids = sorted(
        (i for i in phase_cache if i not in held),
        key=lambda i: -phase_cache[i][0].shape[0],
    )[:3]
    mean_dir = F.normalize(stats.mean, dim=-1)
    print(
        f"{'track':<34} {'id':>4} {'adj r/s':>13} {'mean r/s':>13} "
        f"{'pair r/s':>13} {'nll':>7}"
    )
    for idx in sorted(held) + train_ids:
        rows = phase_cache[idx][0]
        if rows.shape[0] < prefix + steps:
            continue
        real = rows[prefix : prefix + steps]
        track_id = None if idx in held else idx
        sampled = sample_sequence(
            module.model,
            stats,
            rows[:prefix] if prefix else None,
            track_id,
            steps,
            temperature,
            cfg,
            torch.Generator().manual_seed(0),
        )
        with torch.no_grad():
            x = whiten_rows(stats, rows[: prefix + steps])[None]
            ids = torch.tensor([module.model.null_id if track_id is None else idx])
            params = module.model(x[:, :-1], ids, torch.tensor([track_id is None]))
            nll = float(mog_nll(params, x)[0, prefix:].mean())
        r_adj, r_mean, r_pair = _seq_stats(real, mean_dir)
        s_adj, s_mean, s_pair = _seq_stats(sampled, mean_dir)
        label = ("null" if track_id is None else str(idx)).rjust(4)
        print(
            f"{names[idx][:34]:<34} {label} {r_adj:6.3f}/{s_adj:6.3f} "
            f"{r_mean:6.3f}/{s_mean:6.3f} {r_pair:6.3f}/{s_pair:6.3f} {nll:7.2f}"
        )


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "config_style_ar.yaml"))
    parser.add_argument(
        "--build-phases", action="store_true", help="build the phase cache and exit"
    )
    parser.add_argument("--force", action="store_true", help="rebuild the phase cache")
    parser.add_argument("--sample", action="store_true", help="print a sampling report")
    parser.add_argument("--ckpt", default=None, help="checkpoint for --sample")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--prefix", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    phase_cache = build_phase_cache(cfg, force=args.force)
    tracks, held, _, _ = load_corpus(cfg)
    names = {t.track_idx: t.track_name for t in tracks}

    if args.sample:
        ckpt = Path(args.ckpt or REPO / cfg.train.save_path / "style_ar_best.ckpt")
        module = load_style_ar(ckpt)
        print(f"{ckpt} held out {module.held_out}")
        sample_report(
            module,
            phase_cache,
            names,
            args.steps,
            args.prefix,
            args.temperature,
            args.cfg,
        )
        return

    train_rows = torch.cat(
        [p for idx, phases in phase_cache.items() if idx not in held for p in phases]
    )
    stats = fit_style_stats(train_rows, cfg.data.pca_dims, cfg.data.pca_eig_floor)
    full = fit_style_stats(train_rows, train_rows.shape[-1], cfg.data.pca_eig_floor)
    cum = full.eigvals.cumsum(0) / full.eigvals.sum()
    kept = float(cum[cfg.data.pca_dims - 1])
    print(
        f"style stats: {train_rows.shape[0]} rows from {len(phase_cache) - len(held)} "
        f"tracks (held out {held}); pca_dims {cfg.data.pca_dims} keeps {kept:.4f} of "
        f"the variance; 99% in {int((cum < 0.99).sum()) + 1} dims"
    )
    if args.build_phases:
        return

    L.seed_everything(cfg.train.seed, workers=True)
    train_loader, val_loader = build_dataloaders(cfg, phase_cache, stats, held)
    num_tracks = max(phase_cache) + 1
    module = StyleArModule(cfg, stats, num_tracks, list(held))
    print(
        f"StyleAr: {sum(p.numel() for p in module.model.parameters()) / 1e6:.2f}M params"
    )
    if cfg.train.checkpoint:
        ckpt = torch.load(
            REPO / cfg.train.checkpoint, map_location="cpu", weights_only=False
        )
        print(module.load_state_dict(ckpt["state_dict"], strict=False))
    trainer = build_trainer(cfg)
    trainer.fit(module, train_loader, val_loader)
    final = REPO / cfg.train.save_path / "style_ar_final.ckpt"
    trainer.save_checkpoint(str(final))
    print(f"final checkpoint -> {final}")


if __name__ == "__main__":
    main()
