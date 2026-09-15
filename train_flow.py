"""
Flow-matching enhancer that cleans up the frozen tokenizer's decoder output.

`probe_copy_synthesis.py` measured that the VQ round trip alone -- before any
generation -- owns 55% of the log-spectral error and 81% of the crest error, and
`probe_consistency.py` pinned the mechanism: the decoder head emits 2*(n_fft//2+1)
free numbers per hop, 4x more than the waveform it produces has samples, so its
spectrogram is not in the range of the STFT operator (measured consistency 0.5136
against a real-audio floor of 0.0005). Overlap-add destroys whatever is
unrealizable, which spreads energy in time and raises crest 3.5 dB.

This model post-processes that waveform. It is a bridge between two *data*
distributions, not noise -> data: x0 is the round trip, x1 is the true audio, and
the same crop of the same track supplies both. Training pairs come from the
existing whole-track token cache decoded through onnx/decoder.onnx, so the
tokenizer is frozen and never re-trained.

Usage:
    uv run python train_flow.py --build-cache --config config_flow.yaml
    uv run python train_flow.py --config config_flow.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generate_ar import decode_tokens, make_decoder  # noqa: E402
from prepare import (  # noqa: E402
    multi_res_stft_distance,
    phase_distance,
    stft_consistency,
)
from train import EMA, TimeOneCycleLR  # noqa: E402
from train_ar import (  # noqa: E402
    _overlaps,
    enumerate_tracks,
    load_track_audio,
    plan_val_windows,
)


# ===========================================================================
# Config
# ===========================================================================


@dataclass
class TokenizerCfg:
    """Frozen tokenizer artefacts and the caches the pair build reads."""

    decoder_onnx: str = "onnx/decoder.onnx"
    meta: str = "onnx/tokenizer_meta.json"
    token_cache: str = "~/.cache/infected_pbm/tokens_ea052b5c4acf"
    tracks_dir: str = "~/.cache/infected_pbm/tracks"
    slices_dir: str = "~/.cache/infected_pbm/slices"
    cache_root: str = "~/.cache/infected_pbm"
    chunk_frames: int = 4096
    margin: int = 256


@dataclass
class DataCfg:
    """Cropping, the spectral compression, and the train/val/held-out split.

    val_frac / val_windows_per_track / min_val_window / split_seed carry the
    names train_ar.plan_val_windows reads, so the same held-out time windows are
    planned here without a second implementation.
    """

    crop_frames: int = 256
    alpha: float = 0.3
    beta: float | None = None
    held_out_tracks: int = 5
    val_frac: float = 0.04
    val_windows_per_track: int = 2
    min_val_window: int = 512
    split_seed: int = 1234
    steps_per_epoch: int = 500
    val_crops_per_window: int = 2
    single_track: str | None = None


@dataclass
class ModelCfg:
    """UNet2D geometry."""

    base_ch: int = 64
    ch_mults: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    blocks_per_level: int = 2
    attn_at_bottleneck: bool = True
    attn_heads: int = 8
    t_embed_dim: int = 256
    dropout: float = 0.0
    groups: int = 32
    condition_on_x0: bool = False


@dataclass
class FlowCfg:
    """The interpolant and the sampler."""

    sigma: float = 0.0
    t_eps: float = 1.0e-3
    steps: int = 8
    project_every_step: bool = True
    rollout_prob: float = 0.5


@dataclass
class LossCfg:
    """
    Term weights. Every metric comes from prepare.py.

    w_phase exists because the other terms cannot see the defect. Measured on a
    round trip: 88% of the complex STFT error survives even if the magnitude
    spectrum is made perfect, so mrstft (magnitude only) is structurally blind to
    it, stft_consistency is already ~0 for any real waveform, and complex L1
    answers phase uncertainty by shrinking magnitude rather than guessing. Left
    at those three terms the model collapses to returning its input, which is
    what it did: phase_distance 0.4507 for the round trip, 0.4518 after eight
    steps of enhancement.
    """

    w_spec: float = 1.0
    w_wav: float = 0.5
    w_cons: float = 0.1
    w_phase: float = 1.0


@dataclass
class TrainCfg:
    """Optimiser, schedule and checkpointing, matching the repo's conventions."""

    devices: int | list[int] = 1
    minutes: float = 1200.0
    lr: float = 2.0e-4
    lr_pct_start: float = 0.05
    lr_div_factor: float = 25.0
    batch_size: int = 8
    accumulate_grad_batches: int = 1
    precision: str = "32-true"
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 4
    ema_decay: float | None = 0.999
    seed: int = 42
    save_path: str = "saved_flow/"
    checkpoint: str | None = None


@dataclass
class FlowConfig:
    """Top-level config, one block per YAML section."""

    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    flow: FlowCfg = field(default_factory=FlowCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


SECTIONS: dict[str, type] = {
    "tokenizer": TokenizerCfg,
    "data": DataCfg,
    "model": ModelCfg,
    "flow": FlowCfg,
    "loss": LossCfg,
    "train": TrainCfg,
}


def _build_section(cls: type, raw: dict[str, Any] | None, name: str):
    """
    Instantiate one config dataclass, rejecting unknown keys.

    A silently dropped key is how a config typo becomes a run that looks fine and
    answers the wrong question, so unknown keys raise (same contract as
    train_ar._build_section).

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


def load_config(path: str | Path) -> FlowConfig:
    """
    Read a flow-enhancer config from YAML.

    Args:
      path (str | Path): path to the .yaml file.

    Returns:
      FlowConfig: fully populated config with defaults filled in.
    """
    with open(path, "r") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")
    return FlowConfig(
        **{
            name: _build_section(cls, raw.get(name), name)
            for name, cls in SECTIONS.items()
        }
    )


def config_from_dict(raw: dict[str, Any]) -> FlowConfig:
    """
    Rebuild a config from the dict stashed in a checkpoint's hyperparameters.

    Args:
      raw (dict[str, Any]): the asdict() form written at training time.

    Returns:
      FlowConfig: reconstructed config.
    """
    return FlowConfig(
        **{name: cls(**raw[name]) for name, cls in SECTIONS.items()}
    )


# ===========================================================================
# Spectral representation
# ===========================================================================


@dataclass(frozen=True)
class SpecGeom:
    """
    STFT geometry plus the amplitude compression, shared by every stage.

    The Nyquist bin is dropped so a 4-level UNet halves the frequency axis
    cleanly (513 -> 512); it is restored as zeros before the inverse. Source mp3s
    are lowpassed far below 22.05 kHz, so nothing audible lives there.

    Args:
      n_fft (int): analysis size, matched to the tokenizer's 1024.
      hop (int): analysis hop, matched to the tokenizer's 256.
      alpha (float): amplitude compression exponent, |X|^alpha.
      beta (float): scale applied after compression, measured at cache build.
      sample_rate (int): 44100.
    """

    n_fft: int = 1024
    hop: int = 256
    alpha: float = 0.3
    beta: float = 1.0
    sample_rate: int = 44100

    @property
    def bins(self) -> int:
        """int: frequency bins the network sees, Nyquist excluded."""
        return self.n_fft // 2

    def crop_samples(self, frames: int) -> int:
        """
        Samples whose centred STFT is exactly `frames` frames long.

        Args:
          frames (int): desired frame count.

        Returns:
          int: sample count.
        """
        return (frames - 1) * self.hop


def _window(n_fft: int, device: torch.device) -> torch.Tensor:
    """
    Hann analysis/synthesis window.

    Args:
      n_fft (int): window length.
      device (torch.device): where to build it.

    Returns:
      torch.Tensor: (n_fft,) float32 window.
    """
    return torch.hann_window(n_fft, device=device, dtype=torch.float32)


def compress(spec: torch.Tensor, geom: SpecGeom) -> torch.Tensor:
    """
    Compress a complex spectrogram to the network's real/imag tensor.

    Xc = beta * |X|^alpha * exp(i * angle(X)). Exactly invertible by
    `uncompress`; the compression is what stops a 90 dB dynamic range from
    making the L1 loss report only the loudest bins.

    Args:
      spec (torch.Tensor): (B, F, T) complex spectrogram.
      geom (SpecGeom): geometry and compression constants.

    Returns:
      torch.Tensor: (B, 2, F, T) float32, real and imaginary parts.
    """
    mag = spec.abs().clamp_min(1e-12)
    scaled = spec / mag * (geom.beta * mag.pow(geom.alpha))
    return torch.stack([scaled.real, scaled.imag], dim=1)


def uncompress(x: torch.Tensor, geom: SpecGeom) -> torch.Tensor:
    """
    Invert `compress` and restore the dropped Nyquist bin as zeros.

    The exponent here is 1/alpha > 1, so the gradient vanishes rather than
    explodes as a bin approaches silence -- the reason compression is applied to
    data only and this direction to model output.

    Args:
      x (torch.Tensor): (B, 2, F, T) real/imag of the compressed spectrogram.
      geom (SpecGeom): geometry and compression constants.

    Returns:
      torch.Tensor: (B, F+1, T) complex spectrogram at the original scale.
    """
    c = torch.complex(x[:, 0].float(), x[:, 1].float())
    mag = c.abs().clamp_min(1e-12)
    full = c / mag * (mag / geom.beta).pow(1.0 / geom.alpha)
    return torch.cat([full, torch.zeros_like(full[:, :1])], dim=1)


def stft_c(wav: torch.Tensor, geom: SpecGeom) -> torch.Tensor:
    """
    Waveform to the network's compressed spectral tensor.

    Args:
      wav (torch.Tensor): (B, 1, L) or (B, L) float waveform.
      geom (SpecGeom): geometry and compression constants.

    Returns:
      torch.Tensor: (B, 2, geom.bins, L//hop + 1) float32.
    """
    if wav.dim() == 3:
        wav = wav.squeeze(1)
    spec = torch.stft(
        wav.float(),
        n_fft=geom.n_fft,
        hop_length=geom.hop,
        window=_window(geom.n_fft, wav.device),
        return_complex=True,
    )
    return compress(spec[:, : geom.bins], geom)


def istft_c(x: torch.Tensor, geom: SpecGeom, length: int) -> torch.Tensor:
    """
    Network tensor back to a waveform.

    Args:
      x (torch.Tensor): (B, 2, geom.bins, T) compressed spectral tensor.
      geom (SpecGeom): geometry and compression constants.
      length (int): output sample count, normally geom.crop_samples(T).

    Returns:
      torch.Tensor: (B, 1, length) float32 waveform.
    """
    wav = torch.istft(
        uncompress(x, geom),
        n_fft=geom.n_fft,
        hop_length=geom.hop,
        window=_window(geom.n_fft, x.device),
        length=length,
    )
    return wav.unsqueeze(1)


def project(x: torch.Tensor, geom: SpecGeom, length: int) -> torch.Tensor:
    """
    Project a spectral tensor onto the range of the STFT operator.

    One pass of STFT(ISTFT(.)). The decoder's own spectrogram misses this
    manifold by 0.51 relative L1, and everything unrealizable is destroyed by
    overlap-add, so every sampler iterate is put back on it.

    Args:
      x (torch.Tensor): (B, 2, F, T) spectral tensor.
      geom (SpecGeom): geometry and compression constants.
      length (int): sample count of the intermediate waveform.

    Returns:
      torch.Tensor: (B, 2, F, T) realizable spectral tensor.
    """
    return stft_c(istft_c(x, geom, length), geom)


# ===========================================================================
# Paired round-trip / clean cache
# ===========================================================================


def load_meta(cfg: FlowConfig) -> dict[str, Any]:
    """
    Read the frozen tokenizer's geometry.

    Args:
      cfg (FlowConfig): full config.

    Returns:
      dict[str, Any]: onnx/tokenizer_meta.json contents.
    """
    return json.loads((REPO / cfg.tokenizer.meta).read_text())


def cache_tag(cfg: FlowConfig) -> str:
    """
    Content address for the pair cache.

    Keyed on the decoder graph and the token cache it decodes, the two things
    that decide what the round-trip stream actually is.

    Args:
      cfg (FlowConfig): full config.

    Returns:
      str: 12 hex characters.
    """
    digest = hashlib.sha256()
    digest.update(Path(cfg.tokenizer.token_cache).expanduser().name.encode())
    with open(REPO / cfg.tokenizer.decoder_onnx, "rb") as handle:
        digest.update(handle.read(4 << 20))
    return digest.hexdigest()[:12]


def pair_cache_dir(cfg: FlowConfig) -> Path:
    """
    Args:
      cfg (FlowConfig): full config.

    Returns:
      Path: directory the pair cache lives in.
    """
    root = Path(cfg.tokenizer.cache_root).expanduser()
    return root / f"flowpairs_{cache_tag(cfg)}"


def _crest_db(wav: torch.Tensor) -> float:
    """
    Args:
      wav (torch.Tensor): (L,) waveform.

    Returns:
      float: peak-to-RMS ratio in dB, 0.0 for silence.
    """
    rms = float(wav.pow(2).mean().sqrt())
    peak = float(wav.abs().max())
    return 20.0 * math.log10(peak / rms) if rms > 0 and peak > 0 else 0.0


def measure_beta(
    tracks: Sequence[torch.Tensor], geom: SpecGeom, samples: int = 64
) -> float:
    """
    Choose the compression scale so the network's inputs have unit variance.

    For Xc = beta*|X|^alpha*exp(i*phi) with phase roughly uniform, the real and
    imaginary parts each have variance beta^2 * E[|X|^(2*alpha)] / 2, so
    beta = sqrt(2 / E[|X|^(2*alpha)]). Measured once and pinned in the manifest
    rather than re-guessed per run.

    Args:
      tracks (Sequence[torch.Tensor]): clean (L,) waveforms to sample from.
      geom (SpecGeom): geometry; only alpha is used.
      samples (int): number of random 1 s excerpts to average over.

    Returns:
      float: the scale.
    """
    rng = random.Random(0)
    span = geom.sample_rate
    total, count = 0.0, 0
    for _ in range(samples):
        wav = tracks[rng.randrange(len(tracks))]
        if wav.numel() <= span:
            continue
        start = rng.randrange(0, wav.numel() - span)
        spec = torch.stft(
            wav[start : start + span].float().unsqueeze(0),
            n_fft=geom.n_fft,
            hop_length=geom.hop,
            window=_window(geom.n_fft, wav.device),
            return_complex=True,
        )
        total += float(spec[:, : geom.bins].abs().pow(2.0 * geom.alpha).mean())
        count += 1
    if not count or total <= 0:
        return 1.0
    return float(math.sqrt(2.0 * count / total))


def build_pair_cache(cfg: FlowConfig, force: bool = False) -> Path:
    """
    Decode the whole corpus through the frozen decoder and cache (rt, clean).

    The token cache already holds whole-track streams, so the round trip is one
    ONNX decode per track with no 32768-sample seams, and load_track_audio trims
    the source to the same whole-frame length -- the two streams are therefore
    sample-aligned with no resampling and no gain change. Nothing is normalised:
    peak-normalising per clip is the bias that invalidated earlier A/Bs.

    Args:
      cfg (FlowConfig): full config.
      force (bool): rebuild even if the cache is already complete.

    Returns:
      Path: the cache directory.
    """
    out = pair_cache_dir(cfg)
    manifest_path = out / "_manifest.json"
    if manifest_path.exists() and not force:
        print(f"pair cache present: {out}")
        return out

    meta = load_meta(cfg)
    hop, rate = int(meta["hop_length"]), int(meta["sample_rate"])
    token_dir = Path(cfg.tokenizer.token_cache).expanduser()
    token_manifest = json.loads((token_dir / "_manifest.json").read_text())
    tracks_dir = Path(cfg.tokenizer.tracks_dir).expanduser()
    slices_dir = Path(cfg.tokenizer.slices_dir).expanduser()
    paths = enumerate_tracks(tracks_dir)
    out.mkdir(parents=True, exist_ok=True)

    session = make_decoder(REPO / cfg.tokenizer.decoder_onnx, use_gpu=True)
    geom = SpecGeom(int(meta["n_fft"]), hop, cfg.data.alpha, 1.0, rate)
    entries: list[dict[str, Any]] = []
    beta_pool: list[torch.Tensor] = []

    print(f"building pair cache -> {out}")
    for entry in token_manifest["tracks"]:
        blob = torch.load(
            token_dir / entry["file"], map_location="cpu", weights_only=False
        )
        idx = int(blob["track_idx"])
        tokens = blob["tokens"].long().unsqueeze(0)
        rt = torch.from_numpy(
            decode_tokens(
                session, tokens, hop, cfg.tokenizer.chunk_frames, cfg.tokenizer.margin
            )
        ).reshape(-1)
        clean = load_track_audio(paths[idx], rate, hop, slices_dir).reshape(-1)
        usable = min(rt.numel(), clean.numel())
        if rt.numel() != clean.numel():
            print(
                f"  [{idx:02d}] length mismatch rt={rt.numel()} clean={clean.numel()}"
                f"; trimming to {usable}"
            )
        rt, clean = rt[:usable].contiguous(), clean[:usable].contiguous()

        name = f"{idx:03d}_{Path(entry['file']).stem.split('_', 1)[1]}.pt"
        torch.save({"rt": rt, "clean": clean}, out / name)
        entries.append(
            {
                "file": name,
                "track_idx": idx,
                "track_name": entry["track_name"],
                "num_frames": usable // hop,
                "samples": usable,
            }
        )
        if len(beta_pool) < 8:
            beta_pool.append(clean)
        print(
            f"  [{idx:02d}] {usable / rate:7.1f}s  crest clean {_crest_db(clean):5.2f}"
            f" -> rt {_crest_db(rt):5.2f} dB"
        )

    beta = measure_beta(beta_pool, geom)
    held = plan_held_out(entries, cfg.data)
    manifest_path.write_text(
        json.dumps(
            {
                "tokenizer_checkpoint": token_manifest["checkpoint"],
                "token_cache": token_dir.name,
                "decoder_onnx": cfg.tokenizer.decoder_onnx,
                "sample_rate": rate,
                "n_fft": int(meta["n_fft"]),
                "hop_length": hop,
                "alpha": cfg.data.alpha,
                "beta": beta,
                "held_out": held,
                "tracks": entries,
            },
            indent=1,
        )
    )
    print(f"beta = {beta:.4f}   held-out tracks = {held}")
    return out


def plan_held_out(entries: Sequence[dict[str, Any]], cfg: DataCfg) -> list[int]:
    """
    Pick whole tracks to withhold from training.

    prepare.py's random_split cuts per 0.74 s slice, so every track lands in both
    halves; a restoration model would then be scored on tracks it trained on.
    The draw is seeded from split_seed alone, so it does not move when the model
    changes.

    Args:
      entries (Sequence[dict[str, Any]]): per-track manifest rows.
      cfg (DataCfg): split settings.

    Returns:
      list[int]: sorted track indices reserved for evaluation.
    """
    idxs = sorted(int(e["track_idx"]) for e in entries)
    rng = random.Random(cfg.split_seed)
    rng.shuffle(idxs)
    return sorted(idxs[: max(0, cfg.held_out_tracks)])


# ===========================================================================
# Datasets
# ===========================================================================


@dataclass
class PairTrack:
    """One track's aligned round-trip and clean streams plus its val windows."""

    rt: torch.Tensor
    clean: torch.Tensor
    track_idx: int
    track_name: str
    num_frames: int
    val_windows: list[tuple[int, int]]


def load_pair_cache(
    cache_dir: Path, cfg: DataCfg, held_out: bool = False
) -> tuple[list[PairTrack], dict[str, Any]]:
    """
    Load the pair cache, memory-mapped, and recompute the val split.

    The waveforms are ~8 GB of float32, so they are mapped rather than read; the
    split is recomputed from config here (as train_ar does) so changing split
    settings never means rebuilding the cache.

    Args:
      cache_dir (Path): directory written by build_pair_cache.
      cfg (DataCfg): split and filtering settings.
      held_out (bool): return the withheld evaluation tracks instead of the
        training tracks.

    Returns:
      tuple[list[PairTrack], dict[str, Any]]: the tracks and the manifest.
    """
    manifest: dict[str, Any] = json.loads((cache_dir / "_manifest.json").read_text())
    reserved = set(manifest["held_out"])
    tracks: list[PairTrack] = []
    for entry in manifest["tracks"]:
        idx = int(entry["track_idx"])
        if (idx in reserved) != held_out:
            continue
        if cfg.single_track and cfg.single_track not in entry["track_name"]:
            continue
        blob = torch.load(
            cache_dir / entry["file"], map_location="cpu", weights_only=True, mmap=True
        )
        frames = int(entry["num_frames"])
        tracks.append(
            PairTrack(
                rt=blob["rt"],
                clean=blob["clean"],
                track_idx=idx,
                track_name=str(entry["track_name"]),
                num_frames=frames,
                val_windows=plan_val_windows(frames, idx, cfg),
            )
        )
    if not tracks:
        raise ValueError(f"no tracks matched (held_out={held_out})")
    return tracks, manifest


def _crop(track: PairTrack, start_frame: int, geom: SpecGeom, frames: int) -> dict:
    """
    Cut the same span out of both streams.

    Args:
      track (PairTrack): the track to cut.
      start_frame (int): first frame of the crop.
      geom (SpecGeom): geometry, for the frame -> sample conversion.
      frames (int): crop length in frames.

    Returns:
      dict: {"rt": (1, L), "clean": (1, L)} float32, L = geom.crop_samples(frames).
    """
    lo = start_frame * geom.hop
    hi = lo + geom.crop_samples(frames)
    return {
        "rt": track.rt[lo:hi].float().unsqueeze(0).clone(),
        "clean": track.clean[lo:hi].float().unsqueeze(0).clone(),
    }


class PairCropDataset(torch.utils.data.Dataset):
    """
    Random aligned crops from the training tracks, avoiding the val windows.

    Tracks are drawn in proportion to length so a 9-minute track is not sampled
    as often as a 3-minute one per unit of audio. Epoch length is nominal: runs
    here are wall-clock capped, and the epoch only exists to trigger checkpoints.
    """

    def __init__(
        self, tracks: list[PairTrack], cfg: DataCfg, geom: SpecGeom, epoch_len: int
    ):
        """
        Args:
          tracks (list[PairTrack]): training tracks.
          cfg (DataCfg): crop and split settings.
          geom (SpecGeom): STFT geometry.
          epoch_len (int): number of crops per epoch.
        """
        self.tracks = tracks
        self.cfg = cfg
        self.geom = geom
        self.epoch_len = epoch_len
        lengths = torch.tensor(
            [max(1, t.num_frames - cfg.crop_frames) for t in tracks], dtype=torch.double
        )
        self.weights = (lengths / lengths.sum()).tolist()

    def __len__(self) -> int:
        return self.epoch_len

    def __getitem__(self, index: int) -> dict:
        """
        Args:
          index (int): ignored; positions are drawn fresh each call.

        Returns:
          dict: {"rt": (1, L), "clean": (1, L)} float32.
        """
        rng = random.Random((torch.initial_seed() + index) & 0xFFFFFFFF)
        frames = self.cfg.crop_frames
        for _ in range(16):
            track = rng.choices(self.tracks, weights=self.weights, k=1)[0]
            span = track.num_frames - frames
            if span <= 0:
                continue
            start = rng.randrange(span)
            if not _overlaps(start, start + frames, track.val_windows):
                return _crop(track, start, self.geom, frames)
        return _crop(self.tracks[0], 0, self.geom, frames)


class ValPairDataset(torch.utils.data.Dataset):
    """
    Fixed crops from the held-out time windows, so val/loss is comparable.

    Positions are enumerated once at construction: a moving validation set makes
    a run's own loss curve unreadable, which is the point of the fixed windows.
    """

    def __init__(self, tracks: list[PairTrack], cfg: DataCfg, geom: SpecGeom):
        """
        Args:
          tracks (list[PairTrack]): training tracks (their val windows are used).
          cfg (DataCfg): crop and split settings.
          geom (SpecGeom): STFT geometry.
        """
        self.geom = geom
        self.frames = cfg.crop_frames
        self.items: list[tuple[PairTrack, int]] = []
        for track in tracks:
            for lo, hi in track.val_windows:
                room = hi - lo - self.frames
                if room <= 0:
                    continue
                for k in range(max(1, cfg.val_crops_per_window)):
                    step = room * k // max(1, cfg.val_crops_per_window)
                    self.items.append((track, lo + step))
        if not self.items:
            raise ValueError("no validation crops; widen val_frac or shrink crop_frames")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        """
        Args:
          index (int): crop index.

        Returns:
          dict: {"rt": (1, L), "clean": (1, L)} float32.
        """
        track, start = self.items[index]
        return _crop(track, start, self.geom, self.frames)


# ===========================================================================
# Model
# ===========================================================================


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding of the flow time.

    Args:
      t (torch.Tensor): (B,) times in [0, 1].
      dim (int): embedding width, even.

    Returns:
      torch.Tensor: (B, dim) float32 embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    ang = t.float().reshape(-1, 1) * freqs.reshape(1, -1) * 1000.0
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


def _groups(requested: int, channels: int) -> int:
    """
    Largest group count that divides `channels` and does not exceed `requested`.

    Args:
      requested (int): configured group count.
      channels (int): channels to normalise.

    Returns:
      int: usable group count.
    """
    for g in range(min(requested, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Pre-norm residual block with FiLM conditioning on the flow time."""

    def __init__(
        self, in_ch: int, out_ch: int, t_dim: int, groups: int, dropout: float
    ):
        """
        Args:
          in_ch (int): input channels.
          out_ch (int): output channels.
          t_dim (int): time-embedding width.
          groups (int): requested GroupNorm groups.
          dropout (float): dropout before the second convolution.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(t_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(_groups(groups, out_ch), out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, in_ch, F, T) features.
          temb (torch.Tensor): (B, t_dim) time embedding.

        Returns:
          torch.Tensor: (B, out_ch, F, T) features.
        """
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(temb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Full self-attention over the flattened frequency-time grid."""

    def __init__(self, channels: int, heads: int, groups: int):
        """
        Args:
          channels (int): feature channels.
          heads (int): attention heads.
          groups (int): requested GroupNorm groups.
        """
        super().__init__()
        self.heads = _groups(heads, channels)
        self.norm = nn.GroupNorm(_groups(groups, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, C, F, T) features.

        Returns:
          torch.Tensor: (B, C, F, T) features.
        """
        b, c, f, t = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, f * t).unbind(1)
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, f, t))


class UNet2D(nn.Module):
    """
    Spectrogram UNet predicting the clean spectrogram from a flow iterate.

    Input is the iterate (2 channels) plus a normalised frequency coordinate (1).
    The coordinate matters: 2D convolutions are translation-equivariant, but
    spectrogram statistics are not remotely stationary along frequency.

    The round trip x0 is deliberately NOT an input, and `condition_on_x0` exists
    only to reproduce the version that showed why. Feeding both x_t and x0 makes
    the target algebraically recoverable -- x1 = (x_t - (1-t)*x0)/t up to the
    bridge noise -- so the network learns that inversion instead of restoration:
    measured val/spec fell to 0.143 against an identity baseline of 0.775 while
    the sampler, whose trajectory leaves the true interpolant, degraded
    monotonically with step count (mrstft 0.93 at 1 step, 1.87 at 8; crest 16.5
    -> 24.8 dB). x0 needs no channel of its own: it is where the trajectory
    starts.
    """

    def __init__(self, cfg: ModelCfg, out_ch: int = 2):
        """
        Args:
          cfg (ModelCfg): UNet geometry.
          out_ch (int): output channels.
        """
        super().__init__()
        in_ch = 5 if cfg.condition_on_x0 else 3
        self.cfg = cfg
        t_dim = cfg.t_embed_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim * 4)
        )
        t_out = t_dim * 4
        chans = [cfg.base_ch * m for m in cfg.ch_mults]

        self.stem = nn.Conv2d(in_ch, chans[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skips = [chans[0]]
        prev = chans[0]
        for level, ch in enumerate(chans):
            blocks = nn.ModuleList()
            for _ in range(cfg.blocks_per_level):
                blocks.append(ResBlock(prev, ch, t_out, cfg.groups, cfg.dropout))
                prev = ch
                skips.append(prev)
            self.down.append(blocks)
            last = level == len(chans) - 1
            self.downsample.append(
                nn.Identity() if last else nn.Conv2d(ch, ch, 3, stride=2, padding=1)
            )
            if not last:
                skips.append(prev)

        self.mid1 = ResBlock(prev, prev, t_out, cfg.groups, cfg.dropout)
        self.mid_attn = (
            AttnBlock(prev, cfg.attn_heads, cfg.groups)
            if cfg.attn_at_bottleneck
            else nn.Identity()
        )
        self.mid2 = ResBlock(prev, prev, t_out, cfg.groups, cfg.dropout)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for level, ch in reversed(list(enumerate(chans))):
            blocks = nn.ModuleList()
            for _ in range(cfg.blocks_per_level + 1):
                blocks.append(
                    ResBlock(prev + skips.pop(), ch, t_out, cfg.groups, cfg.dropout)
                )
                prev = ch
            self.up.append(blocks)
            self.upsample.append(
                nn.Identity() if level == 0 else nn.Conv2d(ch, ch, 3, padding=1)
            )

        self.out_norm = nn.GroupNorm(_groups(cfg.groups, prev), prev)
        self.out_conv = nn.Conv2d(prev, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the clean spectrogram.

        Args:
          x_t (torch.Tensor): (B, 2, F, T) current flow iterate.
          cond (torch.Tensor): (B, 2, F, T) round-trip spectrogram; used only
            when the rejected condition_on_x0 option is on.
          t (torch.Tensor): (B,) flow time in [0, 1].

        Returns:
          torch.Tensor: (B, 2, F, T) predicted clean spectrogram, x1_hat.
        """
        b, _, f, n = x_t.shape
        coord = torch.linspace(0.0, 1.0, f, device=x_t.device, dtype=x_t.dtype)
        coord = coord.reshape(1, 1, f, 1).expand(b, 1, f, n)
        temb = self.t_mlp(timestep_embedding(t, self.cfg.t_embed_dim))

        parts = [x_t, cond, coord] if self.cfg.condition_on_x0 else [x_t, coord]
        h = self.stem(torch.cat(parts, dim=1))
        skips = [h]
        for blocks, down in zip(self.down, self.downsample):
            for block in blocks:
                h = block(h, temb)
                skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)

        for blocks, up in zip(self.up, self.upsample):
            for block in blocks:
                h = block(torch.cat([h, skips.pop()], dim=1), temb)
            if not isinstance(up, nn.Identity):
                h = up(F.interpolate(h, scale_factor=2.0, mode="nearest"))

        # Residual on the iterate with a zero-initialised head: an untrained model
        # returns x_t unchanged, which at t=0 is exactly the round trip -- already
        # a strong answer -- and at t=1 is exactly the target.
        base = cond if self.cfg.condition_on_x0 else x_t
        return base + self.out_conv(F.silu(self.out_norm(h)))


# ===========================================================================
# Flow module
# ===========================================================================


class FlowModule(L.LightningModule):
    """
    Bridge flow from the round trip to the true audio.

    The interpolant runs between two *data* endpoints rather than noise and data:

        gam = sigma * sqrt(t * (1 - t))
        x_t = (1 - t) * x0 + t * x1 + gam * eps

    The Brownian-bridge envelope vanishes at both ends, so t=0 is exactly the
    round trip the sampler starts from and t=1 is exactly the target. sigma > 0
    is what makes the map one-to-many: unconstrained decoder phase is genuinely
    not a deterministic function of the round trip, and a sigma of 0 collapses
    the model onto the conditional mean, which is audible as smoothing.

    The network predicts x1 rather than a velocity. Velocity blows up as t -> 1
    (it carries a 1/(1-t)), whereas an x1 prediction is well scaled everywhere
    and can be handed straight to waveform-domain losses.
    """

    def __init__(self, cfg: FlowConfig, geom: SpecGeom):
        """
        Args:
          cfg (FlowConfig): full config.
          geom (SpecGeom): STFT geometry with the measured beta.
        """
        super().__init__()
        self.cfg = cfg
        self.geom = geom
        self.net = UNet2D(cfg.model)
        self.save_hyperparameters({"cfg": asdict(cfg), "geom": asdict(geom)})

    def gamma(self, t: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
        """
        Brownian-bridge noise envelope, zero at both endpoints.

        Overridable because training and sampling need not use the same scale:
        the noise is what makes the map one-to-many during training, but every
        injection at sampling time is independent complex noise sprayed across
        STFT bins, which inverts to metallic "musical noise" and accumulates with
        step count.

        Args:
          t (torch.Tensor): times in [0, 1], any shape.
          sigma (float | None): scale override; the config's when None.

        Returns:
          torch.Tensor: the standard deviation at each time.
        """
        scale = self.cfg.flow.sigma if sigma is None else sigma
        return scale * (t * (1.0 - t)).clamp_min(0.0).sqrt()

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Draw one point on the bridge.

        Args:
          x0 (torch.Tensor): (B, 2, F, T) round-trip spectrogram.
          x1 (torch.Tensor): (B, 2, F, T) clean spectrogram.
          t (torch.Tensor): (B,) times in [0, 1].
          noise (torch.Tensor | None): the epsilon to use; drawn fresh if None.

        Returns:
          torch.Tensor: (B, 2, F, T) the iterate x_t.
        """
        tt = t.reshape(-1, 1, 1, 1)
        eps = torch.randn_like(x1) if noise is None else noise
        return (1.0 - tt) * x0 + tt * x1 + self.gamma(tt) * eps

    def training_iterate(
        self, x0: torch.Tensor, x1: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce the point to train on: the sampler's own trajectory, or the ideal one.

        The ideal interpolant is built from the TRUE x1, which the sampler never
        has. Training only on it leaves two measured defects: the t=0 estimate
        barely improves at all (1.05x better than copying its input after 8.5 h,
        against 1.9-2.6x everywhere else), and from step two onward the model is
        fed its own error and compounds it -- heard as a metallic layer that
        grows with step count, and seen as spectral centroid climbing 2437 ->
        5450 Hz over eight steps.

        Rolling the sampler forward with no gradient and training at wherever it
        lands fixes both, because the inputs then are the inputs at inference:
        step 0 is exactly x0 at t=0, and later steps carry the model's own drift.
        `rollout_prob` mixes in the cheap ideal-interpolant samples, which cost
        one forward instead of i+1.

        Args:
          x0 (torch.Tensor): (B, 2, F, T) round-trip spectrogram.
          x1 (torch.Tensor): (B, 2, F, T) clean spectrogram.
          length (int): samples the spectrogram covers.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: the iterate and its (B,) times.
        """
        batch = x1.shape[0]
        steps = self.cfg.flow.steps
        if self.training and torch.rand(()).item() < self.cfg.flow.rollout_prob:
            was_training = self.net.training
            self.net.eval()
            x_t, t = self.walk(x0, length, steps, int(torch.randint(steps, ()).item()))
            self.net.train(was_training)
            return x_t, t.expand(batch)
        eps = self.cfg.flow.t_eps
        t = torch.rand(batch, device=x1.device) * (1.0 - 2.0 * eps) + eps
        return self.interpolate(x0, x1, t), t

    def _run(self, batch: dict, stage: str) -> torch.Tensor:
        """
        One training or validation step.

        Args:
          batch (dict): {"rt": (B, 1, L), "clean": (B, 1, L)}.
          stage (str): "train" or "val", used for the log prefix.

        Returns:
          torch.Tensor: () scalar loss.
        """
        rt, clean = batch["rt"], batch["clean"]
        length = rt.shape[-1]
        x0, x1 = stft_c(rt, self.geom), stft_c(clean, self.geom)
        x_t, t = self.training_iterate(x0, x1, length)
        x1_hat = self.net(x_t, x0, t)

        l_spec = (x1_hat - x1).abs().mean()
        wav_hat = istft_c(x1_hat, self.geom, length)
        l_wav = multi_res_stft_distance(wav_hat, clean)
        l_cons = stft_consistency(
            uncompress(x1_hat, self.geom),
            wav_hat,
            n_fft=self.geom.n_fft,
            hop=self.geom.hop,
            shift=0,
        )
        # Magnitude-weighted cos(phase error), normalised, so it cannot be
        # satisfied by shrinking magnitude the way the complex L1 term can.
        l_phase = phase_distance(wav_hat, clean)
        weights = self.cfg.loss
        loss = (
            weights.w_spec * l_spec
            + weights.w_wav * l_wav
            + weights.w_cons * l_cons
            + weights.w_phase * l_phase
        )

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/spec", l_spec, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/wav", l_wav, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/cons", l_cons, on_step=on_step, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/phase", l_phase, on_step=on_step, on_epoch=True, sync_dist=True)
        if stage == "val":
            # The bar to clear: doing nothing at all. If the model is not below
            # this, it is not enhancing anything.
            self.log(
                "val/identity_spec",
                (x0 - x1).abs().mean(),
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def training_step(self, batch: dict, index: int) -> torch.Tensor:
        """
        Args:
          batch (dict): training batch.
          index (int): batch index, unused.

        Returns:
          torch.Tensor: () scalar loss.
        """
        return self._run(batch, "train")

    def validation_step(self, batch: dict, index: int) -> torch.Tensor:
        """
        Args:
          batch (dict): validation batch.
          index (int): batch index; the sampler read-out runs on the first only.

        Returns:
          torch.Tensor: () scalar loss.
        """
        loss = self._run(batch, "val")
        if index == 0:
            self._log_sampler_metrics(batch)
        return loss

    @torch.no_grad()
    def _log_sampler_metrics(self, batch: dict) -> None:
        """
        Score what the model actually delivers, against doing nothing.

        The training loss averages over t, where large t is easy -- x_t is
        already mostly the target, so the loss can fall a long way while the
        t=0 answer the sampler starts from stays bad. This runs the real Euler
        sampler and compares it with the round trip on the same metric, so a run
        shows continuously whether C beats B rather than only at eval time.

        Args:
          batch (dict): {"rt": (B, 1, L), "clean": (B, 1, L)}.
        """
        rt, clean = batch["rt"], batch["clean"]
        # Both step counts, because they answer different questions: one step is
        # the quality of the t=0 estimate alone, the full count is whether
        # iterating on it pays. Later steps trust x_t more (it is trained to
        # carry t*x1), so a weak t=0 estimate is amplified rather than corrected
        # until that estimate is actually good.
        for tag, steps in (("sample1", 1), ("sample", None)):
            out = self.sample(rt, steps=steps)
            self.log(
                f"val/{tag}_mrstft",
                multi_res_stft_distance(out, clean),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val/{tag}_phase", phase_distance(out, clean), on_epoch=True, sync_dist=True
            )
        for tag, fn in (("mrstft", multi_res_stft_distance), ("phase", phase_distance)):
            self.log(f"val/rt_{tag}", fn(rt, clean), on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def walk(
        self,
        x0: torch.Tensor,
        length: int,
        steps: int,
        stop_at: int,
        project_steps: bool | None = None,
        sigma: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Take `stop_at` sampler steps from the round trip and report where it got to.

        Each step re-anchors to x0: the model's current estimate of x1 is
        substituted into the interpolant at the next time, with that time's own
        noise. Every iterate is therefore the shape the network was trained on,
        (1-t)*x0 + t*x1 + gamma(t)*eps, with the true x1 replaced by the estimate.

        A free Euler integration is what the x1 parameterisation invites, and it
        fails here: the marginals it produces carry no gamma(t) noise at any t.
        Measured, that ran mrstft to 2.18 against a round-trip baseline of 0.86.

        Args:
          x0 (torch.Tensor): (B, 2, F, T) round-trip spectrogram.
          length (int): samples the spectrogram covers, for the projection.
          steps (int): total steps the grid is divided into.
          stop_at (int): steps to actually take; 0 returns x0 at t=0.
          project_steps (bool | None): per-step projection, config when None.
          sigma (float | None): bridge-noise scale, config when None.
          generator (torch.Generator | None): draw noise from this instead of
            global state, so a sampled result is reproducible.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: the iterate and its time, a scalar.
        """
        do_project = (
            self.cfg.flow.project_every_step if project_steps is None else project_steps
        )
        grid = torch.linspace(0.0, 1.0, steps + 1, device=x0.device)
        x = x0
        for i in range(stop_at):
            t_next = grid[i + 1]
            x1_hat = self.net(x, x0, grid[i].expand(x.shape[0]))
            eps = torch.randn(
                x.shape, generator=generator, device=x.device, dtype=x.dtype
            )
            x = (1.0 - t_next) * x0 + t_next * x1_hat + self.gamma(t_next, sigma) * eps
            if do_project:
                x = project(x, self.geom, length)
        return x, grid[stop_at]

    @torch.no_grad()
    def sample(
        self,
        rt: torch.Tensor,
        steps: int | None = None,
        project_steps: bool | None = None,
        sigma: float | None = None,
    ) -> torch.Tensor:
        """
        Walk the bridge from the round trip to a clean waveform.

        Each step re-anchors to x0 rather than free-running: the model's current
        estimate of x1 is substituted into the interpolant at the next time, with
        that time's own noise. Every iterate is therefore of exactly the form the
        network was trained on, (1-t)*x0 + t*x1 + gamma(t)*eps, with the true x1
        replaced by the estimate.

        A free Euler integration is what the x1 parameterisation invites, and it
        fails here: the marginals it produces carry no gamma(t) noise at all, so
        each call sees an input unlike anything in training and the error
        compounds. Measured, that ran mrstft to 2.18 against a round-trip
        baseline of 0.86, worsening monotonically with step count, while the
        same model scored 0.73 on the true interpolant.

        The final step has t=1, where gamma is 0 and the interpolant is the
        estimate itself, so the walk lands on the prediction rather than near it.

        Args:
          rt (torch.Tensor): (B, 1, L) round-trip waveform.
          steps (int | None): steps to take; config default when None.
          project_steps (bool | None): project each iterate back onto the range
            of the STFT operator; config default when None.
          sigma (float | None): bridge-noise scale for this walk only; the
            config's when None. Note a single step lands at t=1 where gamma is
            0, so it injects nothing whatever sigma says.

        Returns:
          torch.Tensor: (B, 1, L) enhanced waveform.
        """
        steps = steps or self.cfg.flow.steps
        length = rt.shape[-1]
        x0 = stft_c(rt, self.geom)
        # Seeded rather than drawn from global state: two calls on the same audio
        # must agree, both for reproducible evaluation and so that overlapping
        # windows of a long file do not disagree inside their crossfade.
        gen = torch.Generator(device=x0.device).manual_seed(self.cfg.train.seed)
        x, _ = self.walk(x0, length, steps, steps, project_steps, sigma, gen)
        return istft_c(x, self.geom, length)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        AdamW with decay only on matmul weights, plus the repo's wall-clock cycle.

        Returns:
          dict[str, Any]: Lightning optimizer/scheduler bundle.
        """
        decay, no_decay = [], []
        for _, param in self.net.named_parameters():
            if param.requires_grad:
                (no_decay if param.ndim < 2 else decay).append(param)
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


def enhance_long(
    module: FlowModule,
    wav: torch.Tensor,
    win_frames: int | None = None,
    steps: int | None = None,
    project_steps: bool | None = None,
    sigma: float | None = None,
) -> torch.Tensor:
    """
    Run the sampler over audio longer than one crop, with a crossfaded overlap.

    The UNet is fully convolutional and would accept a whole track, but the
    bottleneck attention is quadratic in the frequency-time grid, so a window
    four times the training crop costs sixteen times the memory. Defaulting the
    window to the training crop keeps that bounded and keeps inference on the
    context length the model was actually trained on. Windows overlap by half and
    are cosine-crossfaded in the waveform domain; the sampler is deterministic,
    so neighbours agree closely wherever they meet.

    Args:
      module (FlowModule): trained model, already on the target device.
      wav (torch.Tensor): (1, 1, L) round-trip waveform.
      win_frames (int | None): window length in STFT frames; the training crop
        when None.
      steps (int | None): Euler steps, config default when None.
      project_steps (bool | None): per-step projection, config default when None.
      sigma (float | None): bridge-noise scale, config default when None.

    Returns:
      torch.Tensor: (1, 1, L) enhanced waveform.
    """
    geom = module.geom
    total = wav.shape[-1]
    win = geom.crop_samples(win_frames or module.cfg.data.crop_frames)
    if total <= win:
        return module.sample(wav, steps, project_steps, sigma)

    stride = win // 2
    out = torch.zeros_like(wav)
    norm = torch.zeros_like(wav)
    ramp = torch.hann_window(win, periodic=False, device=wav.device).reshape(1, 1, -1)
    start = 0
    while True:
        end = min(start + win, total)
        lo = end - win
        out[..., lo:end] += (
            module.sample(wav[..., lo:end], steps, project_steps, sigma) * ramp
        )
        norm[..., lo:end] += ramp
        if end >= total:
            break
        start += stride
    # The Hann ramp is zero at both ends, so the very first and last samples get
    # no weight from any window; the clamp leaves them as written, not as NaN.
    return out / norm.clamp_min(1e-6)


# ===========================================================================
# Entry points
# ===========================================================================


def build_geom(cfg: FlowConfig, manifest: dict[str, Any]) -> SpecGeom:
    """
    Assemble the STFT geometry, preferring an explicit beta from config.

    Args:
      cfg (FlowConfig): full config.
      manifest (dict[str, Any]): the pair-cache manifest.

    Returns:
      SpecGeom: geometry with alpha and beta resolved.
    """
    beta = cfg.data.beta if cfg.data.beta is not None else float(manifest["beta"])
    return SpecGeom(
        n_fft=int(manifest["n_fft"]),
        hop=int(manifest["hop_length"]),
        alpha=float(cfg.data.alpha),
        beta=beta,
        sample_rate=int(manifest["sample_rate"]),
    )


def load_flow_module(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> FlowModule:
    """
    Rebuild a trained enhancer from a Lightning checkpoint.

    The config and geometry travel inside the checkpoint's hyperparameters, so a
    caller does not have to find the YAML that produced it and cannot pair a
    checkpoint with the wrong beta.

    Args:
      checkpoint (str | Path): path to the .ckpt.
      device (str | torch.device): where to place the model.

    Returns:
      FlowModule: model in eval mode with weights loaded.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = state["hyper_parameters"]
    module = FlowModule(config_from_dict(hparams["cfg"]), SpecGeom(**hparams["geom"]))
    module.load_state_dict(state["state_dict"], strict=True)
    return module.to(device).eval()


def build_dataloaders(
    cfg: FlowConfig, tracks: list[PairTrack], geom: SpecGeom
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Args:
      cfg (FlowConfig): full config.
      tracks (list[PairTrack]): training tracks.
      geom (SpecGeom): STFT geometry.

    Returns:
      tuple[DataLoader, DataLoader]: train and validation loaders.
    """
    epoch_len = cfg.data.steps_per_epoch * cfg.train.batch_size
    train_set = PairCropDataset(tracks, cfg.data, geom, epoch_len)
    val_set = ValPairDataset(tracks, cfg.data, geom)
    common = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": True,
        "persistent_workers": cfg.train.num_workers > 0,
    }
    return (
        torch.utils.data.DataLoader(train_set, shuffle=False, drop_last=True, **common),
        torch.utils.data.DataLoader(val_set, shuffle=False, **common),
    )


def build_trainer(cfg: FlowConfig) -> L.Trainer:
    """
    Assemble a Lightning trainer matching the repo's run conventions.

    Args:
      cfg (FlowConfig): full config.

    Returns:
      L.Trainer: configured trainer.
    """
    save_path = REPO / cfg.train.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.Timer(duration={"minutes": cfg.train.minutes}),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="flow_best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        # Unmonitored rolling save: this is the one that survives a crash and a
        # mid-epoch Timer stop.
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="flow_latest",
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
        logger=L.pytorch.loggers.TensorBoardLogger(str(save_path), name="flow"),
        log_every_n_steps=10,
    )


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "config_flow.yaml"))
    parser.add_argument(
        "--build-cache", action="store_true", help="build the pair cache and exit"
    )
    parser.add_argument(
        "--force-rebuild", action="store_true", help="rebuild the cache even if present"
    )
    return parser.parse_args()


def main() -> None:
    """Build the pair cache if needed, then train."""
    args = parse_args()
    cfg = load_config(args.config)
    cache_dir = build_pair_cache(cfg, force=args.force_rebuild)
    if args.build_cache:
        return

    L.seed_everything(cfg.train.seed, workers=True)
    tracks, manifest = load_pair_cache(cache_dir, cfg.data)
    geom = build_geom(cfg, manifest)
    train_loader, val_loader = build_dataloaders(cfg, tracks, geom)
    module = FlowModule(cfg, geom)
    params = sum(p.numel() for p in module.net.parameters())
    print(
        f"tracks {len(tracks)} (held out {manifest['held_out']})  "
        f"crop {cfg.data.crop_frames}f = {geom.crop_samples(cfg.data.crop_frames)} samples  "
        f"beta {geom.beta:.4f}  params {params / 1e6:.1f}M  "
        f"val crops {len(val_loader.dataset)}"
    )

    if cfg.train.checkpoint:
        state = torch.load(
            REPO / cfg.train.checkpoint, map_location="cpu", weights_only=False
        )
        missing = module.load_state_dict(state["state_dict"], strict=False)
        print(f"warm start from {cfg.train.checkpoint}: {missing}")

    build_trainer(cfg).fit(module, train_loader, val_loader)
    save_path = REPO / cfg.train.save_path
    print(f"done -> {save_path}")


if __name__ == "__main__":
    main()
