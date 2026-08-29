"""
Autoregressive generative stage over the frozen VQ-GAN's RVQ tokens.

Implements PLAN_generative_stage.md section 8 step 4: a decoder-only transformer
over whole-track token streams, MusicGen-style delay pattern on the depth axis
(section 2), and classifier-free guidance with two independently-dropped
conditioning streams (sections 11.1-11.2).

Three entry points:

    uv run python train_ar.py --config config_ar.yaml --build-cache
    uv run python train_ar.py --config config_ar.yaml --bench-attn
    uv run python train_ar.py --config config_ar.yaml

The token cache is built automatically by the training path when absent.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset

# Wall-clock one-cycle LR and the EMA callback are the repo's own; reusing them
# keeps the `minutes:` config idiom working identically to train.py.
from train import EMA, TimeOneCycleLR

REPO = Path(__file__).resolve().parent


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class TokenizerCfg:
    """Frozen tokenizer and token-cache build settings."""

    encoder_onnx: str = "onnx/encoder.onnx"
    meta: str = "onnx/tokenizer_meta.json"
    checkpoint: str = "saved_20260827_cont9h/lvl1_vqgan_last.ckpt"
    tracks_dir: str = "~/.cache/infected_pbm/tracks"
    slices_dir: str = "~/.cache/infected_pbm/slices"
    cache_root: str = "~/.cache/infected_pbm"
    chunk_frames: int = 4096
    margin: int = 256
    verify_tolerance: float = 0.005


@dataclass
class DataCfg:
    """Cropping, conditioning windows and the train/val time split."""

    crop_frames: int = 4096
    frames_per_pos: int = 1
    style_window_sec: float = 10.0
    val_frac: float = 0.04
    val_windows_per_track: int = 2
    min_val_window: int = 512
    split_seed: int = 1234
    track_weighting: str = "length"
    steps_per_epoch: int = 500
    single_track: str | None = None


@dataclass
class ModelCfg:
    """Transformer trunk and conditioning geometry."""

    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.0
    style_bottleneck: int = 128
    p_drop_cond: float = 0.1
    rope_theta: float = 10000.0


@dataclass
class TrainCfg:
    """Optimiser, schedule and checkpointing."""

    devices: int = 1
    minutes: float = 360.0
    lr: float = 3.0e-4
    lr_pct_start: float = 0.05
    lr_div_factor: float = 25.0
    batch_size: int = 8
    precision: str = "bf16-mixed"
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 4
    ema_decay: float | None = None
    seed: int = 42
    save_path: str = "saved_ar/"
    checkpoint: str | None = None


@dataclass
class ArConfig:
    """Top-level config, one block per YAML section."""

    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def _build_section(cls: type, raw: dict[str, Any] | None, name: str):
    """
    Instantiate one config dataclass, rejecting unknown keys.

    Silent key drops are how a config typo turns into a run that looks fine and
    answers the wrong question, so unknown keys raise.

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


def load_config(path: str | Path) -> ArConfig:
    """
    Read an AR training config from YAML.

    Args:
      path (str | Path): path to the .yaml file.

    Returns:
      ArConfig: fully populated config with defaults filled in.
    """
    with open(path, "r") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    unknown = set(raw) - {"tokenizer", "data", "model", "train"}
    if unknown:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")
    return ArConfig(
        tokenizer=_build_section(TokenizerCfg, raw.get("tokenizer"), "tokenizer"),
        data=_build_section(DataCfg, raw.get("data"), "data"),
        model=_build_section(ModelCfg, raw.get("model"), "model"),
        train=_build_section(TrainCfg, raw.get("train"), "train"),
    )


# ===========================================================================
# Token cache
# ===========================================================================

CODEBOOK_KEY = "model.vq_module.vq_codebook.code_embedding"


def _preload_cudnn() -> None:
    """
    Register libcudnn under the unversioned name ONNX Runtime dlopens.

    ORT looks for "libcudnn.so"; PyTorch ships only "libcudnn.so.9". Loading the
    symlinked path RTLD_GLOBAL satisfies ORT. Must run before the first session.
    """
    lib = REPO / ".venv/lib/python3.13/site-packages/nvidia/cudnn/lib/libcudnn.so"
    if lib.exists():
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            print(f"cudnn preload failed: {exc}")


def make_session(path: Path, num_rq: int):
    """
    Open an ONNX graph on the GPU when available, else the CPU.

    TF32 is forced off. ORT enables it by default on Ampere and later, which
    drops convolutions to ~10 mantissa bits -- enough to flip codebook argmin
    near-ties (the notebook measured 56/15360 tokens changed, -11.6 dB). A token
    cache written with TF32 on would be quietly wrong.

    Args:
      path (Path): the .onnx file.
      num_rq (int): RVQ depth, for building the warmup input.

    Returns:
      onnxruntime.InferenceSession: a warmed-up session.
    """
    import onnxruntime as ort

    def warmup(sess) -> np.ndarray:
        spec = sess.get_inputs()[0]
        if "int64" in spec.type:
            return np.zeros((1, 128, num_rq), dtype=np.int64)
        return np.zeros((1, 1, 32768), dtype=np.float32)

    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            sess = ort.InferenceSession(
                str(path), providers=[("CUDAExecutionProvider", {"use_tf32": "0"})]
            )
            sess.run(None, {sess.get_inputs()[0].name: warmup(sess)})
            return sess
        except Exception as exc:
            print(f"CUDA unavailable ({type(exc).__name__}), falling back to CPU")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def cache_tag(meta_path: Path, encoder_path: Path) -> str:
    """
    Content-address the cache directory to the exact tokenizer that filled it.

    A re-exported or retrained tokenizer must not silently reuse stale tokens,
    so the tag hashes the tokenizer metadata plus a prefix of the encoder graph.

    Args:
      meta_path (Path): tokenizer_meta.json.
      encoder_path (Path): encoder.onnx.

    Returns:
      str: a 12-character hex tag.
    """
    digest = hashlib.sha256()
    digest.update(meta_path.read_bytes())
    digest.update(str(encoder_path.stat().st_size).encode())
    with open(encoder_path, "rb") as handle:
        digest.update(handle.read(4 << 20))
    return digest.hexdigest()[:12]


def enumerate_tracks(tracks_dir: Path) -> list[Path]:
    """
    List the corpus mp3s in prepare.py's order.

    Mirrors prepare.py:174-182 (os.walk, .mp3 filter, sorted) so that track_idx
    here means the same thing as track_idx everywhere else in the repo.

    Args:
      tracks_dir (Path): directory holding the .mp3 files.

    Returns:
      list[Path]: sorted track paths.
    """
    found: list[str] = []
    for root, _, files in os.walk(tracks_dir):
        for name in files:
            if name.endswith(".mp3"):
                found.append(os.path.join(root, name))
    return [Path(p) for p in sorted(found)]


def _load_from_slice_cache(
    path: Path, slices_dir: Path, hop_length: int
) -> torch.Tensor | None:
    """
    Rebuild a track's waveform from prepare.py's cached slices.

    Seven of the 53 mp3s no longer decode under the installed torchcodec, but all
    of them were decoded successfully by the 2026-05-01 prepare.py run and survive
    as slice tensors. Concatenating those slices restores the identical contiguous
    waveform -- audio is being joined here, not tokens, so this introduces none of
    the seam problems that motivated whole-track encoding in the first place.

    Args:
      path (Path): the .mp3 whose slices are wanted.
      slices_dir (Path): prepare.py's slice cache.
      hop_length (int): STFT hop, 256.

    Returns:
      torch.Tensor | None: (1, L) float32 waveform, or None when no cache exists.
    """
    cached = slices_dir / f"slices_{path.stem.replace(' ', '_')}.pt"
    if not cached.exists():
        return None
    wav = torch.load(cached, map_location="cpu", weights_only=False).reshape(1, -1).float()
    # prepare.py right-pads the last slice with zeros; drop that tail.
    nonzero = (wav[0] != 0).nonzero()
    if nonzero.numel():
        wav = wav[..., : int(nonzero[-1]) + 1]
    usable = (wav.shape[-1] // hop_length) * hop_length
    return wav[..., :usable].contiguous()


def load_track_audio(
    path: Path, sample_rate: int, hop_length: int, slices_dir: Path | None = None
) -> torch.Tensor:
    """
    Load one track as mono at the tokenizer's rate, trimmed to whole frames.

    Follows prepare.py:196-203 for load/resample/mixdown, but trims to a multiple
    of hop_length rather than padding to a 32768 slice -- whole-track streams have
    no slice grid to respect. Falls back to the slice cache for mp3s the installed
    decoder rejects.

    Args:
      path (Path): the .mp3 file.
      sample_rate (int): target rate, 44100 for this tokenizer.
      hop_length (int): STFT hop, 256.
      slices_dir (Path | None): prepare.py's slice cache, used only on decode
        failure. None disables the fallback.

    Returns:
      torch.Tensor: (1, L) float32 mono waveform, L a multiple of hop_length.
    """
    try:
        wav, src_rate = torchaudio.load(str(path), format="mp3")
    except Exception as exc:
        fallback = (
            _load_from_slice_cache(path, slices_dir, hop_length) if slices_dir else None
        )
        if fallback is None:
            raise RuntimeError(f"cannot decode {path.name} and no slice cache") from exc
        print(f"    decode failed ({type(exc).__name__}); used slice cache")
        return fallback
    if src_rate != sample_rate:
        wav = torchaudio.transforms.Resample(src_rate, sample_rate)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    usable = (wav.shape[-1] // hop_length) * hop_length
    return wav[..., :usable].contiguous().float()


def encode_chunked(
    session,
    wav: torch.Tensor,
    hop_length: int,
    chunk_frames: int,
    margin: int,
) -> np.ndarray:
    """
    Encode a whole track in overlapping windows, discarding the margins.

    The encoder is fully convolutional (STFT plus Conv1d), so its receptive field
    is finite -- roughly 120 frames -- and decoding context of `margin` frames on
    each side makes chunked encoding exact rather than merely close. Chunking is
    needed for memory, not speed: an eight-minute track is ~83k frames, whose
    hidden-1024 activations do not fit alongside a running experiment.

    Args:
      session (onnxruntime.InferenceSession): the encoder graph.
      wav (torch.Tensor): (1, L) mono waveform, L a multiple of hop_length.
      hop_length (int): STFT hop, 256.
      chunk_frames (int): frames emitted per window.
      margin (int): context frames encoded and discarded on each side.

    Returns:
      np.ndarray: (T, R) int64 indices, T = L // hop_length.
    """
    total = wav.shape[-1] // hop_length
    audio = wav.unsqueeze(0).numpy().astype(np.float32)
    if total <= chunk_frames:
        return session.run(None, {"waveform": np.ascontiguousarray(audio)})[0][0]

    pieces: list[np.ndarray] = []
    pos = 0
    while pos < total:
        end = min(pos + chunk_frames, total)
        lo, hi = max(0, pos - margin), min(total, end + margin)
        window = np.ascontiguousarray(audio[..., lo * hop_length : hi * hop_length])
        idx = session.run(None, {"waveform": window})[0][0]
        pieces.append(idx[pos - lo : (pos - lo) + (end - pos)])
        pos = end
    return np.concatenate(pieces, axis=0)


def verify_chunking(
    session,
    wav: torch.Tensor,
    hop_length: int,
    chunk_frames: int,
    margin: int,
) -> dict[str, float]:
    """
    Check chunked encoding against a single pass, and against no margin at all.

    Exact equality is not achievable and is not the right bar. The encoder's
    token output is shape-dependent: two single passes over audio sharing the
    same right edge but differing in length disagree in tens of positions,
    because cuDNN picks different convolution algorithms per input shape and the
    codebook argmax flips on near-ties. There is therefore no canonical
    reference to match.

    What is checkable, and what this asserts:

    - level 0 must be bit-identical. It carries the structure the AR model
      depends on, and it was measured stable across every chunk size tried.
    - the margin must do real work: disagreements at margin=0 are ~27x higher,
      which is the receptive-field effect the margin exists to remove.
    - the residual rate must stay under tolerance. Measured at 0.02% overall,
      confined to levels 1-2, and worth ~1.7e-4 mean relative error in z_q.

    Args:
      session (onnxruntime.InferenceSession): the encoder graph.
      wav (torch.Tensor): (1, L) reference waveform.
      hop_length (int): STFT hop.
      chunk_frames (int): frames per window under test.
      margin (int): context frames per side under test.

    Returns:
      dict[str, float]: counts and rates -- "total", "mismatch", "rate",
        "level0_mismatch" and "mismatch_no_margin".
    """
    single = session.run(
        None, {"waveform": np.ascontiguousarray(wav.unsqueeze(0).numpy())}
    )[0][0]
    chunked = encode_chunked(session, wav, hop_length, chunk_frames, margin)
    bare = encode_chunked(session, wav, hop_length, chunk_frames, 0)
    mismatch = int((single != chunked).sum())
    return {
        "total": float(single.size),
        "mismatch": float(mismatch),
        "rate": mismatch / max(single.size, 1),
        "level0_mismatch": float((single[:, 0] != chunked[:, 0]).sum()),
        "mismatch_no_margin": float((single != bare).sum()),
    }


def extract_codebooks(ckpt_path: Path) -> torch.Tensor:
    """
    Pull the RVQ codebooks straight out of a Lightning checkpoint.

    Read by state_dict key rather than through train.py's build_generator: no
    model is constructed and the global RNG is never touched, so this cannot
    perturb anything downstream (cf. the split-RNG confound).

    Args:
      ckpt_path (Path): the .ckpt file.

    Returns:
      torch.Tensor: (R, N, C) float32 per-level codebooks.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    if CODEBOOK_KEY not in state:
        raise KeyError(f"{CODEBOOK_KEY} missing from {ckpt_path}")
    return state[CODEBOOK_KEY].detach().float().clone()


def compute_style_windows(
    tokens: torch.Tensor, codebooks: torch.Tensor, window_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Average the quantized latent over fixed windows to get style descriptors.

    z_q is the sum of the per-level codebook vectors at the cached indices, so
    this is a gather plus a sum -- no model runs. Precomputing per window rather
    than per crop keeps the whole corpus's conditioning at ~10 MB and makes
    __getitem__ a single row lookup.

    Args:
      tokens (torch.Tensor): (T, R) int64 indices for one track.
      codebooks (torch.Tensor): (R, N, C) per-level codebooks.
      window_frames (int): frames averaged per descriptor.

    Returns:
      tuple[torch.Tensor, torch.Tensor]: (W, C) float32 L2-normalized descriptors
        and (W, 2) int64 [start, end) frame bounds. The final window absorbs any
        remainder.
    """
    total, num_rq = tokens.shape
    num_win = max(1, total // window_frames)
    bounds = torch.zeros((num_win, 2), dtype=torch.int64)
    vectors = torch.zeros((num_win, codebooks.shape[-1]), dtype=torch.float32)
    for w in range(num_win):
        start = w * window_frames
        end = total if w == num_win - 1 else (w + 1) * window_frames
        chunk = tokens[start:end]
        z_q = torch.zeros((end - start, codebooks.shape[-1]), dtype=torch.float32)
        for level in range(num_rq):
            z_q += codebooks[level][chunk[:, level]]
        vectors[w] = F.normalize(z_q.mean(dim=0), dim=-1)
        bounds[w] = torch.tensor([start, end])
    return vectors, bounds


def plan_val_windows(
    num_frames: int, track_idx: int, cfg: DataCfg
) -> list[tuple[int, int]]:
    """
    Choose the held-out time windows for one track.

    The existing prepare.py split is per-0.74s slice, so every track sits in both
    train and val -- useless for measuring memorization. This replaces it with
    contiguous windows at random positions inside each track, sized to `val_frac`.

    Randomness comes from a dedicated Random(split_seed) keyed on the track's
    length, never the global RNG: the split is a property of the data alone and
    must not shift when the model's parameter count changes.

    Placement divides the interior into equal segments and draws one window per
    segment, which guarantees non-overlap without rejection sampling.

    Args:
      num_frames (int): length of the track's token stream.
      track_idx (int): index of the track, so two equal-length tracks differ.
      cfg (DataCfg): split settings.

    Returns:
      list[tuple[int, int]]: [start, end) frame ranges reserved for validation.
    """
    edge = int(0.05 * num_frames)
    span = num_frames - 2 * edge
    if span <= 0:
        return []

    count = max(1, cfg.val_windows_per_track)
    budget = int(num_frames * cfg.val_frac)
    length = max(cfg.min_val_window, budget // count)
    segment = span // count
    if length >= segment:
        length = max(1, segment // 2)

    if segment <= length:
        return []

    # Seeded from a stable string digest rather than hash(), which is only
    # deterministic for ints by implementation detail.
    key = f"{cfg.split_seed}:{track_idx}:{num_frames}:{count}:{length}"
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    windows: list[tuple[int, int]] = []
    for i in range(count):
        lo = edge + i * segment
        start = rng.randint(lo, lo + segment - length)
        windows.append((start, start + length))
    return windows


def build_token_cache(cfg: ArConfig, force: bool = False) -> Path:
    """
    Tokenize the whole corpus with the frozen ONNX encoder and cache it to disk.

    One .pt per track holding the full token stream, so training crops come from
    contiguous audio at arbitrary offsets. The previous approach -- encoding
    32768-sample slices independently and concatenating -- put a padding-
    contaminated seam every 0.743 s, the encode-side twin of the 1.35 Hz click
    train render_samples.py already fixed on the decode side.

    Args:
      cfg (ArConfig): full config; tokenizer and data sections are used.
      force (bool): rebuild even if a complete cache already exists.

    Returns:
      Path: the cache directory.
    """
    tok = cfg.tokenizer
    enc_path = REPO / tok.encoder_onnx
    meta_path = REPO / tok.meta
    meta: dict[str, Any] = json.loads(meta_path.read_text())
    hop, rate = meta["hop_length"], meta["sample_rate"]

    cache_dir = (
        Path(os.path.expanduser(tok.cache_root))
        / f"tokens_{cache_tag(meta_path, enc_path)}"
    )
    manifest_path = cache_dir / "_manifest.json"
    if manifest_path.exists() and not force:
        print(f"token cache present: {cache_dir}")
        return cache_dir

    tracks = enumerate_tracks(Path(os.path.expanduser(tok.tracks_dir)))
    if not tracks:
        raise FileNotFoundError(f"no .mp3 under {tok.tracks_dir}")
    print(f"building token cache for {len(tracks)} tracks -> {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    _preload_cudnn()
    session = make_session(enc_path, meta["num_rq"])
    print(f"encoder on {session.get_providers()[0]}")

    slices_dir = Path(os.path.expanduser(tok.slices_dir))
    probe = load_track_audio(tracks[0], rate, hop, slices_dir)[..., : 60 * rate]
    report = verify_chunking(session, probe, hop, tok.chunk_frames, tok.margin)
    print(
        f"chunk check over 60 s: {report['mismatch']:.0f}/{report['total']:.0f} tokens "
        f"differ ({100 * report['rate']:.4f}%), level 0 {report['level0_mismatch']:.0f}, "
        f"margin=0 would give {report['mismatch_no_margin']:.0f}"
    )
    if report["level0_mismatch"]:
        raise RuntimeError(
            f"level 0 differs in {report['level0_mismatch']:.0f} positions; "
            f"raise tokenizer.margin above {tok.margin} and rebuild"
        )
    if report["rate"] > tok.verify_tolerance:
        raise RuntimeError(
            f"chunked encoding differs in {100 * report['rate']:.4f}% of tokens, "
            f"over the {100 * tok.verify_tolerance:.4f}% tolerance; raise "
            f"tokenizer.margin above {tok.margin} and rebuild"
        )

    window_frames = max(1, int(round(cfg.data.style_window_sec * meta["frames_per_second"])))
    codebooks = extract_codebooks(REPO / tok.checkpoint)
    torch.save(codebooks, cache_dir / "codebooks.pt")

    entries: list[dict[str, Any]] = []
    for track_idx, path in enumerate(tracks):
        wav = load_track_audio(path, rate, hop, slices_dir)
        tokens = torch.from_numpy(
            encode_chunked(session, wav, hop, tok.chunk_frames, tok.margin)
        ).long()
        if int(tokens.max()) >= torch.iinfo(torch.int16).max:
            raise ValueError("token ids exceed int16 range")
        style, bounds = compute_style_windows(tokens, codebooks, window_frames)

        name = path.stem
        rel = f"{track_idx:03d}_{name.replace(' ', '_')}.pt"
        torch.save(
            {
                "tokens": tokens.to(torch.int16),
                "style_windows": style,
                "style_bounds": bounds,
                "track_idx": track_idx,
                "track_name": name,
                "track_path": str(path),
                "num_frames": int(tokens.shape[0]),
                "duration_sec": float(tokens.shape[0] / meta["frames_per_second"]),
                "sample_rate": rate,
                "hop_length": hop,
                "frames_per_second": meta["frames_per_second"],
                "num_rq": meta["num_rq"],
                "num_tokens": meta["num_tokens"],
                "style_window_frames": window_frames,
            },
            cache_dir / rel,
        )
        entries.append(
            {
                "file": rel,
                "track_idx": track_idx,
                "track_name": name,
                "num_frames": int(tokens.shape[0]),
                "duration_sec": float(tokens.shape[0] / meta["frames_per_second"]),
                "val_windows": plan_val_windows(
                    int(tokens.shape[0]), track_idx, cfg.data
                ),
            }
        )
        print(f"  [{track_idx + 1:2d}/{len(tracks)}] {name[:48]:48s} {tokens.shape[0]:7d} frames")

    manifest = {
        "tokenizer_meta": meta,
        "checkpoint": tok.checkpoint,
        "encoder_onnx": tok.encoder_onnx,
        "chunk_frames": tok.chunk_frames,
        "margin": tok.margin,
        "verify": report,
        "style_window_frames": window_frames,
        "split_params": asdict(cfg.data),
        "tracks": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    frames = sum(e["num_frames"] for e in entries)
    print(
        f"cache built: {len(entries)} tracks, {frames} frames, "
        f"{frames / meta['frames_per_second'] / 3600:.2f} h"
    )
    return cache_dir


# ===========================================================================
# Dataset
# ===========================================================================


@dataclass
class TrackTokens:
    """One track's cached token stream plus its conditioning and held-out spans."""

    tokens: torch.Tensor
    style: torch.Tensor
    style_bounds: torch.Tensor
    track_idx: int
    track_name: str
    num_frames: int
    fps: float
    val_windows: list[tuple[int, int]]


def load_token_cache(cache_dir: Path, cfg: DataCfg) -> tuple[list[TrackTokens], dict[str, Any]]:
    """
    Load every cached track into RAM and recompute the val split from config.

    The corpus is ~23 MB of int16, so there is no reason for lazy loading. The
    split is recomputed here rather than read from the manifest so that changing
    split settings does not require re-tokenizing; it stays deterministic because
    plan_val_windows depends only on the data and split_seed.

    Args:
      cache_dir (Path): directory written by build_token_cache.
      cfg (DataCfg): split and filtering settings.

    Returns:
      tuple[list[TrackTokens], dict[str, Any]]: loaded tracks and the manifest.
    """
    manifest: dict[str, Any] = json.loads((cache_dir / "_manifest.json").read_text())
    tracks: list[TrackTokens] = []
    for entry in manifest["tracks"]:
        if cfg.single_track and cfg.single_track not in entry["track_name"]:
            continue
        blob = torch.load(cache_dir / entry["file"], map_location="cpu", weights_only=False)
        num_frames = int(blob["num_frames"])
        tracks.append(
            TrackTokens(
                tokens=blob["tokens"].long(),
                style=blob["style_windows"].float(),
                style_bounds=blob["style_bounds"].long(),
                track_idx=int(blob["track_idx"]),
                track_name=str(blob["track_name"]),
                num_frames=num_frames,
                fps=float(blob["frames_per_second"]),
                val_windows=plan_val_windows(num_frames, int(blob["track_idx"]), cfg),
            )
        )
    if not tracks:
        raise ValueError(f"no tracks matched single_track={cfg.single_track!r}")
    return tracks, manifest


def _overlaps(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    """
    Test a half-open range against a list of half-open ranges.

    Args:
      start (int): range start, inclusive.
      end (int): range end, exclusive.
      spans (Sequence[tuple[int, int]]): ranges to test against.

    Returns:
      bool: True if any span intersects [start, end).
    """
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def _pick_style(track: TrackTokens, start: int, end: int, rng: random.Random) -> int:
    """
    Choose a style window disjoint from the crop being predicted.

    Section 11.3: conditioning computed from the same window as the target is a
    compressed copy of the answer, which makes training loss look excellent and
    generation worthless. Drawing from elsewhere in the same track keeps the
    descriptor informative about style without leaking content.

    Args:
      track (TrackTokens): the track being cropped.
      start (int): crop start frame.
      end (int): crop end frame, exclusive.
      rng (random.Random): sampler.

    Returns:
      int: index into track.style. Falls back to the farthest window when the
        track is too short to hold a disjoint one.
    """
    bounds = track.style_bounds
    free = [
        w
        for w in range(bounds.shape[0])
        if not (start < int(bounds[w, 1]) and int(bounds[w, 0]) < end)
    ]
    if free:
        return rng.choice(free)
    centre = (start + end) / 2.0
    centres = (bounds[:, 0] + bounds[:, 1]).float() / 2.0
    return int((centres - centre).abs().argmax())


_WORKER_RNG: random.Random | None = None


def _worker_rng() -> random.Random:
    """
    Per-worker sampler seeded from the DataLoader's own seed.

    Returns:
      random.Random: a process-local RNG, distinct per worker and per epoch.
    """
    global _WORKER_RNG
    if _WORKER_RNG is None:
        _WORKER_RNG = random.Random(torch.initial_seed() & 0xFFFFFFFF)
    return _WORKER_RNG


class TokenCropDataset(Dataset):
    """
    Random contiguous crops of the token corpus, avoiding the held-out windows.

    Length is steps_per_epoch * batch_size rather than a corpus count: 3.88M
    frames is only ~950 non-overlapping crops at 4096, so an "epoch" here is a
    logging interval, and random offsets supply the variety.

    Args:
      tracks (list[TrackTokens]): loaded corpus.
      cfg (DataCfg): crop and weighting settings.
      length (int): number of items per epoch.
    """

    def __init__(self, tracks: list[TrackTokens], cfg: DataCfg, length: int) -> None:
        self._tracks = [t for t in tracks if t.num_frames > cfg.crop_frames]
        if not self._tracks:
            raise ValueError(f"no track longer than crop_frames={cfg.crop_frames}")
        self._cfg = cfg
        self._length = length
        if cfg.track_weighting == "length":
            self._weights = [float(t.num_frames) for t in self._tracks]
        elif cfg.track_weighting == "uniform":
            self._weights = [1.0 for _ in self._tracks]
        else:
            raise ValueError(f"unknown track_weighting {cfg.track_weighting!r}")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = _worker_rng()
        crop = self._cfg.crop_frames
        for _ in range(64):
            track = rng.choices(self._tracks, weights=self._weights, k=1)[0]
            start = rng.randrange(0, track.num_frames - crop)
            if not _overlaps(start, start + crop, track.val_windows):
                break
        else:
            raise RuntimeError("could not draw a crop clear of the val windows")

        style_idx = _pick_style(track, start, start + crop, rng)
        return {
            "tokens": track.tokens[start : start + crop],
            "score_mask": torch.ones(crop, dtype=torch.bool),
            "style": track.style[style_idx],
            "track_idx": track.track_idx,
            "track_name": track.track_name,
            "start_frame": start,
            "start_time_sec": start / track.fps,
            "duration_sec": crop / track.fps,
        }


class ValCropDataset(Dataset):
    """
    Deterministic crops taken from the held-out windows.

    Every item is fixed across runs, so val loss is comparable between
    checkpoints and between configs. All crops share one length -- the shortest
    held-out window in the corpus, capped at crop_frames -- so batches stack.

    Args:
      tracks (list[TrackTokens]): loaded corpus.
      cfg (DataCfg): crop settings.
    """

    def __init__(self, tracks: list[TrackTokens], cfg: DataCfg) -> None:
        spans = [(t, lo, hi) for t in tracks for lo, hi in t.val_windows]
        if not spans:
            raise ValueError("no validation windows; check val_frac")
        self._length = cfg.crop_frames
        # Each item is a full-length crop ENDING at the held-out window, so the
        # model gets the same 23.8 s of context it trains with; only the held-out
        # frames are scored. Without this, val CE would sit above train CE purely
        # because the window is shorter than a training crop, and the gap would
        # not mean overfitting.
        self._items = [
            (t, max(0, hi - self._length), lo, hi)
            for t, lo, hi in spans
            if t.num_frames >= self._length
        ]

    @property
    def crop_frames(self) -> int:
        """Common crop length shared by every validation item."""
        return self._length

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        track, start, lo, hi = self._items[index]
        end = start + self._length
        score = torch.zeros(self._length, dtype=torch.bool)
        score[max(0, lo - start) : max(0, hi - start)] = True
        style_idx = _pick_style(track, lo, hi, random.Random(index))
        return {
            "tokens": track.tokens[start:end],
            "score_mask": score,
            "style": track.style[style_idx],
            "track_idx": track.track_idx,
            "track_name": track.track_name,
            "start_frame": start,
            "start_time_sec": start / track.fps,
            "duration_sec": self._length / track.fps,
        }


# ===========================================================================
# Delay pattern
# ===========================================================================


def build_delay_grid(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Offset depth d by d frames, MusicGen-style (PLAN section 2).

    Depth 1 is literally the residual of depth 0, so predicting them in parallel
    assumes an independence that does not hold. Delaying each depth by its index
    lets depth d at position t attend to depth d-1 of the same frame at position
    t-1, recovering most of the flattened factorization at 1x sequence cost.

    Args:
      tokens (torch.Tensor): (B, T, R) int64 code indices.
      pad_id (int): id used where the delay runs off either end.

    Returns:
      torch.Tensor: (B, T + R - 1, R) int64 delayed grid.
    """
    batch, frames, depth = tokens.shape
    length = frames + depth - 1
    offsets = torch.arange(length, device=tokens.device)[:, None] - torch.arange(
        depth, device=tokens.device
    )[None, :]
    valid = (offsets >= 0) & (offsets < frames)
    gathered = tokens.gather(
        1, offsets.clamp(0, frames - 1).expand(batch, length, depth)
    )
    return torch.where(valid.expand(batch, length, depth), gathered, pad_id)


def undelay_grid(grid: torch.Tensor, frames: int) -> torch.Tensor:
    """
    Invert build_delay_grid, recovering aligned codes for decoding.

    Args:
      grid (torch.Tensor): (B, T + R - 1, R) delayed grid.
      frames (int): original frame count T.

    Returns:
      torch.Tensor: (B, T, R) int64 aligned codes.
    """
    depth = grid.shape[-1]
    offsets = torch.arange(frames, device=grid.device)[:, None] + torch.arange(
        depth, device=grid.device
    )[None, :]
    return grid.gather(1, offsets.expand(grid.shape[0], frames, depth))


# ===========================================================================
# Transformer
# ===========================================================================


class RMSNorm(nn.Module):
    """
    Root-mean-square layer norm.

    Args:
      dim (int): feature width.
      eps (float): numerical floor.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (..., dim) input.

        Returns:
          torch.Tensor: (..., dim) normalized and scaled output.
        """
        normed = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normed * self.weight.float()).type_as(x)


def rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute rotary position embedding factors.

    Args:
      seq_len (int): maximum positions to cover.
      head_dim (int): per-head width, must be even.
      theta (float): frequency base.
      device (torch.device): target device.
      dtype (torch.dtype): target dtype.

    Returns:
      tuple[torch.Tensor, torch.Tensor]: cos and sin, each (seq_len, head_dim // 2).
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    angles = torch.outer(torch.arange(seq_len, device=device).float(), freqs)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotate query/key pairs by their absolute position.

    Args:
      x (torch.Tensor): (B, H, L, Dh) queries or keys.
      cos (torch.Tensor): (L, Dh // 2) cosine factors.
      sin (torch.Tensor): (L, Dh // 2) sine factors.

    Returns:
      torch.Tensor: (B, H, L, Dh) rotated tensor.
    """
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack(
        [even * cos - odd * sin, even * sin + odd * cos], dim=-1
    ).flatten(-2)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal attention over SDPA.

    F.scaled_dot_product_attention dispatches to the FlashAttention-2 kernel for
    bf16 causal inputs, so no third-party kernel is needed; at these sequence
    lengths attention is a small share of the step anyway.

    Args:
      d_model (int): model width.
      n_heads (int): head count.
      dropout (float): attention dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        if (d_model // n_heads) % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, L, d_model) input.
          cos (torch.Tensor): (L, head_dim // 2) rotary cosines.
          sin (torch.Tensor): (L, head_dim // 2) rotary sines.

        Returns:
          torch.Tensor: (B, L, d_model) attended output.
        """
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(batch, length, -1))


class SwiGLU(nn.Module):
    """
    Gated feed-forward block.

    Args:
      d_model (int): model width.
      hidden (int): inner width.
    """

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (..., d_model) input.

        Returns:
          torch.Tensor: (..., d_model) output.
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """
    Pre-norm decoder block.

    Args:
      d_model (int): model width.
      n_heads (int): head count.
      dropout (float): dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm_mlp = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, int(8 * d_model / 3) // 64 * 64)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, L, d_model) input.
          cos (torch.Tensor): rotary cosines.
          sin (torch.Tensor): rotary sines.

        Returns:
          torch.Tensor: (B, L, d_model) output.
        """
        x = x + self.drop(self.attn(self.norm_attn(x), cos, sin))
        return x + self.drop(self.mlp(self.norm_mlp(x)))


PREFIX_POSITIONS = 2  # one conditioning token per stream: track id, style


class ArTransformer(nn.Module):
    """
    Decoder-only transformer over delayed RVQ tokens with two CFG streams.

    Conditioning is attached as projected prefix tokens (PLAN section 10.2), one
    per stream, each with its own learned null. Keeping the streams separate is
    what allows independent guidance weights at inference; a single shared null
    forecloses that and cannot be retrofitted without retraining.

    Args:
      cfg (ModelCfg): trunk and conditioning geometry.
      num_tokens (int): codebook entries per level.
      num_rq (int): RVQ depth.
      num_tracks (int): corpus size, for the track-id embedding.
      style_dim (int): width of the style descriptor.
      frames_per_pos (int): frames folded into one transformer position.
      max_positions (int): longest sequence the rotary cache must cover.
    """

    def __init__(
        self,
        cfg: ModelCfg,
        num_tokens: int,
        num_rq: int,
        num_tracks: int,
        style_dim: int,
        frames_per_pos: int = 1,
        max_positions: int = 8192,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_tokens = num_tokens
        self.num_rq = num_rq
        self.num_tracks = num_tracks
        self.frames_per_pos = frames_per_pos
        self.pad_id = num_tokens  # shared PAD/BOS, only ever an input
        self.max_positions = max_positions

        dim = cfg.d_model
        self.token_emb = nn.ModuleList(
            [nn.Embedding(num_tokens + 1, dim) for _ in range(num_rq)]
        )
        # Folding F frames into one position concatenates their embeddings; at
        # F == 1 this is skipped entirely so the default path stays plain.
        self.fold = (
            nn.Identity()
            if frames_per_pos == 1
            else nn.Linear(frames_per_pos * dim, dim, bias=False)
        )
        self.track_emb = nn.Embedding(num_tracks + 1, dim)  # last index is the null
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, cfg.style_bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.style_bottleneck, dim, bias=False),
        )
        self.style_null = nn.Parameter(torch.zeros(dim))

        self.blocks = nn.ModuleList(
            [Block(dim, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm_out = RMSNorm(dim)
        self.heads = nn.ModuleList(
            [
                nn.Linear(dim, frames_per_pos * num_tokens, bias=False)
                for _ in range(num_rq)
            ]
        )

        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers)
            )
            nn.init.normal_(
                block.mlp.down.weight, std=0.02 / math.sqrt(2 * cfg.n_layers)
            )
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """
        Apply GPT-style initialization.

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
        Fetch rotary factors, rebuilding the cache when it is too small.

        Args:
          length (int): sequence length needed.
          device (torch.device): target device.
          dtype (torch.dtype): target dtype.

        Returns:
          tuple[torch.Tensor, torch.Tensor]: cos and sin sliced to `length`.
        """
        head_dim = self.cfg.d_model // self.cfg.n_heads
        if (
            self._cos is None
            or self._cos.shape[0] < length
            or self._cos.device != device
            or self._cos.dtype != dtype
        ):
            self._cos, self._sin = rope_cache(
                max(length, self.max_positions), head_dim, self.cfg.rope_theta, device, dtype
            )
        assert self._sin is not None
        return self._cos[:length], self._sin[:length]

    def conditioning(
        self,
        track_idx: torch.Tensor,
        style: torch.Tensor,
        drop_id: torch.Tensor,
        drop_style: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the conditioning prefix, replacing dropped streams with their nulls.

        Args:
          track_idx (torch.Tensor): (B,) int64 track ids.
          style (torch.Tensor): (B, style_dim) float style descriptors.
          drop_id (torch.Tensor): (B,) bool, True where the id is nulled.
          drop_style (torch.Tensor): (B,) bool, True where the style is nulled.

        Returns:
          torch.Tensor: (B, 2, d_model) prefix tokens.
        """
        ids = torch.where(drop_id, torch.full_like(track_idx, self.num_tracks), track_idx)
        id_vec = self.track_emb(ids)
        style_vec = torch.where(
            drop_style[:, None], self.style_null.to(style.dtype)[None, :], self.style_proj(style)
        )
        return torch.stack([id_vec, style_vec], dim=1)

    def forward(
        self,
        tokens_in: torch.Tensor,
        track_idx: torch.Tensor,
        style: torch.Tensor,
        drop_id: torch.Tensor,
        drop_style: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score every position of a delayed, shifted token grid.

        Args:
          tokens_in (torch.Tensor): (B, L, R) int64 input grid, already delayed
            and right-shifted by one position.
          track_idx (torch.Tensor): (B,) int64 track ids.
          style (torch.Tensor): (B, style_dim) style descriptors.
          drop_id (torch.Tensor): (B,) bool conditioning-dropout mask for the id.
          drop_style (torch.Tensor): (B,) bool conditioning-dropout mask for style.

        Returns:
          torch.Tensor: (B, L, R, num_tokens) logits aligned with the input grid.
        """
        batch, length, depth = tokens_in.shape
        fold = self.frames_per_pos
        if length % fold:
            raise ValueError(f"sequence {length} not divisible by frames_per_pos {fold}")

        embedded = sum(self.token_emb[d](tokens_in[:, :, d]) for d in range(depth))
        assert isinstance(embedded, torch.Tensor)
        if fold > 1:
            embedded = self.fold(embedded.reshape(batch, length // fold, fold * self.cfg.d_model))

        prefix = self.conditioning(track_idx, style, drop_id, drop_style)
        hidden = torch.cat([prefix.to(embedded.dtype), embedded], dim=1)

        cos, sin = self._rope(hidden.shape[1], hidden.device, hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, cos, sin)
        hidden = self.norm_out(hidden)[:, PREFIX_POSITIONS:]

        logits = torch.stack([head(hidden) for head in self.heads], dim=2)
        if fold > 1:
            # (B, L/F, R, F*V) -> (B, L/F, F, R, V) so frames stay time-ordered
            logits = logits.view(batch, length // fold, depth, fold, self.num_tokens)
            logits = logits.permute(0, 1, 3, 2, 4)
        return logits.reshape(batch, length, depth, self.num_tokens)

    @torch.no_grad()
    def cfg_logits(
        self,
        tokens_in: torch.Tensor,
        track_a: int,
        track_b: int | None,
        style_a: torch.Tensor,
        style_b: torch.Tensor | None,
        strength: float = 3.0,
        blend: float = 1.0,
    ) -> torch.Tensor:
        """
        Combine conditional and unconditional logits for guided sampling.

        Uses the section 11.5 parameterization, which separates the two knobs
        that a raw (w1, w2) pair entangles:

            l_final = l(null) + strength * (blend * dA + (1 - blend) * dB)

        strength is guidance magnitude, set once per model (section 11.4 puts the
        useful audio range near 1-3, far below image-diffusion values); blend is
        the A/B mix and is the knob worth exposing in a UI. Mixing happens in
        output space, after the model, so every forward pass uses a conditioning
        value that actually occurred in training -- unlike averaging two
        embeddings, which invents an unseen input (section 10.1).

        Args:
          tokens_in (torch.Tensor): (1, L, R) input grid for one sequence.
          track_a (int): first track id.
          track_b (int | None): second track id, or None for single-condition CFG.
          style_a (torch.Tensor): (style_dim,) style descriptor for A.
          style_b (torch.Tensor | None): style descriptor for B, or None.
          strength (float): guidance strength.
          blend (float): 1.0 is pure A, 0.0 pure B, 0.5 an even mix.

        Returns:
          torch.Tensor: (1, L, R, num_tokens) guided logits.
        """
        device = tokens_in.device
        two_sided = track_b is not None and style_b is not None
        count = 3 if two_sided else 2

        ids = torch.tensor(
            [self.num_tracks, track_a] + ([track_b] if two_sided else []),
            device=device,
            dtype=torch.long,
        )
        styles = torch.stack(
            [torch.zeros_like(style_a), style_a] + ([style_b] if two_sided else [])  # type: ignore[list-item]
        ).to(device)
        drop_id = torch.tensor([True, False] + ([False] if two_sided else []), device=device)
        drop_style = drop_id.clone()

        logits = self.forward(
            tokens_in.expand(count, -1, -1), ids, styles, drop_id, drop_style
        )
        null = logits[0:1]
        delta_a = logits[1:2] - null
        if not two_sided:
            return null + strength * delta_a
        delta_b = logits[2:3] - null
        return null + strength * (blend * delta_a + (1.0 - blend) * delta_b)


# ===========================================================================
# Lightning module
# ===========================================================================


class ArLightningModule(L.LightningModule):
    """
    Next-token training over the delayed grid, with per-stream CFG dropout.

    Targets and loss are ordinary cross-entropy (PLAN section 11.2): CFG changes
    only what conditioning the model sees, never what it is asked to predict.

    Args:
      model (ArTransformer): the network.
      cfg (ArConfig): full config.
      shuffle_cond (bool): shuffle conditioning across the batch, the section
        11.3 leakage diagnostic.
    """

    def __init__(
        self, model: ArTransformer, cfg: ArConfig, shuffle_cond: bool = False
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.shuffle_cond = shuffle_cond
        self.save_hyperparameters({"config": asdict(cfg), "shuffle_cond": shuffle_cond})
        self._step_start = time.monotonic()

    def _maybe_log(self, name: str, value: Any, **kwargs: Any) -> None:
        """
        Log only when attached to a trainer, so --bench-attn can call _run alone.

        Args:
          name (str): metric name.
          value (Any): metric value.
          **kwargs (Any): forwarded to LightningModule.log.
        """
        if self._trainer is not None:
            self.log(name, value, **kwargs)

    def _prepare(
        self, tokens: torch.Tensor, score: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Delay, right-shift and mask one batch of aligned codes.

        Args:
          tokens (torch.Tensor): (B, T, R) int64 aligned codes.
          score (torch.Tensor | None): (B, T) bool, which frames contribute to
            the loss. None scores every frame. Validation uses it to score only
            held-out frames while still conditioning on preceding context.

        Returns:
          tuple[torch.Tensor, torch.Tensor, torch.Tensor]: input grid (B, L, R),
            target grid (B, L, R) and a bool loss mask (B, L, R). Positions where
            the delay ran off either end are PAD and are masked out.
        """
        pad = self.model.pad_id
        grid = build_delay_grid(tokens, pad)
        bos = torch.full(
            (grid.shape[0], 1, grid.shape[2]), pad, dtype=grid.dtype, device=grid.device
        )
        inputs = torch.cat([bos, grid[:, :-1]], dim=1)

        fold = self.model.frames_per_pos
        if fold > 1 and grid.shape[1] % fold:
            extra = fold - grid.shape[1] % fold
            filler = torch.full(
                (grid.shape[0], extra, grid.shape[2]),
                pad,
                dtype=grid.dtype,
                device=grid.device,
            )
            inputs = torch.cat([inputs, filler], dim=1)
            grid = torch.cat([grid, filler], dim=1)

        mask = grid != pad
        if score is not None:
            # Delay the frame mask the same way the codes were delayed, so a
            # scored frame lines up with the position that predicts it.
            delayed = build_delay_grid(
                score.long().unsqueeze(-1).expand(-1, -1, grid.shape[2]), 0
            )
            if delayed.shape[1] < grid.shape[1]:
                pad_len = grid.shape[1] - delayed.shape[1]
                delayed = F.pad(delayed, (0, 0, 0, pad_len))
            mask = mask & delayed.bool()
        return inputs, grid, mask

    def _run(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        """
        Shared train/val step.

        Args:
          batch (dict[str, Any]): collated batch from TokenCropDataset.
          stage (str): "train" or "val", used for metric names and dropout.

        Returns:
          torch.Tensor: scalar mean cross-entropy over supervised positions.
        """
        tokens = batch["tokens"].long()
        track_idx = batch["track_idx"].long()
        style = batch["style"].float()
        batch_size = tokens.shape[0]

        if self.shuffle_cond:
            perm = torch.randperm(batch_size, device=tokens.device)
            track_idx, style = track_idx[perm], style[perm]

        # Independent coins per stream (section 11.2): a shared coin would make
        # separate guidance weights impossible at inference.
        prob = self.cfg.model.p_drop_cond if stage == "train" else 0.0
        drop_id = torch.rand(batch_size, device=tokens.device) < prob
        drop_style = torch.rand(batch_size, device=tokens.device) < prob

        inputs, targets, mask = self._prepare(tokens, batch.get("score_mask"))
        logits = self.model(inputs, track_idx, style, drop_id, drop_style)

        total = torch.zeros((), device=tokens.device)
        supervised = 0
        for depth in range(self.model.num_rq):
            keep = mask[:, :, depth]
            picked = logits[:, :, depth][keep]
            wanted = targets[:, :, depth][keep]
            depth_loss = F.cross_entropy(picked.float(), wanted)
            total = total + depth_loss * keep.sum()
            supervised += int(keep.sum())
            self._maybe_log(f"{stage}/ce_depth{depth}", depth_loss, prog_bar=False)
            self._maybe_log(
                f"{stage}/acc_depth{depth}",
                (picked.argmax(-1) == wanted).float().mean(),
                prog_bar=False,
            )
        loss = total / max(supervised, 1)
        self._maybe_log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): collated training batch.
          batch_idx (int): index within the epoch.

        Returns:
          torch.Tensor: training loss.
        """
        loss = self._run(batch, "train")
        elapsed = time.monotonic() - self._step_start
        self._step_start = time.monotonic()
        if elapsed > 0:
            supervised = batch["tokens"].numel()
            self.log("train/tokens_per_sec", supervised / elapsed, prog_bar=False)
        self.log(
            "train/lr", self.optimizers().param_groups[0]["lr"], prog_bar=False  # type: ignore[union-attr,index]
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Args:
          batch (dict[str, Any]): collated validation batch.
          batch_idx (int): index within the epoch.

        Returns:
          torch.Tensor: validation loss.
        """
        return self._run(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        """
        AdamW with decay only on matmul weights, plus the repo's wall-clock cycle.

        Returns:
          dict[str, Any]: Lightning optimizer/scheduler bundle.
        """
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
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


# ===========================================================================
# Entry points
# ===========================================================================


def build_model(cfg: ArConfig, tracks: list[TrackTokens], manifest: dict[str, Any]) -> ArTransformer:
    """
    Instantiate the transformer against the cached corpus's geometry.

    num_tracks comes from the manifest, not the loaded subset, so track ids stay
    valid when data.single_track filters the corpus down to one track.

    Args:
      cfg (ArConfig): full config.
      tracks (list[TrackTokens]): loaded corpus, for the style width.
      manifest (dict[str, Any]): cache manifest, for tokenizer geometry.

    Returns:
      ArTransformer: the model.
    """
    meta = manifest["tokenizer_meta"]
    positions = cfg.data.crop_frames + meta["num_rq"] - 1 + PREFIX_POSITIONS
    return ArTransformer(
        cfg=cfg.model,
        num_tokens=meta["num_tokens"],
        num_rq=meta["num_rq"],
        num_tracks=len(manifest["tracks"]),
        style_dim=int(tracks[0].style.shape[-1]),
        frames_per_pos=cfg.data.frames_per_pos,
        max_positions=positions,
    )


def bench_attention(module: ArLightningModule, batch: dict[str, Any], steps: int = 8) -> None:
    """
    Time forward+backward and report which SDPA backend actually runs.

    This is what decides the flash-attn question empirically. If SDPA already
    selects the FlashAttention kernel at this sequence length, adding a
    build-from-source dependency buys nothing.

    Args:
      module (ArLightningModule): model wrapper, already on the target device.
      batch (dict[str, Any]): one batch, already on the target device.
      steps (int): timed iterations after warmup.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    device = batch["tokens"].device
    for name, backend in [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]:
        try:
            with torch.no_grad(), sdpa_kernel([backend]):
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    module._run(batch, "train")
            print(f"  SDPA {name:14s} available")
        except Exception as exc:
            print(f"  SDPA {name:14s} unavailable ({type(exc).__name__})")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(module.model.parameters(), lr=1e-9)
    if device.type == "cuda":
        # Probes (notably the math fallback, which materializes the full
        # attention matrix) must not pollute the training-step VRAM figure.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for index in range(steps + 2):
        if index == 2:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = module._run(batch, "train")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    tokens = batch["tokens"].numel() * steps
    params = sum(p.numel() for p in module.model.parameters())
    print(f"\n  params           {params / 1e6:.1f} M")
    print(f"  step time        {elapsed / steps * 1000:.1f} ms")
    print(f"  tokens/s         {tokens / elapsed:,.0f}")
    if device.type == "cuda":
        print(f"  peak VRAM        {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def build_trainer(cfg: ArConfig) -> L.Trainer:
    """
    Assemble a Lightning trainer matching the repo's run conventions.

    Args:
      cfg (ArConfig): full config.

    Returns:
      L.Trainer: configured trainer.
    """
    save_path = REPO / cfg.train.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.Timer(duration={"minutes": cfg.train.minutes}),
        # Monitored "best". Owns no last.ckpt: Lightning gates save_last on a
        # top-k save actually firing, and val/loss rises monotonically here, so
        # save_last on this callback would freeze at the first epoch forever.
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="ar_best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        # Unmonitored rolling save. Fires every epoch regardless of any metric,
        # which is what keeps last.ckpt current and survives a crash.
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(save_path),
            filename="ar_latest",
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
        gradient_clip_val=cfg.train.grad_clip,
        callbacks=callbacks,
        logger=L.pytorch.loggers.TensorBoardLogger(str(save_path), name="ar"),
        log_every_n_steps=10,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse the CLI.

    Returns:
      argparse.Namespace: parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "config_ar.yaml"))
    parser.add_argument(
        "--build-cache", action="store_true", help="tokenize the corpus and exit"
    )
    parser.add_argument(
        "--force-rebuild", action="store_true", help="rebuild the cache even if present"
    )
    parser.add_argument(
        "--bench-attn", action="store_true", help="time one batch and report the SDPA backend"
    )
    parser.add_argument(
        "--shuffle-cond",
        action="store_true",
        help="shuffle conditioning across the batch (section 11.3 leakage test)",
    )
    return parser.parse_args()


def main() -> None:
    """Build the token cache if needed, then benchmark or train."""
    args = parse_args()
    cfg = load_config(args.config)

    cache_dir = build_token_cache(cfg, force=args.force_rebuild)
    if args.build_cache:
        return

    L.seed_everything(cfg.train.seed, workers=True)
    tracks, manifest = load_token_cache(cache_dir, cfg.data)
    frames = sum(t.num_frames for t in tracks)
    held = sum(hi - lo for t in tracks for lo, hi in t.val_windows)
    print(
        f"{len(tracks)} tracks, {frames} frames "
        f"({frames / manifest['tokenizer_meta']['frames_per_second'] / 3600:.2f} h), "
        f"val {100 * held / frames:.2f}%"
    )

    train_set = TokenCropDataset(
        tracks, cfg.data, cfg.data.steps_per_epoch * cfg.train.batch_size
    )
    val_set = ValCropDataset(tracks, cfg.data)
    print(f"val: {len(val_set)} windows at {val_set.crop_frames} frames")

    loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.train.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=False, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_model(cfg, tracks, manifest)
    module = ArLightningModule(model, cfg, shuffle_cond=args.shuffle_cond)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params")

    if args.bench_attn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module.to(device)
        batch = next(iter(DataLoader(train_set, batch_size=cfg.train.batch_size)))
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        bench_attention(module, batch)
        return

    if cfg.train.checkpoint:
        state = torch.load(REPO / cfg.train.checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = module.load_state_dict(state["state_dict"], strict=False)
        print(f"warm start: {len(missing)} missing, {len(unexpected)} unexpected keys")

    trainer = build_trainer(cfg)
    trainer.fit(module, train_loader, val_loader)
    # Timer stops mid-epoch, so the epoch-end callback misses the final partial
    # epoch. Save explicitly.
    trainer.save_checkpoint(REPO / cfg.train.save_path / "ar_final.ckpt")
    print(f"final checkpoint -> {REPO / cfg.train.save_path / 'ar_final.ckpt'}")


if __name__ == "__main__":
    main()
