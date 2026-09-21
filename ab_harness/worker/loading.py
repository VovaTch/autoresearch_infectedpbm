"""
Checkpoint loading, shared by every front-end that samples tokens.

The rating harness and the slice synthesizer need the same four things out of a
checkpoint -- the model wrapped in a sampler, the corpus it was trained against,
the tokenizer meta and the sliding-window geometry -- and neither should be the
place the other reads it from. Keeping it here is what stops the two apps
drifting into loading the same file differently, which would make a clip
generated in one of them unreproducible in the other.

Three backends load here, told apart by the checkpoint's directory family
(ab_harness.checkpoints.backend_of): AR/DPO token models behind ArGenerator,
latent-flow DiTs (train_zflow.py) behind ZFlowGenerator and masked diffusion
models (train_mdm.py) behind MdmGenerator. All three satisfy SampleSource, so
the services never branch on which one is running.

The corpus and the token cache belong to the tokenizer, not to the AR model, so
the cache is only re-read when the new checkpoint points at a different one.
That is what makes switching models mid-session cost a few seconds rather than a
minute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ab_harness.checkpoints import backend_of
from ab_harness.config import REPO, GeneratorCfg
from ab_harness.model.protocols import SampleSource
from ab_harness.worker.generator import ArGenerator
from generate_ar import config_from_ckpt
from train_ar import (
    ArConfig,
    DataCfg,
    TrackTokens,
    build_model,
    load_token_cache,
    token_cache_dir,
)


@dataclass
class LoadedModel:
    """
    Everything a sampling front-end needs from one checkpoint.

    Args:
      checkpoint (str): repo-relative path this was loaded from.
      generator (SampleSource): the batched sampler of whichever backend.
      tracks (list[TrackTokens]): the corpus in manifest order.
      manifest (dict[str, Any]): the token cache's manifest.
      meta (dict[str, Any]): manifest["tokenizer_meta"], the frame geometry.
      cache_dir (Path): the token cache the corpus came from.
      ar_cfg (ArConfig | None): the AR config the checkpoint was trained with
        (the AR backend), or the one its crops came from (zflow/mdm in
        generator mode); None for a legacy flow checkpoint.
      device (torch.device): where the model lives.
      chunk_frames (int): encoder window for priming from an audio file.
      margin (int): encoder context discarded per side.
    """

    checkpoint: str
    generator: SampleSource
    tracks: list[TrackTokens]
    manifest: dict[str, Any]
    meta: dict[str, Any]
    cache_dir: Path
    ar_cfg: ArConfig | None
    device: torch.device
    chunk_frames: int = 4096
    margin: int = 256

    @property
    def backend(self) -> str:
        """
        Returns:
          str: "ar", "zflow" or "mdm", from the checkpoint path.
        """
        return backend_of(self.checkpoint)

    @property
    def by_idx(self) -> dict[int, TrackTokens]:
        """
        Returns:
          dict[int, TrackTokens]: the corpus keyed by track id. The ids are the
            rows of the model's track embedding, so they are not interchangeable
            with positions in `tracks` once a track is filtered out.
        """
        return {t.track_idx: t for t in self.tracks}

    @property
    def fps(self) -> float:
        """
        Returns:
          float: tokenizer frames per second.
        """
        return float(self.meta["frames_per_second"])

    @property
    def hop(self) -> int:
        """
        Returns:
          int: samples per frame.
        """
        return int(self.meta["hop_length"])

    @property
    def sample_rate(self) -> int:
        """
        Returns:
          int: decoder output rate.
        """
        return int(self.meta["sample_rate"])


def load_checkpoint(
    checkpoint: str, gen_cfg: GeneratorCfg, previous: LoadedModel | None = None
) -> LoadedModel:
    """
    Load a checkpoint of any backend and build its sampler.

    Args:
      checkpoint (str): repo-relative checkpoint path; its directory family
        picks the backend.
      gen_cfg (GeneratorCfg): device and sampling settings.
      previous (LoadedModel | None): the model being replaced, if any, whose
        corpus is reused when the token cache is the same.

    Returns:
      LoadedModel: the sampler and the corpus it draws from.
    """
    backend = backend_of(checkpoint)
    if backend == "zflow":
        return load_zflow_checkpoint(checkpoint, gen_cfg, previous)
    if backend == "mdm":
        return load_mdm_checkpoint(checkpoint, gen_cfg, previous)
    return load_ar_checkpoint(checkpoint, gen_cfg, previous)


def _device(gen_cfg: GeneratorCfg) -> torch.device:
    """
    Args:
      gen_cfg (GeneratorCfg): device setting.

    Returns:
      torch.device: the configured device, or the CPU without CUDA.
    """
    return torch.device(gen_cfg.device if torch.cuda.is_available() else "cpu")


def _corpus(
    cache_dir: Path, previous: LoadedModel | None
) -> tuple[list[TrackTokens], dict[str, Any]]:
    """
    Args:
      cache_dir (Path): token cache to read.
      previous (LoadedModel | None): reused when it came from the same cache.

    Returns:
      tuple[list[TrackTokens], dict[str, Any]]: the full corpus and manifest.
    """
    if previous is not None and previous.cache_dir == cache_dir:
        return previous.tracks, previous.manifest
    if not cache_dir.exists():
        raise FileNotFoundError(f"no token cache at {cache_dir}")
    return load_token_cache(cache_dir, DataCfg())


def _ckpt_path(checkpoint: str) -> Path:
    """
    Args:
      checkpoint (str): repo-relative checkpoint path.

    Returns:
      Path: the absolute path, verified to exist.
    """
    ckpt_path = REPO / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    return ckpt_path


def load_zflow_checkpoint(
    checkpoint: str, gen_cfg: GeneratorCfg, previous: LoadedModel | None = None
) -> LoadedModel:
    """
    Load a train_zflow checkpoint behind ZFlowGenerator.

    Args:
      checkpoint (str): repo-relative checkpoint path.
      gen_cfg (GeneratorCfg): device and flow sampling settings.
      previous (LoadedModel | None): the model being replaced, if any.

    Returns:
      LoadedModel: the sampler and the corpus it draws from.
    """
    from ab_harness.worker.zflow_gen import ZFlowGenerator
    from train_zflow import ar_config, cache_dir as zflow_cache_dir, load_zflow_module

    device = _device(gen_cfg)
    module = load_zflow_module(_ckpt_path(checkpoint), ema=True, device=str(device))
    cfg = module.cfg
    ar_cfg = ar_config(cfg)
    cache = zflow_cache_dir(cfg)
    tracks, manifest = _corpus(cache, previous)
    window = ar_cfg.data.crop_frames if ar_cfg is not None else cfg.data.crop_frames
    if ar_cfg is not None:
        chunk, margin = ar_cfg.tokenizer.chunk_frames, ar_cfg.tokenizer.margin
    else:
        chunk, margin = cfg.tokenizer.chunk_frames, cfg.tokenizer.margin
    return LoadedModel(
        checkpoint=checkpoint,
        generator=ZFlowGenerator(
            module,
            device,
            min(gen_cfg.window_frames, window),
            len(manifest["tracks"]),
            gen_cfg,
        ),
        tracks=tracks,
        manifest=manifest,
        meta=manifest["tokenizer_meta"],
        cache_dir=cache,
        ar_cfg=ar_cfg,
        device=device,
        chunk_frames=chunk,
        margin=margin,
    )


def load_mdm_checkpoint(
    checkpoint: str, gen_cfg: GeneratorCfg, previous: LoadedModel | None = None
) -> LoadedModel:
    """
    Load a train_mdm checkpoint behind MdmGenerator.

    Args:
      checkpoint (str): repo-relative checkpoint path.
      gen_cfg (GeneratorCfg): device and MaskGIT settings.
      previous (LoadedModel | None): the model being replaced, if any.

    Returns:
      LoadedModel: the sampler and the corpus it draws from.
    """
    from ab_harness.worker.mdm_gen import MdmGenerator
    from train_mdm import ar_config, load_mdm_module

    device = _device(gen_cfg)
    module = load_mdm_module(_ckpt_path(checkpoint), ema=True, device=str(device))
    ar_cfg = ar_config(module.cfg)
    cache = token_cache_dir(ar_cfg)
    tracks, manifest = _corpus(cache, previous)
    window = min(gen_cfg.window_frames, ar_cfg.data.crop_frames)
    return LoadedModel(
        checkpoint=checkpoint,
        generator=MdmGenerator(
            module.net,
            module.cfg.mask,
            device,
            window,
            module.cfg.model.prefix_max_frac,
            gen_cfg,
        ),
        tracks=tracks,
        manifest=manifest,
        meta=manifest["tokenizer_meta"],
        cache_dir=cache,
        ar_cfg=ar_cfg,
        device=device,
        chunk_frames=ar_cfg.tokenizer.chunk_frames,
        margin=ar_cfg.tokenizer.margin,
    )


def load_ar_checkpoint(
    checkpoint: str, gen_cfg: GeneratorCfg, previous: LoadedModel | None = None
) -> LoadedModel:
    """
    Load one AR/DPO checkpoint and build the sampler around it.

    Args:
      checkpoint (str): repo-relative checkpoint path.
      gen_cfg (GeneratorCfg): device and sliding-window settings.
      previous (LoadedModel | None): the model being replaced, if any. Its corpus
        is reused when the new checkpoint points at the same token cache.

    Returns:
      LoadedModel: the sampler and the corpus it draws from.

    Raises:
      FileNotFoundError: when the checkpoint or its token cache is missing.
    """
    ckpt = torch.load(_ckpt_path(checkpoint), map_location="cpu", weights_only=False)
    ar_cfg = config_from_ckpt(ckpt)

    # The cache is content-addressed on the encoder graph and the style geometry
    # (train_ar.token_cache_dir), the same rule train_ar.main builds it by. The
    # newest directory is NOT that cache once several tokenizers have been run.
    cache_dir = token_cache_dir(ar_cfg)
    if not cache_dir.exists():
        cache_root = Path(ar_cfg.tokenizer.cache_root).expanduser()
        caches = sorted(cache_root.glob("tokens_*"))
        if not caches:
            raise FileNotFoundError(f"no token cache under {cache_root}")
        print(
            f"  WARN token cache {cache_dir.name} for {checkpoint} is missing; "
            f"falling back to {caches[-1].name}"
        )
        cache_dir = caches[-1]

    if previous is not None and previous.cache_dir == cache_dir:
        tracks, manifest = previous.tracks, previous.manifest
    else:
        # single_track would hide most of the corpus from the sampler
        data_cfg = DataCfg(**{**vars(ar_cfg.data), "single_track": None})
        tracks, manifest = load_token_cache(cache_dir, data_cfg)

    model = build_model(ar_cfg, tracks, manifest)
    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  WARN missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}")
    device = _device(gen_cfg)
    model.to(device).eval()

    return LoadedModel(
        checkpoint=checkpoint,
        generator=ArGenerator(
            model,
            device,
            window_frames=min(gen_cfg.window_frames, ar_cfg.data.crop_frames),
            reprime_frac=gen_cfg.reprime_frac,
        ),
        tracks=tracks,
        manifest=manifest,
        meta=manifest["tokenizer_meta"],
        cache_dir=cache_dir,
        ar_cfg=ar_cfg,
        device=device,
        chunk_frames=ar_cfg.tokenizer.chunk_frames,
        margin=ar_cfg.tokenizer.margin,
    )
