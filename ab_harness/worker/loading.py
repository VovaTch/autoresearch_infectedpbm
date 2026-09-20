"""
Checkpoint loading, shared by every front-end that samples from an AR model.

The rating harness and the slice synthesizer need the same four things out of a
checkpoint -- the model wrapped in a sampler, the corpus it was trained against,
the tokenizer meta and the sliding-window geometry -- and neither should be the
place the other reads it from. Keeping it here is what stops the two apps
drifting into loading the same file differently, which would make a clip
generated in one of them unreproducible in the other.

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

from ab_harness.config import REPO, GeneratorCfg
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
      generator (ArGenerator): the batched, KV-cached sampler.
      tracks (list[TrackTokens]): the corpus in manifest order.
      manifest (dict[str, Any]): the token cache's manifest.
      meta (dict[str, Any]): manifest["tokenizer_meta"], the frame geometry.
      cache_dir (Path): the token cache the corpus came from.
      ar_cfg (ArConfig): the config the checkpoint was trained with.
      device (torch.device): where the model lives.
    """

    checkpoint: str
    generator: ArGenerator
    tracks: list[TrackTokens]
    manifest: dict[str, Any]
    meta: dict[str, Any]
    cache_dir: Path
    ar_cfg: ArConfig
    device: torch.device

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
    ckpt_path = REPO / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    device = torch.device(gen_cfg.device if torch.cuda.is_available() else "cpu")
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
    )
