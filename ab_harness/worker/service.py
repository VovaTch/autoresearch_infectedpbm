"""
The generation service: the only place that owns a GPU.

One implementation with two front-ends. bake_ab_bank.py runs it in-process to
pre-fill the bank; the app runs it in a child process and streams clips from it
while the rater works. Having a single code path is what keeps what the app
produces identical to what the CLI produces.

Everything it emits is persisted as tokens before the audio is handed back, so a
crash costs nothing that was already rated and any clip can be re-decoded later
for reward-model training.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ab_harness.config import REPO, AbConfig
from ab_harness.model.audio import fill_fraction, normalize_lufs, to_int16
from ab_harness.model.bank import ClipBank
from ab_harness.model.pair_sampler import TrackInfo, save_corpus
from ab_harness.model.types import ClipSpec
from ab_harness.worker.decoder import TokenDecoder
from ab_harness.worker.generator import ArGenerator, SampleRequest
from ab_harness.worker.protocol import (
    CheckpointChanged,
    ClipRequest,
    ClipResult,
    Shutdown,
    SwitchCheckpoint,
)
from generate_ar import config_from_ckpt
from train_ar import DataCfg, TrackTokens, build_model, load_token_cache


def style_vector(track: TrackTokens, spec: ClipSpec) -> torch.Tensor:
    """
    Rebuild the style descriptor a ClipSpec was (or would be) generated with.

    Module-level so anything scoring or regenerating a banked clip -- the
    preference trainer included -- can reproduce the conditioning without
    loading a GenerationService and its ONNX decoder.

    Args:
      track (TrackTokens): the cached track the clip was drawn from.
      spec (ClipSpec): the clip being generated.

    Returns:
      torch.Tensor: (style_dim,) descriptor, zeros when the stream is nulled
        (the model substitutes its learned null, so the value is inert).
    """
    window = spec.conditioning.style_window
    if not spec.conditioning.use_style or window < 0:
        return torch.zeros(track.style.shape[-1])
    return track.style[min(window, track.style.shape[0] - 1)]


class GenerationService:
    """
    Loads the checkpoint, corpus and decoder, and turns ClipSpecs into audio.

    Args:
      cfg (AbConfig): harness config.
    """

    def __init__(self, cfg: AbConfig) -> None:
        self.cfg = cfg
        self.bank = ClipBank(cfg.bank_root)
        self.checkpoint = cfg.generator.checkpoint
        self._tracks: dict[int, TrackTokens] = {}
        self._ordered_tracks: list[TrackTokens] = []
        self._manifest: dict[str, Any] = {}
        self._cache_dir: Path | None = None
        self._generator: ArGenerator | None = None
        self._decoder: TokenDecoder | None = None
        self._meta: dict[str, Any] = {}
        self._loaded = False

    # -- startup -------------------------------------------------------------

    def load(self) -> None:
        """
        Load everything heavy. Safe to call twice; the second call is a no-op.
        """
        if self._loaded:
            return
        gen_cfg = self.cfg.generator
        self._load_checkpoint(gen_cfg.checkpoint)
        self._decoder = TokenDecoder(
            REPO / gen_cfg.decoder_onnx,
            hop=int(self._meta["hop_length"]),
            sample_rate=int(self._meta["sample_rate"]),
            use_gpu=gen_cfg.use_gpu_decoder,
        )
        self._loaded = True

    def _load_checkpoint(self, checkpoint: str) -> None:
        """
        Load one AR/DPO checkpoint and build the generator around it.

        The corpus and the token cache belong to the tokenizer, not to the AR
        model, so they are only reloaded when the checkpoint points at a
        different cache -- which is what makes switching models mid-session cost
        a few seconds rather than a minute. The decoder is untouched for the
        same reason.

        Args:
          checkpoint (str): repo-relative checkpoint path.

        Raises:
          FileNotFoundError: when the checkpoint or its token cache is missing.
        """
        ckpt_path = REPO / checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ar_cfg = config_from_ckpt(ckpt)

        cache_root = Path(ar_cfg.tokenizer.cache_root).expanduser()
        caches = sorted(cache_root.glob("tokens_*"))
        if not caches:
            raise FileNotFoundError(f"no token cache under {cache_root}")
        cache_dir = caches[-1]
        if cache_dir != self._cache_dir:
            # single_track would hide most of the corpus from the sampler
            data_cfg = DataCfg(**{**vars(ar_cfg.data), "single_track": None})
            tracks, manifest = load_token_cache(cache_dir, data_cfg)
            self._ordered_tracks = tracks
            self._tracks = {t.track_idx: t for t in tracks}
            self._manifest = manifest
            self._meta = manifest["tokenizer_meta"]
            self._cache_dir = cache_dir
            # The UI process never imports torch, so the corpus it needs to draw
            # spans from has to reach it through the bank.
            save_corpus(self.corpus_path, self.tracks)
        self.bank.init_manifest(cache_dir.name, self._meta, checkpoint)

        model = build_model(ar_cfg, self._ordered_tracks, self._manifest)
        state = {
            k[len("model.") :]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"  WARN missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}"
            )
        device = torch.device(
            self.cfg.generator.device if torch.cuda.is_available() else "cpu"
        )
        model.to(device).eval()
        self._generator = ArGenerator(
            model,
            device,
            window_frames=min(
                self.cfg.generator.window_frames, ar_cfg.data.crop_frames
            ),
            reprime_frac=self.cfg.generator.reprime_frac,
        )
        self.checkpoint = checkpoint

    def switch_checkpoint(self, checkpoint: str) -> None:
        """
        Sample from a different model from here on.

        The old generator stays in place until the new one is built, so a bad
        path costs an error message rather than a dead session.

        Args:
          checkpoint (str): repo-relative checkpoint path.
        """
        if checkpoint == self.checkpoint and self._loaded:
            return
        self.load()
        self._load_checkpoint(checkpoint)

    # -- corpus --------------------------------------------------------------

    @property
    def corpus_path(self) -> Path:
        """
        Returns:
          Path: where the torch-free corpus view is published.
        """
        return self.cfg.bank_root / "corpus.json"

    @property
    def tracks(self) -> list[TrackInfo]:
        """
        Returns:
          list[TrackInfo]: the torch-free corpus view the sampler draws from.
        """
        return [
            TrackInfo(
                track_idx=t.track_idx,
                track_name=t.track_name,
                num_frames=t.num_frames,
                style_bounds=tuple(
                    (int(lo), int(hi)) for lo, hi in t.style_bounds.tolist()
                ),
            )
            for t in sorted(self._tracks.values(), key=lambda t: t.track_idx)
        ]

    @property
    def meta(self) -> dict[str, Any]:
        """
        Returns:
          dict[str, Any]: the tokenizer meta the token cache was built with.
        """
        self.load()
        return self._meta

    def corpus(self) -> list[TrackInfo]:
        """
        Returns:
          list[TrackInfo]: the corpus, loading the service first if needed.
        """
        self.load()
        return self.tracks

    # -- production ----------------------------------------------------------

    def _real_tokens(self, spec: ClipSpec) -> np.ndarray:
        """
        Slice a span out of the cached real tokens.

        Args:
          spec (ClipSpec): the span to cut.

        Returns:
          np.ndarray: (T, R) int16 codes, clamped to the track's length.
        """
        track = self._tracks[spec.conditioning.track_idx]
        start = max(0, min(spec.conditioning.start_frame, track.num_frames - 1))
        end = min(start + spec.n_frames, track.num_frames)
        return track.tokens[start:end].numpy().astype(np.int16)

    def _style_vector(self, spec: ClipSpec) -> torch.Tensor:
        """
        Args:
          spec (ClipSpec): the clip being generated.

        Returns:
          torch.Tensor: (style_dim,) descriptor, zeros when the stream is nulled.
        """
        return style_vector(self._tracks[spec.conditioning.track_idx], spec)

    def _request(self, spec: ClipSpec) -> SampleRequest:
        """
        Args:
          spec (ClipSpec): the recipe.

        Returns:
          SampleRequest: one lane of a sampling batch.
        """
        cond, sampling = spec.conditioning, spec.sampling
        prompt = None
        if cond.prompt_frames > 0:
            prompt = torch.from_numpy(
                self._real_tokens(spec)[: cond.prompt_frames]
            ).long()
        return SampleRequest(
            track_idx=cond.track_idx,
            style=self._style_vector(spec),
            use_track_id=cond.use_track_id,
            use_style=cond.use_style,
            frames=spec.n_frames,
            prompt=prompt,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            cfg_strength=cond.cfg_strength,
            seed=sampling.seed,
        )

    def _audio_result(
        self, spec: ClipSpec, tokens: np.ndarray, was_live: bool
    ) -> ClipResult:
        """
        Decode and loudness-match one clip's tokens.

        Args:
          spec (ClipSpec): the clip.
          tokens (np.ndarray): (T, R) codes.
          was_live (bool): whether the tokens were just sampled.

        Returns:
          ClipResult: the finished clip.
        """
        assert self._decoder is not None
        audio = self._decoder.decode(tokens)
        audio = normalize_lufs(
            audio, self._decoder.sample_rate, self.cfg.session.target_lufs
        )
        pcm = to_int16(audio)
        # Measured here, where the waveform already exists, and cached in the
        # bank so a clip that turned out to be dead air is never decoded again
        # just to rediscover that.
        fill = fill_fraction(pcm)
        self.bank.set_fill(spec.item_id, fill)
        return ClipResult(
            spec=spec,
            tokens=tokens,
            pcm=pcm,
            sample_rate=self._decoder.sample_rate,
            was_live=was_live,
            fill=fill,
        )

    def produce_many(self, specs: Sequence[ClipSpec]) -> list[ClipResult]:
        """
        Produce a batch of clips, sampling everything new in one pass.

        Batching is the whole point: the sampling loop is launch-bound, so eight
        clips cost what one does. Banked tokens are reused rather than
        regenerated, which is what makes a repeat pair byte-identical to its
        first showing and what stops a re-run of the baker from duplicating work.

        Args:
          specs (Sequence[ClipSpec]): what to produce.

        Returns:
          list[ClipResult]: one result per spec, in order. A failure is reported
            per spec rather than raised, so one bad recipe cannot take down a
            whole batch.
        """
        if not specs:
            return []
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001 - reported per spec, not raised
            traceback.print_exc()
            return [
                ClipResult(spec=s, error=f"{type(exc).__name__}: {exc}") for s in specs
            ]

        tokens: dict[str, np.ndarray] = {}
        live: set[str] = set()
        to_sample: list[ClipSpec] = []
        results: list[ClipResult] = []

        for spec in specs:
            try:
                if self.bank.has(spec.item_id):
                    tokens[spec.item_id] = self.bank.tokens(spec.item_id)
                elif spec.is_reference:
                    tokens[spec.item_id] = self._real_tokens(spec)
                    self.bank.add(spec, tokens[spec.item_id])
                else:
                    to_sample.append(spec)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                results.append(
                    ClipResult(spec=spec, error=f"{type(exc).__name__}: {exc}")
                )

        # Lanes run to the longest in their batch, so mixing a 10 s clip with a
        # 90 s one would charge the short clip 90 s of steps. Group by length.
        by_length: dict[int, list[ClipSpec]] = {}
        for spec in to_sample:
            by_length.setdefault(spec.n_frames, []).append(spec)

        for group in by_length.values():
            try:
                assert self._generator is not None
                sampled = self._generator.sample_batch(
                    [self._request(spec) for spec in group]
                )
                for spec, codes in zip(group, sampled):
                    tokens[spec.item_id] = codes.numpy().astype(np.int16)
                    self.bank.add(spec, tokens[spec.item_id])
                    live.add(spec.item_id)
            except Exception as exc:  # noqa: BLE001 - the batch failed, not the loop
                traceback.print_exc()
                results.extend(
                    ClipResult(spec=spec, error=f"{type(exc).__name__}: {exc}")
                    for spec in group
                )

        failed = {r.spec.item_id for r in results}
        for spec in specs:
            if spec.item_id in failed or spec.item_id not in tokens:
                continue
            try:
                results.append(
                    self._audio_result(spec, tokens[spec.item_id], spec.item_id in live)
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                results.append(
                    ClipResult(spec=spec, error=f"{type(exc).__name__}: {exc}")
                )
        order = {spec.item_id: i for i, spec in enumerate(specs)}
        return sorted(results, key=lambda r: order[r.spec.item_id])

    def produce(self, spec: ClipSpec) -> ClipResult:
        """
        Produce one clip.

        Args:
          spec (ClipSpec): what to produce.

        Returns:
          ClipResult: tokens plus normalized int16 PCM, or an error.
        """
        return self.produce_many([spec])[0]

    def close(self) -> None:
        """Release the model and decoder."""
        self._generator = None
        self._decoder = None
        self._tracks = {}
        self._ordered_tracks = []
        self._cache_dir = None
        self._loaded = False


def run_service(
    cfg: AbConfig, requests: "mp.Queue[Any]", results: "mp.Queue[Any]"
) -> None:
    """
    Child-process entry point: serve requests until told to stop.

    Args:
      cfg (AbConfig): harness config.
      requests (mp.Queue[Any]): inbound ClipRequest / SwitchCheckpoint /
        Shutdown messages.
      results (mp.Queue[Any]): outbound ClipResult / CheckpointChanged messages.
    """
    service = GenerationService(cfg)
    try:
        service.load()
    except Exception as exc:  # noqa: BLE001 - report instead of dying silently
        traceback.print_exc()
        results.put(ClipResult(spec=None, error=f"startup failed: {exc}"))  # type: ignore[arg-type]
        return
    max_batch = max(1, cfg.generator.max_batch)
    wait = cfg.generator.batch_wait_s

    def apply_switch(request: SwitchCheckpoint) -> None:
        """
        Args:
          request (SwitchCheckpoint): the model the rater picked. A failure is
            reported and the old model kept, so a typo in a path does not end
            the session.
        """
        try:
            service.switch_checkpoint(request.checkpoint)
            results.put(CheckpointChanged(checkpoint=service.checkpoint))
        except Exception as exc:  # noqa: BLE001 - report, keep serving
            traceback.print_exc()
            results.put(
                CheckpointChanged(
                    checkpoint=service.checkpoint,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    while True:
        try:
            message = requests.get(timeout=1.0)
        except queue.Empty:
            continue
        if isinstance(message, Shutdown):
            break
        if isinstance(message, SwitchCheckpoint):
            apply_switch(message)
            continue
        if not isinstance(message, ClipRequest):
            continue
        # Gather whatever else is already waiting: batching is close to free and
        # is the difference between generating during a session and not. The
        # short wait covers the gap between a pair's two requests arriving.
        batch = [message.spec]
        stop = False
        # A switch that lands mid-gather is applied after the batch: the clips
        # already in it were drawn against the old model and must stay so.
        switch: SwitchCheckpoint | None = None
        while len(batch) < max_batch:
            try:
                extra = requests.get(timeout=wait)
            except queue.Empty:
                break
            if isinstance(extra, Shutdown):
                stop = True
                break
            if isinstance(extra, SwitchCheckpoint):
                switch = extra
                break
            if isinstance(extra, ClipRequest):
                batch.append(extra.spec)
        for result in service.produce_many(batch):
            results.put(result)
        if switch is not None:
            apply_switch(switch)
        if stop:
            break
    service.close()
