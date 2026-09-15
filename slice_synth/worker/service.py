"""
The synthesis service: the only place that owns a GPU.

One implementation with two front-ends, exactly as the rating harness does it.
slice_synth.render runs it in-process, where a failure is a readable traceback;
the app runs it in a child process, where a CUDA OOM kills a worker rather than
the session. A single code path is what keeps what the app produces identical to
what the CLI produces.

Its whole job is turning recipes into audio. A StyleSpec becomes a 1024-d
descriptor here, deterministically from its own seed, and the resolved vector
travels back with the clip -- for the two random modes that vector is the only
record of what was heard, and without it a keeper could never be reproduced.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import random
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ab_harness.config import REPO
from ab_harness.model.audio import fill_fraction, normalize_lufs, to_int16
from ab_harness.model.pair_sampler import TrackInfo, pick_style_window
from ab_harness.worker.decoder import TokenDecoder
from ab_harness.worker.generator import SampleRequest
from ab_harness.worker.loading import LoadedModel, load_ar_checkpoint
from slice_synth.config import SynthConfig
from slice_synth.model.types import SAME_TRACK, RenderSpec, StyleSpec
from slice_synth.worker.encoder import FileEncoder
from slice_synth.worker.protocol import (
    Cancel,
    RenderProgress,
    RenderRequest,
    RenderResult,
    Shutdown,
    SwitchCheckpoint,
    WorkerReady,
)
from train_ar import TrackTokens

PROMPT_CACHE = "~/.cache/infected_pbm/synth_prompts"


def slerp(a: torch.Tensor, b: torch.Tensor, mix: float) -> torch.Tensor:
    """
    Interpolate along the sphere the style descriptors live on.

    Descriptors are L2-normalized, so a straight average of two of them is
    shorter than either -- and the model was never shown a short one. Slerp keeps
    the result on the sphere, which is what makes "halfway between these two
    moods" a thing the model can actually read.

    Args:
      a (torch.Tensor): (D,) unit vector.
      b (torch.Tensor): (D,) unit vector.
      mix (float): 0.0 is a, 1.0 is b.

    Returns:
      torch.Tensor: (D,) unit vector.
    """
    dot = float(torch.clamp((a * b).sum(), -1.0, 1.0))
    omega = float(np.arccos(dot))
    if abs(np.sin(omega)) < 1e-6:
        # (anti)parallel: slerp is undefined, and a lerp is already correct
        return F.normalize(a * (1.0 - mix) + b * mix, dim=-1)
    scale = np.sin(omega)
    return F.normalize(
        (float(np.sin((1.0 - mix) * omega)) * a + float(np.sin(mix * omega)) * b)
        / scale,
        dim=-1,
    )


class SynthService:
    """
    Loads the checkpoint, corpus, decoder and encoder, and renders RenderSpecs.

    Args:
      cfg (SynthConfig): synthesizer config.
    """

    def __init__(self, cfg: SynthConfig) -> None:
        self.cfg = cfg
        self.checkpoint = cfg.generator.checkpoint
        self._model: LoadedModel | None = None
        self._decoder: TokenDecoder | None = None
        self._encoder: FileEncoder | None = None

    # -- startup -------------------------------------------------------------

    def load(self) -> None:
        """
        Load everything heavy. Safe to call twice; the second call is a no-op.
        """
        if self._model is not None:
            return
        self._load_checkpoint(self.cfg.generator.checkpoint)
        assert self._model is not None
        gen_cfg = self.cfg.generator
        self._decoder = TokenDecoder(
            REPO / gen_cfg.decoder_onnx,
            hop=self._model.hop,
            sample_rate=self._model.sample_rate,
            use_gpu=gen_cfg.use_gpu_decoder,
        )
        self._encoder = FileEncoder(
            REPO / gen_cfg.encoder_onnx,
            self._model.meta,
            chunk_frames=self._model.ar_cfg.tokenizer.chunk_frames,
            margin=self._model.ar_cfg.tokenizer.margin,
            cache_dir=Path(PROMPT_CACHE).expanduser(),
        )

    def _load_checkpoint(self, checkpoint: str) -> None:
        """
        Args:
          checkpoint (str): repo-relative checkpoint path.

        Raises:
          FileNotFoundError: when the checkpoint or its token cache is missing.
        """
        self._model = load_ar_checkpoint(checkpoint, self.cfg.generator, self._model)
        self.checkpoint = checkpoint

    def switch_checkpoint(self, checkpoint: str) -> None:
        """
        Sample from a different model from here on.

        The old model stays in place until the new one is built, so a bad path
        costs an error message rather than a dead session.

        Args:
          checkpoint (str): repo-relative checkpoint path.
        """
        if checkpoint == self.checkpoint and self._model is not None:
            return
        self.load()
        self._load_checkpoint(checkpoint)

    def close(self) -> None:
        """Release the model, decoder and encoder."""
        if self._encoder is not None:
            self._encoder.close()
        self._model = None
        self._decoder = None
        self._encoder = None

    # -- corpus --------------------------------------------------------------

    @property
    def model(self) -> LoadedModel:
        """
        Returns:
          LoadedModel: the loaded checkpoint, loading it first if needed.
        """
        self.load()
        assert self._model is not None
        return self._model

    def corpus(self) -> list[TrackInfo]:
        """
        Returns:
          list[TrackInfo]: the torch-free corpus view the UI picks tracks from.
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
            for t in sorted(self.model.tracks, key=lambda t: t.track_idx)
        ]

    def ready(self) -> WorkerReady:
        """
        Returns:
          WorkerReady: everything the UI needs to build its controls.
        """
        return WorkerReady(
            checkpoint=self.checkpoint,
            tracks=self.corpus(),
            meta=dict(self.model.meta),
            window_frames=self.model.generator.window - self.model.generator.depth + 1,
        )

    def _track(self, track_idx: int) -> TrackTokens:
        """
        Args:
          track_idx (int): id from the corpus view.

        Returns:
          TrackTokens: the cached track.

        Raises:
          KeyError: when the id is not in this checkpoint's corpus.
        """
        tracks = self.model.by_idx
        if track_idx not in tracks:
            raise KeyError(f"no track {track_idx} in {self.model.cache_dir.name}")
        return tracks[track_idx]

    # -- conditioning --------------------------------------------------------

    def _span(self, spec: RenderSpec, track_idx: int) -> tuple[int, int]:
        """
        The frame range being generated, expressed in one track's coordinates.

        Only meaningful when the style descriptor and the generated audio come
        from the same track; otherwise there is nothing to be disjoint from and
        an empty span leaves every window eligible.

        Args:
          spec (RenderSpec): the render.
          track_idx (int): the track the style window would be drawn from.

        Returns:
          tuple[int, int]: half-open span, or (0, 0) for "no constraint".
        """
        prompt = spec.prompt
        if prompt.kind == "corpus" and prompt.track_idx == track_idx:
            return prompt.start_frame, prompt.start_frame + spec.n_frames
        return 0, 0

    def _window_vector(
        self, spec: RenderSpec, style: StyleSpec, use_b: bool
    ) -> torch.Tensor:
        """
        Look up one precomputed style window.

        Args:
          spec (RenderSpec): the render, for the disjointness span.
          style (StyleSpec): the recipe.
          use_b (bool): read the interp endpoint B instead of A.

        Returns:
          torch.Tensor: (D,) unit descriptor.
        """
        track_idx = style.track_b if use_b else style.track_idx
        window = style.window_b if use_b else style.window
        if track_idx == SAME_TRACK:
            # "follow the render" -- the render's own track, or the prompt's when
            # the id stream is nulled and there is no track to follow.
            track_idx = (
                spec.track_idx
                if spec.track_idx is not None
                else (spec.prompt.track_idx if spec.prompt.kind == "corpus" else 0)
            )
        track = self._track(int(track_idx))
        if window < 0:
            # Section 11.3: a descriptor computed from the same window as the
            # target is a compressed copy of the answer. Training enforces this
            # rule, so sampling has to as well or it runs off-distribution.
            bounds = [(int(lo), int(hi)) for lo, hi in track.style_bounds.tolist()]
            start, end = self._span(spec, track_idx)
            window = pick_style_window(bounds, start, end, random.Random(style.seed))
        window = max(0, min(window, track.style.shape[0] - 1))
        return track.style[window].float()

    def _style_vector(self, spec: RenderSpec, style: StyleSpec) -> torch.Tensor:
        """
        Resolve one StyleSpec, ignoring its walk.

        Args:
          spec (RenderSpec): the render.
          style (StyleSpec): the recipe.

        Returns:
          torch.Tensor: (D,) unit descriptor.
        """
        dim = int(self.model.tracks[0].style.shape[-1])
        generator = torch.Generator().manual_seed(int(style.seed))
        if style.kind == "random":
            return F.normalize(torch.randn(dim, generator=generator), dim=-1)
        if style.kind == "interp":
            return slerp(
                self._window_vector(spec, style, False),
                self._window_vector(spec, style, True),
                float(style.mix),
            )
        base = self._window_vector(spec, style, False)
        if style.kind == "jitter":
            noise = torch.randn(base.shape, generator=generator)
            return F.normalize(base + float(style.noise) * noise, dim=-1)
        return base

    def _walk_step(self, spec: RenderSpec, style: StyleSpec, segment: int) -> StyleSpec:
        """
        The recipe for one later segment of a walking style.

        Args:
          spec (RenderSpec): the render.
          style (StyleSpec): the base entry.
          segment (int): segment index, 1 or more.

        Returns:
          StyleSpec: a non-walking spec, seeded from (style.seed, segment) so the
            whole walk replays from the saved recipe. "windows" draws a window
            of one of the render's own tracks -- any of them, when several ids
            coexist -- unless the entry pins a track; "random" a fresh vector.
        """
        seed = int(style.seed) * 1009 + segment
        if style.walk == "random":
            return replace(style, kind="random", walk="none", seed=seed)
        source = style.track_idx
        if source == SAME_TRACK:
            pool = list(spec.tracks) or (
                [spec.prompt.track_idx] if spec.prompt.kind == "corpus" else [0]
            )
            source = random.Random(seed).choice(pool)
        return replace(
            style, kind="window", walk="none", track_idx=source, window=-1, seed=seed
        )

    def resolve_style(self, spec: RenderSpec) -> torch.Tensor:
        """
        Turn a StyleSpec into the descriptor(s) the model is conditioned on.

        Every mode returns unit vectors, matching how the corpus descriptors are
        built. Randomness is drawn from the spec's own seed rather than the global
        RNG, so a render is reproducible from its saved recipe alone.

        Args:
          spec (RenderSpec): the render.

        Returns:
          torch.Tensor: (S, D) descriptors, one per style segment -- S is 1
            unless the style walks. Zeros when the stream is nulled: the model
            substitutes its learned null, so the value is inert.
        """
        style = spec.style
        dim = int(self.model.tracks[0].style.shape[-1])
        if style is None:
            return torch.zeros(1, dim)
        first = self._style_vector(spec, style)
        positions = spec.n_frames + self.model.generator.depth - 1
        later = [
            self._style_vector(spec, self._walk_step(spec, style, k))
            for k in range(1, style.segments(positions))
        ]
        return torch.stack([first, *later])

    def resolve_prompt(self, spec: RenderSpec) -> torch.Tensor | None:
        """
        Turn a PromptSpec into the real codes a generation is forced to open with.

        Args:
          spec (RenderSpec): the render.

        Returns:
          torch.Tensor | None: (P, R) int64 codes, or None for a cold start.

        Raises:
          RuntimeError: when a file prompt cannot be decoded or encoded.
        """
        prompt = spec.prompt
        frames = prompt.frames(self.model.fps)
        if frames <= 0:
            return None
        if prompt.kind == "file":
            assert self._encoder is not None
            codes = self._encoder.slice(prompt.path, prompt.start_sec, frames)
            return torch.from_numpy(codes.astype(np.int64)) if codes.size else None
        track = self._track(prompt.track_idx)
        start = max(0, min(prompt.start_frame, track.num_frames - 1))
        return track.tokens[start : start + frames].long()

    def reference_tokens(self, spec: RenderSpec) -> np.ndarray:
        """
        The real codes for the span a render covers -- the tokenizer's ceiling.

        Args:
          spec (RenderSpec): a render with kind "reference".

        Returns:
          np.ndarray: (T, R) int16 codes, clamped to the track's length.
        """
        track_idx = (
            spec.prompt.track_idx if spec.prompt.kind == "corpus" else spec.track_idx
        )
        track = self._track(int(track_idx or 0))
        start = max(0, min(spec.prompt.start_frame, track.num_frames - 1))
        return track.tokens[start : start + spec.n_frames].numpy().astype(np.int16)

    # -- production ----------------------------------------------------------

    def _request(self, spec: RenderSpec, style: torch.Tensor) -> SampleRequest:
        """
        Args:
          spec (RenderSpec): the recipe.
          style (torch.Tensor): (S, D) resolved descriptors.

        Returns:
          SampleRequest: one lane of a sampling batch.
        """
        walking = spec.style is not None and spec.style.walking
        return SampleRequest(
            track_idx=(
                self.model.generator.model.num_tracks
                if spec.track_idx is None
                else int(spec.track_idx)
            ),
            style=style[0],
            use_track_id=spec.track_idx is not None,
            use_style=spec.style is not None,
            frames=spec.n_frames,
            prompt=self.resolve_prompt(spec),
            temperature=spec.temperature,
            top_k=spec.top_k,
            top_p=spec.top_p,
            cfg_strength=spec.cfg_strength,
            seed=spec.seed,
            co_tracks=tuple(int(t) for t in spec.co_tracks),
            style_schedule=tuple(style[1:]),
            style_period=spec.style.period if walking and spec.style else 0,
        )

    def _audio_result(
        self, spec: RenderSpec, tokens: np.ndarray, style: torch.Tensor | None
    ) -> RenderResult:
        """
        Decode and loudness-match one clip's tokens.

        Args:
          spec (RenderSpec): the render.
          tokens (np.ndarray): (T, R) codes.
          style (torch.Tensor | None): (S, D) descriptors that were used.

        Returns:
          RenderResult: the finished clip.
        """
        assert self._decoder is not None
        audio = self._postprocess(
            self._decoder.decode(tokens), self._decoder.sample_rate
        )
        audio = normalize_lufs(
            audio, self._decoder.sample_rate, self.cfg.output.target_lufs
        )
        pcm = to_int16(audio)
        return RenderResult(
            spec=spec,
            tokens=tokens,
            pcm=pcm,
            sample_rate=self._decoder.sample_rate,
            # a single segment is stored flat, as every recipe before walking was
            style_used=(
                None if style is None else style.detach().cpu().squeeze(0).numpy()
            ),
            fill=fill_fraction(pcm),
        )

    def _postprocess(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Hook for cleanup applied after decoding and before loudness matching.

        Identity today. Copy-synthesis put 55% of the spectral error and 81% of
        the crest error in the tokenizer alone, with the decoder's phase
        unconstrained (the GAN is off), so a diffusion decoder is the next real
        quality lever and this is where it attaches.

        Args:
          pcm (np.ndarray): (N,) float32 decoder output.
          sample_rate (int): samples per second.

        Returns:
          np.ndarray: (N,) float32 waveform.
        """
        return pcm

    def _chunks(self, specs: Sequence[RenderSpec]) -> list[list[RenderSpec]]:
        """
        Split sampling work into passes that fit one KV cache.

        Lanes run to the longest in their batch, so mixing a 10 s clip with a
        90 s one would charge the short clip 90 s of steps -- hence the grouping
        by length. Within a length, a *guided* lane occupies two rows (one of
        them the unconditional pass), so the budget counts rows, not clips.

        Args:
          specs (Sequence[RenderSpec]): the clips to sample.

        Returns:
          list[list[RenderSpec]]: one list per sampling pass.
        """
        budget = max(1, self.cfg.generator.max_batch)
        by_length: dict[int, list[RenderSpec]] = {}
        for spec in specs:
            by_length.setdefault(spec.n_frames, []).append(spec)

        passes: list[list[RenderSpec]] = []
        for group in by_length.values():
            current: list[RenderSpec] = []
            rows = 0
            for spec in group:
                cost = 2 if spec.cfg_strength > 0.0 and spec.style is not None else 1
                if current and rows + cost > budget:
                    passes.append(current)
                    current, rows = [], 0
                current.append(spec)
                rows += cost
            if current:
                passes.append(current)
        return passes

    def render_many(
        self,
        specs: Sequence[RenderSpec],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[RenderResult]:
        """
        Produce a batch of clips, sampling everything new in as few passes as fit.

        A failure is reported per spec rather than raised, so one bad recipe --
        an undecodable prompt file, a track id from another corpus -- cannot take
        down the whole batch.

        Args:
          specs (Sequence[RenderSpec]): what to produce.
          progress (Callable[[int, int], None] | None): called with
            (positions done, positions total) across the whole request.

        Returns:
          list[RenderResult]: one result per spec, in the order asked for.
        """
        if not specs:
            return []
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001 - reported per spec, not raised
            traceback.print_exc()
            return [
                RenderResult(spec=s, error=f"{type(exc).__name__}: {exc}")
                for s in specs
            ]

        results: dict[str, RenderResult] = {}
        to_sample: list[RenderSpec] = []
        styles: dict[str, torch.Tensor] = {}

        for spec in specs:
            try:
                if spec.is_reference:
                    results[spec.item_id] = self._audio_result(
                        spec, self.reference_tokens(spec), None
                    )
                else:
                    styles[spec.item_id] = self.resolve_style(spec)
                    to_sample.append(spec)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                results[spec.item_id] = RenderResult(
                    spec=spec, error=f"{type(exc).__name__}: {exc}"
                )

        passes = self._chunks(to_sample)
        totals = [max((s.n_frames for s in group), default=0) for group in passes]
        grand_total = sum(totals)
        done_before = 0

        for index, group in enumerate(passes):
            offset = done_before

            def report(step: int, total: int, _offset: int = offset) -> None:
                if progress is not None:
                    progress(min(_offset + step, grand_total), grand_total)

            try:
                requests = [self._request(spec, styles[spec.item_id]) for spec in group]
                sampled = self.model.generator.sample_batch(requests, report)
                for spec, codes in zip(group, sampled):
                    results[spec.item_id] = self._audio_result(
                        spec, codes.numpy().astype(np.int16), styles[spec.item_id]
                    )
            except Exception as exc:  # noqa: BLE001 - the pass failed, not the loop
                traceback.print_exc()
                for spec in group:
                    results[spec.item_id] = RenderResult(
                        spec=spec, error=f"{type(exc).__name__}: {exc}"
                    )
            done_before += totals[index]
            if progress is not None:
                progress(min(done_before, grand_total), grand_total)

        return [results[spec.item_id] for spec in specs if spec.item_id in results]


def run_service(
    cfg: SynthConfig, requests: "mp.Queue[Any]", results: "mp.Queue[Any]"
) -> None:
    """
    Child-process entry point: serve requests until told to stop.

    Args:
      cfg (SynthConfig): synthesizer config.
      requests (mp.Queue[Any]): inbound RenderRequest / SwitchCheckpoint /
        Cancel / Shutdown messages.
      results (mp.Queue[Any]): outbound WorkerReady / RenderProgress /
        RenderResult messages.
    """
    service = SynthService(cfg)
    try:
        service.load()
        results.put(service.ready())
    except Exception as exc:  # noqa: BLE001 - report instead of dying silently
        traceback.print_exc()
        results.put(WorkerReady(checkpoint=cfg.generator.checkpoint, error=str(exc)))
        return

    def drain_cancels() -> list[Any]:
        """
        Returns:
          list[Any]: messages pulled off the queue that were not cancellations,
            so a Cancel can empty the backlog without eating a switch or a stop.
        """
        kept: list[Any] = []
        while True:
            try:
                message = requests.get_nowait()
            except queue.Empty:
                return kept
            if isinstance(message, RenderRequest):
                continue
            kept.append(message)

    pending: list[Any] = []
    while True:
        if pending:
            message = pending.pop(0)
        else:
            try:
                message = requests.get(timeout=1.0)
            except queue.Empty:
                continue

        if isinstance(message, Shutdown):
            break
        if isinstance(message, Cancel):
            pending = drain_cancels() + pending
            continue
        if isinstance(message, SwitchCheckpoint):
            try:
                service.switch_checkpoint(message.checkpoint)
                results.put(service.ready())
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                traceback.print_exc()
                ready = service.ready()
                ready.error = f"{type(exc).__name__}: {exc}"
                results.put(ready)
            continue
        if not isinstance(message, RenderRequest):
            continue

        def report(step: int, total: int, batch: str = message.batch_id) -> None:
            results.put(RenderProgress(batch_id=batch, step=step, total=total))

        for result in service.render_many(message.specs, report):
            results.put(result)

    service.close()


__all__ = ["SynthService", "run_service", "slerp"]
