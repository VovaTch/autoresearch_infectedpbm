"""
KV-cached, batched, sliding-window sampling from a trained AR checkpoint.

generate_ar.generate re-runs the whole prefix every step, so cost is O(L^2): a
couple of minutes for a 10 s clip and hours for 90 s. That is fine for rendering
one demo and useless for a rating harness, which needs a stream of clips at
listening pace.

Three things make the difference, in descending order of effect:

1. A KV cache, which turns O(L^2) into O(L).

2. Batching. Measured on this checkpoint, one sampling step costs 7.4 ms at
   batch 1 and 7.6 ms at batch 16 -- the loop is bound by kernel launch latency,
   not by arithmetic, so sixteen clips cost what one does. That is 12.9 s per
   10 s clip alone against 0.8 s each in a batch of sixteen, and it is what
   makes generating during a rating session viable at all. bf16 autocast was
   tried and is slightly SLOWER for the same reason: it adds casts to a loop
   that was never compute-bound.

3. A sliding window. The model was trained on crop_frames = 4096 frames =
   23.8 s, so a 90 s clip is four times its context. Rather than extrapolating
   rotary positions far past anything seen in training, the sampler keeps the
   most recent `window` positions and re-primes the cache from them, putting
   every query back inside the geometry training used. Measured cost is nil: a
   90 s clip runs at 8.3 ms/step against 9.0 ms/step for a 10 s one.

Coherence past 24 s will still be weak. That is not a bug to hide; it is the
thing the structure tier's "which is more interesting?" question measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from generate_ar import sample_step
from train_ar import (
    PREFIX_POSITIONS,
    ArTransformer,
    KVCache,
    build_delay_grid,
    undelay_grid,
)


@dataclass
class SampleRequest:
    """
    One clip to sample. Lanes in a batch are independent; nothing is shared.

    Args:
      track_idx (int): conditioning track id.
      style (torch.Tensor): (style_dim,) descriptor for the first segment.
      use_track_id (bool): False nulls the id stream.
      use_style (bool): False nulls the style stream.
      frames (int): aligned frames T to produce.
      prompt (torch.Tensor | None): (P, R) real codes to prime with.
      temperature (float): sampling temperature; <= 0 is argmax.
      top_k (int): top-k cutoff.
      top_p (float): nucleus cutoff.
      cfg_strength (float): guidance strength; 0.0 samples the conditional
        logits directly, which is what generate_ar.generate does. A guided lane
        occupies two rows of the batch, one of them the unconditional pass.
      seed (int | None): per-lane RNG seed.
      co_tracks (tuple[int, ...]): further track ids that coexist with
        track_idx as id tokens at the same rotary position. Ignored when the id
        stream is nulled.
      style_schedule (tuple[torch.Tensor, ...]): descriptors for segments
        1, 2, ... ; segment k takes over at position k * style_period, the last
        one holding to the end. A precomputed tuple today; a model that predicts
        the next style would replace it with something called at each boundary.
      style_period (int): positions per style segment; 0 keeps `style` fixed.
    """

    track_idx: int
    style: torch.Tensor
    use_track_id: bool = True
    use_style: bool = True
    frames: int = 1722
    prompt: torch.Tensor | None = None
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    cfg_strength: float = 0.0
    seed: int | None = None
    co_tracks: tuple[int, ...] = ()
    style_schedule: tuple[torch.Tensor, ...] = ()
    style_period: int = 0

    def style_at(self, step: int) -> torch.Tensor | None:
        """
        Args:
          step (int): position about to be sampled.

        Returns:
          torch.Tensor | None: the descriptor that takes over at exactly this
            step, or None when nothing changes here.
        """
        if not self.style_schedule or self.style_period <= 0 or step <= 0:
            return None
        if step % self.style_period:
            return None
        segment = min(step // self.style_period, len(self.style_schedule))
        return self.style_schedule[segment - 1]

    @property
    def guided(self) -> bool:
        """
        Returns:
          bool: True when guidance is both requested and meaningful. With every
            stream nulled there is no conditional to guide towards, so the
            strength is ignored rather than amplifying a zero delta.
        """
        return self.cfg_strength > 0.0 and (self.use_track_id or self.use_style)


class ArGenerator:
    """
    Sampling front-end for one loaded ArTransformer.

    Args:
      model (ArTransformer): trained model, already on the target device and in
        eval mode.
      device (torch.device): compute device.
      window_frames (int): frames of context to retain; use the checkpoint's
        crop_frames so sampling stays in-distribution.
      reprime_frac (float): fraction of the window dropped at each re-prime.
        Smaller keeps more context and re-primes more often.
    """

    def __init__(
        self,
        model: ArTransformer,
        device: torch.device,
        window_frames: int = 4096,
        reprime_frac: float = 0.25,
    ) -> None:
        self.model = model
        self.device = device
        self.depth = model.num_rq
        self.pad_id = model.pad_id
        # +depth-1 for the delay pattern, +PREFIX_POSITIONS for the conditioning
        self.window = window_frames + self.depth - 1
        self.max_length = self.window + PREFIX_POSITIONS
        self.keep_rows = max(1, int(self.window * (1.0 - reprime_frac)))
        self._cache: KVCache | None = None
        self._fed: list[torch.Tensor] = []
        self._prefix: tuple[torch.Tensor, ...] | None = None

    # -- cache plumbing ------------------------------------------------------

    def _start(self, cache: KVCache | None) -> KVCache:
        """
        Args:
          cache (KVCache | None): buffers to refill, or None for fresh ones.

        Returns:
          KVCache: a cache holding the current prefix.
        """
        assert self._prefix is not None
        ids, styles, drop_id, drop_style, extra_ids, extra_valid = self._prefix
        return self.model.start_incremental(
            ids,
            styles,
            drop_id,
            drop_style,
            self.max_length + extra_ids.shape[1],
            cache,
            extra_ids=extra_ids,
            extra_valid=extra_valid,
        )

    def _open(self, prefix: tuple[torch.Tensor, ...]) -> None:
        """
        Start a batch: build or reset the cache and load the conditioning prefix.

        Args:
          prefix (tuple[torch.Tensor, ...]): (ids, styles, drop_id, drop_style,
            extra_ids, extra_valid).
        """
        self._prefix = prefix
        rows = prefix[0].shape[0]
        length = self.max_length + prefix[4].shape[1]
        reuse = (
            self._cache
            if self._cache is not None
            and self._cache.keys[0].shape[0] == rows
            and self._cache.max_length == length
            else None
        )
        self._cache = self._start(reuse)
        self._fed = []

    def _reprime(self, keep_rows: int | None = None) -> None:
        """
        Rebuild the cache from the tail of what has been fed so far.

        Args:
          keep_rows (int | None): rows of context to re-feed; the sliding-window
            default drops a fraction, while a conditioning change mid-clip keeps
            everything that still fits so no context is lost to it.
        """
        assert self._cache is not None and self._prefix is not None
        keep = self._fed[-(self.keep_rows if keep_rows is None else keep_rows) :]
        self._cache = self._start(self._cache)
        self._fed = keep
        if keep:
            self.model.step_incremental(torch.cat(keep, dim=1), self._cache)

    def _restyle(self, rows: Sequence[int], styles: Sequence[torch.Tensor]) -> None:
        """
        Swap the style token of some rows and re-prime so the change takes.

        Args:
          rows (Sequence[int]): batch rows whose style changes.
          styles (Sequence[torch.Tensor]): (style_dim,) descriptor per row.
        """
        assert self._prefix is not None
        updated = self._prefix[1].clone()
        for row, style in zip(rows, styles):
            updated[row] = style.to(updated.device, updated.dtype)
        self._prefix = (self._prefix[0], updated) + tuple(self._prefix[2:])
        self._reprime(keep_rows=min(len(self._fed), self.window - 1))

    def _feed(self, rows: torch.Tensor) -> torch.Tensor:
        """
        Append one position and return the logits scoring the next one.

        Args:
          rows (torch.Tensor): (B, 1, R) int64 token rows, one per batch row.

        Returns:
          torch.Tensor: (B, R, V) logits for the position after `rows`.
        """
        assert self._cache is not None
        if self._cache.length + 1 > self.max_length:
            self._reprime()
        logits = self.model.step_incremental(rows, self._cache)
        self._fed.append(rows)
        return logits[:, -1]

    # -- sampling ------------------------------------------------------------

    def _build_prefix(
        self, requests: Sequence[SampleRequest]
    ) -> tuple[tuple[torch.Tensor, ...], list[int], list[int]]:
        """
        Lay lanes out across batch rows, expanding guided lanes to two.

        Args:
          requests (Sequence[SampleRequest]): the lanes to sample.

        Extra ids are padded to the widest lane in the batch; a row's unused
        slots are marked invalid and never attended, so a single-track lane in a
        multi-track batch samples exactly as it would alone.

        Returns:
          tuple: the prefix tensors (ids, styles, drop_id, drop_style,
            extra_ids, extra_valid), the conditional row index per lane, and the
            unconditional row index per lane (-1 when the lane is unguided).
        """
        null_id = self.model.num_tracks
        ids: list[int] = []
        styles: list[torch.Tensor] = []
        drop_id: list[bool] = []
        drop_style: list[bool] = []
        extras: list[tuple[int, ...]] = []
        cond_row: list[int] = []
        null_row: list[int] = []

        for request in requests:
            style = request.style.to(self.device).float()
            if request.guided:
                null_row.append(len(ids))
                ids.append(null_id)
                styles.append(torch.zeros_like(style))
                drop_id.append(True)
                drop_style.append(True)
                extras.append(())
            else:
                null_row.append(-1)
            cond_row.append(len(ids))
            ids.append(request.track_idx if request.use_track_id else null_id)
            styles.append(style)
            drop_id.append(not request.use_track_id)
            drop_style.append(not request.use_style)
            extras.append(tuple(request.co_tracks) if request.use_track_id else ())

        width = max((len(e) for e in extras), default=0)
        extra_ids = torch.full((len(ids), width), null_id, dtype=torch.long)
        extra_valid = torch.zeros((len(ids), width), dtype=torch.bool)
        for row, extra in enumerate(extras):
            extra_ids[row, : len(extra)] = torch.tensor(extra, dtype=torch.long)
            extra_valid[row, : len(extra)] = True

        prefix = (
            torch.tensor(ids, device=self.device, dtype=torch.long),
            torch.stack(styles).to(self.device),
            torch.tensor(drop_id, device=self.device),
            torch.tensor(drop_style, device=self.device),
            extra_ids.to(self.device),
            extra_valid.to(self.device),
        )
        return prefix, cond_row, null_row

    @torch.no_grad()
    def sample_batch(
        self,
        requests: Sequence[SampleRequest],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[torch.Tensor]:
        """
        Sample several clips at once, one lane each.

        Mirrors ArLightningModule._prepare: the model sees a BOS row of pad_id
        followed by the delayed grid shifted right by one, so position i is
        scored from grid[i-1]. Any drift from that is a train/generate mismatch.

        Lanes of different lengths run to the longest and are truncated after,
        which costs nothing that batching has not already paid for.

        Args:
          requests (Sequence[SampleRequest]): the clips to sample.
          progress (Callable[[int, int], None] | None): called with
            (step, total) roughly every 512 positions.

        Returns:
          list[torch.Tensor]: (T, R) int64 aligned codes per request, on the CPU.
        """
        if not requests:
            return []
        depth = self.depth
        prefix, cond_row, null_row = self._build_prefix(requests)
        rows = prefix[0].shape[0]
        self._open(prefix)

        # multinomial requires the generator on the same device as its input,
        # so this must follow self.device rather than defaulting to the CPU.
        generators = [
            (
                torch.Generator(device=self.device).manual_seed(r.seed)
                if r.seed is not None
                else None
            )
            for r in requests
        ]
        # Only the entries the prompt actually covers may be forced.
        # build_delay_grid pads the corners of the delayed grid, and its trailing
        # corner lands on real output frames: forcing it writes pad_id -- one
        # past the last codebook entry -- into the depth-1 frames straight after
        # the prompt, which the decoder then gathers out of range. Level d of
        # grid row s holds aligned frame s-d, so an entry is real exactly when
        # 0 <= s-d < len(prompt).
        forced = [
            (
                build_delay_grid(r.prompt[None].to(self.device), self.pad_id)[0]
                if r.prompt is not None and r.prompt.numel()
                else None
            )
            for r in requests
        ]
        prompt_lens = [
            int(r.prompt.shape[0]) if r.prompt is not None and r.prompt.numel() else 0
            for r in requests
        ]
        offsets = torch.arange(depth, device=self.device)
        lengths = [r.frames + depth - 1 for r in requests]
        total = max(lengths)
        # scatter map: which lane feeds each batch row
        lane_of_row = torch.zeros(rows, dtype=torch.long)
        for lane, (cond, null) in enumerate(zip(cond_row, null_row)):
            lane_of_row[cond] = lane
            if null >= 0:
                lane_of_row[null] = lane
        lane_of_row = lane_of_row.to(self.device)

        grids = [
            torch.empty((0, depth), dtype=torch.long, device=self.device)
            for _ in requests
        ]
        current = torch.full(
            (rows, 1, depth), self.pad_id, dtype=torch.long, device=self.device
        )

        for step in range(total):
            changes = [
                (cond_row[lane], style)
                for lane, request in enumerate(requests)
                if (style := request.style_at(step)) is not None
            ]
            if changes:
                self._restyle([row for row, _ in changes], [s for _, s in changes])
            logits = self._feed(current).float()
            picks = []
            for lane, request in enumerate(requests):
                lane_logits = logits[cond_row[lane]]
                if request.guided:
                    null_logits = logits[null_row[lane]]
                    lane_logits = null_logits + request.cfg_strength * (
                        lane_logits - null_logits
                    )
                pick = sample_step(
                    lane_logits,
                    request.temperature,
                    request.top_k,
                    request.top_p,
                    generators[lane],
                )
                prompt = forced[lane]
                if prompt is not None and step < prompt.shape[0]:
                    aligned = step - offsets
                    real = (aligned >= 0) & (aligned < prompt_lens[lane])
                    pick = torch.where(real, prompt[step], pick)
                picks.append(pick)
                if step < lengths[lane]:
                    grids[lane] = torch.cat([grids[lane], pick[None]], dim=0)
            stacked = torch.stack(picks)
            current = stacked[lane_of_row].view(rows, 1, depth)
            if progress is not None and (step + 1) % 512 == 0:
                progress(step + 1, total)

        return [
            undelay_grid(grid[: lengths[lane]][None], requests[lane].frames)[0].cpu()
            for lane, grid in enumerate(grids)
        ]

    def sample(
        self,
        track_idx: int,
        style: torch.Tensor,
        use_track_id: bool,
        use_style: bool,
        frames: int,
        prompt: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_strength: float = 0.0,
        seed: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """
        Sample a single clip. A one-lane call into sample_batch, so there is no
        second code path to keep in step.

        Args:
          track_idx (int): conditioning track id.
          style (torch.Tensor): (style_dim,) descriptor.
          use_track_id (bool): False nulls the id stream.
          use_style (bool): False nulls the style stream.
          frames (int): aligned frames T to produce.
          prompt (torch.Tensor | None): (P, R) real codes to prime with.
          temperature (float): sampling temperature; <= 0 is argmax.
          top_k (int): top-k cutoff.
          top_p (float): nucleus cutoff.
          cfg_strength (float): guidance strength.
          seed (int | None): RNG seed.
          progress (Callable[[int, int], None] | None): progress callback.

        Returns:
          torch.Tensor: (T, R) int64 aligned codes on the CPU.
        """
        request = SampleRequest(
            track_idx=track_idx,
            style=style,
            use_track_id=use_track_id,
            use_style=use_style,
            frames=frames,
            prompt=prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_strength=cfg_strength,
            seed=seed,
        )
        return self.sample_batch([request], progress)[0]
