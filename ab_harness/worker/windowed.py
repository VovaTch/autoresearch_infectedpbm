"""
Window chaining for the non-autoregressive generators.

A bidirectional denoiser produces one fixed-length window (the crop it was
trained on, 512 frames) at a time. A clip longer than that is a chain of
windows, each re-primed with the tail of what came before: the last
`keep` frames of the output so far are handed to the next window as a clean
prefix and only the remainder is generated -- the same idea as ArGenerator's
sliding window, with the prefix conditioning the models were trained with
(train_zflow / train_mdm prefix_max_frac) doing the joining. A prompt is the
first window's prefix.

Lanes chain independently (a prompt of 200 frames and one of 0 generate
different amounts per window), so the loop runs per window over the lanes
still short of their target and hands the backend a batch with per-lane prefix
lengths. Output per lane is exactly `request.frames` frames, prompt included,
as ArGenerator returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from ab_harness.worker.generator import SampleRequest


@dataclass
class WindowBatch:
    """
    One window of work across the active lanes.

    Args:
      prefix (torch.Tensor): (B, W, R) int64 tokens; only the first
        prefix_frames[b] of lane b are meaningful, the rest is 0.
      prefix_frames (torch.Tensor): (B,) prefix length per lane (0 = none).
      track_idx (torch.Tensor): (B,) ids, the null id where the stream is off.
      style (torch.Tensor): (B, C) descriptors for this window.
      drop_id (torch.Tensor): (B,) bool, True where the id stream is nulled.
      drop_style (torch.Tensor): (B,) bool, True where the style is nulled.
      cfg (torch.Tensor): (B,) guidance per lane, 1.0 = conditional only.
      requests (list[SampleRequest]): the lane recipes, for sampling knobs.
      generators (list[torch.Generator]): one seeded RNG per lane.
    """

    prefix: torch.Tensor
    prefix_frames: torch.Tensor
    track_idx: torch.Tensor
    style: torch.Tensor
    drop_id: torch.Tensor
    drop_style: torch.Tensor
    cfg: torch.Tensor
    requests: list[SampleRequest]
    generators: list[torch.Generator]


class WindowedGenerator(ABC):
    """
    Base for samplers that fill fixed windows and chain them.

    Args:
      device (torch.device): compute device.
      window (int): frames per window, the model's training crop.
      depth (int): RVQ levels R.
      num_tracks (int): id vocabulary; the null id is num_tracks.
      pad_id (int): token id outside the codebook (for SampleSource parity).
      reprime_frac (float): fraction of the window generated fresh per chained
        window; the rest is the clean prefix carried over.
      max_prefix (int | None): longest prefix the model was trained to hold
        (prefix_max_frac x window); the carried prefix and a prompt are capped
        at it so sampling stays in-distribution. None = window - 1.
    """

    def __init__(
        self,
        device: torch.device,
        window: int,
        depth: int,
        num_tracks: int,
        pad_id: int,
        reprime_frac: float = 0.25,
        max_prefix: int | None = None,
    ) -> None:
        self.device = device
        self.window = window
        self.window_frames = window
        self.depth = depth
        self.num_tracks = num_tracks
        self.pad_id = pad_id
        cap = window - 1 if max_prefix is None else max(0, min(window - 1, max_prefix))
        self.keep = min(cap, int(round(window * (1.0 - reprime_frac))))

    @abstractmethod
    def rounds_per_window(self) -> int:
        """
        Returns:
          int: progress ticks one window produces.
        """

    @abstractmethod
    def fill_window(
        self, batch: WindowBatch, progress: Callable[[int], None]
    ) -> torch.Tensor:
        """
        Args:
          batch (WindowBatch): the active lanes' prefixes and conditionals.
          progress (Callable[[int], None]): called with rounds done so far.

        Returns:
          torch.Tensor: (B, W, R) int64 tokens, prefix frames as given.
        """

    @torch.no_grad()
    def sample_batch(
        self,
        requests: Sequence[SampleRequest],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[torch.Tensor]:
        """
        Sample several clips, chaining windows until each reaches its length.

        Args:
          requests (Sequence[SampleRequest]): the clips to sample.
          progress (Callable[[int, int], None] | None): called with
            (round, total) after every sampler round.

        Returns:
          list[torch.Tensor]: (frames, R) int64 aligned codes per request, on
            the CPU, prompt frames included.
        """
        if not requests:
            return []
        outputs: list[torch.Tensor] = []
        for r in requests:
            if r.prompt is not None and r.prompt.numel():
                outputs.append(r.prompt.long().to(self.device)[: r.frames])
            else:
                outputs.append(
                    torch.empty((0, self.depth), dtype=torch.int64, device=self.device)
                )
        generators = [
            torch.Generator(device=self.device).manual_seed(
                r.seed if r.seed is not None else int(torch.randint(0, 2**31, (1,)))
            )
            for r in requests
        ]
        windows = [self._windows_needed(r, outputs[i]) for i, r in enumerate(requests)]
        total = max(windows) * self.rounds_per_window()
        done = 0
        while True:
            active = [
                i for i, r in enumerate(requests) if outputs[i].shape[0] < r.frames
            ]
            if not active:
                break
            batch = self._window_batch(active, requests, outputs, generators)
            base = done

            def report(rounds: int) -> None:
                if progress is not None:
                    progress(base + rounds, total)

            filled = self.fill_window(batch, report)
            done = base + self.rounds_per_window()
            for row, lane in enumerate(active):
                p = int(batch.prefix_frames[row])
                new = filled[row, p:]
                room = requests[lane].frames - outputs[lane].shape[0]
                outputs[lane] = torch.cat([outputs[lane], new[:room]], dim=0)
        return [o.cpu() for o in outputs]

    def _windows_needed(self, request: SampleRequest, have: torch.Tensor) -> int:
        """
        Args:
          request (SampleRequest): the lane.
          have (torch.Tensor): (P, R) frames already in hand.

        Returns:
          int: windows the chain will take, for the progress denominator.
        """
        count, frames = 0, int(have.shape[0])
        while frames < request.frames:
            frames += self.window - min(frames, self.keep)
            count += 1
        return count

    def _window_batch(
        self,
        active: list[int],
        requests: Sequence[SampleRequest],
        outputs: list[torch.Tensor],
        generators: list[torch.Generator],
    ) -> WindowBatch:
        """
        Args:
          active (list[int]): lanes still generating.
          requests (Sequence[SampleRequest]): every lane's recipe.
          outputs (list[torch.Tensor]): frames produced so far per lane.
          generators (list[torch.Generator]): per-lane RNGs.

        Returns:
          WindowBatch: this window's inputs for the active lanes.
        """
        size = len(active)
        prefix = torch.zeros(
            (size, self.window, self.depth), dtype=torch.int64, device=self.device
        )
        lengths = torch.zeros(size, dtype=torch.int64, device=self.device)
        ids = torch.full(
            (size,), self.num_tracks, dtype=torch.int64, device=self.device
        )
        styles = []
        drop_id = torch.zeros(size, dtype=torch.bool, device=self.device)
        drop_style = torch.zeros(size, dtype=torch.bool, device=self.device)
        cfg = torch.ones(size, dtype=torch.float32, device=self.device)
        for row, lane in enumerate(active):
            r, have = requests[lane], outputs[lane]
            p = min(int(have.shape[0]), self.keep)
            if p:
                prefix[row, :p] = have[have.shape[0] - p :]
            lengths[row] = p
            if r.use_track_id:
                ids[row] = r.track_idx
            else:
                drop_id[row] = True
            drop_style[row] = not r.use_style
            style = r.style_at(int(have.shape[0])) if have.shape[0] else None
            if style is None:
                style = self._style_before(r, int(have.shape[0]))
            styles.append(style.to(self.device).float())
            if r.guided:
                cfg[row] = r.cfg_strength
        return WindowBatch(
            prefix=prefix,
            prefix_frames=lengths,
            track_idx=ids,
            style=torch.stack(styles),
            drop_id=drop_id,
            drop_style=drop_style,
            cfg=cfg,
            requests=[requests[i] for i in active],
            generators=[generators[i] for i in active],
        )

    @staticmethod
    def _style_before(request: SampleRequest, step: int) -> torch.Tensor:
        """
        Args:
          request (SampleRequest): the lane.
          step (int): first frame of the window about to be generated.

        Returns:
          torch.Tensor: (C,) the descriptor in force at `step`: the base style
            or the last schedule entry whose boundary lies at or before it.
        """
        if not request.style_schedule or request.style_period <= 0:
            return request.style
        segment = min(step // request.style_period, len(request.style_schedule))
        return request.style if segment == 0 else request.style_schedule[segment - 1]
