"""
Latent-flow backend: a train_zflow checkpoint behind the SampleSource seam.

Each window is one flow integration in the whitened z_q space: the prompt (or
the previous window's tail) is embedded through the codebooks, whitened and
held clean while noise elsewhere is integrated to t=1 under the lane's track
id, style and guidance; the result is requantised to RVQ tokens by beam search
(train_zflow.requantize). The prompt frames themselves are returned verbatim,
never round-tripped.

SampleRequest knobs that mean nothing here -- temperature, top_k, top_p,
co_tracks -- are ignored; the flow's own knobs (Euler steps, churn, beam)
come from GeneratorCfg. Legacy adaLN checkpoints (saved_zflow*, trained
without ids or prefixes) still run: the id stream is absent and the prefix is
enforced by replacement only, which joins less cleanly.
"""

from __future__ import annotations

from typing import Callable

import torch

from ab_harness.config import GeneratorCfg
from ab_harness.worker.windowed import WindowBatch, WindowedGenerator
from train_zflow import ZFlowModule, embed_zq, requantize


class ZFlowGenerator(WindowedGenerator):
    """
    Args:
      module (ZFlowModule): trained flow, on `device`, in eval mode.
      device (torch.device): compute device.
      window (int): frames per window (the AR crop it was trained on).
      num_tracks (int): id vocabulary of the corpus.
      gen_cfg (GeneratorCfg): flow_steps, flow_churn, requantize_beam and
        reprime_frac.
    """

    def __init__(
        self,
        module: ZFlowModule,
        device: torch.device,
        window: int,
        num_tracks: int,
        gen_cfg: GeneratorCfg,
    ) -> None:
        model = module.cfg.model
        self.module = module
        self.steps = gen_cfg.flow_steps
        self.churn = gen_cfg.flow_churn
        self.beam = gen_cfg.requantize_beam
        self.patch = model.patch
        self.xattn = module.net.memory is not None
        window -= window % self.patch
        max_prefix = int(model.prefix_max_frac * window) if module.net.prefix else None
        super().__init__(
            device,
            window,
            int(module.codebooks.shape[0]),  # type: ignore[arg-type]
            num_tracks,
            int(module.codebooks.shape[1]),  # type: ignore[arg-type]
            gen_cfg.reprime_frac,
            max_prefix,
        )

    def rounds_per_window(self) -> int:
        """
        Returns:
          int: Euler steps per window.
        """
        return self.steps

    def fill_window(
        self, batch: WindowBatch, progress: Callable[[int], None]
    ) -> torch.Tensor:
        """
        Args:
          batch (WindowBatch): prefixes and conditionals of the active lanes.
          progress (Callable[[int], None]): rounds done so far.

        Returns:
          torch.Tensor: (B, W, R) int64 tokens; prefix frames as given.
        """
        module, size = self.module, batch.prefix.shape[0]
        stats = module.stats
        z = embed_zq(batch.prefix, module.codebooks)  # type: ignore[arg-type]
        prefix = stats.whiten(z.transpose(1, 2))
        # the clamp is patch-aligned; the true prefix is restored below
        held = torch.arange(self.window, device=self.device).unsqueeze(0) < (
            batch.prefix_frames - batch.prefix_frames % self.patch
        ).unsqueeze(1)
        noise = torch.stack(
            [
                torch.randn(
                    prefix.shape[1:],
                    generator=g,
                    device=self.device,
                    dtype=prefix.dtype,
                )
                for g in batch.generators
            ]
        )
        ids = batch.track_idx if self.xattn else None
        cfg: float | torch.Tensor = batch.cfg
        if bool((batch.cfg == 1.0).all()):
            cfg = 1.0
        x = module.sample(
            noise,
            0.0,
            batch.style,
            steps=self.steps,
            churn=self.churn,
            cfg_scale=cfg,
            generator=batch.generators[0] if self.churn > 0 else None,
            track_idx=ids,
            prefix=prefix,
            prefix_mask=held,
            drop_id=batch.drop_id,
            drop_style=batch.drop_style,
        )
        progress(self.steps)
        tokens = requantize(stats.unwhiten(x), module.codebooks, beam=self.beam)  # type: ignore[arg-type]
        keep = torch.arange(self.window, device=self.device).unsqueeze(0) < (
            batch.prefix_frames.unsqueeze(1)
        )
        return torch.where(keep.unsqueeze(-1), batch.prefix, tokens).reshape(
            size, self.window, -1
        )
