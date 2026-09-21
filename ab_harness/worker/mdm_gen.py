"""
Masked-diffusion backend: a train_mdm checkpoint behind the SampleSource seam.

Each window is one MaskGIT fill of the (W, R) grid: prompt or carried-over
frames sit unmasked, everything else starts as MASK and is resolved level by
level in the configured number of rounds under the lane's id, style and
guidance. Temperature and top_k/top_p from the SampleRequest drive the
per-cell draws exactly as they drive the AR sampler; co_tracks is ignored.
"""

from __future__ import annotations

from typing import Callable

import torch

from ab_harness.config import GeneratorCfg
from ab_harness.worker.windowed import WindowBatch, WindowedGenerator
from train_mdm import MaskCfg, MaskedDenoiser


class MdmGenerator(WindowedGenerator):
    """
    Args:
      net (MaskedDenoiser): trained denoiser, on `device`, in eval mode.
      mask_cfg (MaskCfg): the checkpoint's sampler defaults (schedule).
      device (torch.device): compute device.
      window (int): frames per window (the AR crop it was trained on).
      prefix_max_frac (float): longest trained prefix, as a window fraction.
      gen_cfg (GeneratorCfg): mdm_steps, mdm_choice_temperature, reprime_frac.
    """

    def __init__(
        self,
        net: MaskedDenoiser,
        mask_cfg: MaskCfg,
        device: torch.device,
        window: int,
        prefix_max_frac: float,
        gen_cfg: GeneratorCfg,
    ) -> None:
        self.net = net
        self.mask_cfg = MaskCfg(**vars(mask_cfg))
        self.mask_cfg.steps = list(gen_cfg.mdm_steps)
        self.mask_cfg.choice_temperature = gen_cfg.mdm_choice_temperature
        if len(self.mask_cfg.steps) != net.num_rq:
            raise ValueError(
                f"mdm_steps needs {net.num_rq} entries, got {self.mask_cfg.steps}"
            )
        super().__init__(
            device,
            window,
            net.num_rq,
            net.memory.num_tracks,
            net.mask_id,
            gen_cfg.reprime_frac,
            int(prefix_max_frac * window) if prefix_max_frac > 0 else None,
        )

    def rounds_per_window(self) -> int:
        """
        Returns:
          int: MaskGIT rounds per window, summed over levels.
        """
        return sum(self.mask_cfg.steps)

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
        # lanes share a forward pass only when their draw knobs agree
        groups: dict[tuple[float, int, float], list[int]] = {}
        for row, r in enumerate(batch.requests):
            groups.setdefault((r.temperature, r.top_k, r.top_p), []).append(row)
        out = torch.zeros_like(batch.prefix)
        rounds = self.rounds_per_window()
        for (temperature, top_k, top_p), rows in groups.items():
            idx = torch.tensor(rows, device=self.device)
            cfg: float | torch.Tensor = batch.cfg[idx]
            if bool((batch.cfg[idx] == 1.0).all()):
                cfg = 1.0
            out[idx] = self.net.sample(
                self.window,
                batch.track_idx[idx],
                batch.style[idx],
                self.mask_cfg,
                prefix=batch.prefix[idx],
                prefix_frames=batch.prefix_frames[idx],
                cfg_scale=cfg,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=[batch.generators[i] for i in rows],
                progress=lambda done, _total: progress(min(done, rounds)),
                drop_id=batch.drop_id[idx],
                drop_style=batch.drop_style[idx],
            )
        return out
