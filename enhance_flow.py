"""
Run a trained flow enhancer over a wav file.

The model is a post-processor for the tokenizer's decoder: give it a decoded
waveform and it integrates the bridge flow from that round trip toward the clean
manifold. Audio longer than one crop is processed in half-overlapping windows and
crossfaded; the sampler is deterministic, so the windows agree where they meet.

Usage:
    uv run python enhance_flow.py --checkpoint saved_flow/flow_latest.ckpt \
        --input in.wav --out out.wav --steps 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from train_flow import FlowModule, enhance_long, load_flow_module

REPO = Path(__file__).resolve().parent


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    """
    Read a file as mono at the model's rate.

    Args:
      path (Path): audio file.
      sample_rate (int): target rate.

    Returns:
      torch.Tensor: (1, 1, L) float32 waveform.
    """
    wav, rate = torchaudio.load(str(path))
    if rate != sample_rate:
        wav = torchaudio.transforms.Resample(rate, sample_rate)(wav)
    return wav.mean(dim=0, keepdim=True).unsqueeze(0).float()


def enhance(
    module: FlowModule,
    wav: torch.Tensor,
    steps: int | None = None,
    project_steps: bool | None = None,
    win_frames: int | None = None,
    sigma: float | None = None,
) -> torch.Tensor:
    """
    Enhance one waveform, trimmed to the frame grid the STFT needs.

    Args:
      module (FlowModule): trained model on the target device.
      wav (torch.Tensor): (1, 1, L) round-trip waveform.
      steps (int | None): Euler steps, config default when None.
      project_steps (bool | None): per-step projection, config default when None.
      win_frames (int | None): window length in frames for long audio; the
        training crop when None.
      sigma (float | None): bridge-noise scale, config default when None.

    Returns:
      torch.Tensor: (1, 1, L') enhanced waveform, L' <= L.
    """
    geom = module.geom
    frames = wav.shape[-1] // geom.hop + 1
    frames -= frames % 8  # three UNet downsamples need a multiple of 8 frames
    if frames < 8:
        raise ValueError(f"input is too short: {wav.shape[-1]} samples")
    trimmed = wav[..., : geom.crop_samples(frames)]
    return enhance_long(module, trimmed, win_frames, steps, project_steps, sigma)


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--win-frames", type=int, default=None)
    parser.add_argument(
        "--sigma", type=float, default=None, help="bridge-noise scale at sampling time"
    )
    parser.add_argument(
        "--no-project", action="store_true", help="skip the per-step STFT projection"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Load the model, enhance the input file, write the output."""
    args = parse_args()
    module = load_flow_module(args.checkpoint, args.device)
    wav = load_mono(Path(args.input), module.geom.sample_rate).to(args.device)
    out = enhance(
        module,
        wav,
        steps=args.steps,
        project_steps=False if args.no_project else None,
        win_frames=args.win_frames,
        sigma=args.sigma,
    )
    torchaudio.save(args.out, out[0].cpu(), module.geom.sample_rate)
    print(
        f"{args.input} -> {args.out}  "
        f"{out.shape[-1] / module.geom.sample_rate:.2f}s  "
        f"steps={args.steps or module.cfg.flow.steps}"
    )


if __name__ == "__main__":
    main()
