"""
Is the decoder's predicted STFT self-consistent, and is the VQ or the decoder
responsible for the crest-factor excess?

MEASURED FACT (recomputed from renders_copy_synthesis/, 4 tracks): the
encode->decode round trip RAISES crest factor by +3.50 dB (orig 10.37 ->
recon 13.86). Higher crest on brickwalled masters means the sustained body
between transients is being lost, not that transients got sharper. GAN legs
pushed crest further the same wrong way (14.44 -> 15.33 dB, past the
pre-registered 15.0 FAIL line) with rms -12%, which the user heard as a
loudness drop -- so re-enabling the GAN attacks this from the wrong side.

HYPOTHESIS. TemporalDecoder predicts a RAW complex STFT (513 real + 513 imag
per frame) and hands it to a plain weighted-overlap-add ISTFT at hop 256 /
n_fft 1024 -- 4x redundancy with NOTHING forcing the prediction to be the STFT
of any real signal. Where neighbouring frames disagree, overlap-add cancels.
Cancellation removes steady-state energy (many overlapping frames must agree to
sustain a tone) while leaving isolated transients intact => crest UP, centroid
DOWN. Both are what the measurements show.

This probe tests it directly, and separates the two possible culprits:

  consistency   ||STFT(ISTFT(S_hat)) - S_hat|| / ||S_hat||, on the decoder's own
                prediction. Real audio's STFT is consistent by construction, so
                the same figure on the ORIGINAL is the floor (~0). A large gap
                confirms the mechanism.
  bypass_vq     decode(z_e) with the quantizer out of the path. If the crest
                excess survives with continuous latents, the 5.68 kbps token
                budget is NOT the cause and it is the decoder/loss; if it
                vanishes, no loss change can help and the budget is the wall.

Diagnostic only -- trains nothing, writes no checkpoint.

Usage:
  uv run python probe_consistency.py
  uv run python probe_consistency.py --ckpt saved_20260827_cont9h/lvl1_vqgan_last.ckpt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from render_samples import load_module, pick_clips
from train import (
    build_learning_params,
    build_loss_aggregator,
    build_optimizer_cfg,
    build_scheduler_cfg,
)

SR = 44100
N_FFT = 1024
HOP = 256
SHIFT = 128  # train.ISTFT(padding='same') output lag vs torch.stft(center=True)


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="saved_20260827_cont9h/lvl1_vqgan_last.ckpt")
    p.add_argument("--clips", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def crest_db(x: torch.Tensor) -> float:
    """
    Args:
      x (torch.Tensor): (..., L) waveform.

    Returns:
      float: peak-to-rms ratio in dB.
    """
    peak = x.abs().max()
    rms = x.float().pow(2).mean().sqrt()
    return float(20.0 * torch.log10(peak / (rms + 1e-12)))


def centroid(x: torch.Tensor) -> float:
    """
    Args:
      x (torch.Tensor): (1, L) waveform.

    Returns:
      float: energy-weighted mean frequency, normalized to Nyquist.
    """
    spec = torch.stft(
        x.squeeze(0).float(),
        N_FFT,
        HOP,
        window=torch.hann_window(N_FFT, device=x.device),
        return_complex=True,
    ).abs()
    freqs = torch.linspace(0.0, 1.0, spec.shape[0], device=x.device)[:, None]
    return float((spec * freqs).sum() / (spec.sum() + 1e-12))


def consistency(spec: torch.Tensor, y: torch.Tensor) -> float:
    """
    Relative L1 gap between a complex spectrogram and the STFT of its own ISTFT.

    Zero means spec lies in the range of the STFT operator (it is the transform
    of a real signal); large means overlap-add is cancelling disagreements
    between neighbouring frames.

    train.ISTFT(padding="same") is a correct inverse of torch.stft(center=True)
    but its output is shifted +SHIFT samples and 2*HOP longer (verified exactly:
    best-alignment rel_err 0.000000 at scale 1.0000). The shift must be undone
    before re-analysis or the half-hop misalignment alone reads as ~1.24
    inconsistency on REAL audio.

    Args:
      spec (torch.Tensor): (1, F, T) complex spectrogram the decoder predicted.
      y (torch.Tensor): (1, L) waveform the decoder's ISTFT produced from it.

    Returns:
      float: ||STFT(y) - spec||_1 / ||spec||_1.
    """
    back = torch.stft(
        y.squeeze(0).float()[SHIFT:],
        N_FFT,
        HOP,
        window=torch.hann_window(N_FFT, device=y.device),
        return_complex=True,
        center=True,
    )
    t = min(back.shape[-1], spec.shape[-1])
    a, b = back[..., :t], spec[0, ..., :t]
    return float((a - b).abs().sum() / (b.abs().sum() + 1e-12))


@torch.no_grad()
def decode_traced(
    model: torch.nn.Module, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run TemporalDecoder step by step so the intermediate STFT is observable.

    Mirrors TemporalDecoder.forward exactly (train.py:1364).

    Args:
      model (torch.nn.Module): the MultiLvlVQVariationalAutoEncoder.
      z (torch.Tensor): (B, token_dim, T_lat) decoder input.

    Returns:
      tuple: (spec (B,F,T) complex, y_istft (B,1,L), y_final (B,1,L)).
    """
    d = model.decoder
    h = d._proj_in(z)
    h = d._res1(h)
    h = d._up(h)
    h = d._res2(h)
    spec = d._proj_spec(h).float()
    bins = d._spec_bins
    cspec = torch.complex(spec[:, :bins], spec[:, bins:])
    y = d._istft(cspec).unsqueeze(1)
    return cspec, y, d._end_conv(y).float()


def main() -> None:
    args = parse_args()
    dev = torch.device(args.device)
    module = load_module(
        args.ckpt,
        build_learning_params(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        build_loss_aggregator(),
        per_level_codebooks=True,
    )
    model = module.model.to(dev).eval()

    clips = pick_clips(args.clips)
    print(f"\ncheckpoint  {args.ckpt}")
    print(f"clips       {len(clips)}   n_fft {N_FFT} hop {HOP} (4x overlap)\n")

    rows: list[tuple[str, float, float, float, float, float, float, float]] = []
    for name, clip in clips:
        x = (clip / (clip.abs().max() + 1e-8)).to(dev)
        z_e = model.encode(x.reshape(1, 1, -1))
        z_q = model.vq_module(z_e, extract_losses=True)["v_q"][:, -1, ...]

        _, _, y_q = decode_traced(model, z_q)
        cspec_q, y_istft_q, _ = decode_traced(model, z_q)
        _, _, y_e = decode_traced(model, z_e)

        rec = y_q.reshape(1, -1)[:, : x.shape[1]]
        byp = y_e.reshape(1, -1)[:, : x.shape[1]]

        # floor: a real signal's own STFT is consistent by construction
        xs = torch.stft(
            x.squeeze(0).float(),
            N_FFT,
            HOP,
            window=torch.hann_window(N_FFT, device=dev),
            return_complex=True,
        )
        # control through the decoder's OWN overlap-add, so the floor measures
        # the operator the decoder actually uses
        x_ola = model.decoder._istft(xs[None])

        rows.append(
            (
                name,
                crest_db(x),
                crest_db(rec),
                crest_db(byp),
                centroid(x),
                centroid(rec),
                consistency(xs[None], x_ola),
                consistency(cspec_q, y_istft_q.reshape(1, -1)),
            )
        )

    hdr = (
        f"{'clip':<26}{'crestA':>8}{'crestVQ':>9}{'crestZe':>9}"
        f"{'centA':>8}{'centVQ':>8}{'consREAL':>10}{'consDEC':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r[0][:25]:<26}{r[1]:8.2f}{r[2]:9.2f}{r[3]:9.2f}"
            f"{r[4]:8.3f}{r[5]:8.3f}{r[6]:10.4f}{r[7]:9.4f}"
        )
    a = np.array([r[1:] for r in rows], dtype=np.float64)
    m = a.mean(axis=0)
    print("-" * len(hdr))
    print(
        f"{'mean':<26}{m[0]:8.2f}{m[1]:9.2f}{m[2]:9.2f}"
        f"{m[3]:8.3f}{m[4]:8.3f}{m[5]:10.4f}{m[6]:9.4f}"
    )
    print(
        f"\ncrest excess: VQ path {m[1] - m[0]:+.2f} dB, "
        f"z_e bypass {m[2] - m[0]:+.2f} dB  "
        f"(quantizer's share: {m[1] - m[2]:+.2f} dB)"
    )
    print(
        f"inconsistency: decoder {m[6]:.4f} vs real-audio floor {m[5]:.4f} "
        f"= {m[6] / max(m[5], 1e-9):.1f}x"
    )


if __name__ == "__main__":
    main()
