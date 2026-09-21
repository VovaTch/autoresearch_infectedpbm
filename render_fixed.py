"""Render one checkpoint through the candidate peak-outlier treatments.

2026-09-21 rework. The listening paths (ab_harness LUFS+peak guard,
render_samples shared headroom) scale a clip down whenever its loudest sample
would clip, so a dozen drum-hit overshoot samples pull the whole body down
~1.5 dB. This script compares that guard against bounding the decoder output
instead, at inference (no retraining):

  guard    rms-matched, then /peak     -- the current playback behaviour
  hardclip rms-matched, clamp to +-1   -- what an int16 sink does anyway
  tanh     tanh(decoder output) at native level, then rms-matched
  lim      soft knee at LIM_K x rms (tanh above), then rms-matched
  knee     rms-matched, tanh knee from KNEE_THR (full-scale) up to 1.0

Every variant is rms-matched to ORIG (peak-normed source) so cdpam compares
equal-loudness bodies; `peak` is the value the guard would have divided by.

Older notes, still true of the decoder:

Two artifacts were measured on the 2026-08-25 gancap10 renders:

1. Slice-boundary clicks. render_samples.reconstruct decodes independent
   32768-sample slices and hard-concatenates them, so every 0.743 s carries a
   step discontinuity (sample-to-sample jump 4.9x the local mean, vs 1.1x for
   real audio). The model is fully convolutional + ISTFT and has no length
   assumption, so decoding the whole clip in one pass removes it entirely
   (jump ratio 4.14 -> 1.06 measured).
2. Sparse peak overshoot. The excursion distribution matches the original up
   to ~4x rms; the entire crest gap lives in the top 0.01% of samples. Since
   listening renders are peak-normalized, a dozen outlier samples drag the
   whole clip down ~12% in rms -- the audible "volume is lower".

Env: N_CLIPS (default 3 per track), LIM_K (knee in units of rms, default 4.0),
LIM_CEIL (ceiling as a multiple of the knee, default 1.15), CKPT, TAG.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torchaudio

from render_samples import (
    SLICE,
    SR,
    build_learning_params,
    build_loss_aggregator,
    build_optimizer_cfg,
    build_scheduler_cfg,
    load_module,
    multi_res_stft_dist,
    pick_clips,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders_limiter")
CKPT = os.environ.get("CKPT", "saved_20260914_dsteps2_24h/lvl1_vqgan_last.ckpt")
TAG = os.environ.get("TAG", "d24")
LIM_K = float(os.environ.get("LIM_K", "4.0"))
LIM_CEIL = float(os.environ.get("LIM_CEIL", "1.15"))
KNEE_THR = float(os.environ.get("KNEE_THR", "0.8"))
OUTPUT_ACT = os.environ.get("OUTPUT_ACT", "none")  # match the checkpoint's model.output_act


def crest_ratio(x: torch.Tensor) -> float:
    """Peak-to-rms ratio (linear, not dB) of a waveform."""
    return float(x.abs().max() / x.pow(2).mean().sqrt())


def soft_limit(
    x: torch.Tensor, k: float = LIM_K, ceil: float = LIM_CEIL, ref: torch.Tensor | None = None
) -> torch.Tensor:
    """Soft-clip samples above k*rms into a tanh knee; leaves the body untouched.

    A fixed k drives every clip to the same crest, which mangles sources that
    are genuinely peaky (clip6's original crests at 16.7 dB, so a 4x knee is
    clipping the music, not the model's overshoot). Passing `ref` sets the knee
    from the reference's own peak-to-rms instead, so only overshoot beyond what
    the source does is touched.

    Args:
      x (torch.Tensor): waveform (1, L).
      k (float): knee threshold in units of the signal rms; ignored if ref given.
      ceil (float): hard ceiling as a multiple of the knee.
      ref (torch.Tensor | None): reference waveform to match the crest of.

    Returns:
      torch.Tensor: limited waveform (1, L).
    """
    rms = x.pow(2).mean().sqrt()
    if ref is not None:
        k = crest_ratio(ref)
    thr = k * rms
    head = thr * (ceil - 1.0)
    mag = x.abs()
    over = mag > thr
    out = x.clone()
    out[over] = torch.sign(x[over]) * (thr + head * torch.tanh((mag[over] - thr) / head))
    return out


@torch.no_grad()
def decode_sliced(module, clip: torch.Tensor) -> torch.Tensor:
    """Old path: independent slices, hard concatenation."""
    n = clip.shape[1] // SLICE
    batch = clip[:, : n * SLICE].reshape(n, 1, SLICE)
    return module.model(batch.cuda())["slice"].reshape(1, -1).cpu()


@torch.no_grad()
def decode_full(module, clip: torch.Tensor) -> torch.Tensor:
    """One pass over the whole clip; no boundaries to stitch."""
    out = module.model(clip.reshape(1, 1, -1).cuda())["slice"].reshape(1, -1).cpu()
    return out[:, : clip.shape[1]]


def soft_knee(x: torch.Tensor, thr: float = KNEE_THR, ceil: float = 1.0) -> torch.Tensor:
    """Absolute-level soft clipper: identity below thr, tanh knee saturating at ceil.

    Unlike soft_limit the knee is in full-scale units, not rms units, so it is
    deployable without a reference and its ceiling is a real ceiling.

    Args:
      x (torch.Tensor): waveform (1, L).
      thr (float): knee start in full-scale units.
      ceil (float): asymptotic ceiling in full-scale units.

    Returns:
      torch.Tensor: limited waveform (1, L), |out| < ceil.
    """
    head = ceil - thr
    mag = x.abs()
    over = mag > thr
    out = x.clone()
    out[over] = torch.sign(x[over]) * (thr + head * torch.tanh((mag[over] - thr) / head))
    return out


def peak_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.abs().max() + 1e-8)


def rms_match(variants: dict[str, torch.Tensor], ref: torch.Tensor) -> dict[str, torch.Tensor]:
    """Scale every variant to the reference rms. No headroom scale on purpose.

    The shared headroom divide is exactly the guard under test, so it is applied
    explicitly by the "guard" variant instead of to the whole group.

    Args:
      variants (dict[str, torch.Tensor]): named waveforms (1, L).
      ref (torch.Tensor): reference waveform (1, L) to match rms to.

    Returns:
      dict[str, torch.Tensor]: rms-matched waveforms.
    """
    target = ref.pow(2).mean().sqrt()
    return {k: v * (target / (v.pow(2).mean().sqrt() + 1e-8)) for k, v in variants.items()}


def stats(x: torch.Tensor) -> tuple[float, float, float, float]:
    """Returns (crest_dB, rms, peak, ppm of samples with |x| > 1)."""
    a = x.reshape(-1).double().numpy()
    rms = float(np.sqrt((a**2).mean()))
    peak = float(np.abs(a).max())
    over = float((np.abs(a) > 1.0).mean() * 1e6)
    return 20.0 * np.log10(peak / rms), rms, peak, over


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    lp, oc, sc, la = (
        build_learning_params(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        build_loss_aggregator(),
    )

    print("Loading cdpam evaluator...")
    import cdpam

    evaluator = cdpam.CDPAM(dev="cpu")
    resample = torchaudio.transforms.Resample(SR, 22050)

    def cdpam_score(orig: torch.Tensor, rec: torch.Tensor) -> float:
        return float(
            evaluator.forward(
                resample(orig.float()) * 32768.0, resample(rec.float()) * 32768.0
            )
            .mean()
            .item()
        )

    clips = pick_clips(int(os.environ.get("N_CLIPS", "3")))
    print(f"Picked {len(clips)} clips.")
    print(f"Loading {TAG} <- {CKPT}")
    module = load_module(
        CKPT, lp, oc, sc, la, per_level_codebooks=True, output_act=OUTPUT_ACT
    ).cuda().eval()
    print(f"Limiter: knee {LIM_K:.2f}x rms, ceiling {LIM_K * LIM_CEIL:.2f}x rms\n")

    cols = ("cdpam", "mrstft", "crest_dB", "rms", "peak", "over_ppm")
    hdr = f"{'clip':<24}{'variant':<10}" + "".join(f"{c:>10}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    acc: dict[str, list[tuple[float, ...]]] = {}

    for i, (name, clip) in enumerate(clips):
        clip = peak_norm(clip)
        full = decode_full(module, clip)
        ref = clip[:, : full.shape[1]]

        # tanh/lim act on the decoder's native-level output, as a baked-in
        # output nonlinearity would; rms matching comes after.
        group = rms_match(
            {
                "ORIG": ref,
                "raw": full,
                "tanh": torch.tanh(full),
                "lim": soft_limit(full),
            },
            ref,
        )
        raw = group.pop("raw")
        group["guard"] = raw / (raw.abs().max() + 1e-8)
        group["hardclip"] = raw.clamp(-1.0, 1.0)
        group["knee"] = soft_knee(raw)
        _, _, rpk, rover = stats(raw)
        print(f"  raw decoder output (rms-matched): peak {rpk:.3f}, over_ppm {rover:.1f}")
        orig = group["ORIG"]

        for vname, rec in group.items():
            suffix = "ORIG" if vname == "ORIG" else f"{TAG}_{vname}"
            torchaudio.save(os.path.join(OUT, f"clip{i}_{name}_{suffix}.wav"), rec, SR)
            L = min(orig.shape[1], rec.shape[1])
            cd = 0.0 if vname == "ORIG" else cdpam_score(orig[:, :L], rec[:, :L])
            mr = 0.0 if vname == "ORIG" else multi_res_stft_dist(orig[:, :L], rec[:, :L])
            row = (cd, mr, *stats(rec))
            acc.setdefault(vname, []).append(row)
            print(
                f"{('clip' + str(i) + ' ' + name)[:23]:<24}{vname:<10}"
                + "".join(f"{v:>10.4f}" for v in row)
            )

    print("\n" + "=" * len(hdr))
    print(f"{'MEAN':<24}{'variant':<10}" + "".join(f"{c:>10}" for c in cols))
    for vname, rows in acc.items():
        m = np.mean(rows, axis=0)
        print(f"{'':<24}{vname:<10}" + "".join(f"{v:>10.4f}" for v in m))
    print(f"\nWavs in: {OUT}")


if __name__ == "__main__":
    main()
