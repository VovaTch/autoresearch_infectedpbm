"""Re-render the GAN checkpoint with the two post-hoc artifact fixes.

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

Variants written per clip: ORIG, gan_sliced (the old path, for reference),
gan_full (fix 1), gan_lim (fix 1 + fix 2).

Env: N_CLIPS (default 3 per track), LIM_K (knee in units of rms, default 4.0),
LIM_CEIL (ceiling as a multiple of the knee, default 1.15), CKPT, TAG.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torchaudio

from render_samples import (
    OUT,
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

CKPT = os.environ.get("CKPT", "saved_20260827_cont9h/lvl1_vqgan_last.ckpt")
TAG = os.environ.get("TAG", "gan")
LIM_K = float(os.environ.get("LIM_K", "4.0"))
LIM_CEIL = float(os.environ.get("LIM_CEIL", "1.15"))


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


def peak_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.abs().max() + 1e-8)


def loudness_match(variants: dict[str, torch.Tensor], ref: torch.Tensor) -> dict[str, torch.Tensor]:
    """Scale every variant to the reference rms, then apply one shared headroom scale.

    Peak-normalizing each variant independently is what made the old renders
    unusable: a reconstruction with sparse peak overshoot gets scaled down
    harder than the reference, so orig and recon end up at different levels.
    cdpam is level-sensitive (0.2153 peak-normed vs 0.0176 rms-matched on the
    same audio), and a quieter file also loses a blind ear A/B for reasons that
    have nothing to do with fidelity. Matching rms first and then dividing
    everything by a single scalar keeps the comparison honest and still
    guarantees nothing clips.

    Args:
      variants (dict[str, torch.Tensor]): named waveforms (1, L).
      ref (torch.Tensor): reference waveform (1, L) to match rms to.

    Returns:
      dict[str, torch.Tensor]: rms-matched waveforms sharing one headroom scale.
    """
    target = ref.pow(2).mean().sqrt()
    out = {k: v * (target / (v.pow(2).mean().sqrt() + 1e-8)) for k, v in variants.items()}
    peak = max(float(v.abs().max()) for v in out.values())
    return {k: v / (peak + 1e-8) for k, v in out.items()}


def stats(x: torch.Tensor) -> tuple[float, float, float]:
    """Returns (crest_dB, rms, boundary/body jump ratio) of a peak-normed clip."""
    a = x.reshape(-1).double().numpy()
    rms = float(np.sqrt((a**2).mean()))
    crest = 20.0 * np.log10(np.abs(a).max() / rms)
    d = np.abs(np.diff(a))
    n = len(a) // SLICE
    bnd = [j * SLICE - 1 for j in range(1, n) if j * SLICE - 1 < len(d)]
    if not bnd:
        return crest, rms, float("nan")
    mask = np.ones(len(d), bool)
    mask[bnd] = False
    return crest, rms, float(d[bnd].mean() / d[mask].mean())


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
    module = load_module(CKPT, lp, oc, sc, la, per_level_codebooks=True).cuda().eval()
    print(f"Limiter: knee {LIM_K:.2f}x rms, ceiling {LIM_K * LIM_CEIL:.2f}x rms\n")

    hdr = f"{'clip':<24}{'variant':<12}{'cdpam':>9}{'mrstft':>9}{'crest_dB':>10}{'rms':>8}{'bnd_jump':>10}"
    print(hdr)
    print("-" * len(hdr))
    acc: dict[str, list[tuple[float, ...]]] = {}

    for i, (name, clip) in enumerate(clips):
        clip = peak_norm(clip)
        full = decode_full(module, clip)
        ref = clip[:, : full.shape[1]]

        group = {
            "ORIG": ref,
            "gan_full": full,
            "gan_lim": soft_limit(full),
            "gan_limsrc": soft_limit(full, ceil=1.05, ref=ref),
        }
        group = loudness_match(group, ref)
        clip = group["ORIG"]

        for vname, rec in group.items():
            suffix = "ORIG" if vname == "ORIG" else f"{TAG}_{vname}"
            torchaudio.save(os.path.join(OUT, f"clip{i}_{name}_{suffix}.wav"), rec, SR)
            L = min(clip.shape[1], rec.shape[1])
            cd = 0.0 if vname == "ORIG" else cdpam_score(clip[:, :L], rec[:, :L])
            mr = 0.0 if vname == "ORIG" else multi_res_stft_dist(clip[:, :L], rec[:, :L])
            cr, rm, bj = stats(rec)
            acc.setdefault(vname, []).append((cd, mr, cr, rm, bj))
            print(
                f"{('clip' + str(i) + ' ' + name)[:23]:<24}{vname:<12}"
                f"{cd:>9.4f}{mr:>9.4f}{cr:>10.2f}{rm:>8.4f}{bj:>10.2f}"
            )

    print("\n" + "=" * len(hdr))
    print(f"{'MEAN':<24}{'variant':<12}{'cdpam':>9}{'mrstft':>9}{'crest_dB':>10}{'rms':>8}{'bnd_jump':>10}")
    for vname, rows in acc.items():
        m = np.mean(rows, axis=0)
        print(
            f"{'':<24}{vname:<12}{m[0]:>9.4f}{m[1]:>9.4f}{m[2]:>10.2f}{m[3]:>8.4f}{m[4]:>10.2f}"
        )
    print(f"\nWavs in: {OUT}")


if __name__ == "__main__":
    main()
