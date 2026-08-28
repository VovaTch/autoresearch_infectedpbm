"""Render original vs reconstructed audio samples from trained checkpoints.

Saves wavs to ./renders/ and prints per-sample CDPAM + multi-res STFT distance
so we can check whether the cdpam score corresponds to audible quality.
Eval/audio facts (from prepare.py): 44100 Hz mono, slice_length=32768 (~0.74s).
"""

from __future__ import annotations

import glob
import os

import torch
import torchaudio

from train import (
    build_learning_params,
    build_optimizer_cfg,
    build_scheduler_cfg,
    build_loss_aggregator,
    build_module,
)

SR = 44100
SLICE = 32768
SLICES_PER_CLIP = 14  # ~10.4s clips (14 * 32768 / 44100); sequential slices stitched
CACHE = os.path.expanduser("~/.cache/infected_pbm/slices")
OUT = os.path.join(os.path.dirname(__file__), "renders")

# 2026-08-28 cleanup: checkpoints were deleted for every lineage leg EXCEPT
# batch64, gancap10 and cont9h -- those three tags are the only ones that still
# render. The other entries are kept as the lineage record (and their
# tensorboard metrics survive under <dir>/lvl1_vqgan/); rendering them now
# raises FileNotFoundError.
CKPTS = {
    # tag -> checkpoint to render. cfg fields must match config.yaml's model block
    # (token_dim/hidden/num_rq/num_tokens/time_downsample/ze_norm). Paths point
    # at the best-on-monitor checkpoint (model_name + "_best.ckpt").
    # 2026-07-16 baseline reset: nonorm12h is the only surviving lineage
    # (all other checkpoints deleted; ear-confirmed best + all-time best
    # composite/cdpam/chroma).
    "nonorm12h": dict(
        path="saved_12h_nonorm/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
    ),
    # 2026-07-17: +12h continuation of nonorm12h (config_cont24h_nonorm.yaml).
    # All-time best every metric: 1.9146/0.0967/1.1437/0.0139.
    "nonorm24h": dict(
        path="saved_24h_nonorm/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
    ),
    # 2026-08-07: +48h continuation of nonorm24h (config_cont48h_resume.yaml,
    # relaunched after the 2026-08-05 power loss). Composite 1.8227 / mrstft
    # 1.0678 / chroma 0.0115 all beat nonorm24h; cdpam 0.0998 slightly worse.
    "nonorm48h": dict(
        path="saved_48h_nonorm_resume/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
    ),
    # 2026-08-10: +48h continuation of nonorm48h at 1/5 the peak LR (2e-5,
    # config_cont48h_lowlr.yaml). Test metrics essentially flat vs nonorm48h:
    # 1.8210/0.1015/1.0653/0.0115.
    "lowlr48h": dict(
        path="saved_48h_lowlr/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
    ),
    # 2026-08-14: +48h GAN leg from lowlr48h (config_cont48h_gan.yaml, cap 30).
    # cdpam IMPROVED 0.1015 -> 0.0993 while composite/mrstft/chroma regressed
    # (1.9268/1.1719/0.0143) -- the ear decides this one, not the metrics.
    "gan48h": dict(
        path="saved_48h_gan/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
    ),
    # 2026-08-16: 24h GAN anneal from gan48h (config_cont24h_gan_anneal.yaml,
    # cap 30->10, peak 2e-5). Repair pass for the transient artifacts; the
    # verdict is crest factor + HF retention, not the aggregate metrics
    # (1.9054/0.1023/1.1494/0.0137).
    "ganAnneal": dict(
        path="saved_24h_gan_anneal/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
    ),
    # 2026-08-18: 48h second anneal from ganAnneal (config_cont48h_gan_anneal2.yaml,
    # cap 10->5). 1.8800/0.1034/1.1236/0.0140.
    "ganAnneal2": dict(
        path="saved_48h_gan_anneal2/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
    ),
    # 2026-08-19: 24h per-level codebooks from ganAnneal2
    # (config_cont24h_perlevel.yaml). Each RQ level owns a 2048-entry table;
    # same bitrate, GAN off, peak 1e-4.
    "perlevel24h": dict(
        path="saved_24h_perlevel/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
        per_level_codebooks=True,
    ),
    # 2026-08-21: +24h per-level leg 2 (config_20260821_perlevel.yaml), single
    # variable = more wall clock. NEW ALL-TIME BEST 1.7691 / 0.1005 / 1.0245 /
    # 0.0107; align/commit 0.994 and dead codes 183/6144, both still falling.
    "perlevel48h": dict(
        path="saved_20260821_perlevel/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
        per_level_codebooks=True,
    ),
    # 2026-08-23: +24h per-level leg 3, batch 24 -> 64 (config_20260822_batch64.yaml).
    # Composite/mrstft/chroma/phase all all-time best (1.7572 / 1.0173 / 0.0099 /
    # 0.6151) but cdpam regressed to 0.1051. FIRST leg where best != last:
    # best-on-monitor froze at 03:10 = 12.6h of 24.
    "batch64": dict(
        path="saved_20260822_batch64/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
        per_level_codebooks=True,
    ),
    "batch64best": dict(
        path="saved_20260822_batch64/lvl1_vqgan_best.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
        per_level_codebooks=True,
    ),
    # 2026-08-25: 48h GAN re-acquisition from batch64 last (config_20260823_gan.yaml,
    # d_weight_cap 10, peak 1e-4, batch 48). First adversarial leg on the per-level
    # quantizer. Judge on the spectral table (spectral_table.py), not composite.
    "gancap10": dict(
        path="saved_20260823_gan/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
        per_level_codebooks=True,
    ),
    # 2026-08-26: 48h GAN drive reduction from gancap10 last, d_weight_cap 10 -> 5,
    # single variable (config_20260826_gancap5.yaml). Take last.ckpt: on any
    # adversarial leg best.ckpt is the epoch-0 save, because loss_monitor is
    # validation/total loss and the GAN inflates the generator term by construction.
    "gancap5": dict(
        path="saved_20260826_gancap5/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
        per_level_codebooks=True,
    ),
    # Same run, the min-total-loss save at epoch 48 (5.5h in). Unlike a cold
    # adversarial start, this leg warm-started with the GAN already converged,
    # so validation/total loss is NOT inflated from epoch 0 and best.ckpt lands
    # on a real minimum. Epoch 48 is the only point in the run that beat the
    # starting weights; last.ckpt (epoch 288) sits inside an align blowup.
    "gancap5best": dict(
        path="saved_20260826_gancap5/lvl1_vqgan_best.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
        per_level_codebooks=True,
    ),
    # 2026-08-28: 9h continuation from gancap5best (config_20260827_cont9h.yaml),
    # single variable = wall clock. Test composite 1.8496 / cdpam 0.1013 /
    # mrstft 1.1022 / chroma 0.0126. last.ckpt is safe here: its epoch (80) is
    # calm (align 1.430) AND is the run's cdpam argmin, i.e. still improving at
    # the cutoff.
    "cont9h": dict(
        path="saved_20260827_cont9h/lvl1_vqgan_last.ckpt",
        token_dim=1024, num_rq=3, num_tokens=2048, time_downsample=1, hidden=1024,
        per_level_codebooks=True,
    ),
}
# TAGS env var: comma-separated subset of CKPTS to render (default: all).
_only = [t for t in os.environ.get("TAGS", "").split(",") if t]
if _only:
    CKPTS = {t: CKPTS[t] for t in _only}
# tracks to render (substrings of slice filenames); one set of clips per track
TRACKS = ["deeply_disturbed", "Cookie_From_Space"]


def multi_res_stft_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean abs log-magnitude STFT distance over several resolutions (lower=better)."""
    tot = 0.0
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        wa = torch.stft(a.squeeze(0), n_fft, hop, return_complex=True).abs()
        wb = torch.stft(b.squeeze(0), n_fft, hop, return_complex=True).abs()
        tot += (torch.log1p(wa) - torch.log1p(wb)).abs().mean().item()
    return tot / 3.0


def _apply_ema(module, ckpt) -> bool:
    """Copy generator EMA weights from EMAOptimizer state into module.model.

    state_dict holds RAW weights; test-time metrics use EMA (the EMA callback swaps
    them in), so rendering raw weights misrepresents the model — match by copying
    the ema params (stored in model.parameters() order) over the generator."""
    gen_params = list(module.model.parameters())
    for st in ckpt.get("optimizer_states") or []:
        ema = st.get("ema") if isinstance(st, dict) else None
        if ema is not None and len(ema) == len(gen_params):
            with torch.no_grad():
                for p, e in zip(gen_params, ema):
                    p.copy_(e.to(dtype=p.dtype))
            return True
    return False


def load_module(ckpt_path: str, lp, oc, sc, la, token_dim: int = 1024, num_rq: int = 3, num_tokens: int = 2048, time_downsample: int = 1, hidden: int = 1024, ze_norm: str = "none", per_level_codebooks: bool = False):
    module = build_module(
        lp, la, oc, sc,
        token_dim=token_dim, num_rq_steps=num_rq, num_tokens=num_tokens,
        time_downsample=time_downsample, hidden=hidden, ze_norm=ze_norm,
        per_level_codebooks=per_level_codebooks,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    module.load_state_dict(sd, strict=True)
    if isinstance(ckpt, dict):
        print("  EMA weights applied" if _apply_ema(module, ckpt) else "  (raw weights, no EMA found)")
    module.eval()
    return module


def pick_clips(n_clips: int = 3) -> list[tuple[str, torch.Tensor]]:
    """Pick n clips from distinct positions (25/50/75%) of each track in TRACKS."""
    clips = []
    for substr in TRACKS:
        files = [
            f
            for f in sorted(glob.glob(os.path.join(CACHE, "*.pt")))
            if substr in os.path.basename(f)
        ]
        if not files:
            raise ValueError(f"No slice file matched '{substr}'")
        slices = torch.load(files[0], map_location="cpu")  # [N,1,SLICE]
        name = (
            os.path.splitext(os.path.basename(files[0]))[0].replace("slices_", "")[:40]
        )
        n = slices.shape[0]
        fracs = [(i + 1) / (n_clips + 1) for i in range(n_clips)]  # 0.25,0.5,0.75
        for fr in fracs:
            start = min(int(fr * n), n - SLICES_PER_CLIP)
            start -= start % SLICES_PER_CLIP
            clip = slices[start : start + SLICES_PER_CLIP].reshape(1, -1)
            clips.append((f"{name}_p{int(fr*100)}", clip))
    return clips


@torch.no_grad()
def reconstruct(module, clip: torch.Tensor) -> torch.Tensor:
    """clip [1, L] -> recon [1, L]; decoded in ONE pass, not slice-by-slice.

    The old path reshaped to [n,1,SLICE], decoded each slice independently and
    hard-concatenated, which put a step discontinuity every 0.743 s -- a 1.35 Hz
    click train in every rendered clip (sample-to-sample jump at the seams was
    5.11x the local mean, vs 1.08x for real audio). The model is fully
    convolutional + ISTFT with no length assumption, so decoding the whole clip
    at once removes it entirely (measured jump ratio -> 1.01). Training is
    unaffected either way: it feeds independent slices and never saw a seam.
    """
    out = module.model(clip.reshape(1, 1, -1))["slice"].reshape(1, -1)
    return out[:, : clip.shape[1]]


def loudness_match(
    variants: dict[str, torch.Tensor], ref: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Scale each variant to the reference rms, then apply one shared headroom scale.

    Peak-normalizing every file independently -- what this script used to do --
    silently biases the comparison: a reconstruction with sparse peak overshoot
    is divided by a larger number, so orig and recon end up at different levels.
    cdpam is level-sensitive, and the effect is not small: on identical audio,
    peak-normed scored 0.2153 against 0.0176 rms-matched. It penalizes exactly
    the high-crest models (the GAN ones), which is why per-clip render A/Bs
    disagreed with the aggregate test metrics for months. A quieter file also
    loses a blind ear A/B for reasons unrelated to fidelity.

    Args:
      variants (dict[str, torch.Tensor]): named waveforms (1, L).
      ref (torch.Tensor): reference waveform (1, L) whose rms is the target.

    Returns:
      dict[str, torch.Tensor]: rms-matched waveforms sharing one headroom scale.
    """
    target = ref.pow(2).mean().sqrt()
    out = {
        k: v * (target / (v.pow(2).mean().sqrt() + 1e-8)) for k, v in variants.items()
    }
    peak = max(float(v.abs().max()) for v in out.values())
    return {k: v / (peak + 1e-8) for k, v in out.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    lp = build_learning_params()
    oc = build_optimizer_cfg()
    sc = build_scheduler_cfg()
    la = build_loss_aggregator()

    print("Loading cdpam evaluator...")
    import cdpam
    evaluator = cdpam.CDPAM(dev="cpu")

    def cdpam_score(orig: torch.Tensor, rec: torch.Tensor) -> float:
        rs = torchaudio.transforms.Resample(SR, 22050)
        a = rs(orig.float()) * 32768.0
        b = rs(rec.float()) * 32768.0
        return float(evaluator.forward(a, b).mean().item())

    clips = pick_clips(int(os.environ.get("N_CLIPS", "3")))
    print(f"Picked {len(clips)} clips.\n")

    modules = {}
    for tag, cfg in CKPTS.items():
        path = cfg["path"]
        if os.path.exists(path):
            print(f"Loading {tag} <- {path} ({ {k: v for k, v in cfg.items() if k != 'path'} })")
            modules[tag] = load_module(
                path, lp, oc, sc, la,
                token_dim=cfg.get("token_dim", 1024),
                num_rq=cfg.get("num_rq", 3),
                num_tokens=cfg.get("num_tokens", 2048),
                time_downsample=cfg.get("time_downsample", 1),
                hidden=cfg.get("hidden", 1024),
                ze_norm=cfg.get("ze_norm", "none"),
                per_level_codebooks=cfg.get("per_level_codebooks", False),
            )
        else:
            print(f"SKIP {tag}: {path} missing")

    print(f"\n{'clip':<24}{'model':<16}{'cdpam':>10}{'mrstft':>10}")
    print("-" * 60)
    for i, (name, clip) in enumerate(clips):
        clip = clip / (clip.abs().max() + 1e-8)
        group = {"ORIG": clip}
        group.update({tag: reconstruct(m, clip) for tag, m in modules.items()})
        group = loudness_match(group, clip)  # equal rms: see loudness_match
        ref = group["ORIG"]
        for tag, rec in group.items():
            torchaudio.save(os.path.join(OUT, f"clip{i}_{name}_{tag}.wav"), rec, SR)
            if tag == "ORIG":
                continue
            L = min(ref.shape[1], rec.shape[1])
            cd = cdpam_score(ref[:, :L], rec[:, :L])
            mr = multi_res_stft_dist(ref[:, :L], rec[:, :L])
            print(f"{('clip'+str(i)+' '+name)[:23]:<24}{tag:<16}{cd:>10.4f}{mr:>10.4f}")
    print(f"\nWavs in: {OUT}")


if __name__ == "__main__":
    main()
