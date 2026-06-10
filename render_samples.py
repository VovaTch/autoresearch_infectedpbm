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
SLICES_PER_CLIP = 4  # ~2.97s clips
CACHE = os.path.expanduser("~/.cache/infected_pbm/slices")
OUT = os.path.join(os.path.dirname(__file__), "renders")

CKPTS = {
    # tag: (ckpt_path, token_dim, latent_grid) -- must match the run's arch for strict load.
    "fixA_grid4": ("saved_cookie_fixA/last.ckpt", 512, 4),
    "fixB_grid8": ("saved_cookie_fixB/last.ckpt", 512, 8),
}
TRACK_SUBSTR = "Cookie_From_Space"  # restrict clips to this song (slice filenames use underscores)


def multi_res_stft_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean abs log-magnitude STFT distance over several resolutions (lower=better)."""
    tot = 0.0
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        wa = torch.stft(a.squeeze(0), n_fft, hop, return_complex=True).abs()
        wb = torch.stft(b.squeeze(0), n_fft, hop, return_complex=True).abs()
        tot += (torch.log1p(wa) - torch.log1p(wb)).abs().mean().item()
    return tot / 3.0


def load_module(ckpt_path: str, lp, oc, sc, la, token_dim: int = 512, latent_grid: int = 4):
    module = build_module(lp, la, oc, sc, token_dim=token_dim, latent_grid=latent_grid)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    module.load_state_dict(sd, strict=True)
    module.eval()
    return module


def pick_clips(n_clips: int = 3) -> list[tuple[str, torch.Tensor]]:
    """Pick n clips from distinct positions (25/50/75%) of the TRACK_SUBSTR song."""
    files = [
        f
        for f in sorted(glob.glob(os.path.join(CACHE, "*.pt")))
        if TRACK_SUBSTR in os.path.basename(f)
    ]
    if not files:
        raise ValueError(f"No slice file matched '{TRACK_SUBSTR}'")
    slices = torch.load(files[0], map_location="cpu")  # [N,1,SLICE]
    name = os.path.splitext(os.path.basename(files[0]))[0].replace("slices_", "")[:40]
    n = slices.shape[0]
    fracs = [(i + 1) / (n_clips + 1) for i in range(n_clips)]  # 0.25,0.5,0.75
    clips = []
    for fr in fracs:
        start = min(int(fr * n), n - SLICES_PER_CLIP)
        start -= start % SLICES_PER_CLIP
        clip = slices[start : start + SLICES_PER_CLIP].reshape(1, -1)
        clips.append((f"{name}_p{int(fr*100)}", clip))
    return clips


@torch.no_grad()
def reconstruct(module, clip: torch.Tensor) -> torch.Tensor:
    """clip [1, L] -> recon [1, L]; process slice-by-slice as in training."""
    L = clip.shape[1]
    n = L // SLICE
    batch = clip[:, : n * SLICE].reshape(n, 1, SLICE)  # [n,1,SLICE]
    out = module.model(batch)["slice"]  # [n,1,SLICE]
    return out.reshape(1, -1)


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

    clips = pick_clips(3)
    print(f"Picked {len(clips)} clips.\n")

    modules = {}
    for tag, (path, token_dim, latent_grid) in CKPTS.items():
        if os.path.exists(path):
            print(f"Loading {tag} <- {path} (token_dim={token_dim}, grid={latent_grid})")
            modules[tag] = load_module(
                path, lp, oc, sc, la, token_dim=token_dim, latent_grid=latent_grid
            )
        else:
            print(f"SKIP {tag}: {path} missing")

    print(f"\n{'clip':<24}{'model':<16}{'cdpam':>10}{'mrstft':>10}")
    print("-" * 60)
    for i, (name, clip) in enumerate(clips):
        clip = clip / (clip.abs().max() + 1e-8)  # peak normalize for fair listen
        torchaudio.save(os.path.join(OUT, f"clip{i}_{name}_ORIG.wav"), clip, SR)
        for tag, module in modules.items():
            rec = reconstruct(module, clip)
            rec = rec / (rec.abs().max() + 1e-8)
            torchaudio.save(os.path.join(OUT, f"clip{i}_{name}_{tag}.wav"), rec, SR)
            L = min(clip.shape[1], rec.shape[1])
            cd = cdpam_score(clip[:, :L], rec[:, :L])
            mr = multi_res_stft_dist(clip[:, :L], rec[:, :L])
            print(f"{('clip'+str(i)+' '+name)[:23]:<24}{tag:<16}{cd:>10.4f}{mr:>10.4f}")
    print(f"\nWavs in: {OUT}")


if __name__ == "__main__":
    main()
