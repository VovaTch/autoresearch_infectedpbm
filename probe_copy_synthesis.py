"""
Copy-synthesis A/B/C: how much of the roughness is the tokenizer, not the model?

Every quality complaint so far -- "sounds like a mess", beat 0.42 against 0.76
for real audio -- has been measured on AR generations, which pass through the
tokenizer on the way out. That confounds two very different failures, and they
point at different multi-week projects:

  A  original     the source mp3, untouched
  B  round trip   encode -> decode, NO generation at all
  C  generated    an AR sample, decoded the same way

If B already sounds rough, the ceiling is the tokenizer and no generative
architecture can lift it -- the project is a diffusion DECODER. If B is clean
and only C is rough, the tokenizer is fine and the project is the generator.
PhaseLoss sits at weight 0.0 with the GAN off (config.yaml gan_start_step
100000000), so decoder phase is unconstrained and B is not obviously safe.

All three are LUFS-matched to the session target so loudness cannot bias the
comparison, and each is written as its own wav for listening in isolation.

Usage:
  uv run python probe_copy_synthesis.py
  uv run python probe_copy_synthesis.py --tracks 4 --seconds 12
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ab_harness.config import load_config
from ab_harness.model.audio import normalize_lufs, to_int16
from ab_harness.worker.generator import SampleRequest
from ab_harness.worker.service import GenerationService
from probe_cross_track import profile_distance, timbre_profile
from score_renders import features
from train_ar import enumerate_tracks, load_track_audio

REPO = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_ab.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tracks", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--start-frac", type=float, default=0.35)
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", default="renders_copy_synthesis")
    return parser.parse_args()


def log_spectrum(pcm: np.ndarray, sr: int) -> np.ndarray:
    """
    Args:
      pcm (np.ndarray): (N,) mono waveform.
      sr (int): sample rate.

    Returns:
      np.ndarray: (F,) mean log magnitude spectrum.
    """
    n, hop = 2048, 512
    frames = np.array(
        [
            np.abs(np.fft.rfft(pcm[i : i + n] * np.hanning(n)))
            for i in range(0, max(1, len(pcm) - n), hop)
        ]
    )
    return np.log1p(frames.mean(axis=0))


def spectral_rmse(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """
    Args:
      a (np.ndarray): (N,) reference waveform.
      b (np.ndarray): (N,) comparison waveform.
      sr (int): sample rate.

    Returns:
      float: RMSE between mean log spectra, over the overlapping length.
    """
    n = min(len(a), len(b))
    sa, sb = log_spectrum(a[:n], sr), log_spectrum(b[:n], sr)
    return float(np.sqrt(np.mean((sa - sb) ** 2)))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.bank.root = tempfile.mkdtemp(prefix="probe_copysyn_")
    if args.checkpoint:
        cfg.generator.checkpoint = args.checkpoint
    cfg.generator.window_frames = args.window

    service = GenerationService(cfg)
    service.load()
    corpus, tracks = service.corpus(), service._tracks
    gen, dec = service._generator, service._decoder
    assert gen is not None and dec is not None
    sr = dec.sample_rate

    meta = json.loads((REPO / "onnx" / "tokenizer_meta.json").read_text())
    hop = meta["hop_length"]
    ar_cfg = load_config(args.config)
    del ar_cfg
    tok_cfg = json.loads(
        (REPO / "onnx" / "tokenizer_meta.json").read_text()
    )  # rate/hop only
    rate = tok_cfg["sample_rate"]

    tracks_dir = Path(os.path.expanduser("~/.cache/infected_pbm/tracks"))
    slices_dir = Path(os.path.expanduser("~/.cache/infected_pbm/slices"))
    paths = enumerate_tracks(tracks_dir)

    frames = int(args.seconds * cfg.sampler.fps)
    rng = random.Random(0)
    chosen = rng.sample([t for t in corpus if t.num_frames > frames + 1], args.tracks)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    def save(tag: str, pcm: np.ndarray) -> np.ndarray:
        """LUFS-match, write, and return the normalised waveform."""
        audio = normalize_lufs(pcm, sr, cfg.session.target_lufs)
        norm = to_int16(audio).astype(np.float32) / 32768.0
        torchaudio.save(str(out / f"{tag}.wav"), torch.from_numpy(norm)[None], sr)
        return norm

    print(f"checkpoint {cfg.generator.checkpoint}")
    print(f"{args.seconds:.0f} s per clip, LUFS {cfg.session.target_lufs}\n")

    rows: list[tuple[str, float, float, dict[str, float], dict[str, float], dict[str, float]]] = []
    for t in chosen:
        idx = t.track_idx
        start = int(max(0, min(t.num_frames - frames - 1, t.num_frames * args.start_frac)))
        stem = f"t{idx}"

        # A: the source mp3, same span, no tokenizer in the path at all.
        wav = load_track_audio(paths[idx], rate, hop, slices_dir)
        a0, a1 = start * hop, (start + frames) * hop
        a = save(f"{stem}_A_original", wav[0, a0:a1].numpy())

        # B: the cached tokens for that exact span, decoded. encode -> decode.
        toks = tracks[idx].tokens[start : start + frames].numpy().astype(np.int16)
        b = save(f"{stem}_B_roundtrip", dec.decode(toks))

        # C: an AR generation, prompted from the same span so it starts in place.
        prompt = torch.from_numpy(
            tracks[idx].tokens[start : start + int(3.0 * cfg.sampler.fps)]
            .numpy()
            .astype(np.int64)
        ).long()
        req = SampleRequest(
            track_idx=idx,
            style=tracks[idx].style[rng.randrange(tracks[idx].style.shape[0])],
            frames=frames,
            prompt=prompt,
            temperature=args.temperature,
            top_k=250,
            top_p=0.0,
            cfg_strength=args.cfg,
            seed=1,
        )
        c = save(f"{stem}_C_generated", dec.decode(gen.sample_batch([req])[0].numpy().astype(np.int16)))

        rows.append(
            (
                f"[{idx}] {t.track_name[:34]}",
                spectral_rmse(a, b, sr),
                spectral_rmse(a, c, sr),
                features(a, sr),
                features(b, sr),
                features(c, sr),
            )
        )
        print(f"done {stem}  {t.track_name[:44]}", flush=True)

    print(f"\n{'track':<38} {'A->B':>7} {'A->C':>7}")
    for name, ab, ac, *_ in rows:
        print(f"{name:<38} {ab:7.3f} {ac:7.3f}")
    ab_m = float(np.mean([r[1] for r in rows]))
    ac_m = float(np.mean([r[2] for r in rows]))
    print(f"{'mean':<38} {ab_m:7.3f} {ac_m:7.3f}")
    print("\nlog-spectral RMSE against the original. A->B is the tokenizer alone.")

    print(f"\n{'stage':<12} {'beat':>7} {'crest dB':>9} {'centroid':>9} {'hf>5k':>7}")
    for label, col in (("A original", 3), ("B roundtrip", 4), ("C generated", 5)):
        f = [r[col] for r in rows]
        print(
            f"{label:<12} {np.mean([x['beat'] for x in f]):7.3f} "
            f"{np.mean([x['crest'] for x in f]):9.2f} "
            f"{np.mean([x['cent'] for x in f]):9.0f} "
            f"{np.mean([x['hf'] for x in f]):7.3f}"
        )

    service.close()
    print(f"\nwrote {len(rows) * 3} wavs -> {out}")


if __name__ == "__main__":
    main()
