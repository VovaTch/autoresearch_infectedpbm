"""
Cross-track conditioning: does the style/id stream carry anything transferable?

Every clip the harness has ever drawn takes its prompt, its style descriptor and
its track id from ONE track -- ClipSpec has a single conditioning.track_idx, and
service._real_tokens, service._style_vector and SampleRequest.track_idx all read
it. Training matched them too (shuffle_cond is False in every checkpoint, and
_pick_style draws "from elsewhere in the same track"). So the conditioning has
never had to do work the prompt was not already doing, which is the likely
reason the conditioning ladder measured it inert (id p=0.81, style p=0.67).

This probe breaks the tie: prompt always from track A, conditioning from track B.

  matched     id=A style=A   the normal case, the baseline
  xstyle      id=A style=B   only the style descriptor crosses
  xid         id=B style=A   only the track id crosses
  xboth       id=B style=B   both cross

Three outcomes, and they mean different things:
  * tokens identical to matched  -> the stream is being ignored outright
  * output moves toward track B  -> style transfer works, and the harness has
                                    been blind to its most interesting axis
  * output changes but toward
    neither A nor B              -> off-distribution, needs shuffle_cond training

Bypasses GenerationService.produce_many, which cannot express this: it builds a
SampleRequest from one ClipSpec. The generator and decoder are driven directly.
Nothing is written to the preference bank.

Usage:
  uv run python probe_cross_track.py
  uv run python probe_cross_track.py --track deeply_disturbed --seeds 1 2 3 4
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from ab_harness.config import load_config
from ab_harness.model.audio import normalize_lufs, to_int16
from ab_harness.worker.generator import SampleRequest
from ab_harness.worker.service import GenerationService

REPO = Path(__file__).resolve().parent
BANDS = 24


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_ab.yaml")
    parser.add_argument("--track", default="deeply_disturbed", help="track A substring")
    parser.add_argument(
        "--others",
        type=int,
        default=3,
        help="how many B tracks to borrow conditioning from",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--start-frac", type=float, default=0.35)
    parser.add_argument("--prompt-sec", type=float, default=3.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--out", default="renders_cross_track")
    return parser.parse_args()


def timbre_profile(pcm: np.ndarray, sr: int, bands: int = BANDS) -> np.ndarray:
    """
    Log-spaced band energies: a coarse timbre fingerprint.

    Args:
      pcm (np.ndarray): (N,) float32 mono waveform.
      sr (int): sample rate.
      bands (int): number of log-spaced frequency bands.

    Returns:
      np.ndarray: (bands,) mean log band energy, mean-centred so the profile
        compares timbre shape rather than loudness.
    """
    n, hop = 2048, 512
    frames = [
        np.abs(np.fft.rfft(pcm[i : i + n] * np.hanning(n)))
        for i in range(0, max(1, len(pcm) - n), hop)
    ]
    spec = np.mean(np.array(frames), axis=0)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    edges = np.logspace(np.log10(40), np.log10(sr / 2), bands + 1)
    out = np.array(
        [
            spec[m].mean() if (m := (freqs >= lo) & (freqs < hi)).any() else 0.0
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    out = np.log1p(out)
    return out - out.mean()


def profile_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Args:
      a (np.ndarray): (bands,) profile.
      b (np.ndarray): (bands,) profile.

    Returns:
      float: correlation distance in [0, 2]; 0 is identical shape.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1.0 - (a @ b) / denom) if denom > 0 else 1.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    scratch = Path(tempfile.mkdtemp(prefix="probe_xtrack_"))
    cfg.bank.root = str(scratch)

    service = GenerationService(cfg)
    service.load()
    corpus = service.corpus()
    tracks = service._tracks
    gen, dec = service._generator, service._decoder
    assert gen is not None and dec is not None

    fps = cfg.sampler.fps
    frames = int(args.seconds * fps)
    n_prompt = int(args.prompt_sec * fps)

    hits = [t for t in corpus if args.track.lower() in t.track_name.lower()]
    if not hits:
        raise SystemExit(f"no track matching {args.track!r}")
    a = hits[0]
    start = int(max(0, min(a.num_frames - frames - 1, a.num_frames * args.start_frac)))

    rng = random.Random(0)
    pool = [
        t for t in corpus if t.track_idx != a.track_idx and t.num_frames > frames + 1
    ]
    others = rng.sample(pool, min(args.others, len(pool)))

    def style_of(idx: int, span_start: int) -> torch.Tensor:
        """Style descriptor from a window of `idx` disjoint from the A span."""
        tr = tracks[idx]
        bounds = tr.style_bounds
        free = [
            w
            for w in range(bounds.shape[0])
            if not (
                span_start < int(bounds[w, 1])
                and int(bounds[w, 0]) < span_start + frames
            )
        ]
        w = rng.choice(free) if free else 0
        return tr.style[min(w, tr.style.shape[0] - 1)]

    prompt = torch.from_numpy(
        tracks[a.track_idx].tokens[start : start + n_prompt].numpy().astype(np.int64)
    ).long()

    print(f"checkpoint  {cfg.generator.checkpoint}")
    print(f"track A     [{a.track_idx}] {a.track_name}")
    print(
        f"span        {start}-{start + frames}   "
        f"prompt {n_prompt} frames (always from A)"
    )
    for b in others:
        print(f"track B     [{b.track_idx}] {b.track_name}")
    print(f"seeds       {args.seeds}\n")

    cells: list[tuple[str, int, torch.Tensor, int]] = []
    for seed in args.seeds:
        style_a = style_of(a.track_idx, start)
        cells.append(("matched", a.track_idx, style_a, seed))
        for b in others:
            style_b = style_of(b.track_idx, start)
            bi = b.track_idx
            cells.append((f"xstyle_t{bi}", a.track_idx, style_b, seed))
            cells.append((f"xid_t{bi}", bi, style_a, seed))
            cells.append((f"xboth_t{bi}", bi, style_b, seed))

    reqs = [
        SampleRequest(
            track_idx=idx,
            style=style,
            use_track_id=True,
            use_style=True,
            frames=frames,
            prompt=prompt,
            temperature=1.0,
            top_k=250,
            top_p=0.0,
            cfg_strength=2.0,
            seed=seed,
        )
        for _, idx, style, seed in cells
    ]

    print(f"sampling {len(reqs)} clips in chunks of {cfg.generator.max_batch}")
    codes: list[torch.Tensor] = []
    size = max(1, cfg.generator.max_batch)
    for i in range(0, len(reqs), size):
        codes += gen.sample_batch(reqs[i : i + size])
        print(f"  {min(i + size, len(reqs))}/{len(reqs)}", flush=True)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    def save(tag: str, tokens: np.ndarray) -> np.ndarray:
        audio = normalize_lufs(
            dec.decode(tokens), dec.sample_rate, cfg.session.target_lufs
        )
        pcm = to_int16(audio)
        torchaudio.save(
            str(out / f"{tag}.wav"),
            torch.from_numpy(pcm.astype(np.float32) / 32768.0)[None],
            dec.sample_rate,
        )
        return pcm.astype(np.float32) / 32768.0

    # Real-audio anchors: A's own span, and each B's span at the same offset.
    anchors: dict[str, np.ndarray] = {}
    real_a = (
        tracks[a.track_idx].tokens[start : start + frames].numpy().astype(np.int16)
    )
    anchors["A"] = timbre_profile(save("REAL_A", real_a), dec.sample_rate)
    for b in others:
        s_b = min(start, tracks[b.track_idx].num_frames - frames - 1)
        real_b = (
            tracks[b.track_idx].tokens[s_b : s_b + frames].numpy().astype(np.int16)
        )
        anchors[f"t{b.track_idx}"] = timbre_profile(
            save(f"REAL_t{b.track_idx}", real_b), dec.sample_rate
        )

    matched_tokens: dict[int, np.ndarray] = {}
    rows: list[tuple[str, int, float, float, float]] = []
    for (name, _idx, _style, seed), code in zip(cells, codes):
        tok = code.numpy().astype(np.int16)
        pcm = save(f"{name}_s{seed}", tok)
        prof = timbre_profile(pcm, dec.sample_rate)
        if name == "matched":
            matched_tokens[seed] = tok
            rows.append(
                (name, seed, 0.0, profile_distance(prof, anchors["A"]), float("nan"))
            )
            continue
        ref = matched_tokens.get(seed)
        same_shape = ref is not None and ref.shape == tok.shape
        diff = float((tok != ref).mean()) if same_shape else float("nan")
        b_key = "t" + name.split("_t")[1]
        rows.append(
            (
                name,
                seed,
                diff,
                profile_distance(prof, anchors["A"]),
                profile_distance(prof, anchors[b_key]),
            )
        )

    head = f"{'cell':<18} {'seed':>4} {'tok!=matched':>12} {'dist->A':>8}"
    print(f"\n{head} {'dist->B':>8}")
    for name, seed, diff, da, db in rows:
        db_s = "     -- " if np.isnan(db) else f"{db:8.4f}"
        print(f"{name:<18} {seed:>4} {diff:>11.1%} {da:8.4f} {db_s}")

    head = f"{'cell group':<18} {'n':>3} {'tok!=matched':>12} {'dist->A':>8}"
    print(f"\n{head} {'dist->B':>8}")
    for group in ("matched", "xstyle", "xid", "xboth"):
        sel = [r for r in rows if r[0].startswith(group)]
        if not sel:
            continue
        diffs = [r[2] for r in sel if not np.isnan(r[2])]
        das = [r[3] for r in sel]
        dbs = [r[4] for r in sel if not np.isnan(r[4])]
        db_s = "     -- " if not dbs else f"{np.mean(dbs):8.4f}"
        print(
            f"{group:<18} {len(sel):>3} {np.mean(diffs) if diffs else 0:>11.1%} "
            f"{np.mean(das):8.4f} {db_s}"
        )

    service.close()
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"\nwrote {len(rows) + 1 + len(others)} wavs -> {out}")
    print("dist->A / dist->B are correlation distances of the timbre profile to")
    print("REAL_A / REAL_t<B>. Conditioning transfers if dist->B falls below matched.")


if __name__ == "__main__":
    main()
