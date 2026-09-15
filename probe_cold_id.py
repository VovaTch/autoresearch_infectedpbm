"""
Cold-start conditioning: is the id/style stream a track lookup at all?

probe_cross_track measured a swap of the conditioning while the prompt stayed
on track A, and found the output barely moved. That probe was confounded by
construction: every lane got 3 s of real A tokens to continue, so the prompt
carried the answer and the conditioning never had to. The conditioning ladder
hit the same wall (id p=0.81, style p=0.67, all prompted).

This probe removes the prompt. A cold lane has nothing but the id embedding and
the style descriptor, so if training -- which only ever paired a track with its
own conditioning -- taught the model "id=T, style=T means make T", a cold clip
conditioned on T should land nearer T's real timbre than any other track's.

  cond_T     id=T style=T, no prompt      the lookup, if there is one
  null       both streams off, no prompt  what no information sounds like

Scored as identification: for each cold clip, rank its own track among the N
real anchors by timbre distance. Chance is (N+1)/2. A working lookup ranks 1.

Nothing is written to the preference bank.

Usage:
  uv run python probe_cold_id.py
  uv run python probe_cold_id.py --tracks 8 --seeds 1 2 3 4 --cfg 2.0 4.0
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
from probe_cross_track import profile_distance, timbre_profile

REPO = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_ab.yaml")
    parser.add_argument(
        "--checkpoint", default=None, help="override the config's checkpoint"
    )
    parser.add_argument("--tracks", type=int, default=6, help="how many tracks")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--start-frac", type=float, default=0.35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--cfg", type=float, nargs="+", default=[0.0, 2.0, 4.0, 8.0])
    parser.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.7])
    parser.add_argument(
        "--window",
        type=int,
        default=4096,
        help="sampling window; the KV cache is sized to this regardless of "
        "clip length, so a 16384 window costs ~10 GiB at max_batch 8",
    )
    parser.add_argument("--out", default="renders_cold_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    scratch = Path(tempfile.mkdtemp(prefix="probe_coldid_"))
    cfg.bank.root = str(scratch)
    if args.checkpoint:
        cfg.generator.checkpoint = args.checkpoint
    cfg.generator.window_frames = args.window

    service = GenerationService(cfg)
    service.load()
    corpus = service.corpus()
    tracks = service._tracks
    gen, dec = service._generator, service._decoder
    assert gen is not None and dec is not None

    frames = int(args.seconds * cfg.sampler.fps)
    rng = random.Random(0)
    pool = [t for t in corpus if t.num_frames > frames + 1]
    chosen = rng.sample(pool, min(args.tracks, len(pool)))

    def style_of(idx: int) -> torch.Tensor:
        """Style descriptor from a random window of `idx`."""
        tr = tracks[idx]
        return tr.style[rng.randrange(tr.style.shape[0])]

    print(f"checkpoint  {cfg.generator.checkpoint}")
    for t in chosen:
        print(f"track       [{t.track_idx}] {t.track_name}")
    print(f"seeds {args.seeds}   cfg {args.cfg}   temps {args.temps}   "
          f"{args.seconds}s COLD (no prompt)")
    print("cfg 0.0 == cfg 1.0: unguided conditional, one pass instead of two\n")

    cells: list[tuple[str, int, float, float, int]] = []
    for seed in args.seeds:
        for temp in args.temps:
            for g in args.cfg:
                for t in chosen:
                    cells.append(
                        (f"cond_t{t.track_idx}_cfg{g:g}_T{temp:g}",
                         t.track_idx, g, temp, seed)
                    )
            cells.append((f"null_T{temp:g}", chosen[0].track_idx, 0.0, temp, seed))

    reqs = [
        SampleRequest(
            track_idx=idx,
            style=style_of(idx),
            use_track_id=not name.startswith("null"),
            use_style=not name.startswith("null"),
            frames=frames,
            prompt=None,
            temperature=temp,
            top_k=250,
            top_p=0.0,
            cfg_strength=strength,
            seed=seed,
        )
        for name, idx, strength, temp, seed in cells
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

    # Real anchor per track: its own audio at the same relative offset.
    anchors: dict[int, np.ndarray] = {}
    for t in chosen:
        s = int(max(0, min(t.num_frames - frames - 1, t.num_frames * args.start_frac)))
        real = tracks[t.track_idx].tokens[s : s + frames].numpy().astype(np.int16)
        anchors[t.track_idx] = timbre_profile(
            save(f"REAL_t{t.track_idx}", real), dec.sample_rate
        )

    order = [t.track_idx for t in chosen]
    n = len(order)
    rows: list[tuple[str, int, float, int, float, float]] = []
    for (name, idx, _strength, _temp, seed), code in zip(cells, codes):
        prof = timbre_profile(save(f"{name}_s{seed}", code.numpy().astype(np.int16)),
                              dec.sample_rate)
        dists = {k: profile_distance(prof, v) for k, v in anchors.items()}
        own = dists[idx]
        others = [d for k, d in dists.items() if k != idx]
        rank = 1 + sum(1 for d in others if d < own)
        rows.append((name, seed, 0.0, rank, own, float(np.mean(others))))

    print(f"\n{'cell':<28} {'seed':>4} {'rank':>5} {'d(own)':>8} {'d(other)':>9}")
    for name, seed, _s, rank, own, oth in rows:
        mark = "  <-- own track nearest" if rank == 1 else ""
        print(f"{name:<28} {seed:>4} {rank:>3}/{n} {own:8.4f} {oth:9.4f}{mark}")

    print(f"\nchance rank = {(n + 1) / 2:.1f}/{n}\n")
    print(f"{'group':<18} {'n':>3} {'mean rank':>10} {'rank=1':>7} "
          f"{'d(own)':>8} {'d(other)':>9} {'own-other':>10}")
    groups = [
        f"cfg{c:g}_T{t:g}" for t in args.temps for c in args.cfg
    ] + [f"null_T{t:g}" for t in args.temps]
    for g in groups:
        sel = [r for r in rows if r[0].endswith(g)]
        if not sel:
            continue
        ranks = np.array([r[3] for r in sel], dtype=float)
        own = np.array([r[4] for r in sel])
        oth = np.array([r[5] for r in sel])
        delta = own - oth
        se = delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else float("nan")
        t = delta.mean() / se if se else float("nan")
        print(f"{g:<18} {len(sel):>3} {ranks.mean():>10.2f} "
              f"{(ranks == 1).mean():>6.0%} {own.mean():8.4f} {oth.mean():9.4f} "
              f"{delta.mean():>+10.4f}  t={t:.2f}")

    service.close()
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"\nwrote {len(rows) + n} wavs -> {out}")
    print("Lookup works if mean rank << chance and own-other is negative.")


if __name__ == "__main__":
    main()
