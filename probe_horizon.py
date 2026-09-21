"""
Horizon probe: can the AR continue a melodic loop whose period fits its window?

The 512-frame checkpoints see 2.97 s of context (172 fps). At 145 BPM that is
1.8 bars, so any phrase longer than that is invisible by construction. The test
that separates "the model cannot do melody" from "the model cannot see melody":
tile one real bar-multiple of corpus audio into a loop, prime with the last
window of it, and score how much of the loop the continuation keeps. Loops
shorter than the window are fully visible (the model has seen a repeat); loops
longer than it are not, and those conditions collapse to "continue real music".

Conditions are bar counts. The loop period in seconds follows the track's own
tempo (librosa beat tracking), snapped to a beat so the loop is on-grid.

Scores, per generated clip, against the loop unit as template:
  chroma  phase-max Pearson correlation of chroma_stft, tonal content
  mel     the same on log-mel, timbre + rhythm
  tok     fraction of level-0 tokens equal to the token one loop period back
Rows "ceiling" (encode->decode of the tiled loop) and "chance" (the template
against another spot in the track) calibrate each metric.

Usage:
  uv run python probe_horizon.py --tracks deeply_disturbed Cookie --seeds 0 1 2
  uv run python probe_horizon.py --tracks Cookie --start-sec 95 --bars 1 2 8
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

from ab_harness.config import REPO, GeneratorCfg
from ab_harness.worker.decoder import TokenDecoder
from ab_harness.worker.generator import SampleRequest
from ab_harness.worker.loading import LoadedModel, load_ar_checkpoint
from train_ar import (
    TrackTokens,
    encode_chunked,
    enumerate_tracks,
    load_track_audio,
    make_session,
)

FEAT_HOP = 1024  # scoring feature hop at 44.1 kHz, ~43 fps


@dataclass
class Condition:
    """
    One loop condition on one track.

    Args:
      track (TrackTokens): corpus track supplying audio, id and style.
      bars (float): loop length in bars.
      loop_sec (float): loop period in seconds after beat snapping.
      unit (np.ndarray): (L,) float32 loop unit audio.
      prompt (np.ndarray): (P,) float32 tiled prompt audio, ends on a boundary.
      prompt_tokens (torch.Tensor): (P_frames, R) int64 codes of the prompt.
      style (torch.Tensor): (D,) style descriptor of the source window.
    """

    track: TrackTokens
    bars: float
    loop_sec: float
    unit: np.ndarray
    prompt: np.ndarray
    prompt_tokens: torch.Tensor
    style: torch.Tensor


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="saved_ar_20260915_512_aligned/ar_latest.ckpt")
    ap.add_argument("--tracks", nargs="+", default=["deeply_disturbed", "Cookie"])
    ap.add_argument(
        "--start-sec",
        type=float,
        nargs="*",
        default=[],
        help="loop start per track; missing entries are auto-picked (tonal region)",
    )
    ap.add_argument("--bars", type=float, nargs="+", default=[0.5, 1, 1.5, 2, 4, 8])
    ap.add_argument("--seconds", type=float, default=20.0, help="generated length")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=250)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=512, help="sampling window frames")
    ap.add_argument("--out", default="renders_horizon")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-batch", type=int, default=16, help="KV rows per batch")
    return ap.parse_args()


def find_track(tracks: list[TrackTokens], want: str) -> TrackTokens:
    """
    Args:
      tracks (list[TrackTokens]): loaded corpus.
      want (str): case-insensitive substring of the track name.

    Returns:
      TrackTokens: first match.
    """
    hits = [t for t in tracks if want.lower() in t.track_name.lower()]
    if not hits:
        raise SystemExit(f"no track matching {want!r}")
    return hits[0]


def tonal_start(wav: np.ndarray, sr: int, span_sec: float = 20.0) -> float:
    """
    Pick the most pitch-clear region of a track for the loop source.

    Low chroma entropy = a clear tonal centre; RMS above the median keeps it out
    of intros and breakdowns.

    Args:
      wav (np.ndarray): (L,) mono waveform.
      sr (int): sample rate.
      span_sec (float): region length scored.

    Returns:
      float: start time in seconds.
    """
    hop = 4096
    chroma = librosa.feature.chroma_stft(y=wav, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=wav, hop_length=hop)[0]
    probs = chroma / np.maximum(chroma.sum(0, keepdims=True), 1e-9)
    entropy = -(probs * np.log(probs + 1e-9)).sum(0)
    span = int(span_sec * sr / hop)
    lo, hi = span, max(span + 1, len(entropy) - 2 * span)
    best, best_t = np.inf, lo
    for t in range(lo, hi):
        if rms[t : t + span].mean() < np.percentile(rms, 75):
            continue
        score = float(entropy[t : t + span].mean())
        if score < best:
            best, best_t = score, t
    return best_t * hop / sr


def snap_loop(
    wav: np.ndarray, sr: int, start_sec: float, bars: float
) -> tuple[float, float]:
    """
    Snap a loop start to a beat and its length to a bar multiple of the tempo.

    Args:
      wav (np.ndarray): (L,) mono waveform.
      sr (int): sample rate.
      start_sec (float): requested start.
      bars (float): loop length in bars of 4 beats.

    Returns:
      tuple[float, float]: (start_sec on a beat, loop_sec).
    """
    lo = int(max(0.0, start_sec - 10.0) * sr)
    hi = int((start_sec + 40.0) * sr)
    tempo, beats = librosa.beat.beat_track(y=wav[lo:hi], sr=sr, units="time")
    bpm = float(np.atleast_1d(tempo)[0])
    beats = beats + lo / sr
    snapped = float(beats[np.argmin(np.abs(beats - start_sec))]) if len(beats) else start_sec
    return snapped, bars * 4.0 * 60.0 / bpm


def tile(unit: np.ndarray, length: int, fade: int = 441) -> np.ndarray:
    """
    Tile a loop unit with a short crossfade at each seam, ending on a boundary.

    Args:
      unit (np.ndarray): (L,) loop audio.
      length (int): minimum total samples; rounded up to whole units.
      fade (int): crossfade samples per seam (10 ms at 44.1 kHz).

    Returns:
      np.ndarray: (N * L,) tiled audio, N = ceil(length / L).
    """
    reps = int(np.ceil(length / len(unit)))
    out = np.tile(unit, reps).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    for k in range(1, reps):
        seam = k * len(unit)
        tail = unit[-fade:] * (1.0 - ramp)
        out[seam : seam + fade] = out[seam : seam + fade] * ramp + tail
    return out


def features(wav: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
      wav (np.ndarray): (L,) mono waveform.
      sr (int): sample rate.

    Returns:
      tuple[np.ndarray, np.ndarray]: chroma (12, F) and log-mel (64, F).
    """
    chroma = librosa.feature.chroma_stft(y=wav, sr=sr, hop_length=FEAT_HOP)
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=FEAT_HOP, n_mels=64)
    return chroma, np.log(mel + 1e-6)


def centred(block: np.ndarray) -> np.ndarray:
    """
    Remove each feature row's mean so a static spectral envelope scores zero.

    Args:
      block (np.ndarray): (C, F) features.

    Returns:
      np.ndarray: (C, F) row-centred features.
    """
    return block - block.mean(1, keepdims=True)


def phase_max_corr(template: np.ndarray, segment: np.ndarray) -> float:
    """
    Pearson correlation between two (C, F) feature blocks, maximised over a
    circular shift of the segment so loop phase does not matter.

    Args:
      template (np.ndarray): (C, F) loop-unit features.
      segment (np.ndarray): (C, F) features of one loop period of output.

    Returns:
      float: best correlation over shifts.
    """
    f = min(template.shape[1], segment.shape[1])
    template, segment = centred(template[:, :f]), centred(segment[:, :f])
    a = template.ravel()
    a = (a - a.mean()) / (a.std() + 1e-9)
    best = -1.0
    for shift in range(f):
        b = np.roll(segment[:, :f], shift, axis=1).ravel()
        b = (b - b.mean()) / (b.std() + 1e-9)
        best = max(best, float(np.dot(a, b) / len(a)))
    return best


def loop_scores(
    unit: np.ndarray, out: np.ndarray, sr: int, periods: int = 6
) -> tuple[list[float], list[float]]:
    """
    Score successive loop periods of an output against the unit template.

    Args:
      unit (np.ndarray): (L,) loop unit audio.
      out (np.ndarray): (M,) output audio, starting at a loop boundary.
      sr (int): sample rate.
      periods (int): how many periods to score.

    Returns:
      tuple[list[float], list[float]]: per-period chroma and mel correlations.
    """
    t_chroma, t_mel = features(unit, sr)
    frames = t_chroma.shape[1]
    o_chroma, o_mel = features(out, sr)
    chroma, mel = [], []
    for k in range(periods):
        lo, hi = k * frames, (k + 1) * frames
        if hi > o_chroma.shape[1]:
            break
        chroma.append(phase_max_corr(t_chroma, o_chroma[:, lo:hi]))
        mel.append(phase_max_corr(t_mel, o_mel[:, lo:hi]))
    return chroma, mel


def token_periodicity(tokens: torch.Tensor, period: int) -> float:
    """
    Args:
      tokens (torch.Tensor): (T, R) codes.
      period (int): loop period in frames.

    Returns:
      float: fraction of level-0 codes equal to the code one period earlier.
    """
    if tokens.shape[0] <= period:
        return float("nan")
    col = tokens[:, 0]
    return float((col[period:] == col[:-period]).float().mean())


def style_for_start(track: TrackTokens, frame: int) -> torch.Tensor:
    """
    Args:
      track (TrackTokens): corpus track.
      frame (int): a frame inside the wanted style window.

    Returns:
      torch.Tensor: (D,) descriptor of the grid slice holding that frame.
    """
    bounds = np.asarray(track.style_bounds)
    idx = int(np.argmin(np.abs(bounds[:, 0] - frame)))
    return track.style[idx]


def build_condition(
    track: TrackTokens,
    wav: np.ndarray,
    sr: int,
    hop: int,
    start_sec: float,
    bars: float,
    window: int,
    encoder,
    chunk: int,
    margin: int,
) -> Condition:
    """
    Cut, tile and encode one loop condition.

    Args:
      track (TrackTokens): source corpus track.
      wav (np.ndarray): (L,) full track audio.
      sr (int): sample rate.
      hop (int): tokenizer hop.
      start_sec (float): requested loop start.
      bars (float): loop length in bars.
      window (int): prompt length in frames (the sampling window).
      encoder: onnx encoder session.
      chunk (int): encoder chunk frames.
      margin (int): encoder margin frames.

    Returns:
      Condition: the prepared condition.
    """
    snapped, loop_sec = snap_loop(wav, sr, start_sec, bars)
    n_unit = int(round(loop_sec * sr / hop)) * hop
    lo = int(snapped * sr)
    unit = wav[lo : lo + n_unit].astype(np.float32)
    tiled = tile(unit, window * hop)
    prompt = tiled[-window * hop :]
    codes = encode_chunked(
        encoder, torch.from_numpy(prompt)[None], hop, chunk, margin
    )
    return Condition(
        track=track,
        bars=bars,
        loop_sec=n_unit / sr,
        unit=unit,
        prompt=prompt,
        prompt_tokens=torch.from_numpy(np.asarray(codes)).long(),
        style=style_for_start(track, int(snapped * sr / hop)),
    )


def main() -> None:
    args = parse_args()
    gen_cfg = GeneratorCfg(
        checkpoint=args.ckpt,
        device=args.device,
        window_frames=args.window,
        max_batch=args.max_batch,
    )
    loaded: LoadedModel = load_ar_checkpoint(args.ckpt, gen_cfg)
    meta = loaded.meta
    sr, hop, fps = int(meta["sample_rate"]), int(meta["hop_length"]), loaded.fps
    tok_cfg = loaded.ar_cfg.tokenizer
    encoder = make_session(REPO / gen_cfg.encoder_onnx, int(meta["num_rq"]))
    decoder = TokenDecoder(REPO / gen_cfg.decoder_onnx, hop=hop, sample_rate=sr)
    tracks_dir = Path(os.path.expanduser(tok_cfg.tracks_dir))
    slices_dir = Path(os.path.expanduser(tok_cfg.slices_dir))
    paths = enumerate_tracks(tracks_dir)
    out_dir = REPO / args.out
    out_dir.mkdir(exist_ok=True)
    print(f"ckpt {args.ckpt}  window {args.window} = {args.window / fps:.2f}s")

    conditions: list[Condition] = []
    for i, want in enumerate(args.tracks):
        track = find_track(loaded.tracks, want)
        wav = load_track_audio(paths[track.track_idx], sr, hop, slices_dir)[0].numpy()
        start = args.start_sec[i] if i < len(args.start_sec) else tonal_start(wav, sr)
        print(f"track '{track.track_name}' id {track.track_idx} loop start {start:.1f}s")
        for bars in args.bars:
            cond = build_condition(
                track, wav, sr, hop, start, bars, args.window, encoder,
                tok_cfg.chunk_frames, tok_cfg.margin,
            )
            conditions.append(cond)
            print(f"  {bars:>4} bars = {cond.loop_sec:5.2f}s  ({cond.loop_sec * fps:.0f} frames)")

    n_frames = int(args.seconds * fps)
    rows: list[dict[str, object]] = []

    def score(tag: str, cond: Condition, out: np.ndarray, tokens: torch.Tensor | None, seed: int) -> None:
        chroma, mel = loop_scores(cond.unit, out, sr)
        period = int(round(cond.loop_sec * fps))
        row = {
            "track": cond.track.track_name[:24],
            "bars": cond.bars,
            "loop_s": round(cond.loop_sec, 2),
            "kind": tag,
            "seed": seed,
            "chroma_1": round(chroma[0], 3) if chroma else float("nan"),
            "chroma_mean": round(float(np.mean(chroma)), 3) if chroma else float("nan"),
            "mel_1": round(mel[0], 3) if mel else float("nan"),
            "mel_mean": round(float(np.mean(mel)), 3) if mel else float("nan"),
            "tok_period": round(token_periodicity(tokens, period), 3) if tokens is not None else float("nan"),
            "n_periods": len(chroma),
        }
        rows.append(row)
        print(
            f"  {row['track']:<24} {cond.bars:>4} bars {tag:<8} s{seed} "
            f"chroma1 {row['chroma_1']:.3f} mean {row['chroma_mean']:.3f}  "
            f"mel1 {row['mel_1']:.3f} mean {row['mel_mean']:.3f}  tok {row['tok_period']}"
        )

    # calibration rows: ceiling (tokenizer round trip of the loop) and chance
    print("\ncalibration")
    for cond in conditions:
        span = int(np.ceil(args.seconds / cond.loop_sec)) + 1
        tiled = tile(cond.unit, span * len(cond.unit))
        codes = encode_chunked(
            encoder, torch.from_numpy(tiled)[None], hop, tok_cfg.chunk_frames, tok_cfg.margin
        )
        recon = decoder.decode(np.asarray(codes))
        score("ceiling", cond, recon, torch.from_numpy(np.asarray(codes)).long(), -1)
        wav_full = load_track_audio(paths[cond.track.track_idx], sr, hop, slices_dir)[0].numpy()
        far = (len(wav_full) // 2 + int(37.0 * sr)) % (len(wav_full) - len(cond.unit) * 8)
        score("chance", cond, wav_full[far : far + len(cond.unit) * 7], None, -1)

    print("\nsampling")
    requests: list[SampleRequest] = []
    meta_req: list[tuple[Condition, int]] = []
    for cond in conditions:
        for seed in args.seeds:
            requests.append(
                SampleRequest(
                    track_idx=cond.track.track_idx,
                    style=cond.style,
                    frames=cond.prompt_tokens.shape[0] + n_frames,
                    prompt=cond.prompt_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    cfg_strength=args.cfg,
                    seed=seed,
                )
            )
            meta_req.append((cond, seed))

    batch = max(1, gen_cfg.max_batch // (2 if args.cfg > 0 else 1))
    outputs: list[torch.Tensor] = []
    for i in range(0, len(requests), batch):
        chunk = requests[i : i + batch]
        print(f"  batch {i // batch + 1}/{(len(requests) + batch - 1) // batch} ({len(chunk)} lanes)")
        outputs += loaded.generator.sample_batch(chunk)

    for (cond, seed), tokens in zip(meta_req, outputs):
        p = cond.prompt_tokens.shape[0]
        kept = float((tokens[:p] == cond.prompt_tokens).float().mean())
        if kept < 1.0:
            print(f"  WARN prompt reproduced {kept:.1%}")
        gen_tokens = tokens[p:]
        gen_wav = decoder.decode(tokens.numpy())
        score("gen", cond, gen_wav[p * hop :], gen_tokens, seed)
        stem = f"{cond.track.track_name[:16].replace(' ', '_')}_{cond.bars:g}bar_s{seed}"
        torchaudio.save(str(out_dir / f"{stem}.wav"), torch.from_numpy(gen_wav)[None], sr)
        if seed == args.seeds[0]:
            torchaudio.save(
                str(out_dir / f"{stem}_promptraw.wav"), torch.from_numpy(cond.prompt)[None], sr
            )

    with open(out_dir / "scores.tsv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print("\nsummary: mean over seeds, gen rows (chance / ceiling in brackets)")
    print(f"{'track':<24} {'bars':>5} {'loop_s':>6} {'chroma1':>18} {'chroma_mean':>18} {'mel_mean':>18} {'tok':>6}")
    for cond in conditions:
        pick = lambda kind: [r for r in rows if r["track"] == cond.track.track_name[:24] and r["bars"] == cond.bars and r["kind"] == kind]
        g, c, ch = pick("gen"), pick("ceiling")[0], pick("chance")[0]
        mean = lambda key: float(np.nanmean([float(r[key]) for r in g]))
        print(
            f"{cond.track.track_name[:24]:<24} {cond.bars:>5} {cond.loop_sec:>6.2f} "
            f"{mean('chroma_1'):>6.3f} [{ch['chroma_1']:.2f}/{c['chroma_1']:.2f}] "
            f"{mean('chroma_mean'):>6.3f} [{ch['chroma_mean']:.2f}/{c['chroma_mean']:.2f}] "
            f"{mean('mel_mean'):>6.3f} [{ch['mel_mean']:.2f}/{c['mel_mean']:.2f}] "
            f"{mean('tok_period'):>6.3f}"
        )
    print(f"\nwavs + scores.tsv in {out_dir}")


if __name__ == "__main__":
    main()
