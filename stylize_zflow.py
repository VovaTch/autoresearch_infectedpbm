"""
Stylise a foreign track with the latent flow prior (train_zflow.py) by SDEdit,
or draw free samples from it, and decode through the frozen ONNX tokenizer.

    mp3 -> ONNX encoder -> tokens -> z_q -> whiten -> SDEdit (windowed) ->
    unwhiten -> requantize (beam) -> tokens -> ONNX decoder -> wav

Every render is scored against the source (cdpam / mrstft on 32768-sample
slices, token agreement per level = content retention) and against an IM
timbre anchor (probe_cross_track.timbre_profile over training tracks = style
pull). A cell where the timbre distance drops clearly below the original's
while cdpam stays moderate is the one to listen to.

Usage:
  uv run python stylize_zflow.py stylize ~/Downloads/a.mp3 --strength 0.3 0.5 0.7 --cfg 1 3
  uv run python stylize_zflow.py stylize a.mp3 --style-track-idx 8 --cfg 3
  uv run python stylize_zflow.py sample --n 4 --seconds 12 --style-track-idx 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from generate_ar import decode_tokens, make_decoder
from prepare import make_cdpam_evaluator
from probe_cross_track import profile_distance, timbre_profile
from roundtrip_external import slice_metrics
from score_renders import features
from train_ar import encode_chunked, enumerate_tracks, load_track_audio, make_session
from train_zflow import (
    ZFlowModule,
    embed_zq,
    load_corpus,
    load_zflow_module,
    requantize,
    style_vector,
)

REPO = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ckpt", default="saved_zflow/last.ckpt")
        p.add_argument(
            "--no-ema", action="store_true", help="use raw instead of EMA weights"
        )
        p.add_argument("--out", default="renders_zflow")
        p.add_argument(
            "--steps", type=int, default=None, help="Euler steps; config when omitted"
        )
        p.add_argument(
            "--churn", type=float, default=None, help="noise re-draw fraction per step"
        )
        p.add_argument(
            "--cfg", type=float, nargs="+", default=[1.0], help="guidance scales"
        )
        p.add_argument(
            "--style-ref", default=None, help="mp3 whose mean z_q is the style vector"
        )
        p.add_argument(
            "--style-track-idx",
            type=int,
            default=None,
            help="corpus track index as style",
        )
        p.add_argument("--beam", type=int, default=8, help="requantize beam width")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
        )

    st = sub.add_parser("stylize", help="SDEdit foreign tracks")
    st.add_argument("tracks", nargs="+", help="mp3 paths")
    st.add_argument("--strength", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    st.add_argument(
        "--window",
        type=int,
        default=None,
        help="frames per SDEdit window; config crop when omitted",
    )
    st.add_argument("--window-batch", type=int, default=8)
    st.add_argument(
        "--max-seconds", type=float, default=None, help="truncate the source"
    )
    st.add_argument(
        "--anchor-tracks",
        type=int,
        default=10,
        help="training tracks in the IM timbre anchor",
    )
    st.add_argument("--cdpam-batch", type=int, default=16)
    common(st)

    sa = sub.add_parser("sample", help="free draws from the prior")
    sa.add_argument("--n", type=int, default=4)
    sa.add_argument("--seconds", type=float, default=12.0)
    common(sa)
    return parser.parse_args()


# ===========================================================================
# Latent plumbing
# ===========================================================================


class Pipeline:
    """
    Frozen tokenizer sessions plus the flow module, on one device.

    Args:
      module (ZFlowModule): trained prior.
      device (str): torch device for the flow.
    """

    def __init__(self, module: ZFlowModule, device: str) -> None:
        self.module = module
        self.device = device
        tok = module.cfg.tokenizer
        self.meta: dict[str, Any] = json.loads((REPO / tok.meta).read_text())
        self.hop = int(self.meta["hop_length"])
        self.rate = int(self.meta["sample_rate"])
        self.encoder = make_session(REPO / tok.encoder_onnx, int(self.meta["num_rq"]))
        self.decoder = make_decoder(REPO / tok.decoder_onnx)
        self.codebooks = module.codebooks.to(device)  # type: ignore[union-attr]
        self.stats = module.stats

    def load_audio(self, path: Path, max_seconds: float | None = None) -> torch.Tensor:
        """
        Args:
          path (Path): mp3.
          max_seconds (float | None): truncate to this many seconds.

        Returns:
          torch.Tensor: (1, L) mono waveform, L a multiple of hop.
        """
        slices_dir = Path(os.path.expanduser(self.module.cfg.tokenizer.slices_dir))
        wav = load_track_audio(path, self.rate, self.hop, slices_dir)
        if max_seconds is not None:
            keep = int(max_seconds * self.rate) // self.hop * self.hop
            wav = wav[..., :keep]
        return wav

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
          wav (torch.Tensor): (1, L) waveform.

        Returns:
          torch.Tensor: (T, R) int64 tokens on the device.
        """
        tok = self.module.cfg.tokenizer
        idx = encode_chunked(self.encoder, wav, self.hop, tok.chunk_frames, tok.margin)
        return torch.from_numpy(idx).long().to(self.device)

    def decode(self, tokens: torch.Tensor, length: int) -> torch.Tensor:
        """
        Args:
          tokens (torch.Tensor): (T, R) indices.
          length (int): samples to keep.

        Returns:
          torch.Tensor: (1, length) waveform.
        """
        tok = self.module.cfg.tokenizer
        out = decode_tokens(
            self.decoder, tokens.unsqueeze(0), self.hop, tok.chunk_frames, tok.margin
        )
        return torch.from_numpy(out).reshape(1, -1)[:, :length]

    def whiten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
          tokens (torch.Tensor): (T, R) indices.

        Returns:
          torch.Tensor: (1, dims, T) whitened latent.
        """
        z = embed_zq(tokens.unsqueeze(0), self.codebooks).transpose(1, 2)
        return self.stats.whiten(z)

    def tokens_from_latent(self, x: torch.Tensor, beam: int) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (1, dims, T) whitened latent.
          beam (int): requantize beam width.

        Returns:
          torch.Tensor: (T, R) indices.
        """
        return requantize(self.stats.unwhiten(x), self.codebooks, beam=beam)[0]

    def style_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
          tokens (torch.Tensor): (T, R) indices.

        Returns:
          torch.Tensor: (1, C) unit-norm style descriptor.
        """
        return style_vector(embed_zq(tokens, self.codebooks)).unsqueeze(0)


def resolve_style(
    pipe: Pipeline, args: argparse.Namespace
) -> tuple[torch.Tensor | None, str]:
    """
    Pick the style vector the CLI asked for.

    Args:
      pipe (Pipeline): sessions and module.
      args (argparse.Namespace): CLI arguments.

    Returns:
      tuple[torch.Tensor | None, str]: (1, C) descriptor or None, and a tag.
    """
    if args.style_ref:
        path = Path(args.style_ref).expanduser()
        return (
            pipe.style_from_tokens(pipe.encode(pipe.load_audio(path))),
            f"ref-{path.stem[:16]}",
        )
    if args.style_track_idx is not None:
        tracks_dir = Path(os.path.expanduser(pipe.module.cfg.tokenizer.tracks_dir))
        path = enumerate_tracks(tracks_dir)[args.style_track_idx]
        print(f"style track {args.style_track_idx}: {path.name}")
        return (
            pipe.style_from_tokens(pipe.encode(pipe.load_audio(path))),
            f"trk{args.style_track_idx}",
        )
    return None, "uncond"


@torch.no_grad()
def sdedit_windowed(
    module: ZFlowModule,
    x: torch.Tensor,
    strength: float,
    style: torch.Tensor | None,
    window: int,
    batch: int,
    steps: int | None,
    churn: float | None,
    cfg_scale: float,
    seed: int,
) -> torch.Tensor:
    """
    SDEdit a long latent in half-overlapping windows with a triangular crossfade.

    Every window draws its noise from the same seeded generator so a run is
    reproducible; windows are batched for throughput.

    Args:
      module (ZFlowModule): the prior.
      x (torch.Tensor): (1, dims, T) whitened source latent.
      strength (float): SDEdit strength in [0, 1].
      style (torch.Tensor | None): (1, C) descriptor.
      window (int): frames per window, a multiple of the patch.
      batch (int): windows per forward.
      steps (int | None): Euler steps.
      churn (float | None): noise re-draw per step.
      cfg_scale (float): guidance.
      seed (int): noise seed.

    Returns:
      torch.Tensor: (1, dims, T) edited latent.
    """
    if strength <= 0.0:
        return x
    patch = module.cfg.model.patch
    total = x.shape[-1]
    hop = window // 2
    starts = list(range(0, max(1, total - window + 1), hop))
    if starts[-1] + window < total:
        starts.append(total - window)
    starts = [s - s % patch for s in starts]
    if total < window:
        starts = [0]
    ramp = torch.linspace(0.0, 1.0, hop, device=x.device)
    weight = torch.cat(
        [ramp, torch.ones(window - 2 * hop, device=x.device), ramp.flip(0)]
    ).clamp_min(1e-3)
    out = torch.zeros_like(x)
    norm = torch.zeros(1, 1, total, device=x.device)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    for i in range(0, len(starts), batch):
        chunk = starts[i : i + batch]
        crops = torch.cat([x[..., s : s + window] for s in chunk])
        lengths = [c.shape[-1] for c in crops]
        assert all(l == lengths[0] for l in lengths)
        cond = None if style is None else style.expand(crops.shape[0], -1)
        edited = module.sdedit(crops, strength, cond, steps, churn, cfg_scale, gen)
        for j, s in enumerate(chunk):
            span = min(window, total - s)
            out[..., s : s + span] += edited[j : j + 1, :, :span] * weight[:span]
            norm[..., s : s + span] += weight[:span]
    return out / norm.clamp_min(1e-6)


# ===========================================================================
# Scoring
# ===========================================================================


class Anchor:
    """
    What "sounds like IM" is measured against.

    Args:
      timbre (np.ndarray): (bands,) mean timbre profile of the anchor tracks.
      timbre_spread (float): mean profile distance of those tracks to it.
      zstyle (torch.Tensor): (C,) unit-norm centroid of the per-track style
        vectors of every training track (from the token cache, no audio).
      zstyle_spread (float): mean cosine distance of those tracks to it.
    """

    def __init__(
        self,
        timbre: np.ndarray,
        timbre_spread: float,
        zstyle: torch.Tensor,
        zstyle_spread: float,
    ) -> None:
        self.timbre = timbre
        self.timbre_spread = timbre_spread
        self.zstyle = zstyle
        self.zstyle_spread = zstyle_spread


TIMBRE_BANDS = 48


def im_anchor(pipe: Pipeline, out_dir: Path, count: int) -> Anchor:
    """
    Build (or load) the IM anchor: audio timbre over `count` training tracks,
    latent style over all of them.

    Args:
      pipe (Pipeline): for audio loading and the codebooks.
      out_dir (Path): render directory holding the cache file.
      count (int): tracks in the timbre average.

    Returns:
      Anchor: the reference.
    """
    path = out_dir / f"_im_anchor_{count}.pt"
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return Anchor(
            blob["timbre"],
            blob["timbre_spread"],
            blob["zstyle"].to(pipe.device),
            blob["zstyle_spread"],
        )
    tracks, held, _ = load_corpus(pipe.module.cfg)
    styles = torch.stack(
        [
            style_vector(embed_zq(t.tokens.to(pipe.device), pipe.codebooks))
            for t in tracks
            if t.track_idx not in held
        ]
    )
    zstyle = torch.nn.functional.normalize(styles.mean(0), dim=-1)
    zstyle_spread = float((1.0 - styles @ zstyle).mean())
    tracks_dir = Path(os.path.expanduser(pipe.module.cfg.tokenizer.tracks_dir))
    profiles = []
    for track in enumerate_tracks(tracks_dir)[:count]:
        pcm = pipe.load_audio(track).reshape(-1).numpy()
        profiles.append(timbre_profile(pcm, pipe.rate, TIMBRE_BANDS))
    timbre = np.mean(np.stack(profiles), axis=0)
    timbre_spread = float(np.mean([profile_distance(p, timbre) for p in profiles]))
    torch.save(
        {
            "timbre": timbre,
            "timbre_spread": timbre_spread,
            "zstyle": zstyle.cpu(),
            "zstyle_spread": zstyle_spread,
        },
        path,
    )
    print(
        f"IM anchor: timbre spread {timbre_spread:.3f}, zstyle spread {zstyle_spread:.3f}"
    )
    return Anchor(timbre, timbre_spread, zstyle, zstyle_spread)


def score(
    ref: torch.Tensor,
    pred: torch.Tensor,
    src_tokens: torch.Tensor,
    tokens: torch.Tensor,
    evaluator: Any,
    anchor: Anchor,
    codebooks: torch.Tensor,
    rate: int,
    cdpam_batch: int,
) -> dict[str, float]:
    """
    Args:
      ref (torch.Tensor): (1, L) source waveform.
      pred (torch.Tensor): (1, L) render.
      src_tokens (torch.Tensor): (T, R) source tokens.
      tokens (torch.Tensor): (T, R) render tokens.
      evaluator: cdpam evaluator.
      anchor (Anchor): IM reference.
      codebooks (torch.Tensor): (R, N, C) codebooks, for the latent style.
      rate (int): sample rate.
      cdpam_batch (int): slices per cdpam forward.

    Returns:
      dict[str, float]: cdpam, mrstft, timbre / zstyle distances to the
        anchor (in units of the anchor's own spread), beat, crest, token
        agreement per level.
    """
    out = slice_metrics(ref, pred, evaluator, cdpam_batch)
    pcm = pred.reshape(-1).numpy()
    prof = timbre_profile(pcm, rate, TIMBRE_BANDS)
    out["timbre"] = profile_distance(prof, anchor.timbre) / max(
        anchor.timbre_spread, 1e-6
    )
    zs = style_vector(embed_zq(tokens.to(codebooks.device), codebooks))
    out["zstyle"] = float(1.0 - zs @ anchor.zstyle) / max(anchor.zstyle_spread, 1e-6)
    feats = features(pcm, rate)
    out["beat"] = feats["beat"]
    out["crest"] = feats["crest"]
    agree = (tokens.cpu() == src_tokens.cpu()).float().mean(dim=0)
    for level in range(agree.shape[0]):
        out[f"agree_l{level}"] = float(agree[level])
    return out


def print_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    """
    Args:
      rows (list[tuple[str, dict[str, float]]]): (label, metrics) per render.
    """
    print(
        f"\n{'render':<34} {'cdpam':>6} {'mrstft':>6} {'timbre':>6} {'zstyle':>6} "
        f"{'beat':>6} {'crest':>6} {'L0':>5} {'L1':>5} {'L2':>5}"
    )
    print(
        "timbre / zstyle: distance to the IM anchor over the IM tracks' own spread (1 = typical IM track)"
    )
    print("-" * 99)
    for label, m in rows:
        print(
            f"{label:<34} {m['cdpam']:6.3f} {m['mrstft']:6.3f} {m['timbre']:6.2f} "
            f"{m['zstyle']:6.2f} {m['beat']:6.3f} {m['crest']:6.2f} "
            f"{100 * m['agree_l0']:4.0f}% {100 * m['agree_l1']:4.0f}% {100 * m['agree_l2']:4.0f}%"
        )


# ===========================================================================
# Modes
# ===========================================================================


def run_stylize(pipe: Pipeline, args: argparse.Namespace, out_dir: Path) -> None:
    """
    Args:
      pipe (Pipeline): sessions and module.
      args (argparse.Namespace): CLI arguments.
      out_dir (Path): render directory.
    """
    module = pipe.module
    window = args.window or module.cfg.data.crop_frames
    style, style_tag = resolve_style(pipe, args)
    evaluator = make_cdpam_evaluator(args.device)
    anchor = im_anchor(pipe, out_dir, args.anchor_tracks)
    for track in args.tracks:
        path = Path(track).expanduser()
        wav = pipe.load_audio(path, args.max_seconds)
        length = wav.shape[-1]
        stem = path.stem[:32].strip().replace(" ", "_")
        print(
            f"\n{path.name}: {length / pipe.rate:.1f} s, {length // pipe.hop} frames, style {style_tag}"
        )
        src_tokens = pipe.encode(wav)
        patch = module.cfg.model.patch
        pad = (-src_tokens.shape[0]) % patch
        x = pipe.whiten_tokens(torch.cat([src_tokens, src_tokens[-1:].expand(pad, -1)]))
        torchaudio.save(str(out_dir / f"{stem}__A_original.wav"), wav, pipe.rate)
        rt = pipe.decode(src_tokens, length)
        torchaudio.save(str(out_dir / f"{stem}__B_roundtrip.wav"), rt, pipe.rate)
        rows = [
            (
                "A original",
                score(
                    wav,
                    wav,
                    src_tokens,
                    src_tokens,
                    evaluator,
                    anchor,
                    pipe.codebooks,
                    pipe.rate,
                    args.cdpam_batch,
                ),
            ),
            (
                "B roundtrip",
                score(
                    wav,
                    rt,
                    src_tokens,
                    src_tokens,
                    evaluator,
                    anchor,
                    pipe.codebooks,
                    pipe.rate,
                    args.cdpam_batch,
                ),
            ),
        ]
        for strength, cfg_scale in itertools.product(args.strength, args.cfg):
            if cfg_scale != 1.0 and style is None:
                continue
            edited = sdedit_windowed(
                module,
                x,
                strength,
                style,
                window,
                args.window_batch,
                args.steps,
                args.churn,
                cfg_scale,
                args.seed,
            )
            tokens = pipe.tokens_from_latent(edited, args.beam)[: src_tokens.shape[0]]
            out = pipe.decode(tokens, length)
            label = f"s{strength:.2f}_cfg{cfg_scale:g}_{style_tag}"
            torchaudio.save(str(out_dir / f"{stem}__{label}.wav"), out, pipe.rate)
            np.save(out_dir / f"{stem}__{label}_tokens.npy", tokens.cpu().numpy())
            rows.append(
                (
                    label,
                    score(
                        wav,
                        out,
                        src_tokens,
                        tokens,
                        evaluator,
                        anchor,
                        pipe.codebooks,
                        pipe.rate,
                        args.cdpam_batch,
                    ),
                )
            )
            print(
                f"  {label}: cdpam {rows[-1][1]['cdpam']:.3f} timbre {rows[-1][1]['timbre']:.2f} zstyle {rows[-1][1]['zstyle']:.2f}"
            )
        print_table(rows)


def run_sample(pipe: Pipeline, args: argparse.Namespace, out_dir: Path) -> None:
    """
    Args:
      pipe (Pipeline): sessions and module.
      args (argparse.Namespace): CLI arguments.
      out_dir (Path): render directory.
    """
    module = pipe.module
    patch = module.cfg.model.patch
    frames = int(args.seconds * pipe.meta["frames_per_second"]) // patch * patch
    style, style_tag = resolve_style(pipe, args)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    for cfg_scale in args.cfg:
        if cfg_scale != 1.0 and style is None:
            continue
        noise = torch.randn(
            args.n, module.stats.dims, frames, generator=gen, device=args.device
        )
        cond = None if style is None else style.expand(args.n, -1)
        x = module.sample(noise, 0.0, cond, args.steps, args.churn, cfg_scale, gen)
        for i in range(args.n):
            tokens = pipe.tokens_from_latent(x[i : i + 1], args.beam)
            wav = pipe.decode(tokens, frames * pipe.hop)
            name = f"sample{i}_cfg{cfg_scale:g}_{style_tag}_seed{args.seed}.wav"
            torchaudio.save(str(out_dir / name), wav, pipe.rate)
            feats = features(wav.reshape(-1).numpy(), pipe.rate)
            print(
                f"{name}: crest {feats['crest']:.2f} beat {feats['beat']:.3f} std {float(x[i].std()):.3f}"
            )


def main() -> None:
    """Load the prior, then stylise or sample."""
    args = parse_args()
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    module = load_zflow_module(
        REPO / args.ckpt, ema=not args.no_ema, device=args.device
    )
    pipe = Pipeline(module, args.device)
    if args.mode == "stylize":
        run_stylize(pipe, args, out_dir)
    else:
        run_sample(pipe, args, out_dir)


if __name__ == "__main__":
    main()
