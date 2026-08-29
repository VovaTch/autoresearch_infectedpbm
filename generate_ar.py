"""
Sample audio from a trained AR checkpoint, and decode the reference alongside it.

train_ar.py can train the model but has no sampling loop and no tokens->audio
path, so nothing in the repo could turn a checkpoint into something audible.
This closes that gap.

The default run writes THREE wavs per request, which is the point -- a bare
generation is undiagnosable on its own:

  *_recon.wav   cached tokens for the track, decoded. The CEILING: this is the
                best the AR stage could ever sound, since it is the tokenizer's
                own reconstruction of real tokens. If this sounds bad, the
                problem is upstream of the transformer.
  *_prompt.wav  the priming frames alone, decoded (when --prompt-sec > 0).
  *_gen.wav     the model's continuation.

Comparing gen against recon separates "the AR is broken" from "the tokenizer is
the limit", which listening to gen alone cannot do.

Usage:
  uv run python generate_ar.py --ckpt saved_ar_smoke/ar_best.ckpt --seconds 10
  uv run python generate_ar.py --ckpt ... --prompt-sec 0 --temperature 0.9
  uv run python generate_ar.py --ckpt ... --recon-only
"""

from __future__ import annotations

import argparse
import ctypes
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from train_ar import (
    _pick_style,
    ArConfig,
    ArTransformer,
    DataCfg,
    ModelCfg,
    TokenizerCfg,
    TrackTokens,
    TrainCfg,
    build_delay_grid,
    build_model,
    load_token_cache,
    undelay_grid,
)

REPO = Path(__file__).resolve().parent


def config_from_ckpt(ckpt: dict[str, Any]) -> ArConfig:
    """
    Rebuild the ArConfig a checkpoint was trained with.

    Args:
      ckpt (dict[str, Any]): loaded lightning checkpoint.

    Returns:
      ArConfig: config reconstructed from saved hyper_parameters.
    """
    raw = ckpt["hyper_parameters"]["config"]
    return ArConfig(
        tokenizer=TokenizerCfg(**raw["tokenizer"]),
        data=DataCfg(**raw["data"]),
        model=ModelCfg(**raw["model"]),
        train=TrainCfg(**raw["train"]),
    )


def _preload_cudnn() -> None:
    """Load cuDNN into the process before onnxruntime opens its CUDA provider."""
    for name in ("libcudnn.so.9", "libcudnn.so"):
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def make_decoder(path: Path, use_gpu: bool = True):
    """
    Open the ONNX decoder graph.

    TF32 is pinned off: onnxruntime enables it by default on Ampere, which
    silently breaks bit-parity with the PyTorch decoder.

    Args:
      path (Path): decoder.onnx path.
      use_gpu (bool): try the CUDA provider first.

    Returns:
      onnxruntime.InferenceSession: ready session.
    """
    import onnxruntime as ort

    providers: list[Any] = []
    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        _preload_cudnn()
        providers.append(("CUDAExecutionProvider", {"use_tf32": 0}))
    providers.append("CPUExecutionProvider")
    try:
        return ort.InferenceSession(str(path), providers=providers)
    except Exception as exc:  # noqa: BLE001 - provider failures are environmental
        print(f"  decoder: CUDA provider unavailable ({exc}); falling back to CPU")
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def decode_tokens(
    session, tokens: torch.Tensor, hop: int, chunk: int = 4096, margin: int = 256
) -> np.ndarray:
    """
    Decode RVQ indices to a waveform, in overlap-trimmed chunks.

    Args:
      session: onnxruntime session for decoder.onnx.
      tokens (torch.Tensor): (1, T, R) int64 indices.
      hop (int): samples per frame.
      chunk (int): frames decoded per window.
      margin (int): context frames discarded per side.

    Returns:
      np.ndarray: (1, 1, T * hop) float32 waveform.
    """
    name = session.get_inputs()[0].name
    idx = tokens.cpu().numpy().astype(np.int64)
    total = idx.shape[1]
    if total <= chunk:
        return session.run(None, {name: idx})[0]
    out, pos = [], 0
    while pos < total:
        end = min(pos + chunk, total)
        lo, hi = max(0, pos - margin), min(total, end + margin)
        wav = session.run(None, {name: idx[:, lo:hi]})[0]
        out.append(wav[..., (pos - lo) * hop : wav.shape[-1] - (hi - end) * hop])
        pos = end
    return np.concatenate(out, axis=-1)


def sample_step(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample one token per RVQ depth from a single position's logits.

    Args:
      logits (torch.Tensor): (R, V) logits for one position.
      temperature (float): softmax temperature; <= 0 selects argmax.
      top_k (int): keep only the k highest logits (0 disables).
      top_p (float): nucleus threshold (0 disables).
      generator (torch.Generator | None): per-clip RNG. Passing one keeps a clip
        reproducible from its seed even when several clips are sampled in the
        same batch, where the global RNG would interleave their draws.

    Returns:
      torch.Tensor: (R,) int64 sampled indices.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p > 0:
        ordered, order = logits.sort(dim=-1, descending=True)
        cum = ordered.softmax(dim=-1).cumsum(dim=-1)
        # keep the first token past the threshold, so p<top_p never empties the set
        drop = cum - ordered.softmax(dim=-1) > top_p
        ordered = ordered.masked_fill(drop, float("-inf"))
        logits = ordered.gather(-1, order.argsort(dim=-1))
    return torch.multinomial(
        logits.softmax(dim=-1), num_samples=1, generator=generator
    ).squeeze(-1)


@torch.no_grad()
def generate(
    model: ArTransformer,
    track_idx: int,
    style: torch.Tensor,
    frames: int,
    prompt: torch.Tensor | None,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Autoregressively sample a delayed grid and undelay it to aligned codes.

    Mirrors ArLightningModule._prepare exactly: the model sees a BOS row of
    pad_id followed by the delayed grid shifted right by one, so position i is
    scored from grid[i-1]. Any drift from that is a train/generate mismatch.

    No KV cache -- every step re-runs the full prefix, so cost is O(L^2). At
    ~4000 frames that is a couple of minutes, which is cheap enough to not
    justify reaching into the trained attention module.

    Args:
      model (ArTransformer): trained model in eval mode.
      track_idx (int): conditioning track id.
      style (torch.Tensor): (style_dim,) style descriptor.
      frames (int): number of aligned frames T to produce.
      prompt (torch.Tensor | None): (P, R) real codes to prime with.
      temperature (float): sampling temperature.
      top_k (int): top-k cutoff.
      top_p (float): nucleus cutoff.
      device (torch.device): compute device.

    Returns:
      torch.Tensor: (1, T, R) int64 aligned codes.
    """
    depth, pad = model.num_rq, model.pad_id
    length = frames + depth - 1

    ids = torch.tensor([track_idx], device=device, dtype=torch.long)
    styles = style[None].to(device).float()
    keep = torch.zeros(1, dtype=torch.bool, device=device)

    inputs = torch.full((1, 1, depth), pad, dtype=torch.long, device=device)
    grid = torch.empty((1, 0, depth), dtype=torch.long, device=device)

    # Priming: force the first positions to the real delayed grid instead of
    # sampling them, so the model continues actual music rather than its own
    # cold start.
    #
    # Only the entries the prompt actually covers may be forced. build_delay_grid
    # pads the corners of the delayed grid, and its trailing corner lands on real
    # output frames: forcing it wrote pad_id -- one past the last codebook entry
    # -- into the depth-1 frames straight after every prompt, which the decoder
    # then gathered out of range. Level d of grid row s holds aligned frame s-d,
    # so an entry is real exactly when 0 <= s-d < len(prompt).
    forced = None
    prompt_len = 0
    if prompt is not None and prompt.numel():
        forced = build_delay_grid(prompt[None].to(device), pad)[0]
        prompt_len = int(prompt.shape[0])
    offsets = torch.arange(depth, device=device)

    for step in range(length):
        logits = model(inputs, ids, styles, keep, keep)[0, -1]
        nxt = sample_step(logits.float(), temperature, top_k, top_p)
        if forced is not None and step < forced.shape[0]:
            aligned = step - offsets
            real = (aligned >= 0) & (aligned < prompt_len)
            nxt = torch.where(real, forced[step], nxt)
        nxt = nxt.view(1, 1, depth)
        grid = torch.cat([grid, nxt], dim=1)
        if step + 1 < length:
            inputs = torch.cat([inputs, nxt], dim=1)
        if (step + 1) % 512 == 0:
            print(f"    {step + 1}/{length} positions", flush=True)
    return undelay_grid(grid, frames)


def token_stats(tag: str, tokens: torch.Tensor, num_tokens: int) -> None:
    """
    Report degeneracy signatures in a token stream.

    "It sounds bad" has distinct causes that look different here: collapse
    (few distinct codes, low entropy), looping (a short period repeats), and
    drift (stats fine but unlike the reference). Printing these next to the
    reference's own numbers says which one is happening.

    Args:
      tag (str): label for the row.
      tokens (torch.Tensor): (T, R) int64 codes.
      num_tokens (int): codebook size, for the entropy ceiling.
    """
    frames, depth = tokens.shape
    parts = []
    for d in range(depth):
        col = tokens[:, d]
        counts = torch.bincount(col, minlength=num_tokens).float()
        probs = counts[counts > 0] / counts.sum()
        ent = float(-(probs * probs.log2()).sum() / np.log2(num_tokens))
        parts.append(f"L{d} uniq {int((counts > 0).sum()):4d} H {ent:.3f}")
    # loop check: smallest lag whose shifted copy matches often
    loop = "none"
    flat = tokens[:, 0]
    for lag in range(1, min(512, frames // 2)):
        if float((flat[lag:] == flat[:-lag]).float().mean()) > 0.9:
            loop = f"lag {lag} ({lag / 172.265625:.2f}s)"
            break
    print(f"  {tag:<10} {'  '.join(parts)}  loop {loop}")


def pick_track(tracks: list[TrackTokens], want: str | None) -> TrackTokens:
    """
    Choose the track to condition on.

    Args:
      tracks (list[TrackTokens]): loaded corpus.
      want (str | None): case-insensitive substring, or None for the first track.

    Returns:
      TrackTokens: the selected track.
    """
    if want is None:
        return tracks[0]
    hits = [t for t in tracks if want.lower() in t.track_name.lower()]
    if not hits:
        names = ", ".join(t.track_name[:40] for t in tracks[:5])
        raise SystemExit(f"no track matching {want!r}; loaded: {names}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="saved_ar_smoke/ar_best.ckpt")
    ap.add_argument("--decoder", default="onnx/decoder.onnx")
    ap.add_argument("--track", default=None, help="substring; default = first loaded")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--prompt-sec", type=float, default=3.0, help="0 = cold start")
    ap.add_argument("--start-frac", type=float, default=0.35, help="where to prompt from")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=250)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recon-only", action="store_true", help="skip sampling")
    ap.add_argument("--out", default="renders_ar")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(REPO / args.ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_ckpt(ckpt)

    cache_root = Path(cfg.tokenizer.cache_root).expanduser()
    caches = sorted(cache_root.glob("tokens_*"))
    if not caches:
        raise SystemExit(f"no token cache under {cache_root}")
    # Load the whole corpus: single_track would hide the track we want to name.
    data_cfg = DataCfg(**{**vars(cfg.data), "single_track": None})
    tracks, manifest = load_token_cache(caches[-1], data_cfg)
    meta = manifest["tokenizer_meta"]
    fps, hop = meta["frames_per_second"], meta["hop_length"]
    sr = int(meta["sample_rate"])

    model = build_model(cfg, tracks, manifest)
    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  WARN missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}")
    model.to(device).eval()

    want = args.track if args.track is not None else cfg.data.single_track
    track = pick_track(tracks, want)
    trained_on = cfg.data.single_track
    print(
        f"ckpt {args.ckpt}  epoch {ckpt.get('epoch')} step {ckpt.get('global_step')}\n"
        f"track '{track.track_name}' (id {track.track_idx}, {track.num_frames} frames)\n"
        f"trained on: {trained_on or 'full corpus'}  "
        f"p_drop_cond={cfg.model.p_drop_cond}"
    )
    if cfg.model.p_drop_cond == 0.0:
        print(
            "  NOTE p_drop_cond=0: the CFG nulls were never trained, so guided "
            "sampling is meaningless here. Sampling plain conditional."
        )

    out_dir = REPO / args.out
    out_dir.mkdir(exist_ok=True)
    stem = f"{Path(args.ckpt).parent.name}_{track.track_name[:32]}"
    session = make_decoder(REPO / args.decoder)

    n_frames = int(args.seconds * fps)
    n_prompt = int(args.prompt_sec * fps)
    start = int(track.num_frames * args.start_frac)
    start = max(0, min(start, track.num_frames - n_frames - 1))
    ref = track.tokens[start : start + n_frames].long()

    # Section 11.3: training always draws the style descriptor from a window
    # DISJOINT from the crop, so conditioning never carries the answer. Sampling
    # has to honour the same rule or it is conditioned on out-of-distribution
    # input -- and would flatter itself by echoing the target.
    style_idx = _pick_style(track, start, start + n_frames, random.Random(args.seed))
    bounds = track.style_bounds[style_idx]
    style = track.style[style_idx]
    print(
        f"style window {style_idx} = frames {int(bounds[0])}-{int(bounds[1])} "
        f"({int(bounds[0]) / fps:.1f}-{int(bounds[1]) / fps:.1f}s), "
        f"generated span {start}-{start + n_frames}"
    )

    def write(tag: str, tokens: torch.Tensor) -> None:
        wav = decode_tokens(session, tokens, hop)
        path = out_dir / f"{stem}_{tag}.wav"
        torchaudio.save(str(path), torch.from_numpy(wav[0]), sr)
        peak = float(np.abs(wav).max())
        print(f"  wrote {path.name}  {wav.shape[-1] / sr:.1f}s  peak {peak:.3f}")

    print(f"\ndecoding reference tokens (the ceiling) from frame {start}")
    write("recon", ref[None])
    if n_prompt:
        write("prompt", ref[:n_prompt][None])
    if args.recon_only:
        return

    print(f"\nsampling {n_frames} frames (T={args.temperature} k={args.top_k} p={args.top_p})")
    gen = generate(
        model,
        track.track_idx,
        style,
        n_frames,
        ref[:n_prompt] if n_prompt else None,
        args.temperature,
        args.top_k,
        args.top_p,
        device,
    )
    if n_prompt:
        match = (gen[0, :n_prompt] == ref[:n_prompt].to(device)).float().mean()
        print(f"  prompt reproduced exactly: {match:.1%} (should be 100%)")
    novel = (gen[0, n_prompt:] != ref[n_prompt:].to(device)).float().mean()
    print(f"  continuation differs from reference in {novel:.1%} of codes")

    print("\ntoken statistics (compare gen against reference)")
    nt = int(meta["num_tokens"])
    token_stats("reference", ref[n_prompt:], nt)
    token_stats("generated", gen[0, n_prompt:].cpu(), nt)
    write("gen", gen.cpu())


if __name__ == "__main__":
    main()
