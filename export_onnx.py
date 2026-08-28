"""
Export the lvl1_vqgan tokenizer to two self-contained ONNX graphs.

    encoder.onnx : waveform (B, 1, L) float32 -> token indices (B, T, R) int64
    decoder.onnx : token indices (B, T, R) int64 -> waveform (B, 1, L) float32

Split rather than monolithic because the generation stage never runs the two
together: the AR model consumes encoder tokens offline and only ever calls the
decoder in its sampling loop.

Three ops in the training graph do not survive a naive export, and each is
replaced here by an exactly equivalent fixed-weight layer:

    torch.stft (complex)          -> reflect pad + Conv1d          (ConvSTFT)
    torch.fft.irfft + F.fold      -> ConvTranspose1d + envelope    (ConvISTFT)
    VQCodeBookFunc / torch.cdist  -> matmul + argmin               (rq_tokenize)

Every replacement is verified against the original torch op at export time, so
a silent numerical drift fails the run instead of shipping.

Example:
    uv run python export_onnx.py --ckpt saved_20260827_cont9h/lvl1_vqgan_last.ckpt \
        --out-dir onnx_cont9h
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from train import (
    build_learning_params,
    build_loss_aggregator,
    build_module,
    build_optimizer_cfg,
    build_scheduler_cfg,
)


# ===========================================================================
# Fixed-weight replacements for the non-exportable spectral ops
# ===========================================================================


def rfft_basis(n_fft: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Real/imag analysis basis of torch.fft.rfft, built numerically.

    Derived by transforming the identity rather than from a closed form, so the
    sign and normalization conventions match torch exactly by construction.

    Args:
      n_fft (int): transform size.

    Returns:
      tuple[torch.Tensor, torch.Tensor]: real and imag bases, each
        (n_fft // 2 + 1, n_fft).
    """
    spec = torch.fft.rfft(torch.eye(n_fft, dtype=torch.float64), dim=-1)
    return spec.real.T.contiguous().float(), spec.imag.T.contiguous().float()


def irfft_basis(n_fft: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Real/imag synthesis basis of torch.fft.irfft(..., norm="backward").

    Args:
      n_fft (int): transform size.

    Returns:
      tuple[torch.Tensor, torch.Tensor]: bases mapping the real and imag parts
        of a spectrum to the time frame, each (n_fft // 2 + 1, n_fft).
    """
    bins = n_fft // 2 + 1
    eye = torch.eye(bins, dtype=torch.float64)
    zero = torch.zeros_like(eye)
    b_real = torch.fft.irfft(torch.complex(eye, zero), n=n_fft, dim=-1, norm="backward")
    b_imag = torch.fft.irfft(torch.complex(zero, eye), n=n_fft, dim=-1, norm="backward")
    return b_real.contiguous().float(), b_imag.contiguous().float()


class ConvSTFT(nn.Module):
    """
    torch.stft(center=True, pad_mode="reflect") as a reflect pad plus Conv1d.

    Emits the real and imag parts stacked on the channel axis, matching the
    encoder's own `cat([spec.real, spec.imag], dim=1)` layout.

    Args:
      n_fft (int): transform size.
      hop_length (int): frame stride in samples.
      win_length (int): Hann window length.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad = n_fft // 2
        window = torch.hann_window(win_length, dtype=torch.float64).float()
        b_real, b_imag = rfft_basis(n_fft)
        # conv1d is cross-correlation, so the basis needs no flip
        weight = torch.cat([b_real, b_imag], dim=0) * window[None, :]
        self.register_buffer("weight", weight.unsqueeze(1))  # (2*bins, 1, n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x (torch.Tensor): (B, 1, L) waveform.

        Returns:
          torch.Tensor: (B, 2 * (n_fft // 2 + 1), L // hop_length) spectrogram
            with real and imag stacked on channels.
        """
        x = F.pad(x, (self.pad, self.pad), mode="reflect")
        spec = F.conv1d(x, self.weight, stride=self.hop_length)
        # torch.stft yields L // hop + 1 frames; the encoder keeps L // hop
        return spec[..., :-1]


class ConvISTFT(nn.Module):
    """
    The decoder's ISTFT(padding="same") as a single ConvTranspose1d.

    The window, the inverse DFT and the overlap-add are all linear with fixed
    coefficients, so they fuse into one transposed convolution. The Hann
    envelope is rebuilt in-graph from a ones tensor, which keeps the frame
    count dynamic.

    Args:
      n_fft (int): transform size.
      hop_length (int): frame stride in samples.
      win_length (int): Hann window length.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.trim = (win_length - hop_length) // 2
        window = torch.hann_window(win_length, dtype=torch.float64).float()
        b_real, b_imag = irfft_basis(n_fft)
        weight = torch.cat([b_real, b_imag], dim=0) * window[None, :]
        # conv_transpose1d: out[t * stride + k] += sum_c inp[c, t] * w[c, 0, k]
        self.register_buffer("weight", weight.unsqueeze(1))  # (2*bins, 1, n_fft)
        self.register_buffer("env_weight", window.square().reshape(1, 1, -1))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
          spec (torch.Tensor): (B, 2 * (n_fft // 2 + 1), T) real|imag spectrum.

        Returns:
          torch.Tensor: (B, 1, T * hop_length) waveform.
        """
        y = F.conv_transpose1d(spec, self.weight, stride=self.hop_length)
        ones = torch.ones_like(spec[:, :1])
        env = F.conv_transpose1d(ones, self.env_weight, stride=self.hop_length)
        y = y[..., self.trim : -self.trim] / env[..., self.trim : -self.trim]
        return y


def rq_tokenize(z_e: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """
    Residual-quantize latents to indices without cdist or autograd.Function.

    Replaces the Euclidean distance with ||e||^2 - 2 z.e; the dropped ||z||^2
    term is constant per position and cannot change the argmin.

    Args:
      z_e (torch.Tensor): (B, C, T) encoder latents.
      codebooks (torch.Tensor): (R, N, C) per-level codes.

    Returns:
      torch.Tensor: (B, T, R) int64 indices.
    """
    residual = z_e.transpose(1, 2)  # (B, T, C)
    picks: list[torch.Tensor] = []
    for level in range(codebooks.shape[0]):
        codes = codebooks[level]  # (N, C)
        scores = residual @ codes.T - 0.5 * codes.pow(2).sum(-1)[None, None, :]
        idx = scores.argmax(dim=-1)  # argmax(-dist^2/2) == argmin(dist)
        residual = residual - codes[idx]
        picks.append(idx)
    return torch.stack(picks, dim=-1)


# ===========================================================================
# Export wrappers
# ===========================================================================


class OnnxEncoder(nn.Module):
    """
    Waveform to RQ token indices, reusing the trained submodules by reference.

    Args:
      net (nn.Module): a MultiLvlVQVariationalAutoEncoder.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        enc = net.encoder
        if enc._ze_norm != "none":
            raise NotImplementedError(f"ze_norm={enc._ze_norm!r} not wired for export")
        self.stft = ConvSTFT(enc._n_fft, enc._hop_length, enc._win_length)
        self.proj_in, self.res1 = enc._proj_in, enc._res1
        self.down, self.res2, self.proj_out = enc._down, enc._res2, enc._proj_out
        cb = net.vq_module.vq_codebook
        self.register_buffer("codebooks", cb.code_embedding.detach().clone())

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
          wav (torch.Tensor): (B, 1, L) waveform, L a multiple of hop_length.

        Returns:
          torch.Tensor: (B, L // hop_length, R) int64 token indices.
        """
        h = self.proj_in(self.stft(wav))
        h = self.res2(self.down(self.res1(h)))
        return rq_tokenize(self.proj_out(h), self.codebooks)


class OnnxDecoder(nn.Module):
    """
    RQ token indices back to a waveform.

    Args:
      net (nn.Module): a MultiLvlVQVariationalAutoEncoder.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        dec = net.decoder
        self.bins = dec._spec_bins
        self.proj_in, self.res1 = dec._proj_in, dec._res1
        self.up, self.res2, self.proj_spec = dec._up, dec._res2, dec._proj_spec
        self.end_conv = dec._end_conv
        istft = dec._istft
        self.istft = ConvISTFT(istft.n_fft, istft.hop_length, istft.win_length)
        cb = net.vq_module.vq_codebook
        self.register_buffer("codebooks", cb.code_embedding.detach().clone())

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
          indices (torch.Tensor): (B, T, R) int64 token indices.

        Returns:
          torch.Tensor: (B, 1, T * hop_length) waveform.
        """
        z_q = torch.zeros(
            indices.shape[0],
            indices.shape[1],
            self.codebooks.shape[-1],
            dtype=self.codebooks.dtype,
            device=indices.device,
        )
        for level in range(self.codebooks.shape[0]):
            z_q = z_q + self.codebooks[level][indices[..., level]]
        h = self.proj_in(z_q.transpose(1, 2))
        h = self.res2(self.up(self.res1(h)))
        return self.end_conv(self.istft(self.proj_spec(h)))


# ===========================================================================
# Verification
# ===========================================================================


def check(name: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> float:
    """
    Compare a replacement against its reference and raise if it drifted.

    Args:
      name (str): label for the report line.
      got (torch.Tensor): replacement output.
      ref (torch.Tensor): reference output.
      tol (float): max absolute difference allowed.

    Returns:
      float: the observed max absolute difference.
    """
    if got.shape != ref.shape:
        raise AssertionError(f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}")
    err = float((got.float() - ref.float()).abs().max())
    flag = "ok " if err <= tol else "FAIL"
    print(f"  [{flag}] {name:28s} max|diff| = {err:.3e}  (tol {tol:.0e})")
    if err > tol:
        raise AssertionError(f"{name} exceeded tolerance: {err:.3e} > {tol:.0e}")
    return err


def verify_ops(net: nn.Module, wav: torch.Tensor) -> None:
    """
    Check ConvSTFT / ConvISTFT / rq_tokenize against the training ops.

    Args:
      net (nn.Module): the loaded autoencoder.
      wav (torch.Tensor): (B, 1, L) probe waveform.
    """
    enc, dec = net.encoder, net.decoder
    print("op-level equivalence:")

    stft = ConvSTFT(enc._n_fft, enc._hop_length, enc._win_length)
    ref_c = torch.stft(
        wav.flatten(start_dim=1),
        n_fft=enc._n_fft,
        hop_length=enc._hop_length,
        win_length=enc._win_length,
        window=torch.hann_window(enc._win_length),
        return_complex=True,
    )
    n_frames = wav.shape[-1] // enc._hop_length
    ref = torch.cat([ref_c.real, ref_c.imag], dim=1)[..., :n_frames]
    check("ConvSTFT vs torch.stft", stft(wav), ref, 1e-3)

    istft = ConvISTFT(dec._istft.n_fft, dec._istft.hop_length, dec._istft.win_length)
    spec = torch.randn(wav.shape[0], dec._spec_bins * 2, n_frames)
    ref_wav = dec._istft(torch.complex(spec[:, : dec._spec_bins], spec[:, dec._spec_bins :]))
    check("ConvISTFT vs ISTFT", istft(spec).squeeze(1), ref_wav, 1e-3)

    with torch.no_grad():
        z_e = net.encode(wav)
        ref_idx = net.vq_module(z_e)["indices"]
        got_idx = rq_tokenize(z_e, net.vq_module.vq_codebook.code_embedding)
    same = int((got_idx == ref_idx).sum()), ref_idx.numel()
    print(f"  [{'ok ' if same[0] == same[1] else 'FAIL'}] rq_tokenize vs cdist"
          f"          {same[0]}/{same[1]} indices identical")
    if same[0] != same[1]:
        raise AssertionError("rq_tokenize picked different codes than cdist")


def verify_wrappers(
    net: nn.Module, encoder: nn.Module, decoder: nn.Module, wav: torch.Tensor
) -> None:
    """
    Check the two wrappers reproduce tokenize() and from_tokens() end to end.

    Args:
      net (nn.Module): the loaded autoencoder.
      encoder (nn.Module): OnnxEncoder wrapper.
      decoder (nn.Module): OnnxDecoder wrapper.
      wav (torch.Tensor): (B, 1, L) probe waveform.
    """
    print("wrapper equivalence (vs train.py):")
    with torch.no_grad():
        ref_idx = net.tokenize(wav)
        got_idx = encoder(wav)
        n_same = int((got_idx == ref_idx).sum())
        print(f"  [{'ok ' if n_same == ref_idx.numel() else 'FAIL'}] encoder vs tokenize"
              f"           {n_same}/{ref_idx.numel()} indices identical")
        if n_same != ref_idx.numel():
            raise AssertionError("encoder wrapper disagrees with tokenize()")
        check("decoder vs from_tokens", decoder(ref_idx), net.from_tokens(ref_idx), 2e-3)
        check("round-trip vs forward", decoder(encoder(wav)),
              net(wav)["slice"].reshape(wav.shape), 2e-3)


def verify_onnx(path: Path, inputs: dict[str, torch.Tensor], ref: torch.Tensor,
                tol: float) -> None:
    """
    Run the exported graph under onnxruntime and compare against PyTorch.

    Args:
      path (Path): the .onnx file.
      inputs (dict[str, torch.Tensor]): feed dict.
      ref (torch.Tensor): expected output.
      tol (float): max absolute difference allowed.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {k: v.numpy() for k, v in inputs.items()}
    got = torch.from_numpy(sess.run(None, feed)[0])
    check(f"onnxruntime {path.name}", got, ref, tol)


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    """Build, verify and export the encoder and decoder graphs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="saved_20260827_cont9h/lvl1_vqgan_last.ckpt")
    ap.add_argument("--out-dir", default="onnx")
    ap.add_argument("--token-dim", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--num-rq", type=int, default=3)
    ap.add_argument("--num-tokens", type=int, default=2048)
    ap.add_argument("--time-downsample", type=int, default=1)
    ap.add_argument("--ze-norm", default="none")
    ap.add_argument("--per-level-codebooks", action="store_true", default=True)
    ap.add_argument("--slice-length", type=int, default=32768)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-dynamo", action="store_true", help="use the legacy exporter")
    ap.add_argument(
        "--external-data",
        action="store_true",
        help="write weights to .onnx.data sidecars instead of one self-contained "
        "file; needed only if a graph ever exceeds the 2GB protobuf limit",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.ckpt}")
    module = build_module(
        build_learning_params(),
        build_loss_aggregator(),
        build_optimizer_cfg(),
        build_scheduler_cfg(),
        token_dim=args.token_dim,
        num_rq_steps=args.num_rq,
        num_tokens=args.num_tokens,
        time_downsample=args.time_downsample,
        hidden=args.hidden,
        ze_norm=args.ze_norm,
        per_level_codebooks=args.per_level_codebooks,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    module.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt,
                           strict=True)
    module.eval()
    net = module.model
    for p in net.parameters():
        p.requires_grad_(False)

    wav = torch.randn(1, 1, args.slice_length) * 0.1
    verify_ops(net, wav)

    encoder, decoder = OnnxEncoder(net).eval(), OnnxDecoder(net).eval()
    verify_wrappers(net, encoder, decoder, wav)

    with torch.no_grad():
        ref_idx = encoder(wav)
        ref_wav = decoder(ref_idx)

    enc_path, dec_path = out_dir / "encoder.onnx", out_dir / "decoder.onnx"
    kw = {
        "opset_version": args.opset,
        "dynamo": not args.no_dynamo,
        "external_data": args.external_data,
    }
    print(f"exporting (opset {args.opset}, dynamo={not args.no_dynamo})")

    torch.onnx.export(
        encoder, (wav,), str(enc_path),
        input_names=["waveform"], output_names=["indices"],
        dynamic_axes={"waveform": {0: "batch", 2: "samples"},
                      "indices": {0: "batch", 1: "frames"}},
        **kw,
    )
    torch.onnx.export(
        decoder, (ref_idx,), str(dec_path),
        input_names=["indices"], output_names=["waveform"],
        dynamic_axes={"indices": {0: "batch", 1: "frames"},
                      "waveform": {0: "batch", 2: "samples"}},
        **kw,
    )

    print("onnxruntime parity:")
    verify_onnx(enc_path, {"waveform": wav}, ref_idx, 0)
    verify_onnx(dec_path, {"indices": ref_idx}, ref_wav, 2e-3)

    enc_h = net.encoder
    meta = {
        "checkpoint": args.ckpt,
        "sample_rate": 44100,
        "n_fft": enc_h._n_fft,
        "hop_length": enc_h._hop_length,
        "win_length": enc_h._win_length,
        "frames_per_second": 44100 / enc_h._hop_length / args.time_downsample,
        "token_dim": args.token_dim,
        "num_rq": args.num_rq,
        "num_tokens": args.num_tokens,
        "per_level_codebooks": args.per_level_codebooks,
        "indices_layout": "(batch, frames, num_rq) int64",
        "waveform_layout": "(batch, 1, frames * hop_length) float32",
        "samples_must_be_multiple_of": enc_h._hop_length,
    }
    (out_dir / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2))

    for path in (enc_path, dec_path):
        parts = [path] + sorted(out_dir.glob(f"{path.name}.data*"))
        total = sum(p.stat().st_size for p in parts) / 1e6
        extra = f"  (+{len(parts) - 1} sidecar)" if len(parts) > 1 else ""
        print(f"  {path}  {total:.1f} MB{extra}")
    print(f"  {out_dir / 'tokenizer_meta.json'}")
    print(f"\n{meta['frames_per_second']:.1f} fps x {args.num_rq} codes x "
          f"log2({args.num_tokens}) bits = "
          f"{meta['frames_per_second'] * args.num_rq * 11 / 1000:.2f} kbps")


if __name__ == "__main__":
    main()
