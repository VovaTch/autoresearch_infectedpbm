"""
CPU tests for the non-AR backends: checkpoint families, window chaining and
the ZFlow / MDM generators over tiny models. No cache, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import train_mdm as mdm
import train_zflow as zf
from ab_harness.checkpoints import backend_of, discover_all
from ab_harness.config import GeneratorCfg, checkpoint_menus
from ab_harness.model.protocols import SampleSource
from ab_harness.worker.generator import SampleRequest
from ab_harness.worker.mdm_gen import MdmGenerator
from ab_harness.worker.windowed import WindowBatch, WindowedGenerator
from ab_harness.worker.zflow_gen import ZFlowGenerator

DIM, RQ, CODES, TRACKS = 16, 3, 32, 4
CPU = torch.device("cpu")


# ------------------------------------------------------------- checkpoints


def test_backend_is_read_off_the_directory_family():
    assert backend_of("saved_ar_20260915/ar_latest.ckpt") == "ar"
    assert backend_of("saved_dpo_x/last.ckpt") == "ar"
    assert backend_of("saved_zflow_512_xattn/zflow_latest.ckpt") == "zflow"
    assert backend_of("saved_mdm_512/mdm_best.ckpt") == "mdm"
    assert backend_of("somewhere/else.ckpt") == "ar"
    assert backend_of("") == "ar"


def test_discover_all_lists_every_backend(tmp_path: Path):
    for d in ("saved_ar_a", "saved_zflow_b", "saved_mdm_c", "saved_20260101_tok"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "last.ckpt").write_bytes(b"")
    found = discover_all(tmp_path)
    assert found == {
        "ar": ["saved_ar_a/last.ckpt"],
        "zflow": ["saved_zflow_b/last.ckpt"],
        "mdm": ["saved_mdm_c/last.ckpt"],
    }


def test_checkpoint_menus_put_the_current_one_first_in_its_backend():
    menus = checkpoint_menus("saved_mdm_zzz/never.ckpt")
    assert menus["mdm"][0] == "saved_mdm_zzz/never.ckpt"
    assert "saved_mdm_zzz/never.ckpt" not in menus["ar"]


# ---------------------------------------------------------------- chaining


class CountingGenerator(WindowedGenerator):
    """Fills every generated frame with the window index; prefix untouched."""

    def __init__(self, window: int, keep_frac: float, max_prefix: int | None = None):
        super().__init__(CPU, window, RQ, TRACKS, CODES, 1.0 - keep_frac, max_prefix)
        self.windows = 0
        self.batches: list[WindowBatch] = []

    def rounds_per_window(self) -> int:
        return 2

    def fill_window(self, batch: WindowBatch, progress):
        self.windows += 1
        self.batches.append(batch)
        out = torch.full_like(batch.prefix, 100 + self.windows)
        held = torch.arange(self.window).unsqueeze(0) < batch.prefix_frames.unsqueeze(1)
        progress(2)
        return torch.where(held.unsqueeze(-1), batch.prefix, out)


def request(frames: int, prompt: int = 0, **kw) -> SampleRequest:
    p = torch.randint(0, CODES, (prompt, RQ)) if prompt else None
    return SampleRequest(1, torch.randn(DIM), frames=frames, prompt=p, seed=0, **kw)


def test_chain_returns_exactly_the_requested_frames_with_the_prompt_first():
    gen = CountingGenerator(window=8, keep_frac=0.5)
    reqs = [request(20, prompt=3), request(5), request(8, prompt=8)]
    ticks: list[tuple[int, int]] = []
    outs = gen.sample_batch(reqs, lambda a, b: ticks.append((a, b)))
    assert [o.shape for o in outs] == [(20, RQ), (5, RQ), (8, RQ)]
    assert torch.equal(outs[0][:3], reqs[0].prompt)
    assert torch.equal(outs[2], reqs[2].prompt)  # nothing to generate
    # window 1: lane 0 has prefix 3 -> 5 new frames; lane 1 prefix 0 -> 8 -> done
    assert (outs[0][3:8] == 101).all() and (outs[1] == 101).all()
    # window 2: carries keep=4 frames, adds 4 -> frames 8..11 are 102, etc.
    assert (outs[0][8:12] == 102).all() and (outs[0][12:16] == 103).all()
    assert (outs[0][16:20] == 104).all()
    assert gen.windows == 4
    assert ticks[-1] == (8, 8) and all(b == 8 for _, b in ticks)
    assert gen.batches[1].prefix_frames.tolist() == [4]
    assert torch.equal(gen.batches[1].prefix[0, :4], outs[0][4:8])


def test_carried_prefix_is_capped_at_the_trained_maximum():
    gen = CountingGenerator(window=8, keep_frac=0.75, max_prefix=2)
    assert gen.keep == 2
    req = request(20, prompt=6)
    out = gen.sample_batch([req])[0]
    assert torch.equal(out[:6], req.prompt)
    assert gen.batches[0].prefix_frames.tolist() == [2]


def test_window_batch_carries_conditioning_flags_and_guidance():
    gen = CountingGenerator(window=8, keep_frac=0.5)
    reqs = [
        request(8, use_track_id=False, cfg_strength=3.0),
        request(8, use_style=False, cfg_strength=0.0),
        request(8, style_schedule=(torch.ones(DIM),), style_period=4),
    ]
    reqs[2] = SampleRequest(
        **{**vars(reqs[2]), "frames": 12, "style": torch.zeros(DIM)}
    )
    gen.sample_batch(reqs)
    first = gen.batches[0]
    assert first.track_idx.tolist() == [TRACKS, 1, 1]
    assert first.drop_id.tolist() == [True, False, False]
    assert first.drop_style.tolist() == [False, True, False]
    assert first.cfg.tolist() == [
        3.0,
        1.0,
        1.0,
    ]  # unguided lanes sample the conditional
    assert torch.equal(first.style[2], torch.zeros(DIM))
    second = gen.batches[1]
    assert second.requests == [reqs[2]]
    assert torch.equal(second.style[0], torch.ones(DIM))  # step 8 >= period 4


# ------------------------------------------------------------- generators


def gen_cfg(**kw) -> GeneratorCfg:
    return GeneratorCfg(
        **{"flow_steps": 2, "mdm_steps": [2, 1, 1], "requantize_beam": 2, **kw}
    )


def zflow_generator(cond_mode: str = "xattn", prefix: float = 0.5) -> ZFlowGenerator:
    torch.manual_seed(0)
    cfg = zf.ZFlowConfig()
    cfg.model = zf.ModelCfg(
        d_model=32,
        n_layers=1,
        n_heads=4,
        patch=2,
        mlp_hidden=48,
        dropout=0.0,
        t_embed_dim=16,
        style_bottleneck=8,
        cond_mode=cond_mode,
        num_tracks=TRACKS,
        prefix_max_frac=prefix,
    )
    cfg.data.pca_dims = DIM
    cb = torch.randn(RQ, CODES, DIM) * torch.tensor([4.0, 1.0, 0.25]).reshape(-1, 1, 1)
    z = zf.embed_zq(torch.randint(0, CODES, (512, RQ)), cb)
    mean = z.mean(0)
    eig, vecs = torch.linalg.eigh(torch.cov((z - mean).T))
    stats = zf.LatentStats(mean, vecs.flip(1), eig.flip(0).clamp_min(0), DIM, 1e-3)
    module = zf.ZFlowModule(cfg, stats, cb).eval()
    for p in module.net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return ZFlowGenerator(module, CPU, 8, TRACKS, gen_cfg())


def mdm_generator() -> MdmGenerator:
    torch.manual_seed(0)
    cfg = mdm.MdmConfig()
    cfg.model = mdm.ModelCfg(
        d_model=32,
        n_layers=1,
        n_heads=4,
        mlp_hidden=48,
        dropout=0.0,
        style_bottleneck=8,
        num_tracks=TRACKS,
        prefix_max_frac=0.5,
    )
    net = mdm.MaskedDenoiser(cfg.model, CODES, RQ, DIM).eval()
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return MdmGenerator(net, cfg.mask, CPU, 8, 0.5, gen_cfg())


@pytest.mark.parametrize("make", [zflow_generator, mdm_generator])
def test_backend_satisfies_sample_source_and_returns_prompt_plus_frames(make):
    gen = make()
    assert isinstance(gen, SampleSource)
    assert gen.window_frames == 8 and gen.depth == RQ and gen.num_tracks == TRACKS
    reqs = [request(19, prompt=3, cfg_strength=2.0), request(10, use_style=False)]
    outs = gen.sample_batch(reqs)
    assert [o.shape for o in outs] == [(19, RQ), (10, RQ)]
    assert all(o.dtype == torch.int64 and o.device.type == "cpu" for o in outs)
    assert torch.equal(outs[0][:3], reqs[0].prompt)
    assert all(int(o.min()) >= 0 and int(o.max()) < CODES for o in outs)


@pytest.mark.parametrize("make", [zflow_generator, mdm_generator])
def test_backend_is_reproducible_from_the_seed(make):
    gen = make()
    req = request(12, prompt=2)
    a = gen.sample_batch([req])[0]
    b = gen.sample_batch([req, request(12)])[0]  # batch-mates change nothing
    c = gen.sample_batch([SampleRequest(**{**vars(req), "seed": 5})])[0]
    assert torch.equal(a, b)
    assert not torch.equal(a[2:], c[2:])


def test_mdm_lanes_with_different_draw_knobs_are_grouped_not_dropped():
    gen = mdm_generator()
    reqs = [request(8, temperature=0.0), request(8, temperature=1.0, top_k=5)]
    outs = gen.sample_batch(reqs)
    assert len(outs) == 2 and not (outs[0] == outs[1]).all()


def test_legacy_adaln_flow_checkpoint_still_samples_with_a_prompt():
    gen = zflow_generator(cond_mode="adaln", prefix=0.0)
    assert gen.keep == 6  # window - 1 cap, 0.25 reprime -> 6
    out = gen.sample_batch([request(14, prompt=4, cfg_strength=2.0)])[0]
    assert out.shape == (14, RQ)
