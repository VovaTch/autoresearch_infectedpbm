"""
CPU tests for train_zflow: whitening, requantisation, the DiT, the flow and
the sampler. Every model here is tiny; nothing touches a cache or a GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

import train_zflow as zf

DIM = 16  # token_dim stand-in
RQ = 3
CODES = 32


def codebooks(seed: int = 0) -> torch.Tensor:
    """
    Returns:
      torch.Tensor: (RQ, CODES, DIM) well-separated random codebooks, so
        nearest-neighbour assignment has no near-ties.
    """
    gen = torch.Generator().manual_seed(seed)
    scale = torch.tensor([4.0, 1.0, 0.25]).reshape(-1, 1, 1)
    return torch.randn(RQ, CODES, DIM, generator=gen) * scale


def random_tokens(frames: int, seed: int = 1) -> torch.Tensor:
    """
    Returns:
      torch.Tensor: (frames, RQ) int64 indices.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, CODES, (frames, RQ), generator=gen)


def stats_from(z: torch.Tensor, dims: int = DIM, floor: float = 1e-3) -> zf.LatentStats:
    """
    Args:
      z (torch.Tensor): (N, DIM) latent rows.

    Returns:
      zf.LatentStats: PCA of the rows.
    """
    mean = z.mean(0)
    cov = torch.cov((z - mean).T)
    eig, vecs = torch.linalg.eigh(cov)
    return zf.LatentStats(mean, vecs.flip(1), eig.flip(0).clamp_min(0.0), dims, floor)


def tiny_cfg(**model: int | float) -> zf.ZFlowConfig:
    """
    Returns:
      zf.ZFlowConfig: a config with a 2-layer, 32-wide DiT.
    """
    cfg = zf.ZFlowConfig()
    cfg.model = zf.ModelCfg(
        d_model=32,
        n_layers=2,
        n_heads=4,
        patch=2,
        mlp_hidden=48,
        dropout=0.0,
        t_embed_dim=16,
        style_bottleneck=8,
    )
    for key, value in model.items():
        setattr(cfg.model, key, value)
    cfg.data.pca_dims = DIM
    cfg.data.style_context_frames = 4
    cfg.flow.steps = 4
    return cfg


def tiny_module(seed: int = 0, **model: int | float) -> zf.ZFlowModule:
    """
    Returns:
      zf.ZFlowModule: module over the toy codebooks with exact PCA stats.
    """
    torch.manual_seed(seed)
    cb = codebooks()
    z = zf.embed_zq(random_tokens(512), cb)
    return zf.ZFlowModule(tiny_cfg(**model), stats_from(z), cb)


# --------------------------------------------------------------- codebooks


def test_embed_zq_sums_one_code_per_level():
    cb = codebooks()
    tok = random_tokens(5)
    z = zf.embed_zq(tok, cb)
    expected = cb[0][tok[:, 0]] + cb[1][tok[:, 1]] + cb[2][tok[:, 2]]
    assert torch.allclose(z, expected)


@pytest.mark.parametrize("beam", [1, 4])
def test_requantize_recovers_the_tokens_that_built_the_latent(beam):
    cb = codebooks()
    tok = random_tokens(300)
    z = zf.embed_zq(tok, cb).T.unsqueeze(0)  # (1, DIM, T)
    assert torch.equal(zf.requantize(z, cb, chunk=64, beam=beam), tok.unsqueeze(0))


def test_beam_search_never_ends_with_a_larger_residual_than_greedy():
    gen = torch.Generator().manual_seed(3)
    cb = torch.randn(RQ, CODES, DIM, generator=gen)  # equal scales: greedy is ambiguous
    z = torch.randn(2, DIM, 50, generator=gen) * 3
    for beam in (1, 4, 8):
        idx = zf.requantize(z, cb, chunk=16, beam=beam)
        err = (zf.embed_zq(idx, cb).transpose(1, 2) - z).norm(dim=1)
        if beam == 1:
            greedy = err
        else:
            assert torch.all(err <= greedy + 1e-4)


def test_style_vector_is_unit_norm_and_masked():
    z = torch.randn(2, 6, DIM)
    valid = torch.tensor([[True] * 6, [True, True, False, False, False, False]])
    s = zf.style_vector(z, valid)
    assert torch.allclose(s.norm(dim=-1), torch.ones(2), atol=1e-5)
    manual = torch.nn.functional.normalize(z[1, :2].mean(0), dim=-1)
    assert torch.allclose(s[1], manual, atol=1e-5)


# --------------------------------------------------------------- whitening


def test_whiten_then_unwhiten_is_the_identity_at_full_rank():
    z = torch.randn(1000, DIM) @ torch.randn(DIM, DIM) + 3.0
    stats = stats_from(z)
    x = z.T.unsqueeze(0)
    assert torch.allclose(stats.unwhiten(stats.whiten(x)), x, atol=1e-3)


def test_whitened_coordinates_have_unit_variance():
    z = torch.randn(4000, DIM) @ torch.randn(DIM, DIM)
    stats = stats_from(z, floor=0.0)
    w = stats.whiten(z.T.unsqueeze(0))[0]
    assert torch.allclose(w.std(dim=1), torch.ones(DIM), atol=0.1)


def test_the_eigenvalue_floor_caps_the_whitening_gain():
    z = torch.randn(1000, DIM)
    z[:, -1] *= 1e-4  # a near-null direction
    stats = stats_from(z, floor=1e-2)
    ratio = float(stats.scale.max() / stats.scale.min())
    assert ratio <= 1.0 / math.sqrt(1e-2) + 1e-3


def test_rank_truncation_drops_only_the_tail():
    scales = torch.tensor([4.0] * 4 + [0.1] * (DIM - 4))
    z = torch.randn(1000, DIM) * scales
    stats = stats_from(z, dims=4)
    x = z.T.unsqueeze(0)
    back = stats.unwhiten(stats.whiten(x))
    assert back.shape == x.shape
    rel = float((back - x).norm() / x.norm())
    assert rel < 0.1


def test_stats_survive_a_save_load_round_trip(tmp_path):
    z = torch.randn(200, DIM)
    stats = stats_from(z)
    stats.save(tmp_path / "s.pt")
    back = zf.LatentStats.load(tmp_path / "s.pt", DIM, 1e-3)
    x = torch.randn(1, DIM, 8)
    assert torch.allclose(back.whiten(x), stats.whiten(x))


# --------------------------------------------------------------------- DiT


def test_dit_preserves_the_latent_shape():
    net = zf.LatentDiT(tiny_cfg().model, DIM, DIM)
    x = torch.randn(3, DIM, 12)
    out = net(x, torch.rand(3), torch.randn(3, DIM))
    assert out.shape == x.shape


def test_dit_rejects_frames_not_divisible_by_patch():
    net = zf.LatentDiT(tiny_cfg().model, DIM, DIM)
    with pytest.raises(ValueError):
        net(torch.randn(1, DIM, 7), torch.rand(1))


def test_untrained_dit_predicts_zero_velocity():
    net = zf.LatentDiT(tiny_cfg().model, DIM, DIM)
    out = net(torch.randn(2, DIM, 8), torch.rand(2), torch.randn(2, DIM))
    assert torch.equal(out, torch.zeros_like(out))


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    net = zf.LatentDiT(tiny_cfg().model, DIM, DIM)
    # Un-zero the output so the loss is not identically zero.
    for p in net.parameters():
        if p.ndim >= 1:
            torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(2, DIM, 8)
    drop = torch.tensor([False, True])
    net(x, torch.rand(2), torch.randn(2, DIM), drop).pow(2).mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_dropped_style_is_the_null_embedding():
    torch.manual_seed(0)
    net = zf.LatentDiT(tiny_cfg().model, DIM, DIM)
    torch.nn.init.normal_(net.null_style, std=1.0)
    t = torch.rand(2)
    style = torch.randn(2, DIM)
    c_drop = net.condition(t, style, torch.tensor([True, True]))
    c_none = net.condition(t, None, None)
    c_keep = net.condition(t, style, torch.tensor([False, False]))
    assert torch.allclose(c_drop, c_none)
    assert not torch.allclose(c_keep, c_none)


def test_rope_cache_grows_and_is_sliced():
    net = zf.LatentDiT(tiny_cfg().model, DIM, DIM)
    cos8, _ = net.rope(8, torch.device("cpu"))
    cos16, _ = net.rope(16, torch.device("cpu"))
    assert cos8.shape[0] == 8 and cos16.shape[0] == 16
    assert torch.allclose(cos16[:8], cos8)


# -------------------------------------------------------------------- flow


def test_interpolant_endpoints():
    eps, x1 = torch.randn(2, DIM, 4), torch.randn(2, DIM, 4)
    assert torch.allclose(zf.ZFlowModule.interpolate(eps, x1, torch.zeros(2)), eps)
    assert torch.allclose(zf.ZFlowModule.interpolate(eps, x1, torch.ones(2)), x1)


def test_time_sampling_stays_inside_the_open_interval():
    module = tiny_module()
    t = module.sample_t(4096, torch.device("cpu"))
    eps = module.cfg.flow.t_eps
    assert float(t.min()) >= eps and float(t.max()) <= 1 - eps
    module.cfg.flow.t_sampling = "uniform"
    t = module.sample_t(4096, torch.device("cpu"))
    assert 0.4 < float(t.mean()) < 0.6


def test_prepare_returns_whitened_crop_and_source_tokens():
    module = tiny_module()
    crop, ctx = 8, module.cfg.data.style_context_frames
    tok = random_tokens(crop + 2 * ctx).unsqueeze(0)
    valid = torch.ones(1, crop + 2 * ctx, dtype=torch.bool)
    prep = module.prepare({"tokens": tok, "valid": valid})
    x1, style, crop_tok = prep.x1, prep.style, prep.tokens
    assert prep.track_idx is None and prep.score_mask.all()
    assert x1.shape == (1, DIM, crop)
    assert style.shape == (1, DIM)
    assert torch.equal(crop_tok, tok[:, ctx : ctx + crop])
    z = zf.embed_zq(crop_tok, module.codebooks).transpose(1, 2)
    assert torch.allclose(module.stats.unwhiten(x1), z, atol=1e-3)


class OracleNet(torch.nn.Module):
    """Knows x1 and returns the exact velocity x1 - eps for any iterate."""

    def __init__(self, x1: torch.Tensor) -> None:
        super().__init__()
        self.x1 = x1
        self.calls: list[float] = []

    def forward(self, x, t, style=None, drop=None, *_):
        self.calls.append(float(t[0]))
        tt = t.reshape(-1, 1, 1)
        # x = (1 - t) eps + t x1  ->  eps = (x - t x1) / (1 - t)
        eps = (x - tt * self.x1) / (1.0 - tt)
        return self.x1 - eps


@pytest.mark.parametrize("steps", [1, 3, 8])
def test_an_oracle_flow_lands_on_the_data_for_any_step_count(steps):
    module = tiny_module()
    x1 = torch.randn(2, DIM, 4)
    module.net = OracleNet(x1)  # type: ignore[assignment]
    out = module.sample(torch.randn(2, DIM, 4), 0.0, None, steps=steps, churn=0.0)
    assert torch.allclose(out, x1, atol=1e-4)


def test_churn_zero_is_exactly_an_euler_step():
    module = tiny_module()
    torch.manual_seed(1)
    for p in module.net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(2, DIM, 4)
    manual = x.clone()
    for t in (0.25, 0.5, 0.75):
        manual = manual + 0.25 * module.velocity(manual, torch.full((2,), t), None)
    out = module.sample(x, 0.25, None, steps=3, churn=0.0)
    assert torch.allclose(out, manual, atol=1e-5)


def test_full_churn_puts_every_iterate_on_the_training_interpolant():
    module = tiny_module()
    x1 = torch.randn(4, DIM, 4)
    module.net = OracleNet(x1)  # type: ignore[assignment]
    gen = torch.Generator().manual_seed(0)
    out = module.sample(
        torch.randn(4, DIM, 4), 0.0, None, steps=4, churn=1.0, generator=gen
    )
    # With an oracle the final step lands exactly on x1 whatever the churn.
    assert torch.allclose(out, x1, atol=1e-4)
    assert module.net.calls == pytest.approx([0.0, 0.25, 0.5, 0.75])


def test_sdedit_strength_zero_returns_the_source():
    module = tiny_module()
    src = torch.randn(1, DIM, 4)
    assert torch.equal(module.sdedit(src, 0.0), src)


def test_sdedit_with_an_oracle_returns_the_source_at_any_strength():
    module = tiny_module()
    src = torch.randn(2, DIM, 4)
    module.net = OracleNet(src)  # type: ignore[assignment]
    out = module.sdedit(src, 0.6, steps=5)
    assert torch.allclose(out, src, atol=1e-4)
    assert module.net.calls[0] == pytest.approx(0.4)


def test_cfg_scale_one_equals_the_conditional_pass():
    module = tiny_module()
    torch.manual_seed(2)
    for p in module.net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    x, t, style = torch.randn(2, DIM, 4), torch.rand(2), torch.randn(2, DIM)
    plain = module.net(x, t, style)
    assert torch.allclose(module.velocity(x, t, style, 1.0), plain)
    guided = module.velocity(x, t, style, 3.0)
    uncond = module.net(x, t, None)
    assert torch.allclose(guided, uncond + 3.0 * (plain - uncond), atol=1e-5)


def test_training_step_loss_is_finite_and_backpropagates():
    module = tiny_module()
    crop, ctx = 8, module.cfg.data.style_context_frames
    tok = torch.stack([random_tokens(crop + 2 * ctx, s) for s in range(3)])
    valid = torch.ones(3, crop + 2 * ctx, dtype=torch.bool)
    module.log = lambda *a, **k: None  # type: ignore[assignment]
    loss = module.training_step({"tokens": tok, "valid": valid}, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert module.net.in_proj.weight.grad is not None


# ------------------------------------------------------------- data + cfg


def fake_track(idx: int, frames: int) -> zf.TrackTokens:
    return zf.TrackTokens(
        tokens=random_tokens(frames, idx),
        style=torch.zeros(1, DIM),
        style_bounds=torch.zeros(1, 2, dtype=torch.int64),
        track_idx=idx,
        track_name=f"t{idx}",
        num_frames=frames,
        fps=172.265625,
        val_windows=[],
    )


def test_span_tokens_zero_fills_outside_the_track():
    track = fake_track(0, 20)
    tok, valid = zf._span_tokens(track, 0, 8, 4)
    assert tok.shape == (16, RQ) and valid.shape == (16,)
    assert not valid[:4].any() and valid[4:].all()
    assert torch.equal(tok[4:], track.tokens[:12])
    tok, valid = zf._span_tokens(track, 12, 8, 4)
    assert valid[:12].all() and not valid[12:].any()


def test_crop_dataset_offsets_are_patch_aligned_and_in_range():
    cfg = zf.DataCfg(crop_frames=8, style_context_frames=2, steps_per_epoch=50)
    ds = zf.LatentCropDataset(
        [fake_track(0, 41), fake_track(1, 9)], cfg, patch=4, seed=0
    )
    for i in range(len(ds)):
        item = ds[i]
        assert item["tokens"].shape == (12, RQ)
        assert item["valid"][2:10].all()


def test_val_dataset_is_deterministic_and_skips_short_tracks():
    cfg = zf.DataCfg(crop_frames=8, style_context_frames=2, val_crops_per_track=3)
    ds = zf.ValLatentDataset([fake_track(0, 40), fake_track(1, 4)], cfg, patch=4)
    assert len(ds) == 3
    assert all(start % 4 == 0 for _, start in ds.items)
    assert torch.equal(ds[1]["tokens"], ds[1]["tokens"])


def test_param_groups_and_apply_ema_follow_optimizer_order():
    module = tiny_module()
    decay, no_decay = zf.param_groups(module.net)
    assert all(p.ndim >= 2 for p in decay) and all(p.ndim < 2 for p in no_decay)
    ema = [torch.full_like(p, 7.0) for p in decay + no_decay]
    assert zf.apply_ema(module.net, {"optimizer_states": [{"ema": ema}]})
    assert all(torch.all(p == 7.0) for p in module.net.parameters())
    assert not zf.apply_ema(module.net, {"optimizer_states": [{"ema": ema[:-1]}]})


def write_config(tmp_path, body: str):
    path = tmp_path / "c.yaml"
    path.write_text(body)
    return path


def test_config_defaults_load_from_an_empty_file(tmp_path):
    cfg = zf.load_config(write_config(tmp_path, ""))
    assert cfg.data.crop_frames == 2048 and cfg.model.patch == 2


def test_config_rejects_an_unknown_key(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        zf.load_config(write_config(tmp_path, "model:\n  width: 3\n"))


def test_config_rejects_an_unknown_section(tmp_path):
    with pytest.raises(ValueError, match="section"):
        zf.load_config(write_config(tmp_path, "sampler:\n  steps: 3\n"))


def test_config_survives_a_round_trip_through_a_checkpoint_dict():
    from dataclasses import asdict

    cfg = tiny_cfg()
    assert zf.config_from_dict(asdict(cfg)) == cfg


def test_held_out_tracks_depend_only_on_the_seed():
    entries = [{"track_idx": i} for i in range(53)]
    a = zf.plan_held_out(entries, zf.DataCfg(split_seed=1234, held_out_tracks=5))  # type: ignore[arg-type]
    b = zf.plan_held_out(entries, zf.DataCfg(split_seed=1234, held_out_tracks=5))  # type: ignore[arg-type]
    assert a == b == [18, 33, 34, 45, 51]


# ------------------------------------------------- xattn + prefix (generator)


def xattn_module(**model: int | float) -> zf.ZFlowModule:
    """
    Returns:
      zf.ZFlowModule: tiny module in generator mode (xattn, 4 ids, prefix on).
    """
    return tiny_module(cond_mode="xattn", num_tracks=4, prefix_max_frac=0.5, **model)


def ar_batch(batch: int, frames: int, seed: int = 5) -> dict:
    """
    Returns:
      dict: an AR-stage loader batch (train_ar.TokenCropDataset layout).
    """
    gen = torch.Generator().manual_seed(seed)
    return {
        "tokens": torch.randint(0, CODES, (batch, frames, RQ), generator=gen),
        "style": torch.nn.functional.normalize(
            torch.randn(batch, DIM, generator=gen), dim=-1
        ),
        "track_idx": torch.randint(0, 4, (batch,), generator=gen),
        "score_mask": torch.ones(batch, frames, dtype=torch.bool),
    }


def test_xattn_dit_preserves_shape_and_starts_at_zero():
    net = xattn_module().net
    x = torch.randn(3, DIM, 12)
    out = net(x, torch.rand(3), torch.randn(3, DIM), None, torch.tensor([0, 1, 3]))
    assert out.shape == x.shape
    assert torch.equal(out, torch.zeros_like(out))
    assert net.in_proj.in_features == (DIM + 1) * net.cfg.patch


def test_xattn_memory_nulls_each_stream_independently():
    torch.manual_seed(0)
    mem = xattn_module().net.memory
    assert mem is not None
    torch.nn.init.normal_(mem.null_style, std=1.0)
    ids = torch.tensor([1, 2])
    style = torch.randn(2, DIM)
    full = mem(ids, style)
    no_id = mem(ids, style, drop_id=torch.tensor([True, True]))
    no_style = mem(ids, style, drop_style=torch.tensor([True, True]))
    null = mem(None, None, batch=2)
    assert full.shape == (2, 2, mem.null_style.shape[0])
    assert torch.allclose(no_id[:, 0], null[:, 0]) and torch.allclose(
        no_id[:, 1], full[:, 1]
    )
    assert torch.allclose(no_style[:, 1], null[:, 1]) and torch.allclose(
        no_style[:, 0], full[:, 0]
    )
    assert not torch.allclose(full, null)


def test_xattn_gradients_reach_every_parameter():
    torch.manual_seed(0)
    net = xattn_module().net
    for p in net.parameters():
        if p.ndim >= 1:
            torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(2, DIM, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, :4] = True
    drop = torch.tensor([False, True])
    out = net(
        x, torch.rand(2), torch.randn(2, DIM), drop, torch.tensor([0, 2]), drop, mask
    )
    out.pow(2).mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_prepare_accepts_ar_stage_items():
    module = xattn_module()
    batch = ar_batch(2, 8)
    prep = module.prepare(batch)
    assert prep.x1.shape == (2, DIM, 8)
    assert torch.equal(prep.tokens, batch["tokens"])
    assert prep.track_idx is not None and torch.equal(
        prep.track_idx, batch["track_idx"]
    )
    assert torch.equal(prep.style, batch["style"])


def test_training_step_runs_in_generator_mode():
    module = xattn_module()
    loss = module._run(ar_batch(4, 8), "train")
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_prefix_frames_survive_sampling_untouched():
    module = xattn_module()
    for p in module.net.parameters():
        if p.ndim >= 1:
            torch.nn.init.normal_(p, std=0.05)
    prefix = torch.randn(2, DIM, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :4] = True
    noise = torch.randn(2, DIM, 8)
    out = module.sample(
        noise,
        0.0,
        torch.randn(2, DIM),
        steps=3,
        track_idx=torch.tensor([1, 2]),
        prefix=prefix,
        prefix_mask=mask,
    )
    assert torch.equal(out[:, :, :4], prefix[:, :, :4])
    assert not torch.allclose(out[:, :, 4:], prefix[:, :, 4:])


def test_cfg_null_branch_drops_id_and_style():
    torch.manual_seed(0)
    module = xattn_module()
    for p in module.net.parameters():
        if p.ndim >= 1:
            torch.nn.init.normal_(p, std=0.05)
    x, t = torch.randn(2, DIM, 8), torch.rand(2)
    style, ids = torch.randn(2, DIM), torch.tensor([0, 3])
    cond = module.net(x, t, style, None, ids)
    both_dropped = torch.ones(2, dtype=torch.bool)
    null = module.net(x, t, style, both_dropped, ids, both_dropped)
    guided = module.velocity(x, t, style, 3.0, ids)
    assert torch.allclose(guided, null + 3.0 * (cond - null), atol=1e-5)


def test_legacy_adaln_config_round_trips_without_new_keys():
    raw = zf.asdict(tiny_cfg())
    for key in ("cond_mode", "num_tracks", "p_drop_cond", "prefix_max_frac"):
        raw["model"].pop(key)
    cfg = zf.config_from_dict(raw)
    assert cfg.model.cond_mode == "adaln" and cfg.model.prefix_max_frac == 0.0
    net = zf.LatentDiT(cfg.model, DIM, DIM)
    assert net.memory is None and net.in_proj.in_features == DIM * cfg.model.patch
