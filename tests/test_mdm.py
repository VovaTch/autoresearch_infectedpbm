"""
CPU tests for train_mdm: the corruption, the loss mask, the sampler and the
config round-trip. Every model here is tiny; nothing touches a cache or a GPU.
"""

from __future__ import annotations

import pytest
import torch

import train_mdm as mdm

DIM = 16  # style_dim stand-in
RQ = 3
CODES = 32
TRACKS = 4


def tiny_cfg(**model: int | float) -> mdm.MdmConfig:
    """
    Returns:
      mdm.MdmConfig: a config with a 2-layer, 32-wide denoiser and 4 ids.
    """
    cfg = mdm.MdmConfig()
    cfg.model = mdm.ModelCfg(
        d_model=32,
        n_layers=2,
        n_heads=4,
        mlp_hidden=48,
        dropout=0.0,
        style_bottleneck=8,
        num_tracks=TRACKS,
    )
    for key, value in model.items():
        setattr(cfg.model, key, value)
    cfg.mask.steps = [3, 2, 2]
    return cfg


def tiny_module(seed: int = 0, **model: int | float) -> mdm.MdmModule:
    """
    Returns:
      mdm.MdmModule: module over the toy vocabulary.
    """
    torch.manual_seed(seed)
    return mdm.MdmModule(tiny_cfg(**model), CODES, RQ, DIM)


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
        "track_idx": torch.randint(0, TRACKS, (batch,), generator=gen),
        "score_mask": torch.ones(batch, frames, dtype=torch.bool),
    }


# -------------------------------------------------------------- corruption


def test_mask_ratio_schedules_run_from_all_to_none():
    u = torch.tensor([0.0, 1.0])
    assert torch.allclose(
        mdm.mask_ratio(u, "cosine"), torch.tensor([1.0, 0.0]), atol=1e-6
    )
    assert torch.allclose(mdm.mask_ratio(u, "linear"), torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError):
        mdm.mask_ratio(u, "square")


def test_corruption_respects_prefix_and_level_order():
    module = tiny_module()
    tokens = ar_batch(64, 16)["tokens"]
    gen = torch.Generator().manual_seed(1)
    corrupted, target, level = module.corrupt(tokens, gen)
    mask_id = module.net.mask_id
    masked = corrupted == mask_id
    idx = torch.arange(RQ).reshape(1, 1, -1)
    lv = level.reshape(-1, 1, 1)
    # nothing below the drawn level is ever masked; the target is exactly the
    # masked cells of the drawn level
    assert not masked[(idx < lv).expand_as(masked)].any()
    assert torch.equal(target, masked & (idx == lv))
    # levels above the drawn level are masked wherever the drawn level's frame
    # is not a prefix frame -- i.e. a fully masked frame above, clean prefix
    above = (idx > lv).expand_as(masked)
    held = ~masked.any(dim=-1) & (lv.squeeze(-1) < RQ - 1)  # frames with nothing masked
    assert not masked[held.unsqueeze(-1) & above].any()
    # prefix frames are a contiguous head: once a frame has a masked cell
    # above the level, every later frame does too
    for b in range(64):
        if int(level[b]) == RQ - 1:
            continue
        col = masked[b, :, RQ - 1]
        first = int(col.long().argmax()) if col.any() else 16
        assert col[first:].all()
    # unmasked cells keep their value
    assert torch.equal(corrupted[~masked], tokens[~masked])


def test_loss_scores_only_the_masked_cells_of_the_drawn_level():
    module = tiny_module()
    batch = ar_batch(4, 8)
    torch.manual_seed(0)
    loss = module._run(batch, "val")
    assert loss.ndim == 0 and torch.isfinite(loss)
    # with nothing scored the loss is 0, not NaN
    batch["score_mask"] = torch.zeros(4, 8, dtype=torch.bool)
    assert float(module._run(batch, "val")) == 0.0


def test_training_step_runs_and_reaches_every_parameter():
    module = tiny_module()
    loss = module._run(ar_batch(8, 8), "train")
    loss.backward()
    for name, p in module.net.named_parameters():
        assert p.grad is not None, name


# ----------------------------------------------------------------- sampler


class OracleNet(torch.nn.Module):
    """Always votes for a fixed answer grid, whatever the input."""

    def __init__(self, answer: torch.Tensor, mask_id: int) -> None:
        super().__init__()
        self.answer = answer
        self.mask_id = mask_id
        self.num_rq = answer.shape[-1]
        self.calls = 0

    def forward(self, tokens, track_idx, style, drop_id=None, drop_style=None):
        self.calls += 1
        logits = torch.full((*tokens.shape, CODES), -5.0)
        return logits.scatter(
            -1, self.answer.expand(tokens.shape[0], -1, -1).unsqueeze(-1), 5.0
        )


def oracle_sampler(answer: torch.Tensor) -> mdm.MaskedDenoiser:
    """
    Returns:
      mdm.MaskedDenoiser: a real sampler whose network is the oracle.
    """
    net = mdm.MaskedDenoiser(tiny_cfg().model, CODES, RQ, DIM)
    oracle = OracleNet(answer, net.mask_id)
    net.forward = oracle.forward  # type: ignore[method-assign]
    net._oracle = oracle  # type: ignore[attr-defined]
    return net


def test_sampler_recovers_the_oracle_answer_and_keeps_the_prefix():
    answer = torch.randint(0, CODES, (1, 12, RQ))
    net = oracle_sampler(answer)
    prefix = torch.randint(0, CODES, (2, 12, RQ))
    lengths = torch.tensor([4, 0])
    out = net.sample(
        12,
        torch.tensor([0, 1]),
        torch.randn(2, DIM),
        tiny_cfg().mask,
        prefix=prefix,
        prefix_frames=lengths,
        temperature=0.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert out.shape == (2, 12, RQ)
    assert not (out == net.mask_id).any()
    assert torch.equal(out[0, :4], prefix[0, :4])
    assert torch.equal(out[0, 4:], answer[0, 4:])
    assert torch.equal(out[1], answer[0])
    assert net._oracle.calls == sum(tiny_cfg().mask.steps)  # type: ignore[attr-defined]


def test_sampler_is_reproducible_from_its_generator():
    torch.manual_seed(0)
    net = mdm.MaskedDenoiser(tiny_cfg().model, CODES, RQ, DIM)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    kwargs = dict(
        track_idx=torch.tensor([2]), style=torch.randn(1, DIM), mask_cfg=tiny_cfg().mask
    )
    a = net.sample(8, generator=torch.Generator().manual_seed(7), **kwargs)
    b = net.sample(8, generator=torch.Generator().manual_seed(7), **kwargs)
    c = net.sample(8, generator=torch.Generator().manual_seed(8), **kwargs)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert net._rope is not None


def test_guidance_at_scale_one_is_the_conditional_branch():
    torch.manual_seed(0)
    net = mdm.MaskedDenoiser(tiny_cfg().model, CODES, RQ, DIM)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    tokens = torch.randint(0, CODES + 1, (2, 6, RQ))
    ids, style = torch.tensor([0, 3]), torch.randn(2, DIM)
    cond = net(tokens, ids, style)
    assert torch.allclose(net.guided_logits(tokens, ids, style, 1.0), cond)
    drop = torch.ones(2, dtype=torch.bool)
    null = net(tokens, ids, style, drop, drop)
    guided = net.guided_logits(tokens, ids, style, 2.5)
    assert torch.allclose(guided, null + 2.5 * (cond - null), atol=1e-5)
    assert not torch.allclose(cond, null)


def test_sample_tokens_honours_top_k_and_argmax():
    base = torch.tensor([[[0.0, 1.0, 5.0, 2.0]]])
    logits = base.expand(3, 2, -1)
    assert torch.equal(mdm.sample_tokens(logits, 0.0, 0, 0.0), torch.full((3, 2), 2))
    gen = torch.Generator().manual_seed(0)
    draws = mdm.sample_tokens(base.expand(200, 2, -1), 1.0, 2, 0.0, gen)
    assert set(draws.unique().tolist()) <= {2, 3}


# ------------------------------------------------------------------ config


def test_config_round_trips_through_yaml_and_dict(tmp_path):
    cfg = tiny_cfg()
    cfg.mask.steps = [4, 1, 1]
    path = tmp_path / "c.yaml"
    import yaml

    path.write_text(yaml.safe_dump(mdm.asdict(cfg)))
    back = mdm.load_config(path)
    assert back == cfg
    assert mdm.config_from_dict(mdm.asdict(cfg)) == cfg
    path.write_text(yaml.safe_dump({"model": {"width": 3}}))
    with pytest.raises(ValueError):
        mdm.load_config(path)
