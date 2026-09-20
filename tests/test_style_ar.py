"""
The style-AR (train_style_ar.py) on the CPU with tiny synthetic data.

What has to hold: the phase cache reproduces the token cache's own descriptors
at phase 0; whitening round-trips onto the unit sphere the token-AR expects;
the mixture NLL is a real likelihood (finite, and it falls on learnable data);
sampling is shaped right, seed-reproducible, and guidance at 1 is exactly the
conditional model; and a checkpoint reloads with its whitening intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from train_ar import compute_style_windows
from train_style_ar import (
    DataCfg,
    ModelCfg,
    MogParams,
    StyleAr,
    StyleArConfig,
    StyleArModule,
    collate_pad,
    decode_rows,
    fit_style_stats,
    guide,
    load_style_ar,
    mog_nll,
    mog_sample,
    phase_windows,
    sample_sequence,
    whiten_rows,
)

C = 12  # raw descriptor width
DIMS = 8
TRACKS = 4


def tiny_cfg(dims: int = DIMS) -> StyleArConfig:
    """
    Args:
      dims (int): whitened width.

    Returns:
      StyleArConfig: a CPU-sized config.
    """
    return StyleArConfig(
        data=DataCfg(pca_dims=dims, crop_positions=16),
        model=ModelCfg(d_model=32, n_layers=2, n_heads=4, mog_components=3),
    )


def toy_rows(n: int, seed: int = 0) -> torch.Tensor:
    """
    Args:
      n (int): rows.
      seed (int): RNG seed.

    Returns:
      torch.Tensor: (n, C) unit descriptors with an anisotropic spread.
    """
    generator = torch.Generator().manual_seed(seed)
    scale = torch.linspace(1.0, 0.05, C)
    return F.normalize(torch.randn(n, C, generator=generator) * scale + 0.5, dim=-1)


@pytest.fixture
def module() -> StyleArModule:
    stats = fit_style_stats(toy_rows(256), DIMS, 1e-4)
    return StyleArModule(tiny_cfg(), stats, TRACKS, [3]).eval()


def test_phase_zero_matches_the_token_cache_descriptors() -> None:
    generator = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 5, (100, 2), generator=generator)
    codebooks = torch.randn(2, 5, C, generator=generator)
    phases = phase_windows(tokens, codebooks, 16, 4, 4)
    expected, _ = compute_style_windows(tokens, codebooks, 16, 4, grid=True)
    assert torch.allclose(phases[0], expected)
    assert len(phases) == 4 and phases[1].shape[0] in (
        expected.shape[0] - 1,
        expected.shape[0],
    )
    # phase k is the descriptor grid of the track with its first 4k frames dropped
    shifted, _ = compute_style_windows(tokens[8:], codebooks, 16, 4, grid=True)
    assert torch.allclose(phases[2], shifted)


def test_full_rank_whitening_round_trips_onto_the_sphere() -> None:
    rows = toy_rows(64)
    stats = fit_style_stats(rows, C, 1e-6)
    back = decode_rows(stats, whiten_rows(stats, rows))
    assert torch.allclose(back, rows, atol=1e-4)
    assert torch.allclose(back.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_truncated_whitening_still_lands_on_the_sphere() -> None:
    rows = toy_rows(64)
    stats = fit_style_stats(rows, DIMS, 1e-4)
    back = decode_rows(stats, whiten_rows(stats, rows))
    assert stats.basis.shape == (C, DIMS) and stats.eigvals.shape == (DIMS,)
    assert torch.allclose(back.norm(dim=-1), torch.ones(64), atol=1e-5)
    assert float((back * rows).sum(-1).mean()) > 0.9


def test_forward_predicts_one_more_position_than_it_reads() -> None:
    model = StyleAr(tiny_cfg().model, DIMS, TRACKS, 18)
    ids = torch.tensor([0, 1])
    drop = torch.tensor([False, True])
    for length in (0, 1, 5):
        params = model(torch.randn(2, length, DIMS), ids, drop)
        assert params.logits.shape == (2, length + 1, 3)
        assert params.means.shape == (2, length + 1, 3, DIMS)
        assert params.log_std.shape == (2, length + 1, 3, DIMS)
    assert torch.isfinite(params.log_std).all()


def test_nll_is_finite_and_falls_on_a_linear_ar_process() -> None:
    torch.manual_seed(0)
    rotation = torch.linalg.qr(torch.randn(DIMS, DIMS))[0]
    seq = [torch.randn(DIMS)]
    for _ in range(63):
        seq.append(seq[-1] @ rotation + 0.05 * torch.randn(DIMS))
    x = torch.stack(seq)[None].repeat(8, 1, 1)
    model = StyleAr(tiny_cfg().model, DIMS, TRACKS, 66)
    ids = torch.zeros(8, dtype=torch.long)
    drop = torch.ones(8, dtype=torch.bool)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)

    def loss() -> torch.Tensor:
        return mog_nll(model(x[:, :-1], ids, drop), x).mean()

    start = float(loss().detach())
    assert torch.isfinite(torch.tensor(start))
    for _ in range(150):
        optim.zero_grad()
        value = loss()
        value.backward()
        optim.step()
    end = float(loss().detach())
    assert end < start - 1.0, (start, end)


def test_mog_nll_matches_a_single_gaussian_in_closed_form() -> None:
    target = torch.zeros(1, DIMS)
    params = MogParams(
        torch.zeros(1, 1), torch.zeros(1, 1, DIMS), torch.zeros(1, 1, DIMS)
    )
    expected = 0.5 * DIMS * torch.log(torch.tensor(2 * torch.pi))
    assert torch.allclose(mog_nll(params, target)[0], expected)


def test_mog_sample_is_greedy_at_zero_temperature() -> None:
    params = MogParams(
        torch.tensor([[0.0, 3.0]]),
        torch.stack([torch.zeros(DIMS), torch.ones(DIMS)])[None],
        torch.zeros(1, 2, DIMS),
    )
    assert torch.equal(mog_sample(params, None, 0.0)[0], torch.ones(DIMS))


def test_guide_at_one_is_the_conditional_and_at_zero_the_null() -> None:
    cond = MogParams(
        torch.randn(1, 3), torch.randn(1, 3, DIMS), torch.randn(1, 3, DIMS)
    )
    null = MogParams(
        torch.randn(1, 3), torch.randn(1, 3, DIMS), torch.randn(1, 3, DIMS)
    )
    at_one = guide(cond, null, 1.0)
    assert torch.equal(at_one.logits, cond.logits) and torch.equal(
        at_one.means, cond.means
    )
    at_zero = guide(cond, null, 0.0)
    assert torch.equal(at_zero.means, null.means) and torch.equal(
        at_zero.log_std, cond.log_std
    )


def test_sample_sequence_shapes_norms_and_replay(module: StyleArModule) -> None:
    prefix = toy_rows(3, seed=5)
    rows = sample_sequence(
        module.model,
        module.stats,
        prefix,
        0,
        5,
        1.0,
        1.0,
        torch.Generator().manual_seed(7),
    )
    assert rows.shape == (5, C)
    assert torch.allclose(rows.norm(dim=-1), torch.ones(5), atol=1e-5)
    again = sample_sequence(
        module.model,
        module.stats,
        prefix,
        0,
        5,
        1.0,
        1.0,
        torch.Generator().manual_seed(7),
    )
    assert torch.equal(rows, again)
    other = sample_sequence(
        module.model,
        module.stats,
        prefix,
        0,
        5,
        1.0,
        1.0,
        torch.Generator().manual_seed(8),
    )
    assert not torch.allclose(rows, other)
    assert sample_sequence(module.model, module.stats, None, None, 0).shape == (0, C)
    cold = sample_sequence(module.model, module.stats, None, None, 2)
    assert cold.shape == (2, C)


def test_cfg_one_equals_the_plain_conditional_pass(module: StyleArModule) -> None:
    prefix = toy_rows(2, seed=9)
    plain = sample_sequence(
        module.model,
        module.stats,
        prefix,
        1,
        4,
        1.0,
        1.0,
        torch.Generator().manual_seed(3),
    )
    guided = sample_sequence(
        module.model,
        module.stats,
        prefix,
        1,
        4,
        1.0,
        1.0 + 1e-9,
        torch.Generator().manual_seed(3),
    )
    assert torch.allclose(plain, guided, atol=1e-5)


def test_cfg_zero_equals_the_null_pass(module: StyleArModule) -> None:
    prefix = toy_rows(2, seed=9)
    null = sample_sequence(
        module.model,
        module.stats,
        prefix,
        None,
        4,
        1.0,
        1.0,
        torch.Generator().manual_seed(3),
    )
    zero = sample_sequence(
        module.model,
        module.stats,
        prefix,
        1,
        4,
        1.0,
        0.0,
        torch.Generator().manual_seed(3),
    )
    assert torch.allclose(null, zero, atol=1e-5)


def test_a_walk_longer_than_the_context_keeps_going(module: StyleArModule) -> None:
    steps = module.model.context + 5
    rows = sample_sequence(module.model, module.stats, toy_rows(3), 0, steps)
    assert rows.shape == (steps, C) and torch.isfinite(rows).all()


def test_collate_pads_to_the_longest_and_masks_the_rest() -> None:
    items = [
        {"x": torch.ones(3, DIMS), "raw": torch.ones(3, C), "track_idx": 1},
        {"x": torch.ones(5, DIMS), "raw": torch.ones(5, C), "track_idx": 2},
    ]
    batch = collate_pad(items)
    assert batch["x"].shape == (2, 5, DIMS) and batch["raw"].shape == (2, 5, C)
    assert batch["mask"].tolist() == [[True] * 3 + [False] * 2, [True] * 5]
    assert batch["track_idx"].tolist() == [1, 2]
    assert float(batch["x"][0, 3:].abs().sum()) == 0.0


def test_training_step_uses_the_null_id_on_validation(module: StyleArModule) -> None:
    batch = collate_pad(
        [
            {"x": torch.randn(6, DIMS), "raw": toy_rows(6), "track_idx": 0},
            {"x": torch.randn(4, DIMS), "raw": toy_rows(4), "track_idx": 3},
        ]
    )
    module.train()
    assert torch.isfinite(module.training_step(batch, 0))
    module.eval()
    with torch.no_grad():
        assert torch.isfinite(module.validation_step(batch, 0))


def test_checkpoint_round_trip_keeps_whitening_and_split(
    module: StyleArModule, tmp_path: Path
) -> None:
    path = tmp_path / "style_ar.ckpt"
    torch.save(
        {"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)},
        path,
    )
    again = load_style_ar(path, ema=False)
    assert again.held_out == [3]
    assert torch.equal(again.stats.basis, module.stats.basis)
    assert torch.equal(again.stats.mean, module.stats.mean)
    prefix = toy_rows(2)
    a = sample_sequence(
        module.model,
        module.stats,
        prefix,
        0,
        3,
        1.0,
        1.0,
        torch.Generator().manual_seed(1),
    )
    b = sample_sequence(
        again.model,
        again.stats,
        prefix,
        0,
        3,
        1.0,
        1.0,
        torch.Generator().manual_seed(1),
    )
    assert torch.allclose(a, b)


class _Walker:
    """A StyleWalker that records its call and returns numbered unit rows."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None, int, float, float]] = []

    def sample(
        self,
        prefix: torch.Tensor | None,
        track_idx: int | None,
        steps: int,
        temperature: float,
        cfg: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        assert prefix is not None
        self.calls.append((prefix.shape[0], track_idx, steps, temperature, cfg))
        rows = torch.zeros(steps, prefix.shape[1])
        rows[:, 0] = 1.0
        return rows


def test_walk_from_window_keeps_the_real_window_first() -> None:
    from ab_harness.worker.style_ar import walk_from_window

    style = torch.nn.functional.normalize(torch.randn(10, 6), dim=-1)
    walker = _Walker()
    rows = walk_from_window(walker, style, 7, 3, 5, 4, 1.2, 1.0, 9)
    assert rows.shape == (5, 6)
    assert torch.equal(rows[0], style[7])
    assert walker.calls == [(4, 3, 4, 1.2, 1.0)]  # windows 4..7 as prefix
    assert torch.equal(
        walk_from_window(walker, style, 7, 3, 1, 4, 1.2, 1.0, 9), style[7][None]
    )
    assert len(walker.calls) == 1  # one segment never asks the model
    walk_from_window(walker, style, 1, None, 3, 4, 1.0, 0.0, 9)
    assert walker.calls[-1] == (
        2,
        None,
        2,
        1.0,
        0.0,
    )  # prefix clipped at the track start
