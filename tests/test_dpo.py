"""
Preference-pair extraction and the DPO objective.

CPU only and under a second: the transformer here is 32-wide and two layers
deep, which is enough to check that masks mask and that gradients flow to the
parameters freezing was supposed to leave alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ab_harness.model.bank import ClipBank
from ab_harness.model.judgement import JudgementLog
from ab_harness.model.types import ClipSpec, Conditioning, Judgement, Sampling, Tier
from train_ar import ArConfig, ArTransformer, ModelCfg, prepare_grid
from train_dpo import (
    ArCheckpointWriter,
    DpoCfg,
    LengthBucketSampler,
    PairsCfg,
    PreferenceDataset,
    collate_pairs,
    dpo_loss,
    freeze_trunk,
    load_preference_pairs,
    measure_diversity,
    sequence_logprob,
    split_by_session,
)

NUM_TOKENS = 16
DEPTH = 2


def _spec(
    item: str,
    group: str = "g0",
    tier: Tier = Tier.BULK,
    frames: int = 12,
    prompt: int = 0,
    checkpoint: str = "ckpt_a",
    generator: str = "ar",
) -> ClipSpec:
    """
    Args:
      item (str): item id.
      group (str): shared-conditioning group.
      tier (Tier): rating tier.
      frames (int): clip length.
      prompt (int): forced prefix frames.
      checkpoint (str): originating checkpoint.
      generator (str): "ar" or "reference".

    Returns:
      ClipSpec: a minimal spec.
    """
    return ClipSpec(
        item_id=item,
        tier=tier,
        group_id=group,
        n_frames=frames,
        conditioning=Conditioning(
            track_idx=0, start_frame=0, style_window=0, prompt_frames=prompt
        ),
        sampling=Sampling(seed=hash(item) % 1000),
        generator=generator,
        checkpoint=checkpoint,
    )


def _judgement(
    pair: str,
    session: str,
    left: str,
    right: str,
    choice: str,
    ts: str,
    is_repeat: bool = False,
    is_anchor: bool = False,
    tier: Tier = Tier.BULK,
) -> Judgement:
    """
    Args:
      pair (str): pair id.
      session (str): session id.
      left (str): left item id.
      right (str): right item id.
      choice (str): "left", "right" or "tie".
      ts (str): ISO timestamp, which read_all sorts on.
      is_repeat (bool): repeat showing.
      is_anchor (bool): anchor pair.
      tier (Tier): rating tier.

    Returns:
      Judgement: one logged decision.
    """
    chosen = {"left": left, "right": right}.get(choice, "")
    return Judgement(
        pair_id=pair,
        session_id=session,
        tier=tier,
        question="Which sounds cleaner?",
        item_left=left,
        item_right=right,
        choice=choice,  # type: ignore[arg-type]
        chosen_item_id=chosen,
        response_ms=1000,
        is_repeat=is_repeat,
        is_anchor=is_anchor,
        lufs_target=-23.0,
        was_live=True,
        ts=ts,
    )


@pytest.fixture
def loaded_bank(bank: ClipBank) -> ClipBank:
    """
    A bank holding every clip the filter tests refer to.

    Args:
      bank (ClipBank): empty bank from conftest.

    Returns:
      ClipBank: the same bank, populated.
    """
    rng = np.random.default_rng(0)
    specs = [
        _spec("gen_a", "g0"),
        _spec("gen_b", "g0"),
        _spec("gen_c", "g1"),
        _spec("gen_d", "g1"),
        _spec("gen_e", "g2"),
        _spec("gen_f", "g3"),  # different group from gen_e on purpose
        _spec("gen_g", "g4"),
        _spec("ref_h", "g4", generator="reference"),
        _spec("gen_i", "g5"),
        _spec("gen_j", "g5"),
    ]
    for spec in specs:
        bank.add(spec, rng.integers(0, NUM_TOKENS, (spec.n_frames, DEPTH)))
    return bank


def _log(bank: ClipBank, session: str, rows: list[Judgement]) -> None:
    """
    Args:
      bank (ClipBank): destination bank.
      session (str): session id, and the log's file stem.
      rows (list[Judgement]): decisions to append.
    """
    log = JudgementLog(bank.sessions_dir / f"{session}.jsonl")
    for row in rows:
        log.append(row)


def test_filter_chain_keeps_only_clean_same_group_pairs(loaded_bank: ClipBank) -> None:
    _log(
        loaded_bank,
        "s1",
        [
            _judgement("p1", "s1", "gen_a", "gen_b", "left", "2026-01-01T00:00:01"),
            _judgement("p2", "s1", "gen_c", "gen_d", "tie", "2026-01-01T00:00:02"),
            _judgement("p3", "s1", "gen_e", "gen_f", "right", "2026-01-01T00:00:03"),
            _judgement(
                "p4",
                "s1",
                "gen_g",
                "ref_h",
                "right",
                "2026-01-01T00:00:04",
                is_anchor=True,
            ),
        ],
    )
    pairs, report = load_preference_pairs(loaded_bank, PairsCfg())

    assert [p.pair_id for p in pairs] == ["p1"]
    assert pairs[0].winner.item_id == "gen_a"
    assert pairs[0].loser.item_id == "gen_b"
    assert report.dropped["tie"] == 1
    assert report.dropped["conditioning differs"] == 1
    assert report.dropped["anchor pair"] == 1


def test_a_contradictory_repeat_drops_the_whole_comparison(
    loaded_bank: ClipBank,
) -> None:
    _log(
        loaded_bank,
        "s1",
        [
            _judgement("p1", "s1", "gen_a", "gen_b", "left", "2026-01-01T00:00:01"),
            _judgement(
                "p2",
                "s1",
                "gen_b",
                "gen_a",
                "left",  # sides swapped, so this picks gen_b: a contradiction
                "2026-01-01T00:00:02",
                is_repeat=True,
            ),
            _judgement("p3", "s1", "gen_i", "gen_j", "left", "2026-01-01T00:00:03"),
            _judgement(
                "p4",
                "s1",
                "gen_j",
                "gen_i",
                "right",  # swapped and consistent: still gen_i
                "2026-01-01T00:00:04",
                is_repeat=True,
            ),
        ],
    )
    pairs, report = load_preference_pairs(loaded_bank, PairsCfg())

    assert {p.winner.item_id for p in pairs} == {"gen_i"}
    assert report.dropped["contradictory repeat"] == 2
    assert report.dropped["consistent repeat"] == 1


def test_a_fatigued_session_is_discarded_whole(loaded_bank: ClipBank) -> None:
    _log(
        loaded_bank,
        "good",
        [_judgement("p1", "good", "gen_a", "gen_b", "left", "2026-01-01T00:00:01")],
    )
    _log(
        loaded_bank,
        "tired",
        [
            # the anchor's generated side won, which is the wrong answer
            _judgement(
                "p2",
                "tired",
                "gen_g",
                "ref_h",
                "left",
                "2026-01-01T00:00:02",
                is_anchor=True,
            ),
            _judgement("p3", "tired", "gen_i", "gen_j", "left", "2026-01-01T00:00:03"),
        ],
    )
    pairs, report = load_preference_pairs(loaded_bank, PairsCfg(min_anchor_acc=0.6))

    assert report.failed_sessions == ["tired"]
    assert {p.session_id for p in pairs} == {"good"}


def test_off_policy_pairs_are_kept_by_default_and_droppable(
    bank: ClipBank,
) -> None:
    rng = np.random.default_rng(1)
    for item, ckpt in [
        ("gen_a", "ckpt_a"),
        ("gen_b", "ckpt_a"),
        ("gen_c", "ckpt_b"),
        ("gen_d", "ckpt_b"),
    ]:
        spec = _spec(item, "g0" if ckpt == "ckpt_a" else "g1", checkpoint=ckpt)
        bank.add(spec, rng.integers(0, NUM_TOKENS, (spec.n_frames, DEPTH)))
    _log(
        bank,
        "s1",
        [
            _judgement("p1", "s1", "gen_a", "gen_b", "left", "2026-01-01T00:00:01"),
            _judgement("p2", "s1", "gen_c", "gen_d", "left", "2026-01-01T00:00:02"),
        ],
    )

    kept, _ = load_preference_pairs(bank, PairsCfg(reference_checkpoint="ckpt_a"))
    assert len(kept) == 2

    strict, report = load_preference_pairs(
        bank,
        PairsCfg(reference_checkpoint="ckpt_a", restrict_to_reference_checkpoint=True),
    )
    assert [p.pair_id for p in strict] == ["p1"]
    assert report.dropped["off-policy checkpoint"] == 1


def test_the_split_never_shares_a_session(loaded_bank: ClipBank) -> None:
    _log(
        loaded_bank,
        "s1",
        [_judgement("p1", "s1", "gen_a", "gen_b", "left", "2026-01-01T00:00:01")],
    )
    _log(
        loaded_bank,
        "s2",
        [_judgement("p2", "s2", "gen_i", "gen_j", "left", "2026-01-01T00:00:02")],
    )
    pairs, _ = load_preference_pairs(loaded_bank, PairsCfg())
    train, val = split_by_session(pairs, val_frac=0.5, seed=0)

    assert train and val
    assert not ({p.session_id for p in train} & {p.session_id for p in val})


def test_a_single_session_is_never_split_away(loaded_bank: ClipBank) -> None:
    _log(
        loaded_bank,
        "s1",
        [_judgement("p1", "s1", "gen_a", "gen_b", "left", "2026-01-01T00:00:01")],
    )
    pairs, _ = load_preference_pairs(loaded_bank, PairsCfg())
    train, val = split_by_session(pairs, val_frac=0.5, seed=0)

    assert len(train) == 1 and val == []


# -- grid, scoring and loss -------------------------------------------------


def _model() -> ArTransformer:
    """
    Returns:
      ArTransformer: a tiny model, deterministic under a fixed seed.
    """
    torch.manual_seed(0)
    return ArTransformer(
        cfg=ModelCfg(d_model=32, n_layers=2, n_heads=4, style_bottleneck=8),
        num_tokens=NUM_TOKENS,
        num_rq=DEPTH,
        num_tracks=3,
        style_dim=8,
        max_positions=64,
    ).eval()


def _side(tokens: torch.Tensor, score: torch.Tensor) -> dict[str, object]:
    """
    Args:
      tokens (torch.Tensor): (B, T, R) codes.
      score (torch.Tensor): (B, T) bool loss mask over frames.

    Returns:
      dict[str, object]: a collated side for sequence_logprob.
    """
    batch = tokens.shape[0]
    return {
        "tokens": tokens,
        "score": score,
        "track_idx": torch.zeros(batch, dtype=torch.long),
        "style": torch.zeros(batch, 8),
        "drop_id": torch.zeros(batch, dtype=torch.bool),
        "drop_style": torch.zeros(batch, dtype=torch.bool),
    }


def test_prepare_grid_matches_the_lightning_method() -> None:
    from train_ar import ArLightningModule

    model = _model()
    module = ArLightningModule(model, ArConfig())
    tokens = torch.randint(0, NUM_TOKENS, (2, 9, DEPTH))
    score = torch.ones(2, 9, dtype=torch.bool)
    score[:, :3] = False

    for expected, actual in zip(
        module._prepare(tokens, score),
        prepare_grid(tokens, model.pad_id, model.frames_per_pos, score),
    ):
        assert torch.equal(expected, actual)


def test_pad_corners_never_reach_the_loss() -> None:
    model = _model()
    tokens = torch.randint(0, NUM_TOKENS, (1, 9, DEPTH))
    _, targets, mask = prepare_grid(tokens, model.pad_id, 1)

    assert (targets[mask] < NUM_TOKENS).all()
    # every real code is scored exactly once, at its delayed position
    assert int(mask.sum()) == tokens.numel()


def test_prompt_frames_do_not_contribute() -> None:
    model = _model()
    tokens = torch.randint(0, NUM_TOKENS, (1, 12, DEPTH))
    full = torch.ones(1, 12, dtype=torch.bool)
    trimmed = full.clone()
    trimmed[:, :4] = False

    with torch.no_grad():
        lp_full, count_full = sequence_logprob(model, _side(tokens, full))
        lp_trim, count_trim = sequence_logprob(model, _side(tokens, trimmed))

    assert count_full.item() == tokens.numel()
    assert count_trim.item() == tokens.numel() - 4 * DEPTH
    assert lp_trim.item() > lp_full.item()  # fewer negative terms in the sum


def test_changing_a_masked_frame_changes_nothing_downstream() -> None:
    model = _model()
    tokens = torch.randint(0, NUM_TOKENS, (1, 12, DEPTH))
    score = torch.ones(1, 12, dtype=torch.bool)
    score[:, -3:] = False  # mask the tail so no scored position depends on it

    edited = tokens.clone()
    edited[:, -3:] = (edited[:, -3:] + 1) % NUM_TOKENS
    with torch.no_grad():
        before, _ = sequence_logprob(model, _side(tokens, score))
        after, _ = sequence_logprob(model, _side(edited, score))

    assert torch.allclose(before, after, atol=1e-4)


def test_dpo_loss_is_log_two_when_the_policy_is_the_reference() -> None:
    logprob = torch.tensor([-10.0, -20.0])
    loss, metrics = dpo_loss(logprob, logprob, logprob, logprob, beta=0.1)

    assert loss.item() == pytest.approx(0.6931, abs=1e-3)
    assert metrics["margin"].item() == pytest.approx(0.0)
    assert metrics["acc"].item() == 0.0


def test_dpo_loss_falls_as_the_margin_grows() -> None:
    ref = torch.zeros(1)
    losses = [
        dpo_loss(torch.tensor([w]), torch.tensor([-w]), ref, ref, beta=0.1)[0].item()
        for w in (0.0, 1.0, 5.0, 20.0)
    ]
    assert losses == sorted(losses, reverse=True)

    _, metrics = dpo_loss(torch.tensor([1.0]), torch.tensor([-1.0]), ref, ref, beta=0.1)
    assert metrics["acc"].item() == 1.0


def test_label_smoothing_bounds_the_loss_away_from_zero() -> None:
    ref = torch.zeros(1)
    huge = dpo_loss(torch.tensor([100.0]), torch.tensor([-100.0]), ref, ref, 0.1, 0.1)[
        0
    ]
    assert huge.item() > 0.5  # the flipped-label term keeps paying


# -- dataset, sampler, freezing, diversity ----------------------------------


def _tracks() -> dict[int, object]:
    """
    Returns:
      dict[int, object]: one synthetic TrackTokens keyed by track index.
    """
    from train_ar import TrackTokens

    return {
        0: TrackTokens(
            tokens=torch.zeros(64, DEPTH, dtype=torch.long),
            style=torch.arange(16, dtype=torch.float32).reshape(2, 8),
            style_bounds=torch.tensor([[0, 32], [32, 64]]),
            track_idx=0,
            track_name="t0",
            num_frames=64,
            fps=172.265625,
            val_windows=[],
        )
    }


def test_both_sides_share_one_window_and_the_prompt_is_masked(
    loaded_bank: ClipBank,
) -> None:
    _log(
        loaded_bank,
        "s1",
        [_judgement("p1", "s1", "gen_a", "gen_b", "left", "2026-01-01T00:00:01")],
    )
    pairs, _ = load_preference_pairs(loaded_bank, PairsCfg())
    pairs = [
        type(pairs[0])(
            pair_id=pairs[0].pair_id,
            session_id=pairs[0].session_id,
            tier=pairs[0].tier,
            group_id=pairs[0].group_id,
            winner=_spec("gen_a", prompt=4),
            loser=_spec("gen_b", prompt=4),
        )
    ]
    dataset = PreferenceDataset(pairs, _tracks(), loaded_bank, crop_frames=8, seed=3)
    item = dataset[0]

    assert item["win"]["start"].item() == item["lose"]["start"].item()
    assert item["win"]["tokens"].shape == (8, DEPTH)
    start = int(item["win"]["start"])
    expected = torch.arange(start, start + 8) >= 4
    assert torch.equal(item["win"]["score"], expected)


def test_batches_never_mix_two_clip_lengths(loaded_bank: ClipBank) -> None:
    _log(
        loaded_bank,
        "s1",
        [
            _judgement("p1", "s1", "gen_a", "gen_b", "left", "2026-01-01T00:00:01"),
            _judgement(
                "p2",
                "s1",
                "gen_i",
                "gen_j",
                "left",
                "2026-01-01T00:00:02",
                tier=Tier.STRUCTURE,
            ),
        ],
    )
    pairs, _ = load_preference_pairs(loaded_bank, PairsCfg())
    # crop below one clip's length forces two buckets
    dataset = PreferenceDataset(pairs, _tracks(), loaded_bank, crop_frames=6)
    dataset.pairs[1] = type(dataset.pairs[1])(
        pair_id="p2",
        session_id="s1",
        tier=Tier.STRUCTURE,
        group_id="g5",
        winner=_spec("gen_i", "g5", Tier.STRUCTURE, frames=4),
        loser=_spec("gen_j", "g5", Tier.STRUCTURE, frames=4),
    )
    sampler = LengthBucketSampler(dataset, batch_size=4, shuffle=False)

    for batch in sampler:
        lengths = {dataset.scored_length(i) for i in batch}
        assert len(lengths) == 1
    assert sum(len(b) for b in sampler) == len(dataset)


def test_collate_keeps_pairs_aligned(loaded_bank: ClipBank) -> None:
    _log(
        loaded_bank,
        "s1",
        [
            _judgement("p1", "s1", "gen_a", "gen_b", "left", "2026-01-01T00:00:01"),
            _judgement("p2", "s1", "gen_i", "gen_j", "right", "2026-01-01T00:00:02"),
        ],
    )
    pairs, _ = load_preference_pairs(loaded_bank, PairsCfg())
    dataset = PreferenceDataset(pairs, _tracks(), loaded_bank, crop_frames=12)
    batch = collate_pairs([dataset[0], dataset[1]])

    assert batch["win"]["tokens"].shape == (2, 12, DEPTH)
    assert batch["win"]["item_id"] == ["gen_a", "gen_j"]
    assert batch["lose"]["item_id"] == ["gen_b", "gen_i"]


def test_freezing_leaves_only_the_top_blocks_and_the_heads() -> None:
    model = _model()
    trainable, total = freeze_trunk(model, trainable_blocks=1)

    assert trainable < total
    assert all(not p.requires_grad for p in model.blocks[0].parameters())
    assert all(p.requires_grad for p in model.blocks[1].parameters())
    assert all(p.requires_grad for p in model.heads.parameters())
    assert not model.token_emb[0].weight.requires_grad


def test_freezing_everything_is_a_no_op_at_full_depth() -> None:
    model = _model()
    trainable, total = freeze_trunk(model, trainable_blocks=len(model.blocks))
    assert trainable == total


def test_identical_samples_read_as_zero_diversity() -> None:
    same = torch.randint(0, NUM_TOKENS, (20, DEPTH))
    reading = measure_diversity([same, same.clone(), same.clone()], NUM_TOKENS)
    assert reading.disagreement == pytest.approx(0.0)

    varied = [torch.randint(0, NUM_TOKENS, (20, DEPTH)) for _ in range(3)]
    assert measure_diversity(varied, NUM_TOKENS).disagreement > 0.5
    assert measure_diversity(varied, NUM_TOKENS).entropy > reading.entropy


def test_the_written_checkpoint_loads_as_an_ar_checkpoint(tmp_path: Path) -> None:
    from generate_ar import config_from_ckpt
    from train_dpo import DpoConfig, DpoLightningModule

    module = DpoLightningModule(_model(), _model(), DpoConfig())
    writer = ArCheckpointWriter(tmp_path, ArConfig())

    class _Trainer:
        current_epoch = 3
        global_step = 40

    path = writer._write(module, _Trainer(), "dpo_latest.ckpt")  # type: ignore[arg-type]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    assert isinstance(config_from_ckpt(ckpt), ArConfig)
    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    missing, unexpected = _model().load_state_dict(state, strict=True)
    assert not missing and not unexpected
    assert ckpt["dpo_config"]["dpo"]["beta"] == DpoCfg().beta
