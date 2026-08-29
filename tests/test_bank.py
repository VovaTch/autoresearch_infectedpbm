"""The token store: what survives a restart, and what refuses to mix."""

from __future__ import annotations

import numpy as np
import pytest

from ab_harness.model.bank import ClipBank, TokenizerMismatch
from ab_harness.model.types import ClipSpec, Conditioning, Sampling, Tier


def _spec(item: str, group: str = "g0", tier: Tier = Tier.BULK) -> ClipSpec:
    """
    Args:
      item (str): item id.
      group (str): group id.
      tier (Tier): rating tier.

    Returns:
      ClipSpec: a minimal spec.
    """
    return ClipSpec(
        item_id=item,
        tier=tier,
        group_id=group,
        n_frames=8,
        conditioning=Conditioning(track_idx=2, start_frame=100, style_window=3),
        sampling=Sampling(seed=42),
        checkpoint="ckpt_test",
    )


def test_roundtrip_survives_a_reopen(bank: ClipBank) -> None:
    spec = _spec("i0")
    tokens = np.random.randint(0, 2048, (8, 3))
    bank.add(spec, tokens)

    reopened = ClipBank(bank.root)
    assert reopened.spec("i0") == spec
    assert np.array_equal(reopened.tokens("i0"), tokens.astype(np.int16))


def test_tokens_are_stored_as_int16(bank: ClipBank) -> None:
    bank.add(_spec("i0"), np.random.randint(0, 2048, (8, 3)).astype(np.int64))
    assert bank.tokens("i0").dtype == np.int16


def test_has_is_false_until_tokens_land(bank: ClipBank) -> None:
    assert not bank.has("i0")
    bank.add(_spec("i0"), np.zeros((8, 3)))
    assert bank.has("i0")


def test_missing_item_raises(bank: ClipBank) -> None:
    with pytest.raises(KeyError):
        bank.tokens("nope")
    with pytest.raises(KeyError):
        bank.spec("nope")


def test_non_2d_tokens_are_rejected(bank: ClipBank) -> None:
    with pytest.raises(ValueError):
        bank.add(_spec("i0"), np.zeros(8))


def test_groups_bucket_by_group_id(bank: ClipBank) -> None:
    bank.add(_spec("i0", "gA"), np.zeros((8, 3)))
    bank.add(_spec("i1", "gA"), np.zeros((8, 3)))
    bank.add(_spec("i2", "gB", Tier.STRUCTURE), np.zeros((8, 3)))
    groups = bank.groups()
    assert sorted(groups) == ["gA", "gB"]
    assert len(groups["gA"]) == 2
    assert [s.item_id for s in bank.specs(Tier.STRUCTURE)] == ["i2"]


def test_a_bank_refuses_a_different_tokenizer(bank: ClipBank) -> None:
    with pytest.raises(TokenizerMismatch):
        bank.require_tokenizer("tokens_other")
    # re-initialising with the original tag is fine and records new checkpoints
    bank.init_manifest("tokens_test", {"num_rq": 3}, "ckpt_second")
    assert bank.manifest()["checkpoints"] == ["ckpt_second", "ckpt_test"]


def test_readding_an_item_overwrites_rather_than_duplicates(bank: ClipBank) -> None:
    bank.add(_spec("i0"), np.zeros((8, 3)))
    bank.add(_spec("i0"), np.ones((8, 3)))
    assert len(bank.specs()) == 1
    assert bank.tokens("i0").sum() == 24


# -- measured quality --------------------------------------------------------


def test_fill_roundtrips_and_survives_a_reopen(bank: ClipBank) -> None:
    bank.add(_spec("i0"), np.zeros((8, 3)))
    assert bank.fill("i0") is None
    bank.set_fill("i0", 0.42)
    assert bank.fill("i0") == pytest.approx(0.42)
    assert ClipBank(bank.root).fill("i0") == pytest.approx(0.42)


def test_fill_is_kept_out_of_the_spec(bank: ClipBank) -> None:
    """A spec is the recipe; fill is an outcome of running it."""
    spec = _spec("i0")
    bank.add(spec, np.zeros((8, 3)))
    bank.set_fill("i0", 0.1)
    assert ClipBank(bank.root).spec("i0") == spec


def test_fills_accumulate_rather_than_overwrite(bank: ClipBank) -> None:
    bank.set_fill("i0", 0.3)
    bank.set_fill("i1", 0.7)
    reopened = ClipBank(bank.root)
    assert reopened.fill("i0") == pytest.approx(0.3)
    assert reopened.fill("i1") == pytest.approx(0.7)
