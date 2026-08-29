"""
The sampler decides what data the whole ladder is trained on.

Each of these guards a property that cannot be fixed after the fact: pairs drawn
with mismatched conditioning, or with an identifiable side, are not repairable
by any amount of downstream care.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from ab_harness.model.pair_sampler import (
    PairSampler,
    SamplerCfg,
    TrackInfo,
    load_corpus,
    pick_style_window,
    save_corpus,
)
from ab_harness.model.types import Tier

DRAWS = 3000


@pytest.fixture
def specs(sampler: PairSampler) -> list:
    """
    Args:
      sampler (PairSampler): the seeded sampler.

    Returns:
      list: a long run of drawn pair specs.
    """
    return [sampler.next_spec() for _ in range(DRAWS)]


def test_tier_mix_matches_the_configured_share(specs: list) -> None:
    counts = Counter(spec.tier for spec in specs)
    assert counts[Tier.BULK] / DRAWS == pytest.approx(0.8, abs=0.03)


def test_repeat_and_anchor_rates_match_config(specs: list) -> None:
    assert sum(s.is_repeat for s in specs) / DRAWS == pytest.approx(0.1, abs=0.02)
    assert sum(s.is_anchor and not s.is_repeat for s in specs) / DRAWS == pytest.approx(
        0.05, abs=0.02
    )


def test_a_repeat_replays_an_earlier_comparison(specs: list) -> None:
    seen: set[frozenset[str]] = set()
    checked = 0
    for spec in specs:
        key = frozenset({spec.left.item_id, spec.right.item_id})
        if spec.is_repeat:
            assert key in seen, "a repeat must replay a comparison already shown"
            checked += 1
        seen.add(key)
    assert checked > 50


def test_a_repeat_gets_a_fresh_pair_id(specs: list) -> None:
    ids = [s.pair_id for s in specs]
    assert len(set(ids)) == len(ids)


def test_both_clips_of_a_pair_share_conditioning_and_differ_only_in_seed(
    specs: list,
) -> None:
    for spec in specs:
        if spec.is_anchor:
            continue
        assert spec.left.group_id == spec.right.group_id
        assert spec.left.conditioning == spec.right.conditioning
        assert spec.left.n_frames == spec.right.n_frames
        assert spec.left.sampling.seed != spec.right.sampling.seed


def test_an_anchor_puts_a_generation_against_real_tokens(specs: list) -> None:
    anchors = [s for s in specs if s.is_anchor and not s.is_repeat]
    assert anchors
    for spec in anchors:
        generators = {spec.left.generator, spec.right.generator}
        assert generators == {"ar", "reference"}
        reference = spec.left if spec.left.is_reference else spec.right
        assert reference.item_id.startswith("ref_")


def test_sides_are_randomized(sampler: PairSampler, tracks: list[TrackInfo]) -> None:
    # a reference clip is identifiable by its id, so it makes a clean probe of
    # which side the sampler put it on
    cfg = SamplerCfg(fps=172.265625, bulk_seconds=1.0, anchor_rate=1.0, repeat_rate=0.0)
    anchor_only = PairSampler(tracks, cfg, "ckpt", random.Random(7))
    left = sum(anchor_only.next_spec().left.is_reference for _ in range(600))
    assert left / 600 == pytest.approx(0.5, abs=0.06)


def test_the_same_seed_gives_the_same_stream(tracks: list[TrackInfo]) -> None:
    cfg = SamplerCfg(fps=172.265625, bulk_seconds=1.0, structure_seconds=4.0)

    def run() -> list[tuple[str, str]]:
        sampler = PairSampler(tracks, cfg, "c", random.Random(99))
        return [
            (spec.left.item_id, spec.right.item_id)
            for spec in (sampler.next_spec() for _ in range(40))
        ]

    assert run() == run()


def test_all_eight_conditioning_combinations_are_reachable(specs: list) -> None:
    combos = {
        (
            s.left.conditioning.use_track_id,
            s.left.conditioning.use_style,
            s.left.conditioning.prompt_frames > 0,
        )
        for s in specs
    }
    assert len(combos) == 8, f"only saw {sorted(combos)}"


def test_a_fully_unconditional_draw_happens(specs: list) -> None:
    assert any(
        not s.left.conditioning.use_track_id
        and not s.left.conditioning.use_style
        and s.left.conditioning.prompt_frames == 0
        for s in specs
    )


def test_spans_stay_inside_the_track(specs: list, tracks: list[TrackInfo]) -> None:
    lengths = {t.track_idx: t.num_frames for t in tracks}
    for spec in specs:
        cond = spec.left.conditioning
        assert 0 <= cond.start_frame
        assert cond.start_frame + spec.left.n_frames <= lengths[cond.track_idx]


def test_style_window_is_disjoint_from_the_generated_span(
    specs: list, tracks: list[TrackInfo]
) -> None:
    bounds = {t.track_idx: t.style_bounds for t in tracks}
    for spec in specs:
        cond = spec.left.conditioning
        if not cond.use_style:
            continue
        low, high = bounds[cond.track_idx][cond.style_window]
        start, end = cond.start_frame, cond.start_frame + spec.left.n_frames
        assert not (start < high and low < end), "style window leaked into the target"


def test_pick_style_falls_back_to_the_farthest_window_when_none_are_free() -> None:
    # one window covering everything: nothing can be disjoint
    assert pick_style_window([(0, 100)], 0, 100, random.Random(0)) == 0
    assert pick_style_window([], 0, 10, random.Random(0)) == -1


def test_an_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError):
        PairSampler([])


def test_corpus_roundtrips_through_disk(tmp_path, tracks: list[TrackInfo]) -> None:
    path = tmp_path / "corpus.json"
    save_corpus(path, tracks)
    assert load_corpus(path) == tracks
    assert load_corpus(tmp_path / "absent.json") == []
