"""
The pipeline exists so the rater never waits. These pin the ways it could stall.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from ab_harness.model.bank import ClipBank
from ab_harness.model.pair_sampler import PairSampler, SamplerCfg, TrackInfo
from ab_harness.model.pipeline import PairPipeline
from ab_harness.model.types import Tier
from ab_harness.model.types import Clip, Pair


def tracks_for(bank: ClipBank) -> list[TrackInfo]:
    """
    Args:
      bank (ClipBank): unused; keeps the helper callable from fixtures.

    Returns:
      list[TrackInfo]: a small synthetic corpus.
    """
    window = int(10.0 * 172.265625)
    return [
        TrackInfo(
            i,
            f"track_{i}",
            40_000,
            tuple((w * window, (w + 1) * window) for w in range(23)),
        )
        for i in range(5)
    ]


from tests.conftest import FakeProducer


@pytest.fixture
def pipeline(sampler: PairSampler, bank: ClipBank) -> PairPipeline:
    """
    Args:
      sampler (PairSampler): seeded sampler.
      bank (ClipBank): empty bank the fake producer records into.

    Returns:
      PairPipeline: a pipeline with a fake producer, structure tier live so the
        empty bank does not starve it.
    """
    producer = FakeProducer(store=bank)
    return PairPipeline(sampler, producer, bank, depth=3, structure_live=True)


def test_depth_is_maintained(pipeline: PairPipeline) -> None:
    pipeline.pump()
    assert pipeline.ready + pipeline.inflight == 3
    for _ in range(5):
        assert pipeline.next_pair() is not None
        assert pipeline.ready + pipeline.inflight == 3


def test_a_pair_is_never_served_twice(pipeline: PairPipeline) -> None:
    seen = {pipeline.next_pair().spec.pair_id for _ in range(20)}  # type: ignore[union-attr]
    assert len(seen) == 20


def test_pairs_come_back_with_both_clips_decoded(pipeline: PairPipeline) -> None:
    pair = pipeline.next_pair()
    assert pair is not None
    assert pair.left.spec.item_id == pair.spec.left.item_id
    assert pair.right.spec.item_id == pair.spec.right.item_id
    assert pair.left.pcm.size and pair.right.pcm.size


def test_a_producer_that_never_delivers_yields_no_pair_instead_of_blocking(
    sampler: PairSampler, bank: ClipBank
) -> None:
    class DeadProducer(FakeProducer):
        def submit(self, specs) -> None:  # type: ignore[no-untyped-def]
            self.requested.extend(s.item_id for s in specs)

    pipeline = PairPipeline(sampler, DeadProducer(), bank, depth=2, structure_live=True)
    assert pipeline.next_pair() is None
    assert pipeline.inflight == 2  # still trying, not wedged


def test_a_failed_clip_can_be_dropped_without_losing_the_queue(
    sampler: PairSampler, bank: ClipBank
) -> None:
    class DeadProducer(FakeProducer):
        def submit(self, specs) -> None:  # type: ignore[no-untyped-def]
            self.requested.extend(s.item_id for s in specs)

    dead = DeadProducer()
    pipeline = PairPipeline(sampler, dead, bank, depth=2, structure_live=True)
    pipeline.pump()
    assert pipeline.inflight == 2

    doomed = next(iter(pipeline._inflight.values()))
    pipeline.drop_stalled([doomed.left.item_id])
    assert pipeline.dropped == 1
    assert pipeline.inflight == 1

    # the queue recovers as soon as a working producer is available again
    pipeline.producer = FakeProducer(store=bank)
    pipeline.pump()
    assert pipeline.next_pair() is not None


def test_structure_tier_is_not_served_on_demand_when_it_is_not_banked(
    tracks: list[TrackInfo], bank: ClipBank
) -> None:
    # every draw is structure, nothing is banked, and live structure is off:
    # the rating lane must decline rather than queue a 90 s generation
    cfg = SamplerCfg(
        fps=172.265625, structure_seconds=4.0, bulk_share=0.0, repeat_rate=0.0
    )
    sampler = PairSampler(tracks, cfg, "ckpt", random.Random(3))
    producer = FakeProducer(store=bank)
    pipeline = PairPipeline(
        sampler, producer, bank, depth=2, structure_live=False, structure_backfill=0
    )
    pipeline.pump()
    assert pipeline.inflight == 0
    assert producer.requested == []


def test_a_banked_structure_pair_is_served_even_when_live_is_off(
    tracks: list[TrackInfo], bank: ClipBank
) -> None:
    cfg = SamplerCfg(
        fps=172.265625, structure_seconds=4.0, bulk_share=0.0, repeat_rate=0.0
    )
    # bank everything the sampler will draw, by draining a twin sampler first
    warm = PairSampler(tracks, cfg, "ckpt", random.Random(3))
    for _ in range(8):
        spec = warm.next_spec()
        for side in (spec.left, spec.right):
            bank.add(side, np.zeros((side.n_frames, 3), dtype=np.int16))

    sampler = PairSampler(tracks, cfg, "ckpt", random.Random(3))
    pipeline = PairPipeline(sampler, FakeProducer(store=bank), bank, depth=2)
    pair = pipeline.next_pair()
    assert pair is not None and pair.spec.tier is Tier.STRUCTURE


def test_audio_of_served_pairs_is_released(pipeline: PairPipeline) -> None:
    for _ in range(10):
        pipeline.next_pair()
    # only what is still in flight may be held
    assert len(pipeline._clips) <= 2 * pipeline.depth


# -- structure backfill ------------------------------------------------------


def _structure_only(tracks: list[TrackInfo], bank: ClipBank, **kwargs) -> PairPipeline:
    """
    Args:
      tracks (list[TrackInfo]): synthetic corpus.
      bank (ClipBank): the store.
      **kwargs: overrides passed to PairPipeline.

    Returns:
      PairPipeline: a pipeline whose sampler only ever draws the structure tier.
    """
    cfg = SamplerCfg(
        fps=172.265625,
        bulk_seconds=1.0,
        structure_seconds=4.0,
        bulk_share=0.0,
        repeat_rate=0.0,
        anchor_rate=0.0,
    )
    sampler = PairSampler(tracks, cfg, "ckpt", random.Random(3))
    return PairPipeline(sampler, FakeProducer(store=bank), bank, **kwargs)


def test_declined_structure_draws_are_counted_not_swallowed(
    tracks: list[TrackInfo], bank: ClipBank
) -> None:
    """
    Regression: a fifth of all draws were being discarded silently, so a whole
    session could go by without the structure tier appearing and no way to see
    why from the app.
    """
    pipeline = _structure_only(tracks, bank, depth=2, structure_backfill=0)
    pipeline.pump()
    assert pipeline.skipped_structure > 0
    assert pipeline.inflight == 0


def test_backfill_banks_structure_pairs_in_the_background(
    sampler: PairSampler, bank: ClipBank
) -> None:
    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=2, structure_backfill=1
    )
    assert pipeline.banked_structure_pairs() == 0
    for _ in range(4):
        pipeline.pump()
    assert pipeline.banked_structure_pairs() >= 1


def test_backfill_waits_until_the_rating_queue_is_full(
    sampler: PairSampler, bank: ClipBank
) -> None:
    """
    The worker is FIFO, so a structure batch queued ahead of bulk clips would
    hold the rater up for two minutes.
    """

    class DeadProducer(FakeProducer):
        def submit(self, specs) -> None:  # type: ignore[no-untyped-def]
            self.requested.extend(s.item_id for s in specs)

    producer = DeadProducer()
    pipeline = PairPipeline(sampler, producer, bank, depth=3, structure_backfill=1)
    pipeline.pump()
    # nothing ever completes, so the queue is never full and backfill holds off
    assert pipeline.backfilling == 0


def test_backfill_audio_is_not_kept(sampler: PairSampler, bank: ClipBank) -> None:
    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=2, structure_backfill=1
    )
    for _ in range(4):
        pipeline.pump()
    banked = pipeline.banked_structure_pairs()
    assert banked >= 1
    held = {
        clip.spec.item_id
        for clip in pipeline._clips.values()
        if clip.spec.tier is Tier.STRUCTURE
    }
    assert not held, "backfill audio was retained; it only needs to reach the bank"


def test_backfill_can_be_disabled(sampler: PairSampler, bank: ClipBank) -> None:
    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=2, structure_backfill=0
    )
    for _ in range(4):
        pipeline.pump()
    assert pipeline.backfilling == 0
    assert pipeline.banked_structure_pairs() == 0


def test_a_banked_structure_pair_becomes_servable(
    tracks: list[TrackInfo], bank: ClipBank
) -> None:
    """The point of the whole lane: what it banks is what later draws serve."""
    pipeline = _structure_only(tracks, bank, depth=1, structure_backfill=1)
    for _ in range(12):
        pipeline.pump()
    assert pipeline.banked_structure_pairs() >= 1
    served = [pipeline.next_pair() for _ in range(6)]
    assert any(p is not None and p.spec.tier is Tier.STRUCTURE for p in served)


def test_structure_banked_by_another_process_becomes_servable(
    sampler: PairSampler, bank: ClipBank, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The worker owns every bank write, so the UI's index is a startup snapshot.
    Before the refresh this test's structure draws were declined forever and the
    tier never appeared in a session, however much the backfill produced.
    """
    import ab_harness.model.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "REFRESH_INTERVAL_S", 0.0)
    pipe = PairPipeline(
        sampler,
        FakeProducer(store=bank),
        bank,
        depth=2,
        structure_live=False,
        structure_backfill=0,
    )
    for _ in range(10):
        pipe.next_pair()
    assert pipe.skipped_structure > 0
    assert pipe.banked_structure_pairs() == 0

    # A second ClipBank stands in for the worker process: same directory, its
    # own in-memory index. The pipeline's bank knows nothing of these writes.
    worker_bank = ClipBank(bank.root)
    other = PairSampler(
        sampler.tracks, sampler.cfg, checkpoint="ckpt_test", rng=random.Random(7)
    )
    spec = other.next_spec(tier=Tier.STRUCTURE)
    for clip in (spec.left, spec.right):
        worker_bank.add(clip, np.zeros((clip.n_frames, 3), dtype=np.int16))

    assert pipe.banked_structure_pairs() == 1
    tiers = {
        pair.spec.tier
        for pair in (pipe.next_pair() for _ in range(40))
        if pair is not None
    }
    assert Tier.STRUCTURE in tiers


# -- worklist ----------------------------------------------------------------


def test_worklist_shows_ready_pairs_before_pending_ones(
    sampler: PairSampler, bank: ClipBank
) -> None:
    class SlowProducer(FakeProducer):
        """Delivers nothing, so everything it is given stays in flight."""

        def submit(self, specs) -> None:  # type: ignore[no-untyped-def]
            self.requested.extend(s.item_id for s in specs)

    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=2, structure_live=True
    )
    pipeline.pump()
    ready = [item for item in pipeline.worklist() if item.ready]
    assert len(ready) == pipeline.ready
    assert [item.ready for item in pipeline.worklist()] == sorted(
        (item.ready for item in pipeline.worklist()), reverse=True
    )

    pipeline.producer = SlowProducer()
    pipeline.request(Tier.STRUCTURE, 1)
    pending = [item for item in pipeline.worklist() if not item.ready]
    assert len(pending) == 1
    assert pending[0].tier is Tier.STRUCTURE
    assert pending[0].seconds == pytest.approx(sampler.cfg.structure_seconds, abs=0.02)


def test_take_serves_the_chosen_pair_not_the_head(
    sampler: PairSampler, bank: ClipBank
) -> None:
    """
    The reason the worklist exists: with a random draw the 90 s tier turned up
    once in thirteen pairs, because the mix is a coin flip and not a schedule.
    """
    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=4, structure_live=True
    )
    pipeline.pump()
    items = [item for item in pipeline.worklist() if item.ready]
    assert len(items) >= 2

    wanted = items[-1].pair_id
    pair = pipeline.take(wanted)
    assert pair is not None and pair.spec.pair_id == wanted
    assert wanted not in {item.pair_id for item in pipeline.worklist()}
    assert pipeline.take(wanted) is None


def test_request_queues_the_asked_for_tier_regardless_of_the_mix(
    sampler: PairSampler, bank: ClipBank
) -> None:
    producer = FakeProducer(store=bank)
    pipeline = PairPipeline(
        sampler, producer, bank, depth=1, structure_live=False, structure_backfill=0
    )
    assert pipeline.request(Tier.STRUCTURE, 2) == 2
    pipeline.pump()
    served = [pipeline.next_pair() for _ in range(2)]
    assert [p.spec.tier for p in served if p is not None].count(Tier.STRUCTURE) == 2


def test_request_prefers_banked_structure_the_rater_has_not_heard(
    sampler: PairSampler, bank: ClipBank
) -> None:
    """
    Decoding a banked pair is seconds; generating one is two minutes. Serving
    unheard material first is also what stops a growing bank replaying itself.
    """
    warm = PairSampler(
        sampler.tracks, sampler.cfg, checkpoint="ckpt_test", rng=random.Random(11)
    )
    banked = [warm.next_spec(tier=Tier.STRUCTURE) for _ in range(2)]
    for spec in banked:
        for clip in (spec.left, spec.right):
            bank.add(clip, np.zeros((clip.n_frames, 3), dtype=np.int16))

    producer = FakeProducer(store=bank)
    pipeline = PairPipeline(
        sampler, producer, bank, depth=1, structure_live=False, structure_backfill=0
    )
    pipeline.request(Tier.STRUCTURE, 1)
    # nothing new was sampled: both clips were already in the bank
    assert set(producer.requested) <= {
        clip.item_id for spec in banked for clip in (spec.left, spec.right)
    }

    # once everything banked has been judged, a request generates instead
    for spec in banked:
        pipeline.mark_rated(spec)
    producer.requested.clear()
    pipeline.request(Tier.STRUCTURE, 1)
    assert producer.requested and not set(producer.requested) <= {
        clip.item_id for spec in banked for clip in (spec.left, spec.right)
    }


# -- the dead-air gate -------------------------------------------------------


class QuietProducer(FakeProducer):
    """
    Producer whose clips carry a chosen fill fraction.

    Args:
      fill (float): fill reported for every generated clip.
      store (ClipBank | None): bank to record into.
    """

    def __init__(self, fill: float, store: ClipBank | None = None) -> None:
        super().__init__(store=store)
        self.fill = fill

    def submit(self, specs) -> None:  # type: ignore[no-untyped-def]
        before = len(self._queue)
        super().submit(specs)
        for clip in self._queue[before:]:
            clip.fill = self.fill
            if self.store is not None:
                self.store.set_fill(clip.spec.item_id, self.fill)


def test_dead_pairs_never_reach_the_worklist(
    sampler: PairSampler, bank: ClipBank
) -> None:
    pipeline = PairPipeline(
        sampler,
        QuietProducer(0.05, bank),
        bank,
        depth=2,
        structure_backfill=0,
        min_fill=0.25,
    )
    for _ in range(5):
        pipeline.pump()
    assert pipeline.ready == 0
    assert pipeline.rejected_quiet > 0
    assert pipeline.next_pair() is None


def test_full_pairs_pass_the_gate(sampler: PairSampler, bank: ClipBank) -> None:
    pipeline = PairPipeline(
        sampler,
        QuietProducer(0.9, bank),
        bank,
        depth=2,
        structure_backfill=0,
        min_fill=0.25,
    )
    pipeline.pump()
    assert pipeline.next_pair() is not None
    assert pipeline.rejected_quiet == 0


def test_the_gate_is_off_at_zero(sampler: PairSampler, bank: ClipBank) -> None:
    pipeline = PairPipeline(
        sampler,
        QuietProducer(0.0, bank),
        bank,
        depth=2,
        structure_backfill=0,
        min_fill=0.0,
    )
    pipeline.pump()
    assert pipeline.next_pair() is not None
    assert pipeline.rejected_quiet == 0


def test_an_unmeasured_clip_is_given_the_benefit_of_the_doubt(
    sampler: PairSampler, bank: ClipBank
) -> None:
    """A producer that reports no fill must not have every pair thrown away."""
    pipeline = PairPipeline(
        sampler,
        FakeProducer(store=bank),
        bank,
        depth=2,
        structure_backfill=0,
        min_fill=0.25,
    )
    pipeline.pump()
    assert pipeline.next_pair() is not None
    assert pipeline.rejected_quiet == 0


def test_a_reference_side_is_exempt(sampler: PairSampler, bank: ClipBank) -> None:
    """
    An anchor's reference side is real tokens and its answer is known, so it
    must survive the gate whatever it measures.
    """
    from ab_harness.model.types import Tier

    cfg = SamplerCfg(
        fps=172.265625,
        bulk_seconds=1.0,
        bulk_share=1.0,
        repeat_rate=0.0,
        anchor_rate=1.0,
    )
    anchors = PairSampler(tracks_for(bank), cfg, "ckpt", random.Random(4))
    producer = QuietProducer(0.02, bank)
    pipeline = PairPipeline(
        anchors, producer, bank, depth=1, structure_backfill=0, min_fill=0.25
    )
    pipeline.pump()
    # both sides read as empty, but only the generated one may trigger a drop
    spec = anchors.next_spec()
    assert spec.is_anchor
    reference = spec.left if spec.left.is_reference else spec.right
    assert reference.is_reference
    pair = Pair(
        spec=spec,
        left=Clip(
            spec.left, np.zeros((4, 3), np.int16), np.zeros(8, np.int16), fill=0.02
        ),
        right=Clip(
            spec.right, np.zeros((4, 3), np.int16), np.zeros(8, np.int16), fill=0.02
        ),
    )
    # the generated side is what fails it; a pair of two references would pass
    assert pipeline._too_quiet(pair)
    both_ref = Pair(
        spec=spec,
        left=Clip(
            reference, np.zeros((4, 3), np.int16), np.zeros(8, np.int16), fill=0.0
        ),
        right=Clip(
            reference, np.zeros((4, 3), np.int16), np.zeros(8, np.int16), fill=0.0
        ),
    )
    assert not pipeline._too_quiet(both_ref)


def test_a_banked_clip_known_to_be_dead_is_not_reserved(
    sampler: PairSampler, bank: ClipBank
) -> None:
    pipeline = PairPipeline(
        sampler,
        FakeProducer(store=bank),
        bank,
        depth=1,
        structure_backfill=1,
        min_fill=0.25,
    )
    for _ in range(6):
        pipeline.pump()
    for spec in bank.specs():
        bank.set_fill(spec.item_id, 0.01)
    assert pipeline._banked_structure_spec() is None
