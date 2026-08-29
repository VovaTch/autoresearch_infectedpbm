"""
The rating loop: one judgement per decision, mapped back to the right item.

The blind side mapping is the part worth pinning. The view only ever says
"left" or "right"; if that resolved to the wrong item id the whole dataset would
be silently inverted and nothing downstream would notice.
"""

from __future__ import annotations

import pytest

from ab_harness.model.bank import ClipBank
from ab_harness.model.pair_sampler import PairSampler
from ab_harness.model.pipeline import PairPipeline
from ab_harness.model.types import Pair
from ab_harness.viewmodel.session_vm import SessionViewModel
from tests.conftest import FakeProducer, FakeSink


@pytest.fixture
def session(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> SessionViewModel:
    """
    Args:
      qapp: session QCoreApplication.
      sampler (PairSampler): seeded sampler.
      bank (ClipBank): empty bank.
      fake_sink (FakeSink): in-memory judgement sink.

    Returns:
      SessionViewModel: a view-model over a synchronous fake producer.
    """
    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=2, structure_live=True
    )
    return SessionViewModel(pipeline, fake_sink, target_lufs=-23.0, session_id="sess0")


def test_advance_shows_a_pair_and_emits_it(session: SessionViewModel) -> None:
    seen: list[Pair] = []
    session.pair_changed.connect(seen.append)
    assert session.advance() is True
    assert session.pair is not None
    assert seen and seen[0] is session.pair


@pytest.mark.parametrize("choice,side", [("left", "left"), ("right", "right")])
def test_choice_resolves_to_the_item_on_that_side(
    session: SessionViewModel, fake_sink: FakeSink, choice: str, side: str
) -> None:
    session.advance()
    pair = session.pair
    assert pair is not None
    expected = pair.spec.left.item_id if side == "left" else pair.spec.right.item_id

    session.choose(choice)  # type: ignore[arg-type]
    assert len(fake_sink.rows) == 1
    row = fake_sink.rows[0]
    assert row.choice == choice
    assert row.chosen_item_id == expected
    assert row.item_left == pair.spec.left.item_id
    assert row.item_right == pair.spec.right.item_id


def test_a_tie_records_no_winner(
    session: SessionViewModel, fake_sink: FakeSink
) -> None:
    session.advance()
    session.choose("tie")
    assert fake_sink.rows[0].choice == "tie"
    assert fake_sink.rows[0].chosen_item_id == ""


def test_exactly_one_judgement_per_decision(
    session: SessionViewModel, fake_sink: FakeSink
) -> None:
    session.advance()
    for _ in range(5):
        session.choose("left")
    assert len(fake_sink.rows) == 5
    assert session.rated == 5
    assert len({row.pair_id for row in fake_sink.rows}) == 5


def test_choosing_advances_to_a_new_pair(session: SessionViewModel) -> None:
    session.advance()
    first = session.pair
    assert first is not None
    session.choose("left")
    assert session.pair is not None
    assert session.pair.spec.pair_id != first.spec.pair_id


def test_skip_records_nothing_but_still_advances(
    session: SessionViewModel, fake_sink: FakeSink
) -> None:
    session.advance()
    first = session.pair
    session.skip()
    assert fake_sink.rows == []
    assert session.rated == 0
    assert session.pair is not None and session.pair is not first


def test_choosing_with_no_pair_on_screen_does_nothing(
    session: SessionViewModel, fake_sink: FakeSink
) -> None:
    session.choose("left")
    assert fake_sink.rows == []


def test_response_time_is_recorded(
    session: SessionViewModel, fake_sink: FakeSink
) -> None:
    session.advance()
    session.choose("right")
    assert fake_sink.rows[0].response_ms >= 0


def test_session_metadata_is_carried_onto_every_row(
    session: SessionViewModel, fake_sink: FakeSink
) -> None:
    session.advance()
    pair = session.pair
    assert pair is not None
    session.choose("left")
    row = fake_sink.rows[0]
    assert row.session_id == "sess0"
    assert row.lufs_target == -23.0
    assert row.tier is pair.spec.tier
    assert row.question == pair.spec.question
    assert row.is_repeat == pair.spec.is_repeat
    assert row.is_anchor == pair.spec.is_anchor
    assert row.was_live is True


def test_waiting_is_reported_when_nothing_is_ready(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    class DeadProducer(FakeProducer):
        def submit(self, specs) -> None:  # type: ignore[no-untyped-def]
            self.requested.extend(s.item_id for s in specs)

    pipeline = PairPipeline(sampler, DeadProducer(), bank, depth=1, structure_live=True)
    session = SessionViewModel(pipeline, fake_sink)
    states: list[bool] = []
    session.waiting.connect(states.append)
    assert session.advance() is False
    assert states == [True]
    assert session.pair is None


def test_queue_depth_is_published(session: SessionViewModel) -> None:
    depths: list[tuple[int, int]] = []
    session.queue_changed.connect(lambda r, i: depths.append((r, i)))
    session.advance()
    assert depths


def test_jumping_to_another_queue_entry_does_not_lose_the_current_pair(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    """
    A worklist has to be non-destructive: picking a second entry while the first
    is on screen must put the first back, not throw it away unjudged.
    """
    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=3, structure_live=True
    )
    session = SessionViewModel(pipeline, fake_sink)
    assert session.advance()
    first = session.pair
    assert first is not None

    other = [item.pair_id for item in pipeline.worklist() if item.ready][-1]
    assert session.select(other)
    assert session.pair is not None and session.pair.spec.pair_id == other
    assert first.spec.pair_id in {item.pair_id for item in pipeline.worklist()}

    assert session.select(first.spec.pair_id)
    assert session.pair is not None and session.pair.spec.pair_id == first.spec.pair_id
    assert not fake_sink.rows, "switching entries logs nothing"
