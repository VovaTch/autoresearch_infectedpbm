"""
The worklist: the rater's way of choosing what to rate and what to generate.

The random tier mix is a coin flip per pair, so the 90 s tier could go a dozen
pairs without appearing. These pin the three things that fix it -- the queue is
visible, an entry can be picked, and a tier can be asked for -- plus the filter
that keeps mostly-silent generations out of the way, and the blind.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from ab_harness.model.bank import ClipBank
from ab_harness.model.pair_sampler import PairSampler
from ab_harness.model.pipeline import PairPipeline, WorkItem
from ab_harness.model.types import Clip, Tier
from ab_harness.view.main_window import MainWindow
from ab_harness.view.worklist import PAIR_ID_ROLE, WorklistPanel
from ab_harness.viewmodel.player_vm import PlayerViewModel
from ab_harness.viewmodel.session_vm import SessionViewModel
from tests.conftest import FakeProducer, FakeSink


def _rows(panel: WorklistPanel) -> list[str]:
    """
    Args:
      panel (WorklistPanel): the panel to read.

    Returns:
      list[str]: the visible row labels.
    """
    return [panel._list.item(i).text() for i in range(panel._list.count())]


def _pipeline(sampler: PairSampler, bank: ClipBank, **kwargs) -> PairPipeline:
    """
    Args:
      sampler (PairSampler): seeded sampler.
      bank (ClipBank): store the fake producer records into.
      **kwargs: overrides passed to PairPipeline.

    Returns:
      PairPipeline: a pipeline over a fake producer.
    """
    options = {"depth": 2, "structure_live": True} | kwargs
    return PairPipeline(sampler, FakeProducer(store=bank), bank, **options)


def test_rows_show_length_and_fullness_and_never_an_id(
    qapp, sampler: PairSampler, bank: ClipBank
) -> None:
    pipeline = _pipeline(sampler, bank)
    pipeline.pump()
    items = pipeline.worklist()
    assert items

    panel = WorklistPanel()
    panel.set_items(items)
    rows = _rows(panel)
    assert len(rows) == len(items)
    assert all("fill" in row or "generating" in row for row in rows)
    joined = " ".join(rows)
    for item in items:
        assert item.pair_id not in joined
        if item.pair is not None:
            assert item.pair.spec.left.item_id not in joined
            assert str(item.pair.spec.left.sampling.seed) not in joined


def test_mostly_empty_pairs_can_be_filtered_out(
    qapp, sampler: PairSampler, bank: ClipBank
) -> None:
    pipeline = _pipeline(sampler, bank)
    pipeline.pump()
    loud = [item for item in pipeline.worklist() if item.ready][0]
    silent_pair = loud.pair
    assert silent_pair is not None
    hushed = WorkItem(
        pair_id="quiet",
        tier=Tier.BULK,
        seconds=10.0,
        pair=type(silent_pair)(
            spec=silent_pair.spec,
            left=Clip(
                spec=silent_pair.spec.left,
                tokens=silent_pair.left.tokens,
                pcm=np.zeros(44100, dtype=np.int16),
            ),
            right=silent_pair.right,
        ),
    )

    panel = WorklistPanel(quiet_fill=0.25)
    panel.set_items([loud, hushed])
    assert len(_rows(panel)) == 2

    panel._hide_quiet.setChecked(True)
    assert len(_rows(panel)) == 1
    assert "quiet hidden" in panel._status.text()


def test_clicking_a_row_asks_for_that_pair(
    qapp, sampler: PairSampler, bank: ClipBank
) -> None:
    pipeline = _pipeline(sampler, bank)
    pipeline.pump()
    items = [item for item in pipeline.worklist() if item.ready]
    panel = WorklistPanel()
    panel.set_items(items)

    seen: list[str] = []
    panel.selected.connect(seen.append)
    row = panel._list.item(len(items) - 1)
    panel._emit_selected(row)
    assert seen == [row.data(PAIR_ID_ROLE)]


def test_the_generate_buttons_name_a_tier_and_a_count(qapp) -> None:
    from PySide6.QtWidgets import QPushButton

    panel = WorklistPanel()
    panel._count.setValue(3)
    asked: list[tuple[str, int]] = []
    panel.generate.connect(lambda tier, count: asked.append((tier, count)))
    for button in panel.findChildren(QPushButton):
        button.click()
    assert asked == [(str(Tier.BULK), 3), (str(Tier.STRUCTURE), 3)]


def test_the_window_wires_picking_and_generating_to_the_session(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    """
    A mis-wired signal here is invisible: the click does nothing and the rater
    reads it as the harness ignoring them.
    """
    pipeline = _pipeline(sampler, bank, depth=3)
    session = SessionViewModel(pipeline, fake_sink)
    window = MainWindow(session, PlayerViewModel(), quiet_fill=0.25)
    session.advance()

    rows = _rows(window.view.worklist)
    assert rows, "the queue reached the panel"

    queued = [item.pair_id for item in pipeline.worklist() if item.ready]
    assert queued
    window.view.picked.emit(queued[-1])
    assert session.pair is not None
    assert session.pair.spec.pair_id == queued[-1]

    before = pipeline.inflight + pipeline.ready
    window.view.generate.emit(str(Tier.STRUCTURE), 2)
    assert pipeline.inflight + pipeline.ready >= before + 2
