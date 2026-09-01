"""
Shared fixtures.

Every test here runs on the CPU in well under a second: no checkpoints, no GPU,
no audio device, no subprocess. The Qt tests use the offscreen platform and a
bare QCoreApplication, since nothing under test is a widget.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ab_harness.model.bank import ClipBank
from ab_harness.model.pair_sampler import PairSampler, SamplerCfg, TrackInfo
from ab_harness.model.types import Clip, ClipSpec, Judgement
from ab_harness.worker.protocol import CheckpointChanged

FPS = 172.265625
WINDOW = int(10.0 * FPS)


@pytest.fixture(scope="session")
def qapp():
    """
    Returns:
      QApplication: one application object for the whole session; Qt refuses to
        create a second. A QApplication rather than a QCoreApplication so the
        widget tests can share it -- it is a superset, and under the offscreen
        platform it needs no display.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def tracks() -> list[TrackInfo]:
    """
    Returns:
      list[TrackInfo]: five synthetic tracks long enough for the structure tier.
    """
    return [
        TrackInfo(
            track_idx=i,
            track_name=f"track_{i}",
            num_frames=40_000,
            style_bounds=tuple((w * WINDOW, (w + 1) * WINDOW) for w in range(23)),
        )
        for i in range(5)
    ]


@pytest.fixture
def sampler(tracks: list[TrackInfo]) -> PairSampler:
    """
    Args:
      tracks (list[TrackInfo]): synthetic corpus.

    Returns:
      PairSampler: a seeded sampler with short clips, so tests stay fast.
    """
    cfg = SamplerCfg(fps=FPS, bulk_seconds=1.0, structure_seconds=4.0)
    return PairSampler(tracks, cfg, checkpoint="ckpt_test", rng=random.Random(1234))


@pytest.fixture
def bank(tmp_path: Path) -> ClipBank:
    """
    Args:
      tmp_path (Path): pytest temp dir.

    Returns:
      ClipBank: an empty initialised bank.
    """
    store = ClipBank(tmp_path / "ab")
    store.init_manifest("tokens_test", {"num_rq": 3, "sample_rate": 44100}, "ckpt_test")
    return store


class FakeProducer:
    """
    ClipProducer that fabricates silence-free noise instead of sampling.

    Args:
      fail (set[str] | None): item ids to never return, simulating a worker-side
        failure.
      store (ClipBank | None): bank to record produced tokens into, so
        pipeline tests see the same reuse behaviour as the real service.
      checkpoint (str): the model it claims to be serving.
      fail_switch (str): error text to answer every switch with; empty means
        switches succeed.
    """

    def __init__(
        self,
        fail: set[str] | None = None,
        store: ClipBank | None = None,
        checkpoint: str = "ckpt_test",
        fail_switch: str = "",
    ) -> None:
        self.fail = fail or set()
        self.store = store
        self.checkpoint = checkpoint
        self.fail_switch = fail_switch
        self.requested: list[str] = []
        self.switches: list[str] = []
        self._queue: list[Clip] = []
        self._change: object | None = None
        self.closed = False

    def switch_checkpoint(self, checkpoint: str) -> None:
        """
        Args:
          checkpoint (str): the model to serve next. Applied immediately, the
            way the in-process producer does.
        """
        self.switches.append(checkpoint)
        if self.fail_switch:
            # A worker that cannot load keeps the model it had.
            self._change = CheckpointChanged(
                checkpoint=self.checkpoint, error=self.fail_switch
            )
            return
        self.checkpoint = checkpoint
        self._change = CheckpointChanged(checkpoint=checkpoint)

    def take_checkpoint_change(self) -> object | None:
        """
        Returns:
          object | None: the outcome of a switch, once.
        """
        change, self._change = self._change, None
        return change

    def submit(self, specs: Sequence[ClipSpec]) -> None:
        rng = np.random.default_rng(0)
        for spec in specs:
            self.requested.append(spec.item_id)
            if spec.item_id in self.fail:
                continue
            tokens = rng.integers(0, 2048, size=(spec.n_frames, 3)).astype(np.int16)
            if self.store is not None:
                self.store.add(spec, tokens)
            pcm = (rng.normal(0, 3000, size=spec.n_frames * 8)).astype(np.int16)
            self._queue.append(Clip(spec=spec, tokens=tokens, pcm=pcm, was_live=True))

    def poll(self, timeout: float = 0.0) -> list[Clip]:
        done, self._queue = self._queue, []
        return done

    def close(self) -> None:
        self.closed = True


class FakeSink:
    """JudgementSink that keeps everything in memory."""

    def __init__(self) -> None:
        self.rows: list[Judgement] = []

    def append(self, judgement: Judgement) -> None:
        self.rows.append(judgement)


@pytest.fixture
def fake_sink() -> FakeSink:
    """
    Returns:
      FakeSink: an in-memory judgement sink.
    """
    return FakeSink()
