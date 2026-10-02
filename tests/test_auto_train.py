"""
Automatic DPO rounds: when they fire, and how their progress is read back.

No training runs here. The launcher is a fake that records calls and returns a
pid, and round status files are written by hand, so the trigger and the status
plumbing are pinned without a GPU.
"""

from __future__ import annotations

import os
import subprocess
import time

import numpy as np

from ab_harness.model.auto_train import (
    AutoTrainCfg,
    AutoTrainer,
    BackendRun,
    RoundStatus,
    TrainEvent,
    process_alive,
)
from ab_harness.model.bank import ClipBank
from ab_harness.model.pair_sampler import PairSampler
from ab_harness.model.pipeline import PairPipeline
from ab_harness.viewmodel.session_vm import SessionViewModel
from dpo_round import round_save_path
from tests.conftest import FakeProducer, FakeSink
from tests.test_dpo import DEPTH, NUM_TOKENS, _judgement, _log, _spec

CONFIGS = ["config_dpo_ar.yaml", "config_dpo_zflow.yaml", "config_dpo_mdm.yaml"]


class FakeLauncher:
    """
    Records launches instead of starting a driver.

    Args:
      pid (int): returned as the driver's pid.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.calls: list[tuple[int, int]] = []

    def launch(self, number: int, pairs: int) -> int:
        """
        Args:
          number (int): round number.
          pairs (int): usable pairs at launch.

        Returns:
          int: the configured pid.
        """
        self.calls.append((number, pairs))
        return self.pid


class Clock:
    """Hand-advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        """
        Returns:
          float: current time in seconds.
        """
        return self.now


def dead_pid() -> int:
    """
    Returns:
      int: the pid of a process that has exited and been reaped.
    """
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def add_pairs(bank: ClipBank, count: int, start: int = 0, choice: str = "left") -> None:
    """
    Bank `count` fresh same-group pairs and one judgement for each.

    Args:
      bank (ClipBank): destination.
      count (int): pairs to add.
      start (int): index of the first, so calls do not collide.
      choice (str): the verdict on every pair.
    """
    rng = np.random.default_rng(start)
    rows = []
    for i in range(start, start + count):
        left, right = _spec(f"a{i}", f"g{i}"), _spec(f"b{i}", f"g{i}")
        for spec in (left, right):
            bank.add(spec, rng.integers(0, NUM_TOKENS, (spec.n_frames, DEPTH)))
        ts = f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}"
        rows.append(_judgement(f"p{i}", "s1", left.item_id, right.item_id, choice, ts))
    _log(bank, "s1", rows)


def trainer(
    bank: ClipBank, launcher: FakeLauncher, every: int = 3, clock: Clock | None = None
) -> AutoTrainer:
    """
    Args:
      bank (ClipBank): the bank.
      launcher (FakeLauncher): fake driver starter.
      every (int): pairs between rounds.
      clock (Clock | None): poll clock.

    Returns:
      AutoTrainer: an enabled trainer.
    """
    cfg = AutoTrainCfg(enabled=True, every_pairs=every, configs=CONFIGS, poll_s=5.0)
    return AutoTrainer(bank, cfg, launcher, clock=clock or Clock())


def test_existing_pairs_become_the_baseline_and_do_not_fire(bank: ClipBank) -> None:
    add_pairs(bank, 5)
    launcher = FakeLauncher(os.getpid())
    auto = trainer(bank, launcher)
    assert auto.on_rated() is None
    assert auto.new_pairs == 0
    assert launcher.calls == []


def test_a_round_fires_after_every_pairs_new_usable_pairs(bank: ClipBank) -> None:
    auto = trainer(bank, launcher := FakeLauncher(os.getpid()))
    add_pairs(bank, 2)
    assert auto.on_rated() is None
    add_pairs(bank, 1, start=2)
    launched = auto.on_rated()
    assert launched is not None and launched.round == 1
    assert launcher.calls == [(1, 3)]
    saved = RoundStatus.load(RoundStatus.path(auto.rounds_dir, 1))
    assert saved.pid == os.getpid()
    assert [run.label for run in saved.runs] == ["ar", "zflow", "mdm"]


def test_ties_do_not_count_toward_the_trigger(bank: ClipBank) -> None:
    auto = trainer(bank, launcher := FakeLauncher(os.getpid()))
    add_pairs(bank, 5, choice="tie")
    assert auto.on_rated() is None
    assert auto.new_pairs == 0
    assert launcher.calls == []


def test_no_second_round_while_one_is_running(bank: ClipBank) -> None:
    auto = trainer(bank, launcher := FakeLauncher(os.getpid()))
    add_pairs(bank, 3)
    auto.on_rated()
    add_pairs(bank, 3, start=3)
    assert auto.on_rated() is None
    assert len(launcher.calls) == 1


def test_a_dead_driver_frees_the_trigger_for_the_next_round(bank: ClipBank) -> None:
    auto = trainer(bank, launcher := FakeLauncher(dead_pid()))
    add_pairs(bank, 3)
    auto.on_rated()
    add_pairs(bank, 3, start=3)
    launched = auto.on_rated()
    assert launched is not None and launched.round == 2
    assert launcher.calls == [(1, 3), (2, 6)]


def test_poll_reports_progress_then_announces_the_finish_once(bank: ClipBank) -> None:
    clock = Clock()
    auto = trainer(bank, FakeLauncher(os.getpid()), clock=clock)
    status = RoundStatus(
        round=1,
        pairs=3,
        pid=os.getpid(),
        runs=[
            BackendRun(CONFIGS[0], "done", metrics={"final_val_acc": 0.55}),
            BackendRun(CONFIGS[1], "running"),
            BackendRun(CONFIGS[2]),
        ],
    )
    status.save(auto.rounds_dir)
    event = auto.poll()
    assert event is not None and event.status == "DPO r1: zflow 2/3"
    assert auto.poll() is None  # throttled

    status.runs[1].state, status.runs[2].state = "done", "failed"
    status.save(auto.rounds_dir)
    clock.now += 10
    event = auto.poll()
    assert event is not None and event.finished is not None
    assert event.finished.summary() == (
        "DPO r1 (3 pairs): ar val acc 0.55, zflow done, mdm failed"
    )
    clock.now += 10
    again = auto.poll()
    assert again is not None and again.finished is None


def test_zombies_count_as_dead() -> None:
    process = subprocess.Popen(["true"])  # never polled, so it stays a zombie
    try:
        deadline = time.monotonic() + 5.0
        while process_alive(process.pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not process_alive(process.pid)
    finally:
        process.wait()
    assert process_alive(os.getpid())


def test_round_save_dirs_keep_the_backend_family() -> None:
    assert round_save_path("config_dpo_zflow.yaml", 3).startswith("saved_zflow_dpo_")
    assert round_save_path("config_dpo_zflow.yaml", 3).endswith("_r03/")
    assert round_save_path("config_dpo_ar.yaml", 1).startswith("saved_dpo_")
    assert round_save_path("config_dpo_mdm.yaml", 1).startswith("saved_mdm_")


class FakeTrigger:
    """Launches on the first rating, then reports a finished round."""

    def __init__(self) -> None:
        self.rated = 0
        self.status = RoundStatus(round=1, pairs=3, pid=os.getpid(), runs=[])

    def on_rated(self) -> RoundStatus | None:
        """
        Returns:
          RoundStatus | None: the round on the first call only.
        """
        self.rated += 1
        return self.status if self.rated == 1 else None

    def poll(self) -> TrainEvent:
        """
        Returns:
          TrainEvent: a finished round.
        """
        return TrainEvent("DPO 0/3 new pairs", self.status)


def test_the_session_tells_the_trigger_and_relays_its_events(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=2, structure_live=True
    )
    trigger = FakeTrigger()
    session = SessionViewModel(pipeline, fake_sink, trainer=trigger)
    statuses: list[str] = []
    finished: list[str] = []
    session.training_status.connect(statuses.append)
    session.round_finished.connect(finished.append)
    session.advance()
    session.choose("left")
    assert trigger.rated == 1
    assert statuses == ["DPO r1: finishing"]
    session._pump()
    assert statuses[-1] == "DPO 0/3 new pairs"
    assert finished == ["DPO r1 (3 pairs): "]
