"""
Preference training that keeps up with the rater.

Every `every_pairs` new usable pairs, a DPO round trains every backend (AR,
zflow, MDM) on the whole bank, each from its fixed base checkpoint, one after
another on a spare GPU (dpo_round.py). "Usable" is the exact filter DPO trains
on (ab_harness.model.preferences), so ties, anchors and repeats never count
toward the trigger.

The round runs detached (its own session) so closing the app, or the app
crashing, does not kill a training run halfway. Its progress lives in one JSON
file per round under <bank>/dpo_rounds/, written by the driver and read here;
the UI and the driver share nothing else.

Torch-free: imported by the UI process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from ab_harness.model.bank import ClipBank
from ab_harness.model.preferences import PairsCfg, load_preference_pairs

REPO = Path(__file__).resolve().parents[2]
TERMINAL = ("done", "failed")
BASELINE = "baseline.json"


@dataclass
class AutoTrainCfg:
    """
    When and where automatic DPO rounds run.

    Args:
      enabled (bool): launch rounds at all.
      every_pairs (int): new usable pairs between rounds.
      device (str): torch device the round trains on; keep it off the worker's.
      configs (list[str]): train_dpo.py configs, trained in this order.
      poll_s (float): seconds between reads of the round status file.
    """

    enabled: bool = False
    every_pairs: int = 100
    device: str = "cuda:1"
    configs: list[str] = field(
        default_factory=lambda: [
            "config_dpo_ar.yaml",
            "config_dpo_zflow.yaml",
            "config_dpo_mdm.yaml",
        ]
    )
    poll_s: float = 5.0


def backend_label(config: str) -> str:
    """
    Args:
      config (str): a train_dpo.py config path, e.g. config_dpo_zflow.yaml.

    Returns:
      str: the short name shown in the status bar, e.g. "zflow".
    """
    return Path(config).stem.removeprefix("config_dpo_").removeprefix("config_")


def process_alive(pid: int) -> bool:
    """
    Args:
      pid (int): process id, 0 for none.

    Returns:
      bool: True if the process runs; a zombie (exited, not yet reaped by the
        app that spawned it) counts as dead.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return stat.read_text().rsplit(")", 1)[-1].split()[0] != "Z"
    return True


@dataclass
class BackendRun:
    """
    One backend's training inside a round.

    Args:
      config (str): its train_dpo.py config.
      state (str): pending, running, done or failed.
      save_dir (str): repo-relative checkpoint directory.
      metrics (dict[str, float]): train_dpo.run_metrics output once done.
    """

    config: str
    state: str = "pending"
    save_dir: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """
        Returns:
          str: backend short name.
        """
        return backend_label(self.config)


@dataclass
class RoundStatus:
    """
    Progress of one DPO round, persisted as <bank>/dpo_rounds/rNN.json.

    Args:
      round (int): round number, from 1.
      pairs (int): usable pairs when it was launched.
      pid (int): the process driving it.
      started (str): launch time, ISO-8601.
      runs (list[BackendRun]): one per backend, in training order.
    """

    round: int
    pairs: int
    pid: int = 0
    started: str = ""
    runs: list[BackendRun] = field(default_factory=list)

    @staticmethod
    def path(rounds_dir: Path, number: int) -> Path:
        """
        Args:
          rounds_dir (Path): <bank>/dpo_rounds.
          number (int): round number.

        Returns:
          Path: the round's status file.
        """
        return rounds_dir / f"r{number:02d}.json"

    @classmethod
    def load(cls, path: Path) -> RoundStatus:
        """
        Args:
          path (Path): a status file.

        Returns:
          RoundStatus: its contents.
        """
        raw = json.loads(path.read_text())
        raw["runs"] = [BackendRun(**run) for run in raw.get("runs", [])]
        return cls(**raw)

    def save(self, rounds_dir: Path) -> None:
        """
        Write atomically, so a reader never sees half a file.

        Args:
          rounds_dir (Path): <bank>/dpo_rounds.
        """
        path = self.path(rounds_dir, self.round)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    @property
    def finished(self) -> bool:
        """
        Returns:
          bool: every backend reached a terminal state, or the driver died
            (crash, power loss) and nothing will move any more.
        """
        settled = all(run.state in TERMINAL for run in self.runs)
        return settled or not process_alive(self.pid)

    def progress(self) -> str:
        """
        Returns:
          str: status-bar text for a round in flight.
        """
        active = [i for i, run in enumerate(self.runs) if run.state not in TERMINAL]
        if not active:
            return f"DPO r{self.round}: finishing"
        index = active[0]
        return (
            f"DPO r{self.round}: {self.runs[index].label} "
            f"{index + 1}/{len(self.runs)}"
        )

    def summary(self) -> str:
        """
        Returns:
          str: one line per round: each backend's held-out accuracy or fate.
        """
        parts = []
        for run in self.runs:
            acc = run.metrics.get("final_val_acc")
            if run.state == "done" and acc is not None:
                parts.append(f"{run.label} val acc {acc:.2f}")
            elif run.state == "done":
                parts.append(f"{run.label} done")
            else:
                parts.append(
                    f"{run.label} {'failed' if run.state == 'failed' else 'died'}"
                )
        return f"DPO r{self.round} ({self.pairs} pairs): " + ", ".join(parts)


@dataclass
class TrainEvent:
    """
    What the status bar should show after a poll.

    Args:
      status (str): persistent status text.
      finished (RoundStatus | None): a round that finished since the last
        poll, to be announced once.
    """

    status: str
    finished: RoundStatus | None = None


class RoundLauncher(Protocol):
    """Starts a round's driver and reports its pid."""

    def launch(self, number: int, pairs: int) -> int:
        """
        Args:
          number (int): round number.
          pairs (int): usable pairs at launch.

        Returns:
          int: pid of the driver process.
        """
        ...


class DetachedLauncher:
    """
    Run dpo_round.py in its own session, logging to logs/dpo_rNN.log.

    Args:
      bank_root (Path): the bank being rated.
      cfg (AutoTrainCfg): device and configs.
      repo (Path): repository root.
    """

    def __init__(self, bank_root: Path, cfg: AutoTrainCfg, repo: Path = REPO) -> None:
        self.bank_root = bank_root
        self.cfg = cfg
        self.repo = repo

    def launch(self, number: int, pairs: int) -> int:
        """
        Args:
          number (int): round number.
          pairs (int): usable pairs at launch.

        Returns:
          int: pid of the driver.
        """
        logs = self.repo / "logs"
        logs.mkdir(exist_ok=True)
        command = [
            sys.executable,
            str(self.repo / "dpo_round.py"),
            "--bank-root",
            str(self.bank_root),
            "--round",
            str(number),
            "--device",
            self.cfg.device,
            "--configs",
            *self.cfg.configs,
        ]
        with (logs / f"dpo_r{number:02d}.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=self.repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        return process.pid


class AutoTrainer:
    """
    Counts usable pairs and launches a DPO round every `every_pairs` of them.

    On a bank with no round history the current count becomes the baseline, so
    enabling this on a bank that was already trained on does not fire at once.

    Args:
      bank (ClipBank): the bank being rated.
      cfg (AutoTrainCfg): trigger settings.
      launcher (RoundLauncher): starts the driver.
      clock (Callable[[], float]): monotonic seconds, injectable for tests.
    """

    def __init__(
        self,
        bank: ClipBank,
        cfg: AutoTrainCfg,
        launcher: RoundLauncher,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bank = bank
        self.cfg = cfg
        self.launcher = launcher
        self.clock = clock
        self.rounds_dir = bank.root / "dpo_rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self._usable = self._count()
        baseline = self.rounds_dir / BASELINE
        if not baseline.exists():
            baseline.write_text(json.dumps({"pairs": self._usable}))
        self._baseline = int(json.loads(baseline.read_text())["pairs"])
        latest = self.latest()
        # Rounds already over at startup were announced by an earlier session.
        self._announced = latest.round if latest and latest.finished else 0
        self._last_poll = -float("inf")

    def _count(self) -> int:
        """
        Returns:
          int: usable pairs in the bank right now.
        """
        return len(load_preference_pairs(self.bank, PairsCfg())[0])

    def latest(self) -> RoundStatus | None:
        """
        Returns:
          RoundStatus | None: the highest-numbered round, None before the first.
        """
        paths = sorted(self.rounds_dir.glob("r[0-9]*.json"))
        return RoundStatus.load(paths[-1]) if paths else None

    @property
    def new_pairs(self) -> int:
        """
        Returns:
          int: usable pairs added since the last launch (or the baseline).
        """
        latest = self.latest()
        return self._usable - (latest.pairs if latest else self._baseline)

    def on_rated(self) -> RoundStatus | None:
        """
        Recount after a judgement and launch a round if one is due.

        Returns:
          RoundStatus | None: the round just launched, if any.
        """
        self._usable = self._count()
        latest = self.latest()
        if latest is not None and not latest.finished:
            return None
        if not self.cfg.enabled or self.new_pairs < self.cfg.every_pairs:
            return None
        number = (latest.round if latest else 0) + 1
        status = RoundStatus(
            round=number,
            pairs=self._usable,
            started=datetime.now().isoformat(timespec="seconds"),
            runs=[BackendRun(config) for config in self.cfg.configs],
        )
        # Written before launch: the driver adopts it, and a second rating
        # landing before the driver starts sees a round already in flight.
        status.pid = os.getpid()
        status.save(self.rounds_dir)
        status.pid = self.launcher.launch(number, self._usable)
        status.save(self.rounds_dir)
        self._last_poll = -float("inf")
        return status

    def poll(self) -> TrainEvent | None:
        """
        Returns:
          TrainEvent | None: current status, None while throttled.
        """
        now = self.clock()
        if now - self._last_poll < self.cfg.poll_s:
            return None
        self._last_poll = now
        latest = self.latest()
        finished = None
        if latest is not None and latest.finished and latest.round > self._announced:
            self._announced = latest.round
            finished = latest
        if latest is not None and not latest.finished:
            return TrainEvent(latest.progress())
        if not self.cfg.enabled:
            return TrainEvent("", finished)
        return TrainEvent(
            f"DPO {max(0, self.new_pairs)}/{self.cfg.every_pairs} new pairs", finished
        )
