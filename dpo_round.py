"""
One automatic DPO round: train every backend on the whole A/B bank, in turn.

Launched by the A/B harness (ab_harness.model.auto_train) every N new usable
pairs, detached from it. Each backend is its own train_dpo.py process, so VRAM
is returned between them and one backend failing does not stop the others.
Each starts from its config's fixed reference checkpoint -- rounds never chain
-- and saves to <config save_path>_rNN/, which keeps the checkpoint family
(saved_dpo_*, saved_zflow_*, saved_mdm_*) the harness selector sorts by.

Progress goes to <bank>/dpo_rounds/rNN.json after every transition; logs to
logs/dpo_rNN_<backend>.log.

    uv run python dpo_round.py --bank-root ~/.cache/infected_pbm/ab_dsteps2_24h \
      --round 1 --device cuda:1 \
      --configs config_dpo_ar.yaml config_dpo_zflow.yaml config_dpo_mdm.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

from ab_harness.model.auto_train import BackendRun, RoundStatus

REPO = Path(__file__).resolve().parent


def round_save_path(config: str, number: int) -> str:
    """
    Args:
      config (str): a train_dpo.py config.
      number (int): round number.

    Returns:
      str: repo-relative save directory, the config's save_path with the date
        expanded and _rNN appended, e.g. saved_zflow_dpo_20261001_r03/.
    """
    raw = yaml.safe_load((REPO / config).read_text())
    base = raw["train"]["save_path"].replace("<date>", date.today().strftime("%Y%m%d"))
    return f"{base.rstrip('/')}_r{number:02d}/"


def train_backend(
    run: BackendRun,
    number: int,
    bank_root: str,
    device: str,
    extra: list[str],
) -> None:
    """
    Train one backend and record its outcome on `run`.

    Args:
      run (BackendRun): updated in place with save_dir, state and metrics.
      number (int): round number.
      bank_root (str): the A/B bank.
      device (str): torch device.
      extra (list[str]): further train_dpo.py arguments (e.g. --smoke).
    """
    command = [
        sys.executable,
        str(REPO / "train_dpo.py"),
        "--config",
        run.config,
        "--device",
        device,
        "--bank-root",
        bank_root,
        "--save-path",
        run.save_dir,
        *extra,
    ]
    log = REPO / "logs" / f"dpo_r{number:02d}_{run.label}.log"
    with log.open("w") as handle:
        code = subprocess.call(
            command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT
        )
    summary = REPO / run.save_dir / "pairs.json"
    if summary.exists():
        run.metrics = json.loads(summary.read_text()).get("metrics", {})
    ckpt = REPO / run.save_dir / "dpo_latest.ckpt"
    run.state = "done" if code == 0 and ckpt.exists() else "failed"


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument(
        "--extra",
        default="",
        help='arguments forwarded to train_dpo.py, as --extra="--smoke ..."',
    )
    return parser.parse_args()


def main() -> None:
    """Run the round, updating its status file after every transition."""
    args = parse_args()
    rounds_dir = Path(args.bank_root).expanduser() / "dpo_rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)
    path = RoundStatus.path(rounds_dir, args.round)
    status = (
        RoundStatus.load(path)
        if path.exists()
        else RoundStatus(
            round=args.round,
            pairs=0,
            started=datetime.now().isoformat(timespec="seconds"),
            runs=[BackendRun(config) for config in args.configs],
        )
    )
    status.pid = os.getpid()
    status.save(rounds_dir)
    extra = shlex.split(args.extra)
    for run in status.runs:
        run.save_dir = round_save_path(run.config, args.round)
        run.state = "running"
        status.save(rounds_dir)
        print(f"round {args.round}: {run.label} -> {run.save_dir}", flush=True)
        try:
            train_backend(run, args.round, args.bank_root, args.device, extra)
        except Exception as error:  # one backend's crash must not end the round
            print(f"{run.label} crashed: {error!r}", flush=True)
            run.state = "failed"
        status.save(rounds_dir)
        print(f"round {args.round}: {run.label} {run.state} {run.metrics}", flush=True)
    print(status.summary(), flush=True)


if __name__ == "__main__":
    main()
