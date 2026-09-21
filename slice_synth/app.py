"""
Entry point for the slice synthesizer.

Usage:
  uv run python -m slice_synth.app
  uv run python -m slice_synth.app --config config_synth.yaml
  uv run python -m slice_synth.app --checkpoint saved_ar_20260907_16k_cont/ar_latest.ckpt

The UI process deliberately never imports torch: the model lives in a child
process (slice_synth.worker.client). The window comes up immediately with its
controls disabled and enables them when the worker publishes the corpus, which
takes a few seconds -- rather than staring at nothing while a checkpoint loads.
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from ab_harness.viewmodel.player_vm import PlayerViewModel
from slice_synth.config import resolve_config
from slice_synth.view.main_window import MainWindow
from slice_synth.viewmodel.synth_vm import SynthViewModel
from slice_synth.worker.client import ProcessRenderProducer


def parse_args() -> argparse.Namespace:
    """
    Returns:
      argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_synth.yaml")
    parser.add_argument("--checkpoint", default="", help="override the config's pick")
    parser.add_argument("--out", default="", help="override the save directory")
    return parser.parse_args()


def main() -> int:
    """
    Returns:
      int: process exit code.
    """
    args = parse_args()
    cfg = resolve_config(args.config)
    if args.checkpoint:
        cfg.generator.checkpoint = args.checkpoint
    if args.out:
        cfg.output.root = args.out

    producer = ProcessRenderProducer(cfg)
    producer.start()

    app = QApplication(sys.argv)
    # The sample rate is the tokenizer's and does not change with the model, so
    # the transport can be built before the worker has finished loading.
    player = PlayerViewModel(44100, cfg.ui.crossfade_ms)
    vm = SynthViewModel(producer, cfg.output_root)
    window = MainWindow(vm, player, cfg.ui, cfg.checkpoints_by_backend)
    window.show()
    vm.start()

    code = app.exec()
    vm.stop()
    producer.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
