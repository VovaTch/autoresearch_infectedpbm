"""
Worker messages, without a worker.

These import ab_harness.worker.protocol only, which is why the module keeps its
dependency on numpy alone: bringing torch or onnxruntime in here would make the
suite slow and would tie message shape to a GPU being present.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from ab_harness.model.types import ClipSpec, Conditioning, Sampling, Tier
from ab_harness.worker.protocol import ClipRequest, ClipResult, Shutdown


def _spec() -> ClipSpec:
    """
    Returns:
      ClipSpec: a minimal spec.
    """
    return ClipSpec(
        item_id="gen_abc",
        tier=Tier.BULK,
        group_id="grp_abc",
        n_frames=16,
        conditioning=Conditioning(track_idx=1, start_frame=32, prompt_frames=4),
        sampling=Sampling(seed=7),
        checkpoint="ckpt",
    )


def test_the_ui_import_path_stays_free_of_torch() -> None:
    """
    The process split only buys anything if the UI half really is torch-free.

    Checked in a subprocess because this suite imports torch elsewhere, so
    sys.modules in-process says nothing. A regression here would not break any
    behaviour -- it would just put a multi-second import and a CUDA context in
    front of every window opening, which is the kind of thing that goes
    unnoticed until the harness feels bad to use.
    """
    probe = (
        "import ab_harness.app, sys; "
        "print(int(any(m in sys.modules for m in ('torch','lightning','onnxruntime'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0", "a GPU stack reached the UI process"


def test_request_survives_pickling() -> None:
    request = pickle.loads(pickle.dumps(ClipRequest(_spec())))
    assert request.spec == _spec()


def test_result_survives_pickling_with_its_arrays_intact() -> None:
    tokens = np.random.randint(0, 2048, (16, 3)).astype(np.int16)
    pcm = np.random.randint(-3000, 3000, 4096).astype(np.int16)
    result = pickle.loads(pickle.dumps(ClipResult(_spec(), tokens, pcm, 44100, True)))
    assert result.ok
    assert np.array_equal(result.tokens, tokens)
    assert np.array_equal(result.pcm, pcm)
    assert result.pcm.dtype == np.int16
    assert result.sample_rate == 44100
    assert result.was_live is True


def test_an_error_result_is_not_ok() -> None:
    result = ClipResult(_spec(), error="RuntimeError: CUDA out of memory")
    assert not result.ok
    assert result.tokens is None and result.pcm is None


def test_a_result_missing_audio_is_not_ok() -> None:
    assert not ClipResult(_spec(), tokens=np.zeros((4, 3), dtype=np.int16)).ok


def test_shutdown_is_picklable() -> None:
    assert isinstance(pickle.loads(pickle.dumps(Shutdown())), Shutdown)
