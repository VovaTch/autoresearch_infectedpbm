"""
Widget smoke test.

The view holds no logic, so there is little to assert about it beyond this: it
builds, it renders a pair without leaking either candidate's identity, and its
keys reach the view-model. That last one is worth pinning because a mis-wired
signal would silently make a key do nothing, and the rater would just think the
harness was unresponsive.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from ab_harness.model.bank import ClipBank
from ab_harness.model.pair_sampler import PairSampler
from ab_harness.model.pipeline import PairPipeline
from ab_harness.view.rating_view import RatingView
from ab_harness.viewmodel.session_vm import SessionViewModel
from tests.conftest import FakeProducer, FakeSink


def test_rating_view_renders_a_pair_without_leaking_identity(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    from PySide6.QtWidgets import QLabel, QPushButton

    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=1, structure_live=True
    )
    session = SessionViewModel(pipeline, fake_sink)
    view = RatingView()
    session.pair_changed.connect(view.show_pair)
    assert session.advance()

    pair = session.pair
    assert pair is not None
    # every text-bearing widget, not just the labels, so the check keeps
    # holding as the surface grows
    shown = " ".join(
        widget.text()
        for kind in (QLabel, QPushButton)
        for widget in view.findChildren(kind)
        if widget.text()
    )
    assert pair.spec.question in shown
    for hidden in (
        pair.spec.left.item_id,
        pair.spec.right.item_id,
        pair.spec.left.group_id,
        str(pair.spec.left.sampling.seed),
        pair.spec.left.checkpoint,
    ):
        assert hidden not in shown, f"{hidden!r} leaked into the rating surface"


@pytest.mark.parametrize(
    "key_name,signal_name,expected",
    [
        ("Key_1", "chose", "left"),
        ("Key_2", "chose", "tie"),
        ("Key_3", "chose", "right"),
        ("Key_A", "listen", 0),
        ("Key_D", "listen", 1),
        ("Key_Left", "listen", 0),
        ("Key_Right", "listen", 1),
        ("Key_X", "skipped", None),
        ("Key_S", "flip", None),
        ("Key_R", "restart", None),
        ("Key_Space", "play_pause", None),
    ],
)
def test_keys_reach_the_view_model(
    qapp, key_name: str, signal_name: str, expected
) -> None:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    view = RatingView()
    fired: list = []
    getattr(view, signal_name).connect(lambda *args: fired.append(args))
    key = getattr(Qt.Key, key_name)
    view.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    )
    assert fired, f"{key_name} emitted nothing on {signal_name}"
    if expected is not None:
        assert fired[0][0] == expected


def test_shift_plays_a_side_from_the_top(qapp) -> None:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    view = RatingView()
    plain: list = []
    from_start: list = []
    view.listen.connect(lambda side: plain.append(side))
    view.listen_from_start.connect(lambda side: from_start.append(side))
    view.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_D, Qt.KeyboardModifier.ShiftModifier)
    )
    assert from_start == [1] and plain == []


def _buttons(view: RatingView) -> dict[str, object]:
    """
    Args:
      view (RatingView): the rating surface.

    Returns:
      dict[str, object]: buttons keyed by their label.
    """
    from PySide6.QtWidgets import QPushButton

    return {b.text(): b for b in view.findChildren(QPushButton)}


@pytest.mark.parametrize(
    "label,signal_name,expected",
    [
        ("Listen A", "listen", 0),
        ("Listen B", "listen", 1),
        ("Play", "play_pause", None),
        ("Restart", "restart", None),
        ("A is better", "chose", "left"),
        ("Tie", "chose", "tie"),
        ("B is better", "chose", "right"),
        ("Skip", "skipped", None),
    ],
)
def test_buttons_emit_the_same_intent_as_the_keys(
    qapp, label: str, signal_name: str, expected
) -> None:
    view = RatingView()
    fired: list = []
    getattr(view, signal_name).connect(lambda *args: fired.append(args))
    buttons = _buttons(view)
    assert label in buttons, f"no button labelled {label!r}; have {sorted(buttons)}"
    buttons[label].click()
    assert fired, f"{label!r} emitted nothing on {signal_name}"
    if expected is not None:
        assert fired[0][0] == expected


def test_buttons_do_not_steal_keyboard_focus(qapp) -> None:
    """
    A button that takes focus would send the next keystroke nowhere, and the
    harness would look frozen the moment anyone touched the mouse.
    """
    from PySide6.QtCore import Qt

    view = RatingView()
    for label, button in _buttons(view).items():
        assert button.focusPolicy() == Qt.FocusPolicy.NoFocus, label


def test_keys_still_work_after_a_button_is_clicked(qapp) -> None:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    view = RatingView()
    chosen: list = []
    view.chose.connect(lambda side: chosen.append(side))
    _buttons(view)["Restart"].click()
    view.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier)
    )
    assert chosen == ["left"]


def test_the_play_button_tracks_the_transport(qapp) -> None:
    view = RatingView()
    assert "Play" in _buttons(view)
    view.set_playing(True)
    assert "Pause" in _buttons(view)
    view.set_playing(False)
    assert "Play" in _buttons(view)


def test_checkpoint_selector_switches_the_session(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    from PySide6.QtWidgets import QComboBox

    from ab_harness.view.main_window import MainWindow
    from ab_harness.viewmodel.player_vm import PlayerViewModel

    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=1, structure_live=True
    )
    session = SessionViewModel(pipeline, fake_sink)
    window = MainWindow(
        session,
        PlayerViewModel(),
        checkpoints=["ckpt_test", "saved_dpo_x/dpo_latest.ckpt"],
    )
    combo = window.findChild(QComboBox, "checkpoints")
    assert combo is not None and combo.count() == 2
    # the run directory is what tells two models apart at a glance
    assert combo.itemText(1) == "saved_dpo_x/dpo_latest"

    combo.setCurrentIndex(1)
    combo.activated.emit(1)

    assert session.checkpoint == "saved_dpo_x/dpo_latest.ckpt"
    assert pipeline.producer.switches == ["saved_dpo_x/dpo_latest.ckpt"]  # type: ignore[attr-defined]
    # disabled until the worker answers: a second switch would drop a queue
    # that is already being refilled
    assert not combo.isEnabled()
    session._pump()
    assert combo.isEnabled()


def test_checkpoint_selector_is_hidden_when_there_is_nothing_to_choose(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    from PySide6.QtWidgets import QComboBox

    from ab_harness.view.main_window import MainWindow
    from ab_harness.viewmodel.player_vm import PlayerViewModel

    pipeline = PairPipeline(sampler, FakeProducer(store=bank), bank, depth=1)
    window = MainWindow(SessionViewModel(pipeline, fake_sink), PlayerViewModel())
    combo = window.findChild(QComboBox, "checkpoints")
    assert combo is not None and combo.count() == 0


def test_clicking_a_waveform_moves_the_playhead(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    """
    End of the chain: strip -> panel -> rating view -> player. Both sides share
    one playhead, so a click on either seeks the comparison, not one candidate.
    """
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from ab_harness.view.main_window import MainWindow
    from ab_harness.view.waveform import WaveformView
    from ab_harness.viewmodel.player_vm import PlayerViewModel

    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=1, structure_live=True
    )
    session = SessionViewModel(pipeline, fake_sink)
    player = PlayerViewModel()
    window = MainWindow(session, player)
    assert session.advance()
    player.stop()

    duration = player.source.duration
    assert duration > 0.0
    for strip in window.findChildren(WaveformView):
        strip.resize(200, 64)
        strip.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(100.0, 8.0),
                QPointF(100.0, 8.0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        assert player.source.seconds == pytest.approx(duration / 2, abs=1e-3)
        player.source.restart()


def test_backend_picker_switches_to_that_backends_newest_checkpoint(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    from PySide6.QtWidgets import QComboBox

    from ab_harness.view.main_window import MainWindow
    from ab_harness.viewmodel.player_vm import PlayerViewModel

    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=1, structure_live=True
    )
    session = SessionViewModel(pipeline, fake_sink)
    window = MainWindow(
        session,
        PlayerViewModel(),
        checkpoints={
            "ar": ["ckpt_test", "saved_ar_x/ar_latest.ckpt"],
            "zflow": ["saved_zflow_y/zflow_latest.ckpt"],
            "mdm": [],
        },
    )
    backends = window.findChild(QComboBox, "backends")
    combo = window.findChild(QComboBox, "checkpoints")
    assert backends is not None and combo is not None
    assert [backends.itemText(i) for i in range(backends.count())] == ["ar", "zflow"]
    assert combo.count() == 2 and combo.itemData(0) == "ckpt_test"

    backends.setCurrentIndex(1)
    backends.activated.emit(1)
    assert combo.count() == 1 and combo.itemData(0) == "saved_zflow_y/zflow_latest.ckpt"
    assert session.checkpoint == "saved_zflow_y/zflow_latest.ckpt"
    session._pump()
    assert combo.isEnabled()

    # a switch arriving from elsewhere re-homes both selectors
    window._on_checkpoint_changed("saved_ar_x/ar_latest.ckpt", "")
    assert (
        backends.currentData() == "ar"
        and combo.currentData() == "saved_ar_x/ar_latest.ckpt"
    )


def test_a_finished_dpo_round_adds_its_checkpoints_without_switching(
    qapp, sampler: PairSampler, bank: ClipBank, fake_sink: FakeSink
) -> None:
    from PySide6.QtWidgets import QComboBox

    from ab_harness.view.main_window import MainWindow
    from ab_harness.viewmodel.player_vm import PlayerViewModel

    pipeline = PairPipeline(
        sampler, FakeProducer(store=bank), bank, depth=1, structure_live=True
    )
    session = SessionViewModel(pipeline, fake_sink)
    after = {
        "ar": ["saved_dpo_r01/dpo_latest.ckpt", "ckpt_test"],
        "zflow": ["saved_zflow_dpo_r01/dpo_latest.ckpt"],
        "mdm": ["saved_mdm_dpo_r01/dpo_latest.ckpt"],
    }
    window = MainWindow(
        session,
        PlayerViewModel(),
        checkpoints={"ar": ["ckpt_test"], "zflow": [], "mdm": []},
        discover=lambda current: after,
    )
    backends = window.findChild(QComboBox, "backends")
    combo = window.findChild(QComboBox, "checkpoints")
    assert backends is not None and combo is not None
    assert backends.count() == 1

    session.round_finished.emit("DPO r1 (331 pairs): ar val acc 0.55")
    assert [backends.itemText(i) for i in range(backends.count())] == [
        "ar",
        "zflow",
        "mdm",
    ]
    assert backends.isVisibleTo(window)
    assert combo.count() == 2
    assert combo.currentData() == "ckpt_test"
    assert session.checkpoint == "ckpt_test"
