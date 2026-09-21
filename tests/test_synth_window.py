"""
The synthesizer UI, offscreen, with a fake worker.

Nothing here touches a GPU, a checkpoint or an audio device. What is being
checked is the wiring: that the panel expands a selection into the batch it
claims it will, that results land on the row that asked for them, that the three
strips share a playhead, and that saving writes the triple.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QListWidgetItem

from ab_harness.model.pair_sampler import TrackInfo
from slice_synth.config import UiCfg
from slice_synth.model.types import SAME_TRACK, RenderSpec, StyleSpec
from slice_synth.view.controls import FOLLOW, ControlPanel
from slice_synth.view.results import ResultsPanel
from slice_synth.viewmodel.synth_vm import SynthViewModel, Variant
from slice_synth.worker.protocol import RenderProgress, RenderResult, WorkerReady

FPS = 172.265625
META = {
    "frames_per_second": FPS,
    "num_tokens": 2048,
    "num_rq": 3,
    "sample_rate": 44100,
    "hop_length": 256,
}


def tracks(count: int = 4) -> list[TrackInfo]:
    """
    Args:
      count (int): how many synthetic tracks.

    Returns:
      list[TrackInfo]: a corpus long enough to prompt from anywhere.
    """
    return [
        TrackInfo(
            track_idx=i,
            track_name=f"track {i}",
            num_frames=40_000,
            style_bounds=tuple((w * 1723, (w + 1) * 1723) for w in range(23)),
        )
        for i in range(count)
    ]


class FakeProducer:
    """
    RenderProducer that fabricates noise instead of sampling.

    Args:
      fail (set[str] | None): item ids to answer with an error.
    """

    def __init__(self, fail: set[str] | None = None, defer: bool = False) -> None:
        self.fail = fail or set()
        self.defer = defer
        self.checkpoint = "ckpt_test"
        self.submitted: list[RenderSpec] = []
        self.switches: list[str] = []
        self.cancels = 0
        self.closed = False
        self._out: list[Any] = []

    def ready(self) -> None:
        self._out.append(WorkerReady("ckpt_test", tracks(), dict(META), 4096))

    def submit(self, specs: Sequence[RenderSpec]) -> None:
        self.submitted += list(specs)
        if self.defer:
            return
        self.produce(specs)

    def produce(self, specs: Sequence[RenderSpec]) -> None:
        rng = np.random.default_rng(0)
        self._out.append(RenderProgress("b", 256, 512))
        for spec in specs:
            if spec.item_id in self.fail:
                self._out.append(RenderResult(spec=spec, error="fabricated failure"))
                continue
            self._out.append(
                RenderResult(
                    spec=spec,
                    tokens=rng.integers(0, 2048, size=(spec.n_frames, 3)).astype(
                        np.int16
                    ),
                    pcm=(rng.normal(0, 3000, spec.n_frames * 256)).astype(np.int16),
                    sample_rate=44100,
                    style_used=np.zeros(4, dtype=np.float32),
                    fill=1.0,
                )
            )

    def poll(self, timeout: float = 0.0) -> list[Any]:
        out, self._out = self._out, []
        return out

    def cancel(self) -> None:
        self.cancels += 1

    def switch_checkpoint(self, checkpoint: str) -> None:
        self.switches.append(checkpoint)
        self._out.append(WorkerReady(checkpoint, tracks(), dict(META), 4096))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def panel(qapp) -> ControlPanel:
    """
    Returns:
      ControlPanel: a panel with a synthetic corpus loaded.
    """
    widget = ControlPanel(UiCfg(), ["a/one.ckpt", "b/two.ckpt"])
    widget.set_corpus(WorkerReady("a/one.ckpt", tracks(), dict(META), 4096))
    return widget


@pytest.fixture
def vm(qapp, tmp_path: Path) -> SynthViewModel:
    """
    Returns:
      SynthViewModel: a view-model over a fake producer, already loaded.
    """
    producer = FakeProducer()
    producer.ready()
    model = SynthViewModel(producer, tmp_path)
    model._pump()
    return model


# -- control panel ---------------------------------------------------------


def check(panel: ControlPanel, *track_ids: int) -> None:
    """
    Args:
      panel (ControlPanel): the panel to edit.
      track_ids (int): track ids to tick, everything else unticked.
    """
    for row in range(panel.track_list.count()):
        item = panel.track_list.item(row)
        wanted = item.data(Qt.ItemDataRole.UserRole) in track_ids
        item.setCheckState(Qt.CheckState.Checked if wanted else Qt.CheckState.Unchecked)


def test_panel_opens_with_one_track_and_one_style(panel: ControlPanel) -> None:
    assert panel.selected_tracks() == [0]
    assert len(panel.styles.styles()) == 1
    assert len(panel.build_specs()) == 1


def test_panel_expands_the_cross_product(panel: ControlPanel) -> None:
    check(panel, 1, 2)
    panel.styles.add(StyleSpec(kind="random", seed=5))
    panel.seeds.setValue(2)
    specs = panel.build_specs()
    assert len(specs) == 8
    assert {s.track_idx for s in specs} == {1, 2}
    assert {s.seed for s in specs} == {0, 1}


def test_together_folds_ticked_tracks_into_one_clip(panel: ControlPanel) -> None:
    check(panel, 1, 2, None)
    panel.reference.setChecked(True)
    panel.combine.button(1).setChecked(True)
    assert panel.together
    specs = panel.build_specs()
    sampled = [s for s in specs if not s.is_reference]
    assert {(s.track_idx, s.co_tracks) for s in sampled} == {(1, (2,)), (None, ())}
    blended = next(s for s in sampled if s.co_tracks)
    assert blended.prompt.track_idx == 1, "the blend primes from its primary track"
    assert [s.track_idx for s in specs if s.is_reference] == [1, 2]
    assert "1+2" in panel.count.text()

    panel.combine.button(0).setChecked(True)
    assert {s.track_idx for s in panel.build_specs() if not s.is_reference} == {
        1,
        2,
        None,
    }


def test_style_editor_round_trips_the_walk(panel: ControlPanel) -> None:
    editor = panel.styles
    editor.add(
        StyleSpec(kind="window", track_idx=SAME_TRACK, walk="windows", period=256)
    )
    assert editor._walk.currentData() == "windows"
    assert editor._period.value() == 256 and editor._period.isVisibleTo(editor)
    editor._walk.setCurrentIndex(editor._walk.findData("random"))
    editor._period.setValue(128)
    spec = editor.styles()[-1]
    assert spec is not None and (spec.walk, spec.period) == ("random", 128)
    editor._walk.setCurrentIndex(editor._walk.findData("none"))
    spec = editor.styles()[-1]
    assert spec is not None and not spec.walking
    assert not editor._period.isVisibleTo(editor)


def test_style_editor_shows_the_ar_rows_only_for_the_ar_walk(
    panel: ControlPanel,
) -> None:
    editor = panel.styles
    editor.add(StyleSpec(kind="window", track_idx=SAME_TRACK, walk="windows"))
    assert not editor._ar_prefix.isVisibleTo(editor)
    editor._walk.setCurrentIndex(editor._walk.findData("ar"))
    assert editor._ar_prefix.isVisibleTo(editor)
    assert editor._ar_cfg.isVisibleTo(editor) and editor._period.isVisibleTo(editor)
    editor._ar_prefix.setValue(0)
    editor._ar_cfg.setValue(2.0)
    editor._ar_temperature.setValue(0.5)
    spec = editor.styles()[-1]
    assert spec is not None and spec.walk == "ar"
    assert (spec.ar_prefix, spec.ar_cfg, spec.ar_temperature) == (0, 2.0, 0.5)
    editor._walk.setCurrentIndex(editor._walk.findData("windows"))
    assert not editor._ar_prefix.isVisibleTo(editor)


def test_reference_rows_are_one_per_track(panel: ControlPanel) -> None:
    check(panel, 1, 2)
    panel.reference.setChecked(True)
    assert sum(1 for s in panel.build_specs() if s.is_reference) == 2


def test_prompt_follows_each_track_by_default(panel: ControlPanel) -> None:
    check(panel, 1, 2)
    assert panel.prompt_track.currentData() == FOLLOW
    prompts = {s.track_idx: s.prompt.track_idx for s in panel.build_specs()}
    assert prompts == {1: 1, 2: 2}


def test_a_pinned_prompt_track_overrides_the_follow(panel: ControlPanel) -> None:
    check(panel, 1, 2)
    panel.prompt_track.setCurrentIndex(panel.prompt_track.findData(3))
    assert {s.prompt.track_idx for s in panel.build_specs()} == {3}


def test_start_slider_maps_to_a_frame_that_leaves_room(panel: ControlPanel) -> None:
    panel.seconds.setValue(10.0)
    panel.start.setValue(1000)  # all the way right
    spec = panel.build_specs()[0]
    assert spec.prompt.start_frame + spec.n_frames < 40_000


def test_cold_start_prompts_are_zero_length(panel: ControlPanel) -> None:
    panel.prompt_kind.button(0).setChecked(True)
    spec = panel.build_specs()[0]
    assert spec.prompt.kind == "none" and spec.prompt.frames(FPS) == 0


def test_file_prompt_carries_the_path(panel: ControlPanel) -> None:
    panel.prompt_kind.button(2).setChecked(True)
    panel.prompt_path.setText("/tmp/loop.wav")
    panel.file_start.setValue(12.0)
    spec = panel.build_specs()[0]
    assert spec.prompt.kind == "file"
    assert spec.prompt.path == "/tmp/loop.wav" and spec.prompt.start_sec == 12.0


def test_nothing_ticked_builds_nothing(panel: ControlPanel) -> None:
    check(panel)
    assert panel.build_specs() == []


def test_generate_with_nothing_selected_emits_no_batch(panel: ControlPanel) -> None:
    check(panel)
    seen: list[Any] = []
    panel.generate_requested.connect(seen.append)
    panel.generate.click()
    assert seen == []
    assert "tick at least one" in panel.count.text()


def test_filter_hides_rows_and_all_only_ticks_visible(panel: ControlPanel) -> None:
    panel._check_visible(False)
    panel.filter.setText("track 2")
    panel._check_visible(True)
    assert panel.selected_tracks() == [2]


def test_none_clears_hidden_rows_too(panel: ControlPanel) -> None:
    # a ticked track filtered out of sight would otherwise generate invisibly
    check(panel, 0, 3)
    panel.filter.setText("track 3")
    panel._check_visible(False)
    panel.filter.setText("")
    assert panel.selected_tracks() == []


def test_prompt_cannot_outlast_the_clip(panel: ControlPanel) -> None:
    panel.seconds.setValue(4.0)
    assert panel.prompt_seconds.maximum() == pytest.approx(2.0)
    assert panel.prompt_seconds.value() <= 2.0
    spec = panel.build_specs()[0]
    assert spec.prompt.frames(FPS) < spec.n_frames


def test_cfg_slider_reaches_eight(panel: ControlPanel) -> None:
    # identity is at chance until roughly cfg 8; a panel capped at 3 could not
    # reach the setting that makes conditioning audible at all
    assert panel.cfg.maximum() >= 8.0


def test_checkpoint_selection_is_reported_once(panel: ControlPanel) -> None:
    picked: list[str] = []
    panel.checkpoint_picked.connect(picked.append)
    panel.checkpoints.activated.emit(1)
    assert picked == ["b/two.ckpt"]
    panel.set_checkpoint("b/two.ckpt")
    panel.checkpoints.activated.emit(1)
    assert picked == ["b/two.ckpt"]


# -- view-model ------------------------------------------------------------


def test_results_land_on_the_rows_that_asked_for_them(vm: SynthViewModel) -> None:
    specs = [RenderSpec(track_idx=t, style=StyleSpec(), n_frames=200) for t in (1, 2)]
    assert vm.generate(specs) == 2
    assert [v.status for v in vm.variants] == ["generating", "generating"]
    vm._pump()
    assert [v.status for v in vm.variants] == ["ready", "ready"]
    assert {v.spec.track_idx for v in vm.variants} == {1, 2}
    assert vm.selected is not None and vm.selected.item_id == specs[0].item_id


def test_a_failure_marks_only_its_own_row(qapp, tmp_path: Path) -> None:
    specs = [RenderSpec(track_idx=t, style=StyleSpec(), n_frames=200) for t in (1, 2)]
    producer = FakeProducer(fail={specs[0].item_id})
    producer.ready()
    model = SynthViewModel(producer, tmp_path)
    model._pump()
    model.generate(specs)
    model._pump()
    statuses = {v.spec.track_idx: v.status for v in model.variants}
    assert statuses == {1: "failed", 2: "ready"}


def test_duplicate_specs_are_not_queued_twice(vm: SynthViewModel) -> None:
    spec = RenderSpec(track_idx=1, style=StyleSpec(), n_frames=200)
    assert vm.generate([spec]) == 1
    vm._pump()
    assert vm.generate([spec]) == 0
    assert len(vm.variants) == 1


def test_progress_is_forwarded(vm: SynthViewModel) -> None:
    seen: list[tuple[float, int, int]] = []
    vm.progress_changed.connect(lambda f, s, t: seen.append((f, s, t)))
    vm.generate([RenderSpec(n_frames=200, style=StyleSpec())])
    vm._pump()
    assert (0.5, 256, 512) in seen


def test_cancel_drops_pending_rows(qapp, tmp_path: Path) -> None:
    producer = FakeProducer()
    producer.ready()
    model = SynthViewModel(producer, tmp_path)
    model._pump()
    model.generate([RenderSpec(track_idx=1, style=StyleSpec(), n_frames=200)])
    model.cancel()
    assert producer.cancels == 1
    assert model.variants == []
    # the worker still answers; the answer must not resurrect the row
    model._pump()
    assert model.variants == []


def test_save_writes_the_triple(vm: SynthViewModel, tmp_path: Path) -> None:
    vm.generate([RenderSpec(track_idx=1, style=StyleSpec(), n_frames=200)])
    vm._pump()
    saved = vm.save_selected()
    assert saved is not None
    assert saved.wav.exists() and saved.tokens.exists() and saved.spec.exists()
    assert saved.wav.parent == tmp_path
    assert vm.variants[0].saved is not None


def test_save_with_nothing_selected_says_so(vm: SynthViewModel) -> None:
    messages: list[str] = []
    vm.message.connect(messages.append)
    assert vm.save_selected() is None
    assert messages == ["nothing to save"]


def test_switching_checkpoint_republishes_the_corpus(vm: SynthViewModel) -> None:
    seen: list[Any] = []
    vm.corpus_changed.connect(seen.append)
    vm.switch_checkpoint("other.ckpt")
    vm._pump()
    assert vm.producer.switches == ["other.ckpt"]
    assert seen and seen[-1].checkpoint == "other.ckpt"


# -- results panel ---------------------------------------------------------


def test_results_panel_shows_a_render_on_every_strip(qapp, vm: SynthViewModel) -> None:
    view = ResultsPanel()
    view.resize(800, 600)
    view.set_geometry_meta(FPS, 2048)
    vm.generate([RenderSpec(track_idx=1, style=StyleSpec(), n_frames=400)])
    vm._pump()
    render = vm.variants[0].render
    assert render is not None
    view.set_variants(vm.variants)
    view.show_render(render)

    assert view.variants.count() == 1
    assert "uniq" in view.stats.text()
    assert "cfg" in view.recipe.text()
    assert view.mel._image is not None and view.tokens._image is not None
    assert view.save.isEnabled()

    view.set_position(1.0, 2.0)
    assert view.tokens._position == pytest.approx(0.5)
    assert view.mel._position == pytest.approx(0.5)


def test_clearing_a_render_empties_the_strips(qapp) -> None:
    view = ResultsPanel()
    view.resize(800, 600)
    view.show_render(None)
    assert view.mel._image is None and view.tokens._image is None
    assert not view.save.isEnabled()
    assert view.stats.text() == ""


def test_prompt_marker_covers_the_forced_region(qapp, vm: SynthViewModel) -> None:
    view = ResultsPanel()
    view.resize(800, 600)
    view.set_geometry_meta(FPS, 2048)
    spec = RenderSpec(track_idx=1, style=StyleSpec(), n_frames=1000)
    vm.generate([spec])
    vm._pump()
    view.show_render(vm.variants[0].render)
    assert view.tokens._marker == pytest.approx(spec.prompt.frames(FPS) / 1000)


def test_scrubbing_any_strip_reports_a_fraction(qapp) -> None:
    view = ResultsPanel()
    view.resize(800, 600)
    seen: list[float] = []
    view.scrubbed.connect(seen.append)
    for strip in (view.wave, view.mel, view.tokens):
        strip.set_pcm(np.zeros(44100, dtype=np.int16)) if strip is view.wave else None
        centre = strip.rect().center().toPointF()
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            centre,
            centre,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        strip.mousePressEvent(event)
    assert len(seen) == 3
    assert all(0.4 < f < 0.6 for f in seen)


@pytest.mark.parametrize(
    "key, signal",
    [
        (Qt.Key.Key_Space, "play_pause"),
        (Qt.Key.Key_R, "restart"),
        (Qt.Key.Key_S, "save_requested"),
    ],
)
def test_transport_keys(qapp, key, signal: str) -> None:
    view = ResultsPanel()
    fired: list[int] = []
    getattr(view, signal).connect(lambda *_: fired.append(1))
    view.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    )
    assert fired == [1]


def test_arrow_keys_nudge_the_playhead(qapp) -> None:
    view = ResultsPanel()
    seen: list[float] = []
    view.seek_by.connect(seen.append)
    for key, modifier in (
        (Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier),
        (Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier),
    ):
        view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, modifier))
    assert seen == [1.0, -5.0]


def test_selecting_a_row_reports_its_item_id(qapp) -> None:
    view = ResultsPanel()
    seen: list[str] = []
    view.selected.connect(seen.append)
    spec = RenderSpec(track_idx=1, style=StyleSpec())
    view.set_variants([Variant(spec=spec)])
    assert seen and seen[-1] == spec.item_id
    assert isinstance(view.variants.item(0), QListWidgetItem)


def test_the_first_playable_result_is_selected_even_if_it_is_not_row_zero(
    qapp, tmp_path: Path
) -> None:
    # reference rows finish before any sampling starts, so the row selected when
    # a batch is queued is usually the last one to become playable
    specs = [RenderSpec(track_idx=t, style=StyleSpec(), n_frames=200) for t in (1, 2)]
    producer = FakeProducer(defer=True)
    producer.ready()
    model = SynthViewModel(producer, tmp_path)
    model._pump()
    model.generate(specs)
    model.select(specs[0].item_id)

    producer.produce([specs[1]])
    model._pump()
    assert model.selected is not None
    assert model.selected.item_id == specs[1].item_id


@pytest.mark.parametrize(
    "kind, hidden",
    [
        (0, ["track", "start", "file", "file at", "length"]),
        (1, ["file", "file at"]),
        (2, ["track", "start"]),
    ],
)
def test_prompt_rows_hide_their_labels_too(
    panel: ControlPanel, kind: int, hidden
) -> None:
    # a row built from a layout needs labelForField(layout); getting that wrong
    # leaves a stranded label beside empty space
    panel.prompt_kind.button(kind).setChecked(True)
    labels = {
        "track": panel.prompt_track,
        "start": panel._start_row,
        "file": panel._path_row,
        "file at": panel.file_start,
        "length": panel.prompt_seconds,
    }
    for name, field in labels.items():
        label = panel._prompt_form.labelForField(field)
        assert label is not None, name
        assert label.isVisibleTo(panel) == (name not in hidden), name


def test_strip_images_survive_their_source_buffer(qapp) -> None:
    # QImage does not own the buffer it wraps; without a deep copy the strip
    # paints freed memory once the numpy array goes away
    import gc

    view = ResultsPanel()
    view.resize(400, 400)
    view.set_geometry_meta(FPS, 2048)
    tokens = np.random.default_rng(0).integers(0, 2048, size=(500, 3))
    view.tokens.set_tokens(tokens)
    del tokens
    gc.collect()
    image = view.tokens._image
    assert image is not None
    assert image.pixelColor(0, 0).isValid()
    assert image.width() == 500


def test_new_style_entries_open_with_the_configured_walk(qapp) -> None:
    ui = UiCfg(style_walk="ar", style_ar_temperature=1.2, style_ar_cfg=1.5)
    widget = ControlPanel(ui, ["a/one.ckpt"])
    widget.set_corpus(WorkerReady("a/one.ckpt", tracks(), dict(META), 4096))
    first = widget.styles.styles()[0]
    assert first is not None and first.kind == "window" and first.walk == "ar"
    assert (first.ar_temperature, first.ar_cfg, first.track_idx) == (
        1.2,
        1.5,
        SAME_TRACK,
    )
    widget.styles.add(StyleSpec(kind="random", track_idx=SAME_TRACK, walk="none"))
    assert widget.styles.styles()[-1] is not None
    plain = ControlPanel(UiCfg(style_walk="none"), ["a/one.ckpt"])
    entry = plain.styles.styles()[0]
    assert entry is not None and not entry.walking


def test_backend_picker_lists_backends_and_loads_their_newest(qapp) -> None:
    from slice_synth.config import UiCfg
    from slice_synth.view.controls import ControlPanel

    panel = ControlPanel(
        UiCfg(),
        {
            "ar": ["saved_ar_a/ar_latest.ckpt"],
            "zflow": ["saved_zflow_b/zflow_latest.ckpt", "saved_zflow_b/last.ckpt"],
            "mdm": [],
        },
    )
    picked: list[str] = []
    panel.checkpoint_picked.connect(picked.append)
    assert [panel.backends.itemText(i) for i in range(panel.backends.count())] == ["ar", "zflow"]
    assert panel.backends.isVisibleTo(panel) and panel.checkpoints.count() == 1
    assert panel.top_p.isEnabled() and panel.temperature.isEnabled()

    panel.backends.setCurrentIndex(1)
    panel.backends.activated.emit(1)
    assert picked == ["saved_zflow_b/zflow_latest.ckpt"]
    assert panel.checkpoints.count() == 2 and not panel.checkpoints.isEnabled()
    panel.set_checkpoint("saved_zflow_b/zflow_latest.ckpt")
    assert panel.checkpoints.isEnabled() and panel.backends.isEnabled()
    # a flow model ignores the token-draw knobs
    assert not panel.temperature.isEnabled() and not panel.top_k.isEnabled()

    # the worker can land on a checkpoint the panel never listed
    panel.set_checkpoint("saved_mdm_c/mdm_latest.ckpt")
    assert panel.backends.currentData() == "zflow"  # no mdm row to move to
    assert panel.checkpoints.count() == 0
    assert panel.temperature.isEnabled() and not panel.top_p.isEnabled()

    flat = ControlPanel(UiCfg(), ["saved_ar_a/ar_latest.ckpt"])
    assert not flat.backends.isVisibleTo(flat) and flat.checkpoints.count() == 1
