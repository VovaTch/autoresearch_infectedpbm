"""
Recipes: the cross product, the content-addressed ids, and the save round trip.

The ids are load-bearing. A result is matched back to its request by item_id
alone, so two specs that differ anywhere must not collide -- otherwise a
sixteen-variant batch could quietly show the same clip twice under two labels.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from slice_synth.model.library import load_spec, safe_stem, save_render
from slice_synth.model.types import (
    SAME_TRACK,
    PromptSpec,
    Render,
    RenderSpec,
    StyleSpec,
    build_variants,
)


def test_build_variants_is_a_cross_product() -> None:
    base = RenderSpec(checkpoint="ckpt")
    specs = build_variants(base, [3, 7], [StyleSpec(), None], [0, 1])
    assert len(specs) == 8
    assert {(s.track_idx, s.style is None, s.seed) for s in specs} == {
        (t, n, s) for t in (3, 7) for n in (False, True) for s in (0, 1)
    }


def test_build_variants_adds_one_reference_per_track() -> None:
    specs = build_variants(RenderSpec(), [3, 7], [StyleSpec()], [0, 1], reference=True)
    references = [s for s in specs if s.is_reference]
    assert len(references) == 2
    assert [s.track_idx for s in references] == [3, 7]
    # a reference decodes real tokens, so conditioning does not apply to it
    assert all(s.style is None for s in references)


def test_build_variants_skips_a_null_track_reference() -> None:
    specs = build_variants(RenderSpec(), [None], [StyleSpec()], [0], reference=True)
    assert not any(s.is_reference for s in specs)


def test_build_variants_together_folds_real_tracks_into_one_cell() -> None:
    specs = build_variants(
        RenderSpec(), [3, 7, None], [StyleSpec()], [0, 1], reference=True, together=True
    )
    sampled = [s for s in specs if not s.is_reference]
    # one blended cell plus the null cell, times two seeds
    assert {(s.track_idx, s.co_tracks) for s in sampled} == {(3, (7,)), (None, ())}
    assert len(sampled) == 4
    assert [s.track_idx for s in specs if s.is_reference] == [3, 7]
    assert all(s.co_tracks == () for s in specs if s.is_reference)


def test_build_variants_together_with_only_null_is_just_null() -> None:
    specs = build_variants(RenderSpec(), [None], [StyleSpec()], [0], together=True)
    assert [(s.track_idx, s.co_tracks) for s in specs] == [(None, ())]


def test_build_variants_empty_axis_yields_nothing() -> None:
    assert build_variants(RenderSpec(), [], [StyleSpec()], [0]) == []
    assert build_variants(RenderSpec(), [3], [], [0]) == []
    assert build_variants(RenderSpec(), [3], [StyleSpec()], []) == []


def test_item_id_is_stable_and_sensitive() -> None:
    spec = RenderSpec(track_idx=3, style=StyleSpec(seed=1), checkpoint="c")
    assert spec.item_id == replace(spec).item_id
    for changed in (
        replace(spec, seed=1),
        replace(spec, cfg_strength=3.0),
        replace(spec, style=StyleSpec(seed=2)),
        replace(spec, prompt=PromptSpec(start_frame=5)),
        replace(spec, kind="reference"),
        replace(spec, checkpoint="d"),
        replace(spec, co_tracks=(5,)),
        replace(spec, style=StyleSpec(seed=1, walk="random")),
        replace(spec, style=StyleSpec(seed=1, walk="random", period=256)),
    ):
        assert changed.item_id != spec.item_id


def test_spec_json_round_trip() -> None:
    spec = RenderSpec(
        track_idx=None,
        style=StyleSpec(kind="interp", track_idx=SAME_TRACK, track_b=4, mix=0.25),
        prompt=PromptSpec(kind="file", path="/tmp/x.wav", start_sec=12.0),
        cfg_strength=8.0,
        kind="reference",
    )
    assert RenderSpec.from_json(spec.to_json()) == spec


def test_co_tracks_and_walk_survive_json() -> None:
    spec = RenderSpec(
        track_idx=3,
        co_tracks=(14, 2),
        style=StyleSpec(kind="window", walk="windows", period=256),
    )
    again = RenderSpec.from_json(spec.to_json())
    assert again == spec and isinstance(again.co_tracks, tuple)
    assert again.tracks == (3, 14, 2)


def test_old_recipes_without_the_new_keys_still_load() -> None:
    raw = RenderSpec(track_idx=3, style=StyleSpec()).to_json()
    del raw["co_tracks"]
    del raw["style"]["walk"]
    del raw["style"]["period"]
    spec = RenderSpec.from_json(raw)
    assert spec.co_tracks == () and spec.style is not None and not spec.style.walking


def test_labels_spell_out_co_tracks_and_walks() -> None:
    assert (
        RenderSpec(
            track_idx=3, co_tracks=(14,), style=StyleSpec(track_idx=1, window=2)
        ).label()
        == "t3+14 t1w2 s0"
    )
    assert StyleSpec(track_idx=1, window=2, walk="windows").label() == "t1w2~win512"
    assert (
        StyleSpec(kind="random", seed=4, walk="random", period=256).label()
        == "rand4~rnd256"
    )
    assert StyleSpec(track_idx=1, window=2, walk="random", period=0).label() == "t1w2"
    assert StyleSpec(track_idx=1, window=2, walk="ar").label() == "t1w2~ar512"
    assert (
        StyleSpec(
            track_idx=1,
            window=2,
            walk="ar",
            ar_prefix=0,
            ar_cfg=2.0,
            ar_temperature=0.5,
        ).label()
        == "t1w2~ar512p0c2T0.5"
    )


def test_ar_walk_fields_survive_json_and_old_recipes_default_them() -> None:
    style = StyleSpec(walk="ar", ar_prefix=2, ar_temperature=0.7, ar_cfg=3.0)
    again = StyleSpec.from_json(style.to_json())
    assert again == style
    raw = StyleSpec(walk="windows").to_json()
    for key in ("ar_prefix", "ar_temperature", "ar_cfg"):
        del raw[key]
    old = StyleSpec.from_json(raw)
    assert (old.ar_prefix, old.ar_temperature, old.ar_cfg) == (4, 1.0, 1.0)


def test_segments_cover_the_clip() -> None:
    assert StyleSpec(walk="random", period=512).segments(1024) == 2
    assert StyleSpec(walk="random", period=512).segments(1025) == 3
    assert StyleSpec(walk="none", period=512).segments(4096) == 1


def test_prompt_frames_is_zero_for_a_cold_start() -> None:
    assert PromptSpec(kind="none", seconds=3.0).frames(172.265625) == 0
    assert PromptSpec(kind="corpus", seconds=3.0).frames(172.265625) == 516


def test_same_track_label_says_own() -> None:
    assert StyleSpec(track_idx=SAME_TRACK).label().startswith("own")
    assert StyleSpec(track_idx=4, window=2).label() == "t4w2"


def test_safe_stem_strips_path_characters() -> None:
    stem = safe_stem("t3 own/auto s0", "r_abc")
    assert "/" not in stem and " " not in stem and stem.endswith("r_abc")


def test_save_render_writes_three_files_that_reload(tmp_path: Path) -> None:
    spec = RenderSpec(track_idx=3, style=StyleSpec(kind="random", seed=9))
    render = Render(
        spec=spec,
        tokens=np.arange(30, dtype=np.int16).reshape(10, 3),
        pcm=(np.random.default_rng(0).normal(0, 3000, 44100)).astype(np.int16),
        style_used=np.arange(4, dtype=np.float32),
        fill=0.9,
    )
    saved = save_render(render, tmp_path)
    assert saved.wav.exists() and saved.tokens.exists() and saved.spec.exists()

    reloaded, style = load_spec(saved.spec)
    assert reloaded == spec
    assert style is not None and np.allclose(style, render.style_used)
    assert np.array_equal(np.load(saved.tokens), render.tokens)


def test_a_walked_style_saves_as_a_matrix(tmp_path: Path) -> None:
    spec = RenderSpec(
        track_idx=3, style=StyleSpec(kind="random", seed=9, walk="random")
    )
    render = Render(
        spec=spec,
        tokens=np.zeros((10, 3), dtype=np.int16),
        pcm=np.zeros(4410, dtype=np.int16),
        style_used=np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    _, style = load_spec(save_render(render, tmp_path).spec)
    assert style is not None and style.shape == (3, 4)
    assert np.allclose(style, render.style_used)


def test_the_model_layer_stays_free_of_torch_and_qt() -> None:
    """
    The UI process imports the model layer and must never end up with torch in
    it: that is the entire reason generation lives in a child process, and a
    stray import would cost seconds of startup and a CUDA context nobody asked
    for.
    """
    import subprocess
    import sys

    code = (
        "import importlib, sys;"
        "[importlib.import_module(f'slice_synth.model.{m}') for m in "
        "('types', 'spectrum', 'tokens', 'library', 'protocols')];"
        "leaked = [m for m in ('torch', 'onnxruntime', 'lightning') if m in sys.modules];"
        "print(','.join(leaked))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"model layer imported {out.stdout.strip()}"
