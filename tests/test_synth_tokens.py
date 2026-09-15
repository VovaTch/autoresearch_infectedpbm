"""
Token statistics and the palette.

These are the numbers that separate "the transformer collapsed" from "the
tokenizer is the ceiling", so they have to mean what generate_ar.token_stats
means -- this is a numpy restatement of it, not a new metric.
"""

from __future__ import annotations

import numpy as np

from slice_synth.model.tokens import (
    colour_lut,
    level_stats,
    loop_lag,
    stats_line,
    token_image,
)

FPS = 172.265625


def test_entropy_of_a_single_code_is_zero() -> None:
    stats = level_stats(np.zeros((100, 2), dtype=np.int64), 2048)
    assert [s.unique for s in stats] == [1, 1]
    assert all(s.entropy == 0.0 for s in stats)


def test_entropy_of_uniform_usage_is_one() -> None:
    codes = np.tile(np.arange(256), 4)[:, None]
    stats = level_stats(codes, 256)
    assert stats[0].unique == 256
    assert abs(stats[0].entropy - 1.0) < 1e-9


def test_two_codes_out_of_four_is_a_half() -> None:
    codes = np.tile(np.array([0, 1]), 50)[:, None]
    assert abs(level_stats(codes, 4)[0].entropy - 0.5) < 1e-9


def test_loop_detector_finds_the_period() -> None:
    rng = np.random.default_rng(0)
    tile = rng.integers(0, 2048, size=(37, 3))
    assert loop_lag(np.tile(tile, (20, 1))) == 37


def test_loop_detector_stays_quiet_on_noise() -> None:
    rng = np.random.default_rng(1)
    assert loop_lag(rng.integers(0, 2048, size=(2000, 3))) == 0


def test_stats_line_reports_every_level_and_the_loop() -> None:
    rng = np.random.default_rng(2)
    line = stats_line(rng.integers(0, 2048, size=(1000, 3)), 2048, FPS)
    assert line.count("uniq") == 3 and "loop none" in line
    assert stats_line(np.zeros((0, 3), dtype=np.int64), 2048, FPS) == "no tokens"


def test_palette_is_stable_and_covers_every_code() -> None:
    a, b = colour_lut(2048), colour_lut(2048)
    assert a.shape == (2048, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    # nothing may be drawn as pure black, which reads as "no data"
    assert a.max(axis=1).min() > 0


def test_token_image_is_one_row_per_level_and_clamps_pad() -> None:
    lut = colour_lut(2048)
    codes = np.array([[0, 1, 2048]], dtype=np.int64)  # 2048 is pad_id, out of range
    image = token_image(codes, lut)
    assert image.shape == (3, 1, 3)
    assert np.array_equal(image[2, 0], lut[2047])
    assert token_image(np.zeros((0, 3), dtype=np.int64), lut).shape == (0, 0, 3)
