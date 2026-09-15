"""
Reading a token stream by eye.

"It sounds bad" has distinct causes that look different in the codes, and they
point at different fixes: collapse (few distinct codes, low entropy), looping (a
short period repeats), and drift (stats fine, but unlike the reference). This is
generate_ar.token_stats restated in numpy -- the UI process holds no torch --
and returning values instead of printing them, so a widget can draw them.

The colour map is a hash, not a ramp. Codebook indices carry no ordering
whatsoever: code 5 is not "between" 4 and 6 in any sense the decoder respects,
so a gradient over id would invent structure that is not there. A hash gives
neighbouring ids unrelated colours, which is what makes a repeating passage
stand out as a repeating pattern rather than a smooth wash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOOP_MATCH = 0.9
MAX_LAG = 512


@dataclass(frozen=True)
class LevelStats:
    """
    Degeneracy signature for one RVQ level.

    Args:
      level (int): RVQ depth index.
      unique (int): distinct codes used.
      entropy (float): Shannon entropy of the code histogram, normalized by
        log2(codebook size), so 1.0 is uniform usage and 0.0 is one code.
    """

    level: int
    unique: int
    entropy: float


def level_stats(tokens: np.ndarray, num_tokens: int) -> list[LevelStats]:
    """
    Args:
      tokens (np.ndarray): (T, R) integer codes.
      num_tokens (int): codebook size, for the entropy ceiling.

    Returns:
      list[LevelStats]: one entry per RVQ level, in depth order. Empty for an
        empty stream.
    """
    if tokens.size == 0:
        return []
    codes = np.asarray(tokens, dtype=np.int64)
    ceiling = np.log2(max(num_tokens, 2))
    out: list[LevelStats] = []
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:, level], minlength=num_tokens).astype(np.float64)
        probs = counts[counts > 0] / counts.sum()
        entropy = float(-(probs * np.log2(probs)).sum() / ceiling)
        out.append(LevelStats(level, int((counts > 0).sum()), entropy))
    return out


def loop_lag(tokens: np.ndarray, max_lag: int = MAX_LAG) -> int:
    """
    Smallest period at which the coarsest level repeats itself.

    Level 0 carries the most energy, so a loop there is a loop you can hear; the
    finer levels drift even inside a repeat.

    Args:
      tokens (np.ndarray): (T, R) integer codes.
      max_lag (int): longest period to test, in frames.

    Returns:
      int: the lag in frames, or 0 when nothing repeats.
    """
    if tokens.size == 0:
        return 0
    flat = np.asarray(tokens[:, 0], dtype=np.int64)
    for lag in range(1, min(max_lag, flat.size // 2)):
        if float(np.mean(flat[lag:] == flat[:-lag])) > LOOP_MATCH:
            return lag
    return 0


def stats_line(tokens: np.ndarray, num_tokens: int, fps: float) -> str:
    """
    Args:
      tokens (np.ndarray): (T, R) integer codes.
      num_tokens (int): codebook size.
      fps (float): tokenizer frames per second, for reporting a loop in seconds.

    Returns:
      str: a single status-line summary of the stream.
    """
    stats = level_stats(tokens, num_tokens)
    if not stats:
        return "no tokens"
    parts = [f"L{s.level} uniq {s.unique} H {s.entropy:.3f}" for s in stats]
    lag = loop_lag(tokens)
    parts.append(f"loop {f'{lag} ({lag / fps:.2f}s)' if lag else 'none'}")
    return "  ·  ".join(parts)


def colour_lut(num_tokens: int, seed: int = 0) -> np.ndarray:
    """
    A stable id-to-colour table.

    Args:
      num_tokens (int): codebook size.
      seed (int): palette seed; fixed in practice so the same code is the same
        colour across every clip in a session.

    Returns:
      np.ndarray: (num_tokens, 3) uint8 RGB. Saturation and value are held high
        so no code is drawn as mud, and the hue is the hashed part.
    """
    rng = np.random.default_rng(seed)
    hue = rng.permutation(num_tokens).astype(np.float32) / max(num_tokens, 1)
    value = 0.55 + 0.45 * rng.random(num_tokens).astype(np.float32)
    # HSV -> RGB at fixed saturation, written out rather than pulled from a
    # colour library the UI process would otherwise not need.
    sector = hue * 6.0
    index = np.floor(sector).astype(np.int64) % 6
    frac = sector - np.floor(sector)
    saturation = 0.75
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * frac)
    t = value * (1.0 - saturation * (1.0 - frac))
    table = np.stack(
        [
            np.choose(index, [value, q, p, p, t, value]),
            np.choose(index, [t, value, value, q, p, p]),
            np.choose(index, [p, p, t, value, value, q]),
        ],
        axis=-1,
    )
    return (np.clip(table, 0.0, 1.0) * 255.0).astype(np.uint8)


def token_image(tokens: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Args:
      tokens (np.ndarray): (T, R) integer codes.
      lut (np.ndarray): (num_tokens, 3) uint8 palette.

    Returns:
      np.ndarray: (R, T, 3) uint8 RGB, one row per RVQ level. Codes outside the
        palette (pad_id is one past the last entry) are clamped rather than
        raising, so a malformed stream is visible instead of fatal.
    """
    if tokens.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    codes = np.clip(np.asarray(tokens, dtype=np.int64), 0, lut.shape[0] - 1)
    return lut[codes].transpose(1, 0, 2).copy()


__all__ = [
    "LevelStats",
    "colour_lut",
    "level_stats",
    "loop_lag",
    "stats_line",
    "token_image",
]
