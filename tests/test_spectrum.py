"""
The numpy mel front end.

It exists so the UI process can draw a spectrogram without torch or librosa, so
the tests have to pin the two things that would silently go wrong: the filter
bank's placement, and the cost staying flat in clip length.
"""

from __future__ import annotations

import numpy as np

from slice_synth.model.spectrum import (
    hz_to_mel,
    mel_filterbank,
    mel_spectrogram,
    mel_to_hz,
    plan_geometry,
)

SR = 44100


def test_mel_scale_round_trips() -> None:
    for hz in (20.0, 440.0, 1000.0, 11025.0):
        assert abs(float(mel_to_hz(hz_to_mel(hz))) - hz) < 1e-6


def test_filterbank_is_ordered_and_bounded() -> None:
    bank = mel_filterbank(SR, 2048, 128)
    assert bank.shape == (128, 1025)
    peaks = bank.argmax(axis=1)
    assert np.all(np.diff(peaks) >= 0)
    assert bank.min() >= 0.0 and bank.max() <= 1.0 + 1e-6
    # every band has to catch something, or part of the picture is always dark
    assert np.all(bank.sum(axis=1) > 0.0)


def test_a_sine_lands_in_the_expected_band() -> None:
    t = np.arange(SR) / SR
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * t) * 32767).astype(np.int16)
    mel = mel_spectrogram(tone, SR, columns=200)
    band = int(mel.mean(axis=1).argmax())
    edges = np.asarray(mel_to_hz(np.linspace(hz_to_mel(20.0), hz_to_mel(SR / 2), 130)))
    assert edges[band] <= 1000.0 <= edges[band + 2]


def test_output_is_scaled_for_drawing() -> None:
    noise = (np.random.default_rng(0).normal(0, 3000, SR)).astype(np.int16)
    mel = mel_spectrogram(noise, SR, columns=128)
    assert mel.dtype == np.float32
    assert mel.min() >= 0.0 and mel.max() <= 1.0
    assert abs(float(mel.max()) - 1.0) < 1e-6  # the peak always reaches the top


def test_columns_bound_the_frame_count_whatever_the_length() -> None:
    for seconds in (2, 30, 90):
        pcm = np.zeros(SR * seconds, dtype=np.int16)
        mel = mel_spectrogram(pcm, SR, columns=400)
        assert mel.shape[0] == 128
        assert mel.shape[1] <= 401


def test_geometry_keeps_windows_overlapping() -> None:
    for samples in (SR, SR * 90, SR * 600):
        n_fft, hop = plan_geometry(samples, 800)
        assert hop >= 256 and n_fft >= 2 * hop


def test_empty_and_short_inputs_do_not_raise() -> None:
    assert mel_spectrogram(np.zeros(0, dtype=np.int16), SR).shape == (128, 0)
    assert mel_spectrogram(np.zeros(64, dtype=np.int16), SR).shape[0] == 128
