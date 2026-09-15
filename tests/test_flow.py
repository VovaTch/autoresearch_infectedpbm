"""
CPU tests for the flow-matching decoder enhancer.

Everything here runs in well under a second on the CPU: no checkpoints, no GPU,
no pair cache. The subject is the arithmetic that the whole model rests on --
the spectral representation, the interpolant, and the Euler sampler -- plus the
config loader's refusal to swallow a typo.
"""

from __future__ import annotations

import math

import pytest
import torch

import train_flow as tf


GEOM = tf.SpecGeom(n_fft=256, hop=64, alpha=0.3, beta=0.4, sample_rate=44100)


def realizable(frames: int, batch: int = 2, seed: int = 0) -> torch.Tensor:
    """
    Build a waveform that the representation can reproduce exactly.

    The Nyquist bin is deliberately dropped by SpecGeom, so a signal carrying
    energy there could never round-trip; real sources are lowpassed far below it
    (measured 2.8e-10 of total energy on a corpus track), and this mirrors that.

    Args:
      frames (int): STFT frames wanted.
      batch (int): batch size.
      seed (int): RNG seed.

    Returns:
      torch.Tensor: (batch, 1, L) waveform with no Nyquist content.
    """
    torch.manual_seed(seed)
    length = GEOM.crop_samples(frames)
    noise = torch.randn(batch, length) * 0.1
    window = torch.hann_window(GEOM.n_fft)
    spec = torch.stft(
        noise, GEOM.n_fft, GEOM.hop, window=window, return_complex=True
    )
    spec[:, int(0.85 * GEOM.bins) :] = 0
    wav = torch.istft(spec, GEOM.n_fft, GEOM.hop, window=window, length=length)
    return wav.unsqueeze(1)


def tiny_net() -> tf.UNet2D:
    """
    Returns:
      tf.UNet2D: a UNet small enough to run on the CPU in a test.
    """
    cfg = tf.ModelCfg(
        base_ch=8, ch_mults=[1, 2], blocks_per_level=1, t_embed_dim=16, groups=4
    )
    return tf.UNet2D(cfg)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_compress_and_uncompress_are_exact_inverses():
    torch.manual_seed(3)
    spec = torch.randn(2, GEOM.bins, 8, dtype=torch.complex64)
    back = tf.uncompress(tf.compress(spec, GEOM), GEOM)
    assert back.shape[1] == GEOM.bins + 1
    err = (back[:, : GEOM.bins] - spec).abs().max() / spec.abs().max()
    assert float(err) < 1e-5


def test_uncompress_restores_the_dropped_nyquist_bin_as_silence():
    spec = torch.randn(1, GEOM.bins, 4, dtype=torch.complex64)
    back = tf.uncompress(tf.compress(spec, GEOM), GEOM)
    assert float(back[:, -1].abs().max()) == 0.0


def test_stft_istft_round_trip_reproduces_the_waveform():
    wav = realizable(frames=32)
    back = tf.istft_c(tf.stft_c(wav, GEOM), GEOM, wav.shape[-1])
    assert back.shape == wav.shape
    assert float((back - wav).abs().max() / wav.abs().max()) < 1e-4


def test_stft_c_produces_the_requested_frame_count():
    for frames in (16, 32, 64):
        wav = realizable(frames=frames)
        assert tf.stft_c(wav, GEOM).shape == (2, 2, GEOM.bins, frames)


def test_projection_leaves_a_realizable_spectrogram_alone():
    """Compared in the waveform domain: the compressed domain amplifies the
    numerical noise of near-silent bins (|X|^0.3 turns 1e-9 into ~1e-3), so a
    max-abs comparison there measures float32 dust, not the projection."""
    wav = realizable(frames=32)
    x = tf.stft_c(wav, GEOM)
    projected = tf.istft_c(tf.project(x, GEOM, wav.shape[-1]), GEOM, wav.shape[-1])
    assert float((projected - wav).abs().max() / wav.abs().max()) < 1e-4


def test_compression_pulls_the_dynamic_range_together():
    """A quiet bin and a loud one differ by less after compression than before."""
    spec = torch.tensor([[[1e-4 + 0j], [1.0 + 0j]]], dtype=torch.complex64)
    comp = tf.compress(spec, GEOM)
    ratio_before = 1.0 / 1e-4
    ratio_after = float(comp[0, 0, 1, 0] / comp[0, 0, 0, 0])
    assert ratio_after < ratio_before
    assert ratio_after == pytest.approx(ratio_before**GEOM.alpha, rel=1e-3)


def test_measure_beta_normalises_the_representation():
    tracks = [realizable(frames=400, batch=1)[0, 0] for _ in range(3)]
    geom = tf.SpecGeom(n_fft=256, hop=64, alpha=0.3, beta=1.0)
    beta = tf.measure_beta(tracks, geom, samples=8)
    scaled = tf.SpecGeom(n_fft=256, hop=64, alpha=0.3, beta=beta)
    x = tf.stft_c(tracks[0].reshape(1, 1, -1), scaled)
    assert 0.3 < float(x.std()) < 3.0


# ---------------------------------------------------------------------------
# Interpolant and sampler
# ---------------------------------------------------------------------------


def module_for(sigma: float = 0.15, steps: int = 4) -> tf.FlowModule:
    """
    Args:
      sigma (float): bridge noise scale.
      steps (int): sampler steps.

    Returns:
      tf.FlowModule: module wrapping a tiny CPU UNet.
    """
    cfg = tf.FlowConfig()
    cfg.flow.sigma = sigma
    cfg.flow.steps = steps
    cfg.model = tf.ModelCfg(
        base_ch=8, ch_mults=[1, 2], blocks_per_level=1, t_embed_dim=16, groups=4
    )
    return tf.FlowModule(cfg, GEOM)


def test_interpolant_endpoints_are_the_two_data_distributions():
    module = module_for()
    x0, x1 = torch.randn(2, 2, 8, 4), torch.randn(2, 2, 8, 4)
    at_zero = module.interpolate(x0, x1, torch.zeros(2))
    at_one = module.interpolate(x0, x1, torch.ones(2))
    assert torch.allclose(at_zero, x0, atol=1e-6)
    assert torch.allclose(at_one, x1, atol=1e-6)


def test_bridge_noise_peaks_in_the_middle_and_scales_with_sigma():
    x0, x1 = torch.zeros(512, 2, 4, 4), torch.zeros(512, 2, 4, 4)
    for sigma in (0.1, 0.5):
        module = module_for(sigma=sigma)
        mid = module.interpolate(x0, x1, torch.full((512,), 0.5))
        assert float(mid.std()) == pytest.approx(sigma * 0.5, rel=0.1)


def test_zero_sigma_makes_the_interpolant_deterministic():
    module = module_for(sigma=0.0)
    x0, x1 = torch.randn(2, 2, 8, 4), torch.randn(2, 2, 8, 4)
    t = torch.full((2,), 0.3)
    first, second = module.interpolate(x0, x1, t), module.interpolate(x0, x1, t)
    assert torch.allclose(first, second)
    assert torch.allclose(first, 0.7 * x0 + 0.3 * x1, atol=1e-6)


class OracleNet(torch.nn.Module):
    """A network that already knows the answer, for testing the integrator."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target
        self.calls = 0

    def forward(self, x_t, cond, t):
        self.calls += 1
        return self.target


@pytest.mark.parametrize("steps", [1, 4, 16])
def test_an_oracle_sampler_lands_exactly_on_the_target(steps):
    """The last step sits at t=1, where the interpolant is the prediction itself."""
    module = module_for(steps=steps)
    wav = realizable(frames=32)
    target = tf.stft_c(realizable(frames=32, seed=7), GEOM)
    module.net = OracleNet(target)
    out = module.sample(wav, steps=steps, project_steps=False)
    expected = tf.istft_c(target, GEOM, wav.shape[-1])
    assert module.net.calls == steps
    assert float((out - expected).abs().max() / expected.abs().max()) < 1e-4


def test_sampler_is_deterministic_so_overlapping_windows_agree():
    module = module_for()
    wav = realizable(frames=32)
    first = module.sample(wav, steps=3)
    second = module.sample(wav, steps=3)
    assert torch.allclose(first, second)


def test_sampler_starts_from_the_round_trip_at_init():
    """The zero-initialised head makes an untrained model return its input.

    Read with sigma=0: with the bridge noise on, an untrained model returns the
    input plus that noise, which is the correct behaviour but not a fixed point.
    """
    module = module_for(sigma=0.0)
    wav = realizable(frames=32)
    out = module.sample(wav, steps=4, project_steps=False)
    assert float((out - wav).abs().max() / wav.abs().max()) < 1e-3


class SpyNet(torch.nn.Module):
    """Records every iterate it is handed, and answers with a fixed target."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target
        self.seen: list[torch.Tensor] = []

    def forward(self, x_t, cond, t):
        self.seen.append(x_t.clone())
        return self.target


def test_a_rollout_of_zero_steps_is_the_round_trip_at_t_zero():
    """Step 0 of the sampler is x0 at t=0 -- the regime uniform-t training starves."""
    module = module_for(sigma=0.0, steps=8)
    wav = realizable(frames=32)
    x0 = tf.stft_c(wav, GEOM)
    x, t = module.walk(x0, wav.shape[-1], steps=8, stop_at=0)
    assert torch.allclose(x, x0)
    assert float(t) == 0.0


def test_training_iterate_uses_the_sampler_trajectory_when_rolling_out():
    """rollout_prob 1.0 must put the model on its own trajectory, not the ideal one.

    The ideal interpolant is built from the true x1, which the sampler never has;
    training only on it is what left the metallic feedback.
    """
    module = module_for(sigma=0.0, steps=4)
    module.cfg.flow.rollout_prob = 1.0
    module.train()
    torch.manual_seed(2)
    for param in module.net.out_conv.parameters():
        torch.nn.init.normal_(param, std=0.05)
    wav = realizable(frames=32)
    x0 = tf.stft_c(wav, GEOM)
    x1 = tf.stft_c(realizable(frames=32, seed=9), GEOM)

    times = set()
    for _ in range(40):
        x_t, t = module.training_iterate(x0, x1, wav.shape[-1])
        times.add(round(float(t[0]), 4))
        # A rollout can only mix x0 with the model's own estimate; the true x1
        # must never appear in the input. At t=0 the two coincide by definition
        # -- both are x0 -- which is exactly why step 0 trains the hard regime.
        if float(t[0]) > 0:
            assert not torch.allclose(x_t, module.interpolate(x0, x1, t), atol=1e-4)
        else:
            assert torch.allclose(x_t, x0, atol=1e-6)
    assert times <= {0.0, 0.25, 0.5, 0.75}
    assert 0.0 in times


def test_training_iterate_falls_back_to_the_ideal_interpolant():
    module = module_for(sigma=0.0, steps=4)
    module.cfg.flow.rollout_prob = 0.0
    module.train()
    x0 = torch.randn(2, 2, 8, 16)
    x1 = torch.randn(2, 2, 8, 16)
    x_t, t = module.training_iterate(x0, x1, GEOM.crop_samples(16))
    assert torch.allclose(x_t, (1 - t.reshape(-1, 1, 1, 1)) * x0 + t.reshape(-1, 1, 1, 1) * x1, atol=1e-6)


def test_rollout_does_not_leak_gradients_into_the_trajectory():
    """The walk is a no_grad rollout; only the final prediction carries gradient."""
    module = module_for(sigma=0.0, steps=4)
    module.cfg.flow.rollout_prob = 1.0
    module.train()
    wav = realizable(frames=32)
    x_t, _ = module.training_iterate(
        tf.stft_c(wav, GEOM), tf.stft_c(wav, GEOM), wav.shape[-1]
    )
    assert not x_t.requires_grad


def test_every_iterate_is_re_anchored_to_the_round_trip():
    """The property a free Euler integration loses.

    Each step substitutes the current estimate into the interpolant rather than
    stepping along a velocity, so with sigma=0 the input at step i is exactly
    (1-t_i)*x0 + t_i*x1_hat -- the shape the network was trained on. Free Euler
    instead produces iterates carrying no bridge noise at any t, which is what
    ran mrstft to 2.18 against a round-trip baseline of 0.86.
    """
    steps = 4
    module = module_for(sigma=0.0, steps=steps)
    wav = realizable(frames=32)
    x0 = tf.stft_c(wav, GEOM)
    target = tf.stft_c(realizable(frames=32, seed=7), GEOM)
    module.net = SpyNet(target)
    module.sample(wav, steps=steps, project_steps=False)

    assert torch.allclose(module.net.seen[0], x0, atol=1e-6)
    for i, seen in enumerate(module.net.seen[1:], start=1):
        t = i / steps
        assert torch.allclose(seen, (1.0 - t) * x0 + t * target, atol=1e-5)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frames", [16, 32, 64])
def test_unet_preserves_the_spectral_grid(frames):
    net = tiny_net()
    x = torch.randn(2, 2, 8, frames)
    out = net(x, x, torch.rand(2))
    assert out.shape == x.shape


def test_gradients_reach_every_parameter():
    net = tiny_net()
    x = torch.randn(2, 2, 8, 16)
    net(x, x, torch.rand(2)).pow(2).mean().backward()
    starved = [name for name, p in net.named_parameters() if p.grad is None]
    assert starved == []


def test_the_untrained_network_returns_its_iterate_unchanged():
    """Zero-initialised head: at t=0 that is the round trip, at t=1 the target."""
    net = tiny_net()
    x_t = torch.randn(2, 2, 8, 16)
    out = net(x_t, torch.randn(2, 2, 8, 16), torch.rand(2))
    assert torch.allclose(out, x_t, atol=1e-6)


def test_the_network_cannot_see_x0_and_so_cannot_invert_the_interpolant():
    """Given both x_t and x0 the target is recoverable as (x_t-(1-t)x0)/t, and the
    model learns that inversion instead of restoration -- measured as a sampler
    that degrades monotonically with step count. x0 must not reach the network."""
    net = tiny_net()
    torch.manual_seed(5)
    for param in net.out_conv.parameters():
        torch.nn.init.normal_(param, std=0.1)
    x_t, t = torch.randn(2, 2, 8, 16), torch.rand(2)
    first = net(x_t, torch.randn(2, 2, 8, 16), t)
    second = net(x_t, torch.randn(2, 2, 8, 16), t)
    assert torch.allclose(first, second)


def test_the_rejected_conditioning_option_does_reach_the_network():
    cfg = tf.ModelCfg(
        base_ch=8,
        ch_mults=[1, 2],
        blocks_per_level=1,
        t_embed_dim=16,
        groups=4,
        condition_on_x0=True,
    )
    net = tf.UNet2D(cfg)
    assert net.stem.in_channels == 5
    x_t, t = torch.randn(2, 2, 8, 16), torch.rand(2)
    cond = torch.randn(2, 2, 8, 16)
    assert torch.allclose(net(x_t, cond, t), cond, atol=1e-6)


def test_the_frequency_coordinate_channel_breaks_translation_symmetry():
    """Shifting content up the frequency axis must change the answer."""
    net = tiny_net()
    torch.manual_seed(11)
    for p in net.out_conv.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.zeros(1, 2, 8, 16)
    x[:, :, 1] = 1.0
    shifted = torch.roll(x, shifts=3, dims=2)
    low = net(x, x, torch.zeros(1)) - x
    high = net(shifted, shifted, torch.zeros(1)) - shifted
    assert not torch.allclose(low, torch.roll(high, shifts=-3, dims=2), atol=1e-5)


def test_timestep_embedding_separates_nearby_times():
    emb = tf.timestep_embedding(torch.tensor([0.0, 0.5, 1.0]), 32)
    assert emb.shape == (3, 32)
    assert float((emb[0] - emb[1]).abs().max()) > 1e-3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def write_config(tmp_path, body: str):
    """
    Args:
      tmp_path: pytest temporary directory.
      body (str): YAML text.

    Returns:
      pathlib.Path: the written file.
    """
    path = tmp_path / "cfg.yaml"
    path.write_text(body)
    return path


def test_the_phase_term_cannot_be_satisfied_by_shrinking_magnitude():
    """The reason complex L1 alone let the model collapse to returning its input.

    Scaling a prediction down drives a complex L1 toward the magnitude-shrinkage
    optimum, but phase_distance is normalised, so a quieter copy of the same
    wrong phase scores exactly the same.
    """
    from prepare import phase_distance

    torch.manual_seed(4)
    target = realizable(frames=32, batch=1)
    wrong = realizable(frames=32, batch=1, seed=13)
    full = float(phase_distance(wrong, target))
    quiet = float(phase_distance(wrong * 0.1, target))
    assert quiet == pytest.approx(full, rel=0.02)
    assert float((wrong * 0.1 - target).abs().mean()) < float((wrong - target).abs().mean())


def test_the_phase_term_is_zero_for_a_perfect_prediction():
    from prepare import phase_distance

    target = realizable(frames=32, batch=1)
    assert float(phase_distance(target, target)) < 1e-4


def test_config_defaults_load_from_an_empty_file(tmp_path):
    cfg = tf.load_config(write_config(tmp_path, ""))
    assert cfg.data.crop_frames == 256
    assert cfg.flow.sigma == pytest.approx(0.0)
    assert cfg.loss.w_phase == pytest.approx(1.0)
    assert cfg.flow.rollout_prob == pytest.approx(0.5)


def test_config_rejects_an_unknown_key(tmp_path):
    path = write_config(tmp_path, "flow:\n  sigmaa: 0.2\n")
    with pytest.raises(ValueError, match="sigmaa"):
        tf.load_config(path)


def test_config_rejects_an_unknown_section(tmp_path):
    path = write_config(tmp_path, "flwo:\n  sigma: 0.2\n")
    with pytest.raises(ValueError, match="flwo"):
        tf.load_config(path)


def test_config_survives_a_round_trip_through_a_checkpoint_dict(tmp_path):
    from dataclasses import asdict

    cfg = tf.load_config(write_config(tmp_path, "model:\n  base_ch: 48\n"))
    assert tf.config_from_dict(asdict(cfg)).model.base_ch == 48


# ---------------------------------------------------------------------------
# Split and cropping
# ---------------------------------------------------------------------------


def test_held_out_tracks_depend_only_on_the_seed():
    entries = [{"track_idx": i} for i in range(53)]
    cfg = tf.DataCfg(held_out_tracks=5, split_seed=1234)
    first = tf.plan_held_out(entries, cfg)
    assert first == tf.plan_held_out(entries, cfg)
    assert len(first) == 5
    other = tf.plan_held_out(entries, tf.DataCfg(held_out_tracks=5, split_seed=7))
    assert other != first


def fake_track(frames: int = 4000) -> tf.PairTrack:
    """
    Args:
      frames (int): track length in frames.

    Returns:
      tf.PairTrack: a track whose two streams differ by a known constant.
    """
    length = frames * GEOM.hop
    clean = torch.arange(length, dtype=torch.float32)
    return tf.PairTrack(
        rt=clean + 1000.0,
        clean=clean,
        track_idx=0,
        track_name="fake",
        num_frames=frames,
        val_windows=[(1000, 1600)],
    )


def test_a_crop_takes_the_same_span_from_both_streams():
    track = fake_track()
    crop = tf._crop(track, start_frame=17, geom=GEOM, frames=64)
    assert crop["rt"].shape == crop["clean"].shape == (1, GEOM.crop_samples(64))
    assert torch.allclose(crop["rt"] - crop["clean"], torch.full_like(crop["clean"], 1000.0))
    assert float(crop["clean"][0, 0]) == 17 * GEOM.hop


def test_training_crops_never_touch_the_validation_windows():
    track = fake_track()
    dataset = tf.PairCropDataset([track], tf.DataCfg(crop_frames=64), GEOM, epoch_len=200)
    for i in range(200):
        start = int(dataset[i]["clean"][0, 0]) // GEOM.hop
        assert not (start < 1600 and 1000 < start + 64)


def test_validation_crops_are_fixed_across_epochs():
    track = fake_track()
    cfg = tf.DataCfg(crop_frames=64, val_crops_per_window=2)
    dataset = tf.ValPairDataset([track], cfg, GEOM)
    assert len(dataset) == 2
    first = [int(dataset[i]["clean"][0, 0]) for i in range(len(dataset))]
    second = [int(dataset[i]["clean"][0, 0]) for i in range(len(dataset))]
    assert first == second


def test_crop_samples_matches_the_stft_frame_count():
    for frames in (8, 64, 256):
        wav = torch.zeros(1, 1, GEOM.crop_samples(frames))
        assert tf.stft_c(wav, GEOM).shape[-1] == frames


def test_crest_db_is_zero_for_silence_and_positive_for_a_spike():
    assert tf._crest_db(torch.zeros(100)) == 0.0
    spike = torch.zeros(100)
    spike[0] = 1.0
    assert tf._crest_db(spike) == pytest.approx(20 * math.log10(10.0), rel=1e-3)
