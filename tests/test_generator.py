"""
The KV cache and the sliding window.

This is the load-bearing file. Everything downstream assumes the fast sampler
produces exactly what the slow one in generate_ar produces; if that quietly
stopped being true, every clip in the bank would be from a model nobody
evaluated, and nothing else in the suite would catch it.

Tiny random models throughout, on the CPU, so the whole file runs in seconds.
"""

from __future__ import annotations

import pytest
import torch

from ab_harness.worker.generator import ArGenerator
from generate_ar import generate
from train_ar import PREFIX_POSITIONS, ArTransformer, ModelCfg

DEPTH = 3
VOCAB = 16
TRACKS = 5
STYLE_DIM = 12


@pytest.fixture
def model() -> ArTransformer:
    """
    Returns:
      ArTransformer: a small deterministic model, in eval mode.
    """
    torch.manual_seed(0)
    cfg = ModelCfg(d_model=32, n_layers=2, n_heads=4, dropout=0.0, style_bottleneck=8)
    return ArTransformer(
        cfg,
        num_tokens=VOCAB,
        num_rq=DEPTH,
        num_tracks=TRACKS,
        style_dim=STYLE_DIM,
        max_positions=256,
    ).eval()


@pytest.fixture
def cond() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      tuple: (track ids, styles, drop_id, drop_style) for one sequence.
    """
    torch.manual_seed(1)
    return (
        torch.tensor([2]),
        torch.randn(1, STYLE_DIM),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )


# -- cache parity ------------------------------------------------------------


def test_incremental_matches_the_full_forward(model: ArTransformer, cond) -> None:
    tokens = torch.randint(0, VOCAB, (1, 11, DEPTH))
    full = model(tokens, *cond)

    cache = model.start_incremental(*cond, max_length=PREFIX_POSITIONS + 11)
    stepped = torch.cat(
        [model.step_incremental(tokens[:, i : i + 1], cache) for i in range(11)], dim=1
    )
    assert torch.allclose(full, stepped, atol=1e-5)
    # what actually matters: the same tokens come out of an argmax
    assert torch.equal(full.argmax(-1), stepped.argmax(-1))


def test_a_chunked_prefill_matches_a_one_shot_forward(
    model: ArTransformer, cond
) -> None:
    tokens = torch.randint(0, VOCAB, (1, 11, DEPTH))
    full = model(tokens, *cond)
    cache = model.start_incremental(*cond, max_length=PREFIX_POSITIONS + 11)
    chunked = torch.cat(
        [
            model.step_incremental(tokens[:, :4], cache),
            model.step_incremental(tokens[:, 4:], cache),
        ],
        dim=1,
    )
    assert torch.allclose(full, chunked, atol=1e-5)


def test_the_uncached_path_is_untouched(model: ArTransformer, cond) -> None:
    # forward() must not depend on any cache state left over from sampling
    tokens = torch.randint(0, VOCAB, (1, 7, DEPTH))
    before = model(tokens, *cond)
    model.start_incremental(*cond, max_length=64)
    assert torch.equal(model(tokens, *cond), before)


def test_a_cache_reset_reuses_the_same_buffers(model: ArTransformer, cond) -> None:
    cache = model.start_incremental(*cond, max_length=32)
    first = cache.keys[0].data_ptr()
    model.step_incremental(torch.randint(0, VOCAB, (1, 5, DEPTH)), cache)
    again = model.start_incremental(*cond, max_length=32, cache=cache)
    assert again is cache
    assert cache.length == PREFIX_POSITIONS
    assert cache.keys[0].data_ptr() == first


def test_cache_overflow_raises_rather_than_corrupting(
    model: ArTransformer, cond
) -> None:
    cache = model.start_incremental(*cond, max_length=PREFIX_POSITIONS + 4)
    with pytest.raises(ValueError, match="overflow"):
        model.step_incremental(torch.randint(0, VOCAB, (1, 5, DEPTH)), cache)


# -- sampler parity ----------------------------------------------------------


def _style() -> torch.Tensor:
    """
    Returns:
      torch.Tensor: (STYLE_DIM,) a fixed style descriptor.
    """
    torch.manual_seed(1)
    return torch.randn(1, STYLE_DIM)[0]


@pytest.mark.parametrize("prompt_frames", [0, 4])
def test_greedy_sampling_matches_generate_ar(
    model: ArTransformer, prompt_frames: int
) -> None:
    device = torch.device("cpu")
    style, frames = _style(), 24
    prompt = torch.randint(0, VOCAB, (prompt_frames, DEPTH)) if prompt_frames else None

    reference = generate(model, 2, style, frames, prompt, 0.0, 0, 0.0, device)[0]
    fast = ArGenerator(model, device, window_frames=64).sample(
        2, style, True, True, frames, prompt, temperature=0.0
    )
    assert torch.equal(reference, fast)


def test_the_prompt_is_reproduced_exactly(model: ArTransformer) -> None:
    prompt = torch.randint(0, VOCAB, (6, DEPTH))
    out = ArGenerator(model, torch.device("cpu"), window_frames=64).sample(
        2, _style(), True, True, 20, prompt, temperature=0.0
    )
    assert torch.equal(out[:6], prompt)


def test_output_shape_and_dtype(model: ArTransformer) -> None:
    out = ArGenerator(model, torch.device("cpu"), window_frames=64).sample(
        2, _style(), True, True, 17, None, temperature=0.0
    )
    assert out.shape == (17, DEPTH)
    assert out.dtype == torch.int64
    assert int(out.max()) < VOCAB  # the pad id must never reach the output


def test_the_same_seed_gives_the_same_clip(model: ArTransformer) -> None:
    device = torch.device("cpu")

    def run() -> torch.Tensor:
        return ArGenerator(model, device, window_frames=64).sample(
            2, _style(), True, True, 20, None, temperature=1.0, seed=99
        )

    assert torch.equal(run(), run())


def test_different_seeds_give_different_clips(model: ArTransformer) -> None:
    device = torch.device("cpu")
    gen = ArGenerator(model, device, window_frames=64)
    a = gen.sample(2, _style(), True, True, 40, None, temperature=1.0, seed=1)
    b = gen.sample(2, _style(), True, True, 40, None, temperature=1.0, seed=2)
    assert not torch.equal(a, b)


# -- sliding window ----------------------------------------------------------


def test_a_window_wider_than_the_clip_never_reprimes(model: ArTransformer) -> None:
    device = torch.device("cpu")
    wide = ArGenerator(model, device, window_frames=200).sample(
        2, _style(), True, True, 24, None, temperature=0.0
    )
    reference = generate(model, 2, _style(), 24, None, 0.0, 0, 0.0, device)[0]
    assert torch.equal(wide, reference)


def test_a_clip_longer_than_the_window_still_produces_the_full_length(
    model: ArTransformer,
) -> None:
    out = ArGenerator(
        model, torch.device("cpu"), window_frames=12, reprime_frac=0.5
    ).sample(2, _style(), True, True, 60, None, temperature=0.0)
    assert out.shape == (60, DEPTH)
    assert int(out.max()) < VOCAB


def test_re_priming_changes_the_result_only_past_the_window(
    model: ArTransformer,
) -> None:
    device = torch.device("cpu")
    frames = 40
    full = ArGenerator(model, device, window_frames=200).sample(
        2, _style(), True, True, frames, None, temperature=0.0
    )
    windowed = ArGenerator(model, device, window_frames=16, reprime_frac=0.5).sample(
        2, _style(), True, True, frames, None, temperature=0.0
    )
    # identical until the first re-prime, and free to diverge after it
    assert torch.equal(full[:16], windowed[:16])


# -- conditioning ------------------------------------------------------------


def test_guidance_strength_zero_is_plain_conditional_sampling(
    model: ArTransformer,
) -> None:
    device = torch.device("cpu")
    plain = ArGenerator(model, device, window_frames=64).sample(
        2, _style(), True, True, 20, None, temperature=0.0, cfg_strength=0.0
    )
    reference = generate(model, 2, _style(), 20, None, 0.0, 0, 0.0, device)[0]
    assert torch.equal(plain, reference)


def test_guidance_changes_the_output(model: ArTransformer) -> None:
    device = torch.device("cpu")
    plain = ArGenerator(model, device, window_frames=64).sample(
        2, _style(), True, True, 40, None, temperature=0.0, cfg_strength=0.0
    )
    guided = ArGenerator(model, device, window_frames=64).sample(
        2, _style(), True, True, 40, None, temperature=0.0, cfg_strength=3.0
    )
    assert not torch.equal(plain, guided)


def test_nulling_both_streams_disables_guidance_instead_of_dividing_by_nothing(
    model: ArTransformer,
) -> None:
    device = torch.device("cpu")
    # with everything nulled there is no conditional to guide towards, so the
    # strength must be ignored rather than amplifying a zero delta
    a = ArGenerator(model, device, window_frames=64).sample(
        2, _style(), False, False, 20, None, temperature=0.0, cfg_strength=3.0
    )
    b = ArGenerator(model, device, window_frames=64).sample(
        2, _style(), False, False, 20, None, temperature=0.0, cfg_strength=0.0
    )
    assert torch.equal(a, b)


def test_nulling_the_track_id_changes_the_output(model: ArTransformer) -> None:
    device = torch.device("cpu")
    with_id = ArGenerator(model, device, window_frames=64).sample(
        2, _style(), True, True, 40, None, temperature=0.0
    )
    without = ArGenerator(model, device, window_frames=64).sample(
        2, _style(), False, True, 40, None, temperature=0.0
    )
    assert not torch.equal(with_id, without)


def test_folding_frames_into_positions_is_refused_by_the_incremental_path() -> None:
    cfg = ModelCfg(d_model=32, n_layers=2, n_heads=4, dropout=0.0, style_bottleneck=8)
    folded = ArTransformer(
        cfg, VOCAB, DEPTH, TRACKS, STYLE_DIM, frames_per_pos=2, max_positions=64
    ).eval()
    with pytest.raises(NotImplementedError):
        folded.start_incremental(
            torch.tensor([1]),
            torch.randn(1, STYLE_DIM),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            max_length=32,
        )


# -- batching ----------------------------------------------------------------


def test_a_batch_matches_the_same_clips_sampled_alone(model: ArTransformer) -> None:
    """
    The property batching has to preserve.

    Throughput is worth nothing if the clips a batch produces differ from what
    the same recipes produce alone -- the bank would then be full of audio from
    a model nobody actually evaluated.
    """
    from ab_harness.worker.generator import SampleRequest

    gen = ArGenerator(model, torch.device("cpu"), window_frames=64)
    requests = [
        SampleRequest(track_idx=t, style=_style(), frames=20, temperature=0.0, seed=t)
        for t in range(3)
    ]
    batched = gen.sample_batch(requests)
    alone = [gen.sample_batch([request])[0] for request in requests]
    assert all(torch.equal(a, b) for a, b in zip(batched, alone))


def test_a_batch_keeps_per_lane_seeds_independent(model: ArTransformer) -> None:
    from ab_harness.worker.generator import SampleRequest

    gen = ArGenerator(model, torch.device("cpu"), window_frames=64)
    style = _style()
    same = gen.sample_batch(
        [
            SampleRequest(track_idx=2, style=style, frames=30, seed=5),
            SampleRequest(track_idx=2, style=style, frames=30, seed=5),
            SampleRequest(track_idx=2, style=style, frames=30, seed=6),
        ]
    )
    assert torch.equal(same[0], same[1]), "identical seeds must give identical clips"
    assert not torch.equal(same[0], same[2])


def test_lanes_may_differ_in_length_conditioning_and_guidance(
    model: ArTransformer,
) -> None:
    from ab_harness.worker.generator import SampleRequest

    gen = ArGenerator(model, torch.device("cpu"), window_frames=64)
    prompt = torch.randint(0, VOCAB, (5, DEPTH))
    out = gen.sample_batch(
        [
            SampleRequest(
                track_idx=1, style=_style(), frames=12, temperature=0.0, seed=1
            ),
            SampleRequest(
                track_idx=3,
                style=_style(),
                frames=30,
                prompt=prompt,
                temperature=0.0,
                cfg_strength=2.0,
                seed=2,
            ),
            SampleRequest(
                track_idx=0,
                style=_style(),
                use_track_id=False,
                use_style=False,
                frames=21,
                temperature=0.0,
                seed=3,
            ),
        ]
    )
    assert [t.shape[0] for t in out] == [12, 30, 21]
    assert torch.equal(out[1][:5], prompt)
    assert all(int(t.max()) < VOCAB for t in out)


def test_an_empty_batch_is_allowed(model: ArTransformer) -> None:
    assert ArGenerator(model, torch.device("cpu")).sample_batch([]) == []


def test_a_batch_longer_than_the_window_re_primes_every_lane(
    model: ArTransformer,
) -> None:
    from ab_harness.worker.generator import SampleRequest

    gen = ArGenerator(model, torch.device("cpu"), window_frames=12, reprime_frac=0.5)
    out = gen.sample_batch(
        [
            SampleRequest(
                track_idx=1, style=_style(), frames=50, temperature=0.0, seed=1
            ),
            SampleRequest(
                track_idx=2, style=_style(), frames=50, temperature=0.0, seed=2
            ),
        ]
    )
    assert all(t.shape == (50, DEPTH) for t in out)
    assert not torch.equal(out[0], out[1])


def test_no_pad_id_leaks_out_after_a_prompt(model: ArTransformer) -> None:
    """
    Regression: build_delay_grid pads the corners of the delayed grid, and
    forcing the trailing corner wrote pad_id -- one past the last codebook
    entry -- into the depth-1 frames straight after every prompt. The decoder
    gathered that out of range, putting a burst of garbage at the prompt seam,
    which is exactly the kind of artifact a fidelity A/B would pick up on.
    """
    device = torch.device("cpu")
    prompt = torch.randint(0, VOCAB, (5, DEPTH))
    out = ArGenerator(model, device, window_frames=64).sample(
        2, _style(), True, True, 20, prompt, temperature=0.0
    )
    assert int(out.max()) < VOCAB, "pad_id reached the output"
    assert torch.equal(out[:5], prompt)

    # generate_ar carried the same bug and is fixed alongside
    reference = generate(model, 2, _style(), 20, prompt, 0.0, 0, 0.0, device)[0]
    assert int(reference.max()) < VOCAB
    assert torch.equal(reference, out)


@pytest.mark.parametrize("prompt_frames", [1, 2, 3, 7])
def test_prompts_shorter_than_the_rvq_depth_are_still_clean(
    model: ArTransformer, prompt_frames: int
) -> None:
    prompt = torch.randint(0, VOCAB, (prompt_frames, DEPTH))
    out = ArGenerator(model, torch.device("cpu"), window_frames=64).sample(
        2, _style(), True, True, 16, prompt, temperature=0.0
    )
    assert int(out.max()) < VOCAB
    assert torch.equal(out[:prompt_frames], prompt)
