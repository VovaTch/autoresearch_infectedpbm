"""
Style resolution, on the CPU with a synthetic corpus.

Two properties carry the whole feature. Every mode has to land on the unit
sphere, because that is where compute_style_windows puts the descriptors the
model was trained on and a vector of the wrong length is off-distribution before
its direction is even considered. And every random draw has to come from the
spec's own seed, because a saved recipe is otherwise unreproducible -- for the
"random" mode the recipe is all there is.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slice_synth.config import SynthConfig
from slice_synth.model.types import SAME_TRACK, PromptSpec, RenderSpec, StyleSpec
from slice_synth.worker.service import SynthService, slerp
from train_ar import TrackTokens

DIM = 16
WINDOW = 100
WINDOWS = 8


def make_track(track_idx: int, seed: int) -> TrackTokens:
    """
    Args:
      track_idx (int): id to give the track.
      seed (int): RNG seed for its synthetic content.

    Returns:
      TrackTokens: a tiny in-memory track with unit-norm style windows.
    """
    generator = torch.Generator().manual_seed(seed)
    style = torch.nn.functional.normalize(
        torch.randn(WINDOWS, DIM, generator=generator), dim=-1
    )
    bounds = torch.tensor([[w * WINDOW, (w + 1) * WINDOW] for w in range(WINDOWS)])
    return TrackTokens(
        tokens=torch.randint(0, 64, (WINDOWS * WINDOW, 3), generator=generator),
        style=style,
        style_bounds=bounds,
        track_idx=track_idx,
        track_name=f"track {track_idx}",
        num_frames=WINDOWS * WINDOW,
        fps=172.265625,
        val_windows=[],
    )


class StubModel:
    """
    Stands in for a LoadedModel without a checkpoint.

    Args:
      tracks (list[TrackTokens]): the synthetic corpus.
    """

    def __init__(self, tracks: list[TrackTokens]) -> None:
        self.tracks = tracks
        self.meta = {"frames_per_second": 172.265625, "num_tokens": 64, "num_rq": 3}
        self.cache_dir = type("P", (), {"name": "tokens_test"})()
        self.generator = type("G", (), {"depth": 3})()

    @property
    def by_idx(self) -> dict[int, TrackTokens]:
        return {t.track_idx: t for t in self.tracks}

    @property
    def fps(self) -> float:
        return 172.265625


@pytest.fixture
def service() -> SynthService:
    """
    Returns:
      SynthService: a service wired to a synthetic corpus, no GPU involved.
    """
    svc = SynthService(SynthConfig())
    svc._model = StubModel([make_track(0, 1), make_track(1, 2)])  # type: ignore[assignment]
    return svc


def spec(
    style: StyleSpec | None, track_idx: int | None = 0, start: int = 300
) -> RenderSpec:
    """
    Args:
      style (StyleSpec | None): the recipe under test.
      track_idx (int | None): the render's conditioning track.
      start (int): prompt start frame, which defines the disjointness span.

    Returns:
      RenderSpec: a render to resolve against.
    """
    return RenderSpec(
        track_idx=track_idx,
        style=style,
        n_frames=200,
        prompt=PromptSpec(kind="corpus", track_idx=0, start_frame=start),
    )


@pytest.mark.parametrize(
    "style",
    [
        StyleSpec(kind="window", track_idx=0, window=3),
        StyleSpec(kind="window", track_idx=0, window=-1),
        StyleSpec(kind="random", seed=7),
        StyleSpec(kind="jitter", track_idx=0, window=2, noise=0.5),
        StyleSpec(kind="interp", track_idx=0, track_b=1, mix=0.3),
    ],
)
def test_every_mode_lands_on_the_unit_sphere(
    service: SynthService, style: StyleSpec
) -> None:
    vector = service.resolve_style(spec(style))
    assert vector.shape == (1, DIM), "one segment when the style does not walk"
    assert abs(float(vector.norm()) - 1.0) < 1e-5


def test_a_nulled_stream_resolves_to_zeros(service: SynthService) -> None:
    vector = service.resolve_style(spec(None))
    assert vector.shape == (1, DIM) and float(vector.abs().sum()) == 0.0


def test_the_same_seed_gives_the_same_vector(service: SynthService) -> None:
    for kind in ("random", "jitter"):
        style = StyleSpec(kind=kind, track_idx=0, window=1, seed=11)  # type: ignore[arg-type]
        first = service.resolve_style(spec(style))
        second = service.resolve_style(spec(style))
        assert torch.equal(first, second)


def test_a_different_seed_gives_a_different_vector(service: SynthService) -> None:
    a = service.resolve_style(spec(StyleSpec(kind="random", seed=1)))
    b = service.resolve_style(spec(StyleSpec(kind="random", seed=2)))
    assert not torch.allclose(a, b)


def test_auto_window_avoids_the_generated_span(service: SynthService) -> None:
    # Section 11.3: a descriptor from the same window as the target is a
    # compressed copy of the answer. The span here is frames 300-500, windows 3-4.
    track = service.model.by_idx[0]
    forbidden = {3, 4}
    for seed in range(24):
        vector = service.resolve_style(spec(StyleSpec(window=-1, seed=seed)))
        matches = [
            w
            for w in range(WINDOWS)
            if torch.allclose(track.style[w], vector, atol=1e-6)
        ]
        assert matches and not (set(matches) & forbidden)


def test_same_track_follows_the_render(service: SynthService) -> None:
    style = StyleSpec(track_idx=SAME_TRACK, window=0)
    assert torch.equal(
        service.resolve_style(spec(style, track_idx=1))[0],
        service.model.by_idx[1].style[0],
    )
    assert torch.equal(
        service.resolve_style(spec(style, track_idx=0))[0],
        service.model.by_idx[0].style[0],
    )


def test_same_track_falls_back_to_the_prompt_when_the_id_is_nulled(
    service: SynthService,
) -> None:
    style = StyleSpec(track_idx=SAME_TRACK, window=0)
    assert torch.equal(
        service.resolve_style(spec(style, track_idx=None))[0],
        service.model.by_idx[0].style[0],
    )


def test_jitter_strength_moves_the_vector_off_its_base(service: SynthService) -> None:
    base = service.resolve_style(spec(StyleSpec(kind="window", window=1)))[0]
    gentle = service.resolve_style(
        spec(StyleSpec(kind="jitter", window=1, noise=0.05))
    )[0]
    wild = service.resolve_style(spec(StyleSpec(kind="jitter", window=1, noise=1.5)))[0]
    assert float(base @ gentle) > float(base @ wild)
    assert float(base @ gentle) > 0.9


def test_interp_endpoints_are_the_windows_themselves(service: SynthService) -> None:
    a = service.model.by_idx[0].style[0]
    b = service.model.by_idx[1].style[0]
    at_a = service.resolve_style(
        spec(
            StyleSpec(
                kind="interp", track_idx=0, window=0, track_b=1, window_b=0, mix=0.0
            )
        )
    )
    at_b = service.resolve_style(
        spec(
            StyleSpec(
                kind="interp", track_idx=0, window=0, track_b=1, window_b=0, mix=1.0
            )
        )
    )
    assert torch.allclose(at_a, a, atol=1e-5)
    assert torch.allclose(at_b, b, atol=1e-5)


def test_slerp_stays_on_the_sphere_between_the_endpoints() -> None:
    generator = torch.Generator().manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(DIM, generator=generator), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(DIM, generator=generator), dim=-1)
    for mix in np.linspace(0.0, 1.0, 11):
        mid = slerp(a, b, float(mix))
        assert abs(float(mid.norm()) - 1.0) < 1e-5
    # a straight average would be shorter than either endpoint; slerp is not
    assert float(slerp(a, b, 0.5).norm()) > float((0.5 * (a + b)).norm())


def test_slerp_handles_parallel_vectors(service: SynthService) -> None:
    a = torch.nn.functional.normalize(torch.ones(DIM), dim=-1)
    assert torch.allclose(slerp(a, a.clone(), 0.5), a, atol=1e-6)


def test_unknown_track_is_reported_not_raised_blindly(service: SynthService) -> None:
    with pytest.raises(KeyError):
        service.resolve_style(spec(StyleSpec(track_idx=99, window=0)))


def test_chunks_group_by_length_and_count_guided_lanes_twice() -> None:
    cfg = SynthConfig()
    cfg.generator.max_batch = 4
    svc = SynthService(cfg)
    guided = [
        RenderSpec(style=StyleSpec(seed=i), cfg_strength=2.0, seed=i, n_frames=100)
        for i in range(4)
    ]
    plain = [
        RenderSpec(style=StyleSpec(seed=i), cfg_strength=0.0, seed=i, n_frames=200)
        for i in range(4)
    ]
    passes = svc._chunks(guided + plain)
    assert sorted(len(p) for p in passes) == [2, 2, 4]
    for group in passes:
        assert len({s.n_frames for s in group}) == 1


# -- walking styles ----------------------------------------------------------


@pytest.mark.parametrize("walk", ["windows", "random"])
def test_a_walking_style_yields_one_unit_row_per_segment(
    service: SynthService, walk: str
) -> None:
    style = StyleSpec(kind="window", track_idx=0, window=3, walk=walk, period=50)  # type: ignore[arg-type]
    rows = service.resolve_style(spec(style))
    # 200 frames + depth-1 = 202 positions, 50 per segment -> 5 segments
    assert rows.shape == (5, DIM)
    assert torch.allclose(rows.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(
        rows[0], service.model.by_idx[0].style[3]
    ), "segment 0 is the entry itself"
    assert not torch.allclose(rows[0], rows[1])


def test_a_walk_replays_from_its_seed(service: SynthService) -> None:
    style = StyleSpec(kind="random", seed=3, walk="random", period=64)
    assert torch.equal(
        service.resolve_style(spec(style)), service.resolve_style(spec(style))
    )
    other = StyleSpec(kind="random", seed=4, walk="random", period=64)
    assert not torch.equal(
        service.resolve_style(spec(style))[1:], service.resolve_style(spec(other))[1:]
    )


def test_a_window_walk_draws_from_the_renders_own_tracks(service: SynthService) -> None:
    style = StyleSpec(
        kind="window", track_idx=SAME_TRACK, window=0, walk="windows", period=10
    )
    rows = service.resolve_style(spec(style, track_idx=1))
    corpus = service.model.by_idx[1].style
    for row in rows:
        assert any(
            torch.allclose(row, w) for w in corpus
        ), "every segment is a real window of track 1"

    both = RenderSpec(
        track_idx=0,
        co_tracks=(1,),
        style=style,
        n_frames=200,
        prompt=PromptSpec(kind="corpus", track_idx=0, start_frame=300),
    )
    rows = service.resolve_style(both)
    from_0 = sum(
        any(torch.allclose(r, w) for w in service.model.by_idx[0].style) for r in rows
    )
    from_1 = sum(
        any(torch.allclose(r, w) for w in service.model.by_idx[1].style) for r in rows
    )
    assert from_0 and from_1, "with two ids the walk visits both tracks"


def test_a_walk_needs_a_positive_period(service: SynthService) -> None:
    style = StyleSpec(kind="window", track_idx=0, window=3, walk="random", period=0)
    assert service.resolve_style(spec(style)).shape == (1, DIM)


# -- the style model's walk --------------------------------------------------


class StubStyleAr:
    """
    Stands in for StyleArSampler: unit rows from the generator, calls recorded.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sample(
        self,
        prefix: torch.Tensor | None,
        track_idx: int | None,
        steps: int,
        temperature: float,
        cfg: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "prefix": None if prefix is None else prefix.clone(),
                "track_idx": track_idx,
                "steps": steps,
                "temperature": temperature,
                "cfg": cfg,
            }
        )
        return torch.nn.functional.normalize(
            torch.randn(steps, DIM, generator=generator), dim=-1
        )


@pytest.fixture
def ar_service(service: SynthService) -> tuple[SynthService, StubStyleAr]:
    stub = StubStyleAr()
    service._style_ar = stub  # type: ignore[assignment]
    return service, stub


def test_an_ar_walk_opens_with_the_real_window_and_continues(
    ar_service: tuple[SynthService, StubStyleAr],
) -> None:
    service, stub = ar_service
    style = StyleSpec(kind="window", track_idx=0, window=5, walk="ar", period=50)
    rows = service.resolve_style(spec(style))
    assert rows.shape == (5, DIM)
    assert torch.allclose(rows.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.equal(rows[0], service.model.by_idx[0].style[5])
    call = stub.calls[-1]
    assert call["steps"] == 4 and call["track_idx"] == 0
    # ar_prefix 4 -> windows 2..5 of the base track, the last being segment 0
    assert torch.equal(call["prefix"], service.model.by_idx[0].style[2:6])


def test_the_prefix_clamps_at_the_start_of_the_track(
    ar_service: tuple[SynthService, StubStyleAr],
) -> None:
    service, stub = ar_service
    style = StyleSpec(kind="window", track_idx=0, window=1, walk="ar", period=50)
    service.resolve_style(spec(style))
    assert torch.equal(stub.calls[-1]["prefix"], service.model.by_idx[0].style[0:2])


def test_a_cold_ar_walk_samples_every_segment(
    ar_service: tuple[SynthService, StubStyleAr],
) -> None:
    service, stub = ar_service
    style = StyleSpec(
        kind="window", track_idx=0, window=5, walk="ar", period=50, ar_prefix=0
    )
    rows = service.resolve_style(spec(style))
    assert rows.shape == (5, DIM)
    assert stub.calls[-1]["prefix"] is None and stub.calls[-1]["steps"] == 5
    assert not torch.equal(rows[0], service.model.by_idx[0].style[5])


def test_an_ar_walk_replays_from_its_seed(
    ar_service: tuple[SynthService, StubStyleAr],
) -> None:
    service, _ = ar_service
    style = StyleSpec(
        kind="window", track_idx=0, window=3, walk="ar", period=50, seed=2
    )
    assert torch.equal(
        service.resolve_style(spec(style)), service.resolve_style(spec(style))
    )
    other = StyleSpec(
        kind="window", track_idx=0, window=3, walk="ar", period=50, seed=3
    )
    assert not torch.equal(
        service.resolve_style(spec(style))[1:], service.resolve_style(spec(other))[1:]
    )


def test_same_track_ar_walk_follows_the_render(
    ar_service: tuple[SynthService, StubStyleAr],
) -> None:
    service, stub = ar_service
    style = StyleSpec(
        kind="window", track_idx=SAME_TRACK, window=0, walk="ar", period=50
    )
    rows = service.resolve_style(spec(style, track_idx=1))
    assert torch.equal(rows[0], service.model.by_idx[1].style[0])
    assert stub.calls[-1]["track_idx"] == 1
    assert torch.equal(stub.calls[-1]["prefix"], service.model.by_idx[1].style[0:1])


def test_non_window_kinds_feed_only_their_own_vector(
    ar_service: tuple[SynthService, StubStyleAr],
) -> None:
    service, stub = ar_service
    style = StyleSpec(
        kind="random", seed=7, walk="ar", period=50, ar_temperature=0.5, ar_cfg=2.0
    )
    rows = service.resolve_style(spec(style, track_idx=None))
    call = stub.calls[-1]
    assert call["prefix"].shape == (1, DIM) and torch.equal(call["prefix"][0], rows[0])
    assert call["track_idx"] is None
    assert call["temperature"] == 0.5 and call["cfg"] == 2.0


def test_a_single_segment_never_calls_the_model(
    ar_service: tuple[SynthService, StubStyleAr],
) -> None:
    service, stub = ar_service
    style = StyleSpec(kind="window", track_idx=0, window=5, walk="ar", period=500)
    assert service.resolve_style(spec(style)).shape == (1, DIM)
    assert not stub.calls


def test_a_missing_style_model_is_a_clear_error(service: SynthService) -> None:
    service.cfg.generator.style_ar_checkpoint = "auto"
    style = StyleSpec(kind="window", track_idx=0, window=5, walk="ar", period=50)
    with pytest.raises(FileNotFoundError, match="train_style_ar"):
        service.resolve_style(spec(style))
