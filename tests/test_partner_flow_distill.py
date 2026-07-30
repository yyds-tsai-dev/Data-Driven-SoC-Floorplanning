"""Gate-1 invariants for the SCFM few-step distillation trainer.

These lock the parts of ``flow_matching_distill`` that are easy to get silently
wrong: the paper's time convention is the mirror of this repo's, so the
consistency window and the interval-weighted target are both stated in flipped
coordinates relative to arXiv:2510.17858.
"""

import pytest
import torch

from flow_matching_distill import (apply_shift, default_max_skip, sample_windows,
                                 scfm_target, skip_ladder)

GRID = 32
CPU = torch.device("cpu")


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(20260729)


def test_skip_ladder_is_powers_of_two_up_to_max():
    assert skip_ladder(GRID, 8) == [2, 4, 8]
    assert skip_ladder(GRID, 16) == [2, 4, 8, 16]
    assert skip_ladder(16, 8) == [2, 4, 8]


@pytest.mark.parametrize("grid,max_skip", [(6, 2), (32, 1), (32, 32), (0, 2)])
def test_skip_ladder_rejects_unusable_grids(grid, max_skip):
    with pytest.raises(ValueError):
        skip_ladder(grid, max_skip)


def test_one_step_target_gets_the_wider_ladder():
    # DEVIATION D3: grid/2 supervises the sampler's own entry-point window,
    # which the paper's grid/4 cap never reaches.
    assert default_max_skip(GRID, 1) == GRID // 2
    assert default_max_skip(GRID, 2) == GRID // 4
    assert default_max_skip(GRID, 4) == GRID // 4


@pytest.mark.parametrize("max_skip", [GRID // 4, GRID // 2])
@pytest.mark.parametrize("jitter", [False, True])
@pytest.mark.parametrize("shift", [(1.0, 1.0), (2.5, 4.5)])
def test_windows_stay_ordered_and_inside_the_path(rng, max_skip, jitter, shift):
    ladder = skip_ladder(GRID, max_skip)
    w = sample_windows(4096, GRID, ladder, 0.4, jitter, shift, CPU, rng)
    t1, t2, t3 = w["t1"], w["t2"], w["t3"]
    assert t1.min() >= -1e-6
    assert t3.max() <= 1.0 + 1e-6          # a window may never leave the path
    assert (t2 > t1).all() and (t3 > t2).all()
    assert (w["skip"][w["is_teacher"]] == 1).all()
    assert set(w["skip"][~w["is_teacher"]].unique().tolist()) <= set(ladder)


def test_teacher_slice_matches_the_requested_fraction(rng):
    w = sample_windows(20000, GRID, skip_ladder(GRID, 8), 0.4, True, (1.0, 1.0), CPU, rng)
    assert w["is_teacher"].float().mean().item() == pytest.approx(0.4, abs=0.02)


@pytest.mark.parametrize("frac", [0.0, 1.0])
def test_degenerate_teacher_fractions(rng, frac):
    w = sample_windows(512, GRID, skip_ladder(GRID, 8), frac, True, (1.0, 1.0), CPU, rng)
    assert bool(w["is_teacher"].all()) is (frac == 1.0)
    assert bool(w["is_teacher"].any()) is (frac == 1.0)


def test_shift_is_monotone_endpoint_fixing_and_noise_concentrating():
    t = torch.linspace(0, 1, 101)
    shifted = apply_shift(t, torch.full_like(t, 3.5))
    assert shifted[0].item() == pytest.approx(0.0, abs=1e-6)
    assert shifted[-1].item() == pytest.approx(1.0, abs=1e-6)
    assert (shifted.diff() > 0).all()
    # in this repo t=0 is noise, so concentrating steps near noise pulls t down
    assert (shifted[1:-1] < t[1:-1]).all()
    assert torch.allclose(apply_shift(t, torch.ones_like(t)), t, atol=1e-6)


def test_straight_field_is_an_exact_fixed_point(rng):
    """A perfectly rectified flow must be unchanged by the distillation target.

    This is the property the whole method rests on: the loss can only move a
    field that is *not* already step-size independent.
    """
    v = torch.randn(8, 20, 4, generator=rng)
    for d1, d2 in [(0.03, 0.03), (0.25, 0.25), (0.11, 0.40)]:
        target = scfm_target(v, v, torch.full((8,), d1), torch.full((8,), d2))
        assert torch.allclose(target, v, atol=1e-6)


def test_target_reproduces_the_two_substep_euler_composition(rng):
    """Paper Eq. 11 in this repo's forward-time convention.

    One jump of length ``d1 + d2`` taken with the target velocity must land
    exactly where two consecutive Euler sub-steps land.
    """
    v_near = torch.randn(8, 20, 4, generator=rng)
    v_far = torch.randn(8, 20, 4, generator=rng)
    d1 = torch.rand(8, generator=rng) * 0.2 + 0.01
    d2 = torch.rand(8, generator=rng) * 0.2 + 0.01
    z1 = torch.randn(8, 20, 4, generator=rng)

    z2 = z1 + d1.view(-1, 1, 1) * v_near
    two_steps = z2 + d2.view(-1, 1, 1) * v_far
    one_step = z1 + (d1 + d2).view(-1, 1, 1) * scfm_target(v_near, v_far, d1, d2)

    assert torch.allclose(two_steps, one_step, atol=1e-5)


def test_target_degenerates_to_each_endpoint(rng):
    v_near = torch.randn(4, 6, 4, generator=rng)
    v_far = torch.randn(4, 6, 4, generator=rng)
    tiny, big = torch.full((4,), 1e-6), torch.full((4,), 1.0)
    assert torch.allclose(scfm_target(v_near, v_far, big, tiny), v_near, atol=1e-5)
    assert torch.allclose(scfm_target(v_near, v_far, tiny, big), v_far, atol=1e-5)


def test_distilled_tag_is_loadable_by_the_flow_checkpoint_guard():
    """A distilled checkpoint must pass the shared production/probe guard."""
    from flow_matching_distill import TRAINING_METHOD
    from flow_matching_train import FLOW_METHODS, checkpoint_method

    assert TRAINING_METHOD in FLOW_METHODS
    assert checkpoint_method({"args": {"training_method": TRAINING_METHOD}}) == TRAINING_METHOD
