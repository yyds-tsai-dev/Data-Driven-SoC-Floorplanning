"""`PARTNER_GPU_ARM=1` -> the idle accelerator becomes a phase-B candidate wave.

The fork is CPU-bound after the first sampling wave: the GPU draws the
direct/flow batch in ~0.1-0.3 s at the head of a case and then idles for the
whole SA/refine span.  The arm spends that idle window by

  1. firing a SECOND sampling wave *during* phase A (the latency is masked by
     the already-dispatched workers, so unlike the 0723 "sample at the head"
     variant it never delays the start of the case), and
  2. handing those candidates to the phase-B round on the pool slots phase B
     has always left idle (8 winner variants on a pool of up to 24).

Three properties carry the design and are what this file pins:

  * **off is byte-identical** -- the sampler is called exactly once, no phase-B
    carve is taken, and no arm state is written;
  * **the carve is earned, not assumed** -- the arm refuses it unless the
    MEASURED wave-1 latency says both waves fit and phase A still keeps
    `PARTNER_GPU_ARM_MIN_A` of the budget as real SA time (the 0723 `dmoff`
    carve paid -0.010 for nothing, and that is the failure mode here);
  * **every GPU failure degrades to the plain variant round** -- an exception,
    a legacy one-argument sampler, or a wave that would overrun phase A all
    return `[]` rather than harm the case.

See docs/design/2026-08-05-gpu-arm-phase-b.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as lg      # noqa: E402

_ENV = ("PARTNER_GPU_ARM", "PARTNER_GPU_ARM_TARGET", "PARTNER_GPU_ARM_MIN_A",
        "PARTNER_GPU_ARM_K", "PARTNER_GPU_ARM_TS0", "PARTNER_GPU_ARM_SEED",
        "PARTNER_GPU_ARM_DEBUG", "PARTNER_PHASE_B",
        "PARTNER_PHASE_B_MIN_BUDGET", "PARTNER_REFINE_KERNEL",
        "PARTNER_FUSION", "PARTNER_NREF", "PARTNER_NREF_MIN_N",
        "PARTNER_PERIMETER", "PARTNER_PERIMETER_COL", "PARTNER_POOL_GATE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    lg._GPU_ARM_TS.clear()
    yield
    lg._GPU_ARM_TS.clear()


# ==========================================================================
# 1. env wiring -- off is off
# ==========================================================================
def test_default_is_off():
    assert lg.gpu_arm_on() is False
    assert lg._GPU_ARM_TS == {}


@pytest.mark.parametrize("value", ["1", "on", "true", "True", "ON"])
def test_flag_spellings(monkeypatch, value):
    monkeypatch.setenv("PARTNER_GPU_ARM", value)
    assert lg.gpu_arm_on() is True


@pytest.mark.parametrize("value", ["0", "", "no", "off"])
def test_flag_negative_spellings(monkeypatch, value):
    monkeypatch.setenv("PARTNER_GPU_ARM", value)
    assert lg.gpu_arm_on() is False


# ==========================================================================
# 2. the phase-B floor is adaptive to the refine kernel, never to the arm
#    being off
# ==========================================================================
def test_phase_b_floor_is_unchanged_with_the_arm_off(monkeypatch):
    """`PARTNER_PHASE_B` is a pre-existing lever; the arm must not move its
    8 s floor (which is what has kept it unreachable at both promoted tiers)
    when the arm is not the caller."""
    assert lg.phase_b_min_budget_default(False) == 8.0
    monkeypatch.setenv("PARTNER_REFINE_KERNEL", "numba")
    assert lg.phase_b_min_budget_default(False) == 8.0


def test_phase_b_floor_drops_under_the_refine_kernel(monkeypatch):
    """A phase-B slice must hold at least one refine rung.  RK cut a rung
    from 0.36-0.8 s to ~0.035 s, so the floor can drop to a 0.4 s case."""
    assert lg.phase_b_min_budget_default(True) == 1.5
    monkeypatch.setenv("PARTNER_REFINE_KERNEL", "numba")
    assert lg.phase_b_min_budget_default(True) == 0.35


# ==========================================================================
# 3. slice / K formulas at both operating points
# ==========================================================================
def test_slice_at_the_promoted_tier():
    """BUDGET_MAX=3.5 -> ~3.2 s worker span: the absolute 0.80 s target wins
    (25 % of the budget, ~20 RK rungs)."""
    assert lg.gpu_arm_slice(3.2) == pytest.approx(0.80)


def test_slice_at_the_goal_tier():
    """tau12 tail ~0.44 s: the 35 % ceiling wins (~0.154 s, ~4 RK rungs)."""
    assert lg.gpu_arm_slice(0.44) == pytest.approx(0.154)


def test_slice_never_exceeds_a_third_of_the_budget():
    for rem in (0.2, 0.44, 1.0, 2.0, 3.2, 8.0, 24.0):
        s = lg.gpu_arm_slice(rem)
        assert 0.0 <= s <= 0.35 * rem + 1e-12
        if rem >= 0.1:
            assert s >= min(0.20 * rem, 0.35 * rem) - 1e-12


def test_slice_floor_on_long_budgets():
    """On a long budget the absolute target must not collapse the slice."""
    assert lg.gpu_arm_slice(24.0) == pytest.approx(0.20 * 24.0)


def test_slice_target_is_tunable(monkeypatch):
    monkeypatch.setenv("PARTNER_GPU_ARM_TARGET", "0.40")
    assert lg.gpu_arm_slice(3.2) == pytest.approx(0.64)   # 20 % floor binds


def test_wave2_k_scales_with_the_slice_and_clamps():
    assert lg.gpu_arm_wave2_k(0.154, 24) == 4       # goal tier
    assert lg.gpu_arm_wave2_k(0.80, 24) == 12       # promoted tier, K cap
    assert lg.gpu_arm_wave2_k(0.80, 3) == 3         # pool-size limited
    assert lg.gpu_arm_wave2_k(0.80, 0) == 0
    assert lg.gpu_arm_wave2_k(0.80, -5) == 0
    assert lg.gpu_arm_wave2_k(0.0, 24) == 2         # floor


def test_wave2_k_cap_is_tunable(monkeypatch):
    monkeypatch.setenv("PARTNER_GPU_ARM_K", "5")
    assert lg.gpu_arm_wave2_k(0.80, 24) == 5


def test_phase_b_split_never_displaces_variants_on_a_real_pool():
    """PARTNER_POOL is 24-46 on the contest hardware, and phase B has always
    dispatched only 8 payloads onto it -- the GPU wave rides the idle slots,
    so the winner-variant round is untouched.  This is the difference from
    PARTNER_NREF, which was convicted at the goal tier for eating column
    breadth."""
    for pool in (24, 32, 46):
        k2, n_var = lg.gpu_arm_phase_b_split(0.80, pool)
        assert n_var == lg._PHASE_B_VARIANTS
        assert k2 == 12 and k2 + n_var <= pool
    k2, n_var = lg.gpu_arm_phase_b_split(0.154, 24)
    assert (k2, n_var) == (4, lg._PHASE_B_VARIANTS)


def test_phase_b_split_keeps_half_the_pool_for_the_incumbent():
    """When the pool cannot hold both rounds, exploitation of the known-good
    layout keeps at least half of it."""
    for pool in (2, 4, 8, 10):
        k2, n_var = lg.gpu_arm_phase_b_split(0.80, pool)
        assert k2 + n_var <= pool
        assert n_var >= 1
        assert k2 <= pool // 2
    assert lg.gpu_arm_phase_b_split(0.80, 0) == (0, 1)


def test_malformed_env_falls_back(monkeypatch):
    monkeypatch.setenv("PARTNER_GPU_ARM_TARGET", "bogus")
    monkeypatch.setenv("PARTNER_GPU_ARM_K", "bogus")
    monkeypatch.setenv("PARTNER_GPU_ARM_TS0", "bogus")
    assert lg.gpu_arm_slice(3.2) == pytest.approx(0.80)
    assert lg.gpu_arm_wave2_k(0.80, 16) == 12
    assert lg.gpu_arm_sample_estimate(100) == pytest.approx(0.25)


# ==========================================================================
# 4. sampler-latency estimate: a reusable instance statistic (block-count
#    decade), never a case id
# ==========================================================================
def test_estimate_starts_from_the_prior(monkeypatch):
    assert lg.gpu_arm_sample_estimate(97) == pytest.approx(0.25)
    monkeypatch.setenv("PARTNER_GPU_ARM_TS0", "0.4")
    assert lg.gpu_arm_sample_estimate(97) == pytest.approx(0.4)


def test_estimate_is_an_ema_per_block_decade():
    lg.gpu_arm_record_sample(97, 0.10)
    assert lg.gpu_arm_sample_estimate(97) == pytest.approx(0.10)
    assert lg.gpu_arm_sample_estimate(95) == pytest.approx(0.10)   # same band
    assert lg.gpu_arm_sample_estimate(45) == pytest.approx(0.25)   # other band
    lg.gpu_arm_record_sample(93, 0.20)
    assert lg.gpu_arm_sample_estimate(97) == pytest.approx(0.15)


def test_estimate_ignores_nonsense():
    lg.gpu_arm_record_sample(97, float("nan"))
    lg.gpu_arm_record_sample(97, -1.0)
    assert lg._GPU_ARM_TS == {}


# ==========================================================================
# 5. the second wave: containment is the contract
# ==========================================================================
def _sampler(calls, k_out=None):
    def fn(K, gen_seed=0):
        calls.append((K, gen_seed))
        n = K if k_out is None else k_out
        return [np.full((3, 4), float(gen_seed + i)) for i in range(n)]
    return fn


def test_second_wave_returns_candidates_when_it_fits():
    calls = []
    out = lg.gpu_arm_second_wave(_sampler(calls), 4, 8117, 0.05,
                                 time.time() + 5.0)
    assert len(out) == 4
    assert calls == [(4, 8117)]
    assert all(isinstance(P, np.ndarray) and P.dtype == np.float64
               for P in out)


def test_second_wave_declines_when_it_would_overrun_phase_a():
    """The wave costs about what wave 1 cost; if that does not land before
    `deadline_A` it must not be started at all."""
    calls = []
    out = lg.gpu_arm_second_wave(_sampler(calls), 4, 8117, 0.50,
                                 time.time() + 0.20)
    assert out == []
    assert calls == []


def test_second_wave_declines_a_legacy_one_argument_sampler():
    """A sampler with a pinned generator seed would re-draw wave 1 verbatim
    -- burning the carve for a duplicate batch."""
    calls = []

    def legacy(K):
        calls.append(K)
        return [np.zeros((3, 4))]

    assert lg.gpu_arm_sampler_takes_seed(legacy) is False
    assert lg.gpu_arm_second_wave(legacy, 4, 8117, 0.01,
                                  time.time() + 5.0) == []
    assert calls == []


def test_second_wave_accepts_kwargs_samplers():
    def anything(K, **kw):
        return [np.zeros((3, 4))]

    assert lg.gpu_arm_sampler_takes_seed(anything) is True
    assert lg.gpu_arm_sampler_takes_seed(None) is False


def test_second_wave_swallows_a_gpu_failure():
    def boom(K, gen_seed=0):
        raise RuntimeError("CUDA out of memory")

    assert lg.gpu_arm_second_wave(boom, 4, 8117, 0.01,
                                  time.time() + 5.0) == []


def test_second_wave_is_never_started_for_zero_candidates():
    calls = []
    assert lg.gpu_arm_second_wave(_sampler(calls), 0, 8117, 0.01,
                                  time.time() + 5.0) == []
    assert calls == []


def test_second_wave_truncates_an_overlong_batch():
    calls = []
    out = lg.gpu_arm_second_wave(_sampler(calls, k_out=9), 4, 8117, 0.01,
                                 time.time() + 5.0)
    assert len(out) == 4


# ==========================================================================
# 6. end to end through `_parallel_solve` on a real (tiny) worker pool
# ==========================================================================
def _instance(n: int = 8):
    import torch
    rects = [(2.0 * i, 0.0, 2.0, 2.0) for i in range(n)]
    at = torch.full((n,), 4.0)
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    b2b = torch.zeros((n, n))
    b2b[0, 1] = b2b[1, 0] = 1.0
    b2b[2, 3] = b2b[3, 2] = 1.0
    p2b = torch.zeros((2, n))
    pins = torch.zeros((2, 2))
    return rects, at, cons, tpos, b2b, p2b, pins


@pytest.fixture(scope="module")
def _pool():
    lg.init_worker_pool(2, warmup_timeout=60.0)
    if not lg._POOL_READY:
        pytest.skip("worker pool unavailable in this environment")
    yield
    lg._shutdown_pool()


def _run_parallel(budget: float, seed: int = 5):
    """Drive `_parallel_solve` with a sampler that records every call."""
    rects, at, cons, tpos, b2b, p2b, pins = _instance()
    deadline = time.time() + budget
    opt1 = lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                               deadline, seed=seed)
    calls = []

    def sample_fn(K, gen_seed=0):
        calls.append((K, gen_seed))
        # a legal-ish prediction: the real layout, jittered per draw
        base = np.asarray([list(r) for r in rects], dtype=np.float64)
        return [base + np.array([0.01 * (gen_seed + i), 0.0, 0.0, 0.0])
                for i in range(K)]

    out = lg._parallel_solve(opt1, rects, at, cons, tpos, b2b, p2b, pins,
                             deadline, seed, sample_fn=sample_fn)
    return out, calls


def _assert_no_overlap(pos, tol: float = 1e-7) -> None:
    pos = np.asarray(pos, dtype=np.float64)
    for i in range(len(pos)):
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, len(pos)):
            xj, yj, wj, hj = pos[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            assert not (ox > tol and oy > tol), f"blocks {i},{j} overlap"


def test_off_path_samples_exactly_once_and_takes_no_carve(_pool):
    """The load-bearing off-path claim: one sampling wave, no phase-B round,
    no arm state."""
    out, calls = _run_parallel(1.6)
    assert len(calls) == 1
    assert calls[0][1] == 0                 # baseline generator offset
    assert lg._GPU_ARM_TS == {}
    assert len(out) == 8
    _assert_no_overlap(out)


def test_arm_fires_a_second_wave_and_keeps_the_layout_legal(
        monkeypatch, _pool):
    monkeypatch.setenv("PARTNER_GPU_ARM", "1")
    monkeypatch.setenv("PARTNER_PHASE_B_MIN_BUDGET", "0.35")
    monkeypatch.setenv("PARTNER_GPU_ARM_TS0", "0.001")
    out, calls = _run_parallel(2.0)
    assert len(calls) == 2, calls
    assert calls[1][1] != 0                 # a genuinely different draw
    assert calls[1][0] > 0
    assert lg._GPU_ARM_TS                   # the wave-1 latency was recorded
    assert len(out) == 8
    _assert_no_overlap(out)


def test_arm_declines_the_carve_when_sampling_cannot_be_masked(
        monkeypatch, _pool):
    """A sampler estimated to cost more than phase A can hide must leave the
    budget alone -- this is the `dmoff` "carve for nothing" guard."""
    monkeypatch.setenv("PARTNER_GPU_ARM", "1")
    monkeypatch.setenv("PARTNER_PHASE_B_MIN_BUDGET", "0.35")
    monkeypatch.setenv("PARTNER_GPU_ARM_TS0", "5.0")
    out, calls = _run_parallel(1.6)
    assert len(calls) == 1
    _assert_no_overlap(out)


def test_arm_degrades_to_the_variant_round_when_the_gpu_fails(
        monkeypatch, _pool):
    """A second wave that raises must still leave a legal layout (and the
    phase-B round still runs on the winner variants)."""
    monkeypatch.setenv("PARTNER_GPU_ARM", "1")
    monkeypatch.setenv("PARTNER_PHASE_B_MIN_BUDGET", "0.35")
    monkeypatch.setenv("PARTNER_GPU_ARM_TS0", "0.001")
    rects, at, cons, tpos, b2b, p2b, pins = _instance()
    deadline = time.time() + 2.0
    opt1 = lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                               deadline, seed=5)
    calls = []

    def sample_fn(K, gen_seed=0):
        calls.append(gen_seed)
        if gen_seed:
            raise RuntimeError("CUDA out of memory")
        base = np.asarray([list(r) for r in rects], dtype=np.float64)
        return [base.copy() for _ in range(K)]

    out = lg._parallel_solve(opt1, rects, at, cons, tpos, b2b, p2b, pins,
                             deadline, 5, sample_fn=sample_fn)
    assert len(calls) == 2 and calls[1] != 0
    assert len(out) == 8
    _assert_no_overlap(out)


def test_arm_is_inert_without_a_sampler(monkeypatch, _pool):
    """No direct model (below `PARTNER_DIRECT_MIN`) -> nothing to draw, so no
    carve: the arm must not shorten phase A for a round it cannot feed."""
    monkeypatch.setenv("PARTNER_GPU_ARM", "1")
    monkeypatch.setenv("PARTNER_PHASE_B_MIN_BUDGET", "0.35")
    monkeypatch.setenv("PARTNER_GPU_ARM_TS0", "0.001")
    rects, at, cons, tpos, b2b, p2b, pins = _instance()
    deadline = time.time() + 1.6
    opt1 = lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                               deadline, seed=5)
    out = lg._parallel_solve(opt1, rects, at, cons, tpos, b2b, p2b, pins,
                             deadline, 5, sample_fn=None)
    assert lg._GPU_ARM_TS == {}
    _assert_no_overlap(out)


def test_arm_yields_to_fusion(monkeypatch, _pool):
    """`PARTNER_FUSION` consumes the same carve with a different round; the
    arm must not spend GPU time on candidates nobody will refine."""
    monkeypatch.setenv("PARTNER_GPU_ARM", "1")
    monkeypatch.setenv("PARTNER_PHASE_B_MIN_BUDGET", "0.35")
    monkeypatch.setenv("PARTNER_GPU_ARM_TS0", "0.001")
    monkeypatch.setenv("PARTNER_FUSION", "1")
    out, calls = _run_parallel(1.6)
    assert len(calls) == 1
    _assert_no_overlap(out)


def test_arm_composes_with_early_exit_and_the_anytime_ladder(
        monkeypatch, _pool):
    """The arm only reshapes the time axis of `_parallel_solve`; the two
    other opt-in time levers must still produce a legal layout with it on."""
    monkeypatch.setenv("PARTNER_GPU_ARM", "1")
    monkeypatch.setenv("PARTNER_PHASE_B_MIN_BUDGET", "0.35")
    monkeypatch.setenv("PARTNER_GPU_ARM_TS0", "0.001")
    monkeypatch.setenv("PARTNER_EARLY_EXIT", "1")
    monkeypatch.setenv("PARTNER_ANYTIME_LADDER", "1")
    try:
        out, calls = _run_parallel(2.0)
    finally:
        monkeypatch.delenv("PARTNER_EARLY_EXIT", raising=False)
        monkeypatch.delenv("PARTNER_ANYTIME_LADDER", raising=False)
    assert len(calls) == 2
    assert len(out) == 8
    _assert_no_overlap(out)
