"""`PARTNER_REFINE_KERNEL_QSWEEP=1`: the `_axis_pass(hold=False)` port.

`PARTNER_REFINE_KERNEL=numba` moved the *legalization* half of the refiner
(`hold=True`, `_evict`, the overlap tests) into numba; this sub-switch moves
the *quality* half -- the HPWL weighted-median sweep `_Refiner.run` spends its
rounds on.  Both halves now share one `_axis_constraints` transcription
(`_axis_build_k`), so the two sweeps provably optimize over the same polytope.

The contract is the same as the parent flag's -- **bit equality**, `==` and not
`allclose` -- but its status is different, and that difference is the whole
reason this file exists.  For the `hold=True` port bit equality is a theorem
(every tie-break was transcribed).  Here one step is not: `_wmedian` calls
`np.argsort(vals)`, numpy's *default* introsort, whose permutation among equal
values the kernel does not reproduce (it sorts stably instead).  The module
docstring of `src/solver/refine_numeric_kernel.py` carries the argument that this
is value-preserving in exact arithmetic -- `v = vals[order]` is unique, and a
re-ordering inside a tie block cannot move `searchsorted`'s answer out of that
block -- with one residual: `cumsum` re-associates, so `c[-1]` (hence `half`)
can shift by an ULP.

So bit equality here is *measured*:

  * `test_wmedian_*` hammers `_wmedian_k` against `_Refiner._wmedian` on
    tie-saturated inputs, which is where the permutation can differ at all;
  * `test_axis_pass_soft_bit_exact*` compares whole sweeps, where any ULP
    divergence would be amplified by the Gauss-Seidel feedback;
  * `test_refine_prediction_bit_exact_*` compares whole `refine_prediction`
    outputs by hash, off vs on.

Default OFF, and gated under the parent flag: with `PARTNER_REFINE_KERNEL`
unset the sub-switch does nothing at all, and with the parent on but the
sub-switch off `RefineKernel.qsweep` is False, the median-sweep CSR is never
built, and `_axis_pass(hold=False)` runs the Python path it shipped with.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout_refiner as rf  # noqa: E402
import refine_numeric_kernel as rnk  # noqa: E402
from synth_instances import build_instance, make_optimizer  # noqa: E402

pytestmark = pytest.mark.skipif(
    not rnk.NUMBA_AVAILABLE, reason="numba not installed")

_ENV = ("PARTNER_REFINE_KERNEL", "PARTNER_REFINE_KERNEL_QSWEEP",
        "PARTNER_REFINE_KERNEL_WARMUP", "PARTNER_REFINE_FASTBUILD",
        "PARTNER_ANYTIME_LADDER", "PARTNER_EARLY_EXIT",
        "PARTNER_REFINE_STALL_STOP", "PARTNER_REFINE_PROF",
        "PARTNER_REFINE_PROF_FILE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


CASES = [
    dict(n=100, seed=0),
    dict(n=40, seed=3),
    dict(n=120, seed=7),
    dict(n=64, seed=11, n_preplaced=6, n_clusters=10, n_mib=5),
    dict(n=30, seed=13, n_preplaced=0, frac_fixed=0.0),   # no obstacles
    dict(n=50, seed=17, frac_boundary=0.45),              # tag-saturated
    dict(n=21, seed=19, n_clusters=1, n_mib=1),           # minimum blocks
    dict(n=80, seed=53, n_preplaced=5, n_clusters=8, anchor_clusters=4),
    dict(n=110, seed=59, n_preplaced=8, n_clusters=12, anchor_clusters=8,
         frac_boundary=0.30),
]
_IDS = [f"n{c['n']}s{c['seed']}" for c in CASES]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _refiner(case, kernel, monkeypatch, expand=0.0, span=1000.0, qsweep=True):
    inst = build_instance(**case)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    opt = make_optimizer(inst, seed=3, deadline=time.time() + span)
    if kernel:
        monkeypatch.setenv("PARTNER_REFINE_KERNEL", "numba")
        if qsweep:
            monkeypatch.setenv("PARTNER_REFINE_KERNEL_QSWEEP", "1")
    r = rf._Refiner(opt, pred, seed=11)
    monkeypatch.delenv("PARTNER_REFINE_KERNEL", raising=False)
    monkeypatch.delenv("PARTNER_REFINE_KERNEL_QSWEEP", raising=False)
    if expand:
        r.xmax += (r.xmax - r.xmin) * expand
        r.ymax += (r.ymax - r.ymin) * expand
    return r


def _pair(case, monkeypatch, expand=0.0):
    """(pure-Python refiner, QSWEEP refiner) on the same instance."""
    ref = _refiner(case, False, monkeypatch, expand)
    ker = _refiner(case, True, monkeypatch, expand)
    assert ref._nk is None
    assert ker._nk is not None and ker._nk.qsweep is True
    assert np.array_equal(ref.P, ker.P)
    return ref, ker


def _digest(a) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------
# 1. the switch is a sub-switch, and it is off by default
# --------------------------------------------------------------------------
def test_qsweep_off_by_default_under_the_parent_flag(monkeypatch):
    ker = _refiner(CASES[1], True, monkeypatch, qsweep=False)
    assert ker._nk is not None
    assert ker._nk.qsweep is False


def test_qsweep_alone_does_nothing_without_the_parent_flag(monkeypatch):
    monkeypatch.setenv("PARTNER_REFINE_KERNEL_QSWEEP", "1")
    inst = build_instance(n=30, seed=2)
    opt = make_optimizer(inst, seed=3, deadline=time.time() + 100.0)
    r = rf._Refiner(opt, np.asarray([list(x) for x in inst.rects],
                                    dtype=np.float64), seed=11)
    assert r._nk is None
    assert rnk.qsweep_enabled() is False


def test_median_sweep_stays_python_when_qsweep_is_off(monkeypatch):
    """The guarantee the parent flag's own test file asserts, restated against
    the new dispatch site: with the sub-switch off, `hold=False` must not reach
    either kernel entry point."""
    ker = _refiner(CASES[0], True, monkeypatch, qsweep=False)
    calls, qcalls = ker._nk.calls, ker._nk.qcalls
    ker._axis_pass(0, hold=False)
    ker._axis_pass(1, max_step=1.0, hold=False)
    assert (ker._nk.calls, ker._nk.qcalls) == (calls, qcalls)
    ker._axis_pass(0, hold=True)
    assert (ker._nk.calls, ker._nk.qcalls) == (calls + 1, qcalls)


def test_dispatch_fires_when_qsweep_is_on(monkeypatch):
    ker = _refiner(CASES[0], True, monkeypatch)
    q = ker._nk.qcalls
    ker._axis_pass(0)
    ker._axis_pass(1)
    assert ker._nk.qcalls == q + 2


# --------------------------------------------------------------------------
# 2. `_wmedian`: the one step whose tie permutation is NOT reproduced
# --------------------------------------------------------------------------
def _wmedian_k_py(vals, wts):
    """Host-side call of the njit leaf with the scratch it expects."""
    m = len(vals)
    v = np.ascontiguousarray(vals, dtype=np.float64)
    w = np.ascontiguousarray(wts, dtype=np.float64)
    return float(rnk._wmedian_k(v, w, m, np.zeros(m), np.zeros(m)))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 17, 64, 257])
def test_wmedian_matches_on_random_input(n):
    rng = np.random.default_rng(20260807)
    for _ in range(60):
        vals = rng.normal(size=n) * rng.choice([1.0, 1e-6, 1e6])
        wts = rng.random(size=n) + 1e-9
        assert _wmedian_k_py(vals, wts) == rf._Refiner._wmedian(vals, wts)


@pytest.mark.parametrize("levels", [1, 2, 3, 8])
@pytest.mark.parametrize("n", [2, 7, 33, 128])
def test_wmedian_matches_on_tie_saturated_input(n, levels):
    """The adversarial half: `vals` drawn from a tiny alphabet, so almost every
    entry is a tie and the kernel's stable sort and numpy's introsort really do
    choose different permutations."""
    rng = np.random.default_rng(7 * n + levels)
    for _ in range(120):
        vals = rng.integers(0, levels, size=n).astype(np.float64)
        # weights that make partial sums re-associate badly: mixed magnitudes
        wts = rng.choice([1.0, 1e-13, 3.7, 1e7], size=n) * (1.0 + rng.random(n))
        assert _wmedian_k_py(vals, wts) == rf._Refiner._wmedian(vals, wts)


def test_wmedian_matches_on_equal_weight_ties():
    """The knife edge the docstring calls out: uniform weights put `half`
    exactly on a cumulative-sum boundary."""
    for n in (2, 4, 8, 16, 100):
        for levels in (1, 2, 4):
            vals = np.arange(n, dtype=np.float64) % levels
            wts = np.ones(n)
            assert _wmedian_k_py(vals, wts) == rf._Refiner._wmedian(vals, wts)
            wts2 = np.full(n, 0.1)
            assert _wmedian_k_py(vals, wts2) == rf._Refiner._wmedian(vals, wts2)


# --------------------------------------------------------------------------
# 3. `_median_shift`: the per-group target, group by group
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_median_shift_bit_exact_per_group(case, monkeypatch):
    ref, ker = _pair(case, monkeypatch)
    nk = ker._nk
    seen_edge = seen_pin = seen_none = 0
    rng = np.random.default_rng(5)
    for trial in range(3):
        if trial:
            # re-evaluate on a perturbed layout: the sweep calls this AFTER
            # earlier groups moved, so a static comparison would be weak
            ker.P += rng.normal(scale=0.5, size=ker.P.shape) * \
                np.array([1.0, 1.0, 0.0, 0.0])
            ref.P[...] = ker.P
        for axis in (0, 1):
            for gi, g in enumerate(ref.groups):
                want = ref._median_shift(g, axis)
                found, got = rnk._median_shift_k(
                    ker.P, axis, gi, nk.ge_ptr, nk.ge_M, nk.ge_J, nk.ge_W,
                    nk.gp_ptr, nk.gp_M, nk.gp_X, nk.gp_Y, nk.gp_W,
                    nk.des, nk.wts, nk.sv, nk.sc)
                if want is None:
                    assert not found
                    seen_none += 1
                    continue
                assert found and got == want, \
                    f"group {gi} axis {axis} trial {trial}: {got} != {want}"
                seen_edge += int(len(g.eM) > 0)
                seen_pin += int(len(g.pM) > 0)
    assert seen_edge > 0, "no group had external edges -- test is vacuous"
    assert seen_pin > 0, "no group had pins -- test is vacuous"


def test_median_shift_csr_mirrors_the_group_arrays(monkeypatch):
    ker = _refiner(CASES[3], True, monkeypatch)
    nk = ker._nk
    for gi, g in enumerate(ker.groups):
        e0, e1 = int(nk.ge_ptr[gi]), int(nk.ge_ptr[gi + 1])
        p0, p1 = int(nk.gp_ptr[gi]), int(nk.gp_ptr[gi + 1])
        assert np.array_equal(nk.ge_M[e0:e1], g.eM)
        assert np.array_equal(nk.ge_J[e0:e1], g.eJ)
        assert np.array_equal(nk.ge_W[e0:e1], g.eW)
        assert np.array_equal(nk.gp_M[p0:p1], g.pM)
        assert np.array_equal(nk.gp_X[p0:p1], g.pX)
        assert np.array_equal(nk.gp_Y[p0:p1], g.pY)
        assert np.array_equal(nk.gp_W[p0:p1], g.pW)


def test_qsweep_off_does_not_build_the_csr(monkeypatch):
    """The sub-switch must cost the promoted `hold=True` path nothing, not even
    `__init__` time."""
    ker = _refiner(CASES[2], True, monkeypatch, qsweep=False)
    nk = ker._nk
    assert nk.qsweep is False
    assert int(nk.ge_ptr[-1]) == 0 and int(nk.gp_ptr[-1]) == 0
    assert nk.des.size == 1


# --------------------------------------------------------------------------
# 4. the whole sweep, both axes, many iterations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=_IDS)
@pytest.mark.parametrize("expand", [0.0, 0.28])
def test_axis_pass_soft_bit_exact(case, expand, monkeypatch):
    ref, ker = _pair(case, monkeypatch, expand=expand)
    for it in range(14):
        for axis in (0, 1):
            ref._axis_pass(axis)
            ker._axis_pass(axis)
            assert np.array_equal(ref.P, ker.P), \
                f"P mismatch at it={it} axis={axis}"
    assert ker._nk.qcalls == 28


@pytest.mark.parametrize("case", CASES[:5], ids=_IDS[:5])
def test_axis_pass_soft_bit_exact_with_max_step(case, monkeypatch):
    """`max_step` is unused in production today (`run` always passes None), so
    the port carries it only to keep the signature honest -- assert it."""
    ref, ker = _pair(case, monkeypatch, expand=0.28)
    for it, ms in enumerate((0.001, 0.5, 4.0, 1e9, None) * 2):
        axis = it % 2
        ref._axis_pass(axis, max_step=ms)
        ker._axis_pass(axis, max_step=ms)
        assert np.array_equal(ref.P, ker.P), f"it={it} max_step={ms}"


@pytest.mark.parametrize("case", CASES[:5], ids=_IDS[:5])
def test_soft_and_hold_sweeps_interleave_bit_exact(case, monkeypatch):
    """`run` alternates the two halves through `_tag_snap` / `_perturb`; the
    shared `_axis_build_k` must behave identically in both callers."""
    ref, ker = _pair(case, monkeypatch, expand=0.28)
    for it in range(8):
        for axis in (0, 1):
            ref._axis_pass(axis, hold=True, invert=(it % 3 == 2))
            ker._axis_pass(axis, hold=True, invert=(it % 3 == 2))
            assert np.array_equal(ref.P, ker.P), f"hold it={it} axis={axis}"
            ref._axis_pass(axis)
            ker._axis_pass(axis)
            assert np.array_equal(ref.P, ker.P), f"soft it={it} axis={axis}"


@pytest.mark.parametrize("case", CASES[:5], ids=_IDS[:5])
def test_rng_stream_untouched_by_the_sweep(case, monkeypatch):
    ref, ker = _pair(case, monkeypatch, expand=0.28)
    for _ in range(6):
        ref._axis_pass(0)
        ref._axis_pass(1)
        ker._axis_pass(0)
        ker._axis_pass(1)
    assert ref.rng.getstate() == ker.rng.getstate()


# --------------------------------------------------------------------------
# 5. end to end: `refine_prediction`, off vs on, by hash
#
# Same precondition as the parent flag's end-to-end test: `refine_prediction`
# is deadline-driven, so the claim is only testable where the deadline is
# inert -- which is exactly where the base arm is self-reproducible.  The base
# arm here is `PARTNER_REFINE_KERNEL=numba` (what `.env` ships), so the delta
# under test is the sub-switch alone.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", [dict(n=40, seed=0), dict(n=55, seed=5),
                                  dict(n=70, seed=1),
                                  dict(n=64, seed=11, n_preplaced=6,
                                       n_clusters=10, n_mib=5)],
                         ids=["n40", "n55", "n70", "n64clu"])
def test_refine_prediction_bit_exact_off_vs_on(case, monkeypatch):
    span = 60.0
    inst = build_instance(**case)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)

    def run(qsweep):
        monkeypatch.setenv("PARTNER_REFINE_KERNEL", "numba")
        if qsweep:
            monkeypatch.setenv("PARTNER_REFINE_KERNEL_QSWEEP", "1")
        else:
            monkeypatch.delenv("PARTNER_REFINE_KERNEL_QSWEEP", raising=False)
        dl = time.time() + span
        opt = make_optimizer(inst, seed=3, deadline=dl)
        out = rf.refine_prediction(opt, pred, dl, seed=11)
        monkeypatch.delenv("PARTNER_REFINE_KERNEL", raising=False)
        monkeypatch.delenv("PARTNER_REFINE_KERNEL_QSWEEP", raising=False)
        return None if out is None else np.asarray(out, dtype=np.float64)

    a1 = run(False)
    a2 = run(False)
    if a1 is None or a2 is None or not np.array_equal(a1, a2):
        pytest.skip("base arm not deadline-converged at this span")
    b = run(True)
    assert b is not None
    assert _digest(b) == _digest(a1), \
        f"hash mismatch: off={_digest(a1)} on={_digest(b)}"
    assert np.array_equal(a1, b)


def test_refine_prediction_bit_exact_vs_pure_python(monkeypatch):
    """The end of the chain: sub-switch on vs *no* kernel at all."""
    span = 60.0
    inst = build_instance(n=50, seed=17, frac_boundary=0.45)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)

    def run(env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        dl = time.time() + span
        opt = make_optimizer(inst, seed=3, deadline=dl)
        out = rf.refine_prediction(opt, pred, dl, seed=11)
        for k in env:
            monkeypatch.delenv(k, raising=False)
        return None if out is None else np.asarray(out, dtype=np.float64)

    a1 = run({})
    a2 = run({})
    if a1 is None or a2 is None or not np.array_equal(a1, a2):
        pytest.skip("python path not deadline-converged at this span")
    b = run({"PARTNER_REFINE_KERNEL": "numba",
             "PARTNER_REFINE_KERNEL_QSWEEP": "1"})
    assert b is not None and _digest(b) == _digest(a1)


# --------------------------------------------------------------------------
# 6. PARTNER_REFINE_PROF: instrumentation is inert when off, and measures
#    without changing the answer when on
# --------------------------------------------------------------------------
def test_prof_is_none_by_default(monkeypatch):
    r = _refiner(CASES[1], False, monkeypatch)
    assert r._prof is None


def test_prof_writes_a_record_and_does_not_change_the_layout(
        monkeypatch, tmp_path):
    span = 45.0
    inst = build_instance(n=40, seed=0)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)

    def run(env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        dl = time.time() + span
        opt = make_optimizer(inst, seed=3, deadline=dl)
        out = rf.refine_prediction(opt, pred, dl, seed=11)
        for k in env:
            monkeypatch.delenv(k, raising=False)
        return None if out is None else np.asarray(out, dtype=np.float64)

    a1 = run({})
    a2 = run({})
    if a1 is None or a2 is None or not np.array_equal(a1, a2):
        pytest.skip("python path not deadline-converged at this span")
    path = tmp_path / "prof.jsonl"
    b = run({"PARTNER_REFINE_PROF": "1",
             "PARTNER_REFINE_PROF_FILE": str(path)})
    assert b is not None and np.array_equal(a1, b)
    # one file per pid (network-mount safe); aggregate with a glob
    files = sorted(tmp_path.glob("prof.jsonl.*"))
    assert files
    import json
    recs = [json.loads(ln) for f in files
            for ln in f.read_text().splitlines() if ln]
    assert recs
    for rec in recs:
        assert rec["span"] > 0.0
        assert 0.0 <= rec["f_axis_soft"] <= 1.0
        # the sections plus `other` must account for the whole run
        total = sum(rec[k] for k in rf._PROF_KEYS) + rec["other"]
        assert abs(total - rec["span"]) < 1e-3 * max(rec["span"], 1.0)
    assert any(r["c_axis_soft"] > 0 for r in recs)
