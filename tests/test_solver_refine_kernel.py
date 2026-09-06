"""`PARTNER_REFINE_KERNEL=numba`: flat-array (SoA) numba kernel for the rung.

A direct-refine ladder rung is one `_Refiner.legalize_soft()` call, and its
measured hot loops are all numeric: the `hold=True` axis pass (constraint
build + forward assignment + rigid moves), the `_evict` anchor x variant scan,
and the two overlap tests.  `src/solver/refine_numeric_kernel.py` re-expresses
them over CSR-flattened arrays; the `_Group` / `_AxisCons` object graph is only
a container for numbers, so the rewrite is a transcription.

Because it is a transcription, the contract asserted here is **bit equality**,
not tolerance: the same `P` array, block for block.  That matters because
`legalize_soft`'s own control flow (`_overlap_count() <= 10`, the `<= 6` pair
salvage) branches on the layout, and `refine_prediction` ranks candidates on
it -- anything less than bit equality would make the toggle a search-trajectory
change rather than a pure speed change.

Default OFF: with `PARTNER_REFINE_KERNEL` unset, `_Refiner._nk is None` and
every dispatch site is a single identity test, so the decision logic AND the
rng stream are those of the pre-port fork.
"""

from __future__ import annotations

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

_ENV = ("PARTNER_REFINE_KERNEL", "PARTNER_REFINE_KERNEL_WARMUP",
        "PARTNER_ANYTIME_LADDER", "PARTNER_EARLY_EXIT",
        "PARTNER_REFINE_STALL_STOP")


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
    # anchored clusters: the `_move` contact re-snap path (g.cH / g.cV)
    dict(n=80, seed=53, n_preplaced=5, n_clusters=8, anchor_clusters=4),
    dict(n=110, seed=59, n_preplaced=8, n_clusters=12, anchor_clusters=8,
         frac_boundary=0.30),
]
_IDS = [f"n{c['n']}s{c['seed']}" for c in CASES]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _refiner(case, kernel, monkeypatch, expand=0.0, span=1000.0):
    inst = build_instance(**case)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    opt = make_optimizer(inst, seed=3, deadline=time.time() + span)
    if kernel:
        monkeypatch.setenv("PARTNER_REFINE_KERNEL", "numba")
    else:
        monkeypatch.delenv("PARTNER_REFINE_KERNEL", raising=False)
    r = rf._Refiner(opt, pred, seed=11)
    monkeypatch.delenv("PARTNER_REFINE_KERNEL", raising=False)
    if expand:
        r.xmax += (r.xmax - r.xmin) * expand
        r.ymax += (r.ymax - r.ymin) * expand
    return r


def _pair(case, monkeypatch, expand=0.0):
    ref = _refiner(case, False, monkeypatch, expand)
    ker = _refiner(case, True, monkeypatch, expand)
    assert ref._nk is None
    assert ker._nk is not None, "kernel failed to attach"
    assert np.array_equal(ref.P, ker.P)
    return ref, ker


# --------------------------------------------------------------------------
# 1. default off == the pre-port Python path
# --------------------------------------------------------------------------
def test_default_off_leaves_python_path(monkeypatch):
    r = _refiner(CASES[1], False, monkeypatch)
    assert r._nk is None


def test_kernel_absent_when_flag_is_not_numba(monkeypatch):
    monkeypatch.setenv("PARTNER_REFINE_KERNEL", "1")
    inst = build_instance(n=30, seed=2)
    opt = make_optimizer(inst, seed=3, deadline=time.time() + 100.0)
    r = rf._Refiner(opt, np.asarray([list(x) for x in inst.rects],
                                    dtype=np.float64), seed=11)
    assert r._nk is None


def test_off_path_runs_without_numba_imported(monkeypatch):
    """Flag off: every dispatch site is a single `is None` test, so the four
    ported entry points must run with no kernel object in existence.  (The
    stronger claim -- byte-identical layouts vs the pre-port module -- is
    measured in `scratchpad/refine_kernel_offpath.py`.)"""
    r = _refiner(CASES[0], False, monkeypatch)
    assert r._nk is None
    r._axis_pass(0, hold=True)
    r._has_overlap()
    r._overlap_count()
    r._evict(0)
    assert r._nk is None


# --------------------------------------------------------------------------
# 2. the primitives, one call at a time
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_overlap_tests_agree(case, monkeypatch):
    ref, ker = _pair(case, monkeypatch)
    assert ref._has_overlap() == ker._has_overlap()
    assert ref._overlap_count() == ker._overlap_count()
    # and again after a few sweeps have thinned the overlaps out
    for _ in range(3):
        ref._axis_pass(0, hold=True)
        ref._axis_pass(1, hold=True)
        ker._axis_pass(0, hold=True)
        ker._axis_pass(1, hold=True)
        assert np.array_equal(ref.P, ker.P)
        assert ref._has_overlap() == ker._has_overlap()
        assert ref._overlap_count() == ker._overlap_count()


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_axis_pass_hold_bit_exact_including_invert(case, monkeypatch):
    ref, ker = _pair(case, monkeypatch)
    for it in range(12):
        inv = (it % 5 == 4)
        for axis in (0, 1):
            ref._axis_pass(axis, hold=True, invert=inv)
            ker._axis_pass(axis, hold=True, invert=inv)
            assert np.array_equal(ref.P, ker.P), \
                f"P mismatch at it={it} axis={axis} invert={inv}"


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_evict_scan_bit_exact(case, monkeypatch):
    ref, ker = _pair(case, monkeypatch)
    tried = 0
    for i in range(ref.n):
        a = ref._evict(i)
        b = ker._evict(i)
        assert a == b, f"evict verdict differs for block {i}"
        assert np.array_equal(ref.P, ker.P), f"evict P differs for block {i}"
        tried += int(a)
    assert tried > 0, "no block was evictable -- test is vacuous"


def test_evict_bit_exact_with_max_anchors_and_target(monkeypatch):
    """The two non-default entries: `max_anchors=self.n` (the `legalize_soft`
    salvage) and an explicit `target` (`_deflate`)."""
    ref, ker = _pair(CASES[0], monkeypatch)
    tgt = (float(ref.xmin + 0.3 * (ref.xmax - ref.xmin)),
           float(ref.ymin + 0.7 * (ref.ymax - ref.ymin)))
    hits = 0
    for i in range(0, ref.n, 3):
        a = ref._evict(i, max_anchors=ref.n)
        b = ker._evict(i, max_anchors=ker.n)
        assert a == b and np.array_equal(ref.P, ker.P)
        hits += int(a)
        a = ref._evict(i, target=tgt)
        b = ker._evict(i, target=tgt)
        assert a == b and np.array_equal(ref.P, ker.P)
        hits += int(a)
    assert hits > 0


# --------------------------------------------------------------------------
# 3. a whole rung, and a whole `_tighten`
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=_IDS)
@pytest.mark.parametrize("expand", [0.0, 0.28])
def test_legalize_soft_rung_bit_exact(case, expand, monkeypatch):
    ref, ker = _pair(case, monkeypatch, expand=expand)
    dl = time.time() + 1000.0
    ok_r = ref.legalize_soft(14, deadline=dl, fine=False)
    ok_k = ker.legalize_soft(14, deadline=dl, fine=False)
    assert ok_r == ok_k
    assert np.array_equal(ref.P, ker.P)


@pytest.mark.parametrize("case", CASES[:5], ids=_IDS[:5])
def test_legalize_soft_fine_rung_bit_exact(case, monkeypatch):
    """`fine=True` is the ladder's rung 0 (5-step regrow)."""
    ref, ker = _pair(case, monkeypatch)
    dl = time.time() + 1000.0
    assert ref.legalize_soft(14, deadline=dl, fine=True) == \
        ker.legalize_soft(14, deadline=dl, fine=True)
    assert np.array_equal(ref.P, ker.P)


@pytest.mark.parametrize("case", CASES[:5], ids=_IDS[:5])
def test_tighten_bit_exact(case, monkeypatch):
    ref, ker = _pair(case, monkeypatch, expand=0.28)
    dl = time.time() + 1000.0
    ref.legalize_soft(14, deadline=dl, fine=False)
    ker.legalize_soft(14, deadline=dl, fine=False)
    assert np.array_equal(ref.P, ker.P)
    ref._tighten(time.time() + 1000.0)
    ker._tighten(time.time() + 1000.0)
    assert np.array_equal(ref.P, ker.P)
    assert (ref.xmax, ref.ymax) == (ker.xmax, ker.ymax)


# --------------------------------------------------------------------------
# 4. the rng stream is untouched
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES[:5], ids=_IDS[:5])
def test_rng_stream_identical_after_a_rung(case, monkeypatch):
    ref, ker = _pair(case, monkeypatch, expand=0.28)
    dl = time.time() + 1000.0
    ref.legalize_soft(14, deadline=dl, fine=False)
    ker.legalize_soft(14, deadline=dl, fine=False)
    assert ref.rng.getstate() == ker.rng.getstate()


# --------------------------------------------------------------------------
# 5. hold=False must stay on the Python path
# --------------------------------------------------------------------------
def test_median_sweep_never_reaches_the_kernel(monkeypatch):
    """`_axis_pass(hold=False)` ends in `_wmedian`'s non-stable argsort, so it
    is deliberately out of scope; the dispatch must not fire for it."""
    ker = _refiner(CASES[0], True, monkeypatch)
    assert ker._nk is not None
    before = ker._nk.calls
    ker._axis_pass(0, hold=False)
    ker._axis_pass(1, max_step=1.0, hold=False)
    assert ker._nk.calls == before
    ker._axis_pass(0, hold=True)
    assert ker._nk.calls == before + 1


# --------------------------------------------------------------------------
# 5b. `_build_edges`: the O(G*|E|) -> O(|E|+G) bucketing must produce the
#     same per-group arrays element for element (it is the dominant cost of
#     `_Refiner.__init__`, which the kernel does not otherwise touch).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_fast_build_edges_is_element_identical(case, monkeypatch):
    ref, ker = _pair(case, monkeypatch)
    assert ref._fast_build is False and ker._fast_build is True
    assert len(ref.groups) == len(ker.groups)
    nonempty = 0
    for a, b in zip(ref.groups, ker.groups):
        for attr in ("eM", "eJ", "eW", "pM", "pX", "pY", "pW"):
            x, y = getattr(a, attr), getattr(b, attr)
            assert x.dtype == y.dtype, attr
            assert np.array_equal(x, y), attr
        nonempty += int(len(a.eM) > 0)
    assert nonempty > 0, "no group had external edges -- test is vacuous"
    assert ref.badj == ker.badj
    assert ref.bpin == ker.bpin


def test_fastbuild_can_be_disabled_independently(monkeypatch):
    monkeypatch.setenv("PARTNER_REFINE_KERNEL", "numba")
    monkeypatch.setenv("PARTNER_REFINE_FASTBUILD", "0")
    inst = build_instance(n=40, seed=3)
    opt = make_optimizer(inst, seed=3, deadline=time.time() + 100.0)
    r = rf._Refiner(opt, np.asarray([list(x) for x in inst.rects],
                                    dtype=np.float64), seed=11)
    assert r._nk is not None and r._fast_build is False


# --------------------------------------------------------------------------
# 6. the stable-sort assumption, asserted rather than argued
# --------------------------------------------------------------------------
def test_kernel_mergesort_matches_numpy_stable_argsort():
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 40, 121):
        for _ in range(20):
            # heavy tie density: ties are the only place stability shows
            a = rng.integers(0, 4, size=n).astype(np.float64)
            assert np.array_equal(
                rnk._argsort_stable_k(a),
                np.argsort(a, kind="stable"))


# --------------------------------------------------------------------------
# 7. end to end: `refine_prediction` on a span long enough to converge
#
# NOTE the precondition.  `refine_prediction` is deadline-driven, so at a span
# where the deadline still binds the kernel legitimately does MORE work in the
# same wall clock and the layouts differ -- that is the point of the flag, not
# a fidelity failure.  The transcription claim is only testable where the
# deadline is inert, which is exactly where the Python path is self-
# reproducible; if it is not, the machine is too loaded and the test skips.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", [dict(n=40, seed=0), dict(n=70, seed=1)],
                         ids=["n40", "n70"])
def test_refine_prediction_bit_exact(case, monkeypatch):
    span = 60.0
    inst = build_instance(**case)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)

    def run(kernel):
        if kernel:
            monkeypatch.setenv("PARTNER_REFINE_KERNEL", "numba")
        else:
            monkeypatch.delenv("PARTNER_REFINE_KERNEL", raising=False)
        dl = time.time() + span
        opt = make_optimizer(inst, seed=3, deadline=dl)
        out = rf.refine_prediction(opt, pred, dl, seed=11)
        monkeypatch.delenv("PARTNER_REFINE_KERNEL", raising=False)
        return None if out is None else np.asarray(out, dtype=np.float64)

    a1 = run(False)
    a2 = run(False)
    if a1 is None or a2 is None or not np.array_equal(a1, a2):
        pytest.skip("python path not deadline-converged at this span")
    b = run(True)
    assert b is not None
    assert np.array_equal(a1, b)
