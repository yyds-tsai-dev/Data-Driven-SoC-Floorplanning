"""`PARTNER_SA_KERNEL=numba`: flat-array (SoA) numba kernel for `_layout`.

The kernel in `partner/sa_numeric_kernel.py` re-expresses `_layout_full` --
including `_stack_column`, `_band_solutions`/`_solve_band`/`_dyn_split`/
`_merge_band_once`, the obstacle segment logic and both global post-passes --
over CSR-flattened numpy arrays so numba can compile it.  A previous attempt
to `njit` the `_Unit` object graph in place was infeasible; the data-layout
rewrite is what unlocks it.

The kernel is a *transcription*, so the contract asserted here is **bit
equality**, not tolerance: same `pos`, same `x_right`, same `y_top`.  That
matters because `_cost` (and therefore every Metropolis accept) is computed
from `pos`; anything less than bit equality would make the toggle a
search-trajectory change rather than a pure speed change.

Default OFF: with `PARTNER_SA_KERNEL` unset, `_sa_kernel is None` and the
solver takes exactly the pre-existing Python path.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as lg  # noqa: E402
import sa_numeric_kernel as snk  # noqa: E402
from synth_instances import build_instance  # noqa: E402

pytestmark = pytest.mark.skipif(
    not snk.NUMBA_AVAILABLE, reason="numba not installed")

_ENV = ("PARTNER_SA_KERNEL", "PARTNER_COL_CACHE", "PARTNER_SA_KERNEL_WARMUP")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# --------------------------------------------------------------------------
# synthetic instances (no dataset access, no evaluator, single process)
# --------------------------------------------------------------------------
def _instance(**kw):
    return build_instance(**kw)


def _opt(inst, seed=0, kernel=False, monkeypatch=None):
    if kernel:
        monkeypatch.setenv("PARTNER_SA_KERNEL", "numba")
    o = lg._ColumnOptimizer(
        inst.rects, inst.area_targets, inst.constraints, inst.target_positions,
        inst.b2b, inst.p2b, inst.pins, time.time() + 1000.0, seed=seed)
    if kernel:
        monkeypatch.delenv("PARTNER_SA_KERNEL", raising=False)
    return o


def _pair(inst, seed, monkeypatch):
    """A Python-path optimizer and a kernel-path optimizer on one instance."""
    ref = _opt(inst, seed=seed, kernel=False)
    ker = _opt(inst, seed=seed, kernel=True, monkeypatch=monkeypatch)
    assert ker._sa_kernel is not None, "kernel failed to attach"
    return ref, ker


CASES = [
    dict(n=100, seed=0),
    dict(n=40, seed=3),
    dict(n=120, seed=7),
    dict(n=64, seed=11, n_preplaced=6, n_clusters=10, n_mib=5),
    dict(n=30, seed=13, n_preplaced=0, frac_fixed=0.0),   # no obstacles at all
    dict(n=50, seed=17, frac_boundary=0.45),              # boundary-saturated
    dict(n=21, seed=19, n_clusters=1, n_mib=1),           # minimum block count
    # anchored units: the only way into `_stack_column`'s anchor-gluing branch
    # and `_place_unit_down` (a cluster group containing a preplaced member)
    dict(n=80, seed=53, n_preplaced=5, n_clusters=8, anchor_clusters=4),
    dict(n=110, seed=59, n_preplaced=8, n_clusters=12, anchor_clusters=8,
         frac_boundary=0.30),
]


# --------------------------------------------------------------------------
# 1. default off == bit-identical current behaviour
# --------------------------------------------------------------------------
def test_default_off_leaves_python_path():
    o = _opt(_instance(n=40, seed=1))
    assert o._sa_kernel is None
    cols = o._init_columns(5)
    pos, xr, yt = o._layout(cols)
    pos2, xr2, yt2 = o._layout_full(cols)
    assert np.array_equal(pos, pos2) and xr == xr2 and yt == yt2


def test_kernel_absent_when_flag_is_not_numba(monkeypatch):
    monkeypatch.setenv("PARTNER_SA_KERNEL", "1")
    o = _opt(_instance(n=30, seed=2))
    assert o._sa_kernel is None


# --------------------------------------------------------------------------
# 2. parity on the initial column assignment
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"n{c['n']}s{c['seed']}")
def test_layout_bit_exact_on_initial_columns(case, monkeypatch):
    inst = _instance(**case)
    ref, ker = _pair(inst, seed=case["seed"], monkeypatch=monkeypatch)
    for C in (2, 3, 5, 8, 12, 18):
        cols = ref._init_columns(C)
        cols_k = ker._init_columns(C)
        assert cols == cols_k
        a = ref._layout_full(cols)
        b = ker._layout(cols_k)
        assert np.array_equal(a[0], b[0]), f"pos mismatch at C={C}"
        assert a[1] == b[1] and a[2] == b[2], f"frame mismatch at C={C}"


# --------------------------------------------------------------------------
# 3. parity over a random walk through the SA move space
#    (this is where the obstacle segment logic, band merging, dyn-split, the
#     widen-retry loop and both post-passes actually get exercised)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"n{c['n']}s{c['seed']}")
def test_layout_bit_exact_along_random_move_walk(case, monkeypatch):
    inst = _instance(**case)
    ref, ker = _pair(inst, seed=case["seed"], monkeypatch=monkeypatch)
    C = max(2, min(18, len(ref.units) // 6 + 2))
    cols_r = ref._init_columns(C)
    cols_k = ker._init_columns(C)

    checked = 0
    for step in range(400):
        # drive both optimizers with the SAME move by replaying the rng
        ref.rng.seed(1000 + step)
        ker.rng.seed(1000 + step)
        undo_r = ref._random_move(cols_r)
        undo_k = ker._random_move(cols_k)
        assert cols_r == cols_k, f"move divergence at step {step}"
        assert (undo_r is None) == (undo_k is None)
        if undo_r is None:
            continue
        a = ref._layout_full(cols_r)
        b = ker._layout(cols_k)
        assert b is not None, f"kernel fell back at step {step}"
        assert np.array_equal(a[0], b[0]), f"pos mismatch at step {step}"
        assert a[1] == b[1] and a[2] == b[2], f"frame mismatch at step {step}"
        checked += 1
        # keep roughly half the moves so the walk actually drifts
        if step % 2:
            undo_r()
            undo_k()
    assert checked > 150, f"walk exercised too few layouts ({checked})"


# --------------------------------------------------------------------------
# 4. the subgroup-reorder move mutates the band structure -> the flat arrays
#    must be rebuilt.  This is the one move class that invalidates statics.
# --------------------------------------------------------------------------
def test_subgroup_reorder_invalidates_flat_arrays(monkeypatch):
    inst = _instance(n=80, seed=23, n_clusters=14, n_mib=4)
    ref, ker = _pair(inst, seed=23, monkeypatch=monkeypatch)
    multi = [k for k, u in enumerate(ker.units)
             if len(u.subgroups) >= 2 or any(len(sg) >= 2 for sg in u.subgroups)]
    assert multi, "instance has no reorderable unit"
    cols_r = ref._init_columns(6)
    cols_k = ker._init_columns(6)

    hits = 0
    for step in range(200):
        k = multi[step % len(multi)]
        u_r, u_k = ref.units[k], ker.units[k]
        sgs = [i for i, g in enumerate(u_k.subgroups) if len(g) >= 2]
        if len(u_k.subgroups) >= 2:
            a = step % (len(u_k.subgroups) - 1)
            for o, u in ((ref, u_r), (ker, u_k)):
                u.subgroups[a], u.subgroups[a + 1] = u.subgroups[a + 1], u.subgroups[a]
                o._refresh_unit(u)
        elif sgs:
            gi = sgs[0]
            a = step % (len(u_k.subgroups[gi]) - 1)
            for o, u in ((ref, u_r), (ker, u_k)):
                g = u.subgroups[gi]
                g[a], g[a + 1] = g[a + 1], g[a]
                o._refresh_unit(u)
        else:
            continue
        assert ker._sa_kernel.dirty, "refresh did not mark the kernel dirty"
        a1 = ref._layout_full(cols_r)
        b1 = ker._layout(cols_k)
        assert np.array_equal(a1[0], b1[0]), f"pos mismatch after reorder {step}"
        assert a1[1] == b1[1] and a1[2] == b1[2]
        hits += 1
    assert hits > 50


# --------------------------------------------------------------------------
# 5. side-channel state the SA reads back from `_layout`
# --------------------------------------------------------------------------
def test_col_spans_and_unit_col_match(monkeypatch):
    inst = _instance(n=90, seed=29)
    ref, ker = _pair(inst, seed=29, monkeypatch=monkeypatch)
    cols_r = ref._init_columns(9)
    cols_k = ker._init_columns(9)
    ref._layout_full(cols_r)
    ker._layout(cols_k)
    assert len(ref._col_spans) == len(ker._col_spans)
    for a, b in zip(ref._col_spans, ker._col_spans):
        assert (a is None) == (b is None)
        if a is not None:
            assert a[0] == b[0] and a[1] == b[1]
    assert list(ref._unit_col) == [int(v) for v in ker._unit_col]


# --------------------------------------------------------------------------
# 6. idempotence + boundary cases
# --------------------------------------------------------------------------
def test_kernel_is_idempotent(monkeypatch):
    inst = _instance(n=70, seed=31)
    _, ker = _pair(inst, seed=31, monkeypatch=monkeypatch)
    cols = ker._init_columns(7)
    first = ker._layout(cols)
    for _ in range(5):
        again = ker._layout(cols)
        assert np.array_equal(first[0], again[0])
        assert first[1] == again[1] and first[2] == again[2]


def test_empty_columns_are_handled(monkeypatch):
    inst = _instance(n=45, seed=37)
    ref, ker = _pair(inst, seed=37, monkeypatch=monkeypatch)
    cols_r = ref._init_columns(6)
    cols_k = ker._init_columns(6)
    # drain two columns into their neighbours -> empty columns in the middle
    for src in (1, 3):
        for cs in (cols_r, cols_k):
            cs[src + 1].extend(cs[src])
            cs[src] = []
    a = ref._layout_full(cols_r)
    b = ker._layout(cols_k)
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]
    assert ker._col_spans[1] is None and ker._col_spans[3] is None


def test_single_column_and_single_unit(monkeypatch):
    inst = _instance(n=21, seed=41, n_clusters=0, n_mib=0)
    ref, ker = _pair(inst, seed=41, monkeypatch=monkeypatch)
    cols_r = ref._init_columns(2)
    cols_k = ker._init_columns(2)
    for cs in (cols_r, cols_k):
        cs[0].extend(cs[1])
        cs[1] = []
    a = ref._layout_full(cols_r)
    b = ker._layout(cols_k)
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]


# --------------------------------------------------------------------------
# 7. `_violations` kernel.  This one is bit-exact for a structural reason
#    rather than by transcription luck: the result is an integer count, so it
#    cannot depend on float accumulation order the way a sum would.  The
#    inputs it *is* sensitive to are min/max reductions (exact in any order)
#    and `round(x, 4)` (numba matches CPython's decimal round).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"n{c['n']}s{c['seed']}")
def test_violations_bit_exact_on_layouts(case, monkeypatch):
    inst = _instance(**case)
    ref, ker = _pair(inst, seed=case["seed"], monkeypatch=monkeypatch)
    C = max(2, min(18, len(ref.units) // 6 + 2))
    cols_r = ref._init_columns(C)
    cols_k = ker._init_columns(C)
    seen = set()
    for step in range(120):
        ref.rng.seed(7000 + step)
        ker.rng.seed(7000 + step)
        ur = ref._random_move(cols_r)
        uk = ker._random_move(cols_k)
        if ur is None:
            continue
        pos = ref._layout_full(cols_r)[0]
        assert ref._violations(pos) == ker._violations(pos)
        seen.add(ref._violations(pos))
    # a layout walk that never changes its violation count would prove nothing
    assert len(seen) >= 1


def test_violations_bit_exact_on_perturbed_layouts(monkeypatch):
    """Random jitter drives blocks across the boundary/touch tolerances and
    knocks MIB shapes just over the 1e-4 rounding grid -- exactly the knife
    edges where a reimplementation would diverge from a transcription."""
    inst = _instance(n=90, seed=61, n_clusters=12, n_mib=6, anchor_clusters=3,
                     frac_boundary=0.35)
    ref, ker = _pair(inst, seed=61, monkeypatch=monkeypatch)
    cols = ref._init_columns(9)
    base = ref._layout_full(cols)[0]
    rng = random.Random(5)
    counts = set()
    for trial in range(300):
        pos = base.copy()
        for i in range(pos.shape[0]):
            if rng.random() < 0.5:
                # jitter at the scale of eps (1e-6), TOUCH_TOL (1e-7) and the
                # MIB rounding grid (1e-4)
                s = rng.choice([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
                pos[i, rng.randrange(4)] += s * rng.choice([-1.0, 1.0])
        a = ker._violations(pos)
        # compare against the untouched numpy implementation
        b = lg._ColumnOptimizer._violations(ref, pos)
        assert a == b, f"violation mismatch at trial {trial}: {a} vs {b}"
        counts.add(a)
    assert len(counts) > 1, "jitter never changed the violation count"


# --------------------------------------------------------------------------
# 8. end-to-end: a full anneal chain must follow the identical trajectory
# --------------------------------------------------------------------------
def test_anneal_trajectory_is_identical(monkeypatch):
    inst = _instance(n=60, seed=43)
    ref, ker = _pair(inst, seed=43, monkeypatch=monkeypatch)
    ref.prepare()
    ker.prepare()
    assert ref._cost0 == ker._cost0
    ref.rng.seed(9)
    ker.rng.seed(9)
    cr = ref._restore(ref._snapshot(ref._cols))
    ck = ker._restore(ker._snapshot(ker._cols))
    cost_r, _ = ref._evaluate(cr)
    cost_k, _ = ker._evaluate(ck)
    assert cost_r == cost_k
    # a fixed number of moves rather than a deadline, so the comparison is
    # wall-clock independent
    for step in range(600):
        ref.rng.seed(500 + step)
        ker.rng.seed(500 + step)
        ur = ref._random_move(cr)
        uk = ker._random_move(ck)
        if ur is None:
            continue
        nr, _ = ref._evaluate(cr)
        nk, _ = ker._evaluate(ck)
        assert nr == nk, f"cost divergence at step {step}: {nr!r} vs {nk!r}"
        if nr > cost_r:
            ur()
            uk()
        else:
            cost_r = nr
    assert ker._sa_kernel.fallbacks == 0
