"""`PARTNER_REFINE_KERNEL_DISC=1`: the discrete-move port.

Why this port and not the one the name suggests
-----------------------------------------------
`PARTNER_REFINE_PROF` prices the `discrete` bucket at 58% of `_Refiner.run`'s
wall clock on the real large-band cases.  Splitting that bucket
(`PARTNER_REFINE_PROF_DISC`, `_DPROF_KEYS` in `src/solver/layout_refiner.py`)
shows where inside it the time actually is:

    lc_axis  (the `_axis_pass(hold=True)` pairs of `_legal_check`)   91.4 %
    screen_delta (`_block_hp` pair deltas)                            3.7 %
    enum_optpt / enum_gain / screen_np / edits / bookkeeping        ~ 4.3 %

So the candidate-enumeration half is a ~7% target, and the legalization half
is a ~92% one that *already runs in numba* -- what it still pays is the
BOUNDARY: 6 numba dispatches and 6 O(G) Python `pin_x`/`pin_y` re-syncs per
swap attempt (~28% of one `axis_pass_hold` at n=118/G=99).  Hence two parts:

  * `fuse` -- `_legal_check`'s 3x(axis 0, axis 1)+overlap loop as one njit
    call, pins synced once (nothing inside it can write a `_Group` flag);
  * `hp`   -- `_block_hp`, `_optimal_point`, the candidate scan and the pair
    delta over a per-block CSR.

`PARTNER_REFINE_DISC_PARTS` selects them for A/B attribution.

Contract
--------
Both parts are *transformations*, not policy: the same arithmetic in the same
order, so the contract is **bit equality** (`==`, never `allclose`), asserted
here on the inner functions against random and adversarial inputs, on whole
`_legal_check` / `_discrete_batch` steps, and on whole `refine_positions` /
`refine_prediction` outputs by hash.

`hp` inherits exactly one residual from QSWEEP: `_optimal_point` ends in
`_wmedian`, whose `np.argsort(vals)` is numpy's introsort and whose tie
permutation the kernel's stable sort does not reproduce.  The invariance
argument (and its float-re-association residual) is in
`src/solver/refine_numeric_kernel.py`'s module docstring; here it is measured on
tie-saturated inputs, the only place it can bite.

Default OFF, and gated under `PARTNER_REFINE_KERNEL=numba`: with the parent
unset the sub-switch does nothing at all, and with the parent on but the
sub-switch off the block CSR is never built and every dispatch site is one
`self._disc_* is False` test.
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
        "PARTNER_REFINE_KERNEL_DISC", "PARTNER_REFINE_DISC_PARTS",
        "PARTNER_REFINE_KERNEL_WARMUP", "PARTNER_REFINE_FASTBUILD",
        "PARTNER_ANYTIME_LADDER", "PARTNER_EARLY_EXIT",
        "PARTNER_REFINE_STALL_STOP", "PARTNER_REFINE_PROF",
        "PARTNER_REFINE_PROF_DISC", "PARTNER_REFINE_PROF_FILE",
        "PARTNER_MATCH")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


CASES = [
    dict(n=100, seed=0),
    dict(n=40, seed=3),
    dict(n=118, seed=7),
    dict(n=64, seed=11, n_preplaced=6, n_clusters=10, n_mib=5),
    dict(n=30, seed=13, n_preplaced=0, frac_fixed=0.0),   # no obstacles
    dict(n=50, seed=17, frac_boundary=0.45),              # tag-saturated
    dict(n=21, seed=19, n_clusters=1, n_mib=1),           # minimum blocks
    dict(n=110, seed=59, n_preplaced=8, n_clusters=12, anchor_clusters=8,
         frac_boundary=0.30),
]
_IDS = [f"n{c['n']}s{c['seed']}" for c in CASES]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _refiner(case, disc, monkeypatch, parts=None, span=1000.0, legalize=True):
    inst = build_instance(**case)
    pred = np.asarray([list(r) for r in inst.rects], dtype=np.float64)
    opt = make_optimizer(inst, seed=3, deadline=time.time() + span)
    monkeypatch.setenv("PARTNER_REFINE_KERNEL", "numba")
    monkeypatch.setenv("PARTNER_REFINE_KERNEL_QSWEEP", "1")
    if disc:
        monkeypatch.setenv("PARTNER_REFINE_KERNEL_DISC", "1")
        if parts is not None:
            monkeypatch.setenv("PARTNER_REFINE_DISC_PARTS", parts)
    r = rf._Refiner(opt, pred, seed=11)
    for v in ("PARTNER_REFINE_KERNEL", "PARTNER_REFINE_KERNEL_QSWEEP",
              "PARTNER_REFINE_KERNEL_DISC", "PARTNER_REFINE_DISC_PARTS"):
        monkeypatch.delenv(v, raising=False)
    if legalize:
        r.legalize(60, deadline=time.time() + 60.0)
    return r


def _pair(case, monkeypatch, parts=None, **kw):
    """(no-DISC refiner, DISC refiner) on the same instance and layout."""
    a = _refiner(case, False, monkeypatch, **kw)
    b = _refiner(case, True, monkeypatch, parts=parts, **kw)
    assert np.array_equal(a.P, b.P), "fixture diverged before the test ran"
    return a, b


def _digest(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(
        np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# 1. the switch is inert when off, and cannot fire without its parent
# --------------------------------------------------------------------------
def test_off_by_default(monkeypatch):
    r = _refiner(CASES[1], False, monkeypatch, legalize=False)
    assert r._nk is not None
    assert r._nk.disc is False
    assert r._disc_fuse is False and r._disc_hp is False
    # the block CSR is not built when off
    assert r._nk.ba_j.shape[0] == 1 and r._nk.bp_X.shape[0] == 1


def test_sub_switch_needs_the_parent(monkeypatch):
    monkeypatch.setenv("PARTNER_REFINE_KERNEL_DISC", "1")
    assert rnk.disc_enabled() is False
    inst = build_instance(n=30, seed=13)
    pred = np.asarray([list(x) for x in inst.rects], dtype=np.float64)
    opt = make_optimizer(inst, seed=3, deadline=time.time() + 100.0)
    r = rf._Refiner(opt, pred, seed=11)
    assert r._nk is None
    assert r._disc_fuse is False and r._disc_hp is False


def test_parts_selection(monkeypatch):
    r = _refiner(CASES[1], True, monkeypatch, parts="fuse", legalize=False)
    assert r._disc_fuse is True and r._disc_hp is False
    r = _refiner(CASES[1], True, monkeypatch, parts="hp", legalize=False)
    assert r._disc_fuse is False and r._disc_hp is True
    r = _refiner(CASES[1], True, monkeypatch, legalize=False)
    assert r._disc_fuse is True and r._disc_hp is True


def test_csr_reproduces_badj_order(monkeypatch):
    """The CSR must be `_build_badj`'s append order, entry for entry -- that
    order IS the float accumulation order of `_block_hp`."""
    r = _refiner(CASES[0], True, monkeypatch, legalize=False)
    nk = r._nk
    for i in range(r.n):
        lo, hi = int(nk.ba_ptr[i]), int(nk.ba_ptr[i + 1])
        assert hi - lo == len(r.badj[i])
        for t, (j, w) in enumerate(r.badj[i]):
            assert int(nk.ba_j[lo + t]) == j
            assert float(nk.ba_w[lo + t]) == w
        lo, hi = int(nk.bp_ptr[i]), int(nk.bp_ptr[i + 1])
        assert hi - lo == len(r.bpin[i])
        for t, (px, py, w) in enumerate(r.bpin[i]):
            assert float(nk.bp_X[lo + t]) == px
            assert float(nk.bp_Y[lo + t]) == py
            assert float(nk.bp_W[lo + t]) == w


# --------------------------------------------------------------------------
# 2. inner functions: `==` on random and adversarial inputs
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_block_hp_bit_exact(case, monkeypatch):
    a, b = _pair(case, monkeypatch, parts="hp")
    rng = np.random.default_rng(0xB10C)
    for i in range(a.n):
        # random query points, and the degenerate ones (own centre, origin,
        # far field) where a re-association would show up first
        cxi = float(a.P[i, 0] + 0.5 * a.P[i, 2])
        cyi = float(a.P[i, 1] + 0.5 * a.P[i, 3])
        pts = [(cxi, cyi), (0.0, 0.0), (1e9, -1e9)]
        pts += [tuple(rng.uniform(-100.0, 300.0, 2)) for _ in range(6)]
        excls = [(), (i,), (i, (i + 1) % a.n), tuple(range(min(8, a.n)))]
        for (cx, cy) in pts:
            for ex in excls:
                assert a._block_hp(i, cx, cy, ex) == b._block_hp(i, cx, cy, ex)


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_optimal_point_bit_exact(case, monkeypatch):
    a, b = _pair(case, monkeypatch, parts="hp")
    for i in range(a.n):
        oa = a._optimal_point(i)
        ob = b._optimal_point(i)
        assert (oa is None) == (ob is None)
        if oa is not None:
            assert oa[0] == ob[0] and oa[1] == ob[1]


def test_optimal_point_bit_exact_on_tie_saturated_inputs():
    """The one place the stable-vs-introsort residual can bite.

    Driven straight at the kernel with a hand-built CSR, against a Python
    transcription of `_optimal_point`, on inputs engineered so that the tie
    blocks are large and the weights re-associate: integer weights (where the
    invariance is exact), then float weights (where only the ULP residual is
    left), then all-equal coordinates.
    """
    rng = np.random.default_rng(7)

    def py_optimal(cx, cy, w):
        wa = np.asarray(w)
        return (rf._Refiner._wmedian(np.asarray(cx), wa),
                rf._Refiner._wmedian(np.asarray(cy), wa))

    for trial in range(400):
        m = int(rng.integers(1, 12))
        # tie-saturated: draw from a tiny alphabet so equal values abound
        alpha = rng.choice([2, 3], size=1)[0]
        cx = rng.choice(np.arange(alpha, dtype=np.float64), size=m)
        cy = rng.choice(np.arange(alpha, dtype=np.float64), size=m)
        if trial % 3 == 0:
            w = rng.integers(1, 5, size=m).astype(np.float64)
        elif trial % 3 == 1:
            w = rng.random(m) * 3.0 + 1e-3
        else:
            w = np.full(m, 0.1)
        # CSR with block 0 carrying every entry as a PIN (pins land in the
        # buffers verbatim, so the test controls the values exactly)
        P = np.zeros((1, 4))
        ba_ptr = np.zeros(2, dtype=np.int64)
        ba_j = np.zeros(1, dtype=np.int64)
        ba_w = np.zeros(1)
        bp_ptr = np.array([0, m], dtype=np.int64)
        z = np.zeros(max(m, 1))
        found, ox, oy = rnk._optimal_point_k(
            P, 0, ba_ptr, ba_j, ba_w, bp_ptr,
            np.ascontiguousarray(cx), np.ascontiguousarray(cy),
            np.ascontiguousarray(w),
            z.copy(), z.copy(), z.copy(), z.copy(), z.copy())
        assert bool(found) is True
        px, py = py_optimal(cx, cy, w)
        assert ox == px, (trial, m, cx, w, ox, px)
        assert oy == py, (trial, m, cy, w, oy, py)


def test_optimal_point_none_when_isolated():
    P = np.zeros((1, 4))
    z = np.zeros(1)
    found, ox, oy = rnk._optimal_point_k(
        P, 0, np.zeros(2, dtype=np.int64), np.zeros(1, dtype=np.int64),
        np.zeros(1), np.zeros(2, dtype=np.int64), z, z, z,
        z.copy(), z.copy(), z.copy(), z.copy(), z.copy())
    assert bool(found) is False


@pytest.mark.parametrize("case", CASES[:4], ids=_IDS[:4])
def test_swap_delta_bit_exact(case, monkeypatch):
    a, b = _pair(case, monkeypatch, parts="hp")
    P = a.P
    cx = P[:, 0] + 0.5 * P[:, 2]
    cy = P[:, 1] + 0.5 * P[:, 3]
    rng = np.random.default_rng(99)
    for _ in range(300):
        i, j = (int(x) for x in rng.integers(0, a.n, 2))
        excl = (i, j)
        want = (a._block_hp(i, cx[j], cy[j], excl)
                + a._block_hp(j, cx[i], cy[i], excl)
                - a._block_hp(i, cx[i], cy[i], excl)
                - a._block_hp(j, cx[j], cy[j], excl))
        assert b._nk.swap_delta(i, j, cx[i], cy[i], cx[j], cy[j]) == want


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_discrete_gains_matches_the_python_scan(case, monkeypatch):
    a, b = _pair(case, monkeypatch, parts="hp")
    a._build_swappable()
    b._build_swappable()
    idxs = np.nonzero(a.swappable)[0]
    if len(idxs) < 2:
        pytest.skip("no swappable blocks in this instance")
    P = a.P
    cx = P[:, 0] + 0.5 * P[:, 2]
    cy = P[:, 1] + 0.5 * P[:, 3]
    want = []
    for i in idxs:
        o = a._optimal_point(int(i))
        if o is None:
            continue
        gain = (a._block_hp(int(i), cx[i], cy[i])
                - a._block_hp(int(i), o[0], o[1]))
        if gain > 1e-9:
            want.append((gain, int(i), o[0], o[1]))
    keep, gains, opx, opy = b._nk.discrete_gains(idxs)
    got = [(float(gains[t]), int(idxs[t]), float(opx[t]), float(opy[t]))
           for t in range(len(idxs)) if keep[t]]
    assert got == want


# --------------------------------------------------------------------------
# 3. whole steps: `_legal_check` and `_discrete_batch`
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_legal_check_bit_exact(case, monkeypatch):
    """Same edit, same starting layout: the fused sweep must land the same
    `P`, the same accept decision and the same key."""
    a, b = _pair(case, monkeypatch, parts="fuse")
    rng = np.random.default_rng(4242)
    checked = 0
    for _ in range(24):
        i, j = (int(x) for x in rng.integers(0, a.n, 2))
        if i == j:
            continue
        snap_a, snap_b = a.P.copy(), b.P.copy()
        ka = a._key()
        kb = b._key()
        assert ka == kb
        oka, ra = a._try_swap(i, j, ka)
        okb, rb = b._try_swap(i, j, kb)
        assert oka == okb and ra == rb
        assert np.array_equal(a.P, b.P)
        checked += 1
        if not oka:
            assert np.array_equal(a.P, snap_a)
            assert np.array_equal(b.P, snap_b)
    assert checked > 0


@pytest.mark.parametrize("case", CASES, ids=_IDS)
@pytest.mark.parametrize("parts", ["fuse", "hp", "fuse,hp"])
def test_discrete_batch_bit_exact(case, parts, monkeypatch):
    a, b = _pair(case, monkeypatch, parts=parts)
    a._build_swappable()
    b._build_swappable()
    dl = time.time() + 300.0
    ka, kb = a._key(), b._key()
    assert ka == kb
    for _ in range(3):
        acc_a, ka = a._discrete_batch(ka, dl)
        acc_b, kb = b._discrete_batch(kb, dl)
        assert acc_a == acc_b
        assert ka == kb
        assert np.array_equal(a.P, b.P)


@pytest.mark.parametrize("case", CASES[:4], ids=_IDS[:4])
def test_matching_batch_bit_exact(case, monkeypatch):
    a, b = _pair(case, monkeypatch)
    a._build_swappable()
    b._build_swappable()
    dl = time.time() + 300.0
    ka, kb = a._key(), b._key()
    acc_a, ka = a._matching_batch(ka, dl)
    acc_b, kb = b._matching_batch(kb, dl)
    assert acc_a == acc_b and ka == kb
    assert np.array_equal(a.P, b.P)


# --------------------------------------------------------------------------
# 4. end to end: whole `run` / `refine_positions` / `refine_prediction`
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=_IDS)
@pytest.mark.parametrize("parts", ["fuse", "hp", "fuse,hp"])
def test_run_bit_exact(case, parts, monkeypatch):
    a, b = _pair(case, monkeypatch, parts=parts)
    oa = a.run(time.time() + 300.0)
    ob = b.run(time.time() + 300.0)
    assert _digest(oa) == _digest(ob)


def _e2e(case, span, disc, monkeypatch, parts=None, fn="positions"):
    inst = build_instance(**case)
    pred = np.asarray([list(x) for x in inst.rects], dtype=np.float64)
    monkeypatch.setenv("PARTNER_REFINE_KERNEL", "numba")
    monkeypatch.setenv("PARTNER_REFINE_KERNEL_QSWEEP", "1")
    if disc:
        monkeypatch.setenv("PARTNER_REFINE_KERNEL_DISC", "1")
        if parts is not None:
            monkeypatch.setenv("PARTNER_REFINE_DISC_PARTS", parts)
    else:
        monkeypatch.delenv("PARTNER_REFINE_KERNEL_DISC", raising=False)
        monkeypatch.delenv("PARTNER_REFINE_DISC_PARTS", raising=False)
    dl = time.time() + span
    opt = make_optimizer(inst, seed=3, deadline=dl)
    if fn == "positions":
        r0 = rf._Refiner(opt, pred, seed=11)
        r0.legalize(60, deadline=time.time() + 60.0)
        out = rf.refine_positions(opt, r0.P.copy(), dl, seed=11)
    else:
        out = rf.refine_prediction(opt, pred, dl, seed=11)
    return None if out is None else np.asarray(out, dtype=np.float64)


@pytest.mark.parametrize("case", [CASES[1], CASES[3], CASES[5]],
                         ids=["n40s3", "n64s11", "n50s17"])
def test_refine_positions_bit_exact(case, monkeypatch):
    span = 120.0
    a1 = _e2e(case, span, False, monkeypatch)
    a2 = _e2e(case, span, False, monkeypatch)
    if a1 is None or a2 is None or not np.array_equal(a1, a2):
        pytest.skip("python path not deadline-converged at this span")
    for parts in ("fuse", "hp", "fuse,hp"):
        b = _e2e(case, span, True, monkeypatch, parts=parts)
        assert b is not None and _digest(b) == _digest(a1), parts


@pytest.mark.parametrize("case", [dict(n=40, seed=0), dict(n=64, seed=11)],
                         ids=["n40s0", "n64s11"])
def test_refine_prediction_bit_exact(case, monkeypatch):
    span = 90.0
    a1 = _e2e(case, span, False, monkeypatch, fn="prediction")
    a2 = _e2e(case, span, False, monkeypatch, fn="prediction")
    if a1 is None or a2 is None or not np.array_equal(a1, a2):
        pytest.skip("python path not deadline-converged at this span")
    b = _e2e(case, span, True, monkeypatch, fn="prediction")
    assert b is not None and _digest(b) == _digest(a1)


# --------------------------------------------------------------------------
# 5. warmup compiles the new entry points and never raises
# --------------------------------------------------------------------------
def test_warm_process_covers_disc(monkeypatch):
    rnk._WARMED = False
    rnk._WARMED_Q = False
    rnk._WARMED_D = False
    try:
        assert rnk.warm_process(qsweep=False, disc=True) is True
        assert rnk._WARMED_D is True
    finally:
        rnk._WARMED = False
        rnk._WARMED_Q = False
        rnk._WARMED_D = False
        rnk.warm_process(qsweep=True, disc=True)


# --------------------------------------------------------------------------
# 6. PARTNER_REFINE_PROF_DISC: the sub-buckets partition `discrete` and leave
#    the parent record's invariant alone
# --------------------------------------------------------------------------
def test_prof_disc_is_none_by_default(monkeypatch):
    r = _refiner(CASES[1], False, monkeypatch, legalize=False)
    assert r._dprof is None


def test_prof_disc_needs_the_parent_flag(monkeypatch):
    monkeypatch.setenv("PARTNER_REFINE_PROF_DISC", "1")
    r = _refiner(CASES[1], False, monkeypatch, legalize=False)
    assert r._prof is None and r._dprof is None


def test_prof_disc_sub_buckets_partition_discrete(monkeypatch, tmp_path):
    span = 45.0
    inst = build_instance(n=40, seed=0)
    pred = np.asarray([list(x) for x in inst.rects], dtype=np.float64)

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
             "PARTNER_REFINE_PROF_DISC": "1",
             "PARTNER_REFINE_PROF_FILE": str(path)})
    assert b is not None and np.array_equal(a1, b)
    import json
    recs = [json.loads(ln) for f in sorted(tmp_path.glob("prof.jsonl.*"))
            for ln in f.read_text().splitlines() if ln]
    assert recs
    for rec in recs:
        # the parent invariant the PROF test asserts must still hold
        total = sum(rec[k] for k in rf._PROF_KEYS) + rec["other"]
        assert abs(total - rec["span"]) < 1e-3 * max(rec["span"], 1.0)
        # and the sub-buckets partition `discrete`, not `span`
        # every field is `round(t, 6)`, so the partition can only be checked
        # to the accumulated rounding of the 17 sub-keys plus the remainder
        sub = sum(rec["d_" + k] for k in rf._DPROF_KEYS)
        tol = 1e-6 * (len(rf._DPROF_KEYS) + 2)
        assert sub <= rec["discrete"] + tol
        assert abs(sub + rec["d_unattributed"] - rec["discrete"]) < tol
        for k in rf._DPROF_KEYS:
            assert 0.0 <= rec["df_" + k] <= 1.0
