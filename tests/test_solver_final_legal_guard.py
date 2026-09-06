"""`PARTNER_FINAL_LEGAL_GUARD` (default ON): the last-line hard-legality
guard in `contest_optimizer.solve()`.

Why it exists: the evaluator scores an infeasible layout at the x8
`M_PENALTY`, so one illegal output costs more than any quality knob can win
back.  The column backbone is overlap-free by construction, but the shipped
pipeline runs eight post-passes after it (`_column_edge_seat` -> `_pick_best`
-> `_violation_kill` -> `_coord_polish` -> `_final_seat` -> `_tag_compress`
-> `_wall_repair_final` -> `_final_area_guard`), and this is the single place
where one check covers all of them.

Provenance (2026-08-28 synthetic stress sweep,
/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/stress/results.csv): all 37
infeasible results in a 106-instance adversarial sweep -- including
`fixedheavy_n21_s1`, which was first reported as a genuine solver bug -- are
INPUT artifacts: `tests.synth_instances.build_instance` scatters its
preplaced obstacles without a mutual-overlap check, so no legal layout
exists.  The guard is defence-in-depth, not a fix for that; the last test
below pins the unsatisfiability so the classification cannot be lost again.

What must hold:
  * legal layout  -> PURE CHECK: the SAME object comes back (the shipped
    output is bit-exact; the solver itself is deadline-driven and therefore
    not run-to-run reproducible, so object identity is the only sound
    bit-exactness proof).
  * illegal layout with a legal fallback -> the fallback ships and one
    `[legal-guard]` line goes to stderr.
  * illegal layout, no legal candidate anywhere -> the ORIGINAL comes back;
    the guard can never make an output worse than it already was.
  * `row_fallback` is lazy: never called on the legal path.
  * `=0` disables it; the input is returned untouched.
  * the primitives mirror the evaluator exactly (touching edges legal,
    1e-6 overlap tolerance, 1e-4 fixed/preplaced dimension tolerance, and
    the 1% area test SKIPS fixed/preplaced blocks).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import contest_optimizer as CO                     # noqa: E402

_ENV = ("PARTNER_FINAL_LEGAL_GUARD",)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# fixtures: a 4-block instance, one soft / one fixed-shape / one preplaced
# --------------------------------------------------------------------------
def _instance():
    """areas, constraints (fixed, preplaced, mib, cluster, boundary),
    target_positions.  Block 1 is fixed-shape (2x8 = 16, deliberately NOT
    equal to its 10.0 area target so the soft-area skip is exercised);
    block 2 is preplaced at (10, 0, 2, 5)."""
    areas = np.array([12.0, 10.0, 10.0, 6.0])
    cons = np.zeros((4, 5))
    cons[1, 0] = 1.0            # fixed shape
    cons[2, 1] = 1.0            # preplaced
    tpos = np.full((4, 4), -1.0)
    tpos[1] = (-1.0, -1.0, 2.0, 8.0)
    tpos[2] = (10.0, 0.0, 2.0, 5.0)
    return areas, cons, tpos


def _legal_layout():
    """Four disjoint rectangles honouring both hard constraints."""
    return [(0.0, 0.0, 3.0, 4.0),      # soft, 12
            (4.0, 0.0, 2.0, 8.0),      # fixed 2x8
            (10.0, 0.0, 2.0, 5.0),     # preplaced, exact
            (0.0, 5.0, 3.0, 2.0)]      # soft, 6


def _overlapping_layout():
    """Same blocks, but the two soft blocks are driven on top of each other."""
    lay = _legal_layout()
    lay[3] = (0.5, 0.5, 3.0, 2.0)      # overlaps block 0 by 2.5 x 1.5
    return lay


def _guard(result, fallbacks=(), row_fallback=None):
    areas, cons, tpos = _instance()
    return CO._final_legal_guard(result, fallbacks, 4, areas, cons, tpos,
                                 row_fallback=row_fallback)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def test_overlap_count_matches_evaluator_semantics():
    """`check_overlap` counts a pair only when BOTH axes overlap by more
    than 1e-6; touching edges and sub-tolerance slivers are legal."""
    touching = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0)]
    assert CO._overlap_count(touching, 2) == 0

    sliver = [(0.0, 0.0, 2.0, 2.0), (2.0 - 1e-9, 0.0, 2.0, 2.0)]
    assert CO._overlap_count(sliver, 2) == 0

    real = [(0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 2.0, 2.0)]
    assert CO._overlap_count(real, 2) == 1

    # one-axis-only overlap is NOT an overlap
    side_by_side = [(0.0, 0.0, 2.0, 2.0), (5.0, 1.0, 2.0, 2.0)]
    assert CO._overlap_count(side_by_side, 2) == 0

    assert CO._overlap_count(_overlapping_layout(), 4) == 1
    assert CO._overlap_count(_legal_layout(), 4) == 0


def test_overlap_count_rejects_malformed_input():
    assert CO._overlap_count([(0.0, 0.0, 1.0, 1.0)], 4) == -1


def test_hard_dims_mirror_the_evaluator_tolerance():
    areas, cons, tpos = _instance()
    lay = _legal_layout()
    assert CO._hard_dims_ok(lay, cons, tpos, 4)

    # inside the 1e-4 tolerance -> still legal
    ok = list(lay)
    ok[1] = (4.0, 0.0, 2.0 + 5e-5, 8.0)
    assert CO._hard_dims_ok(ok, cons, tpos, 4)

    # fixed-shape dimension drift beyond tolerance
    bad_w = list(lay)
    bad_w[1] = (4.0, 0.0, 2.5, 8.0)
    assert not CO._hard_dims_ok(bad_w, cons, tpos, 4)

    # preplaced LOCATION drift beyond tolerance
    bad_xy = list(lay)
    bad_xy[2] = (10.5, 0.0, 2.0, 5.0)
    assert not CO._hard_dims_ok(bad_xy, cons, tpos, 4)

    # no constraints / no targets -> nothing to check (evaluator returns 0)
    assert CO._hard_dims_ok(lay, None, tpos, 4)
    assert CO._hard_dims_ok(lay, cons, None, 4)


def test_soft_area_check_skips_fixed_and_preplaced():
    """Block 1 is fixed-shape with w*h = 16 against a 10.0 area target and
    block 2 is preplaced with w*h = 10 against 10.0; the evaluator excludes
    both from the 1% test, so the layout is area-legal."""
    areas, cons, tpos = _instance()
    assert CO._soft_area_ok(_legal_layout(), areas, cons, 4)

    # a SOFT block outside the 1% tolerance does fail
    bad = _legal_layout()
    bad[0] = (0.0, 0.0, 3.0, 4.5)       # 13.5 vs 12.0
    assert not CO._soft_area_ok(bad, areas, cons, 4)

    # ... and it fails the whole predicate
    assert not CO._legal_ok(bad, 4, areas, cons, tpos)


def test_legal_ok_rejects_structurally_broken_layouts():
    areas, cons, tpos = _instance()
    assert CO._legal_ok(_legal_layout(), 4, areas, cons, tpos)
    assert not CO._legal_ok(None, 4, areas, cons, tpos)
    assert not CO._legal_ok(_legal_layout()[:3], 4, areas, cons, tpos)
    nan = _legal_layout()
    nan[0] = (float("nan"), 0.0, 3.0, 4.0)
    assert not CO._legal_ok(nan, 4, areas, cons, tpos)
    zero = _legal_layout()
    zero[0] = (0.0, 0.0, 0.0, 4.0)
    assert not CO._legal_ok(zero, 4, areas, cons, tpos)


# --------------------------------------------------------------------------
# guard semantics
# --------------------------------------------------------------------------
def test_no_op_on_a_legal_layout_returns_the_same_object():
    """Bit-exactness proof for the shipped path: identity, not equality."""
    lay = _legal_layout()
    assert _guard(lay, fallbacks=(_legal_layout(),)) is lay


def test_row_fallback_is_lazy_on_the_legal_path():
    calls = []

    def _row():
        calls.append(1)
        return _legal_layout()

    lay = _legal_layout()
    assert _guard(lay, row_fallback=_row) is lay
    assert calls == []


def test_disabled_by_env_returns_the_input_untouched(monkeypatch):
    bad = _overlapping_layout()
    for off in ("0", "false", "False", "off"):
        monkeypatch.setenv("PARTNER_FINAL_LEGAL_GUARD", off)
        assert _guard(bad, fallbacks=(_legal_layout(),)) is bad


def test_fires_on_an_injected_overlap_and_ships_a_legal_fallback(capsys):
    areas, cons, tpos = _instance()
    bad = _overlapping_layout()
    good = _legal_layout()

    out = _guard(bad, fallbacks=(good,))

    assert out is not bad
    assert CO._legal_ok(out, 4, areas, cons, tpos)
    assert [tuple(r) for r in out] == [tuple(r) for r in good]
    assert "[legal-guard]" in capsys.readouterr().err


def test_skips_illegal_fallbacks_and_reaches_the_row_fallback(capsys):
    """The pool/column fallbacks can be illegal too (they went through the
    same post-passes); the guard must keep walking until something PASSES."""
    areas, cons, tpos = _instance()
    bad = _overlapping_layout()
    also_bad = _overlapping_layout()
    calls = []

    def _row():
        calls.append(1)
        return _legal_layout()

    out = _guard(bad, fallbacks=(also_bad, None), row_fallback=_row)

    assert calls == [1]
    assert CO._legal_ok(out, 4, areas, cons, tpos)
    assert "[legal-guard]" in capsys.readouterr().err


def test_never_degrades_when_no_candidate_is_legal(capsys):
    """An input whose PREPLACED obstacles overlap each other admits no legal
    layout at all.  The guard must then return the ORIGINAL object -- never
    swap in a row layout that would ALSO break the preplaced mandate."""
    areas = np.array([4.0, 4.0])
    cons = np.zeros((2, 5))
    cons[0, 1] = cons[1, 1] = 1.0                       # both preplaced
    tpos = np.array([[0.0, 0.0, 2.0, 2.0],
                     [1.0, 1.0, 2.0, 2.0]])             # mandated to overlap
    mandated = [(0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 2.0, 2.0)]

    assert not CO._legal_ok(mandated, 2, areas, cons, tpos)
    out = CO._final_legal_guard(
        mandated, (), 2, areas, cons, tpos,
        row_fallback=lambda: [(0.0, 0.0, 2.0, 2.0), (0.0, 2.0, 2.0, 2.0)])
    assert out is mandated
    assert "[legal-guard]" in capsys.readouterr().err


def test_a_raising_row_fallback_cannot_break_the_guard():
    bad = _overlapping_layout()

    def _boom():
        raise RuntimeError("fallback exploded")

    assert _guard(bad, fallbacks=(), row_fallback=_boom) is bad


# --------------------------------------------------------------------------
# provenance: the reported "genuine" stress failure is an impossible INPUT
# --------------------------------------------------------------------------
def test_fixedheavy_n21_s1_input_is_unsatisfiable():
    """`build_instance(n=21, seed=1, frac_fixed=0.6)` places its two default
    preplaced obstacles (blocks 15 and 20) so that their MANDATED rectangles
    overlap by 3.4988 x 2.7551.  Both are preplaced (`constraints[:, 1]`),
    not soft, so the evaluator's own hard constraints are contradictory and
    NO optimizer output can be feasible.  The solver's shipped layout matched
    the overlap exactly with preplaced_violations = 0, i.e. it obeyed the
    input."""
    from synth_instances import build_instance

    inst = build_instance(n=21, seed=1, frac_fixed=0.6)
    cons = np.asarray(inst.constraints, dtype=np.float64)
    tp = np.asarray(inst.target_positions, dtype=np.float64)

    assert cons[15, 1] != 0.0 and cons[20, 1] != 0.0
    assert cons[15, 0] == 0.0 and cons[20, 0] == 0.0

    (x1, y1, w1, h1), (x2, y2, w2, h2) = tp[15], tp[20]
    ox = min(x1 + w1, x2 + w2) - max(x1, x2)
    oy = min(y1 + h1, y2 + h2) - max(y1, y2)
    assert ox == pytest.approx(3.4988, abs=1e-3)
    assert oy == pytest.approx(2.7551, abs=1e-3)

    # the mandated placement is itself illegal -> the guard reports it and,
    # having nothing legal to swap in, leaves the layout alone
    mandated = [tuple(float(v) for v in tp[i]) if cons[i, 1] != 0.0
                else (0.0, 0.0, 1.0, 1.0) for i in range(21)]
    assert CO._overlap_count([mandated[15], mandated[20]], 2) == 1


# --------------------------------------------------------------------------
# area repair: rescue the layout instead of discarding it
#
# Provenance (2026-08-30, shadow v3 tid 85 at a x10 diagnostic budget): the
# shipped layout had soft block 32 at 24x13 = 312 against a 650 target, and
# BOTH fallbacks had inherited the same defect, so the guard could only reach
# the row layout (cost 9.999999).  Resizing the one offending block in free
# space keeps everything the layout earned.
# --------------------------------------------------------------------------
def _boxed_in_instance():
    """Block 0 (area target 8) sits at 2x2 = 4 with a neighbour flush against
    each of its four sides, so every resize variant -- keep w / keep h /
    uniform, anchored at (x, y) and at the far corner -- clashes."""
    areas = np.array([8.0, 4.0, 4.0, 4.0, 4.0])
    cons = np.zeros((5, 5))
    tpos = np.full((5, 4), -1.0)
    lay = [(10.0, 10.0, 2.0, 2.0),      # offender: 4 vs 8
           (8.0, 10.0, 2.0, 2.0),       # left
           (12.0, 10.0, 2.0, 2.0),      # right
           (10.0, 8.0, 2.0, 2.0),       # below
           (10.0, 12.0, 2.0, 2.0)]      # above
    return areas, cons, tpos, lay


def test_area_repair_resizes_the_offender_and_leaves_the_rest_alone(capsys):
    areas, cons, tpos = _instance()
    bad = _legal_layout()
    bad[0] = (0.0, 0.0, 3.0, 2.0)               # 6.0 against a 12.0 target
    assert not CO._legal_ok(bad, 4, areas, cons, tpos)

    sentinel = _legal_layout()                  # a legal fallback exists ...
    sentinel[3] = (0.0, 6.0, 3.0, 2.0)          # ... distinguishable from it
    assert CO._legal_ok(sentinel, 4, areas, cons, tpos)
    out = CO._final_legal_guard(bad, (sentinel,), 4, areas, cons, tpos)

    # ... and must NOT be what ships: the repair outranks the fallback chain
    assert [tuple(r) for r in out] != [tuple(r) for r in sentinel]
    assert CO._legal_ok(out, 4, areas, cons, tpos)
    # keep w = 3.0, grow upward into the free space below block 3
    assert tuple(out[0]) == pytest.approx((0.0, 0.0, 3.0, 4.0))
    assert [tuple(r) for r in out[1:]] == [tuple(r) for r in bad[1:]]
    assert "shipping area-repaired layout" in capsys.readouterr().err


def test_area_repair_never_moves_fixed_or_preplaced_blocks():
    """The 1% test skips them, so they are never offenders; the repair must
    not resize them into legality either (that would break `_hard_dims_ok`)."""
    areas, cons, tpos = _instance()
    bad = _legal_layout()
    bad[0] = (0.0, 0.0, 3.0, 2.0)
    fixed = CO._repair_soft_areas(bad, 4, areas, cons)
    assert fixed is not None
    assert tuple(fixed[1]) == tuple(bad[1])     # fixed-shape 2x8 untouched
    assert tuple(fixed[2]) == tuple(bad[2])     # preplaced untouched
    assert CO._hard_dims_ok(fixed, cons, tpos, 4)


def test_area_repair_declines_when_the_offender_is_boxed_in(capsys):
    areas, cons, tpos, lay = _boxed_in_instance()
    assert not CO._legal_ok(lay, 5, areas, cons, tpos)
    assert CO._repair_soft_areas(lay, 5, areas, cons) is None

    # -> the pre-existing fallback behaviour, unchanged
    good = [(0.0, 0.0, 2.0, 4.0), (4.0, 0.0, 2.0, 2.0), (7.0, 0.0, 2.0, 2.0),
            (10.0, 0.0, 2.0, 2.0), (13.0, 0.0, 2.0, 2.0)]
    assert CO._legal_ok(good, 5, areas, cons, tpos)
    out = CO._final_legal_guard(lay, (good,), 5, areas, cons, tpos)
    assert [tuple(r) for r in out] == [tuple(r) for r in good]
    err = capsys.readouterr().err
    assert "verified-legal fallback" in err
    assert "area-repaired" not in err


def test_area_repair_declines_when_the_defect_is_not_area():
    """An overlap is not an area violation: the repair must report failure
    rather than hand `_legal_ok` a layout that still overlaps."""
    areas, cons, _tpos = _instance()
    assert CO._repair_soft_areas(_overlapping_layout(), 4, areas, cons) is None


def test_area_repair_is_never_reached_on_the_legal_path(monkeypatch):
    """Bit-exactness of the shipped path: the fast path returns the SAME
    object without so much as consulting the repair."""
    def _boom(*_a, **_k):
        raise AssertionError("repair must not run on a legal layout")

    monkeypatch.setattr(CO, "_repair_soft_areas", _boom)
    lay = _legal_layout()
    assert _guard(lay, fallbacks=(_legal_layout(),)) is lay
