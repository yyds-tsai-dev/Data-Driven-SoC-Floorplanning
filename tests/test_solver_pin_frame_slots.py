"""`PARTNER_PIN_FRAME_SLOTS=k` (default 0): the pin-frame ladder as a
PORTFOLIO SLOT instead of a global switch.

`PARTNER_PIN_FRAME` (see tests/test_partner_pin_frame.py) turns every
reserved refine worker of every case onto the pinned ladder.  Measured that
way it is net-negative -- official -0.0022 / shadow-v3 +0.0121 -- and the
per-case decomposition says why: the locked cases win big (official tid 86
1.316 -> 1.054) while cases with zero boundary violations lose just as big
(tid 99 1.187 -> 1.318), because for them the clamp only narrows the packing.
Nothing about that is a failure of the mechanism; it is a failure of applying
it unconditionally.

SLOTS puts k pinned workers and (n_ref - k) shipped workers in the SAME
candidate pool and lets `_parallel_solve`'s existing arbitration pick per
case, so a losing pinned candidate is simply not selected.

What must hold:
  * off path (k unset/0) -> `pin_frame_flags` is (False, False) and the
    payload keeps its shipped shape, so the solver is bit-exact;
  * the flag travels in the PAYLOAD, not the environment (the fork pool
    inherits env, so env cannot distinguish two workers of the same case) --
    mirroring `opt._tag_anchor` / `PARTNER_TAG_ANCHOR_EXTRA`;
  * k is clamped so at least one shipped worker always survives;
  * slots are taken from the TAIL of the spec list (the tag-anchored extras
    first), so a slot costs no fresh prediction;
  * a case whose preplaced blocks carry no boundary tag has no `lock_*` at
    all -- `_has_tag_locks` is False and no slot is spent there.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as CSL                 # noqa: E402
import layout_refiner as LR                       # noqa: E402
from column_sa_legalizer import _ColumnOptimizer  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in ("PARTNER_PIN_FRAME", "PARTNER_PIN_FRAME_RETRY",
              "PARTNER_PIN_FRAME_SLOTS", "PARTNER_PIN_FRAME_SLOT_MODE",
              "PARTNER_PINFRAME_DEBUG",
              "PARTNER_TAG_ANCHOR", "PARTNER_TAG_PACK",
              "PARTNER_ANYTIME_LADDER", "PARTNER_RUNG05",
              "PARTNER_REFINE_GUARD", "PARTNER_LEGAL_ADMIT",
              "PARTNER_WALL_REPAIR", "PARTNER_FRAME_SCALE_LADDER",
              "PARTNER_FRAME_SCALE_SET", "REFINER_DEBUG"):
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# synthetic instances (same shapes as tests/test_partner_pin_frame.py)
# ---------------------------------------------------------------------------

def _opt(rects, cons, tpos, budget: float = 20.0):
    at = torch.tensor([float(w * h) for _x, _y, w, h in rects])
    z3 = torch.zeros((1, 3))
    return _ColumnOptimizer(rects, at, cons, tpos, z3, z3,
                            torch.zeros((1, 2)),
                            time.time() + budget, seed=0)


def _left_tagged():
    """Block 0 preplaced at x=[0,20] and tagged LEFT; blocks 1/2 predicted
    PAST the tag line, so the shipped rung 0 takes the wall from the
    prediction (-20) and the tag can never seat."""
    rects = [
        (0.0, 0.0, 20.0, 40.0),
        (-20.0, 0.0, 20.0, 20.0),
        (-20.0, 20.0, 20.0, 20.0),
    ]
    cons = torch.zeros((len(rects), 5))
    cons[0, 1] = 1.0                # preplaced
    cons[0, 4] = 1.0                # boundary: left
    tpos = torch.full((len(rects), 4), -1.0)
    tpos[0] = torch.tensor([0.0, 0.0, 20.0, 40.0])
    return _opt(rects, cons, tpos), np.asarray(rects, dtype=np.float64)


def _soft_tagged():
    """A boundary tag on a SOFT (movable) block: the block can walk to the
    wall, so `_anchor_frame_to_tags` records no lock and pinning is a
    no-op -- the case class the slot must not be spent on."""
    rects = [
        (0.0, 0.0, 20.0, 40.0),
        (20.0, 0.0, 20.0, 40.0),
    ]
    cons = torch.zeros((len(rects), 5))
    cons[0, 4] = 1.0                # boundary: left, but kind 0
    tpos = torch.full((len(rects), 4), -1.0)
    return _opt(rects, cons, tpos), np.asarray(rects, dtype=np.float64)


def _no_overlap(P):
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > 1e-7) & (oy > 1e-7)
    np.fill_diagonal(bad, False)
    return not bad.any()


# ---------------------------------------------------------------------------
# pin_frame_flags: the per-call resolution
# ---------------------------------------------------------------------------

def test_flags_off_without_env_or_payload():
    opt, _P = _left_tagged()
    assert LR.pin_frame_flags(opt) == (False, False)


def test_payload_flag_turns_one_call_on_without_env():
    """The whole point: no env var is set, yet THIS optimizer runs pinned."""
    opt, _P = _left_tagged()
    opt._pin_frame = "1"
    assert LR.pin_frame_flags(opt) == (True, True)


def test_payload_min_mode_pins_only_the_rung0_corner():
    opt, _P = _left_tagged()
    opt._pin_frame = "min"
    assert LR.pin_frame_flags(opt) == (True, False)


def test_payload_empty_string_is_off():
    """`_worker_refine` never sets the attribute for a shipped payload, but
    an empty value must be off too (it is the spec-list default)."""
    opt, _P = _left_tagged()
    opt._pin_frame = ""
    assert LR.pin_frame_flags(opt) == (False, False)


def test_global_env_still_applies_to_every_call(monkeypatch):
    """SLOTS is additive: the global switch keeps working for the arms that
    already measured it."""
    monkeypatch.setattr(LR, "_PIN_FRAME_MODE", "1")
    opt, _P = _left_tagged()
    assert LR.pin_frame_flags(opt) == (True, True)


# ---------------------------------------------------------------------------
# the payload flag actually reaches the ladder
# ---------------------------------------------------------------------------

def test_ladder_honours_the_payload_flag_end_to_end():
    """Same instance, same process, no env var, no module reload: only the
    per-payload attribute differs, and only the pinned run seats the wall on
    the preplaced block's edge."""
    opt_a, P = _left_tagged()
    out_a = LR.refine_prediction(opt_a, P, time.time() + 10.0, seed=0)

    opt_b, P = _left_tagged()
    opt_b._pin_frame = "1"
    out_b = LR.refine_prediction(opt_b, P, time.time() + 10.0, seed=0)

    assert out_a is not None and out_b is not None
    Qa = np.asarray(out_a, dtype=np.float64)
    Qb = np.asarray(out_b, dtype=np.float64)
    assert _no_overlap(Qa) and _no_overlap(Qb)
    # shipped: the wall overshoots the tag line
    assert float(Qa[:, 0].min()) < -1e-6
    # pinned: the wall IS the tag line and the preplaced block never moved
    assert float(Qb[:, 0].min()) == pytest.approx(0.0, abs=1e-6)
    assert Qb[0, 0] == pytest.approx(0.0, abs=1e-6)
    for i in range(opt_b.n):
        assert abs(Qb[i, 2] * Qb[i, 3] - opt_b.areas[i]) / opt_b.areas[i] <= 0.01


def test_worker_refine_accepts_both_payload_arities(monkeypatch):
    """`_worker_refine` is fed 12-wide payloads by the reserved-worker path
    and 11-wide ones by phase B and the offline trace probes."""
    seen = []

    def _spy(opt, pred, deadline, seed=0, _depth=0):
        seen.append(getattr(opt, "_pin_frame", None))
        return np.asarray(pred, dtype=np.float64)

    monkeypatch.setattr(LR, "refine_prediction", _spy)

    opt, P = _left_tagged()
    cons = torch.zeros((opt.n, 5)).numpy()
    base = (P, np.asarray([float(w * h) for _x, _y, w, h in P]),
            cons, None, None, None, None,
            time.time() + 5.0, 0, 1.0, False)

    assert CSL._worker_refine(base) is not None          # 11-wide, shipped
    assert CSL._worker_refine(base + ("1",)) is not None  # 12-wide, pinned
    assert CSL._worker_refine(base + ("",)) is not None   # 12-wide, shipped
    assert seen == [None, "1", None]


# ---------------------------------------------------------------------------
# slot bookkeeping
# ---------------------------------------------------------------------------

def _specs(n_plain, n_anchor):
    return ([(f"p{k}", False, "") for k in range(n_plain)]
            + [(f"p{k}", True, "") for k in range(n_anchor)])


def test_zero_slots_is_identity():
    s = _specs(6, 3)
    assert CSL._pin_slot_specs(s, 0) is s
    assert CSL._pin_slot_specs(s, -1) is s


def test_slots_convert_the_anchored_tail_first():
    """k=2 with the shipping portfolio (6 plain + 3 tag-anchored): the two
    converted slots are anchored EXTRAS, so no fresh prediction is lost."""
    out = CSL._pin_slot_specs(_specs(6, 3), 2)
    assert [p for _P, _a, p in out] == ["", "", "", "", "", "", "", "1", "1"]
    # the anchored flag of a converted slot is preserved
    assert [a for _P, a, _p in out] == [False] * 6 + [True] * 3
    # every prediction the shipped portfolio refined is still refined
    assert [P for P, _a, _p in out] == [P for P, _a, _p in _specs(6, 3)]


def test_slots_spill_into_the_plain_draws_once_extras_run_out():
    out = CSL._pin_slot_specs(_specs(6, 3), 4)
    assert [p for _P, _a, p in out] == ["", "", "", "", "", "1", "1", "1", "1"]


def test_k_is_clamped_to_keep_one_shipped_worker():
    out = CSL._pin_slot_specs(_specs(6, 3), 99)
    assert [p for _P, _a, p in out] == [""] + ["1"] * 8


def test_slot_mode_is_carried_through():
    out = CSL._pin_slot_specs(_specs(6, 3), 1, "min")
    assert out[-1][2] == "min"


# ---------------------------------------------------------------------------
# the instance-statistic gate (no case identity)
# ---------------------------------------------------------------------------

def test_has_tag_locks_true_for_a_tagged_preplaced_block():
    opt, P = _left_tagged()
    assert CSL._has_tag_locks(opt) is True
    # ... and that is exactly when `_anchor_frame_to_tags` records a lock
    r = LR._Refiner(opt, P, 0)
    r._anchor_frame_to_tags()
    assert any(v is not None for v in (r.lock_xmin, r.lock_xmax,
                                       r.lock_ymin, r.lock_ymax))


def test_has_tag_locks_false_when_only_soft_blocks_are_tagged():
    opt, P = _soft_tagged()
    assert CSL._has_tag_locks(opt) is False
    r = LR._Refiner(opt, P, 0)
    r._anchor_frame_to_tags()
    assert all(v is None for v in (r.lock_xmin, r.lock_xmax,
                                   r.lock_ymin, r.lock_ymax))
    assert r._pin_frame_to_locks() is False


# ---------------------------------------------------------------------------
# PARTNER_PIN_FRAME_SLOT_FROM=plain -- which worker pays for the slot
#
# The anchored source was measured to cost the shadow suites (v3 +0.0020 /
# a1 +0.0056, hpwl-driven with flat v_rel): the pool loses the tag-anchored
# candidate that used to win.  `plain` gives the slot the weakest PLAIN draw
# instead, so all PARTNER_TAG_ANCHOR_EXTRA anchored workers survive.
# ---------------------------------------------------------------------------

def test_plain_source_converts_the_weakest_plain_draw():
    """6 plain + 3 anchored, k=1: `specs[n_base-1]` is the one converted and
    every anchored extra is still shipped."""
    out = CSL._pin_slot_specs(_specs(6, 3), 1, source="plain", n_base=6)
    assert [p for _P, _a, p in out] == ["", "", "", "", "", "1", "", "", ""]
    assert [a for _P, a, _p in out] == [False] * 6 + [True] * 3


def test_plain_source_walks_downward_and_never_touches_the_extras():
    out = CSL._pin_slot_specs(_specs(6, 3), 2, source="plain", n_base=6)
    assert [p for _P, _a, p in out] == ["", "", "", "", "1", "1", "", "", ""]
    # k larger than the plain block still leaves one shipped PLAIN draw
    out = CSL._pin_slot_specs(_specs(6, 3), 99, source="plain", n_base=6)
    assert [p for _P, _a, p in out] == [""] + ["1"] * 5 + ["", "", ""]


def test_plain_source_keeps_positions_so_seeds_and_vw_mix_are_untouched():
    """A converted spec keeps its INDEX: `seed + 301 + 7*k` and the
    every-third-worker PARTNER_REFINE_VW_MIX rule are index-based."""
    base = _specs(6, 3)
    out = CSL._pin_slot_specs(base, 2, source="plain", n_base=6)
    assert [P for P, _a, _p in out] == [P for P, _a, _p in base]
    assert [a for _P, a, _p in out] == [a for _P, a, _p in base]


def test_plain_source_with_no_room_is_identity():
    """One plain draw only: converting it would leave the pool with no
    shipped plain draw at all, so nothing is converted."""
    s = _specs(1, 3)
    assert CSL._pin_slot_specs(s, 2, source="plain", n_base=1) is s


def test_default_source_is_the_measured_anchored_behaviour():
    assert (CSL._pin_slot_specs(_specs(6, 3), 2)
            == CSL._pin_slot_specs(_specs(6, 3), 2, source="anchored"))
    # an unrecognised value falls back to anchored rather than failing shut
    assert (CSL._pin_slot_specs(_specs(6, 3), 2, source="typo")
            == CSL._pin_slot_specs(_specs(6, 3), 2))


def test_plain_mode_is_carried_through():
    out = CSL._pin_slot_specs(_specs(6, 3), 1, "min", source="plain", n_base=6)
    assert out[5][2] == "min"


# ---------------------------------------------------------------------------
# `_tag_lock_box_util` -- the lock-box utilization instance statistic
# ---------------------------------------------------------------------------

def _boxed(pred_span=40.0):
    """Block 0 preplaced at x=[0,20], y=[0,40] and tagged LEFT+BOTTOM; two
    soft blocks predicted to the RIGHT.  The lock box is
    [0, pred_xmax] x [0, pred_ymax]: xmin/ymin come from the preplaced
    block's edges, xmax/ymax stay on the prediction bbox."""
    rects = [
        (0.0, 0.0, 20.0, 40.0),
        (20.0, 0.0, 20.0, 20.0),
        (20.0, 20.0, 20.0, pred_span - 20.0),
    ]
    cons = torch.zeros((len(rects), 5))
    cons[0, 1] = 1.0                # preplaced
    cons[0, 4] = 1.0 + 8.0          # boundary: left | bottom
    tpos = torch.full((len(rects), 4), -1.0)
    tpos[0] = torch.tensor([0.0, 0.0, 20.0, 40.0])
    return _opt(rects, cons, tpos)


def test_lock_box_util_matches_the_hand_computed_box():
    opt = _boxed()
    # blocks: 800 + 400 + 400 = 1600; lock box = 40 x 40 = 1600
    assert opt.total_area == pytest.approx(1600.0)
    assert CSL._tag_lock_box_util(opt) == pytest.approx(1.0, abs=1e-9)


def test_lock_box_util_falls_with_a_looser_prediction_bbox():
    """Free sides come from the prediction bbox: stretching the prediction
    to twice the height (1600 -> 2400 of block area in a 40x80 lock box)
    drops the utilization from 1.00 to 0.75."""
    opt = _boxed(pred_span=80.0)
    assert opt.total_area == pytest.approx(2400.0)
    assert CSL._tag_lock_box_util(opt) == pytest.approx(0.75, abs=1e-9)


def test_lock_box_util_is_zero_without_a_tagged_preplaced_block():
    opt, _P = _soft_tagged()
    assert CSL._tag_lock_box_util(opt) == 0.0


def test_lock_box_util_never_raises_on_a_malformed_optimizer():
    class _Broken:
        n = 3
        kind = [2, 0, 0]
    assert CSL._tag_lock_box_util(_Broken()) == 0.0


# ---------------------------------------------------------------------------
# `_pin_slot_gate` -- the instance-statistic gate itself
# ---------------------------------------------------------------------------

def test_gate_is_open_by_default_on_a_tag_locked_instance():
    opt = _boxed()
    assert CSL._pin_slot_gate(opt, 2) == 2
    assert CSL._pin_slot_gate(opt, 2, 0.0, 0.0) == 2


def test_gate_is_shut_without_tag_locks_whatever_the_bounds():
    opt, _P = _soft_tagged()
    assert CSL._pin_slot_gate(opt, 2) == 0
    assert CSL._pin_slot_gate(opt, 2, 0.0, 2.0) == 0


def test_max_util_gate_drops_the_tight_lock_boxes():
    """util 1.0: a MAX bound below it shuts the gate, one above keeps it."""
    opt = _boxed()
    assert CSL._pin_slot_gate(opt, 2, 0.0, 0.80) == 0
    assert CSL._pin_slot_gate(opt, 2, 0.0, 1.50) == 2


def test_min_util_gate_drops_the_loose_lock_boxes():
    opt = _boxed(pred_span=80.0)          # util 0.75
    assert CSL._pin_slot_gate(opt, 2, 0.80, 0.0) == 0
    assert CSL._pin_slot_gate(opt, 2, 0.50, 0.0) == 2


def test_bounds_compose_into_a_band():
    tight, loose = _boxed(), _boxed(pred_span=80.0)
    assert CSL._pin_slot_gate(tight, 2, 0.40, 0.80) == 0   # util 1.00
    assert CSL._pin_slot_gate(loose, 2, 0.40, 0.80) == 2   # util 0.75


def test_zero_slots_stays_zero_through_the_gate():
    opt = _boxed()
    assert CSL._pin_slot_gate(opt, 0, 0.0, 2.0) == 0
    assert CSL._pin_slot_gate(opt, -1) == 0
