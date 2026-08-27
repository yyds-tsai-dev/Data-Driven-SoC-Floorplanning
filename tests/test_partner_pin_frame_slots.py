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
for _p in (ROOT / "partner", ROOT / "FloorSet",
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
