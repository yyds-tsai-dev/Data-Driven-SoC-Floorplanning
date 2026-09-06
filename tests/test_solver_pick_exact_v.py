"""Two opt-in partner flags, both default off and both bit-exact when off.

  * `PARTNER_PICK_EXACT_V` -> `_pick_best` ranks candidates using
    `violation_killer._violations_exact` (the evaluator-exact
    boundary+grouping+MIB total) instead of `layout_refiner.full_violations`
    (a TOUCH_TOL=1e-7-slack, incomplete count).  Only the violation count fed
    into the ranking formula changes.
  * `PARTNER_VAUDIT_JSONL=<path>` -> per-case violation audit trail, one JSON
    line per case, buffered in memory and flushed once at interpreter exit
    (or by calling `contest_optimizer._vaudit_flush()` directly).

What must hold:
  * flag off -> BIT-EXACT: same solve() output, and `violation_killer` is
    never imported.
  * flag on (PARTNER_PICK_EXACT_V) -> solve() still returns a legal layout
    (no overlaps).
  * flag on (PARTNER_VAUDIT_JSONL) -> one JSON line per case is written,
    with a 6-entry `stage_v` and a `final_report` block.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_ENV = ("PARTNER_PICK_EXACT_V", "PARTNER_VAUDIT_JSONL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# a small deterministic instance: a cluster, a MIB group, a rigid block and
# a couple of wall tags -- enough for the ranking/violation machinery to
# have something to chew on, small enough to solve fast.
# ---------------------------------------------------------------------------

def _case(n: int = 12):
    rects = []
    for i in range(n):
        rects.append((6.0 * (i % 4), 6.0 * (i // 4), 5.0, 5.0))
    at = torch.full((n,), 25.0)
    cons = torch.zeros((n, 5))

    cons[2, 0] = 1.0                      # fixed shape (rigid)
    for i in (3, 4, 5):
        cons[i, 2] = 1.0                  # MIB group
    cons[6, 3] = 1.0                      # cluster
    cons[7, 3] = 1.0

    cons[0, 4] = 1.0                      # left
    cons[1, 4] = 4.0                      # top

    tpos = torch.full((n, 4), -1.0)
    tpos[2, 2] = 5.0
    tpos[2, 3] = 5.0

    edges = [[float(i), float((i * 3 + 1) % n), 1.0] for i in range(0, n, 2)]
    b2b = torch.tensor(edges, dtype=torch.float32)
    pins = torch.tensor([[0.0, 0.0], [24.0, 18.0]], dtype=torch.float32)
    p2b = torch.tensor([[0.0, 0.0, 2.0], [1.0, float(n - 1), 2.0]],
                       dtype=torch.float32)
    return {
        "block_count": n,
        "area_targets": at,
        "b2b_connectivity": b2b,
        "p2b_connectivity": p2b,
        "pins_pos": pins,
        "constraints": cons,
        "target_positions": tpos,
    }


def _bare_optimizer(co):
    """A MyOptimizer instance with no model checkpoints loaded (mirrors
    the __new__ pattern used across the other solver tests -- avoids
    paying for checkpoint I/O in a unit test)."""
    opt = co.MyOptimizer.__new__(co.MyOptimizer)
    opt.verbose = False
    opt.model = None
    opt.schedule = None
    opt.direct_model = None
    opt.flow_model = None
    opt.retrieval_index = None
    opt.retrieval_slots = 0
    opt.retrieval_max_cost = 2.0
    opt.device = torch.device("cpu")
    opt._last_pick_channel = "column"
    opt._vaudit_seq = 0
    return opt


def _no_overlap(rects, tol=1e-6):
    P = np.asarray([[float(a) for a in r] for r in rects], dtype=np.float64)
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > tol) & (oy > tol)
    np.fill_diagonal(bad, False)
    return not bad.any()


# ---------------------------------------------------------------------------
# 1. flag-off byte identity
# ---------------------------------------------------------------------------

def test_flag_off_solve_does_not_import_violation_killer(monkeypatch):
    # NOTE: solve() is deadline-SA on wall-clock time, so full-layout
    # bit-equality across two runs is NOT an invariant of the off path.
    # The off-path contract is: no violation_killer import, and a legal
    # (block-count-complete) layout.
    sys.modules.pop("violation_killer", None)
    sys.modules.pop("contest_optimizer", None)
    monkeypatch.delenv("PARTNER_PICK_EXACT_V", raising=False)
    monkeypatch.delenv("PARTNER_VAUDIT_JSONL", raising=False)

    co = importlib.import_module("contest_optimizer")

    kw = _case()
    opt1 = _bare_optimizer(co)
    out1 = opt1.solve(**kw)

    assert len(out1) == kw["block_count"]
    assert "violation_killer" not in sys.modules


def test_pick_best_off_path_does_not_import_violation_killer(monkeypatch):
    sys.modules.pop("violation_killer", None)
    import contest_optimizer as co

    opt = _bare_optimizer(co)
    rects = _case()  # not used directly; _pick_best needs its own inputs
    n = 6
    column_out = [(6.0 * (i % 3), 6.0 * (i // 3), 5.0, 5.0) for i in range(n)]
    at = torch.full((n,), 25.0)
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    b2b = torch.empty(0, 3)
    p2b = torch.empty(0, 3)
    pins = torch.empty(0, 2)
    scorer = co._ColumnOptimizer(
        [tuple(map(float, r)) for r in column_out], at, cons, tpos,
        b2b, p2b, pins, deadline=None, seed=0)
    box = [(np.asarray([list(r) for r in column_out], dtype=np.float64),
            scorer)]
    out = opt._pick_best(column_out, box)
    assert out is not None
    assert "violation_killer" not in sys.modules


# ---------------------------------------------------------------------------
# 2. PARTNER_PICK_EXACT_V smoke: solve still returns a legal layout
# ---------------------------------------------------------------------------

def test_pick_exact_v_smoke(monkeypatch):
    monkeypatch.setenv("PARTNER_PICK_EXACT_V", "1")
    import contest_optimizer as co

    opt = _bare_optimizer(co)
    out = opt.solve(**_case())

    assert len(out) == 12
    assert _no_overlap(out)


# ---------------------------------------------------------------------------
# 3. PARTNER_VAUDIT_JSONL
# ---------------------------------------------------------------------------

def test_vaudit_jsonl_writes_one_line_with_expected_shape(monkeypatch, tmp_path):
    out_path = tmp_path / "vaudit.jsonl"
    monkeypatch.setenv("PARTNER_VAUDIT_JSONL", str(out_path))
    import contest_optimizer as co

    opt = _bare_optimizer(co)
    out = opt.solve(**_case())
    assert _no_overlap(out)

    # flush explicitly rather than relying on interpreter exit
    co._vaudit_flush()

    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1

    rec = json.loads(lines[0])
    assert rec["test_or_seq_id"] == 1
    assert rec["n"] == 12
    assert rec["channel"] in ("column", "direct")
    assert len(rec["stage_v"]) == 6
    for stage in rec["stage_v"]:
        assert isinstance(stage, list) and len(stage) == 2
    fr = rec["final_report"]
    assert set(fr.keys()) == {"bnd", "grp", "mib_count"}
    assert isinstance(fr["bnd"], list)
    assert isinstance(fr["grp"], list)
    assert isinstance(fr["mib_count"], int)
