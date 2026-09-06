"""The IC/DC energy has one job: rank layouts the way the official evaluator
ranks them, while staying differentiable.

The decode tests pin the by-construction constraints (exact area, MIB
uniformity, frozen fixed/preplaced geometry) -- those are what let the energy
drop three of the five constraint families entirely.  The HPWL and N_soft tests
are exact agreement with the evaluator's own functions.  The correlation test
needs the FloorSet validation set and is skipped without it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet", ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from icdc_engine import energy as EN          # noqa: E402


def _evaluator():
    path = ROOT / "scripts" / "iccad2026_evaluate.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("repo_ev_test", path)
        ev = importlib.util.module_from_spec(spec)
        sys.modules["repo_ev_test"] = ev
        spec.loader.exec_module(ev)
        return ev
    except Exception:
        return None


# ---------------------------------------------------------------------------
# decode: the by-construction constraint set
# ---------------------------------------------------------------------------
def _toy(B=2, N=8):
    area = torch.tensor([[10., 20., 30., 40., 50., 60., 70., 80.]] * B,
                        dtype=torch.float64)
    cons = torch.zeros((B, N, 5), dtype=torch.float64)
    cons[:, 0, 0] = 1.0                      # fixed shape
    cons[:, 1, 1] = 1.0                      # preplaced
    cons[:, 4:7, 2] = 1.0                    # one MIB group of three
    tp = torch.full((B, N, 4), -1.0, dtype=torch.float64)
    tp[:, 0, 2:] = torch.tensor([2.5, 4.0], dtype=torch.float64)   # 10
    tp[:, 1, :] = torch.tensor([3.0, 7.0, 4.0, 5.0], dtype=torch.float64)  # 20
    scale = torch.sqrt(area.sum(dim=1))
    return area, cons, tp, scale


def test_decode_area_is_exact_for_soft_blocks():
    area, cons, tp, scale = _toy()
    g = torch.Generator().manual_seed(0)
    z = torch.randn((2, 8, 4), generator=g, dtype=torch.float64)
    r = EN.decode_rects(z, area, cons, tp, scale)
    soft = (cons[..., 0] == 0) & (cons[..., 1] == 0)
    rel = ((r[..., 2] * r[..., 3] - area) / area)[soft]
    assert float(rel.abs().max()) < 1e-12


def test_decode_makes_a_mib_group_uniform():
    area, cons, tp, scale = _toy()
    area[:, 4:7] = 33.0                      # MIB members share an area
    g = torch.Generator().manual_seed(1)
    z = torch.randn((2, 8, 4), generator=g, dtype=torch.float64)
    r = EN.decode_rects(z, area, cons, tp, scale)
    wh = r[:, 4:7, 2:]
    assert float((wh - wh[:, :1]).abs().max()) < 1e-12
    # ... and off, they differ (so the test is measuring the unification)
    r2 = EN.decode_rects(z, area, cons, tp, scale, mib_unify=False)
    assert float((r2[:, 4:7, 2:] - r2[:, 4:5, 2:]).abs().max()) > 1e-6


def test_decode_freezes_fixed_and_preplaced_geometry():
    area, cons, tp, scale = _toy()
    g = torch.Generator().manual_seed(2)
    z = torch.randn((2, 8, 4), generator=g, dtype=torch.float64) * 3.0
    r = EN.decode_rects(z, area, cons, tp, scale)
    assert torch.equal(r[:, 0, 2:], tp[:, 0, 2:])      # fixed shape
    assert torch.equal(r[:, 1, :], tp[:, 1, :])        # preplaced x,y,w,h


def test_decode_masks_padded_blocks():
    area, cons, tp, scale = _toy()
    area[:, 6:] = -1.0
    g = torch.Generator().manual_seed(3)
    z = torch.randn((2, 8, 4), generator=g, dtype=torch.float64)
    r = EN.decode_rects(z, area, cons, tp, scale)
    assert float(r[:, 6:].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# exact agreement with the evaluator's own helpers
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_evaluator() is None, reason="evaluator not importable")
def test_hpwl_matches_the_evaluator():
    ev = _evaluator()
    g = torch.Generator().manual_seed(4)
    N, P, E = 9, 5, 12
    rects = torch.rand((1, N, 4), generator=g, dtype=torch.float64) * 20 + 1
    b2b = torch.stack([
        torch.randint(0, N, (E,), generator=g), torch.randint(0, N, (E,), generator=g),
        torch.randint(1, 5, (E,), generator=g)], dim=-1).to(torch.float64).view(1, E, 3)
    p2b = torch.stack([
        torch.randint(0, P, (E,), generator=g), torch.randint(0, N, (E,), generator=g),
        torch.randint(1, 5, (E,), generator=g)], dim=-1).to(torch.float64).view(1, E, 3)
    pins = (torch.rand((1, P, 2), generator=g, dtype=torch.float64) * 20)
    got = float(EN.hpwl(rects, b2b, p2b, pins)[0])
    lst = [tuple(map(float, r)) for r in rects[0]]
    want = (ev.calculate_hpwl_b2b(lst, b2b[0])
            + ev.calculate_hpwl_p2b(lst, p2b[0], pins[0]))
    assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.skipif(_evaluator() is None, reason="evaluator not importable")
def test_bbox_area_matches_the_evaluator():
    ev = _evaluator()
    g = torch.Generator().manual_seed(5)
    rects = torch.rand((1, 7, 4), generator=g, dtype=torch.float64) * 10 + 1
    mask = torch.ones((1, 7), dtype=torch.bool)
    got = float(EN.bbox_area(rects, mask)[0])
    want = ev.calculate_bbox_area([tuple(map(float, r)) for r in rects[0]])
    assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.skipif(_evaluator() is None, reason="evaluator not importable")
def test_n_soft_matches_the_evaluator_on_real_cases():
    """`N_soft` is the denominator of the term with 4x the marginal weight of
    everything else -- getting it wrong would silently mis-scale the energy."""
    from icdc_engine import data as D
    ev = _evaluator()
    try:
        cases = D.load_test_cases(ev)[:20]
    except Exception:
        pytest.skip("validation dataset unavailable")
    batch = D.collate(cases, dtype=torch.float64)
    got = EN.n_soft(batch["cons"], batch["area"])
    for k, c in enumerate(cases):
        m = ev.evaluate_solution(
            {"positions": [tuple(map(float, r)) for r in c["golden"]], "runtime": 1.0},
            {"hpwl_baseline": c["hpwl_ref"], "area_baseline": c["area_ref"]},
            c["cons"].to(torch.float32), c["b2b"].to(torch.float32),
            c["p2b"].to(torch.float32), c["pins"].to(torch.float32),
            c["area"].to(torch.float32), c["golden"], median_runtime=1.0)
        want = m.total_soft_violations / max(m.violations_relative, 1e-12) \
            if m.violations_relative > 0 else None
        if want is not None:
            assert float(got[k]) == pytest.approx(want, rel=1e-6)


# ---------------------------------------------------------------------------
# the property that actually matters: ranking
# ---------------------------------------------------------------------------
def _spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1])


@pytest.mark.skipif(_evaluator() is None, reason="evaluator not importable")
def test_energy_ranks_layouts_like_the_official_cost():
    """Pooled Spearman over three genuinely different layout populations.

    Threshold 0.90 is the G0.5 bar from the design memo.  The populations are
    kept separate on purpose -- a correlation measured on one solver's output
    only proves the energy can rank that solver's failure modes.
    """
    from icdc_engine import data as D
    ev = _evaluator()
    try:
        cases = D.load_test_cases(ev)
    except Exception:
        pytest.skip("validation dataset unavailable")
    by_id = {c["test_id"]: c for c in cases}
    roots = [ROOT / "scratchpad" / "icdc",
             Path("/nashome/NVL4/vdalab/yyds-dev/.claude/jobs/8b13fef4/tmp"),
             Path("/nashome/NVL4/vdalab/yyds-dev/.claude/jobs/8b13fef4/tmp/icdc_assets")]
    names = ["gr_prod_layouts.json", "icdc_t35.json", "icdc_self03.json"]
    found = [p for n in names for r in roots if (p := r / n).exists()]
    if not found:
        pytest.skip("no reference layout banks on this machine")
    E_all, C_all = [], []
    for path in found:
        raw = json.load(open(path))
        sel, lay = [], []
        for k, v in raw.items():
            c = by_id.get(int(k))
            if c is None:
                continue
            arr = np.asarray(v, float)
            arr = arr[0] if arr.ndim == 3 else arr
            if arr.shape == (c["n"], 4):
                sel.append(c)
                lay.append(arr)
        if len(sel) < 10:
            continue
        batch = D.collate(sel, dtype=torch.float64)
        R = torch.zeros((len(sel), batch["area"].shape[1], 4), dtype=torch.float64)
        for k, arr in enumerate(lay):
            R[k, :arr.shape[0]] = torch.from_numpy(arr)
        Es = EN.energy(R, batch)["E"].numpy()
        costs = []
        for c, arr in zip(sel, lay):
            m = ev.evaluate_solution(
                {"positions": [tuple(map(float, r)) for r in arr], "runtime": 1.0},
                {"hpwl_baseline": c["hpwl_ref"], "area_baseline": c["area_ref"]},
                c["cons"].to(torch.float32), c["b2b"].to(torch.float32),
                c["p2b"].to(torch.float32), c["pins"].to(torch.float32),
                c["area"].to(torch.float32), c["golden"], median_runtime=1.0)
            costs.append(float(m.cost_no_runtime))
        rho = _spearman(Es, costs)
        assert rho >= 0.90, f"{path.name}: spearman {rho:.4f}"
        E_all += list(Es)
        C_all += costs
    assert len(E_all) >= 10
    assert _spearman(E_all, C_all) >= 0.90
