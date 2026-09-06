"""TFDL invariants: the layer's whole value is that its output is legal *by
construction*, so these tests pin the construction rather than a score.

The load-bearing one is `test_guard_hard_ok_contract`: the engine's output has
to be admissible to rung (-1) (`PARTNER_LEGAL_ADMIT`), whose gate is
`layout_refiner._guard_hard_ok`.  That function -- not a reimplementation of it
-- is what the test calls, so the interface cannot drift.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from icdc_engine import energy as EN          # noqa: E402
from icdc_engine import tfdl as T             # noqa: E402


# ---------------------------------------------------------------------------
def _random_layout(B=3, N=24, seed=0, overlap=True, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    area = torch.exp(torch.randn((B, N), generator=g, dtype=dtype) * 0.5 + 3.0)
    aspect = torch.exp(torch.randn((B, N), generator=g, dtype=dtype) * 0.4)
    w = torch.sqrt(area * aspect)
    h = torch.sqrt(area / aspect)
    span = torch.sqrt(area.sum(dim=1)).view(B, 1) * (0.6 if overlap else 1.6)
    x = torch.rand((B, N), generator=g, dtype=dtype) * span
    y = torch.rand((B, N), generator=g, dtype=dtype) * span
    return torch.stack([x, y, w, h], dim=-1), area


def _overlaps(P, tol=1e-7):
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = x0 + P[:, 2], y0 + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    bad = (ox > tol) & (oy > tol)
    np.fill_diagonal(bad, False)
    return int(bad.sum() // 2)


# ---------------------------------------------------------------------------
# the closure is the one piece of clever code; pin it to the obvious one
# ---------------------------------------------------------------------------
def test_closure_matches_floyd_warshall():
    g = torch.Generator().manual_seed(7)
    B, N = 2, 17
    adj = (torch.rand((B, N, N), generator=g) < 0.25)
    adj &= torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    size = torch.rand((B, N), generator=g, dtype=torch.float64) + 0.1
    fast = T._closure(adj, size, chunk=5)
    slow = T._closure_sequential(adj, size)
    reachable = fast > T.NEG / 2
    assert torch.equal(reachable, slow > T.NEG / 2)
    assert torch.allclose(fast[reachable], slow[reachable], atol=1e-12)


# ---------------------------------------------------------------------------
# the unconditional guarantees
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_output_is_overlap_free_from_a_blob(seed):
    """The interesting input is the one the model actually produces: a pile of
    mutually overlapping blocks."""
    rects, area = _random_layout(B=3, N=24, seed=seed, overlap=True)
    mask = area > 0
    assert _overlaps(rects[0].numpy()) > 0, "test input should overlap"
    legal, drift = T.tfdl(rects, mask)
    for b in range(rects.shape[0]):
        assert _overlaps(legal[b].numpy()) == 0
    assert bool((drift == 0).all())          # no pins -> no drift possible


def test_sizes_are_never_touched():
    rects, area = _random_layout(seed=5)
    legal, _ = T.tfdl(rects, area > 0)
    assert torch.equal(legal[..., 2:], rects[..., 2:])


def test_identity_on_an_already_legal_layout():
    """A legal input must come out unchanged, bit for bit.

    Not a nicety: the fine-tune's whole goal is to make the model emit legal
    layouts, and a projection that taxed its own fixed points would cap the
    achievable score.  `exact=True` is what makes it bitwise -- the closure
    alone lands 1-2 ulp off, and the official grouping check (a shapely union)
    calls a 2e-13 gap two components.
    """
    B, N = 2, 12
    rects = torch.zeros((B, N, 4), dtype=torch.float64)
    for b in range(B):
        for i in range(N):
            r, c = divmod(i, 4)
            rects[b, i] = torch.tensor([c * 10.0, r * 7.0, 10.0, 7.0],
                                       dtype=torch.float64)
    mask = torch.ones((B, N), dtype=torch.bool)
    legal, drift = T.tfdl(rects, mask, exact=True)
    assert torch.equal(legal, rects)
    assert bool((drift == 0).all())


def test_pinned_blocks_stay_exact_when_drift_is_zero():
    rects, area = _random_layout(B=4, N=20, seed=11, overlap=True)
    mask = area > 0
    pinned = torch.zeros_like(mask)
    pinned[:, :2] = True
    pin_xy = rects[..., :2].clone()
    legal, drift = T.tfdl(rects, mask, pinned, pin_xy=pin_xy)
    ok = drift.amax(dim=(1, 2)) <= 0
    assert bool(ok.any()), "the fixture should leave at least one sample clean"
    got = legal[..., :2][pinned & ok.view(-1, 1)]
    want = pin_xy[pinned & ok.view(-1, 1)]
    assert torch.equal(got, want)
    # and drift is exactly the residual, never negative
    assert bool((drift >= 0).all())
    moved = legal[..., :2] - pin_xy
    assert torch.allclose(torch.where(pinned.unsqueeze(-1), moved,
                                      torch.zeros_like(moved)), drift)


def test_exact_pass_snaps_sub_tolerance_pin_roundoff_bit_exact(monkeypatch):
    rects = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]]],
        dtype=torch.float64,
    )
    mask = torch.ones((1, 2), dtype=torch.bool)
    pinned = torch.tensor([[True, False]])
    pin_xy = rects[..., :2].clone()
    original = T._exact_pass

    def one_ulp(lo, size, adj):
        out = original(lo, size, adj)
        out[:, 0] = torch.nextafter(out[:, 0], torch.full_like(out[:, 0], float("inf")))
        return out

    monkeypatch.setattr(T, "_exact_pass", one_ulp)
    legal, drift = T.tfdl(
        rects, mask, pinned, pin_xy=pin_xy, exact=True,
    )

    assert torch.equal(legal[0, 0, :2], pin_xy[0, 0])
    assert torch.equal(drift, torch.zeros_like(drift))


def test_padded_blocks_are_ignored():
    rects, area = _random_layout(B=2, N=16, seed=3)
    mask = area > 0
    mask[:, 12:] = False
    area = torch.where(mask, area, torch.full_like(area, -1.0))
    legal, _ = T.tfdl(rects, mask)
    assert torch.equal(legal[:, 12:], rects[:, 12:])
    for b in range(2):
        assert _overlaps(legal[b, :12].numpy()) == 0


def test_exact_pass_produces_bitwise_abutment():
    """After the exact pass, a block that was pushed sits at literally
    ``c_pred + w_pred`` -- the float shapely needs to see a shared edge."""
    rects, area = _random_layout(B=1, N=14, seed=9, overlap=True)
    legal, _ = T.tfdl(rects, area > 0, exact=True)
    P = legal[0].numpy()
    x1 = P[:, 0] + P[:, 2]
    touching = sum(1 for i in range(len(P)) for j in range(len(P))
                   if i != j and P[j, 0] == x1[i])
    assert touching > 0


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------
def test_gradient_reaches_coordinates_and_sizes():
    rects, area = _random_layout(B=2, N=16, seed=2, overlap=True)
    rects = rects.clone().requires_grad_(True)
    legal, _ = T.tfdl(rects, area > 0)
    legal.sum().backward()
    g = rects.grad
    assert torch.isfinite(g).all()
    assert float(g[..., :2].abs().sum()) > 0, "no gradient to coordinates"
    assert float(g[..., 2:].abs().sum()) > 0, "no gradient to sizes"


def test_gradient_matches_finite_difference_on_a_free_block():
    """A block nothing pushes moves 1:1 with its prediction."""
    B, N = 1, 6
    rects = torch.tensor([[[0., 0., 4., 4.], [10., 0., 4., 4.],
                           [20., 0., 4., 4.], [0., 10., 4., 4.],
                           [10., 10., 4., 4.], [20., 10., 4., 4.]]],
                         dtype=torch.float64, requires_grad=True)
    mask = torch.ones((B, N), dtype=torch.bool)
    legal, _ = T.tfdl(rects, mask)
    legal[0, 4, 0].backward()
    assert pytest.approx(float(rects.grad[0, 4, 0]), abs=1e-9) == 1.0


# ---------------------------------------------------------------------------
# the rung-(-1) interface contract
# ---------------------------------------------------------------------------
def _opt_stub(area, cons, tp):
    """The exact attribute surface `_guard_hard_ok` reads."""
    n = len(area)
    fixed = cons[:, 0] != 0
    pre = cons[:, 1] != 0
    kind = np.where(pre, 2, np.where(fixed, 1, 0))
    return SimpleNamespace(n=n, kind=kind, areas=area,
                           rw=np.where(kind != 0, tp[:, 2], 0.0),
                           rh=np.where(kind != 0, tp[:, 3], 0.0),
                           lx=np.where(kind == 2, tp[:, 0], 0.0),
                           ly=np.where(kind == 2, tp[:, 1], 0.0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_guard_hard_ok_contract(seed):
    """decode -> TFDL output is admissible to rung (-1), or its pin drifted.

    Those are the only two outcomes allowed: the layer may fail to honour a
    preplaced origin (and must say so through `drift`), but it may never hand
    back something that violates overlap, soft-block area, or a fixed shape.
    """
    import layout_refiner as rf
    from synth_instances import build_instance

    inst = build_instance(n=48, seed=seed)
    n = 48
    area = inst.area_targets[:n].to(torch.float64).reshape(n)
    cons = inst.constraints[:n].to(torch.float64)
    tp = inst.target_positions[:n].to(torch.float64)
    batch = {"area": area.view(1, n), "cons": cons.view(1, n, -1),
             "tp": tp.view(1, n, 4),
             "scale": torch.sqrt(area.sum()).view(1)}
    g = torch.Generator().manual_seed(seed)
    z = torch.randn((1, n, 4), generator=g, dtype=torch.float64) * 0.4
    rects = EN.decode_rects(z, batch["area"], batch["cons"], batch["tp"],
                            batch["scale"])
    mask = batch["area"] > 0
    pin = EN.preplaced_mask(batch["cons"], batch["tp"], batch["area"])
    legal, drift = T.tfdl(rects, mask, pin, pin_xy=batch["tp"][..., :2],
                          exact=True)

    P = np.ascontiguousarray(legal[0].numpy(), dtype=np.float64)
    opt = _opt_stub(area.numpy(), cons.numpy(), tp.numpy())
    verdict = rf._guard_hard_ok(opt, P)
    d = float(drift.max())
    # the guard's own preplaced tolerance is 1e-5; both directions of the
    # implication are the contract the bank dump relies on
    if d <= T.PIN_TOL:
        assert verdict, f"zero drift ({d:.3e}) but the guard refused the layout"
    if not verdict:
        assert d > T.PIN_TOL, f"guard refused a layout with drift {d:.3e}"
    # regardless of the pin, these must hold
    assert _overlaps(P) == 0
    soft = (cons[:, 0] == 0) & (cons[:, 1] == 0)
    rel = ((P[:, 2] * P[:, 3] - area.numpy()) / area.numpy())[soft.numpy()]
    assert np.abs(rel).max() < 1e-9
