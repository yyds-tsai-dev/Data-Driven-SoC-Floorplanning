"""Evaluator-faithful differentiable energy for the IC/DC data-free engine.

The official no-runtime cost is

    C = (1 + 0.5*(max(0,hpwl_gap) + max(0,area_gap))) * exp(2.0 * V_rel)
    V_rel = (V_boundary + V_grouping + V_mib) / N_soft

(`FloorSet/iccad2026contest/iccad2026_evaluate.py:306-342, 427-540`).  Taking
logs makes it additive, and the 4x marginal weight of V over the gaps -- the
thing that actually decides our score -- comes out automatically:

    E = log(1 + 0.5*(g_h + g_a)) + 2.0 * V_rel

so `exp(E) == C` exactly when the relaxations below are tight.  Everything is
evaluated on the TFDL output, i.e. on a hard-legal layout, never on the raw
prediction (design memo Sec. 4.1).

Deliberate deviations from the official function, each with a reason:

  * the `max(0, gap)` clips become leaky (slope 0.05).  A sample that already
    beats the golden baseline would otherwise sit in a dead zone with no
    gradient at all, and raw direct_v2 predictions do beat it on hpwl.
  * `V_boundary` / `V_grouping` are *counting* relaxations, not distance
    penalties: the evaluator counts discrete bits, so a distance loss spends
    its gradient dragging an already-violating block from 10% off to 5% off,
    which is worth exactly zero score.  `1 - exp(-d/tau)` saturates instead.
  * `V_mib` is identically zero -- `decode_rects` gives every member of a MIB
    group the same (w, h) by construction, so the whole bucket leaves the
    energy (all 100 validation MIB groups are area-uniform, verified).

`hpwl_ref` / `area_ref` are per-instance scalars taken from `metrics_sol`.
They appear only in the loss and never in a model input, so inference needs
no golden information; see the memo's Sec. 4.2 note on the endogenous-scale
alternative kept for the G2 generalisation control.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _leaky_pos(v: torch.Tensor, slope: float = 0.05) -> torch.Tensor:
    """`max(0, v)` with a small negative slope so beating the baseline still
    produces gradient."""
    return torch.where(v >= 0, v, slope * v)


# ---------------------------------------------------------------------------
# decode: z -> rectangles, with every structural constraint by construction
# ---------------------------------------------------------------------------
def decode_rects(z: torch.Tensor, area: torch.Tensor, cons: torch.Tensor,
                 tp: torch.Tensor, scale: torch.Tensor,
                 mib_unify: bool = True) -> torch.Tensor:
    """z ``[B,N,>=3]`` -> rectangles ``[B,N,4]`` as ``(x, y, w, h)``.

    * exact area: ``w = sqrt(A*e^a)``, ``h = sqrt(A/e^a)`` makes ``w*h == A``
      an identity, so the evaluator's 1% soft-area tolerance can never fire;
    * MIB uniformity: the group's log-aspect is replaced by its mean before
      decoding, so every member of a group gets bit-identical (w, h) and
      ``V_mib == 0``;
    * fixed / preplaced shapes and preplaced origins are overwritten with the
      values the contest handed us.
    """
    B, N = area.shape
    mask = area > 0
    log_aspect = z[..., 2].clamp(-3.0, 3.0)

    if mib_unify and cons.shape[-1] > 2:
        gid = cons[..., 2].long().clamp_min(0)
        gid = torch.where(mask, gid, torch.zeros_like(gid))
        gmax = int(gid.max().item()) if gid.numel() else 0
        if gmax > 0:
            flat = (torch.arange(B, device=z.device).view(B, 1) * (gmax + 1)
                    + gid)
            num = torch.zeros(B * (gmax + 1), dtype=z.dtype, device=z.device)
            den = torch.zeros_like(num)
            active = (gid > 0) & mask
            num.scatter_add_(0, flat.reshape(-1),
                             (log_aspect * active).reshape(-1))
            den.scatter_add_(0, flat.reshape(-1),
                             active.to(z.dtype).reshape(-1))
            mean = (num / den.clamp_min(1.0)).gather(0, flat.reshape(-1))
            log_aspect = torch.where(active, mean.view(B, N), log_aspect)

    a = torch.where(mask, area, torch.ones_like(area)).clamp_min(1e-9)
    aspect = torch.exp(log_aspect)
    w = torch.sqrt(a * aspect)
    h = torch.sqrt(a / aspect)
    x = z[..., 0] * scale.view(B, 1)
    y = z[..., 1] * scale.view(B, 1)

    if cons.shape[-1] > 1:
        fixed = cons[..., 0] != 0
        pre = cons[..., 1] != 0
        wh_known = (fixed | pre) & (tp[..., 2] > 0) & (tp[..., 3] > 0) & mask
        w = torch.where(wh_known, tp[..., 2], w)
        h = torch.where(wh_known, tp[..., 3], h)
        xy_known = pre & (tp[..., 0] >= 0) & (tp[..., 1] >= 0) & mask
        x = torch.where(xy_known, tp[..., 0], x)
        y = torch.where(xy_known, tp[..., 1], y)

    rects = torch.stack([x, y, w, h], dim=-1)
    return torch.where(mask.unsqueeze(-1), rects, torch.zeros_like(rects))


def preplaced_mask(cons: torch.Tensor, tp: torch.Tensor,
                   area: torch.Tensor) -> torch.Tensor:
    if cons.shape[-1] <= 1:
        return torch.zeros_like(area, dtype=torch.bool)
    return ((cons[..., 1] != 0) & (tp[..., 0] >= 0) & (tp[..., 1] >= 0)
            & (area > 0))


# ---------------------------------------------------------------------------
# quality terms (line-for-line against the evaluator)
# ---------------------------------------------------------------------------
def hpwl(rects: torch.Tensor, b2b: torch.Tensor, p2b: torch.Tensor,
         pins: torch.Tensor) -> torch.Tensor:
    """Centroid-Manhattan HPWL, matching `calculate_hpwl_b2b/p2b`."""
    B, N, _ = rects.shape
    cx = rects[..., 0] + 0.5 * rects[..., 2]
    cy = rects[..., 1] + 0.5 * rects[..., 3]
    total = rects.new_zeros(B)

    if b2b is not None and b2b.numel():
        i = b2b[..., 0].long()
        j = b2b[..., 1].long()
        wt = b2b[..., 2].to(rects.dtype)
        ok = (i >= 0) & (j >= 0) & (i < N) & (j < N)
        wt = torch.where(ok, wt, torch.zeros_like(wt))
        ii, jj = i.clamp(0, N - 1), j.clamp(0, N - 1)
        d = ((cx.gather(1, ii) - cx.gather(1, jj)).abs()
             + (cy.gather(1, ii) - cy.gather(1, jj)).abs())
        total = total + (d * wt).sum(dim=1)

    if p2b is not None and p2b.numel() and pins is not None and pins.numel():
        P = pins.shape[1]
        p = p2b[..., 0].long()
        b = p2b[..., 1].long()
        wt = p2b[..., 2].to(rects.dtype)
        pp, bb = p.clamp(0, max(P - 1, 0)), b.clamp(0, N - 1)
        px = pins[..., 0].gather(1, pp)
        py = pins[..., 1].gather(1, pp)
        ok = (p >= 0) & (p < P) & (b >= 0) & (b < N) & (px != -1)
        wt = torch.where(ok, wt, torch.zeros_like(wt))
        d = ((px - cx.gather(1, bb)).abs() + (py - cy.gather(1, bb)).abs())
        total = total + (d * wt).sum(dim=1)
    return total


def bbox(rects: torch.Tensor, mask: torch.Tensor):
    big = torch.full_like(rects[..., 0], 1e30)
    x0 = torch.where(mask, rects[..., 0], big).min(dim=1).values
    y0 = torch.where(mask, rects[..., 1], big).min(dim=1).values
    x1 = torch.where(mask, rects[..., 0] + rects[..., 2], -big).max(dim=1).values
    y1 = torch.where(mask, rects[..., 1] + rects[..., 3], -big).max(dim=1).values
    return x0, y0, x1, y1


def bbox_area(rects: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = bbox(rects, mask)
    return (x1 - x0) * (y1 - y0)


# ---------------------------------------------------------------------------
# soft-violation relaxations
# ---------------------------------------------------------------------------
def _sat(d: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """`d / (d + tau)`: a saturating 0/1 indicator with a *polynomial* tail.

    Chosen over `1 - exp(-d/tau)` deliberately.  The evaluator's boundary
    epsilon is 1e-6 absolute, so a faithful counter needs tau tiny -- and with
    an exponential, tau tiny means the gradient is numerically dead everywhere
    except within a few tau of the wall, which is precisely where the model
    is not.  The rational form saturates just as fast near zero but keeps a
    ~tau/d^2 gradient far away, so one term can be both the count and the pull.
    """
    d = d.clamp_min(0.0)
    return d / (d + tau)


def boundary_distance(rects: torch.Tensor, mask: torch.Tensor,
                      code: torch.Tensor) -> torch.Tensor:
    """Per-block total distance from its tagged bbox edges (0 when seated)."""
    x0, y0, x1, y1 = bbox(rects, mask)
    bx, by, bw, bh = rects.unbind(dim=-1)
    d = rects.new_zeros(bx.shape)
    for bit, val, edge in ((1, bx, x0), (2, bx + bw, x1),
                           (4, by + bh, y1), (8, by, y0)):
        on = ((code & bit) != 0) & mask
        d = d + torch.where(on, (val - edge.unsqueeze(1)).abs(),
                            torch.zeros_like(d))
    return d


def v_boundary_soft(rects: torch.Tensor, mask: torch.Tensor,
                    code: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Saturating count of boundary-tag violations.

    The evaluator scores one bit per tagged block: it violates unless *every*
    requested edge of its bitmask touches the layout bbox.
    """
    d = boundary_distance(rects, mask, code)
    active = ((code != 0) & mask).to(rects.dtype)
    return (_sat(d, tau.view(-1, 1).clamp_min(1e-12)) * active).sum(dim=1)


def v_group_soft(rects: torch.Tensor, mask: torch.Tensor, clust: torch.Tensor,
                 area: torch.Tensor, tau: torch.Tensor,
                 mu: float = 0.0) -> torch.Tensor:
    """Saturating grouping relaxation.

    The true quantity is (connected components - 1) per cluster under
    edge-sharing adjacency, which has no useful gradient.  Two terms stand in:

      * a per-block "am I touching anyone in my group" count, which is the
        discrete part (0 when every member abuts a sibling);
      * a compactness term (group bbox area over member area), which supplies
        the long-range gradient that pulls a scattered group together before
        any contact exists.

    Honest limitation (design memo Sec. 4.3): this is a lower bound on the
    component count, not the count itself -- a group split into two perfectly
    abutted halves scores 0 here and 1 officially.
    """
    B, N, _ = rects.shape
    if clust is None or clust.numel() == 0:
        return rects.new_zeros(B)
    x, y, w, h = rects.unbind(dim=-1)
    x1, y1 = x + w, y + h
    same = (clust.unsqueeze(1) == clust.unsqueeze(2)) & (clust.unsqueeze(1) > 0)
    same = same & mask.unsqueeze(1) & mask.unsqueeze(2)
    same = same & ~torch.eye(N, dtype=torch.bool,
                             device=rects.device).unsqueeze(0)
    if not bool(same.any()):
        return rects.new_zeros(B)
    t = tau.view(B, 1, 1).clamp_min(1e-9)
    sep_x = torch.maximum(x.unsqueeze(1) - x1.unsqueeze(2),
                          x.unsqueeze(2) - x1.unsqueeze(1))
    sep_y = torch.maximum(y.unsqueeze(1) - y1.unsqueeze(2),
                          y.unsqueeze(2) - y1.unsqueeze(1))
    ovl_x = torch.minimum(x1.unsqueeze(1), x1.unsqueeze(2)) \
        - torch.maximum(x.unsqueeze(1), x.unsqueeze(2))
    ovl_y = torch.minimum(y1.unsqueeze(1), y1.unsqueeze(2)) \
        - torch.maximum(y.unsqueeze(1), y.unsqueeze(2))
    # abutment = zero separation on one axis AND positive shared edge on the
    # other; both factors use the same saturating indicator so a 1e-13 gap
    # (which shapely already calls two components) reads as a violation
    contact = ((1.0 - _sat(sep_x, t)) * _sat(ovl_y, t)
               + (1.0 - _sat(sep_y, t)) * _sat(ovl_x, t)).clamp(0.0, 1.0)
    contact = torch.where(same, contact, torch.zeros_like(contact))
    node = same.any(dim=2)
    count = ((1.0 - contact.max(dim=2).values) * node).sum(dim=1)

    # compactness: per group, bbox area / member area - 1
    gid = torch.where(mask, clust.clamp_min(0), torch.zeros_like(clust))
    gmax = int(gid.max().item())
    if gmax <= 0:
        return count
    flat = (torch.arange(B, device=rects.device).view(B, 1) * (gmax + 1) + gid)
    M = B * (gmax + 1)
    active = (gid > 0) & mask
    big = 1e30

    def _red(v, red, init):
        buf = torch.full((M,), init, dtype=rects.dtype, device=rects.device)
        return buf.scatter_reduce(
            0, flat.reshape(-1),
            torch.where(active, v, torch.full_like(v, init)).reshape(-1),
            reduce=red, include_self=True)

    gx0, gy0 = _red(x, "amin", big), _red(y, "amin", big)
    gx1, gy1 = _red(x1, "amax", -big), _red(y1, "amax", -big)
    ga = torch.zeros(M, dtype=rects.dtype, device=rects.device)
    ga = ga.scatter_add(0, flat.reshape(-1),
                        torch.where(active, area, torch.zeros_like(area))
                        .reshape(-1))
    has = ga > 0
    slack = torch.where(
        has, ((gx1 - gx0) * (gy1 - gy0) / ga.clamp_min(1e-9) - 1.0)
        .clamp_min(0.0), torch.zeros_like(ga))
    slack = slack.view(B, gmax + 1)
    slack[:, 0] = 0.0
    return count + mu * slack.sum(dim=1)


def n_soft(cons: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """`N_soft` exactly as the evaluator builds it (:459-471)."""
    mask = area > 0
    B, N = area.shape
    if cons.shape[-1] <= 4:
        return torch.ones(B, dtype=area.dtype, device=area.device)
    nb = (((cons[..., 4] != 0) & mask).sum(dim=1)).to(area.dtype)
    out = nb
    for col in (2, 3):
        gid = torch.where(mask, cons[..., col].long().clamp_min(0),
                          torch.zeros_like(cons[..., col].long()))
        gmax = int(gid.max().item()) if gid.numel() else 0
        if gmax <= 0:
            continue
        oh = torch.zeros(B, gmax + 1, dtype=area.dtype, device=area.device)
        oh.scatter_add_(1, gid, mask.to(area.dtype))
        oh[:, 0] = 0.0
        out = out + (oh - 1.0).clamp_min(0.0).sum(dim=1)
    return out.clamp_min(1.0)


# ---------------------------------------------------------------------------
def energy(rects: torch.Tensor, batch: Dict[str, torch.Tensor],
           mu_group: float = 0.5,
           leak: float = 0.05) -> Dict[str, torch.Tensor]:
    """Log-cost energy of a (legal) layout batch.

    `batch` carries `area, cons, b2b, p2b, pins, hpwl_ref, area_ref, n_soft,
    tau_sharp, tau_soft`.  Two numbers come back:

      * ``E`` -- the faithful log-cost, ``exp(E) ~ cost_no_runtime``.  Its
        violation terms use `tau_sharp`, so they see the near-misses the
        evaluator punishes.
      * ``shaping`` -- the same violations measured with `tau_soft`, plus a
        cluster-compactness term.  Nothing about it is evaluator-faithful; it
        exists because a faithful counter is nearly flat far from the wall,
        and something has to carry the long-range gradient that walks a block
        to the frame in the first place.  The trainer adds it with a small
        weight and it never enters a reported score.
    """
    area = batch["area"]
    mask = area > 0
    hp = hpwl(rects, batch.get("b2b"), batch.get("p2b"), batch.get("pins"))
    ar = bbox_area(rects, mask)
    g_h = _leaky_pos(hp / batch["hpwl_ref"].clamp_min(1e-9) - 1.0, leak)
    g_a = _leaky_pos(ar / batch["area_ref"].clamp_min(1e-9) - 1.0, leak)
    cons = batch["cons"]
    code = cons[..., 4].long() if cons.shape[-1] > 4 else torch.zeros_like(area).long()
    clust = cons[..., 3].long() if cons.shape[-1] > 3 else torch.zeros_like(area).long()
    sharp, soft = batch["tau_sharp"], batch["tau_soft"]
    v_b = v_boundary_soft(rects, mask, code, sharp)
    v_g = v_group_soft(rects, mask, clust, area, sharp, mu=0.0)
    v_rel = (v_b + v_g) / batch["n_soft"]
    quality = torch.log1p(0.5 * (g_h + g_a))
    E = quality + 2.0 * v_rel

    s_b = v_boundary_soft(rects, mask, code, soft)
    s_g = v_group_soft(rects, mask, clust, area, soft, mu=mu_group)
    shaping = (s_b + s_g) / batch["n_soft"]
    return {"E": E, "shaping": shaping, "quality": quality, "v_rel": v_rel,
            "v_bnd": v_b, "v_grp": v_g, "hpwl": hp, "bbox_area": ar,
            "g_h": g_h, "g_a": g_a}
