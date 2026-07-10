"""Min-cut partition-driven layout probe (last untested big global mechanism).

Hypothesis under test: recursive min-cut BISECTION builds a global hierarchical
(slicing-tree) structure that can drive the tail (n>=100) HPWL below the
production column-backbone -- the mechanism SA (a local move engine) and QP
(a spring blob that does not realize) both failed to supply.

Pipeline:
  connectivity graph (b2b + cluster cohesion + MIB contraction)
    -> recursive guillotine bisection (cut perpendicular to the longer side)
         * cut = connectivity-diffusion embedding, area-balanced split, FM polish
         * boundary tags -> HARD side-locks (L/R on vertical cuts, B/T on horizontal)
         * preplaced -> side-locked by golden coord, pinned EXACT in the hint
         * pins (p2b) -> soft side votes via connected pin coordinate
    -> slicing rooms (area = block area / density) tiling the chip
    -> per-block exact-area hint at the room aspect (MIB members share one shape)
    -> gen_decoder_probe.decode(realize=...) [order-extract + ASAP + repair + refine]
    -> official evaluator score_case (no-runtime).

The min-cut STRUCTURE is carried by the pairwise room ORDER (which block is
left/below which); the decoder re-derives tight coordinates from that order, so
imperfect leaf tiling is harmless -- the slicing tree is what is under test.

Paired vs the CURRENT production layout (colmax_s4.json test_results[*].positions),
re-scored through the identical score_case instrument. Also runs a GOLDEN-CUT
control (--cut golden): the same rooms/realize pipeline driven by golden-centroid
median bisection instead of FM, isolating CUT QUALITY from REALIZATION LOSS.

Self-contained: reuses gen_decoder_probe.{decode,score_case,helpers} and
column_slicing._parse_constraints/_target only. No src/ or FloorSet/ edits.

Usage (from repo root; run via bash even though the shell is tcsh):
  cd FloorSet/iccad2026contest
  source ../../.env
  PYTHONPATH="$PWD:$PWD/..:<repo>/src" ~/.local/bin/uv run python \
      <repo>/scripts/probes/partition_probe.py \
      --production-json <scratch>/colmax_s4.json \
      --idxs 95,96,97,98,99 --cut both --realize anchored --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- make src/ and this probe dir importable (mirrors analytic_hints.py) ---
_THIS = Path(__file__).resolve()
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))
_REPO = None
for _p in _THIS.parents:
    if (_p / "src" / "floorset_arch").is_dir():
        _REPO = _p
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break
if _REPO is None:
    _REPO = Path("/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning")
    sys.path.insert(0, str(_REPO / "src"))

from iccad2026_evaluate import compute_total_score  # noqa: E402

from gen_decoder_probe import (  # noqa: E402
    _n_of,
    _golden_rects,
    _opt_target_positions,
    _edge_lists,
    _band,
    decode,
    score_case,
)
from floorset_arch.legalizer.column_slicing import (  # noqa: E402
    _parse_constraints,
    _target,
)

Rect = Tuple[float, float, float, float]

BOUND_LEFT = 1
BOUND_RIGHT = 2
BOUND_TOP = 4
BOUND_BOTTOM = 8
LOG_ASPECT_CLAMP = 3.0


# =============================================================================
# Super-node graph (MIB contraction + cluster cohesion)
# =============================================================================
class Graph:
    """Contracted connectivity graph over super-nodes.

    A super-node is either a single block, or a contracted MIB group (members
    forced to share a room). Cluster (grouping) cohesion is added as extra edge
    weight between same-cluster super-nodes so the min-cut prefers to keep a
    group together (soft, HPWL may still split it), which lowers grouping v_rel.
    """

    def __init__(self):
        self.super_of: Dict[int, int] = {}     # block -> super id
        self.members: Dict[int, List[int]] = {}  # super id -> block list
        self.area: Dict[int, float] = {}
        self.adj: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.bcode: Dict[int, int] = {}         # OR of member boundary codes
        self.preplaced: Dict[int, bool] = {}
        self.pin_anchor: Dict[int, Tuple[float, float, float]] = {}  # sx,sy,wsum


def build_graph(n, b2b_e, p2b_e, pin_l, area_targets, mib, cluster, preplaced,
                boundary, golden, cfg) -> Graph:
    g = Graph()
    # --- super-node assignment: contract MIB groups whose members are all free
    #     (a preplaced member is pinned exact, so leave those members separate) ---
    mib_members: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        if mib[i] > 0:
            mib_members[mib[i]].append(i)
    contract: Dict[int, int] = {}  # block -> representative block
    for grp, mem in mib_members.items():
        if any(preplaced[i] for i in mem):
            continue  # do not contract; decoder handles MIB shape per-block
        rep = mem[0]
        for i in mem:
            contract[i] = rep
    sid = 0
    for i in range(n):
        rep = contract.get(i, i)
        if rep == i:
            g.super_of[i] = sid
            g.members[sid] = [i]
            sid += 1
        else:
            pass
    # attach contracted members to their rep's super
    for i in range(n):
        if i in contract and contract[i] != i:
            s = g.super_of[contract[i]]
            g.super_of[i] = s
            g.members[s].append(i)
    # --- per-super aggregates ---
    for s, mem in g.members.items():
        g.area[s] = sum(float(area_targets[i]) for i in mem)
        code = 0
        pre = False
        for i in mem:
            code |= boundary[i]
            pre = pre or preplaced[i]
        g.bcode[s] = code
        g.preplaced[s] = pre
    # --- b2b edges lifted to supers ---
    for i, j, w in b2b_e:
        if not (0 <= i < n and 0 <= j < n) or w <= 0:
            continue
        si, sj = g.super_of[i], g.super_of[j]
        if si == sj:
            continue
        g.adj[si][sj] += w
        g.adj[sj][si] += w
    # --- cluster cohesion: extra all-pairs weight inside each grouping set ---
    if cfg["cluster_cohesion"] > 0:
        cl_members: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            if cluster[i] > 0:
                cl_members[cluster[i]].append(i)
        for grp, mem in cl_members.items():
            supers = sorted({g.super_of[i] for i in mem})
            if len(supers) < 2:
                continue
            # normalize so a group contributes a fixed cohesion budget regardless
            # of size (avoid huge cliques dominating the cut).
            wt = cfg["cluster_cohesion"] / (len(supers) - 1)
            for a in range(len(supers)):
                for b in range(a + 1, len(supers)):
                    g.adj[supers[a]][supers[b]] += wt
                    g.adj[supers[b]][supers[a]] += wt
    # --- pin anchors (p2b): summed pin coordinate + weight per super ---
    acc: Dict[int, Tuple[float, float, float]] = defaultdict(
        lambda: (0.0, 0.0, 0.0))
    npins = len(pin_l)
    for p, b, w in p2b_e:
        if not (0 <= b < n and 0 <= p < npins) or w <= 0:
            continue
        s = g.super_of[b]
        sx, sy, sw = acc[s]
        acc[s] = (sx + w * pin_l[p][0], sy + w * pin_l[p][1], sw + w)
    for s in g.members:
        sx, sy, sw = acc.get(s, (0.0, 0.0, 0.0))
        if sw > 0:
            g.pin_anchor[s] = (sx / sw, sy / sw, sw)
    return g


# =============================================================================
# Recursive guillotine bisection
# =============================================================================
def _diffusion_embed(supers, adj, anchors, iters=40) -> Dict[int, float]:
    """Harmonic connectivity embedding on [0,1]: locked anchors held fixed,
    free supers relax to the weighted average of neighbors + soft anchors.
    Returns a coordinate per super (relative ordering is what matters)."""
    sset = set(supers)
    x = {s: 0.5 for s in supers}
    fixed = {}
    for s in supers:
        a = anchors.get(s)
        if a is not None and a[1] >= 1e9:  # hard lock
            fixed[s] = a[0]
            x[s] = a[0]
    for _ in range(iters):
        for s in supers:
            if s in fixed:
                continue
            num = 0.0
            den = 0.0
            for t, w in adj[s].items():
                if t in sset:
                    num += w * x[t]
                    den += w
            a = anchors.get(s)
            if a is not None and a[1] < 1e9:  # soft anchor (target, weight)
                num += a[1] * a[0]
                den += a[1]
            if den > 0:
                x[s] = num / den
    return x


def _fiedler_embed(supers, adj) -> Dict[int, float]:
    """Fallback when the sub-problem has no anchors: 2nd Laplacian eigenvector."""
    m = len(supers)
    idx = {s: k for k, s in enumerate(supers)}
    L = np.zeros((m, m))
    for s in supers:
        for t, w in adj[s].items():
            if t in idx and t != s:
                L[idx[s], idx[s]] += w
                L[idx[s], idx[t]] -= w
    try:
        vals, vecs = np.linalg.eigh(L)
        v = vecs[:, 1] if m > 1 else np.zeros(m)
    except np.linalg.LinAlgError:
        v = np.array([idx[s] for s in supers], dtype=float)
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo if hi > lo else 1.0
    return {s: (float(v[idx[s]]) - lo) / span for s in supers}


def _fm_refine(A: set, B: set, supers, adj, area, lockA, lockB,
               total, tol, passes) -> None:
    """Light Fiedler-Mattheyses cut polish: move free supers across the cut to
    reduce cut weight while keeping area balance within [0.5-tol, 0.5+tol].
    Mutates A/B in place."""
    def side_area(S):
        return sum(area[s] for s in S)
    for _ in range(passes):
        moved = False
        aA = side_area(A)
        for s in list(supers):
            if s in lockA or s in lockB:
                continue
            inA = s in A
            # gain = reduction in cut weight if moved to the other side
            to_same = 0.0
            to_other = 0.0
            for t, w in adj[s].items():
                if t in A:
                    if inA:
                        to_same += w
                    else:
                        to_other += w
                elif t in B:
                    if inA:
                        to_other += w
                    else:
                        to_same += w
            gain = to_other - to_same  # >0 means moving cuts fewer nets
            if gain <= 1e-12:
                continue
            new_aA = aA - area[s] if inA else aA + area[s]
            frac = new_aA / total if total > 0 else 0.5
            if abs(frac - 0.5) > tol:
                continue
            if inA:
                A.discard(s); B.add(s)
            else:
                B.discard(s); A.add(s)
            aA = new_aA
            moved = True
        if not moved:
            break


def _bisect(supers, g, cut_axis, cfg, golden, use_golden) -> Tuple[set, set]:
    """Split `supers` into (A=lower coord, B=higher coord) on cut_axis (0=x,1=y)
    respecting boundary/preplaced hard locks and area balance."""
    total = sum(g.area[s] for s in supers)
    lockA, lockB = set(), set()
    anchors: Dict[int, Tuple[float, float]] = {}
    LOCK = 1e9
    for s in supers:
        code = g.bcode[s]
        lo_tag = (code & BOUND_LEFT) if cut_axis == 0 else (code & BOUND_BOTTOM)
        hi_tag = (code & BOUND_RIGHT) if cut_axis == 0 else (code & BOUND_TOP)
        if lo_tag and not hi_tag:
            lockA.add(s); anchors[s] = (0.0, LOCK)
        elif hi_tag and not lo_tag:
            lockB.add(s); anchors[s] = (1.0, LOCK)
    # golden-cut control OR pure-FM embedding
    if use_golden:
        coord = {}
        for s in supers:
            mem = g.members[s]
            c = sum((golden[i][cut_axis] + golden[i][cut_axis + 2] / 2.0)
                    for i in mem) / len(mem)
            coord[s] = c
    else:
        # soft anchors from preplaced golden coord + pin votes (normalized)
        for s in supers:
            if s in anchors:
                continue
            tgt = None
            wt = 0.0
            if g.preplaced[s]:
                mem = g.members[s]
                tgt = sum(golden[i][cut_axis] for i in mem) / len(mem)
                wt = cfg["preplaced_w"]
            elif s in g.pin_anchor:
                pa = g.pin_anchor[s]
                tgt = pa[cut_axis]
                wt = min(cfg["pin_w"] * pa[2], cfg["pin_w_cap"])
            if tgt is not None and wt > 0:
                anchors[s] = (tgt, wt)
        # normalize soft-anchor targets to [0,1] over the node set
        soft = [(s, a) for s, a in anchors.items() if a[1] < LOCK]
        if soft:
            vals = [a[0] for _, a in soft]
            lo, hi = min(vals), max(vals)
            span = hi - lo if hi > lo else 1.0
            for s, a in soft:
                anchors[s] = ((a[0] - lo) / span, a[1])
        has_anchor = len(anchors) > 0
        if has_anchor:
            coord = _diffusion_embed(supers, g.adj, anchors,
                                     iters=cfg["embed_iters"])
        else:
            coord = _fiedler_embed(supers, g.adj)
    # area-balanced split at the coord ordering; keep locks on their side
    order = sorted(supers, key=lambda s: (coord[s], s))
    A, B = set(), set()
    # place hard locks first
    for s in supers:
        if s in lockA:
            A.add(s)
        elif s in lockB:
            B.add(s)
    aA = sum(g.area[s] for s in A)
    target = total / 2.0
    free = [s for s in order if s not in lockA and s not in lockB]
    # greedily assign free (in coord order) to A until half, rest to B
    for s in free:
        if aA + g.area[s] / 2.0 <= target or not A:
            A.add(s); aA += g.area[s]
        else:
            B.add(s)
    # ensure non-empty split
    if not A or not B:
        A, B = set(), set()
        for k, s in enumerate(order):
            (A if k < len(order) // 2 else B).add(s)
    if not use_golden and cfg["fm_passes"] > 0:
        _fm_refine(A, B, supers, g.adj, g.area, lockA, lockB, total,
                   cfg["balance_tol"], cfg["fm_passes"])
    return A, B


def partition_layout(g: Graph, n, cfg, golden, use_golden) -> Dict[int, Rect]:
    """Recursively bisect into rooms; return room rect per super-node."""
    total_area = sum(g.area.values())
    side = math.sqrt(total_area / cfg["density"])
    rooms: Dict[int, Rect] = {}

    def rec(supers, region):
        x0, y0, W, H = region
        if len(supers) == 1:
            rooms[supers[0]] = (x0, y0, W, H)
            return
        axis = 0 if W >= H else 1  # cut perpendicular to the longer side
        A, B = _bisect(supers, g, axis, cfg, golden, use_golden)
        A = list(A); B = list(B)
        if not A or not B:
            mid = len(supers) // 2
            A, B = supers[:mid], supers[mid:]
        aA = sum(g.area[s] for s in A)
        aB = sum(g.area[s] for s in B)
        fA = aA / (aA + aB) if (aA + aB) > 0 else 0.5
        if axis == 0:
            WA = W * fA
            rec(A, (x0, y0, WA, H))
            rec(B, (x0 + WA, y0, W - WA, H))
        else:
            HA = H * fA
            rec(A, (x0, y0, W, HA))
            rec(B, (x0, y0 + HA, W, H - HA))

    rec(list(g.members.keys()), (0.0, 0.0, side, side))
    return rooms


# =============================================================================
# Leaf placement -> per-block hint rects (exact-area, room aspect)
# =============================================================================
def _shape_at_aspect(area, la) -> Tuple[float, float]:
    la = max(-LOG_ASPECT_CLAMP, min(LOG_ASPECT_CLAMP, la))
    e = math.exp(la)
    return math.sqrt(area * e), math.sqrt(area / e)


def rooms_to_hints(g, rooms, n, area_targets, preplaced, boundary, mib,
                   golden, tpos) -> List[Rect]:
    hints: List[Optional[Rect]] = [None] * n
    for s, (rx, ry, rW, rH) in rooms.items():
        mem = g.members[s]
        room_la = math.log(rW / rH) if (rW > 1e-9 and rH > 1e-9) else 0.0
        if len(mem) == 1:
            i = mem[0]
            if preplaced[i]:
                tx, ty, tw, th = _target(tpos, i)
                hints[i] = (float(tx), float(ty), float(tw), float(th))
                continue
            a = float(area_targets[i])
            w, h = _shape_at_aspect(a, room_la)
            # place against the room edge facing the chip boundary
            code = boundary[i]
            px = rx + (rW - w if (code & BOUND_RIGHT) else 0.0)
            py = ry + (rH - h if (code & BOUND_TOP) else 0.0)
            hints[i] = (px, py, w, h)
        else:
            # contracted MIB group: identical shape, grid-packed in the room
            k = len(mem)
            a = sum(float(area_targets[i]) for i in mem) / k
            cols = max(1, int(round(math.sqrt(k))))
            rowsN = int(math.ceil(k / cols))
            cw = rW / cols
            ch = rH / rowsN
            la = math.log(cw / ch) if (cw > 1e-9 and ch > 1e-9) else 0.0
            w, h = _shape_at_aspect(a, la)
            for t, i in enumerate(mem):
                cc = t % cols
                rr = t // cols
                hints[i] = (rx + cc * cw, ry + rr * ch, w, h)
    for i in range(n):
        if hints[i] is None:
            hints[i] = (0.0, 0.0, math.sqrt(max(float(area_targets[i]), 1e-9)),
                        math.sqrt(max(float(area_targets[i]), 1e-9)))
    return [hints[i] for i in range(n)]


def partition_hints(sample, n, cfg, use_golden) -> List[Rect]:
    at = sample["input"][0][:n]
    b2b, p2b, pins = sample["input"][1], sample["input"][2], sample["input"][3]
    cons = sample["input"][4][:n]
    b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
    golden = _golden_rects(sample, n)
    tpos = _opt_target_positions(sample, n, golden)
    fixed, preplaced, mib, cluster, boundary = _parse_constraints(cons, n)
    area_targets = [float(at[i]) for i in range(n)]
    g = build_graph(n, b2b_e, p2b_e, pin_l, area_targets, mib, cluster,
                    preplaced, boundary, golden, cfg)
    rooms = partition_layout(g, n, cfg, golden, use_golden)
    return rooms_to_hints(g, rooms, n, area_targets, preplaced, boundary, mib,
                          golden, tpos)


# =============================================================================
# Driver (paired vs current production positions, D1-style)
# =============================================================================
def _load_prod(path) -> Dict[int, List[Rect]]:
    with open(path) as f:
        d = json.load(f)
    out: Dict[int, List[Rect]] = {}
    if isinstance(d, dict) and "test_results" in d:
        for r in d["test_results"]:
            pos = r.get("positions")
            if pos:
                out[int(r["test_id"])] = [tuple(p) for p in pos]
    else:  # plain {idx: layout} cache
        for k, v in d.items():
            out[int(k)] = [tuple(p) for p in v]
    return out


def run(args):
    os.environ.setdefault("FLOORSET_COLUMN_BACKBONE", "1")
    os.environ.setdefault("FLOORSET_SLACK_REFINE_VSNAP", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    prod = _load_prod(args.production_json)
    from lite_dataset_test import FloorplanDatasetLiteTest
    ds = FloorplanDatasetLiteTest(str(args.data_path))

    if args.idxs:
        idxs = [int(x) for x in args.idxs.split(",")]
    else:
        idxs = sorted(prod.keys())
    cfg = dict(
        density=args.density, cluster_cohesion=args.cluster_cohesion,
        preplaced_w=args.preplaced_w, pin_w=args.pin_w, pin_w_cap=args.pin_w_cap,
        embed_iters=args.embed_iters, fm_passes=args.fm_passes,
        balance_tol=args.balance_tol,
    )
    modes = ["fm", "golden"] if args.cut == "both" else [args.cut]

    t0 = time.time()
    rows = []
    for idx in idxs:
        sample = ds[idx]
        n = _n_of(sample)
        b2b, p2b, pins = sample["input"][1], sample["input"][2], sample["input"][3]
        b2b_e, p2b_e, pin_l = _edge_lists(b2b, p2b, pins)
        base = score_case(sample, prod[idx], n) if idx in prod else None
        row = {"idx": idx, "n": n, "band": _band(n)}
        if base is not None:
            row["base_cost"] = base["cost"]
            row["base_hpwl_gap"] = base["hpwl_gap"]
            row["base_area_gap"] = base["area_gap"]
            row["base_v_rel"] = base["v_rel"]
        for mode in modes:
            th = time.time()
            hints = partition_hints(sample, n, cfg, use_golden=(mode == "golden"))
            pos, tr = decode(hints, sample, n, b2b_e, p2b_e, pin_l,
                             do_polish=False, realize=args.realize,
                             do_repair=(args.realize == "compact"),
                             do_refine=(args.realize == "compact" and args.refine),
                             anchor_hint_weight=args.anchor_hint_weight,
                             anchor_wall_weight=args.anchor_wall_weight)
            dms = 1000.0 * (time.time() - th)
            if pos is None:
                sc = {"cost": 10.0, "feasible": False, "hpwl_gap": 0.0,
                      "area_gap": 0.0, "v_rel": 1.0, "boundary_v": -1,
                      "grouping_v": -1, "mib_v": -1, "overlap": -1}
            else:
                sc = score_case(sample, pos, n)
            row[f"{mode}_cost"] = sc["cost"]
            row[f"{mode}_feas"] = sc["feasible"]
            row[f"{mode}_hpwl_gap"] = sc["hpwl_gap"]
            row[f"{mode}_area_gap"] = sc["area_gap"]
            row[f"{mode}_v_rel"] = sc["v_rel"]
            row[f"{mode}_bnd"] = sc["boundary_v"]
            row[f"{mode}_grp"] = sc["grouping_v"]
            row[f"{mode}_mib"] = sc["mib_v"]
            row[f"{mode}_fallback"] = tr.get("legal_fallback")
            row[f"{mode}_ms"] = round(dms, 0)
        rows.append(row)
        if args.verbose:
            b = f" base={row.get('base_cost', float('nan')):.4f}(hg{row.get('base_hpwl_gap',0):+.3f})"
            parts = []
            for mode in modes:
                parts.append(f"{mode}={row[f'{mode}_cost']:.4f}"
                             f"(hg{row[f'{mode}_hpwl_gap']:+.3f} "
                             f"ag{row[f'{mode}_area_gap']:+.3f} "
                             f"vr{row[f'{mode}_v_rel']:.3f} "
                             f"bnd{row[f'{mode}_bnd']} grp{row[f'{mode}_grp']} "
                             f"fb{row[f'{mode}_fallback']})")
            print(f"  idx{idx:3d} n{n}{b}  " + "  ".join(parts), flush=True)
    _report(rows, modes, time.time() - t0, args, cfg)


def _report(rows, modes, elapsed, args, cfg):
    print("\n" + "=" * 92)
    print(f"MIN-CUT PARTITION probe   cut={args.cut}  realize={args.realize}  "
          f"cases={len(rows)}  wall={elapsed:.1f}s")
    print(f"  cfg: {cfg}")
    print("=" * 92)
    have_base = all("base_cost" in r for r in rows)
    ns = [r["n"] for r in rows]
    if have_base:
        base_total = compute_total_score([r["base_cost"] for r in rows], ns)
        print(f"  production(base) weighted total = {base_total:.4f}")
    for mode in modes:
        cs = [r[f"{mode}_cost"] for r in rows]
        feas = sum(1 for r in rows if r[f"{mode}_feas"])
        tot = compute_total_score(cs, ns)
        line = f"  [{mode:6s}] weighted={tot:.4f}  feas={feas}/{len(rows)}"
        if have_base:
            wins = sum(1 for r in rows
                       if r[f"{mode}_cost"] < r["base_cost"] - 1e-9)
            port = compute_total_score(
                [min(r[f"{mode}_cost"], r["base_cost"]) for r in rows], ns)
            line += (f"  wins={wins}/{len(rows)}  portfolio(min)={port:.4f}"
                     f"  deploy_delta={port - base_total:+.4f}")
        print(line)
    print("-" * 92)
    hdr = f"  {'idx':>3} {'n':>3} " + ("{:>9} ".format("prod") if have_base else "")
    for mode in modes:
        hdr += f"{mode:>9} "
    print(hdr + "   (cost; hg=hpwl_gap ag=area_gap vr=v_rel)")
    for r in sorted(rows, key=lambda r: r["idx"]):
        line = f"  {r['idx']:>3} {r['n']:>3} "
        if have_base:
            line += f"{r['base_cost']:>9.4f} "
        for mode in modes:
            flag = ""
            if have_base and r[f"{mode}_cost"] < r["base_cost"] - 1e-9:
                flag = "*"
            line += f"{r[f'{mode}_cost']:>8.4f}{flag} "
        print(line)
        for mode in modes:
            print(f"        {mode:6s}: hg{r[f'{mode}_hpwl_gap']:+.3f} "
                  f"ag{r[f'{mode}_area_gap']:+.3f} vr{r[f'{mode}_v_rel']:.3f} "
                  f"bnd{r[f'{mode}_bnd']} grp{r[f'{mode}_grp']} "
                  f"mib{r[f'{mode}_mib']} fb{r[f'{mode}_fallback']} "
                  f"{r[f'{mode}_ms']:.0f}ms"
                  + (f"   | prod hg{r['base_hpwl_gap']:+.3f} "
                     f"ag{r['base_area_gap']:+.3f} vr{r['base_v_rel']:.3f}"
                     if have_base and mode == modes[0] else ""))
    print("=" * 92)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(cfg=cfg, rows=rows), f, indent=2, default=str)
        print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-json", required=True,
                    help="colmax_s4.json (eval result w/ positions) or a plain "
                         "{idx: layout} cache")
    ap.add_argument("--data-path",
                    default=os.environ.get("FLOORSET_DATA_PATH", "../"))
    ap.add_argument("--idxs", default="", help="comma list, e.g. 95,96,97,98,99")
    ap.add_argument("--cut", choices=["fm", "golden", "both"], default="both")
    ap.add_argument("--realize", choices=["compact", "faithful", "anchored"],
                    default="anchored")
    ap.add_argument("--refine", action="store_true",
                    help="run production refine stack (compact realize only)")
    ap.add_argument("--density", type=float, default=0.90,
                    help="room area = block area / density (whitespace prior)")
    ap.add_argument("--cluster-cohesion", dest="cluster_cohesion", type=float,
                    default=2.0, help="grouping cohesion budget per group edge")
    ap.add_argument("--preplaced-w", dest="preplaced_w", type=float, default=6.0)
    ap.add_argument("--pin-w", dest="pin_w", type=float, default=0.02)
    ap.add_argument("--pin-w-cap", dest="pin_w_cap", type=float, default=4.0)
    ap.add_argument("--embed-iters", dest="embed_iters", type=int, default=40)
    ap.add_argument("--fm-passes", dest="fm_passes", type=int, default=4)
    ap.add_argument("--balance-tol", dest="balance_tol", type=float, default=0.18)
    ap.add_argument("--anchor-hint-weight", dest="anchor_hint_weight",
                    type=float, default=2.0)
    ap.add_argument("--anchor-wall-weight", dest="anchor_wall_weight",
                    type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
