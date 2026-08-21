"""Does the model prediction already reproduce the GOLDEN relative order?

For every ordered pair the golden layout separates (i strictly left-of / below
j with no overlap in the other axis is not required -- we use the full
separation relation), check whether the prediction preserves it.  A model that
already has golden topology has a PRECISION problem, not a STRUCTURE problem.
"""
import json, sys
from pathlib import Path
import numpy as np
SS = Path("/tmp/claude-1100/-nashome-NVL4-vdalab-yyds-dev-Data-Driven-SoC-Floorplanning/c70131c1-0951-4c49-9b1e-f46a14d6ebfc/scratchpad")
sys.path.insert(0, str(SS)); import gr_lib as G

ev = G.load_evaluator(); cases = {c.test_id: c for c in G.load_cases(ev, "../")}
pred = {int(k): np.asarray(v,float) for k,v in json.load(open(SS/"alpha_pred0.json")).items()}
gold = {int(k): np.asarray(v,float) for k,v in json.load(open(SS/"gr_layouts3.json")).items()}

def sep(P):
    """boolean [n,n]: R[i,j] True iff i entirely left of j (x-axis)."""
    x1 = P[:,0]+P[:,2]
    return x1[:,None] <= P[None,:,0] + 1e-9

tot_e = tot_bad = 0
rows=[]
for tid in sorted(pred):
    c = cases[tid]
    P0, P1 = pred[tid], gold[tid][:c.n]
    # golden separation relations on each axis
    gx = sep(P1); gy = sep(P1[:,[1,0,3,2]])
    px = sep(P0); py = sep(P0[:,[1,0,3,2]])
    # a pair is "ordered" in golden if separated on >=1 axis; the pair is
    # preserved if the SAME axis relation still holds in the prediction.
    gpair = gx | gy
    keep  = (gx & px) | (gy & py)
    n = c.n
    iu = np.triu_indices(n,1)
    # consider both directions
    tot = int(gpair.sum())
    ok  = int((gpair & keep).sum())
    bad = tot - ok
    tot_e += tot; tot_bad += bad
    # magnitude of the violation, in layout units
    viol = np.where(gpair & ~keep)
    mag = 0.0
    if len(viol[0]):
        dx = (P0[viol[0],0]+P0[viol[0],2]) - P0[viol[1],0]
        dy = (P0[viol[0],1]+P0[viol[0],3]) - P0[viol[1],1]
        mag = float(np.minimum(np.abs(dx), np.abs(dy)).mean())
    rows.append((tid, c.n, tot, bad, 100.0*bad/max(tot,1), mag))

print(f"{'tid':>4} {'n':>4} {'gold_pairs':>11} {'broken':>7} {'%':>6} {'mean_mag':>9}")
for r in rows:
    if r[1] >= 103:
        print(f"{r[0]:>4} {r[1]:>4} {r[2]:>11} {r[3]:>7} {r[4]:>6.2f} {r[5]:>9.3f}")
print(f"\nlive band total: golden ordered pairs={sum(r[2] for r in rows if r[1]>=103)} "
      f"broken by the model={sum(r[3] for r in rows if r[1]>=103)} "
      f"({100.0*sum(r[3] for r in rows if r[1]>=103)/max(1,sum(r[2] for r in rows if r[1]>=103)):.3f}%)")
