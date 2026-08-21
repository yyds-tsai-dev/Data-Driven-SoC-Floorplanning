import json, sys
from pathlib import Path
import numpy as np
SS = Path("/tmp/claude-1100/-nashome-NVL4-vdalab-yyds-dev-Data-Driven-SoC-Floorplanning/c70131c1-0951-4c49-9b1e-f46a14d6ebfc/scratchpad")
sys.path.insert(0, str(SS)); import gr_lib as G
ev = G.load_evaluator(); cases = {c.test_id: c for c in G.load_cases(ev, "../")}
pred = {int(k): np.asarray(v,float) for k,v in json.load(open(SS/"alpha_pred0.json")).items()}
gold = {int(k): np.asarray(v,float) for k,v in json.load(open(SS/"gr_layouts3.json")).items()}
for tid in (82, 90, 99):
    c = cases[tid]; P0 = pred[tid]; P1 = gold[tid][:c.n]
    print(f"--- tid {tid} n={c.n} frame={tuple(round(f,2) for f in c.frame)}")
    print("  pred[:4] ", np.round(P0[:4],3).tolist())
    print("  gold[:4] ", np.round(P1[:4],3).tolist())
    cen0 = P0[:,:2]+0.5*P0[:,2:]; cen1 = P1[:,:2]+0.5*P1[:,2:]
    d = np.linalg.norm(cen0-cen1,axis=1)
    x0,y0,x1,y1 = c.frame; diag = np.hypot(x1-x0,y1-y0)
    print(f"  centroid dist: mean={d.mean():.3f} med={np.median(d):.3f} p90={np.percentile(d,90):.3f} max={d.max():.3f} diag={diag:.3f}")
    print(f"  pred bbox=({P0[:,0].min():.2f},{P0[:,1].min():.2f})-({(P0[:,0]+P0[:,2]).max():.2f},{(P0[:,1]+P0[:,3]).max():.2f})")
    print(f"  pred area={((P0[:,0]+P0[:,2]).max()-P0[:,0].min())*((P0[:,1]+P0[:,3]).max()-P0[:,1].min()):.1f} gold area={(x1-x0)*(y1-y0):.1f}")
    m0 = G.evaluate(ev, c, [tuple(map(float,r)) for r in P0])
    m1 = G.evaluate(ev, c, [tuple(map(float,r)) for r in P1])
    print(f"  pred cost={m0.cost_no_runtime:.4f} feasible={getattr(m0,'feasible',None)}  gold cost={m1.cost_no_runtime:.4f}")
