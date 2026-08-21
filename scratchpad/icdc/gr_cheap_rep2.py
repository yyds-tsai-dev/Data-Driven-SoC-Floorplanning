"""Production-shaped cheap variant: no greedy, no B/C, one alternating round.

Per case at most 3 LP pairs:  enforce-all -> (if infeasible/worse) polish-only.
This is what an in-solver post-pass could actually afford.
"""
import sys, json, time, statistics as st
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_lib as G
from gr_repair2 import split

SC = Path(__file__).resolve().parent
ev = G.load_evaluator()
cases = G.load_cases(ev, "../")
prod = {int(t["test_id"]): t for t in json.load(open(SC/"eval_prod_rep2.json"))["test_results"]}

rows=[]
for c in cases:
    base=[tuple(p) for p in prod[c.test_id]["positions"]]; n=c.n
    bm=G.evaluate(ev,c,base)
    t0=time.perf_counter()
    a=G.build_assign_robust(base)
    hor=G.transitive_reduce(n,split(a)[0]); ver=G.transitive_reduce(n,split(a)[1])
    sw=np.array([r[2] for r in base]); sh=np.array([r[3] for r in base])
    x0=min(r[0] for r in base); x1=max(r[0]+r[2] for r in base)
    y0=min(r[1] for r in base); y1=max(r[1]+r[3] for r in base)
    pre=set(np.nonzero(c.preplaced)[0].tolist())
    eqx={i:float(base[i][0]) for i in pre}; eqy={i:float(base[i][1]) for i in pre}
    stt=G.boundary_status(base,c.bound)
    want={i:{b for b in (1,2,4,8) if cd&b} for i,(cd,m) in stt.items()}
    ct,_=G.cluster_contacts(base,c.clust)
    rx=G.contact_rows(ct,base,"x"); ry=G.contact_rows(ct,base,"y")
    nx_,px_=G.collect_nets(c,"x"); ny_,py_=G.collect_nets(c,"y")
    hb=max(c.baseline["hpwl_baseline"],1e-6); ab=max(c.baseline["area_baseline"],1e-6)
    def attempt(sel):
        sx=G.solve_axis_ext(n,sw,np.full(n,x0),np.full(n,x1),hor,nx_,px_,eqx,
                            [i for i in sel if 1 in want[i]],[i for i in sel if 2 in want[i]],
                            rx,True,0.5/hb,0.5*(y1-y0)/ab)
        if sx is None: return None
        cw=max(sx+sw)-min(sx)
        sy=G.solve_axis_ext(n,sh,np.full(n,y0),np.full(n,y1),ver,ny_,py_,eqy,
                            [i for i in sel if 8 in want[i]],[i for i in sel if 4 in want[i]],
                            ry,True,0.5/hb,0.5*cw/ab)
        if sy is None: return None
        lay=[(float(sx[i]),float(sy[i]),float(sw[i]),float(sh[i])) for i in range(n)]
        for (i,j,ax,_d) in ct:
            if ax=="x": lay[j]=(lay[i][0]+lay[i][2],lay[j][1],lay[j][2],lay[j][3])
            else:       lay[j]=(lay[j][0],lay[i][1]+lay[i][3],lay[j][2],lay[j][3])
        return lay
    best=(bm.cost_no_runtime,list(base))
    for sel in (sorted(want), []):
        lay=attempt(sel)
        if lay is None: continue
        m=G.evaluate(ev,c,lay)
        if m.is_feasible and m.cost_no_runtime<best[0]-1e-12: best=(m.cost_no_runtime,lay)
        if sel and best[1] is not base and best[0]<bm.cost_no_runtime-1e-12: break
    el=time.perf_counter()-t0
    fm=G.evaluate(ev,c,best[1])
    rows.append({"test_id":c.test_id,"n":n,"base":bm.cost_no_runtime,"final":best[0],"t":el,
                 "feas":bool(fm.is_feasible),"ov":int(fm.overlap_violations),
                 "av":int(fm.area_violations),"dv":int(fm.dimension_violations),
                 "bvb":int(bm.boundary_violations),"bvg":int(bm.grouping_violations),
                 "vb":int(fm.boundary_violations),"vg":int(fm.grouping_violations),
                 "vm":int(fm.mib_violations)})
ns=[r["n"] for r in rows]
print("total base    ", round(ev.compute_total_score([r["base"] for r in rows],ns),6))
print("total cheap-A-r2 ", round(ev.compute_total_score([r["final"] for r in rows],ns),6),
      " gain", round(ev.compute_total_score([r["final"] for r in rows],ns)-ev.compute_total_score([r["base"] for r in rows],ns),6))
tailv=[r["final"] if r["n"]>=95 else r["base"] for r in rows]
print("total cheap-A-r2 tail n>=95", round(ev.compute_total_score(tailv,ns),6),
      " gain", round(ev.compute_total_score(tailv,ns)-ev.compute_total_score([r["base"] for r in rows],ns),6))
t=[r["t"] for r in rows]
print("time median %.3f p90 %.3f max %.3f sum %.1f"%(st.median(t),sorted(t)[89],max(t),sum(t)))
tt=[r["t"] for r in rows if r["n"]>=95]
print("time n>=95: median %.3f max %.3f sum %.1f (%d cases)"%(st.median(tt),max(tt),sum(tt),len(tt)))
print("feasible",sum(1 for r in rows if r["feas"]),"overlap",sum(r["ov"] for r in rows),
      "area",sum(r["av"] for r in rows),"dim",sum(r["dv"] for r in rows))
print("viol bnd %d->%d  grp %d->%d  mib ->%d"%(sum(r["bvb"] for r in rows),sum(r["vb"] for r in rows),
      sum(r["bvg"] for r in rows),sum(r["vg"] for r in rows),sum(r["vm"] for r in rows)))
print("improved",sum(1 for r in rows if r["final"]<r["base"]-1e-12))
json.dump(rows,open(SC/"gr_cheap_rep2.json","w"),indent=1)
