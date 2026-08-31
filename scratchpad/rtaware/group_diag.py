#!/usr/bin/env python
"""Diagnose residual GROUPING violations: for each split cluster, list components and test whether the
minor component's block(s) can be re-seated flush against the major component inside the current bbox
without overlapping anything (free) or not (packed). Usage (from FloorSet/iccad2026contest):
  uv run python .../group_diag.py RESULT.json --data-path ../ [--min-n 96] [--ids 90,97]"""
import argparse, json, sys, math
from pathlib import Path
import numpy as np, torch
_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO/"FloorSet"/"iccad2026contest"), str(_REPO/"FloorSet")):
    if _p not in sys.path: sys.path.insert(0,_p)
from litetestLoader import FloorplanDatasetLiteTest
EPS=1e-6
def touch(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ox=min(ax+aw,bx+bw)-max(ax,bx); oy=min(ay+ah,by+bh)-max(ay,by)
    return (abs(ox)<1e-6 and oy>1e-6) or (abs(oy)<1e-6 and ox>1e-6) or (ox>1e-6 and oy>1e-6)
def overlaps(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    return min(ax+aw,bx+bw)-max(ax,bx)>1e-6 and min(ay+ah,by+bh)-max(ay,by)>1e-6
def components(idx,P):
    idx=list(idx); comp=[]; seen=set()
    for s in idx:
        if s in seen: continue
        st=[s]; c=set([s]); seen.add(s)
        while st:
            u=st.pop()
            for v in idx:
                if v not in seen and touch(P[u],P[v]): seen.add(v); c.add(v); st.append(v)
        comp.append(sorted(c))
    return comp
def free_seats(i,major,P,bbox,fixed):
    """candidate placements of block i flush to some block of `major`, inside bbox, no overlap."""
    x0,y0,x1,y1=bbox; w0,h0=P[i][2],P[i][3]; shapes=[(w0,h0)] if fixed else [(w0,h0),(h0,w0)]
    others=[j for j in range(len(P)) if j!=i]; found=0
    for (w,h) in shapes:
        for m in major:
            mx,my,mw,mh=P[m]
            cands=[]
            for t in np.linspace(0,1,7):
                cands.append((mx+mw, my+t*(mh-h))); cands.append((mx-w, my+t*(mh-h)))   # right/left of m
                cands.append((mx+t*(mw-w), my+mh)); cands.append((mx+t*(mw-w), my-h))   # above/below
            for (cx,cy) in cands:
                if cx<x0-EPS or cy<y0-EPS or cx+w>x1+EPS or cy+h>y1+EPS: continue
                r=(cx,cy,w,h)
                if any(overlaps(r,P[j]) for j in others): continue
                found+=1
    return found
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("result"); ap.add_argument("--data-path",default="../"); ap.add_argument("--min-n",type=int,default=0); ap.add_argument("--ids",default="")
    a=ap.parse_args(); d=json.load(open(a.result)); rows={r["test_id"]:r for r in d["test_results"]}
    want=[int(t) for t in a.ids.split(",") if t.strip()] if a.ids else sorted(rows)
    ds=FloorplanDatasetLiteTest(a.data_path); tally={"free":0,"packed":0}
    for tid in want:
        r=rows[tid]
        if r["block_count"]<a.min_n or not r.get("grouping_violations"): continue
        s=ds[tid]; area_target,b2b,p2b,pins,cons=s["input"]; n=r["block_count"]; cons=cons[:n]
        P=[tuple(map(float,p)) for p in (r.get("positions") or r["raw_positions"])]
        xs=[p[0] for p in P]; ys=[p[1] for p in P]; bbox=(min(xs),min(ys),max(p[0]+p[2] for p in P),max(p[1]+p[3] for p in P))
        cl=cons[:,3].tolist(); groups={}
        for i,c in enumerate(cl):
            if c>0: groups.setdefault(int(c),[]).append(i)
        print(f"tid={tid} n={n} grp_eval={r['grouping_violations']} v_rel={r['violations_relative']:.4f} bbox={bbox[2]-bbox[0]:.1f}x{bbox[3]-bbox[1]:.1f}")
        for g,idx in groups.items():
            comp=components(idx,P)
            if len(comp)<=1: continue
            comp.sort(key=len,reverse=True); major=comp[0]
            for minor in comp[1:]:
                for i in minor:
                    fixed=bool(cons[i,0] or cons[i,1]); pre=bool(cons[i,1])
                    fs=0 if pre else free_seats(i,major,P,bbox,fixed)
                    cls="preplaced" if pre else ("free" if fs else "packed")
                    tally["free" if fs else "packed"]+=1
                    print(f"   group={g} size={len(idx)} comps={[len(c) for c in comp]} minor_block={i} area={P[i][2]*P[i][3]:.0f} w,h={P[i][2]:.1f},{P[i][3]:.1f} fixed={fixed} cls={cls} seats={fs}")
    print("TALLY",tally)
main()
