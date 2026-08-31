#!/usr/bin/env python
"""Paired comparison of budget-table arms: raw (no-runtime) + runtime-aware expected totals.
Usage: uv run python rt_pairs.py [--M 1.45] base1.json base2.json -- cand1.json cand2.json [-- cand2_1.json ...]
Prior over D (field speed-up vs 0823 medians): {0.6:0.15, 0.7:0.30, 0.8:0.30, 0.9:0.15, 1.0:0.10}."""
import csv, json, math, sys
PRIOR={0.6:0.15,0.7:0.30,0.8:0.30,0.9:0.15,1.0:0.10}
args=sys.argv[1:]; M=1.45
while args and args[0].startswith("--") and args[0]!="--":
    k=args.pop(0)
    if k=="--M": M=float(args.pop(0))
groups=[[]]
for a in args:
    if a=="--": groups.append([])
    else: groups[-1].append(a)
med={int(r["test_id"]):float(r["median_runtime_s"]) for r in csv.DictReader(open("docs/official/beta_test/C_median_runtimes_beta_hidden_update_20260823.csv"))}
def cnr(r): return 10.0 if not r["is_feasible"] else (1+0.5*(max(0,r["hpwl_gap"])+max(0,r["area_gap"])))*math.exp(2*r["violations_relative"])
def wt(n): return math.exp((n-120)/12)
def load(p):
    d=json.load(open(p)); rs=d["test_results"]; W=sum(wt(r["block_count"]) for r in rs)
    out={}
    for r in rs:
        w=wt(r["block_count"])/W; c=cnr(r); rt=r["runtime_seconds"]
        fac={D:max(0.7,(rt*M/(med[r["test_id"]]*D))**0.3) for D in PRIOR}
        out[r["test_id"]]=dict(n=r["block_count"],w=w,c=c,rt=rt,fac=fac)
    return out
def totals(o):
    raw=sum(v["w"]*v["c"] for v in o.values())
    byD={D:sum(v["w"]*v["c"]*v["fac"][D] for v in o.values()) for D in PRIOR}
    ev=sum(PRIOR[D]*byD[D] for D in PRIOR)
    tail=[v for v in o.values() if v["n"]>=105]; trt=sum(v["rt"] for v in tail)/len(tail)
    bands={}
    for lo,hi in ((21,75),(76,89),(90,104),(105,120)):
        vs=[v for v in o.values() if lo<=v["n"]<=hi]; bands[(lo,hi)]=sum(v["w"]*v["c"] for v in vs)
    return raw,byD,ev,trt,bands
names=["base"]+[f"cand{i}" for i in range(1,len(groups))]
res={}
for nm,g in zip(names,groups):
    res[nm]=[totals(load(p)) for p in g]
def mean(xs): return sum(xs)/len(xs)
print("%-7s %-7s %-7s | %s | %-7s %-7s | %s" % ("arm","raw","tail_rt","  ".join("D%.1f"%D for D in PRIOR),"EV","dEV","band raw <76/76-89/90-104/105+"))
for nm in names:
    rs=res[nm]; raw=mean([r[0] for r in rs]); byD={D:mean([r[1][D] for r in rs]) for D in PRIOR}; ev=mean([r[2] for r in rs]); trt=mean([r[3] for r in rs])
    bands=[mean([r[4][b] for r in rs]) for b in ((21,75),(76,89),(90,104),(105,120))]
    dev=ev-mean([r[2] for r in res["base"]])
    print("%-7s %.4f %.3f   | %s | %.4f %+.4f | %s" % (nm,raw,trt,"  ".join("%.4f"%byD[D] for D in PRIOR),ev,dev," / ".join("%.4f"%b for b in bands)))
# per-rep detail
for nm in names:
    for i,r in enumerate(res[nm]): print("  %s r%d raw=%.4f EV=%.4f tail_rt=%.3f D1.0=%.4f D0.7=%.4f" % (nm,i+1,r[0],r[2],r[3],r[1][1.0],r[1][0.7]))
