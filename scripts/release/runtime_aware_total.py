#!/usr/bin/env python
"""Runtime-aware totals from full-100 result JSONs (recomputes no-runtime cost from components).
Usage: uv run python rt_total.py [--M 1.45] [--D 0.6,0.7,0.8,1.0] [--band 105] A.json ...
Per-case total = cost_noRT * max(0.7, (rt_local*M/(median*D))**0.3), weight exp((n-120)/12)."""
import csv, json, math, sys
args=sys.argv[1:]; M=1.45; Ds=[0.6,0.7,0.8,1.0]; band=105
while args and args[0].startswith("--"):
    k=args.pop(0)
    if k=="--M": M=float(args.pop(0))
    elif k=="--D": Ds=[float(x) for x in args.pop(0).split(",")]
    elif k=="--band": band=int(args.pop(0))
med={int(r["test_id"]):float(r["median_runtime_s"]) for r in csv.DictReader(open("docs/official/beta_test/C_median_runtimes_beta_hidden_update_20260823.csv"))}
def cnr(r): return 10.0 if not r["is_feasible"] else (1+0.5*(max(0,r["hpwl_gap"])+max(0,r["area_gap"])))*math.exp(2*r["violations_relative"])
print("%-26s %7s %7s %7s | %s | %s" % ("file","raw","floor","tailrt"," ".join("D%.1f"%D for D in Ds), "tail_raw tail_rt_avg"))
for p in args:
    d=json.load(open(p)); rs=d["test_results"]; mx=120; W=sum(math.exp((r["block_count"]-mx)/12) for r in rs)
    raw=sum(math.exp((r["block_count"]-mx)/12)/W*cnr(r) for r in rs)
    tail=[r for r in rs if r["block_count"]>=band]; Wt=sum(math.exp((r["block_count"]-mx)/12) for r in tail)
    traw=sum(math.exp((r["block_count"]-mx)/12)/Wt*cnr(r) for r in tail); trt=sum(r["runtime_seconds"] for r in tail)/len(tail)
    outs=[]
    for D in Ds:
        tot=0
        for r in rs:
            w=math.exp((r["block_count"]-mx)/12)/W; R=r["runtime_seconds"]*M/(med[r["test_id"]]*D); tot+=w*cnr(r)*max(0.7,R**0.3)
        outs.append(tot)
    print("%-26s %7.4f %7.4f %7.3f | %s | %.4f %.3f" % (p.split("/")[-1][:26], raw, 0.7*raw, d["summary"]["avg_runtime"], " ".join("%.4f"%x for x in outs), traw, trt))
