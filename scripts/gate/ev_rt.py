#!/usr/bin/env python
"""Runtime-aware expected total for one or more full-100 result JSONs.
Usage: uv run python ev_rt.py [--M 1.45] [--D 0.7,0.8,1.0] A.json [B.json ...]
Per-case total = cost_no_runtime * max(0.7, (rt_local*M/(median*D))**0.3), weighted exp((n-120)/12)."""
import csv, json, math, sys
args=sys.argv[1:]; M=1.45; Ds=[0.7,0.8,1.0]
while args and args[0].startswith("--"):
    k=args.pop(0)
    if k=="--M": M=float(args.pop(0))
    elif k=="--D": Ds=[float(x) for x in args.pop(0).split(",")]
med={int(r["test_id"]):float(r["median_runtime_s"]) for r in csv.DictReader(open("docs/official/beta_test/C_median_runtimes_beta_hidden_update_20260823.csv"))}
print("%-22s %8s %8s " % ("file","noRT","avg_rt") + " ".join("D=%.1f"%D for D in Ds) + "   over_floor_cases")
for p in args:
    d=json.load(open(p)); rs=d["test_results"]; mx=max(r["block_count"] for r in rs); W=sum(math.exp((r["block_count"]-mx)/12) for r in rs)
    outs=[]; over=[]
    for D in Ds:
        tot=0; ov=0
        for r in rs:
            w=math.exp((r["block_count"]-mx)/12)/W
            R=r["runtime_seconds"]*M/(med[r["test_id"]]*D)
            f=max(0.7, R**0.3); tot+=w*r["cost_no_runtime"]*f
            if f>0.7: ov+=1
        outs.append(tot); over.append(ov)
    print("%-22s %8.4f %8.3f " % (p.split("/")[-1], d["total_score_no_runtime"], d["summary"]["avg_runtime"]) + " ".join("%.4f"%x for x in outs) + "   " + "/".join(str(x) for x in over))
