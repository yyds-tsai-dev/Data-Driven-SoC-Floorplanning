import json, math, sys
sh={r["test_id"]:r for r in json.load(open(sys.argv[1]))["test_results"]}
gd={r["tid"]:r for r in (json.loads(l) for l in open(sys.argv[2]))}
runs=[(f, [json.loads(l) for l in open(f)]) for f in sys.argv[3:]]
tids=sorted(gd)
mx=max(gd[t]["n"] for t in tids); W=lambda n: math.exp((n-mx)/12)
tw=sum(W(gd[t]["n"]) for t in tids)
def wav(f): return sum(W(gd[t]["n"])*f(t) for t in tids)/tw
print("cases=%d  golden-seed p_cost=%.4f   shipped=%.4f"%(len(tids),
  wav(lambda t: gd[t]["p_cost"]), wav(lambda t: sh[t]["cost_no_runtime"])))
for f,rows in runs:
    d={r["tid"]:r for r in rows}
    ok=[t for t in tids if t in d]
    tw2=sum(W(gd[t]["n"]) for t in ok)
    def w2(fn,default=None): return sum(W(gd[t]["n"])*fn(d[t]) for t in ok)/tw2
    nref=sum(1 for t in ok if d[t].get("p_cost") is None)
    wins=[t for t in ok if d[t].get("p_cost") is not None and d[t]["p_cost"]<sh[t]["cost_no_runtime"]]
    print("%-34s n=%d  fail=%d  seed[hpwl %+.3f area %+.3f ovf %.2f]  post[cost %.4f hpwl %+.4f area %+.4f V %.4f]  ls_ms=%.1f wins=%s"%(
      f.split("/")[-1], len(ok), nref,
      w2(lambda r:r["s_hpwl"]), w2(lambda r:r["s_area"]), w2(lambda r:r["s_ovf"]),
      w2(lambda r:r.get("p_cost", r.get("r_cost", 8.0))),
      w2(lambda r:r.get("p_hpwl",0.0)), w2(lambda r:r.get("p_area",0.0)),
      w2(lambda r:r.get("p_V",0.0)), w2(lambda r:r["ls_ms"]), wins))
print()
hdr="tid   n | ship   golden | "+" | ".join(f.split("/")[-1][7:-6] for f,_ in runs)
print(hdr)
for t in tids:
    line="%3d %4d | %.4f %.4f"%(t, gd[t]["n"], sh[t]["cost_no_runtime"], gd[t]["p_cost"])
    for f,rows in runs:
        d={r["tid"]:r for r in rows}
        r=d.get(t)
        if r is None: line+=" |   --  "
        elif r.get("p_cost") is None: line+=" | FAIL(%.2f ovf)"%r["s_ovf"]
        else: line+=" | %.4f"%r["p_cost"]
    print(line)
