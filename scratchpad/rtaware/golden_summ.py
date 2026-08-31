import json, math, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
ref=None
if len(sys.argv)>2:
    d=json.load(open(sys.argv[2])); ref={r["test_id"]:r for r in d["test_results"]}
mx=max(r["n"] for r in rows); W=lambda n: math.exp((n-mx)/12)
tw=sum(W(r["n"]) for r in rows)
def wavg(f): return sum(W(r["n"])*f(r) for r in rows)/tw
print("cases=%d"%len(rows))
print("golden  cost=%.4f V=%.4f"%(wavg(lambda r:r["g_cost"]), wavg(lambda r:r["g_V"])))
for p,lab in (("r_","refine"),("p_","postpass")):
    if all(p+"cost" in r for r in rows):
        print("%-8s cost=%.4f hpwl=%.4f area=%.4f V=%.4f"%(lab,
          wavg(lambda r:r[p+"cost"]), wavg(lambda r:r[p+"hpwl"]),
          wavg(lambda r:r[p+"area"]), wavg(lambda r:r[p+"V"])))
if ref:
    sub=[ref[r["tid"]] for r in rows]
    tw2=sum(W(x["block_count"]) for x in sub)
    print("shipped cost=%.4f hpwl=%.4f area=%.4f V=%.4f"%(
      sum(W(x["block_count"])*x["cost_no_runtime"] for x in sub)/tw2,
      sum(W(x["block_count"])*max(0,x["hpwl_gap"]) for x in sub)/tw2,
      sum(W(x["block_count"])*max(0,x["area_gap"]) for x in sub)/tw2,
      sum(W(x["block_count"])*x["violations_relative"] for x in sub)/tw2))
print()
hdr="tid  n  gV  gcost | rH rA rV rcost | pH pA pV pcost | ship"
print(hdr)
for r in sorted(rows,key=lambda r:r["tid"]):
    s=ref[r["tid"]]["cost_no_runtime"] if ref else float('nan')
    print("%3d %4d %.3f %.4f | %+.4f %+.4f %.3f %.4f | %+.4f %+.4f %.3f %.4f | %.4f  %s"%(
      r["tid"],r["n"],r["g_V"],r["g_cost"],
      r.get("r_hpwl",float('nan')),r.get("r_area",float('nan')),r.get("r_V",float('nan')),r.get("r_cost",float('nan')),
      r.get("p_hpwl",float('nan')),r.get("p_area",float('nan')),r.get("p_V",float('nan')),r.get("p_cost",float('nan')),
      s, json.dumps(r["g_cls"])))
