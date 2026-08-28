"""Re-run the stress grid through the WORKING TREE partner/ (packaged env),
splitting the report by whether the INPUT admits a legal layout at all."""
import os, sys, csv, time
sys.argv = [sys.argv[0]]
exec(open("repro.py").read().split('if __name__')[0])

src = open("/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/stress/stress.py").read()
ns = {}
exec(src[src.index("def build_grid()"):src.index("def dummy_baseline()")], ns)

def impossible_pairs(inst, n):
    c, tp = inst.constraints, inst.target_positions
    pre = [i for i in range(n) if float(c[i, 1]) != 0.0]
    bad = 0
    for a in range(len(pre)):
        for b in range(a + 1, len(pre)):
            i, j = pre[a], pre[b]
            x1, y1, w1, h1 = [float(v) for v in tp[i]]
            x2, y2, w2, h2 = [float(v) for v in tp[j]]
            if (min(x1+w1, x2+w2) - max(x1, x2)) > 1e-6 and \
               (min(y1+h1, y2+h2) - max(y1, y2)) > 1e-6:
                bad += 1
    return bad

opt = MyOptimizer()
rows = []
for name, kw in ns["build_grid"]():
    n = kw["n"]; inst = build_instance(**kw)
    tp = make_target_positions(n, inst.constraints, inst.target_positions)
    imp = impossible_pairs(inst, n)
    t0 = time.time()
    try:
        pos = opt.solve(n, inst.area_targets, inst.b2b, inst.p2b, inst.pins, inst.constraints, tp)
        exc = ""
    except Exception as e:
        pos = None; exc = f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    if pos is None:
        rows.append((name, n, imp, None, None, dt, exc)); continue
    m = evaluate_solution({'positions': pos, 'runtime': dt}, {}, inst.constraints, inst.b2b,
                          inst.p2b, inst.pins, inst.area_targets, inst.target_positions,
                          median_runtime=1.0)
    rows.append((name, n, imp, bool(m.is_feasible), int(m.overlap_violations), dt, exc))
    print(f"{name}: imp_pairs={imp} feasible={m.is_feasible} ov={m.overlap_violations} rt={dt:.3f}", flush=True)

with open("sweep_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name","n","impossible_input_pairs","is_feasible","overlap_violations","runtime","exception"])
    for r in rows: w.writerow(r)

poss = [r for r in rows if r[2] == 0]
imps = [r for r in rows if r[2] > 0]
rts = sorted(r[5] for r in rows)
print("\n===== SWEEP SUMMARY (working-tree partner/, packaged env) =====")
print(f"instances={len(rows)}  exceptions={sum(1 for r in rows if r[6])}")
print(f"SATISFIABLE inputs: {len(poss)}  feasible={sum(1 for r in poss if r[3])}  infeasible={sum(1 for r in poss if r[3] is False)}")
bad = [r for r in poss if r[3] is False]
for r in bad: print("   GENUINE INFEASIBLE:", r)
print(f"IMPOSSIBLE inputs (overlapping preplaced mandate): {len(imps)}  feasible={sum(1 for r in imps if r[3])}")
print(f"runtime: mean={sum(rts)/len(rts):.3f} p90={rts[int(0.9*(len(rts)-1))]:.3f} max={rts[-1]:.3f}")
