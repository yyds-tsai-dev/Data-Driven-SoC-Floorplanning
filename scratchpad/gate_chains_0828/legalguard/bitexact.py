import os, sys, time, json, hashlib
sys.argv=[sys.argv[0]]
exec(open("repro.py").read().split('if __name__')[0])
cases = [dict(n=21, seed=1, frac_fixed=0.6), dict(n=21, seed=1), dict(n=60, seed=1),
         dict(n=90, seed=2), dict(n=105, seed=1, frac_fixed=0.6), dict(n=120, seed=1)]
out = {}
opt = MyOptimizer()
for kw in cases:
    n = kw["n"]; inst = build_instance(**kw)
    tp = make_target_positions(n, inst.constraints, inst.target_positions)
    pos = opt.solve(n, inst.area_targets, inst.b2b, inst.p2b, inst.pins, inst.constraints, tp)
    pos = opt.solve(n, inst.area_targets, inst.b2b, inst.p2b, inst.pins, inst.constraints, tp)
    m = evaluate_solution({'positions': pos, 'runtime': 1.0}, {}, inst.constraints, inst.b2b,
                          inst.p2b, inst.pins, inst.area_targets, inst.target_positions, median_runtime=1.0)
    key = str(kw)
    out[key] = dict(h=hashlib.md5(json.dumps([[float(v) for v in r] for r in pos]).encode()).hexdigest(),
                    feas=bool(m.is_feasible), ov=int(m.overlap_violations))
print(json.dumps(out, indent=0))
