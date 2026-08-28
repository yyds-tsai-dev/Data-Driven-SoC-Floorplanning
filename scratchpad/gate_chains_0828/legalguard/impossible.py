"""For every instance in the stress grid: count INPUT-level impossibilities,
i.e. pairs of PREPLACED blocks whose mandated rectangles overlap (evaluator
tolerance 1e-6).  Such an instance has no feasible solution at all."""
import os, sys, importlib.util
REPO = "/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning"
sys.path.insert(0, os.path.join(REPO, "tests"))
spec = importlib.util.spec_from_file_location(
    "stressmod", "/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/stress/stress.py")
# don't exec stress.py (it imports the package); re-declare the grid by exec'ing
# only build_grid's source
src = open("/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/stress/stress.py").read()
start = src.index("def build_grid()"); end = src.index("def dummy_baseline()")
ns = {}
exec(src[start:end], ns)
build_grid = ns["build_grid"]
from synth_instances import build_instance

rows = []
for name, kw in build_grid():
    inst = build_instance(**kw)
    c = inst.constraints; tp = inst.target_positions
    pre = [i for i in range(kw["n"]) if float(c[i, 1]) != 0.0]
    bad = []
    for a in range(len(pre)):
        for b in range(a + 1, len(pre)):
            i, j = pre[a], pre[b]
            x1, y1, w1, h1 = [float(v) for v in tp[i]]
            x2, y2, w2, h2 = [float(v) for v in tp[j]]
            ox = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
            oy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
            if ox > 1e-6 and oy > 1e-6:
                bad.append((i, j))
    rows.append((name, len(pre), len(bad), bad[:3]))

import csv
with open("input_impossible.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name", "n_preplaced", "n_overlapping_preplaced_pairs", "example_pairs"])
    for r in rows: w.writerow(r)
imp = [r for r in rows if r[2] > 0]
print(f"instances total={len(rows)} with impossible input={len(imp)}")
for r in imp: print("  IMPOSSIBLE", r[0], "pre=%d pairs=%d" % (r[1], r[2]), r[3])
