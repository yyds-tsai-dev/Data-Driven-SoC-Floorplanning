#!/usr/bin/env python
"""Parse REFINER_DEBUG [pick] blocks + [seat] lines; join with results. Usage: pick_parse.py dbg.err dbg.json [dbg.out]
[pick] lines come in blocks per case (cand0 = column, cand1.. = direct). Blocks are attributed to cases in evaluation order via [seat] n=... markers."""
import json, re, sys
err, res = sys.argv[1], sys.argv[2]
d = json.load(open(res)); byn = {r["block_count"]: r for r in d["test_results"]}
cur_n = None; blocks = {}
for line in open(err):
    m = re.match(r"\[seat\] n=(\d+) ", line)
    if m: cur_n = int(m.group(1)); blocks.setdefault(cur_n, []); continue
    m = re.match(r"\[pick\] cand(\d+) direct=(\w+) hp=([\d.]+) area=([\d.]+) V=([\d.]+) \(bnd=(\d+) rest=([-\d.]+)\) score=([\d.]+)", line)
    if m and cur_n is not None:
        blocks[cur_n].append((int(m.group(1)), m.group(2) == "True", float(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6)), float(m.group(8))))
print("tid  n   cost  hpwl  | ncand  col_score  best_direct  ratio   col(hp,area,V)  best(hp,area,V)  | verdict")
for n in sorted(byn):
    if n < 76: continue
    r = byn[n]; b = blocks.get(n, [])
    if not b:
        print("%3d %3d  %.3f %.3f | no [pick] block (box empty -> column shipped)%s" % (r["test_id"], n, r["cost_no_runtime"], r["hpwl_gap"], "  FALLBACK" if r["hpwl_gap"] > 0.15 else ""))
        continue
    col = [x for x in b if x[0] == 0][0]; dirs = [x for x in b if x[1]]
    best = min(dirs, key=lambda x: x[6]) if dirs else None
    ratio = best[6] / col[6] if best else float("nan")
    verdict = "direct wins" if best and best[6] < col[6] * 0.985 else ("direct within margin" if best and best[6] < col[6] else "column better")
    print("%3d %3d  %.3f %.3f | %2d  %.4f  %.4f  %.3f  (%.0f,%.0f,%.0f)  (%.0f,%.0f,%.0f) | %s%s" % (
        r["test_id"], n, r["cost_no_runtime"], r["hpwl_gap"], len(dirs), col[6], best[6] if best else 0, ratio,
        col[2], col[3], col[4], best[2] if best else 0, best[3] if best else 0, best[4] if best else 0, verdict,
        "  FALLBACK" if r["hpwl_gap"] > 0.15 else ""))
