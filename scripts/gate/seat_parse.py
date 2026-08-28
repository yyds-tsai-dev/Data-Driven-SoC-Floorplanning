#!/usr/bin/env python
"""Join [seatgate]/[seat] stderr lines with per-case results. Usage: seat_parse.py dbg.err result.json"""
import json, re, sys
err, res = sys.argv[1], sys.argv[2]
gate, seat = {}, {}
for line in open(err):
    m = re.match(r"\[seatgate\] n=(\d+) rem=([\d.]+) span=([-\d.]+) need=([\d.inf]+) ok=(\d) fix=(\d)", line)
    if m:
        gate[int(m.group(1))] = dict(rem=float(m.group(2)), span=float(m.group(3)), need=float(m.group(4)), ok=int(m.group(5)))
    m = re.match(r"\[seat\] n=(\d+) rem=([\d.]+) wd=([\d.]+) ts=([\d.]+) nref=(\d+) npred=(\d+) left=([-\d.]+)", line)
    if m:
        seat[int(m.group(1))] = dict(rem=float(m.group(2)), wd=float(m.group(3)), ts=float(m.group(4)), nref=int(m.group(5)), npred=int(m.group(6)), left=float(m.group(7)))
d = json.load(open(res))
print("tid  n   cost   hpwl   rt   | gate rem  span  need ok | seat wd    ts    left")
for r in sorted(d["test_results"], key=lambda r: r["block_count"]):
    n = r["block_count"]; g = gate.get(n, {}); s = seat.get(n, {})
    if n < 70: continue
    print("%3d %3d %.3f  %.3f  %.2f | %5s %5s %5s %2s | %5s %5s %6s %s" % (
        r["test_id"], n, r["cost_no_runtime"], r["hpwl_gap"], r["runtime_seconds"],
        "%.3f"%g["rem"] if g else "-", "%.3f"%g["span"] if g else "-", "%.3f"%g["need"] if g else "-", g.get("ok","-"),
        "%.3f"%s["wd"] if s else "-", "%.3f"%s["ts"] if s else "-", "%.3f"%s["left"] if s else "-",
        "FALLBACK" if r["hpwl_gap"] > 0.15 else ""))
