import re, sys, collections, statistics as st
path = sys.argv[1]
lad = re.compile(r"\[lad\] p=(\d+) n=(\d+) d=(\d+) ridx=(\d+) e=([\d.]+) pins=(\d) ok=(\d) win=(-?[\d.]+) t=([\d.]+)")
end = re.compile(r"\[lad\] p=(\d+) n=(\d+) d=(\d+) END legal=(\d) reb=(\d) slice=([\d.]+) res=([\d.]+) lad=([\d.]+) left=(-?[\d.]+)")
cur = collections.defaultdict(list); calls=[]
for line in open(path, errors="replace"):
    for m in lad.finditer(line):
        cur[m.group(1)].append((int(m.group(4)), float(m.group(5)), int(m.group(7)), float(m.group(9))))
    for m in end.finditer(line):
        p=m.group(1)
        calls.append(dict(n=int(m.group(2)), legal=int(m.group(4)), slice=float(m.group(6)),
                          res=float(m.group(7)), lad=float(m.group(8)), left=float(m.group(9)),
                          rungs=cur.pop(p, [])))
def band(n):
    return "a<76" if n<76 else "b76-89" if n<90 else "c90-104" if n<105 else "d105-120"
rows=collections.defaultdict(list)
for c in calls:
    span = c['slice']-c['res']
    r0ok = (len(c['rungs'])==0 and c['legal']==1)
    if r0ok: rows[(band(c['n']),'R0OK')].append((span, c['lad'], c['lad']))
    elif c['legal']==0:
        t_r0 = c['lad']-sum(x[3] for x in c['rungs'])
        rows[(band(c['n']),'LADFAIL')].append((span, t_r0, c['lad']))
    else:
        t_r0 = c['lad']-sum(x[3] for x in c['rungs'])
        rows[(band(c['n']),'EXPOK')].append((span, t_r0, c['lad']))
print(f"{'band':10s} {'kind':8s} {'N':>4s} {'span':>7s} {'t_r0':>7s} {'r0/span':>8s} {'lad':>7s} {'lad/span':>8s}")
for k in sorted(rows):
    v=rows[k]; sp=st.median(x[0] for x in v); t0=st.median(x[1] for x in v); ld=st.median(x[2] for x in v)
    print(f"{k[0]:10s} {k[1]:8s} {len(v):4d} {sp:7.4f} {t0:7.4f} {t0/max(sp,1e-9):8.2f} {ld:7.4f} {ld/max(sp,1e-9):8.2f}")
# expand-rung timing detail
det=collections.defaultdict(list)
for c in calls:
    for (ridx,e,ok,t) in c['rungs']:
        det[(band(c['n']),e,ok)].append(t)
print("\nexpand rungs: band e ok N median_t")
for k in sorted(det): print(f"  {k[0]:10s} e={k[1]:<5} ok={k[2]} N={len(det[k]):4d} t={st.median(det[k]):.4f}")
