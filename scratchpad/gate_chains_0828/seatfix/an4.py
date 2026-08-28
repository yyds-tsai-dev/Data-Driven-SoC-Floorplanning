import re,sys,collections,statistics as st
end = re.compile(r"\[lad\] p=(\d+) n=(\d+) d=(\d+) END legal=(\d) reb=(\d) slice=([\d.]+) res=([\d.]+) lad=([\d.]+) left=(-?[\d.]+)")
sffb = re.compile(r"\[sf\] n=(\d+) FALLBACK tag=(\S+) ok=(\d) ")
ev=[]
for line in open(sys.argv[1],errors="replace"):
    for m in end.finditer(line): ev.append(("end", int(m.group(2)), int(m.group(4))))
    for m in sffb.finditer(line): ev.append(("fb", int(m.group(1)), int(m.group(3))))
# segment into cases by contiguous n
cases=[]; cur=None
for kind,n,ok in ev:
    if cur is None or cur['n']!=n:
        cur=dict(n=n, calls=0, legal=0, fb=0, fbok=0); cases.append(cur)
    if kind=="end": cur['calls']+=1; cur['legal']+=ok
    else: cur['fb']+=1; cur['fbok']+=ok
print("segments:", len(cases))
empty=[c for c in cases if c['legal']+c['fbok']==0]
low=[c for c in cases if 0 < c['legal']+c['fbok'] <= 2]
print("empty-direct segments:", len(empty), [(c['n'],c['calls']) for c in empty])
print("<=2 candidate segments:", len(low), [(c['n'],c['calls'],c['legal']+c['fbok']) for c in low])
sup=collections.defaultdict(list)
for c in cases:
    b="a<76" if c['n']<76 else "b76-89" if c['n']<90 else "c90-104" if c['n']<105 else "d105-120"
    sup[b].append((c['calls'], c['legal']+c['fbok']))
for b in sorted(sup):
    v=sup[b]; print(f"{b:10s} segs={len(v):3d} seats={sum(x[0] for x in v):4d} supply={sum(x[1] for x in v):4d} rate={sum(x[1] for x in v)/max(sum(x[0] for x in v),1):.3f} per-seg supply med={st.median(x[1] for x in v):.1f}")
