import re,sys,collections
lad = re.compile(r"\[lad\] p=(\d+) n=(\d+) d=(\d+) ridx=(\d+) e=([\d.]+) pins=(\d) ok=(\d) win=(-?[\d.]+) t=([\d.]+)")
end = re.compile(r"\[lad\] p=(\d+) n=(\d+) d=(\d+) END legal=(\d) reb=(\d) slice=([\d.]+) res=([\d.]+) lad=([\d.]+) left=(-?[\d.]+)")
sffb = re.compile(r"\[sf\] n=(\d+) FALLBACK tag=(\S+) ok=(\d) bbr=(-?[\d.]+) V=(-?\d+)")
def load(p):
    cur=collections.defaultdict(list); ev=[]
    for line in open(p,errors="replace"):
        for m in lad.finditer(line): cur[m.group(1)].append((float(m.group(5)), int(m.group(7))))
        for m in end.finditer(line):
            rr=cur.pop(m.group(1),[]); legal=int(m.group(4))
            ev.append(("end", int(m.group(2)), (0.0 if legal and not rr else (rr[-1][0] if legal else None))))
        for m in sffb.finditer(line):
            ev.append(("fb", int(m.group(1)), (float(m.group(2)[2:]) if m.group(3)=="1" and m.group(2).startswith("sf") else None)))
    cases=[]; c=None
    for kind,n,tag in ev:
        if c is None or c['n']!=n: c=dict(n=n,rungs=[],fb=[]); cases.append(c)
        (c['rungs'] if kind=="end" else c['fb']).append(tag)
    return cases
for p in sys.argv[1:]:
    cs=load(p); print("==", p)
    for c in cs:
        if c['n'] in (56,81,88,89,96,97,100,104):
            r=[x for x in c['rungs'] if x is not None]
            print(f"  n={c['n']:3d} seats={len(c['rungs'])} closes={sorted(r)} tight<=0.05={sum(1 for x in r if x<=0.05)} fb={sorted(x for x in c['fb'] if x is not None)} ladfail={sum(1 for x in c['rungs'] if x is None)}")
