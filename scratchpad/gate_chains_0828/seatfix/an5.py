import re,sys,collections
lad = re.compile(r"\[lad\] p=(\d+) n=(\d+) d=(\d+) ridx=(\d+) e=([\d.]+) pins=(\d) ok=(\d) win=(-?[\d.]+) t=([\d.]+)")
end = re.compile(r"\[lad\] p=(\d+) n=(\d+) d=(\d+) END legal=(\d) reb=(\d) slice=([\d.]+) res=([\d.]+) lad=([\d.]+) left=(-?[\d.]+)")
sffb = re.compile(r"\[sf\] n=(\d+) FALLBACK tag=(\S+) ok=(\d) bbr=(-?[\d.]+) V=(-?\d+)")
cur=collections.defaultdict(list); ev=[]
for line in open(sys.argv[1],errors="replace"):
    for m in lad.finditer(line): cur[m.group(1)].append((float(m.group(5)), int(m.group(7))))
    for m in end.finditer(line):
        p=m.group(1); rr=cur.pop(p,[])
        legal=int(m.group(4))
        if legal and not rr: tag=0.0
        elif legal: tag=rr[-1][0]
        else: tag=None
        ev.append(("end", int(m.group(2)), tag))
    for m in sffb.finditer(line):
        ev.append(("fb", int(m.group(1)), (float(m.group(2)[2:]) if m.group(3)=="1" and m.group(2).startswith("sf") else None), float(m.group(4)), int(m.group(5))))
cases=[]; c=None
for e in ev:
    n=e[1]
    if c is None or c['n']!=n:
        c=dict(n=n, rungs=[], fbr=[], fbbbr=[], fbV=[]); cases.append(c)
    if e[0]=="end":
        if e[2] is not None: c['rungs'].append(e[2])
    else:
        if e[2] is not None: c['fbr'].append(e[2]); c['fbbbr'].append(e[3]); c['fbV'].append(e[4])
def bestrung(c):
    allr = c['rungs']+c['fbr']
    return min(allr) if allr else None
dist=collections.Counter(); byband=collections.defaultdict(collections.Counter)
for c in cases:
    b="a<76" if c['n']<76 else "b76-89" if c['n']<90 else "c90-104" if c['n']<105 else "d105-120"
    br=bestrung(c); dist[br]+=1; byband[b][br]+=1
    c['best']=br
print("best-rung-per-case distribution:", dict(sorted(dist.items(), key=lambda kv:(kv[0] is None, kv[0]))))
for b in sorted(byband): print(" ", b, dict(sorted(byband[b].items(), key=lambda kv:(kv[0] is None, kv[0]))))
print()
print("cases whose BEST is loose (>=0.12):")
for c in cases:
    if c['best'] is not None and c['best']>=0.12:
        print(f"  n={c['n']:3d} rungs={sorted(c['rungs'])} fb={sorted(c['fbr'])} nseats={len(c['rungs'])+len(c['fbr'])}")
# how many tight (<=0.05) closes per case
print()
tc=collections.Counter()
for c in cases:
    k=sum(1 for r in c['rungs'] if r<=0.05)
    tc[k]+=1
print("count of tight(<=0.05) closes per case:", dict(sorted(tc.items())))
