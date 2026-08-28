import re, sys, collections, statistics as st
path = sys.argv[1]
lad = re.compile(r"\[lad\] p=(\d+) n=(\d+) d=(\d+) ridx=(\d+) e=([\d.]+) pins=(\d) ok=(\d) win=(-?[\d.]+) t=([\d.]+)")
end = re.compile(r"\[lad\] p=(\d+) n=(\d+) d=(\d+) END legal=(\d) reb=(\d) slice=([\d.]+) res=([\d.]+) lad=([\d.]+) left=(-?[\d.]+)")
sffail = re.compile(r"\[sf\] n=(\d+) LADDER-FAIL depth=(\d+) slice=([\d.]+) res=([\d.]+) ran=\[(.*?)\] ovl=(-?\d+) lad=([\d.]+) left=(-?[\d.]+)")
sfrung = re.compile(r"\[sf\] n=(\d+) rung expand=([\d.]+) ok=(\d) ovl=(-?\d+) win=(-?[\d.]+) t=([\d.]+)")
sffb = re.compile(r"\[sf\] n=(\d+) FALLBACK tag=(\S+) ok=(\d) bbr=(-?[\d.]+) V=(-?\d+) left=(-?[\d.]+)")

cur = collections.defaultdict(list)   # pid -> ridx events
calls = []
fb = []
for line in open(path, errors="replace"):
    for m in lad.finditer(line):
        p=m.group(1); cur[p].append((int(m.group(4)), float(m.group(5)), int(m.group(6)), int(m.group(7)), float(m.group(9))))
    for m in end.finditer(line):
        p=m.group(1)
        calls.append(dict(pid=p, n=int(m.group(2)), depth=int(m.group(3)), legal=int(m.group(4)),
                          slice=float(m.group(6)), res=float(m.group(7)), lad=float(m.group(8)),
                          left=float(m.group(9)), rungs=cur.pop(p, [])))
    for m in sffb.finditer(line):
        fb.append(dict(n=int(m.group(1)), tag=m.group(2), ok=int(m.group(3)), bbr=float(m.group(4)), V=int(m.group(5))))

print("calls", len(calls), "depth0", sum(1 for c in calls if c['depth']==0), "fbrec", len(fb))
def band(n):
    if n<76: return "a<76"
    if n<90: return "b76-89"
    if n<105: return "c90-104"
    return "d105-120"
agg = collections.defaultdict(lambda: collections.Counter())
tim = collections.defaultdict(list)
for c in calls:
    if c['depth']!=0: continue
    b=band(c['n']); a=agg[b]
    a['calls']+=1
    r0ok = (len(c['rungs'])==0 and c['legal']==1)
    # rung0 failed if any expand rung ran, OR legal==0 with no expand rung (died on time in rung0)
    a['r0_close']+= int(r0ok)
    if not r0ok:
        a['r0_fail']+=1
        t_exp = sum(x[4] for x in c['rungs'])
        t_r0 = max(0.0, c['lad'] - t_exp)
        tim[b].append((c['slice'], c['lad'], t_r0, t_exp, c['left'], c['legal'], len(c['rungs'])))
        if c['legal']==1:
            e = c['rungs'][-1][1]; a[f'close_e{e}']+=1
        else:
            a['ladfail']+=1
            if not c['rungs']: a['ladfail_noexp']+=1
    a['legal']+=c['legal']
for b in sorted(agg):
    a=agg[b]
    print(b, dict(a))
    if tim[b]:
        sl=[x[0] for x in tim[b]]; r0=[x[2] for x in tim[b]]; ex=[x[3] for x in tim[b]]; lf=[x[4] for x in tim[b]]
        print("   r0-fail calls: n=%d slice med=%.4f | t_r0 med=%.4f mean=%.4f (%.0f%% of slice) | t_expand med=%.4f | left@END med=%.4f"%(
            len(sl), st.median(sl), st.median(r0), sum(r0)/len(r0), 100*st.median(r0)/max(st.median(sl),1e-9), st.median(ex), st.median(lf)))
fbc = collections.Counter()
for f in fb:
    fbc[(band(f['n']), f['tag'], f['ok'])]+=1
for k in sorted(fbc): print("FB", k, fbc[k])
