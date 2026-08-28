import re,sys,collections,statistics as st
rp=re.compile(r"\[rp0\] n=(\d+) slice=([\d.]+) span=([\d.]+) build=([\d.]+) r0=(\d) t=([\d.]+)")
rows=collections.defaultdict(list)
for line in open(sys.argv[1],errors="replace"):
    for m in rp.finditer(line):
        n=int(m.group(1)); sp=float(m.group(3)); t=float(m.group(6)); ok=int(m.group(5))
        b="a<76" if n<76 else "b76-89" if n<90 else "c90-104" if n<105 else "d105-120"
        rows[(b,ok)].append((sp,t,float(m.group(4))))
for k in sorted(rows):
    v=rows[k]; print(f"{k[0]:10s} r0={k[1]} N={len(v):4d} span_med={st.median(x[0] for x in v):.4f} t_med={st.median(x[1] for x in v):.4f} t/span_med={st.median(x[1]/max(x[0],1e-9) for x in v):.2f} t/span_p90={sorted(x[1]/max(x[0],1e-9) for x in v)[int(0.9*len(v))]:.2f} build={st.median(x[2] for x in v):.4f}")
