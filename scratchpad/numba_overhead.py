"""Measure numba dispatch overhead as a function of numpy-array arg count,
and the round(x, 4) semantics vs CPython."""

import time

import numpy as np
from numba import njit


def make(k):
    args = ", ".join(f"a{i}" for i in range(k))
    body = " + ".join(f"a{i}[0]" for i in range(k))
    src = f"def f({args}):\n    return {body}\n"
    ns = {}
    exec(src, ns)
    return njit(cache=False)(ns["f"])


for k in (1, 4, 8, 12, 16, 22, 30):
    f = make(k)
    arrs = [np.ones(4) for _ in range(k)]
    f(*arrs)
    reps = 200000
    t0 = time.perf_counter()
    for _ in range(reps):
        f(*arrs)
    dt = (time.perf_counter() - t0) / reps
    print(f"{k:3d} array args -> {dt*1e6:6.3f} us/call")


@njit(cache=False)
def r4(x):
    return round(x, 4)


r4(1.0)
rng = np.random.default_rng(0)
vals = rng.random(20000) * 1000.0
bad = sum(1 for v in vals if r4(v) != round(float(v), 4))
print("round(x,4) mismatches vs CPython:", bad, "/", len(vals))
