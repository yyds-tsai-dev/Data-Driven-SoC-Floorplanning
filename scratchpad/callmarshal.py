"""Isolate njit call-marshalling cost with many mixed-dtype array args, and
whether a large cold branch poisons the hot path when force-inlined."""

import time

import numpy as np
from numba import njit

U = 82
B = (np.ones((U, 4)), np.zeros((U, 11), dtype=np.int64),
     np.zeros((86, 5), dtype=np.int64), np.ones((100, 3)),
     np.zeros((100, 2), dtype=np.int64), np.ones((98, 3)),
     np.zeros((98, 3), dtype=np.int64), np.full((U, 2), -1.0),
     np.zeros(98, dtype=np.int64), np.zeros(688, dtype=np.int64),
     np.zeros(688), np.zeros((86, 2), dtype=np.int64), np.zeros(86),
     np.zeros((8, 4)), np.zeros(98, dtype=np.int64))

ARGS = ("u, ww, uf, ui, bandi, chunkf, chunki, entf, enti, hcf, plan_ent, "
        "plan_ck, plan_ckw, plani, planf, eckf, tmp")

BIG = ("    s = 0.0\n"
       "    for b in range(bandi.shape[0]):\n"
       "        for c in range(2, 40):\n"
       "            s += chunkf[c, 0] * entf[c, 1] + plan_ckw[c]\n"
       "        plani[b, 0] = int(s)\n"
       "        planf[b] = s\n"
       "    hcf[u, 1] = s\n"
       "    return s\n")


def build(inline, big):
    body = BIG if big else "    return 0.0\n"
    src = ("def leaf(" + ARGS + "):\n"
           "    if ui[u, 0] == 0:\n"
           "        return uf[u, 1] + uf[u, 0] / ww\n" + body)
    ns = {"range": range, "int": int}
    exec(src, ns)
    kw = {"cache": False}
    if inline:
        kw["inline"] = "always"
    leaf = njit(**kw)(ns["leaf"])
    src2 = ("def drive(m, U, ww, uf, ui, bandi, chunkf, chunki, entf, enti, "
            "hcf, plan_ent, plan_ck, plan_ckw, plani, planf, eckf, tmp):\n"
            "    y = 0.0\n"
            "    for _ in range(m):\n"
            "        for u in range(U):\n"
            "            y += leaf(" + ARGS + ")\n"
            "    return y\n")
    ns2 = {"leaf": leaf, "range": range}
    exec(src2, ns2)
    return njit(cache=False)(ns2["drive"])


for inline in (False, True):
    for big in (False, True):
        f = build(inline, big)
        f(1, U, 30.0, *B)
        t0 = time.perf_counter()
        for _ in range(200):
            f(20, U, 30.0, *B)
        dt = (time.perf_counter() - t0) / 200
        print("inline=" + str(inline).ljust(5) + " cold_big_branch="
              + str(big).ljust(5) + " -> "
              + str(round(dt / (20 * U) * 1e9, 2)) + " ns/unit")
