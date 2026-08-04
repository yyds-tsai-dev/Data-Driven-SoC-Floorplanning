"""Single-threaded cProfile of the partner SA hot loop on a synthetic case."""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "partner"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sa_kernel_synth import build_instance, make_optimizer  # noqa: E402


def run(n: int, seconds: float, seed: int, label: str):
    inst = build_instance(n=n, seed=seed)
    opt = make_optimizer(inst, seed=seed, deadline=time.time() + 1000.0)
    opt.prepare()
    cols = opt._cols
    cost, _ = opt._evaluate(cols)

    # count moves by wrapping _evaluate
    stats = {"evals": 0}
    orig_eval = opt._evaluate

    def counting_eval(c):
        stats["evals"] += 1
        return orig_eval(c)

    opt._evaluate = counting_eval

    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    opt._anneal(cols, t0 + seconds, cost)
    pr.disable()
    dt = time.time() - t0

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(22)
    print(f"\n===== {label}: n={n} seed={seed} wall={dt:.2f}s "
          f"evals={stats['evals']} moves/s={stats['evals']/dt:.0f} "
          f"colcache={'on' if opt._dc_enabled else 'off'} "
          f"(hits={opt._dc_hits} recompute={opt._dc_recompute}) =====")
    print(s.getvalue())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    run(n, secs, seed, os.environ.get("LABEL", "profile"))
