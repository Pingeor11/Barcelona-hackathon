"""Two-fold, by-half parameter selection for fill methods: choose on one half, score the other.

The scored half never influences its own parameters, so the reported numbers are out-of-sample.

    python -m ballfill.tune
"""

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.evaluate import context_ball, real_gaps, synthetic_gaps  # noqa: E402

CAP_M = 30.0   # selection loss: mean of per-frame error capped here, so one wild gap cannot decide


def errors(d, gaps, which, method):
    out = []
    for g in gaps:
        est = method(d, context_ball(d, g, which), g)
        out.append(np.linalg.norm(est[g.truth_frames - g.a] - g.truth, axis=1))
    return out


def loss(errs):
    e = np.concatenate(errs)
    return np.minimum(e, CAP_M).mean()


def two_fold(d, gaps, which, make, grid):
    """make(**params) -> method. Returns (per-gap errors with out-of-fold params, chosen params)."""
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    by_half = {h: [g for g in gaps if g.half == h] for h in (1, 2)}
    table = {}
    for c in combos:
        method = make(**c)
        if hasattr(method, "prepare"):
            method.prepare(d, which, gaps)
        table[tuple(c.items())] = {h: errors(d, by_half[h], which, method) for h in (1, 2)}
    out, chosen = [], {}
    for h in (1, 2):
        other = 3 - h
        best = min(table, key=lambda k: loss(table[k][other]))
        chosen[h] = dict(best)
        out += table[best][h]
    return out, chosen, table


def summary(errs):
    e = np.concatenate(errs)
    return "median {:.2f} m  mean {:.2f} m  <=2m {:.0%}  <=5m {:.0%}".format(
        np.median(e), e.mean(), (e <= 2).mean(), (e <= 5).mean())


if __name__ == "__main__":
    from ballfill.carrier import CarrierHMM
    from ballfill.data import load_ball_data
    d = load_ball_data()
    shared = CarrierHMM()
    grid = dict(switch=[0.002, 0.005, 0.01, 0.02, 0.05], beta=[0.0, 0.5, 1.0],
                static_e=[0.0, 0.03, 0.1, 0.3], pass_scale=[10.0, 30.0])

    def make(**hmm):
        shared.hmm = hmm
        return shared

    for which, gaps in (("A", real_gaps(d)), ("B", synthetic_gaps(d))):
        errs, chosen, table = two_fold(d, gaps, which, make, grid)
        print("\nset {}: carrier_hmm, out-of-fold  {}".format(which, summary(errs)))
        for h in (1, 2):
            print("   half {} scored with params chosen on half {}: {}".format(h, 3 - h, chosen[h]))
        ranked = sorted(table, key=lambda k: loss(table[k][1] + table[k][2]))
        print("   top 5 on the whole set (in-sample, for reading only):")
        for k in ranked[:5]:
            print("     {:<70} {}".format(str(dict(k)), summary(table[k][1] + table[k][2])))
