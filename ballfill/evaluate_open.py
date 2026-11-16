"""Test set C: fill methods on the 7 degraded open matches, live-ball frames in gaps.

Each method is prepared per match (its classifier trains on the other half of that match, as on
ours), so C is out-of-sample the same way A and B are.

    python -m ballfill.evaluate_open
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.evaluate import print_table, run_method, score  # noqa: E402
from ballfill.opendata import IDSSE_IDS, load_open_ball, open_gaps  # noqa: E402


def evaluate_open(methods, ids=IDSSE_IDS, seed=0):
    per_method = {name: ([], []) for name in methods}     # errors, gaps
    for mid in ids:
        ob = load_open_ball(mid, seed=seed)
        gaps = open_gaps(ob)
        for name, method in methods.items():
            if hasattr(method, "set_source"):
                method.set_source(mid)
            if hasattr(method, "prepare"):
                method.prepare(ob.d, "C", gaps)
            if hasattr(method, "set_open"):
                method.set_open(ob)
            errs, gl = per_method[name]
            errs += run_method(ob.d, gaps, "C", method)
            gl += gaps
    results = [(name, score(e, g)) for name, (e, g) in per_method.items()]
    n_gaps = len(next(iter(per_method.values()))[1])
    print_table("Test set C -- {} open matches, live frames in gaps: {} gaps".format(
        len(ids), n_gaps), results)
    return per_method


if __name__ == "__main__":
    from ballfill import fill
    evaluate_open(fill.METHODS)
