"""Ablation of the carrier HMM's components on test sets A, B and C.

One shared CarrierHMM (so classifier emissions are computed once per gap); each variant only
changes decoding / the dead-ball model / its weight. Variants are compared on all three sets;
anything chosen is chosen on C (independent matches) and then read off A and B.

    python -m ballfill.ablate
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.carrier import CarrierHMM  # noqa: E402
from ballfill.evaluate import real_gaps, synthetic_gaps  # noqa: E402
from ballfill.opendata import IDSSE_IDS, load_open_ball, open_gaps  # noqa: E402
from ballfill.tune import errors, summary  # noqa: E402

BASE = dict(switch=0.1, beta=0.1, static_e=0.03, pass_scale=5.0)
VARIANTS = {
    "v2 mean": dict(decode="mean", dead=False),
    "dead full e=0.3": dict(decode="mean", dead=True, dead_e=0.3),
    "dead no-exit e=0.3": dict(decode="mean", dead=True, dead_e=0.3, exit=False),
    "dead rest-only e=0.3": dict(decode="mean", dead=True, dead_e=0.3, players=False),
    "dead rest-only e=1": dict(decode="mean", dead=True, dead_e=1.0, players=False),
    "dead rest-only no-exit e=0.3": dict(decode="mean", dead=True, dead_e=0.3, players=False,
                                         exit=False),
    "dead rest-only no-exit e=1": dict(decode="mean", dead=True, dead_e=1.0, players=False,
                                       exit=False),
}


def apply(h, v):
    h.decode, h.dead, h.use_exit = v["decode"], v["dead"], v.get("exit", True)
    extra = {}
    if "dead_e" in v:
        extra = dict(dead_e=v["dead_e"], dead_players=v.get("players", True))
    h.hmm = dict(BASE, **extra)


def run_set(h, d, gaps, which, out):
    for name, v in VARIANTS.items():
        apply(h, v)
        out.setdefault(name, []).extend(errors(d, gaps, which, h))


def main():
    from ballfill.data import load_ball_data
    h = CarrierHMM(flight_s=2.0, dead=True)        # dead=True so prepare() loads the dead model
    results = {"A": {}, "B": {}, "C": {}}
    d = load_ball_data()
    h.set_source("BAR-ATH")
    h.prepare(d, "A", None)
    run_set(h, d, real_gaps(d), "A", results["A"])
    run_set(h, d, synthetic_gaps(d), "B", results["B"])
    for mid in IDSSE_IDS:
        ob = load_open_ball(mid)
        h.set_source(mid)
        h._ready = None
        h.dead = True
        h.prepare(ob.d, "C", None)
        run_set(h, ob.d, open_gaps(ob), "C", results["C"])
    for which in ("C", "A", "B"):
        print("\nset {}".format(which))
        for name, errs in results[which].items():
            print("  {:<26} {}".format(name, summary(errs)))


if __name__ == "__main__":
    main()
