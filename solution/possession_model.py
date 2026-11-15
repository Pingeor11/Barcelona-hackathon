"""Ball possession from automatic tracking: a hidden Markov model over control and stoppage clues.

    python3 solution/possession_model.py <team_name>.csv

Every frame is labelled H (home in possession), A (away in possession) or D (dead ball); frames
outside the two halves are X. The method has two parts:

1. CLUES. Each frame gets a score for H, A and D from what the tracking shows:
   - who is closest to the ball, counted fully only when the ball is slow (under control)
   - Metrica's automatic Smart Tagging phases (e.g. "Build Up" for a team), as a weak hint
   - dead-ball signs: calm players, ball unseen, ball outside the pitch
2. DECODING. A three-state hidden Markov model picks the single best sequence of states for each
   half with the Viterbi algorithm. Changing possession costs 12 points and entering or leaving a
   dead ball costs 10, so a team only takes over when the clues for it clearly outweigh the cost:
   sustained control, not a single touch or deflection.

All numbers in PARAMS were set from football reasoning and whole-match sanity checks. None was
tuned on the public 5-minute ground truth, and no hand labels were used (RULES.md).
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import half_video_blocks, GAME
from features import features, rolling_mean

H, A, D = 0, 1, 2

PARAMS = dict(
    # --- who is closest to the ball ---
    dist_scale=2.0,         # metres of distance advantage worth one point
    near=4.0,               # metres: the clue fades when nobody is this close to the ball
    fast_ball=6.0,          # m/s: faster than this the ball is in flight (pass, shot) ...
    fast_weight=0.3,        # ... and the closeness clue counts only 30%
    tag_weight=0.03,        # per-frame nudge from Smart Tagging team phases
    # --- dead ball ---
    speed_dead=1.3,         # m/s: typical median player speed during a stoppage
    speed_scale=0.35,
    speed_weight=0.6,
    missing_dead=0.15,      # per-frame nudge while the ball is not seen
    out_weight=2.0,         # ball tracked outside the pitch
    # --- decoding ---
    switch_cost=12.0,       # cost of changing possession H <-> A
    dead_cost=10.0,         # cost of entering or leaving a dead ball
)


# ---------------------------------------------------------------- clues

def closeness_clue(F, p):
    """Positive = home closer to a controlled ball, negative = away."""
    dH, dA = F["dH"], F["dA"]
    vis = F["vis"] & np.isfinite(dH) & np.isfinite(dA)
    diff = np.clip((dA - dH) / p["dist_scale"], -3, 3)
    near = np.exp(-np.fmin(dH, dA) / p["near"])
    bs = np.nan_to_num(F["bspeed"], nan=0.0)
    control = np.where(bs > p["fast_ball"], p["fast_weight"], 1.0)
    s = np.where(vis, diff * near * control, 0.0)
    if p["tag_weight"]:
        from tag_check import tag_prior
        s = s + p["tag_weight"] * tag_prior(len(s))
    return s


def dead_ball_clue(F, p):
    ps = rolling_mean(F["pspeed"], 25)                       # players' median speed, 2 s window
    ps_f = np.where(np.isfinite(ps), ps, p["speed_dead"])
    d = np.clip((p["speed_dead"] - ps_f) / p["speed_scale"], -3, 3) * p["speed_weight"]
    d += np.where(F["vis"], 0.0, p["missing_dead"])
    d += np.where(F["out"], p["out_weight"], 0.0)
    return d


def frame_scores(F, p):
    """n x 3 array: how much each frame's clues favour H, A and D."""
    s = closeness_clue(F, p)
    e = np.zeros((len(s), 3))
    e[:, H] += s / 2
    e[:, A] -= s / 2
    e[:, D] += dead_ball_clue(F, p)
    return e


# ---------------------------------------------------------------- decoding

def viterbi(e, switch, dead):
    """Best state sequence given per-frame scores e (n x 3) and switching costs."""
    n = len(e)
    T = np.array([[0, -switch, -dead], [-switch, 0, -dead], [-dead, -dead, 0]], float)
    back = np.zeros((n, 3), np.int8)
    score = e[0] + np.array([-dead, -dead, 0.0])           # a half starts from a dead ball
    for t in range(1, n):
        cand = score[:, None] + T
        back[t] = cand.argmax(0)
        score = cand.max(0) + e[t]
    path = np.empty(n, np.int8); path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(F=None, **overrides):
    """Labels for every video frame."""
    p = dict(PARAMS, **overrides)
    F = F or features()
    e = frame_scores(F, p)
    labels = np.full(len(e), "X", dtype="<U1")
    for _, a, b in half_video_blocks(GAME):
        path = viterbi(e[a:b + 1], p["switch_cost"], p["dead_cost"])
        labels[a:b + 1] = np.array(["H", "A", "D"])[path]
    return labels


def write(path, labels):
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(labels))


def counts(labels):
    g = labels[labels != "X"]
    seg = g[np.r_[True, g[1:] != g[:-1]]]
    return dict(possessions=int(((seg == "H") | (seg == "A")).sum()), stoppages=int((seg == "D").sum()))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evaluation", "results", "possession.csv")
    labels = predict()
    print(counts(labels), {k: int((labels == k).sum()) for k in "HADX"})
    write(out, labels)
    print("->", out)
