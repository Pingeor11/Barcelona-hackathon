"""Unsupervised check against Metrica's automatic Smart Tagging (machine-made, not ground truth).
Agreement = share of frames inside team-tagged attacking phases where we name the same team."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging as st

ATTACK = {"build up", "progression", "attacking transition", "creating chances", "long ball",
          "ball in final third", "players in the box"}
DEFEND = {"defending in defensive third", "defending in middle third", "defending in attacking third"}

def tag_prior(n):
    """Per-frame expected possessor from tags: +1 home, -1 away, 0 unknown."""
    v = np.zeros(n)
    for ev in st.load():
        code = ev["code"].lower(); team = str(ev["tags"].get("Team", "")).lower()
        sgn = 1 if team.startswith("barc") else -1 if team.startswith("atl") else 0
        if not sgn: continue
        if code in DEFEND: sgn = -sgn
        elif code not in ATTACK: continue
        v[ev["start_frame"]:ev["end_frame"]] += sgn
    return np.sign(v)

def agreement(labels, prior):
    m = (prior != 0) & np.isin(labels, ["H", "A"])
    if not m.any():
        return float("nan"), 0
    pred = np.where(labels[m] == "H", 1, -1)
    return (pred == prior[m]).mean(), m.sum()

if __name__ == "__main__":
    prior = tag_prior(199319)
    for f in sys.argv[1:]:
        lab = pd.read_csv(f).label.values.astype(str)
        a, k = agreement(lab, prior)
        print(f"{os.path.basename(f):28s} tag agreement {a:.3f} over {k} frames")
