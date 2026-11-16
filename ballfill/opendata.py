"""The 7 open IDSSE (Bundesliga, DFL) matches, degraded to look like our tracking: test set C.

DFL's ball is complete and accurate while play is live. While it is dead DFL freezes the ball at
the spot play stopped (it moves on 0% of dead frames) -- a placeholder, not a measurement. So:
  * test set C scores LIVE frames inside gaps only;
  * dead/alive labels train a "play is stopped" model;
  * the frozen position is where play stopped (exit point / foul spot), usable as a label for that.

Gaps are imposed by possession.degrade (your teammate's): gap starts drawn from a hazard model
fitted to where OUR tracker loses the ball, gap lengths resampled from our gaps, players dropped
and the ball smoothed to match our tracker's statistics.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.data import BallData, runs  # noqa: E402
from ballfill.evaluate import Gap  # noqa: E402
from possession.load_data import Match  # noqa: E402

IDSSE_IDS = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")


@dataclass
class OpenBall:
    d: BallData            # degraded, in the same shape our match uses
    truth: np.ndarray      # (N, 2) clean DFL ball
    dead: np.ndarray       # (N,) bool, DFL ball status dead
    match_id: str


_PROFILE = None


def _profile():
    global _PROFILE
    if _PROFILE is None:
        from possession.degrade import noise_profile
        from possession.load_data import load_match
        _PROFILE = noise_profile(load_match())
    return _PROFILE


def load_open_ball(match_id, seed=0):
    from possession.degrade import degrade
    from possession.open_data import load_open_match
    o = load_open_match(match_id)
    dg = degrade(o, _profile(), seed=seed)
    n = o.n_frames
    match = Match(ball_xy=dg.ball_xy, ball_speed=np.full(n, np.nan, np.float32),
                  players_xy=dg.players_xy, players_speed=np.full(dg.players_xy.shape[:2], np.nan),
                  track_team=dg.track_team, half=o.half.astype(np.int8),
                  graded=np.ones(n, dtype=bool), tags=pd.DataFrame(columns=["start_frame"]),
                  homography=None, homography_valid=None, gt_public=None, pitch=o.pitch)
    nan2 = np.full((n, 2), np.nan, dtype=np.float32)
    d = BallData(match=match, tac=dg.ball_xy.astype(np.float32), bc=nan2, bc_raw=nan2,
                 bc_live=np.zeros(n, bool), bc_players=None, bc_team=None, offset=None)
    return OpenBall(d=d, truth=o.ball_xy.astype(np.float64), dead=o.labels == "D",
                    match_id=match_id)


def open_gaps(ob, live_only=True):
    """Test set C for one match: every gap with sightings on both sides; truth on live frames."""
    d, m = ob.d, ob.d.match
    seen = ~np.isnan(d.tac[:, 0])
    gaps = []
    for half in (1, 2):
        in_half = m.half == half
        for a, b in runs(~seen & in_half):
            if a == 0 or b >= d.n or not (in_half[a - 1] and in_half[b]):
                continue
            frames = np.arange(a, b)
            if live_only:
                frames = frames[~ob.dead[frames]]
            if len(frames):
                gaps.append(Gap(a, b, half, frames, ob.truth[frames]))
    return gaps


if __name__ == "__main__":
    for mid in IDSSE_IDS:
        ob = load_open_ball(mid)
        g = open_gaps(ob)
        seen = ~np.isnan(ob.d.tac[:, 0])
        print("{}: ball missing {:.1%}, {} gaps with live truth, {} live truth frames, "
              "dead share of gap frames {:.0%}".format(
                  mid, (~seen).mean(), len(g), sum(len(x.truth) for x in g),
                  ob.dead[~seen].mean()))
