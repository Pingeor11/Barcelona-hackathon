"""Load every provided asset for the match into plain numpy/pandas, indexed by video frame.

    export MTAH_DATA_DIR=/path/to/data        # folder holding tactical/ and broadcast/
    python possession/load_data.py            # parse (once, ~10 s), cache, print a summary

    from possession.load_data import load_match
    m = load_match()
    m.ball_xy[i]          # (2,) ball position in metres on video frame i, NaN if not seen
    m.players_xy[i]       # (46, 2) every track slot in metres, NaN where the slot is empty
    m.track_team          # (46,) 0 = home (Barcelona), 1 = away (Athletic)

Every per-frame array has length 199,319 and row i is video frame i, i.e. row i of a submission.
The ATD's frame_id - 1 offset is applied here, once, so nothing downstream has to remember it.

Coordinates are in metres, origin at a pitch corner, x along the length (0..105.3) and y across
it (0..68.0). The raw ATD is normalised 0..1; the pitch size comes from games.py.

The parsed tracking is cached under cache/ (gitignored). It is derived from the licensed match
data, so it must never be committed or shared.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import load_homography  # noqa: E402
import load_smart_tagging  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
HOME, AWAY = 0, 1


@dataclass
class Match:
    ball_xy: np.ndarray        # (N, 2) metres, NaN when the ball is not tracked
    ball_speed: np.ndarray     # (N,) m/s as delivered (mostly NaN, recompute from positions)
    players_xy: np.ndarray     # (N, 46, 2) metres, NaN where a slot is empty
    players_speed: np.ndarray  # (N, 46) m/s
    track_team: np.ndarray     # (46,) HOME or AWAY per track slot
    half: np.ndarray           # (N,) 1 or 2 on graded frames, 0 elsewhere
    graded: np.ndarray         # (N,) bool, the frames the scorer looks at
    tags: pd.DataFrame         # Smart Tagging, one row per instance, frames half-open
    homography: np.ndarray     # (N, 3, 3) normalised pitch -> pixels
    homography_valid: np.ndarray  # (N,) bool
    gt_public: np.ndarray      # (N,) 'H'/'A'/'D' on the public 5 minutes, 'X' elsewhere
    pitch: tuple               # (length, width) metres

    @property
    def n_frames(self):
        return len(self.ball_xy)


def _track_team(meta_path):
    """HOME/AWAY per track slot. The first <Team> in the EPTS metadata is the home side."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_atd(raw_path, n_slots, n_frames):
    """Raw EPTS text -> (frame-indexed) float32 table of shape (n_frames, 1 + 3*n_slots + 3).

    Each line is "frame_id:x,y,s;x,y,s;...:bx,by,bs". Turning both separators into commas makes
    it a plain CSV that pandas parses in C.
    """
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32,
                        na_values=["NaN"]).values
    n_cols = 1 + 3 * n_slots + 3
    if table.shape[1] != n_cols:
        raise ValueError("expected {} columns, got {}".format(n_cols, table.shape[1]))

    out = np.full((n_frames, n_cols - 1), np.nan, dtype=np.float32)
    video_frame = table[:, 0].astype(np.int64) - 1        # ATD counts from 1, video from 0
    out[video_frame] = table[:, 1:]
    return out


def _load_tracking(game, data_dir, refresh):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    track_team = _track_team(meta_path)
    n_slots, n_frames = len(track_team), source_frames("tactical", game)
    cache = os.path.join(CACHE_DIR, "{}_tactical_atd.npz".format(game))

    if os.path.exists(cache) and not refresh:
        arrays = np.load(cache)
        return arrays["table"], track_team

    print("parsing {} (first run only) ...".format(os.path.basename(raw_path)))
    table = _parse_atd(raw_path, n_slots, n_frames)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, table=table)
    return table, track_team


def _tags_frame(game, data_dir):
    rows = []
    for event in load_smart_tagging.load(game, data_dir):
        row = {"code": event["code"].upper(), "start_frame": event["start_frame"],
               "end_frame": event["end_frame"]}
        row.update({key.lower(): value for key, value in event["tags"].items()})
        rows.append(row)
    tags = pd.DataFrame(rows).sort_values("start_frame").reset_index(drop=True)
    if "team" in tags:
        tags["team_side"] = tags["team"].map({"Barcelona": "H", "Atletic Club": "A"})  # N/A -> NaN
    return tags


def load_match(game=GAME, data_dir=None, refresh=False):
    """Everything for one match, aligned to video frames. Cached after the first call."""
    length, width = GAMES[game]["pitch"]
    table, track_team = _load_tracking(game, data_dir, refresh)
    n_frames, n_slots = len(table), len(track_team)

    players = table[:, :3 * n_slots].reshape(n_frames, n_slots, 3)
    players_xy = players[:, :, :2] * np.array([length, width], dtype=np.float32)
    ball_xy = table[:, 3 * n_slots:3 * n_slots + 2] * np.array([length, width], dtype=np.float32)

    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in half_video_blocks(game):
        half[first:last + 1] = number

    homography, _, homography_valid = load_homography.load(game, data_dir)
    gt_public = (pd.read_csv(PUBLIC_GT)["label"].astype(str).values if os.path.exists(PUBLIC_GT) else np.full(n_frames, "X", dtype=str))

    return Match(
        ball_xy=ball_xy, ball_speed=table[:, -1], players_xy=players_xy,
        players_speed=players[:, :, 2], track_team=track_team, half=half, graded=half > 0,
        tags=_tags_frame(game, data_dir), homography=homography,
        homography_valid=homography_valid, gt_public=gt_public, pitch=(length, width))


def summary(m):
    graded = m.graded
    ball_seen = ~np.isnan(m.ball_xy[:, 0])
    n_tracked = (~np.isnan(m.players_xy[:, :, 0])).sum(axis=1)
    print("frames            {:,}  (graded {:,})".format(m.n_frames, graded.sum()))
    print("pitch             {} x {} m".format(*m.pitch))
    print("track slots       {}  home {}  away {}".format(
        len(m.track_team), (m.track_team == HOME).sum(), (m.track_team == AWAY).sum()))
    print("ball seen         {:.1%} of all frames, {:.1%} of graded".format(
        ball_seen.mean(), ball_seen[graded].mean()))
    print("ball speed given  {:.1%} of graded frames".format(
        (~np.isnan(m.ball_speed[graded])).mean()))
    print("players tracked   mean {:.1f} per graded frame".format(n_tracked[graded].mean()))
    print("smart tagging     {} instances, {} codes".format(len(m.tags), m.tags["code"].nunique()))
    print("homography valid  {:.1%}".format(m.homography_valid.mean()))
    labels, counts = np.unique(m.gt_public[m.gt_public != "X"], return_counts=True)
    print("public GT         {}".format(dict(zip(labels, counts.tolist()))))


if __name__ == "__main__":
    summary(load_match(refresh="--refresh" in sys.argv))
