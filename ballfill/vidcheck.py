"""How good are the raw video detections? Scored against truth the detector never saw.

- Test set A frames (real tactical gaps): truth = live broadcast ball.
- Observed "agree" frames: truth = the agreed ball (the detector for each half was trained on
  the other half, so these are held out too).
Reports, by top-peak score: how often the top peak is within 1 / 2 / 5 m of the truth.

    python -m ballfill.vidcheck
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.video import load_detections, to_pitch  # noqa: E402

SCORE_BINS = (0.02, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01)


def pitch_peaks(match, det):
    """(n, K, 3) pixel peaks -> (n, K, 2) metres, (n, K) score."""
    H = match.homography.astype(np.float64)
    xy = to_pitch(H[:, None], det[..., :2].astype(np.float64))
    return xy, det[..., 2]


def report(name, xy, score, truth):
    top = xy[:, 0]
    s = np.nan_to_num(score[:, 0], nan=0.0)
    e = np.linalg.norm(top - truth, axis=1)
    best = np.nanmin(np.linalg.norm(xy - truth[:, None], axis=2), axis=1)
    print("\n{}: {} frames; any peak {:.1%}; best-of-5 within 2 m {:.1%}".format(
        name, len(truth), np.isfinite(top[:, 0]).mean(), np.nanmean(best <= 2)))
    print("  {:>12} {:>8} {:>7} {:>7} {:>7} {:>8}".format(
        "top score", "frames", "<=1m", "<=2m", "<=5m", "median"))
    for lo, hi in zip(SCORE_BINS[:-1], SCORE_BINS[1:]):
        k = (s >= lo) & (s < hi)
        if k.sum():
            print("  {:>5.2f}-{:<6.2f} {:>8} {:>6.1%} {:>6.1%} {:>6.1%} {:>7.2f}m".format(
                lo, min(hi, 1), k.sum(), (e[k] <= 1).mean(), (e[k] <= 2).mean(),
                (e[k] <= 5).mean(), np.median(e[k])))
    return e, s


def main():
    from ballfill.data import load_ball_data
    from ballfill.evaluate import real_gaps
    from ballfill.fuse import fuse
    d = load_ball_data()
    det = load_detections(d.n)
    xy, score = pitch_peaks(d.match, det)
    gaps = real_gaps(d)
    f = np.concatenate([g.truth_frames for g in gaps])
    t = np.concatenate([g.truth for g in gaps])
    report("A: real gaps vs live broadcast", xy[f], score[f], t)
    ball, src, sig, _ = fuse(d)
    graded = np.isin(d.match.half, (1, 2))
    obs = np.flatnonzero(graded & (src == 1))
    report("observed agree frames (held-out half)", xy[obs], score[obs], ball[obs])


if __name__ == "__main__":
    main()
