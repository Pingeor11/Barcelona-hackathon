"""Plot the ball's x and y over a time window: what we know, the gaps, and our fill.

Production output (cache/BAR-ATH_ball.csv, from ballfill.run), coloured by source:
- black: tracker ball (agree / single tracker / flight / disagree)
- green: live-broadcast ball (used where the tracker has none)
- blue: the ball seen in the tactical video inside a gap (verified video sighting)
- red: filled by the HMM between sightings (no source at all: nothing to check against)
- shaded: gaps with no tracking source (video + filled)

Test fill (test set A protocol), to see how good the fill is:
- dashed orange: our fill when the broadcast ball is hidden too (tracker context only), drawn
  over the green stretches it is scored against
- grey dots: the video detector's raw top peak where confident (score >= 0.7), for reference

    python -m ballfill.plot_window [start_min] [minutes]      # default: first 5 minutes

Writes cache/plot_window_<start>_<end>min.png (derived from licensed data: stays in cache/).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FPS = 25
VIDEO_MIN_SCORE = 0.7


def _spans(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(e[::2], e[1::2]))


def main(argv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ballfill.data import load_ball_data
    from ballfill.evaluate import context_ball, real_gaps
    from ballfill.run import make_hmm
    from ballfill.video import load_detections, to_pitch

    start_min = float(argv[1]) if len(argv) > 1 else 0.0
    minutes = float(argv[2]) if len(argv) > 2 else 5.0
    d = load_ball_data()
    m = d.match
    df = pd.read_csv(os.path.join(os.path.dirname(HERE), "cache", "BAR-ATH_ball.csv"))
    kick = int(df.frame.iloc[0])                       # first frame of the first half
    a = kick + int(start_min * 60 * FPS)
    b = a + int(minutes * 60 * FPS)
    w = df[(df.frame >= a) & (df.frame < b)].reset_index(drop=True)
    mins = lambda f: (np.asarray(f) - kick) / FPS / 60   # noqa: E731
    t = mins(w.frame.values)
    xy = w[["x", "y"]].values
    vid = (w.source == "video").values
    filled = (w.source == "filled").values | vid
    bc = (w.source == "bc").values

    # test fill: the A protocol on the tracker gaps in this window
    from ballfill.run import video_filler
    hmm = make_hmm()
    hmm.prepare(d, "A", None)
    vfill = video_filler(d, hmm)
    gaps = [g for g in real_gaps(d) if g.a >= a and g.b <= b]
    test = []
    for g in gaps:
        est = vfill(d, context_ball(d, g, "A"), g)
        e = np.linalg.norm(est[g.truth_frames - g.a] - g.truth, axis=1)
        test.append((g, est, e))

    det = load_detections(d.n)
    H = m.homography.astype(np.float64)
    vf = w.frame.values[filled | bc]
    vf = vf[det[vf, 0, 2] >= VIDEO_MIN_SCORE]
    vxy = np.array([to_pitch(H[f], det[f, 0, :2].astype(np.float64)) for f in vf]).reshape(-1, 2)

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    for k, (ax, name) in enumerate(zip(axes, ("x (m, along the pitch)", "y (m, across)"))):
        for s, e in _spans(filled):
            ax.axvspan(t[s], t[e - 1], color="mistyrose", lw=0)
        ax.plot(t, np.where(filled | bc, np.nan, xy[:, k]), color="k", lw=1.0,
                label="tracker ball")
        ax.plot(t, np.where(bc, xy[:, k], np.nan), color="tab:green", lw=1.6,
                label="live-broadcast ball (tracker gap)")
        ax.plot(t, np.where(filled & ~vid, xy[:, k], np.nan), color="tab:red", lw=1.4,
                label="HMM fill between sightings")
        ax.plot(t, np.where(vid, xy[:, k], np.nan), color="tab:blue", lw=2.0,
                label="ball seen in the video")
        for i, (g, est, _) in enumerate(test):
            ax.plot(mins(np.arange(g.a, g.b)), est[:, k], color="tab:orange", lw=1.2, ls="--",
                    label="our fill with broadcast hidden (test)" if i == 0 else None)
        if len(vf):
            ax.scatter(mins(vf), vxy[:, k], s=3, color="0.55", zorder=1,
                       label="raw detector peak, score >= {}".format(VIDEO_MIN_SCORE))
        ax.set_ylabel(name)
        ax.set_ylim(-3, m.pitch[k] + 3)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=9, ncol=5,
                   markerscale=3, frameon=False)
    axes[1].set_xlabel("minutes since kick-off")
    fig.suptitle("BAR-ATH ball, minutes {:g}-{:g}:  tracker gaps {:.0f} s  (broadcast tracking "
                 "{:.0f} s; shaded: video {:.0f} s + HMM {:.0f} s)".format(
                     start_min, start_min + minutes, (filled | bc).sum() / FPS, bc.sum() / FPS,
                     vid.sum() / FPS, (filled & ~vid).sum() / FPS), y=0.995)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), "cache",
                       "plot_window_{:g}_{:g}min.png".format(start_min, start_min + minutes))
    fig.savefig(out, dpi=100)
    print("wrote", out)
    print("source counts in window:", w.source.value_counts().to_dict())
    if test:
        e = np.concatenate([x[2] for x in test])
        print("test fill (broadcast hidden) vs broadcast: {} gaps, {} frames, median {:.2f} m, "
              "mean {:.2f} m".format(len(test), len(e), np.median(e), e.mean()))
        for g, _, e in test:
            print("  {:5.2f}-{:5.2f} min  gap {:5.1f} s  median {:5.2f} m  max {:5.2f} m".format(
                mins(g.a), mins(g.b), (g.b - g.a) / FPS, np.median(e), e.max()))


if __name__ == "__main__":
    main(sys.argv)
