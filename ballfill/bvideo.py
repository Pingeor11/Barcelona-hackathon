"""Ball detection in the broadcast video, mapped onto the tactical timeline.

The broadcast camera follows play, zoomed in (the ball is ~12-15 px against 6-8 px in the tactical
video), and its homography file gives a per-frame pitch->image mapping in the same convention as the
tactical one. Where the broadcast tracking lost the ball but the camera still shows the live play,
the ball is often in shot: on BAR-ATH ~577 s of the frames we fill are live broadcast frames with a
valid homography.

Same detector as ballfill.video (3-frame heatmap U-Net, two rounds, half-out-of-sample):
- positives: tactical frames where both tactical trackers agree, the broadcast frame is live (its
  players sit on the tactical players), and the broadcast's own tracked ball is within 1 m of the
  agreed ball (a ground ball, so the homography puts it on the ball). Label = the broadcast ball
  projected through the broadcast homography, on that broadcast frame.
- negatives: random crops and crops at the feet of players away from the ball.
- detection only on the broadcast frames we can use: live, valid homography, graded halves, and
  either no broadcast ball (where it would add something) or a held-out sample for checking.

Detections are stored per tactical frame as pitch metres: cache/ballbvid_det.npz
(frames, xy (n, TOP_K, 2), score (n, TOP_K), bframe).

    python -m ballfill.bvideo extract | train | detect | check | all
"""

import os
import queue
import sys
import threading
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from ballfill import video as V  # noqa: E402
from games import GAME  # noqa: E402

V.PREFIX = "ballbvid_"
AGREE_BC_M = 1.0          # broadcast ball must sit this close to the agreed ball for a label
MAX_REPROJ = 1.0          # broadcast homography reprojection error cap
EDGE_PX = 12
N_CHECK = 4000            # held-out agree frames per half, detected for checking


def bvideo_path(game=GAME):
    data_dir = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
    for p in (os.path.join(data_dir, "broadcast", "{}_broadcast.mp4".format(game)),
              os.path.join(data_dir, "{}_broadcast.mp4".format(game))):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("broadcast video not found under " + data_dir)


def homographies(game=GAME):
    data_dir = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
    hb = pd.read_csv(os.path.join(data_dir, "broadcast", "{}_broadcast_homography.csv".format(game)))
    H = hb[["h{}{}".format(i, j) for i in range(3) for j in range(3)]].values.reshape(-1, 3, 3)
    ok = hb.valid.values.astype(bool) & (hb.reprojection_error.values <= MAX_REPROJ) \
        & np.isfinite(H).all(axis=(1, 2))
    return H, ok


def broadcast_ball():
    """(n_broadcast, 2) broadcast tracked ball on its own frames, metres."""
    from ballfill.data import load_broadcast
    from possession.load_data import load_match
    m = load_match()
    table, team = load_broadcast()
    n = len(team)
    return table[:, 3 * n:3 * n + 2] * np.array(m.pitch, np.float32)


def usable(d):
    """Per tactical frame: broadcast frame index (or -1) where the broadcast frame is live and has
    a valid homography, with f-1 and f+1 inside the video."""
    H, okH = homographies()
    t = np.arange(d.n)
    b = np.where(np.isfinite(d.offset), np.rint(t - np.nan_to_num(d.offset)), -1).astype(int)
    ok = (b >= 1) & (b < len(okH) - 1) & d.bc_live & np.isin(d.match.half, (1, 2))
    ok[ok] &= okH[b[ok]]
    return np.where(ok, b, -1), H


def _in_image(uv):
    return (uv[:, 0] > EDGE_PX) & (uv[:, 0] < V.IMG_W - EDGE_PX) & \
        (uv[:, 1] > EDGE_PX) & (uv[:, 1] < V.IMG_H - EDGE_PX)


def choose(d, half, seed=0):
    """(pos_bframes, pos_uv, neg_bframes, neg_c) on broadcast frames, from tactical `half`."""
    from ballfill.fuse import fuse
    m = d.match
    ball, src, sig, _ = fuse(d)
    bmap, H = usable(d)
    bball = broadcast_ball()
    rng = np.random.default_rng(seed + half)
    a, b = V.half_frames(m, half)
    t = np.arange(a, b)
    t = t[(bmap[t] >= 0) & (src[t] == 1) & (sig[t] <= 0.45)]
    bf = bmap[t]
    bb = bball[bf].astype(np.float64)
    near = np.isfinite(bb[:, 0]) & (np.linalg.norm(bb - ball[t], axis=1) <= AGREE_BC_M)
    t, bf, bb = t[near], bf[near], bb[near]
    uv = V.to_pixel(H[bf], bb)
    k = _in_image(uv)
    t, bf, uv = t[k], bf[k], uv[k]
    print("half {}: {} labelled broadcast frames".format(half, len(t)))
    pick = np.sort(rng.choice(len(t), min(V.N_POS, len(t)), replace=False))
    pos_f, pos_uv = bf[pick], uv[pick]
    neg_f, neg_c = [], []
    for k in rng.choice(len(t), V.N_NEG, replace=len(t) < V.N_NEG):
        fr, tf, buv = bf[k], t[k], uv[k]
        for _ in range(20):
            if rng.random() < 0.5:
                p = m.players_xy[tf]
                far = np.isfinite(p[:, 0]) & (np.linalg.norm(p - ball[tf], axis=1) > 4)
                if not far.any():
                    continue
                c = V.to_pixel(H[fr], p[rng.choice(np.flatnonzero(far))]) + rng.normal(0, 25, 2)
            else:
                c = np.array([rng.uniform(0, V.IMG_W), rng.uniform(0, V.IMG_H)])
            if not np.isfinite(c).all():
                continue
            c = np.clip(c, V.CROP / 2, [V.IMG_W - V.CROP / 2, V.IMG_H - V.CROP / 2])
            if np.all(np.abs(buv - c) > V.CROP / 2 + 8):
                neg_f.append(fr)
                neg_c.append(c)
                break
    return pos_f, pos_uv, np.array(neg_f), np.array(neg_c)


def detect_frames(d):
    """Tactical frames to detect on, with their broadcast frames."""
    from ballfill.fuse import fuse
    _, src, _, _ = fuse(d)
    bmap, _ = usable(d)
    rng = np.random.default_rng(5)
    use = bmap >= 0
    need = use & np.isnan(d.bc[:, 0])                     # broadcast tracking has no ball
    check = []
    for h in (1, 2):
        c = np.flatnonzero(use & (src == 1) & (d.match.half == h))
        check.append(rng.choice(c, min(N_CHECK, len(c)), replace=False))
    t = np.union1d(np.flatnonzero(need), np.concatenate(check))
    return t, bmap[t]


def detect(d, bs=4):
    """Full-frame peaks on the chosen broadcast frames, each with the other half's network."""
    import cv2
    import torch
    t_all, b_all = detect_frames(d)
    _, H = usable(d)
    nets = {h: V.load_net(h, 2) for h in (1, 2)}
    half = d.match.half[t_all]
    order = np.argsort(b_all, kind="stable")
    t_all, b_all, half = t_all[order], b_all[order], half[order]
    need = set()
    for bf in b_all:
        need.update((bf - 1, bf, bf + 1))
    pk = np.full((len(t_all), V.TOP_K, 3), np.nan, dtype=np.float32)
    cap = cv2.VideoCapture(bvideo_path())
    first, last = int(b_all[0]) - 1, int(b_all[-1]) + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    ring, pend, j, t0 = {}, {1: [], 2: []}, 0, time.time()

    def flush(h):
        idx = [i for i, _ in pend[h]]
        x = torch.stack([x for _, x in pend[h]])
        with torch.no_grad():
            heat = torch.sigmoid(nets[3 - h](x).float())[:, 0]
        pk[idx] = V.peaks(heat).cpu().numpy()
        pend[h].clear()

    for fr in range(first, last + 1):
        if not cap.grab():
            break
        if fr in need:
            img = cap.retrieve()[1]
            tt = torch.from_numpy(img).to("mps").permute(2, 0, 1).half() / 255 - 0.5
            ring[fr] = torch.nn.functional.pad(tt, (0, 0, 0, V.PAD_H - V.IMG_H))
        while j < len(b_all) and b_all[j] + 1 <= fr:
            c = int(b_all[j])
            if all(c + k in ring for k in (-1, 0, 1)):
                h = int(half[j])
                pend[h].append((j, torch.cat([ring[c - 1], ring[c], ring[c + 1]])))
                if len(pend[h]) == bs:
                    flush(h)
            j += 1
        for k in [k for k in ring if k < fr - 2]:
            del ring[k]
        if (fr - first) % 10000 == 0:
            print("  broadcast frame {}/{}: {} of {} detected ({:.0f} s)".format(
                fr, last, j, len(b_all), time.time() - t0), flush=True)
    for h in (1, 2):
        if pend[h]:
            flush(h)
    xy = V.to_pitch(H[b_all][:, None], pk[:, :, :2].astype(np.float64))
    o = np.argsort(t_all)
    np.savez(V.cache("det.npz"), frames=t_all[o], bframe=b_all[o], xy=xy[o].astype(np.float32),
             score=pk[o, :, 2], uv=pk[o, :, :2])
    print("detected {} broadcast frames in {:.0f} s".format(len(t_all), time.time() - t0))


def load(n):
    """(n, TOP_K, 2) pitch xy and (n, TOP_K) score per tactical frame (NaN where not detected)."""
    z = np.load(V.cache("det.npz"))
    xy = np.full((n, V.TOP_K, 2), np.nan, dtype=np.float32)
    s = np.full((n, V.TOP_K), np.nan, dtype=np.float32)
    xy[z["frames"]], s[z["frames"]] = z["xy"], z["score"]
    return xy, s


def check(d):
    """Top-peak accuracy by score bin: held-out agree frames (truth = agreed ball) and tactical-only
    frames where the broadcast tracker lost the ball (truth = the single tactical tracker)."""
    from ballfill.fuse import fuse
    ball, src, sig, _ = fuse(d)
    xy, s = load(d.n)
    has = np.isfinite(s[:, 0])
    sets = {"held-out agree frames": has & (src == 1),
            "tactical-only, broadcast lost the ball": has & (src == 4)}
    bins = (0.02, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01)
    for name, k in sets.items():
        f = np.flatnonzero(k)
        e = np.linalg.norm(xy[f] - ball[f][:, None], axis=2)
        e = np.where(np.isfinite(e), e, 1e3)
        print("\n{}: {} frames; best-of-5 within 2 m {:.1%}".format(name, len(f),
                                                                    (e.min(1) <= 2).mean()))
        print("   {:>11} {:>7} {:>7} {:>7} {:>7} {:>9}".format("top score", "frames", "<=1m",
                                                               "<=2m", "<=5m", "median"))
        for lo, hi in zip(bins[:-1], bins[1:]):
            q = (s[f, 0] >= lo) & (s[f, 0] < hi)
            if q.sum():
                e0 = e[q, 0]
                print("   {:.2f}-{:.2f} {:>7} {:>6.1%} {:>6.1%} {:>6.1%} {:>8.2f}m".format(
                    lo, min(hi, 1), q.sum(), (e0 <= 1).mean(), (e0 <= 2).mean(),
                    (e0 <= 5).mean(), np.median(e0)))
    filled = has & (src == 0)
    print("\nframes with no source at all that the broadcast detector looked at: {} ({:.0f} s); "
          "top score >= 0.5 on {:.0f} s".format(filled.sum(), filled.sum() / 25,
                                                (filled & (s[:, 0] >= 0.5)).sum() / 25))


def main(argv):
    from ballfill.data import load_ball_data
    step = argv[1] if len(argv) > 1 else "all"
    d = load_ball_data()
    if step in ("extract", "all"):
        for h in (1, 2):
            V.extract(d, h, chosen=choose(d, h), path=bvideo_path())
    if step in ("train", "all"):
        for h in (1, 2):
            V.train(h)
    if step in ("detect", "all"):
        detect(d)
    if step in ("check", "all"):
        check(d)


if __name__ == "__main__":
    main(sys.argv)
