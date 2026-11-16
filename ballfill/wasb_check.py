"""Head-to-head: the pretrained WASB football detector (BMVC 2023) against our detector.

WASB (github.com/nttcom/WASB-SBDT, MIT) is an HRNet taking 3 consecutive RGB frames, ImageNet
normalised, at 512x288, trained on a fixed-camera football dataset. Its code and weights live
outside this repository (point WASB_SRC at its src/ and WASB_WEIGHTS at wasb_soccer_best.pth.tar).

Two ways to run it on our 1920x1080 frames:
- resized: the whole frame at 512x288, as the authors do (our 6-8 px ball becomes ~2 px);
- tiles: native-resolution 512x288 tiles covering the frame (16 tiles, overlapping).
Scored on the same frames as our detector: test set A frames (truth = live broadcast) and held-out
observed frames (truth = agreed ball). Metric: top peak within 1 / 2 / 5 m of the truth.

    WASB_SRC=... WASB_WEIGHTS=... python -m ballfill.wasb_check
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ballfill.video import IMG_H, IMG_W, load_detections, to_pitch, to_pixel, video_path  # noqa

TW, TH = 512, 288
TILE_X = (0, 469, 938, 1408)
TILE_Y = (0, 264, 528, 792)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
N_SAMPLE = int(os.environ.get("WASB_N", 1500))


class _Attr(dict):
    """dict with attribute access, as WASB's Hydra configs provide."""
    __getattr__ = dict.__getitem__

    @classmethod
    def wrap(cls, x):
        if isinstance(x, dict):
            return cls({k: cls.wrap(v) for k, v in x.items()})
        if isinstance(x, list):
            return [cls.wrap(v) for v in x]
        return x


def load_wasb():
    import torch
    import yaml
    sys.path.insert(0, os.environ["WASB_SRC"])
    from models.hrnet import HRNet
    with open(os.path.join(os.environ["WASB_SRC"], "configs", "model", "wasb.yaml")) as f:
        cfg = yaml.safe_load(f)
    net = HRNet(_Attr.wrap(cfg))
    ck = torch.load(os.environ["WASB_WEIGHTS"], map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model_state_dict"])
    return net.to("mps").eval()


def prep(frames_bgr):
    """3 BGR uint8 images (same size) -> (9, h, w) float32, RGB, ImageNet normalised."""
    out = []
    for im in frames_bgr:
        rgb = im[..., ::-1].astype(np.float32) / 255.0
        out.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
    return np.concatenate(out).astype(np.float32)


def run(net, x):
    import torch
    with torch.no_grad():
        y = net(torch.from_numpy(x).to("mps"))[0]
        return torch.sigmoid(y[:, 1]).cpu().numpy()     # middle frame's heatmap


def top_resized(net, frames):
    import cv2
    small = [cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA) for f in frames]
    h = run(net, prep(small)[None])[0]
    r, c = np.unravel_index(h.argmax(), h.shape)
    return np.array([(c + 0.5) * IMG_W / TW, (r + 0.5) * IMG_H / TH]), h.max()


def top_tiles(net, frames):
    x = np.stack([prep([f[y0:y0 + TH, x0:x0 + TW] for f in frames])
                  for y0 in TILE_Y for x0 in TILE_X])
    h = run(net, x)
    k = h.reshape(len(h), -1).max(1).argmax()
    r, c = np.unravel_index(h[k].argmax(), h[k].shape)
    y0, x0 = TILE_Y[k // len(TILE_X)], TILE_X[k % len(TILE_X)]
    return np.array([x0 + c, y0 + r], dtype=np.float64), h[k].max()


def main():
    import cv2
    from ballfill.data import load_ball_data
    from ballfill.evaluate import real_gaps
    from ballfill.fuse import fuse
    d = load_ball_data()
    m = d.match
    H = m.homography.astype(np.float64)
    rng = np.random.default_rng(0)
    gaps = real_gaps(d)
    fa = np.concatenate([g.truth_frames for g in gaps])
    ta = np.concatenate([g.truth for g in gaps])
    ia = np.sort(rng.choice(len(fa), N_SAMPLE, replace=False))
    ball, src, sig, _ = fuse(d)
    obs = np.flatnonzero(np.isin(m.half, (1, 2)) & (src == 1))
    io = np.sort(rng.choice(obs, N_SAMPLE, replace=False))
    sets = {"A real gaps": (fa[ia], ta[ia]), "observed (held out)": (io, ball[io])}
    net = load_wasb()
    ours = load_detections(d.n)
    cap = cv2.VideoCapture(video_path())
    for name, (frames, truth) in sets.items():
        modes = [m for m in (("WASB resized", top_resized), ("WASB tiles", top_tiles))
                 if m[0].split()[1] in os.environ.get("WASB_MODES", "resized,tiles")]
        res = {k: [] for k, _ in modes}
        res["ours (round 3)"] = []
        t0 = time.time()
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
            ims = [cap.read()[1] for _ in range(3)]
            for key, fn in modes:
                uv, s = fn(net, ims)
                res[key].append(np.r_[to_pitch(H[f], uv), s])
            uv = ours[f, 0, :2].astype(np.float64)
            res["ours (round 3)"].append(np.r_[to_pitch(H[f], uv), ours[f, 0, 2]])
        print("\n{}: {} frames ({:.0f} s)".format(name, len(frames), time.time() - t0))
        print("  {:<16} {:>7} {:>7} {:>7} {:>9}".format("detector", "<=1m", "<=2m", "<=5m",
                                                       "median"))
        for key, rows in res.items():
            rows = np.array(rows)
            e = np.linalg.norm(rows[:, :2] - truth, axis=1)
            e = np.where(np.isfinite(e), e, 1e3)
            print("  {:<16} {:>6.1%} {:>6.1%} {:>6.1%} {:>8.2f}m".format(
                key, (e <= 1).mean(), (e <= 2).mean(), (e <= 5).mean(), np.median(e)))
        np.savez(os.path.join(os.path.dirname(HERE), "cache",
                              "ballvid_wasb_{}.npz".format(name.split()[0])),
                 frames=frames, truth=truth, **{k.replace(" ", "_"): np.array(v)
                                                for k, v in res.items()})


if __name__ == "__main__":
    main()
