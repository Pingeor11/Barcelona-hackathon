"""Per-frame signals derived from the tracking. No ground truth is used here."""
import numpy as np
from load import load

L, W = 105.3, 68.0

def rolling_mean(x, half):
    """Centred rolling mean ignoring NaN."""
    v = np.isfinite(x); xs = np.where(v, x, 0.0)
    c = np.r_[0, np.cumsum(xs)]; n = np.r_[0, np.cumsum(v)]
    i = np.arange(len(x)); lo = np.maximum(0, i-half); hi = np.minimum(len(x), i+half+1)
    cnt = n[hi]-n[lo]
    return np.where(cnt > 0, (c[hi]-c[lo])/np.maximum(cnt, 1), np.nan)

def features():
    P, B, team = load()
    bx, by = B[:, 0], B[:, 1]
    vis = np.isfinite(bx)
    dx = P[:, :, 0] - bx[:, None]; dy = P[:, :, 1] - by[:, None]
    d = np.sqrt(dx*dx + dy*dy)
    dH = np.nanmin(np.where(team[None, :] == 0, d, np.nan), axis=1)
    dA = np.nanmin(np.where(team[None, :] == 1, d, np.nan), axis=1)
    # mean player speed, computed from positions (m/s) so units are certain
    v = np.full(P.shape[:2], np.nan, np.float32)
    v[1:] = np.sqrt(((P[1:, :, :2]-P[:-1, :, :2])**2).sum(-1))*25
    v[v > 12] = np.nan  # identity jumps, not running
    with np.errstate(all="ignore"):
        pspeed = np.nanmedian(v, axis=1)
    nplayers = np.isfinite(P[:, :, 0]).sum(1)
    # ball speed from positions
    bs = np.full(len(bx), np.nan); bs[1:] = np.hypot(np.diff(bx), np.diff(by))*25
    out = (bx < -0.5) | (bx > L+0.5) | (by < -0.5) | (by > W+0.5)
    return dict(P=P, B=B, team=team, vis=vis, bx=bx, by=by, dH=dH, dA=dA,
                pspeed=pspeed, nplayers=nplayers, bspeed=bs, out=out)
