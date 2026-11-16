"""A learned gap filler in the style of Kim et al. (KDD 2023, "ballradar"), reimplemented here.

Their repository has no licence, so nothing is copied; this follows the paper's design:
a set transformer over the players at each moment, a bidirectional LSTM over time, a "who has the
ball" head over players (plus a "loose ball" state), and the ball as the possession-weighted
position. Unlike the paper, the ball is also an *input* wherever it was observed, so the model
fills gaps between sightings rather than inferring the whole trajectory from players alone.

Training data: the 7 open matches (players degraded like ours, clean DFL ball on live frames) and
our match (tactical players, the fused ball where observed). Random gaps with real gap lengths are
hidden; the loss is on the hidden ball (Huber, metres) plus possession cross-entropy.

Folds (nothing is ever scored by a model that saw it):
- our match, half h: trained on the 7 open matches + our other half;
- open match k: trained on the other 6 + our match.

    python -m ballfill.imputer       # train the folds, score A, B, C
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession.load_data import CACHE_DIR  # noqa: E402

FPS = 25
STEP = 5                      # 5 Hz
L, W = 105.3, 68.0
WIN = 300                     # 60 s training windows
CONTEXT_STEPS = 150           # 30 s either side of a gap at inference
POSS_M = 2.0
DEV = os.environ.get("IMPUTER_DEVICE", "mps")
N_FEAT = 7                    # x, y, vx, vy, attacks +x, home, present
N_CTX = 4                     # straight-line path x, y; time since last / until next sighting


def edge_context(ball_m, ok):
    """(T, 2) metres + (T,) observed -> (T, N_CTX): the straight line between the sightings either
    side (normalised; held flat beyond the first/last), and seconds since / until a sighting / 30."""
    T = len(ok)
    t = np.arange(T)
    out = np.zeros((T, N_CTX), np.float32)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    nb = _norm_xy(ball_m[idx])
    out[:, 0] = np.interp(t, idx, nb[:, 0])
    out[:, 1] = np.interp(t, idx, nb[:, 1])
    prev = np.maximum.accumulate(np.where(ok, t, -10 ** 6))
    nxt = np.minimum.accumulate(np.where(ok, t, 10 ** 6)[::-1])[::-1]
    out[:, 2] = np.minimum((t - prev) * STEP / FPS / 30.0, 1.0)
    out[:, 3] = np.minimum((nxt - t) * STEP / FPS / 30.0, 1.0)
    return out


def _norm_xy(xy):
    return np.stack([xy[..., 0] / L * 2 - 1, xy[..., 1] / W * 2 - 1], -1)


def sequence(match, ball_in, truth, truth_ok):
    """Per half: dict of 5 Hz arrays. ball_in/truth (N, 2) metres; truth_ok (N,) bool."""
    from ballfill.carrier import _velocity, attack_sign
    P = match.players_xy.astype(np.float32)
    V = _velocity(P)
    sign = attack_sign(match)
    team = match.track_team
    out = {}
    for half in (1, 2):
        idx = np.flatnonzero(match.half == half)
        fr = idx[::STEP]
        p = P[fr]
        present = np.isfinite(p[..., 0])
        x = np.zeros(p.shape[:2] + (N_FEAT,), np.float32)
        x[..., :2] = np.nan_to_num(_norm_xy(p))
        x[..., 2:4] = np.clip(np.nan_to_num(V[fr] / 10.0), -1.5, 1.5)   # dropouts fake huge speeds
        tm = np.clip(team, 0, 1)
        x[..., 4] = sign[fr][:, tm]
        x[..., 5] = (team == 0)[None].astype(np.float32)
        x[..., 6] = present
        b = ball_in[fr].astype(np.float32)
        bin_ok = np.isfinite(b[:, 0])
        out[half] = dict(frames=fr, x=x, present=present, P=np.nan_to_num(p),
                         ball_in=np.nan_to_num(b), ball_ok=bin_ok,
                         truth=np.nan_to_num(truth[fr].astype(np.float32)),
                         truth_ok=truth_ok[fr] & np.isfinite(truth[fr, 0]))
    return out


def make_model(d=64, ctx=128):
    import torch
    import torch.nn as nn

    class Imputer(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Sequential(nn.Linear(N_FEAT + 1, d), nn.ReLU(), nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True)
            self.sab = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.ball_emb = nn.Sequential(nn.Linear(3 + N_CTX, 32), nn.ReLU())
            self.lstm = nn.LSTM(2 * d + 32, ctx, num_layers=2, batch_first=True,
                                bidirectional=True)
            self.poss = nn.Sequential(nn.Linear(d + 2 * ctx, 128), nn.ReLU(), nn.Linear(128, 1))
            self.free_logit = nn.Sequential(nn.Linear(2 * ctx, 64), nn.ReLU(), nn.Linear(64, 1))
            # start at the straight-line path: the loose-ball state holds ~95% of the mass at init
            self.free_logit[-1].bias.data.fill_(6.0)
            self.free_xy = nn.Sequential(nn.Linear(2 * ctx + 2 + N_CTX, 64), nn.ReLU(),
                                         nn.Linear(64, 2))

        def forward(self, x, present, ball_in, ball_ok, ectx):
            B, T, S, F = x.shape
            path = ectx[..., :2]                         # straight line between sightings
            dxy = (x[..., :2] - path[:, :, None]) * torch.tensor([L / 2, W / 2], device=x.device)
            dist = (dxy.norm(dim=-1, keepdim=True) / 20.0).clamp(max=3.0)
            x = torch.cat([x, dist], -1)
            F = F + 1
            h = self.emb(x.reshape(B * T, S, F))
            pad = ~present.reshape(B * T, S)
            empty = pad.all(1)
            pad = pad.clone()
            pad[empty, 0] = False                       # avoid all-masked attention rows
            # additive float mask: boolean padding masks give NaN on Apple's GPU
            h = self.sab(h, src_key_padding_mask=pad.float() * -1e4)
            h = h.masked_fill(pad[..., None], 0.0)
            n = (~pad).sum(1, keepdim=True).clamp(min=1)
            mean = h.sum(1) / n
            mx = h.masked_fill(pad[..., None], -1e4).max(1).values
            frame = torch.cat([mean, mx], -1).reshape(B, T, -1)
            bi = torch.cat([ball_in * ball_ok[..., None], ball_ok[..., None].float(), ectx], -1)
            c, _ = self.lstm(torch.cat([frame, self.ball_emb(bi)], -1))
            hp = h.reshape(B, T, S, -1)
            logit = self.poss(torch.cat([hp, c[:, :, None].expand(-1, -1, S, -1)], -1))[..., 0]
            logit = logit.masked_fill(~present, -1e4)
            free = self.free_logit(c)
            allp = torch.softmax(torch.cat([logit, free], -1), -1)
            # loose ball: from context, starting at the observed ball when there is one
            fxy = path + 0.5 * torch.tanh(self.free_xy(torch.cat([c, ball_in * ball_ok[..., None],
                                                                  ectx], -1)))
            return allp, fxy

    return Imputer()


def ball_from(allp, fxy, P):
    """Possession-weighted position, normalised coordinates."""
    import torch
    Pn = torch.stack([P[..., 0] / L * 2 - 1, P[..., 1] / W * 2 - 1], -1)
    return (allp[..., :-1, None] * Pn).sum(-2) + allp[..., -1:] * fxy


def _poss_label(seq):
    """Index of the player within POSS_M of the true ball (nearest), else S (loose)."""
    P, t, pr = seq["P"], seq["truth"], seq["present"]
    dist = np.linalg.norm(P - t[:, None], axis=2)
    dist[~pr] = np.inf
    j = dist.argmin(1)
    lab = np.where(dist[np.arange(len(j)), j] <= POSS_M, j, P.shape[1])
    return lab


def _gap_lengths():
    from ballfill.data import load_ball_data, runs
    d = load_ball_data()
    seen = np.isfinite(d.tac[:, 0])
    g = np.array([b - a for h in (1, 2) for a, b in runs(~seen & (d.match.half == h))])
    return np.maximum(g // STEP, 1)


def _pad_slots(seq, S):
    out = dict(seq)
    s = seq["x"].shape[1]
    if s < S:
        for k in ("x", "present", "P"):
            a = seq[k]
            pad = np.zeros((a.shape[0], S - s) + a.shape[2:], a.dtype)
            out[k] = np.concatenate([a, pad], 1)
    out["poss"] = _poss_label(out)
    return out


def train(seqs, epochs=12, windows_per_epoch=600, bs=8, seed=0, log=""):
    """seqs: list of per-half sequence dicts (same slot count after padding)."""
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    gl = _gap_lengths()
    net = make_model().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    total = epochs * (windows_per_epoch // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total,
                                                pct_start=0.1)
    weights = np.array([len(s["frames"]) for s in seqs], float)
    weights /= weights.sum()
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for _ in range(windows_per_epoch // bs):
            batch = {k: [] for k in ("x", "present", "P", "ball_in", "ball_ok", "truth", "w",
                                     "poss", "ectx")}
            for _ in range(bs):
                s = seqs[rng.choice(len(seqs), p=weights)]
                a = int(rng.integers(0, len(s["frames"]) - WIN))
                sl = slice(a, a + WIN)
                ok = s["ball_ok"][sl].copy()
                hidden = ~ok
                for _ in range(rng.integers(1, 4)):
                    g = int(min(rng.choice(gl), WIN - 20))
                    st = int(rng.integers(5, WIN - g - 5))
                    ok[st:st + g] = False
                    hidden[st:st + g] = True
                w = np.where(hidden, 1.0, 0.1) * s["truth_ok"][sl]
                x = s["x"][sl].copy()
                P = s["P"][sl].copy()
                bi = s["ball_in"][sl].copy()
                tr = s["truth"][sl].copy()
                if rng.random() < 0.5:                  # mirror the pitch in x
                    x[..., 0] *= -1
                    x[..., 2] *= -1
                    x[..., 4] *= -1
                    P[..., 0] = np.where(s["present"][sl], L - P[..., 0], 0)
                    bi[:, 0] = L - bi[:, 0]
                    tr[:, 0] = L - tr[:, 0]
                if rng.random() < 0.5:                  # mirror in y
                    x[..., 1] *= -1
                    x[..., 3] *= -1
                    P[..., 1] = np.where(s["present"][sl], W - P[..., 1], 0)
                    bi[:, 1] = W - bi[:, 1]
                    tr[:, 1] = W - tr[:, 1]
                batch["x"].append(x)
                batch["present"].append(s["present"][sl])
                batch["P"].append(P)
                batch["ball_in"].append(_norm_xy(bi))
                batch["ectx"].append(edge_context(bi, ok))
                batch["ball_ok"].append(ok)
                batch["truth"].append(tr)
                batch["w"].append(w)
                batch["poss"].append(s["poss"][sl])
            T = {}
            for k, v in batch.items():
                a = np.stack(v)
                T[k] = torch.from_numpy(a.astype(np.float32) if a.dtype == np.float64 else a
                                        ).to(DEV)
            T["ball_in"] = T["ball_in"].float()
            allp, fxy = net(T["x"], T["present"], T["ball_in"], T["ball_ok"], T["ectx"])
            pred = ball_from(allp, fxy, T["P"])
            pm = torch.stack([(pred[..., 0] + 1) / 2 * L, (pred[..., 1] + 1) / 2 * W], -1)
            err = torch.nn.functional.huber_loss(pm, T["truth"], reduction="none",
                                                 delta=2.0).sum(-1)
            w = T["w"].float()
            loss_b = (err * w).sum() / w.sum().clamp(min=1)
            ce = -torch.log(allp.gather(-1, T["poss"].long()[..., None])[..., 0].clamp(min=1e-6))
            loss_p = (ce * (w > 0)).sum() / (w > 0).sum().clamp(min=1)
            loss = loss_b + 0.5 * loss_p
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss_b.item()
        print("  {} epoch {:2d} ball loss {:.3f} ({:.0f} s)".format(
            log, ep, tot / (windows_per_epoch // bs), time.time() - t0), flush=True)
    return net.eval()


class ImputerFill:
    """method(d, ball, gap) with a trained network per scored unit (half or match)."""

    def __init__(self, nets, S):
        self.nets, self.S = nets, S
        self.key = None

    def set_key(self, key):
        self.key = key

    def __call__(self, d, ball, gap):
        import torch
        m = d.match
        ck = (id(m), gap.half)
        if getattr(self, "_ck", None) != ck:          # player features: once per match half
            s = sequence(m, ball, ball, np.isfinite(ball[:, 0]))[gap.half]
            self._seq, self._ck = _pad_slots(s, self.S), ck
        seq = dict(self._seq)
        fr = seq["frames"]
        b = ball[fr].astype(np.float32)
        seq["ball_ok"] = np.isfinite(b[:, 0])
        seq["ball_in"] = np.nan_to_num(b)
        a = max(np.searchsorted(fr, gap.a) - CONTEXT_STEPS, 0)
        b = min(np.searchsorted(fr, gap.b) + CONTEXT_STEPS, len(fr))
        sl = slice(a, b)
        net = self.nets[self.key if self.key is not None else gap.half]
        with torch.no_grad():
            t = lambda k: torch.from_numpy(np.ascontiguousarray(seq[k][sl][None])).to(DEV)  # noqa
            ec = edge_context(seq["ball_in"][sl], seq["ball_ok"][sl])
            allp, fxy = net(t("x"), t("present"), torch.from_numpy(
                _norm_xy(seq["ball_in"][sl])[None]).float().to(DEV), t("ball_ok"),
                torch.from_numpy(ec[None]).to(DEV))
            pred = ball_from(allp, fxy, t("P"))[0].cpu().numpy()
        pm = np.stack([(pred[:, 0] + 1) / 2 * L, (pred[:, 1] + 1) / 2 * W], -1)
        steps = fr[sl]
        k = (steps > gap.a - 1) & (steps < gap.b)
        xs = np.r_[gap.a - 1, steps[k], gap.b]
        ys = np.r_[ball[gap.a - 1][None], pm[k], ball[gap.b][None]]
        frames = np.arange(gap.a, gap.b)
        return np.c_[np.interp(frames, xs, ys[:, 0]), np.interp(frames, xs, ys[:, 1])]


def build_sequences():
    """{('BAR-ATH', half) or (match_id, half): seq}, slot count S, open-ball cache."""
    from ballfill.data import load_ball_data
    from ballfill.fuse import fuse
    from ballfill.opendata import IDSSE_IDS, load_open_ball
    d = load_ball_data()
    ball, src, sig, _ = fuse(d)
    seqs = {}
    for h, s in sequence(d.match, d.tac, ball, np.isfinite(ball[:, 0])).items():
        seqs[("BAR-ATH", h)] = s
    opens = {}
    for mid in IDSSE_IDS:
        ob = load_open_ball(mid)
        opens[mid] = ob
        for h, s in sequence(ob.d.match, ob.d.tac, ob.truth, ~ob.dead).items():
            seqs[(mid, h)] = s
    S = max(s["x"].shape[1] for s in seqs.values())
    seqs = {k: _pad_slots(s, S) for k, s in seqs.items()}
    return d, opens, seqs, S


def main(argv):
    import torch
    from ballfill.evaluate import real_gaps, synthetic_gaps
    from ballfill.opendata import IDSSE_IDS, open_gaps
    from ballfill.tune import errors, summary
    mode = argv[1] if len(argv) > 1 else "all"      # quick | ab | all
    quick = mode == "quick"
    d, opens, seqs, S = build_sequences()
    ep = 4 if quick else 24
    nets = {}
    for h in (1, 2):
        train_on = [s for k, s in seqs.items() if k != ("BAR-ATH", h)]
        nets[h] = train(train_on, epochs=ep, log="BAR-ATH h{}".format(h))
        torch.save(nets[h].state_dict(), os.path.join(CACHE_DIR, "imputer_h{}.pt".format(h)))
    f = ImputerFill(nets, S)
    print("A imputer", summary(errors(d, real_gaps(d), "A", f)), flush=True)
    print("B imputer", summary(errors(d, synthetic_gaps(d), "B", f)), flush=True)
    if mode != "all":
        return
    errs = []
    for mid in IDSSE_IDS:
        train_on = [s for k, s in seqs.items() if k[0] != mid]
        net = train(train_on, epochs=ep, log=mid)
        g = ImputerFill({mid: net}, S)
        g.set_key(mid)
        ob = opens[mid]
        e = errors(ob.d, open_gaps(ob), "C", g)
        print("C {} {}".format(mid, summary(e)), flush=True)
        errs += e
    print("C imputer", summary(errs), flush=True)


if __name__ == "__main__":
    main(sys.argv)
