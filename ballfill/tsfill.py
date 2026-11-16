"""Pure time-series fillers: the ball's own track either side of a gap, nothing else.

- arima: auto-ARIMA per coordinate (order by lowest AIC), fitted by state-space maximum likelihood on the track around
  the gap (missing frames included as missing), then Kalman-smoothed through the gap. This uses
  both sides of the gap, unlike a forecast.
- prophet: Meta's Prophet per coordinate on the same window (seasonalities off: a ball has none),
  predicting the gap frames.

Scored on A, B and C against linear interpolation and the production HMM. They use no player data,
so they're baselines for how much the ball's own history can say.

    python -m ballfill.tsfill
"""

import logging
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FPS = 25
CONTEXT_S = 10.0
DECIMATE = 5           # fit on 5 Hz: the 25 Hz track is smooth, fitting is 5x cheaper
ORDERS = ((1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (0, 2, 1), (0, 2, 2))


def _window(ball, gap):
    c = int(CONTEXT_S * FPS)
    lo = max(gap.a - c, 0)
    hi = min(gap.b + c, len(ball))
    return lo, hi


def _fallback(ball, gap):
    from ballfill.fill import linear
    return linear(None, ball, gap)


def arima(d, ball, gap):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    lo, hi = _window(ball, gap)
    t = np.arange(lo, hi)
    keep = (t % DECIMATE == 0) | (t == gap.a - 1) | (t == gap.b)
    ts = t[keep]
    out = np.empty((gap.length, 2))
    frames = np.arange(gap.a, gap.b)
    for k in range(2):
        y = ball[ts, k].astype(np.float64)
        if np.isfinite(y).sum() < 8:
            return _fallback(ball, gap)
        try:
            best = None
            for order in ORDERS:            # auto-ARIMA: lowest AIC on this window
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = SARIMAX(y, order=order, trend="n").fit(disp=False, maxiter=50)
                if best is None or r.aic < best.aic:
                    best = r
            res = best
            # Kalman-smoothed observation: design matrix x smoothed state (uses both sides)
            Z = res.model.ssm["design"]
            Z = Z[0] if Z.ndim == 2 else Z[0, :, 0]
            fitted = Z @ res.smoothed_state
        except Exception:
            return _fallback(ball, gap)
        out[:, k] = np.interp(frames, ts, fitted)
    out[np.isnan(out)] = _fallback(ball, gap)[np.isnan(out)]
    return out


def prophet(d, ball, gap):
    import pandas as pd
    from prophet import Prophet
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    lo, hi = _window(ball, gap)
    t = np.arange(lo, hi)
    keep = (t % DECIMATE == 0) | (t == gap.a - 1) | (t == gap.b)
    ts = t[keep]
    base = pd.Timestamp("2025-01-01")
    out = np.empty((gap.length, 2))
    frames = np.arange(gap.a, gap.b)
    future = pd.DataFrame({"ds": base + pd.to_timedelta(frames / FPS, unit="s")})
    for k in range(2):
        y = ball[ts, k].astype(np.float64)
        ok = np.isfinite(y)
        if ok.sum() < 8:
            return _fallback(ball, gap)
        df = pd.DataFrame({"ds": base + pd.to_timedelta(ts[ok] / FPS, unit="s"), "y": y[ok]})
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=0.5, n_changepoints=min(25, ok.sum() // 3))
        m.fit(df)
        out[:, k] = m.predict(future)["yhat"].values
    return out


def main():
    from ballfill.carrier import CarrierHMM
    from ballfill.data import load_ball_data
    from ballfill.evaluate import real_gaps, synthetic_gaps
    from ballfill.evaluate_open import evaluate_open
    from ballfill.fill import linear
    from ballfill.run import CONFIG
    from ballfill.tune import errors, summary
    d = load_ball_data()
    methods = {"linear": linear, "arima": arima, "prophet": prophet}
    for which, gaps in (("A", real_gaps(d)), ("B", synthetic_gaps(d))):
        for name, m in methods.items():
            print("{} {:<8} {}".format(which, name, summary(errors(d, gaps, which, m))),
                  flush=True)
    res = evaluate_open(methods)
    for name, (errs, _) in res.items():
        print("C {:<8} {}".format(name, summary(errs)), flush=True)


if __name__ == "__main__":
    main()
