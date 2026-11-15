"""Make clean official tracking (DFL / TRACAB) look like the BAR-ATH automatic tracking.

Every constant is calibrated on LABEL-FREE statistics of the BAR-ATH tracking:

* ball smoothing: the BAR-ATH ball changes velocity much less sharply than an optical ball track
  (90th percentile of the 6-frame velocity change 4.3 m/s against 7.5-8.3), so the clean ball is
  low-pass filtered until that statistic matches;
* ball missing: a model of "is the ball missing?" from PLAYER-ONLY context (speeds, block movement,
  a few seconds before and after) is fitted on BAR-ATH itself (no label involved) and sampled on
  the clean match with a slowly varying threshold, so gaps come in long runs as in BAR-ATH
  (84 gaps per hour, 41% of frames); a ball more than MARGIN metres outside the pitch is always
  dropped (BAR-ATH: 4% of visible frames are off the pitch, the clean data: 31%);
* players: 7% of observations dropped in bursts of about 2 s (20.5 tracked of 22).
"""

import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feats  # noqa: E402

FPS = 25
PLAYER_ONLY = ("mean_speed", "p90_speed", "share_slow", "block_shift", "block_spread", "speed_mean_")
PLAYER_ABSENT_SHARE, PLAYER_ABSENT_MEAN_S = 0.07, 2.0


def player_columns(names):
    return [i for i, n in enumerate(names) if n.startswith(PLAYER_ONLY)]


def fit_missing_model(bar, graded, pitch=(105.3, 68.0), rounds=150):
    """P(ball missing | player-only context), fitted on BAR-ATH graded frames. No labels.

    Returns (model, column indices, AUC of a first-half model on the second half)."""
    x, names = feats.build(bar, pitch=pitch)
    cols = player_columns(names)
    y = np.isnan(bar["ball"][:, 0]).astype(np.int8)
    sel = np.flatnonzero(graded)[::3]
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 500,
              "feature_fraction": 0.8, "lambda_l2": 5.0, "num_threads": 6, "verbosity": -1, "seed": 0}
    half = sel[len(sel) // 2]
    early, late = sel[sel < half], sel[sel >= half]
    first = lgb.train(params, lgb.Dataset(x[early][:, cols], label=y[early]), rounds)
    p, t = first.predict(x[late][:, cols]), y[late]
    ranks = np.empty(len(p))
    ranks[np.argsort(p)] = np.arange(len(p))
    auc = (ranks[t == 1].mean() - (t.sum() - 1) / 2) / (len(t) - t.sum())
    model = lgb.train(params, lgb.Dataset(x[sel][:, cols], label=y[sel]), rounds)
    return model, cols, float(auc)


def smooth_ball(ball, window):
    """Centred moving average of the ball position (NaN-safe), `window` frames wide."""
    if window <= 1:
        return ball.copy()
    out = ball.copy()
    k = window // 2
    for c in range(2):
        v = ball[:, c].astype(np.float64)
        ok = np.isfinite(v)
        s = np.concatenate(([0.0], np.cumsum(np.where(ok, v, 0.0))))
        m = np.concatenate(([0], np.cumsum(ok)))
        idx = np.arange(len(v))
        lo, hi = np.clip(idx - k, 0, len(v)), np.clip(idx + k + 1, 0, len(v))
        mean = (s[hi] - s[lo]) / np.maximum(m[hi] - m[lo], 1)
        out[:, c] = np.where(ok, mean, np.nan)
    return out


def _slow_uniform(n, corr_s, rng):
    """Uniform(0,1) values held constant over stretches of about corr_s seconds."""
    out = np.empty(n)
    pos = 0
    while pos < n:
        length = max(1, int(rng.exponential(corr_s * FPS)))
        out[pos: pos + length] = rng.random()
        pos += length
    return out


def _runs(flag):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flag.astype(np.int8), [0]))))
    return edges[::2], edges[1::2]


def _close(missing, keep, min_visible, min_gap):
    """Visible stretches shorter than min_visible between two gaps are dropped too; gaps shorter than
    min_gap are restored (except where `keep`, the ball off the pitch)."""
    out = missing.copy()
    for start, stop in zip(*_runs(~out)):
        if stop - start < min_visible and start > 0 and stop < len(out):
            out[start:stop] = True
    for start, stop in zip(*_runs(out)):
        if stop - start < min_gap and not keep[start:stop].any():
            out[start:stop] = False
    return out


def _absence(n, n_players, rng):
    mean_absent = PLAYER_ABSENT_MEAN_S * FPS
    mean_present = mean_absent * (1.0 - PLAYER_ABSENT_SHARE) / PLAYER_ABSENT_SHARE
    out = np.zeros((n, n_players), dtype=bool)
    for p in range(n_players):
        pos, absent = 0, bool(rng.random() < PLAYER_ABSENT_SHARE)
        while pos < n:
            length = max(1, int(rng.exponential(mean_absent if absent else mean_present)))
            out[pos: pos + length, p] = absent
            pos += length
            absent = not absent
    return out


def block_missing_rates(bar, graded, block=7500):
    """Share of frames with the ball missing in consecutive 5-minute blocks of the graded frames."""
    missing = np.isnan(bar["ball"][:, 0])
    edges = np.flatnonzero(np.diff(np.concatenate(([0], graded.astype(np.int8), [0]))))
    rates = []
    for start, stop in zip(edges[::2], edges[1::2]):
        for lo in range(start, stop - block // 2, block):
            rates.append(float(missing[lo: min(lo + block, stop)].mean()))
    return np.array(rates)


def damage(trk, missing_model, cols, seed=0, smooth=17, corr_s=15.0, margin=0.3, scale=1.0,
           min_visible_s=2.0, min_gap_s=1.0, pitch=(105.0, 68.0), block_rates=None, block=7500):
    """block_rates: BAR-ATH's 5-minute missing shares. When given, every 5-minute block of the match
    draws one of them and the threshold is scaled until the block loses the ball that often: the
    BAR-ATH tracker has good and bad stretches (11% to 65% missing), and in the good ones the ball
    stays visible through most stoppages."""
    rng = np.random.default_rng(seed)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in trk.items()}
    n, n_players = out["players"].shape[:2]
    blank = {"players": out["players"], "ball": np.full_like(out["ball"], np.nan), "team": out["team"]}
    x, _ = feats.build(blank, pitch=pitch)
    p = missing_model.predict(x[:, cols])
    del x
    k = FPS                                              # +/-1 s average: the tracker does not flicker
    c = np.concatenate(([0.0], np.cumsum(p)))
    idx = np.arange(n)
    lo, hi = np.clip(idx - k, 0, n), np.clip(idx + k + 1, 0, n)
    p = (c[hi] - c[lo]) / (hi - lo)
    bx, by = out["ball"][:, 0], out["ball"][:, 1]
    outside = (bx < -margin) | (bx > pitch[0] + margin) | (by < -margin) | (by > pitch[1] + margin)
    u = _slow_uniform(n, corr_s, rng)
    if block_rates is None:
        missing = (scale * p > u) | outside
        keep = outside
    else:
        p = np.where(outside, np.maximum(p, 0.9), p)
        missing = np.zeros(n, dtype=bool)
        for start in range(0, n, block):
            sl = slice(start, start + block)
            target = float(rng.choice(block_rates))
            low, high = 0.0, 8.0
            for _ in range(20):
                mid = (low + high) / 2
                if (mid * p[sl] > u[sl]).mean() < target:
                    low = mid
                else:
                    high = mid
            missing[sl] = high * p[sl] > u[sl]
        keep = np.zeros(n, dtype=bool)
    missing = _close(missing, keep, int(min_visible_s * FPS), int(min_gap_s * FPS))
    ball = smooth_ball(out["ball"], smooth)
    ball[missing] = np.nan
    out["ball"] = ball
    out["players"][_absence(n, n_players, rng)] = np.nan
    return out
