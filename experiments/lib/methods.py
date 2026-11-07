"""Methods shared by several experiments: the vote baseline, an HMM runner, and a scorer."""

import numpy as np

import common
import features
import hmm

pm = common.pm


def vote_baseline(tracking, window, periods):
    """The repository baseline (nearest player + centred majority vote), per period."""
    nearest = features.nearest_team_codes(tracking)
    out = np.full(len(nearest), common.D, dtype=np.int8)
    for period in np.unique(periods):
        sel = np.flatnonzero(periods == period)
        codes = nearest[sel]
        n = len(codes)
        if window <= 0:
            res = np.where(codes == 0, common.H, np.where(codes == 1, common.A, common.D))
        else:
            home = np.concatenate(([0], np.cumsum(codes == 0)))
            away = np.concatenate(([0], np.cumsum(codes == 1)))
            idx = np.arange(n)
            lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
            h, a = home[hi] - home[lo], away[hi] - away[lo]
            res = np.full(n, common.D, dtype=np.int8)
            seen = (h + a) > 0
            res[seen] = np.where(h[seen] >= a[seen], common.H, common.A)
        out[sel] = res
    return out


def period_runs(periods):
    """[(start, stop)] row spans of each period."""
    return [(int(np.flatnonzero(periods == p)[0]), int(np.flatnonzero(periods == p)[-1]) + 1)
            for p in np.unique(periods)]


def hmm_codes(tracking, names, alpha, possession_s, runs, iterations, block=1, channels=None,
              min_dwell_s=0.0):
    """Decode with the HMM. Pass `channels` to reuse an already built symbol matrix."""
    if channels is None:
        channels = features.build_channels(tracking, names, events=None)
    codes, _, _ = hmm.decode_match(channels, runs, names, features.N_SYMBOLS, alpha=alpha,
                                   iterations=iterations, verbose=False,
                                   possession_s=possession_s, block=block,
                                   min_dwell_s=min_dwell_s)
    return codes


def score(gt, pred, bootstrap=0):
    """Full metric dict against `gt`; the prediction is masked to X wherever gt is X."""
    pred = np.array(pred, dtype=np.int8)
    pred[gt == common.X] = common.X
    return pm.evaluate(gt, pred, bootstrap_n=bootstrap)
