"""Make clean Metrica tracking look like the BAR-ATH automatic tracking.

Metrica's sample tracking has the ball on essentially every frame and no identity problems. The
challenge's tracking does not. This module removes and perturbs, using constants measured on the
BAR-ATH TRACKING (never on its ground truth):

* ball missing in runs resampled from BAR-ATH's own visible/missing run lengths (so about 41% of
  frames lose the ball, in bursts rather than independently);
* each player absent in bursts so that about 7% of player observations are missing (BAR-ATH has
  20.5 tracked objects per graded frame against 22 players);
* Gaussian position noise (declared, not measured): 0.25 m on players, 0.5 m on the ball.

There is no identity to remove: every method in this repository already uses only team + position.
The degradation parameters are declared once here and are not tuned on any score.
"""

import numpy as np

PLAYER_ABSENT_SHARE = 0.07
PLAYER_ABSENT_MEAN_S = 2.0
PLAYER_NOISE_M = 0.25
TARGET_MISSING = 0.41
MISSING_TOLERANCE = 0.03
BALL_NOISE_M = 0.5
FPS = 25


def run_lengths(visible):
    """Lengths of consecutive True runs and False runs in a boolean array."""
    change = np.flatnonzero(np.diff(visible.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change, [len(visible)]))
    lengths = np.diff(bounds)
    states = visible[bounds[:-1]]
    return lengths[states], lengths[~states]


def reference_runs(bar_ath_tracking, graded):
    """Visible / missing run lengths of the BAR-ATH ball, inside the graded frames only."""
    visible = ~np.isnan(bar_ath_tracking["ball"][:, 0])
    vis_runs, gap_runs = [], []
    start = None
    for i, flag in enumerate(graded):
        if flag and start is None:
            start = i
        if (not flag or i == len(graded) - 1) and start is not None:
            stop = i if not flag else i + 1
            v, g = run_lengths(visible[start:stop])
            vis_runs.append(v)
            gap_runs.append(g)
            start = None
    return np.concatenate(vis_runs), np.concatenate(gap_runs)


def ball_visibility_mask(n, vis_runs, gap_runs, rng):
    """Alternating visible / missing runs sampled from the BAR-ATH empirical distributions."""
    mask = np.empty(n, dtype=bool)
    pos, visible = 0, bool(rng.integers(2))
    while pos < n:
        length = int(rng.choice(vis_runs if visible else gap_runs))
        mask[pos: pos + length] = visible
        pos += length
        visible = not visible
    return mask


def player_absence_mask(n, n_players, rng):
    """(n, P) True where a player observation is dropped, in bursts of ~2 s, ~7% of the total."""
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


def degrade(tracking, vis_runs, gap_runs, seed=0):
    """New tracking dict with missing ball / players and position noise. Speeds are recomputed."""
    from metrica_io import displacement_speed

    rng = np.random.default_rng(seed)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in tracking.items() if not k.startswith("_")}   # drop cached derived arrays
    n, n_players = out["players"].shape[:2]

    out["players"][..., :2] += rng.normal(0.0, PLAYER_NOISE_M, size=out["players"][..., :2].shape)
    out["ball"][:, :2] += rng.normal(0.0, BALL_NOISE_M, size=out["ball"][:, :2].shape)

    out["players"][player_absence_mask(n, n_players, rng)] = np.nan
    for _ in range(200):      # redraw until the missing share is close to BAR-ATH's 41% (heavy tails)
        visible = ball_visibility_mask(n, vis_runs, gap_runs, rng)
        if abs((~visible).mean() - TARGET_MISSING) <= MISSING_TOLERANCE:
            break
    out["ball"][~visible] = np.nan

    periods = out["period"]
    for period in np.unique(periods):
        sel = np.flatnonzero(periods == period)
        out["players"][sel, :, 2] = displacement_speed(out["players"][sel, :, :2])
    return out
