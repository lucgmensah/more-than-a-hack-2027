"""Experiment 7: a teammate's temporal models, re-tuned on damaged Metrica instead of the public window.

    python experiments/exp07_friend_port/run.py            # about 15-25 min
    python experiments/exp07_friend_port/run.py --quick    # one copy per game, small grid

A teammate's report (not in this repository) tuned two models on the 17 public possessions:

  * "Viterbi + control": continuous Home / Away / Dead scores from distance, time to the ball (ETA),
    player density and team centroids, a Viterbi with a switch penalty, a dead-ball penalty and a
    minimum possession length, then a centred majority vote. Best public config: switch 0.50,
    dead 0.30, minimum 50 frames, vote +/-120 -> public spellF1 0.520 (360 configurations tried).
  * "adaptive smoothing": the same Viterbi, then a vote whose half-width is short near a decoded
    boundary, medium in stable play and long when the ball is missing or the evidence is weak.
    Best balanced public config: switch 1.0, dead 0.75, minimum 50, stable 35, boundary 15,
    uncertain 50 -> public spellF1 0.409, macroF1 0.636 (384 configurations tried).

Only the report is available, so the scores below are a re-implementation from its description:
the feature weights and score scales are ours, declared once and not tuned. The Dead score is the
mean player speed plus the ball speed (both in the teammate's feature store); with the published
dead penalties (>= 0.3) Dead was almost never predicted on Metrica, so the grid also allows a bonus. The Smart Tagging
prior and the video features are left out (Metrica has neither).

Protocol (as exp02): choose the configuration on the copies of one game, report it on the other
game, both ways. The teammate's two published configurations are also scored as they are.
"""

import argparse
import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXPERIMENTS, "lib"))
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp02_metrica_validation"))
import common  # noqa: E402
import degrade  # noqa: E402
import features  # noqa: E402
import ground_truth  # noqa: E402
import metrica_io  # noqa: E402
from methods import hmm_codes, period_runs, score, vote_baseline  # noqa: E402

FPS = 25
BLOCK = 5                       # Viterbi step = 0.2 s, as in the HMM experiments

# Declared feature weights for the team score (distance, ETA, density, centroid); not tuned.
W_DIST, W_ETA, W_DENS, W_CENT = 1.0, 1.0, 0.5, 0.25
SLOW_SPEED, SLOW_SCALE = 1.5, 1.0     # m/s: mean player speed below 1.5 m/s pushes towards Dead
BALL_STILL, BALL_MOVING = 1.0, 4.0    # m/s: a ball slower than 1 m/s pushes towards Dead, faster than 4 away
BALL_WEIGHT = 0.5

SWITCH = (0.5, 1.0, 2.0, 4.0, 8.0)
DEAD = (-0.6, -0.3, 0.0, 0.3, 0.75)  # negative = a bonus for Dead (needed: see README)
MIN_ALIVE = (0, 25, 50, 100)          # frames
WINDOW = (0, 50, 120)                 # half-width of the final centred vote
ADAPTIVE = tuple(itertools.product((25, 35, 50), (0, 15, 25), (50, 75, 125)))  # stable, boundary, uncertain
BOUNDARY_ZONE = 25                    # frames around a decoded change that count as "boundary"
WEAK_EVIDENCE = 0.2

TEAMMATE = {
    "teammate viterbi (s0.5 d0.3 m50 w120)": (0.5, 0.3, 50, ("global", 120)),
    "teammate adaptive (s1 d0.75 m50 35/15/50)": (1.0, 0.75, 50, ("adaptive", (35, 15, 50))),
}


# -- continuous evidence ---------------------------------------------------------------------------

def frame_scores(tracking):
    """team evidence e in [-1, 1] (+ = home; 0 when the ball is missing) and a slowness term in [-1, 1]."""
    players, ball, team = tracking["players"], tracking["ball"], tracking["team"]
    d_home, d_away = features._nearest_by_team(tracking)

    vel = np.nan_to_num(features.player_velocity(tracking), nan=0.0)
    delta = players[..., :2] + vel * features.REACTION_S - ball[:, None, :2]
    eta = features.REACTION_S + np.sqrt(np.einsum("nsc,nsc->ns", delta, delta)) / features.MAX_SPEED
    eta = np.where(np.isnan(players[..., 0]), np.nan, eta)
    t_home, t_away = features._team_min(eta, team, 0), features._team_min(eta, team, 1)

    delta = players[..., :2] - ball[:, None, :2]
    near = np.sqrt(np.einsum("nsc,nsc->ns", delta, delta)) <= features.ZONE_RADIUS
    density = near[:, team == 0].sum(axis=1) - near[:, team == 1].sum(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c_home = np.nanmean(players[:, team == 0, :2], axis=1)
        c_away = np.nanmean(players[:, team == 1, :2], axis=1)
        mean_speed = np.nanmean(players[..., 2], axis=1)
    dc_home = np.linalg.norm(c_home - ball[:, :2], axis=1)
    dc_away = np.linalg.norm(c_away - ball[:, :2], axis=1)

    parts = (W_DIST * np.tanh((d_away - d_home) / 2.0)
             + W_ETA * np.tanh((t_away - t_home) / 0.5)
             + W_DENS * np.clip(density, -2, 2) / 2.0
             + W_CENT * np.tanh((dc_away - dc_home) / 10.0))
    e = np.nan_to_num(parts / (W_DIST + W_ETA + W_DENS + W_CENT), nan=0.0)
    e[np.isnan(ball[:, 0])] = 0.0
    slow = np.clip((SLOW_SPEED - np.nan_to_num(mean_speed, nan=SLOW_SPEED)) / SLOW_SCALE, -1.0, 1.0)
    speed = features.ball_speed(tracking)
    slow = slow + BALL_WEIGHT * (np.where(speed < BALL_STILL, 1.0, 0.0) - np.where(speed > BALL_MOVING, 1.0, 0.0))
    return e.astype(np.float32), slow.astype(np.float32)


# -- penalty Viterbi with a minimum length for H and A ----------------------------------------------

def penalty_viterbi(e, slow, switch, dead, min_alive):
    """Maximise sum of state scores - switch * changes, no H/A stay shorter than min_alive frames."""
    n = len(e)
    frame = np.stack([e, -e, slow - dead], axis=1)                  # H, A, D
    starts = np.arange(0, n, BLOCK)
    em = np.add.reduceat(frame, starts, axis=0)
    lengths = np.minimum(BLOCK, n - starts)
    m = [max(1, int(round(min_alive / BLOCK)))] * 2 + [1]
    first = np.cumsum([0] + m[:-1])
    total = sum(m)
    owner = np.repeat(np.arange(3), m)
    trans = np.full((total, total), -np.inf)
    for s in range(3):
        for i in range(m[s] - 1):
            trans[first[s] + i, first[s] + i + 1] = 0.0
        last = first[s] + m[s] - 1
        trans[last, last] = 0.0
        for o in range(3):
            if o != s:
                trans[last, first[o]] = -switch
    big_em = em[:, owner]
    delta = np.full(total, -np.inf)
    delta[first] = big_em[0, first]
    back = np.empty((len(em), total), dtype=np.int16)
    for t in range(1, len(em)):
        cand = delta[:, None] + trans
        back[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + big_em[t]
    path = np.empty(len(em), dtype=np.int16)
    path[-1] = delta.argmax()
    for t in range(len(em) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.repeat(owner[path], lengths)[:n].astype(np.int8)


# -- final votes ------------------------------------------------------------------------------------

def mode_filter(labels, half):
    """Centred majority over H/A/D with a per-frame half-width (array or int); ties keep the label."""
    n = len(labels)
    half = np.broadcast_to(np.asarray(half), (n,))
    onehot = np.zeros((n + 1, 3), dtype=np.int32)
    onehot[np.arange(n) + 1, labels] = 1
    cum = np.cumsum(onehot, axis=0)
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - half), np.minimum(n, idx + half + 1)
    counts = (cum[hi] - cum[lo]).astype(np.float32)
    counts[idx, labels] += 0.5
    return counts.argmax(axis=1).astype(np.int8)


def adaptive_half(path, e, stable, boundary, uncertain):
    n = len(path)
    change = np.flatnonzero(np.diff(path)) + 1
    near = np.zeros(n + 1, dtype=np.int32)
    np.add.at(near, np.maximum(0, change - BOUNDARY_ZONE), 1)
    np.add.at(near, np.minimum(n, change + BOUNDARY_ZONE), -1)
    at_boundary = np.cumsum(near)[:n] > 0
    half = np.full(n, stable)
    half[np.abs(e) < WEAK_EVIDENCE] = uncertain
    half[at_boundary] = boundary
    return half


def post(path, e, mode):
    kind, arg = mode
    if kind == "global":
        return path if arg == 0 else mode_filter(path, arg)
    return mode_filter(path, adaptive_half(path, e, *arg))


def decode(sets_entry, switch, dead, min_alive, modes, cache):
    """Codes for each post-processing mode; the Viterbi path is cached per (copy, switch, dead, min)."""
    key, gt, runs, periods, trk, e, slow = sets_entry
    ck = (key, switch, dead, min_alive)
    if ck not in cache:
        path = np.empty(len(e), dtype=np.int8)
        for a, b in runs:
            path[a:b] = penalty_viterbi(e[a:b], slow[a:b], switch, dead, min_alive)
        cache[ck] = path
    path = cache[ck]
    out = {}
    for mode in modes:
        codes = np.empty_like(path)
        for a, b in runs:
            codes[a:b] = post(path[a:b], e[a:b], mode)
        out[mode] = codes
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--extend", action="store_true",
                        help="second pass past the edges of the first grid (dead <= -0.6, switch >= 8)")
    args = parser.parse_args()
    global SWITCH, DEAD, MIN_ALIVE
    if args.extend:
        SWITCH, DEAD, MIN_ALIVE = (8.0, 16.0, 32.0), (-0.6, -0.9, -1.2, -1.6), (0, 50, 100)
    seeds = range(1 if args.quick else 3)

    bar = common.load_tracking()
    vis_runs, gap_runs = degrade.reference_runs(bar, common.graded_mask())
    sets = []
    for game in (1, 2):
        clean = metrica_io.load_game(game)
        periods = clean["period"]
        gt = ground_truth.with_period_gap(
            ground_truth.frame_labels(metrica_io.load_events(game), clean), periods)
        for seed in seeds:
            trk = degrade.degrade(clean, vis_runs, gap_runs, seed=seed)
            e, slow = frame_scores(trk)
            sets.append(("G{}s{}".format(game, seed), gt, period_runs(periods), periods, trk, e, slow))
        print("loaded game", game, flush=True)

    results = []          # (label, family, per-copy dicts)
    started = time.time()

    def record(label, family, preds):
        results.append((label, family, [score(s[1], p) for s, p in zip(sets, preds)]))

    for w in (50, 120):
        record("vote w=%d" % w, "reference", [vote_baseline(s[4], w, s[3]) for s in sets])
    record("HMM a=0.15 dwell 3s (ours)", "reference",
           [hmm_codes(s[4], ("nearest", "ball_speed", "mean_speed"), 0.15, 15.0, s[2], 8, BLOCK,
                      min_dwell_s=3.0) for s in sets])
    print("references done [{:.0f}s]".format(time.time() - started), flush=True)

    cache = {}
    for label, (sw, dd, mm, mode) in TEAMMATE.items():
        record(label, "teammate", [decode(s, sw, dd, mm, (mode,), cache)[mode] for s in sets])

    grid_switch, grid_dead = (SWITCH[:2], DEAD[:1]) if args.quick else (SWITCH, DEAD)
    grid_min = MIN_ALIVE[2:3] if args.quick else MIN_ALIVE
    global_modes = tuple(("global", w) for w in WINDOW)
    adaptive_modes = tuple(("adaptive", a) for a in ADAPTIVE)
    for sw, dd, mm in itertools.product(grid_switch, grid_dead, grid_min):
        modes = global_modes + (adaptive_modes if mm == 50 else ())
        per_copy = [decode(s, sw, dd, mm, modes, cache) for s in sets]
        for mode in modes:
            name = ("viterbi s%g d%g m%d w%d" % (sw, dd, mm, mode[1]) if mode[0] == "global" else
                    "adaptive s%g d%g m%d %d/%d/%d" % ((sw, dd, mm) + mode[1]))
            record(name, mode[0], [pc[mode] for pc in per_copy])
        for s in sets:
            cache.pop((s[0], sw, dd, mm), None)
        print("s{} d{} m{} done [{:.0f}s]".format(sw, dd, mm, time.time() - started), flush=True)

    keys = [s[0] for s in sets]
    g1 = np.array([k.startswith("G1") for k in keys])
    rows = []
    for label, family, res in results:
        spell = np.array([r["spell_f1_mean"] for r in res])
        macro = np.array([r["macro_f1"] for r in res])
        rows.append({"config": label, "family": family,
                     "spell_all": spell.mean(), "spell_sd": spell.std(),
                     "spell_G1": spell[g1].mean(), "spell_G2": spell[~g1].mean(),
                     "macro_all": macro.mean(), "macro_G1": macro[g1].mean(), "macro_G2": macro[~g1].mean(),
                     "poss": np.mean([r["n_pred_spells"] for r in res])})
    table = pd.DataFrame(rows)
    out = os.path.join(common.OUT_DIR, "exp07")
    os.makedirs(out, exist_ok=True)
    table.to_csv(os.path.join(out, "results_extend.csv" if args.extend else "results.csv"), index=False)

    pd.set_option("display.width", 220)
    cols = ["config", "spell_all", "spell_sd", "spell_G1", "spell_G2", "macro_all", "poss"]
    print("\nReferences and the teammate's published configurations:")
    print(table[table.family.isin(("reference", "teammate"))][cols].round(3).to_string(index=False))

    print("\nTransfer: choose on one game (best spellF1), report on the other")
    for fam in ("global", "adaptive"):
        sub = table[table.family == fam]
        if sub.empty:
            continue
        for pick, report in (("G1", "G2"), ("G2", "G1")):
            best = sub.loc[sub["spell_" + pick].idxmax()]
            print("  {:<8} chosen on {}: {:<34} {} spellF1 {:.3f} macroF1 {:.3f}   (on {} {:.3f})".format(
                fam, pick, best.config, report, best["spell_" + report], best["macro_" + report],
                pick, best["spell_" + pick]))
    print("\nTop 10 over all copies (for information; selection above is the honest number):")
    print(table[table.family.isin(("global", "adaptive"))].sort_values("spell_all", ascending=False)
          .head(10)[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
