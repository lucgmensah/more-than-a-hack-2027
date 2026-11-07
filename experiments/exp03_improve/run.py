"""Experiment 3: what actually improves the score? A fast development loop on Metrica games.

    python experiments/exp03_improve/run.py            # about 10-15 min
    python experiments/exp03_improve/run.py --quick    # game 1, one seed

Development set: Metrica sample games 1 and 2 with BAR-ATH-like tracking damage (exp02), several
degradation seeds each. Every configuration is scored on the same four to six damaged copies and
the table reports the mean and the spread, so a gain smaller than the spread is not a gain.

Configurations change one thing at a time:
  * alpha (how much the model trusts the evidence) at a fine grid, since exp02 hit the edge of
    its grid at 0.1;
  * `ttb` replaces/extends `nearest`: time to reach the ball instead of distance (Spearman-style);
  * `count_diff`: who outnumbers whom around the ball.

The HMM runs on 0.2 s blocks (hmm.py `block`), about 5x faster than frame by frame.
Smart Tagging channels cannot be evaluated here (no tagging for these games).
"""

import argparse
import os
import sys
import time

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

OUT = os.path.join(common.OUT_DIR, "exp03")
BLOCK = 5
ITERATIONS = 8
BASE = ("nearest", "ball_speed", "mean_speed")

# (label, channels, alpha, mean possession seconds, minimum dwell seconds)
CHANNEL_CONFIGS = (
    [("base a=%g" % a, BASE, a, 15.0, 0.0) for a in (0.03, 0.05, 0.1, 0.15, 0.2)]
    + [("base+count_diff a=%g" % a, BASE + ("count_diff",), a, 15.0, 0.0) for a in (0.05, 0.1)]
    + [("base+ttb a=%g" % a, BASE + ("ttb",), a, 15.0, 0.0) for a in (0.05, 0.1)]
    + [("ttb-for-nearest a=%g" % a, ("ttb", "ball_speed", "mean_speed"), a, 15.0, 0.0) for a in (0.05, 0.1)]
    + [("base+ttb+count a=%g" % a, BASE + ("ttb", "count_diff"), a, 15.0, 0.0) for a in (0.05, 0.1)]
)


# Stage 2: a minimum length for every possession and stoppage, with the best channel set of stage 1.
DWELL_CONFIGS = tuple(
    ("base a=%g dwell>=%gs" % (a, d), BASE, a, 15.0, d)
    for a in (0.1, 0.15, 0.25, 0.5) for d in (1.0, 2.0, 3.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--stage", choices=("channels", "dwell"), default="channels")
    args = parser.parse_args()

    bar = common.load_tracking()
    vis_runs, gap_runs = degrade.reference_runs(bar, common.graded_mask())
    games = (1,) if args.quick else (1, 2)
    seeds = range(1 if args.quick else args.seeds)
    all_configs = CHANNEL_CONFIGS if args.stage == "channels" else DWELL_CONFIGS
    configs = all_configs[:2] if args.quick else all_configs

    sets = []        # one entry per (game, seed): gt, periods, damaged tracking
    for game in games:
        clean = metrica_io.load_game(game)
        periods = clean["period"]
        gt = ground_truth.with_period_gap(
            ground_truth.frame_labels(metrica_io.load_events(game), clean), periods)
        for seed in seeds:
            damaged = degrade.degrade(clean, vis_runs, gap_runs, seed=seed)
            sets.append(("G{}s{}".format(game, seed), gt, period_runs(periods), periods, damaged))
        print("loaded game", game, flush=True)

    rows = []
    started = time.time()

    for window in (25, 50, 125):
        values = [score(gt, vote_baseline(trk, window, periods))
                  for _, gt, _, periods, trk in sets]
        rows.append(summarise("vote w=%d" % window, values, [r[0] for r in sets]))

    cache = {}
    for label, names, alpha, possession_s, min_dwell in configs:
        values = []
        for key, gt, runs, periods, trk in sets:
            if (key, names) not in cache:
                cache[(key, names)] = features.build_channels(trk, names, events=None)
            codes = hmm_codes(trk, names, alpha, possession_s, runs, ITERATIONS, BLOCK,
                              channels=cache[(key, names)], min_dwell_s=min_dwell)
            values.append(score(gt, codes))
        rows.append(summarise(label, values, [r[0] for r in sets]))
        r = rows[-1]
        print("{:<26} spellF1 {:.3f} +/-{:.3f}  macroF1 {:.3f}  poss {:>4.0f} (gt {:.0f})  a/d {:.3f}  [{:.0f}s]"
              .format(label, r["spellF1"], r["spell_sd"], r["macroF1"], r["poss"], r["gt_poss"],
                      r["alive_dead"], time.time() - started), flush=True)

    table = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    table.to_csv(os.path.join(OUT, "results_{}.csv".format(args.stage)), index=False)
    pd.set_option("display.width", 200)
    print("\nRanked by mean spellF1 over {} damaged copies:".format(len(sets)))
    print(table.sort_values("spellF1", ascending=False).round(3).to_string(index=False))


def summarise(label, results, keys):
    spell = np.array([r["spell_f1_mean"] for r in results])
    return {"config": label,
            "spellF1": spell.mean(), "spell_sd": spell.std(),
            "per_copy": " ".join("{}:{:.2f}".format(k, v) for k, v in zip(keys, spell)),
            "macroF1": np.mean([r["macro_f1"] for r in results]),
            "poss": np.mean([r["n_pred_spells"] for r in results]),
            "gt_poss": np.mean([r["n_gt_spells"] for r in results]),
            "alive_dead": np.mean([r["alive_dead_f1"] for r in results])}


if __name__ == "__main__":
    main()
