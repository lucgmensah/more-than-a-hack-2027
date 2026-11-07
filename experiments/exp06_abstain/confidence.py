"""Experiment 6b: abstain (X) on the frames where the HMM posterior is lowest.

    python experiments/exp06_abstain/confidence.py

Same setting as run.py, but the X rule is the model's own uncertainty: the lowest q share of
frames by smoothed posterior of the decoded state become X. q is the same for every copy.
"""

import os
import sys

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
import hmm  # noqa: E402
import metrica_io  # noqa: E402
from methods import period_runs, score  # noqa: E402

NAMES = ("nearest", "ball_speed", "mean_speed")
ALPHA, DWELL_S, BLOCK, ITERATIONS = 0.15, 3.0, 5, 8
QUANTILES = (0.0, 0.05, 0.10, 0.20, 0.30)


def main():
    bar = common.load_tracking()
    vis_runs, gap_runs = degrade.reference_runs(bar, common.graded_mask())
    rows = {q: [] for q in QUANTILES}
    for game in (1, 2):
        clean = metrica_io.load_game(game)
        periods = clean["period"]
        runs = period_runs(periods)
        gt = ground_truth.with_period_gap(
            ground_truth.frame_labels(metrica_io.load_events(game), clean), periods)
        for seed in range(3):
            trk = degrade.degrade(clean, vis_runs, gap_runs, seed=seed)
            ch = features.build_channels(trk, NAMES, events=None)
            sym = [ch[a:b] for a, b in runs]
            tables, trans = hmm.fit(sym, NAMES, features.N_SYMBOLS, ALPHA, ITERATIONS, False,
                                    15.0, BLOCK)
            codes = np.full(len(ch), common.X, dtype=np.int8)
            conf = np.zeros(len(ch))
            for (a, b), s in zip(runs, sym):
                codes[a:b] = hmm.viterbi(s, tables, trans, NAMES, ALPHA, BLOCK, DWELL_S)
                conf[a:b] = hmm.confidence(s, tables, trans, NAMES, ALPHA, BLOCK)
            live = gt != common.X
            for q in QUANTILES:
                pred = codes.copy()
                if q:
                    pred[conf <= np.quantile(conf[live], q)] = common.X
                r = score(gt, pred)
                rows[q].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
            print("done G%d s%d" % (game, seed), flush=True)
    table = pd.DataFrame([{"abstain_share": q, "spellF1": np.mean([v[0] for v in vs]),
                           "sd": np.std([v[0] for v in vs]), "macroF1": np.mean([v[1] for v in vs]),
                           "poss": np.mean([v[2] for v in vs])} for q, vs in rows.items()])
    table.to_csv(os.path.join(common.OUT_DIR, "exp06", "confidence.csv"), index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
