"""Experiment 7 (kept in exp06_abstain): explicit-duration (semi-Markov) decoding vs the Markov dwell model.

    python experiments/exp06_abstain/semimarkov.py

Same fitted emission tables (alpha 0.15, EM as before); only the decoder changes. Markov + 3 s
minimum is the reference. Six damaged Metrica copies, TARGET missing share 41%.
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
ALPHA, BLOCK, ITERATIONS = 0.15, 5, 8
DECODERS = [("markov dwell 3s", lambda s, t, tr: hmm.viterbi(s, t, tr, NAMES, ALPHA, BLOCK, 3.0))]
for shape in (1.0, 2.0, 4.0):
    for mean in ((15.0, 15.0, 12.0), (10.0, 10.0, 8.0), (20.0, 20.0, 15.0)):
        DECODERS.append(("semi k=%g mean=%g/%g" % (shape, mean[0], mean[2]),
                         lambda s, t, tr, k=shape, m=mean: hmm.semi_viterbi(
                             s, t, tr, NAMES, ALPHA, BLOCK, m, k, 3.0)))


def main():
    bar = common.load_tracking()
    vis_runs, gap_runs = degrade.reference_runs(bar, common.graded_mask())
    rows = {label: [] for label, _ in DECODERS}
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
            tables, trans = hmm.fit(sym, NAMES, features.N_SYMBOLS, ALPHA, ITERATIONS, False, 15.0, BLOCK)
            for label, decode in DECODERS:
                codes = np.full(len(ch), common.X, dtype=np.int8)
                for (a, b), s in zip(runs, sym):
                    codes[a:b] = decode(s, tables, trans)
                r = score(gt, codes)
                rows[label].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
            print("done G%d s%d" % (game, seed), flush=True)
    table = pd.DataFrame([{"decoder": k, "spellF1": np.mean([v[0] for v in vs]),
                           "sd": np.std([v[0] for v in vs]), "macroF1": np.mean([v[1] for v in vs]),
                           "poss": np.mean([v[2] for v in vs])} for k, vs in rows.items()])
    table.to_csv(os.path.join(common.OUT_DIR, "exp06", "semimarkov.csv"), index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
