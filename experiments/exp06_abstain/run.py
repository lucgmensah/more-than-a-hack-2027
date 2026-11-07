"""Experiment 6: emit X (abstain) instead of guessing, on the six damaged Metrica copies.

    python experiments/exp06_abstain/run.py

spellF1 forward-fills X with the previous answer, so an X inside a possession keeps it whole;
macroF1 counts an X as a miss. Policies are applied on top of the exp04 configuration
(alpha 0.15, dwell 3 s) and change nothing else, so each row is a pure post-processing effect.
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
import metrica_io  # noqa: E402
from methods import hmm_codes, period_runs, score  # noqa: E402

FPS = 25
NAMES = ("nearest", "ball_speed", "mean_speed")
ALPHA, DWELL_S, BLOCK, ITERATIONS = 0.15, 3.0, 5, 8


def gap_length(missing):
    """For each frame, the length (frames) of the missing-ball run it belongs to, else 0."""
    out = np.zeros(len(missing), dtype=np.int64)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[s:e] = e - s
    return out


def policies(codes, missing, gaps):
    def x_where(mask):
        out = codes.copy()
        out[mask] = common.X
        return out
    dead = codes == common.D
    yield "none", codes
    yield "X if ball missing", x_where(missing)
    for seconds in (2, 5, 10, 20):
        yield "X if gap >= %ds" % seconds, x_where(gaps >= seconds * FPS)
    yield "X if D and ball missing", x_where(dead & missing)
    yield "X if D and gap >= 5s", x_where(dead & (gaps >= 5 * FPS))


def main():
    bar = common.load_tracking()
    vis_runs, gap_runs = degrade.reference_runs(bar, common.graded_mask())
    rows = {}
    for game in (1, 2):
        clean = metrica_io.load_game(game)
        periods = clean["period"]
        gt = ground_truth.with_period_gap(
            ground_truth.frame_labels(metrica_io.load_events(game), clean), periods)
        for seed in range(3):
            trk = degrade.degrade(clean, vis_runs, gap_runs, seed=seed)
            codes = hmm_codes(trk, NAMES, ALPHA, 15.0, period_runs(periods), ITERATIONS, BLOCK,
                              min_dwell_s=DWELL_S)
            missing = np.isnan(trk["ball"][:, 0])
            gaps = gap_length(missing)
            for label, pred in policies(codes, missing, gaps):
                r = score(gt, pred)
                rows.setdefault(label, []).append(
                    (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
            print("done G%d s%d" % (game, seed), flush=True)

    table = pd.DataFrame(
        [{"policy": k, "spellF1": np.mean([v[0] for v in vs]), "sd": np.std([v[0] for v in vs]),
          "macroF1": np.mean([v[1] for v in vs]), "poss": np.mean([v[2] for v in vs])}
         for k, vs in rows.items()])
    os.makedirs(os.path.join(common.OUT_DIR, "exp06"), exist_ok=True)
    table.to_csv(os.path.join(common.OUT_DIR, "exp06", "results.csv"), index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
