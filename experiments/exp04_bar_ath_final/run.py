"""Experiment 4: the configuration chosen on Metrica, run once on BAR-ATH, with and without Smart Tagging.

    python experiments/exp04_bar_ath_final/run.py

The hyper-parameters below were selected on the damaged Metrica games (exp03, six copies, a
selection never made on BAR-ATH):

    channels  nearest + ball_speed + mean_speed       (tracking only; the only ones Metrica can test)
    alpha     0.15
    dwell     no possession or stoppage shorter than 3 s
    block     0.2 s (speed only)

Three pre-declared runs, no search:
    T   tracking only                    -> comparable with exp03's Metrica numbers
    TSV TS with each tag channel crossed with "ball visible", so EM can learn to trust tags more
        when the ball is missing (decided after exp05 showed tags matter only in the gaps)
    TS  T + set_piece + phase_team       -> adds the Smart Tagging evidence, which exp01 showed to be
                                            the most useful for BAR-ATH and which Metrica cannot test

The public 5-minute window is printed as a pipeline check (17 possessions, noise >> the gaps).
Predictions are written to experiments/out/exp04/ and checked with the official validator.
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import common  # noqa: E402
import features  # noqa: E402
from methods import hmm_codes  # noqa: E402

ALPHA, DWELL_S, BLOCK, POSSESSION_S, ITERATIONS = 0.15, 3.0, 5, 15.0, 8
TRACKING = ("nearest", "ball_speed", "mean_speed")
RUNS = (
    ("T_tracking_only", TRACKING),
    ("TS_tracking+smart_tagging", TRACKING + ("set_piece", "phase_team")),
    ("TP_team_prior_only", TRACKING + ("phase_team@prior",)),
    ("TSV_tagging_by_visibility", TRACKING + ("set_piece@vis", "phase_team@vis")),
)
BASELINES = ("baseline_smooth50", "baseline_smooth125")


def main():
    tracking = common.load_tracking()
    events = common.smart_tags()
    runs = common.half_runs()
    out_dir = os.path.join(common.OUT_DIR, "exp04")

    print("{:<28} {:>8} {:>8} {:>5} {:>6} {:>7}".format(
        "run", "macroF1", "spellF1", "poss", "dur_s", "a/d F1"))
    for baseline in BASELINES:
        codes = common.pm.label_codes(common.pm.read_label_csv(
            os.path.join(common.ROOT, "evaluation", "results", baseline + ".csv")))
        show(baseline + " (repo)", common.score_public(codes))

    for name, names in RUNS:
        channels = features.build_channels(tracking, names, events)
        codes = hmm_codes(tracking, names, ALPHA, POSSESSION_S, runs, ITERATIONS, BLOCK,
                          channels=channels, min_dwell_s=DWELL_S)
        codes = common.mask_to_graded(codes)
        show(name, common.score_public(codes))
        path = common.write_submission(os.path.join(out_dir, name + ".csv"), codes)
        share = np.bincount(codes[common.graded_mask()], minlength=4)[:3] / common.graded_mask().sum()
        print("   share of graded frames  H {:.2f}  A {:.2f}  D {:.2f}".format(*share))
        subprocess.run([sys.executable, os.path.join(common.ROOT, "evaluation", "validate_submission.py"),
                        path], check=False, stdout=subprocess.DEVNULL)


def show(name, result):
    print("{:<28} {:>8.4f} {:>8.4f} {:>5d} {:>6.1f} {:>7.3f}".format(
        name, result["macro_f1"], result["spell_f1_mean"], result["n_pred_spells"],
        result["mean_spell_duration_s"], result["alive_dead_f1"]))


if __name__ == "__main__":
    main()
