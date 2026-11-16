"""Regenerate the submission from the provided data and the two models committed in models/.

    python solution/predict.py

Nothing is fitted here, so the output is the same file on every run:
  1. tactical tracking, with the missing ball taken from the aligned broadcast tracking (broadcast.py);
  2. 159 per-frame features (features.py);
  3. P(dead) from models/dead.txt, P(home | alive) from models/team.txt (train.py says how they were
     trained: DFL open data only, never a BAR-ATH label);
  4. penalty Viterbi with the decoder chosen on the validation copies (DECODER below, decode.py).
Writes solution/out/submission.csv, validates it with the official validator and prints the
public-window check (a pipeline check, never used to choose anything).
"""

import os
import subprocess
import sys

import lightgbm as lgb
import numpy as np

import broadcast
import common
import decode
import model

MODELS = os.path.join(common.HERE, "models")
DECODER = {"switch": 8, "dead": 64, "min_alive": 0}
OUT = os.path.join(common.OUT_DIR, "submission.csv")


def main():
    tracking = common.load_tracking()
    graded = common.graded_mask()
    print("ball missing on graded frames: {:.1%}".format(np.isnan(tracking["ball"][graded, 0]).mean()))
    filled, (first, second, cut) = broadcast.fill_ball(tracking)
    print("  after the broadcast ball: {:.1%}  (aligned: +{} frames before broadcast frame {}, +{} after)".format(
        np.isnan(filled["ball"][graded, 0]).mean(), first, cut, second))
    team_model = lgb.Booster(model_file=os.path.join(MODELS, "team.txt"))
    dead_model = lgb.Booster(model_file=os.path.join(MODELS, "dead.txt"))
    cols = model.dead_columns(model.feature_names(tracking))
    logp = model.combined(team_model, dead_model, cols, filled, pitch=common.BAR_PITCH)
    codes = common.mask_to_graded(decode.decode(logp, DECODER, common.half_runs()))
    path = common.write_submission(OUT, codes)
    share = np.bincount(codes[graded], minlength=4)[:3] / graded.sum()
    r = common.score_public(codes)
    print("written {}".format(path))
    print("whole match: H {:.2f}  A {:.2f}  D {:.2f}".format(*share))
    print("public window (17 possessions, a check only): spellF1 {:.3f}  macroF1 {:.3f}  possessions {} (truth {})".format(
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    subprocess.run([sys.executable, os.path.join(common.ROOT, "evaluation", "validate_submission.py"), path], check=False)


if __name__ == "__main__":
    main()
