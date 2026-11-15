"""Regenerate the submission from the provided data and the two models committed in models/.

    python experiments/exp09_supervised/predict.py

Nothing is fitted here, so the output is the same file on every run:
  1. tactical tracking, with the missing ball taken from the aligned broadcast tracking (broadcast.py);
  2. per-frame features (feats.py);
  3. P(dead) from models/dead.txt, P(home | alive) from models/team.txt (run3.py, run4.py describe
     how they were trained: DFL open data only, never BAR-ATH labels);
  4. penalty Viterbi with the decoder chosen on the validation matches (DECODER below).
Writes experiments/out/exp09/submission.csv, validates it and prints the public-window check.
"""

import os
import subprocess
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import broadcast  # noqa: E402
import feats  # noqa: E402
import model as M  # noqa: E402
import run4  # noqa: E402
import common  # noqa: E402

MODELS = os.path.join(HERE, "models")
DECODER = {"switch": 8, "dead": 64, "min_alive": 0}
BAR_PITCH = (105.3, 68.0)
OUT = os.path.join(common.OUT_DIR, "exp09", "submission.csv")


def main():
    tracking = common.load_tracking()
    filled, (first, second, cut) = broadcast.fill_ball(tracking)
    print("broadcast aligned: +{} frames before broadcast frame {}, +{} after".format(first, cut, second))
    team_model = lgb.Booster(model_file=os.path.join(MODELS, "team.txt"))
    dead_model = lgb.Booster(model_file=os.path.join(MODELS, "dead.txt"))
    _, names = feats.build({"players": tracking["players"][:400], "ball": tracking["ball"][:400],
                            "team": tracking["team"]})
    logp = run4.combined(team_model, dead_model, run4.dead_columns(names), filled, pitch=BAR_PITCH)
    codes = common.mask_to_graded(M.decode(logp, DECODER, common.half_runs()))
    path = common.write_submission(OUT, codes)
    graded = common.graded_mask()
    share = np.bincount(codes[graded], minlength=4)[:3] / graded.sum()
    r = common.score_public(codes)
    print("written {}".format(path))
    print("whole match: H {:.2f}  A {:.2f}  D {:.2f}".format(*share))
    print("public window (17 possessions, a check only): spellF1 {:.3f}  macroF1 {:.3f}  possessions {} (truth {})".format(
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    subprocess.run([sys.executable, os.path.join(common.ROOT, "evaluation", "validate_submission.py"), path], check=False)


if __name__ == "__main__":
    main()
