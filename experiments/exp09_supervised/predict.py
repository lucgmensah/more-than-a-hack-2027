"""Regenerate the submission from the provided data and the two models committed in models/.

    python experiments/exp09_supervised/predict.py                 # the submission
    python experiments/exp09_supervised/predict.py --video-ball    # same, plus the ball found in the video

Nothing is fitted here, so the output is the same file on every run:
  1. tactical tracking, with the missing ball taken from the aligned broadcast tracking (broadcast.py);
  2. per-frame features (feats.py);
  3. P(dead) from models/dead.txt, P(home | alive) from models/team.txt (run3.py, run4.py describe
     how they were trained: DFL open data only, never BAR-ATH labels);
  4. penalty Viterbi with the decoder chosen on the validation matches (DECODER below).
Writes experiments/out/exp09/submission.csv, validates it and prints the public-window check.

--video-ball is off by default. It needs experiments/cache/exp10/video_ball.npz, produced by
experiments/exp10_video_ball/ (a ball detector run on the tactical video, a GPU job). The missing
ball is then filled from the video first and from the broadcast for what remains, and the result
goes to submission_video.csv, so the default submission is never overwritten.
"""

import argparse
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
OUT_DIR = os.path.join(common.OUT_DIR, "exp09")
VIDEO_BALL = os.path.join(common.CACHE_DIR, "exp10", "video_ball.npz")


def with_video_ball(tracking, path=VIDEO_BALL):
    """Tracking whose missing ball is taken from the video detections (exp10) where they exist."""
    if not os.path.isfile(path):
        raise SystemExit("{} not found: run experiments/exp10_video_ball/run.py first".format(path))
    data = np.load(path)
    if "usable" in data.files and not bool(data["usable"]):
        raise SystemExit("the video detector failed its own check (see the output of exp10_video_ball/link.py): not used")
    out = dict(tracking)
    ball = tracking["ball"].copy()
    empty = np.isnan(ball[data["frame"], 0])
    ball[data["frame"][empty], :2] = data["xy"][empty]
    out["ball"] = ball
    return out, int(empty.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video-ball", action="store_true",
                        help="also use the ball detected in the tactical video (exp10)")
    args = parser.parse_args()
    tracking = common.load_tracking()
    graded = common.graded_mask()
    source = tracking
    print("ball missing on graded frames: {:.1%}".format(np.isnan(source["ball"][graded, 0]).mean()))
    if args.video_ball:
        source, n = with_video_ball(tracking)
        print("  after the video detections ({} frames): {:.1%}".format(n, np.isnan(source["ball"][graded, 0]).mean()))
    filled, (first, second, cut) = broadcast.fill_ball(source)
    print("  after the broadcast ball: {:.1%}  (aligned: +{} frames before broadcast frame {}, +{} after)".format(
        np.isnan(filled["ball"][graded, 0]).mean(), first, cut, second))
    team_model = lgb.Booster(model_file=os.path.join(MODELS, "team.txt"))
    dead_model = lgb.Booster(model_file=os.path.join(MODELS, "dead.txt"))
    _, names = feats.build({"players": tracking["players"][:400], "ball": tracking["ball"][:400],
                            "team": tracking["team"]})
    logp = run4.combined(team_model, dead_model, run4.dead_columns(names), filled, pitch=BAR_PITCH)
    codes = common.mask_to_graded(M.decode(logp, DECODER, common.half_runs()))
    path = common.write_submission(
        os.path.join(OUT_DIR, "submission_video.csv" if args.video_ball else "submission.csv"), codes)
    share = np.bincount(codes[graded], minlength=4)[:3] / graded.sum()
    r = common.score_public(codes)
    print("written {}".format(path))
    print("whole match: H {:.2f}  A {:.2f}  D {:.2f}".format(*share))
    print("public window (17 possessions, a check only): spellF1 {:.3f}  macroF1 {:.3f}  possessions {} (truth {})".format(
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    if args.video_ball:
        reference = os.path.join(OUT_DIR, "submission.csv")
        if os.path.isfile(reference):
            base = common.pm.label_codes(common.pm.read_label_csv(reference))
            print("labels changed by the video ball: {:.1%} of graded frames".format((base != codes)[graded].mean()))
    subprocess.run([sys.executable, os.path.join(common.ROOT, "evaluation", "validate_submission.py"), path], check=False)


if __name__ == "__main__":
    main()
