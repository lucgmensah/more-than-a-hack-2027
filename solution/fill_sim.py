"""Is filling the missing ball from a second, noisier sensor worth it? A simulation with labels.

    python solution/fill_sim.py        # after train.py, a few minutes

On BAR-ATH the broadcast tracking has a ball on 12.7% of the frames where the tactical tracking has
none, and where both see it they are 0.9 m apart in the median (2.4 m at the 75th percentile, 6.7 m
at the 90th: balls in the air, seen from two angles). Whether using it helps cannot be measured on
BAR-ATH without labels, and choosing on the public window is not allowed. So the same situation is
built on the two held-out DFL matches, where labels exist:

  * primary sensor   = the damaged copy the model is validated on;
  * second sensor    = the clean ball, smoothed, displaced in proportion to its height (parallax)
                       plus a slow error, tuned until its distance to the primary matches the three
                       BAR-ATH percentiles above; it is kept on 12.7% of the primary's missing frames.

The models are the ones fitted without the two matches (train.py, "hold"). Reported: spellF1 /
macroF1 with the primary alone and with the fill. Result when the submission was built: team
accuracy +0.8 points, macroF1 +0.4, spellF1 +0.015 and -0.003, so predict.py uses the fill.
"""

import os

import lightgbm as lgb
import numpy as np

import common
import damage
import decode
import dfl
import model
from train import CACHE, HOLD

TARGET_SHARE = 0.127
PARALLAX, SLOW_M = 3.2, 0.70          # tuned on the distance percentiles, see the printout


def second_sensor(clean, rng):
    ball = damage.smooth_ball(clean["ball"], damage.SMOOTH)[:, :2].astype(np.float64)
    n = len(ball)
    angle = rng.uniform(0, 2 * np.pi)
    z = np.nan_to_num(clean["ball"][:, 2], nan=0.0).clip(0, 15)[:, None]
    slow = rng.normal(size=(n, 2))
    kernel = np.ones(75) / 75
    slow = np.column_stack([np.convolve(slow[:, c], kernel, mode="same") for c in range(2)])
    slow *= SLOW_M / slow.std()
    return ball + PARALLAX * z * np.array([np.cos(angle), np.sin(angle)]) + slow


def main():
    team_model = lgb.Booster(model_file=os.path.join(CACHE, "team_hold.txt"))
    dead_model = lgb.Booster(model_file=os.path.join(CACHE, "dead_hold.txt"))
    min_frames = int(model.MIN_POSSESSION_S * common.FPS)
    results = {}
    for match in HOLD:
        rng = np.random.default_rng(7)
        d = np.load(os.path.join(CACHE, "dfl_{}_0.npz".format(match)))
        other = np.load(os.path.join(CACHE, "dfl_{}_1.npz".format(match)))
        clean = dfl.load_match(match)
        primary = {k: d[k] for k in ("players", "ball", "team")}
        pitch = tuple(d["pitch"])
        truth, runs = model.controlled(d["label"], min_frames), common.period_runs(d["period"])
        second = second_sensor(clean, rng)
        seen = np.isfinite(primary["ball"][:, 0])
        dist = np.linalg.norm(second[seen] - primary["ball"][seen, :2], axis=1)
        print("{}: second sensor vs primary, distance pct 50/75/90: {} (BAR-ATH: 0.91 / 2.38 / 6.70)".format(
            match, np.round(np.percentile(dist, [50, 75, 90]), 2)))
        # the second sensor has its own gaps (the other damaged copy), then thinned to the BAR-ATH share
        candidate = ~seen & np.isfinite(other["ball"][:, 0])
        edges = np.flatnonzero(np.diff(np.concatenate(([0], candidate.astype(np.int8), [0]))))
        keep_prob = min(1.0, TARGET_SHARE * (~seen).sum() / max(candidate.sum(), 1))
        fill = np.zeros(len(seen), dtype=bool)
        for start, stop in zip(edges[::2], edges[1::2]):
            if rng.random() < keep_prob:
                fill[start:stop] = True
        filled = dict(primary)
        ball = primary["ball"].copy()
        ball[fill, :2] = second[fill]
        filled["ball"] = ball
        live = d["label"] < 2
        print("  primary ball missing {:.1%} (live play {:.1%}); filled on {:.1%} of the missing frames "
              "({:.0%} of the filled frames are live play)".format(
                  (~seen).mean(), (~seen[live]).mean(), fill.sum() / (~seen).sum(), live[fill].mean()))
        cols = model.dead_columns(model.feature_names(primary))
        results[match] = {name: (model.combined(team_model, dead_model, cols, trk, pitch=pitch), truth, runs)
                          for name, trk in (("primary", primary), ("filled", filled))}
    for match, chosen_on in ((HOLD[0], HOLD[1]), (HOLD[1], HOLD[0])):
        lp, truth, runs = results[chosen_on]["primary"]
        decoder, _ = decode.best_decoder([decode.grid(lp, truth, runs)])
        for name in ("primary", "filled"):
            lp, truth, runs = results[match][name]
            r = common.score(truth, decode.decode(lp, decoder, runs))
            print("{} {:<8} spellF1 {:.3f}  macroF1 {:.3f}  team accuracy {:.3f}  possessions {} (truth {})".format(
                match, name, r["spell_f1_mean"], r["macro_f1"], r["team_acc_gt_alive"], r["n_pred_spells"], r["n_gt_spells"]))


if __name__ == "__main__":
    main()
