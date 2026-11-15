"""Experiment 9, fourth pass: dead ball decided without looking at how the ball moves.

    python experiments/exp09_supervised/run4.py [--tag v4] [--base v3]

Found on the public window (as a pipeline check): the run3 model gives P(dead) = 0.00 to stoppages
in which the BAR-ATH tracker still sees the ball. In the DFL data the ball is frozen while play is
stopped (median speed 0.0 m/s), so the model learned "a moving ball means play is on". In BAR-ATH
the tracker follows the real ball, which rolls away and is carried back at 1-10 m/s while the
players walk. That is a property of the training data, not of football, and it cannot be learned
away with more DFL data.

So the label is factorised:
  * dead or alive: a binary model that never sees ball motion or ball-player distances, only the
    players (speeds, block movement, team shape, with context), whether the ball is tracked (and
    for how long it has been / will stay missing) and where it is relative to the lines;
  * home or away, given alive: the three-class model of run3 (--base), renormalised over H and A.
The decoder and the validation are those of run3.py.
"""

import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import broadcast  # noqa: E402
import dfl_io  # noqa: E402
import feats  # noqa: E402
import model as M  # noqa: E402
import common  # noqa: E402
from methods import period_runs, score  # noqa: E402
from run2 import controlled, metrica_copy  # noqa: E402
from run3 import BAR_PITCH, CACHE, HOLD, KEEP, OUT, SMOOTH, STEP  # noqa: E402

DEAD_PREFIXES = ("mean_speed", "p90_speed", "share_slow", "block_shift", "block_spread", "speed_mean_",
                 "n_tracked", "team_distance", "centroid_gap", "obs", "since_ball", "until_ball",
                 "gap_length", "since_gap", "until_gap", "ball_to_touchline", "ball_to_goalline",
                 "ball_to_boundary", "last_ball_to_boundary", "next_ball_to_boundary", "ball_off_centre")
PARAMS = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
          "max_bin": 127, "num_threads": 6, "verbosity": -1, "seed": 0, "deterministic": True,
          "force_row_wise": True}
ROUNDS = 220


def dead_columns(names):
    return [i for i, n in enumerate(names) if n.startswith(DEAD_PREFIXES)]


def fit_dead(keys, tag, cols, min_frames):
    xs, ys = [], []
    for match, seed in keys:
        d = np.load(os.path.join(CACHE, "{}_{}_{}.npz".format(tag, match, seed)))
        label = controlled(d["label"], min_frames)
        rows = len(label[::STEP])
        xs.append(d["x"][:rows][:, cols])               # the un-flipped half: these columns are symmetric
        ys.append((label[::STEP] == 2).astype(np.int8))
    return lgb.train(PARAMS, lgb.Dataset(np.vstack(xs), label=np.concatenate(ys)), num_boost_round=ROUNDS)


def combined(team_model, dead_model, cols, trk, pitch=(105.0, 68.0)):
    """(N, 3) log-probabilities: P(D) from the dead model, P(H), P(A) from the team model given alive."""
    x, _ = feats.build(trk, pitch=pitch)
    p_dead = np.clip(dead_model.predict(x[:, cols]), 1e-6, 1 - 1e-6)
    p = team_model.predict(x)
    x, _ = feats.build(trk, flip=True, pitch=pitch)
    q = team_model.predict(x)
    p = (p + q[:, [1, 0, 2]]) / 2.0
    home = np.clip(p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9), 1e-6, 1 - 1e-6)
    return np.log(np.column_stack([(1 - p_dead) * home, (1 - p_dead) * (1 - home), p_dead]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v4")
    parser.add_argument("--base", default="v3")
    parser.add_argument("--min-s", type=float, default=1.5)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--data", default="v3", help="tag of the cached damaged copies")
    args = parser.parse_args()
    min_frames = int(args.min_s * 25)
    t0 = time.time()
    bar = common.load_tracking()
    graded = common.graded_mask()
    _, names = feats.build({"players": bar["players"][:400], "ball": bar["ball"][:400], "team": bar["team"]})
    cols = dead_columns(names)
    print("dead-ball model uses {} of {} features: {}".format(len(cols), len(names), ", ".join(names[i] for i in cols[:12]) + ", ..."))
    keys = [(m, s) for m in dfl_io.MATCHES for s in range(args.copies)]

    models = {}
    for which, subset in (("hold", [k for k in keys if k[0] not in HOLD]), ("final", keys)):
        path = os.path.join(CACHE, "dead_{}_{}.txt".format(args.tag, which))
        if not os.path.isfile(path):
            fit_dead(subset, args.data, cols, min_frames).save_model(path)
        suffix = "_hold" if which == "hold" else ""
        models[which] = (lgb.Booster(model_file=os.path.join(CACHE, "model_{}{}.txt".format(args.base, suffix))),
                         lgb.Booster(model_file=path))
    print("[{:.0f}s] dead-ball models ready".format(time.time() - t0), flush=True)

    grids, sets, rows = {}, {}, []
    for match in HOLD:
        d = np.load(os.path.join(CACHE, "{}_{}_0.npz".format(args.data, match)))
        trk = {k: d[k] for k in ("players", "ball", "team")}
        truth, runs = controlled(d["label"], min_frames), period_runs(d["period"])
        lp = combined(*models["hold"], cols, trk, pitch=tuple(d["pitch"]))
        dead = truth == 2
        said = np.exp(lp[:, 2]) > 0.5
        print("  held-out {}: dead-ball frames found {:.1%}, live frames called dead {:.1%}; with the ball visible: {:.1%} / {:.1%}".format(
            match, said[dead].mean(), said[~dead].mean(),
            said[dead & np.isfinite(d["ball"][:, 0])].mean(), said[~dead & np.isfinite(d["ball"][:, 0])].mean()))
        sets["DFL " + match] = (lp, truth, runs)
    for key in ("G1s0", "G1s1", "G2s0", "G2s1"):
        trk, gt, runs = metrica_copy(key, SMOOTH)
        sets["Metrica " + key] = (combined(*models["final"], cols, trk), gt, runs)
    for name, (lp, truth, runs) in sets.items():
        grids[name] = M.grid(lp, truth, runs)
        print("[{:.0f}s] {}: best decoder there {:.3f}".format(time.time() - t0, name, max(g[1] for g in grids[name])), flush=True)

    def report(name, chosen_on):
        decoder, _ = M.best_decoder([grids[k] for k in chosen_on])
        lp, truth, runs = sets[name]
        r = score(truth, M.decode(lp, decoder, runs))
        rows.append(dict(validation=name, decoder=str(decoder), **{k: r[k] for k in KEEP}))

    report("DFL " + HOLD[0], ["DFL " + HOLD[1]])
    report("DFL " + HOLD[1], ["DFL " + HOLD[0]])
    for key in ("G1s0", "G1s1"):
        report("Metrica " + key, ["Metrica G2s0", "Metrica G2s1"])
    for key in ("G2s0", "G2s1"):
        report("Metrica " + key, ["Metrica G1s0", "Metrica G1s1"])
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nValidation (decoder chosen on other data than the row it is reported on)")
    print(table.round(3).to_string(index=False))
    print("mean spellF1: DFL held-out {:.3f}, Metrica {:.3f}".format(
        table.spell_f1_mean[:2].mean(), table.spell_f1_mean[2:].mean()), flush=True)
    table.to_csv(os.path.join(OUT, "validation_{}.csv".format(args.tag)), index=False)
    decoder, fit_score = M.best_decoder(list(grids.values()))
    print("final decoder (mean over the six validation copies): {}  spellF1 there {:.3f}".format(decoder, fit_score))

    filled, _ = broadcast.fill_ball(bar)
    gt = common.pm.label_codes(common.pm.read_label_csv(common.GT_PUBLIC))
    for name, trk, suffix in (("tactical ball only", bar, "tactical"), ("+ broadcast ball", filled, "broadcast_fill")):
        lp = combined(*models["final"], cols, trk, pitch=BAR_PITCH)
        np.save(os.path.join(CACHE, "bar_logp_{}_{}.npy".format(args.tag, suffix)), lp)
        codes = common.mask_to_graded(M.decode(lp, decoder, common.half_runs()))
        r = common.score_public(codes)
        share = np.bincount(codes[graded], minlength=4)[:3] / graded.sum()
        print("BAR-ATH {:<19} public check: spellF1 {:.3f} macroF1 {:.3f} possessions {} (truth {}) a/d {:.3f} team acc {:.3f}"
              " | match H {:.2f} A {:.2f} D {:.2f}".format(name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                                                           r["n_gt_spells"], r["alive_dead_f1"], r["team_acc_gt_alive"], *share),
              flush=True)
        common.write_submission(os.path.join(OUT, "gbdt_{}_{}.csv".format(args.tag, suffix)), codes)
    print("[{:.0f}s] done".format(time.time() - t0))


if __name__ == "__main__":
    main()
