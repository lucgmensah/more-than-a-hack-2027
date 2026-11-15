"""Experiment 9, second pass: labels harmonised to "controlled possession", validation on Metrica.

    python experiments/exp09_supervised/run2.py [--min-s 1.5] [--tag v2]

The DFL labels switch team at every touch (9-11 possessions per minute of live play, a third
shorter than 2 s). The challenge asks for controlled possession: a touch, a deflection or a lost
duel does not change it. With possessions shorter than --min-s seconds given back to the team that
had the ball, the DFL statistics match the Metrica event labels (5-6 possessions per live minute,
about 15% shorter than 2 s), so both sources describe the same thing.

Model selection uses Metrica only (other provider, never used to fit the trees): the decoder is
chosen on one Metrica game and reported on the other, both ways. Its ball is smoothed like the
training data. BAR-ATH is run once, with and without the ball of the aligned broadcast tracking.
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
import damage  # noqa: E402
import dfl_io  # noqa: E402
import feats  # noqa: E402
import model as M  # noqa: E402
import common  # noqa: E402
from methods import score  # noqa: E402

CACHE = os.path.join(M.EXPERIMENTS, "cache", "exp09")
OUT = os.path.join(common.OUT_DIR, "exp09")
STEP = 5
BAR_PITCH = (105.3, 68.0)
KEEP = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "alive_dead_f1", "team_acc_gt_alive")
PARAMS = dict(M.PARAMS, learning_rate=0.08, max_bin=127)
ROUNDS = 220


def spells(label):
    change = np.flatnonzero(np.diff(label)) + 1
    bounds = np.concatenate(([0], change, [len(label)]))
    return bounds[:-1], bounds[1:], label[bounds[:-1]]


def controlled(label, min_frames):
    """A possession shorter than min_frames that interrupts the other team's possession (or ends it
    just before a stoppage) is given back to that team: a touch is not a possession."""
    out = label.copy()
    if min_frames <= 0:
        return out
    for _ in range(3):
        start, stop, value = spells(out)
        changed = False
        for i in range(len(value)):
            if value[i] < 2 and stop[i] - start[i] < min_frames:
                before = value[i - 1] if i > 0 else 2
                after = value[i + 1] if i + 1 < len(value) else 2
                if before < 2 and before != value[i] and (after == before or after == 2):
                    out[start[i]: stop[i]] = before
                    changed = True
        if not changed:
            break
    return out


def metrica_copy(key, smooth):
    path = os.path.join(M.EXPERIMENTS, "cache", "exp08", key + ".npz")
    if not os.path.isfile(path):          # the faithful Metrica copies are built by exp08's code
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "exp08_run", os.path.join(M.EXPERIMENTS, "exp08_friend_learned", "run.py"))
        exp08 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp08)
        exp08.prepare((0, 1))
    d = np.load(path)
    trk = {k: d[k] for k in ("players", "ball", "team")}
    trk["ball"] = damage.smooth_ball(trk["ball"], smooth)
    return trk, d["gt"], [tuple(r) for r in d["runs"].tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-s", type=float, default=1.5)
    parser.add_argument("--tag", default="v2")
    args = parser.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    # -- fit on the 7 DFL matches, labels harmonised ----------------------------------------------
    path = os.path.join(CACHE, "model_{}.txt".format(args.tag))
    if not os.path.isfile(path):
        xs, ys = [], []
        for match in dfl_io.MATCHES:
            d = np.load(os.path.join(CACHE, "dfl_{}.npz".format(match)))
            label = controlled(d["label"], int(args.min_s * 25))
            xs.append(d["x"])
            ys.append(np.concatenate([label[::STEP], feats.flip_labels(label)[::STEP]]))
            assert len(ys[-1]) == len(xs[-1])
        x, y = np.vstack(xs), np.concatenate(ys)
        del xs
        lgb.train(PARAMS, lgb.Dataset(x, label=y), num_boost_round=ROUNDS).save_model(path)
        del x, y
    booster = lgb.Booster(model_file=path)
    print("[{:.0f}s] model {} ready (possessions shorter than {} s given back)".format(
        time.time() - t0, args.tag, args.min_s), flush=True)

    # -- Metrica: choose the decoder on one game, report on the other -----------------------------
    games = {1: ("G1s0", "G1s1"), 2: ("G2s0", "G2s1")}
    logp, truth, grids = {}, {}, {}
    for key in games[1] + games[2]:
        trk, gt, runs = metrica_copy(key, damage_smooth())
        logp[key], truth[key] = M.log_proba(booster, trk), (gt, runs)
        grids[key] = M.grid(logp[key], gt, runs)
        print("[{:.0f}s] Metrica {}: best decoder there {:.3f}".format(
            time.time() - t0, key, max(g[1] for g in grids[key])), flush=True)
    rows = []
    for chosen_on, test in ((1, 2), (2, 1)):
        decoder, fit = M.best_decoder([grids[k] for k in games[chosen_on]])
        for key in games[test]:
            gt, runs = truth[key]
            r = score(gt, M.decode(logp[key], decoder, runs))
            rows.append(dict(copy=key, decoder=str(decoder), **{k: r[k] for k in KEEP}))
    table = pd.DataFrame(rows)
    print("\nMetrica, model from DFL only, decoder chosen on the other Metrica game")
    print(table.round(3).to_string(index=False))
    print("mean spellF1 {:.3f}  macroF1 {:.3f}".format(table.spell_f1_mean.mean(), table.macro_f1.mean()), flush=True)
    table.to_csv(os.path.join(OUT, "metrica_{}.csv".format(args.tag)), index=False)
    decoder, fit = M.best_decoder(list(grids.values()))
    print("final decoder (both Metrica games): {}  spellF1 there {:.3f}".format(decoder, fit))

    # -- BAR-ATH, once ----------------------------------------------------------------------------
    bar = common.load_tracking()
    filled, constants = broadcast.fill_ball(bar)
    graded = common.graded_mask()
    for name, trk in (("tactical ball only", bar), ("+ broadcast ball", filled)):
        lp = M.log_proba(booster, trk, pitch=BAR_PITCH)
        codes = common.mask_to_graded(M.decode(lp, decoder, common.half_runs()))
        r = common.score_public(codes)
        share = np.bincount(codes[graded], minlength=4)[:3] / graded.sum()
        print("BAR-ATH {:<20} public check: spellF1 {:.3f} macroF1 {:.3f} possessions {} (truth {}) a/d {:.3f} team acc {:.3f}"
              " | match H {:.2f} A {:.2f} D {:.2f}".format(name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                                                           r["n_gt_spells"], r["alive_dead_f1"], r["team_acc_gt_alive"], *share),
              flush=True)
        suffix = "tactical" if trk is bar else "broadcast_fill"
        common.write_submission(os.path.join(OUT, "gbdt_{}_{}.csv".format(args.tag, suffix)), codes)
        np.save(os.path.join(CACHE, "bar_logp_{}_{}.npy".format(args.tag, suffix)), lp)
    print("[{:.0f}s] done".format(time.time() - t0))


def damage_smooth():
    return 17


if __name__ == "__main__":
    main()
