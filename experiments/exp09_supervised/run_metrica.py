"""Step 1: does a boosted model with temporal features beat exp08's logistic regression on Metrica?

    python experiments/exp09_supervised/run_metrica.py

Same faithful copies and protocol as exp08 (fit on one game, report on the other, both ways). The
decoder for a test game is chosen on the OTHER game's out-of-fold predictions.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
from methods import score  # noqa: E402

CACHE = os.path.join(M.EXPERIMENTS, "cache", "exp08")


def load(key):
    d = np.load(os.path.join(CACHE, key + ".npz"))
    return ({k: d[k] for k in ("players", "ball", "team")}, d["gt"], [tuple(r) for r in d["runs"].tolist()])


def main():
    t0 = time.time()
    games = {1: ("G1s0", "G1s1"), 2: ("G2s0", "G2s1")}
    logp, truth = {}, {}
    for train, test in ((1, 2), (2, 1)):
        xs, ys = [], []
        for key in games[train]:
            trk, gt, _ = load(key)
            x, y, names = M.training_rows(trk, gt)
            xs.append(x); ys.append(y)
        booster = M.fit(np.vstack(xs), np.concatenate(ys))
        print("[{:.0f}s] model fitted on game {} ({} rows, {} features)".format(
            time.time() - t0, train, sum(len(y) for y in ys), len(names)), flush=True)
        if train == 1:
            gain = booster.feature_importance("gain")
            order = np.argsort(gain)[::-1][:25]
            print("  top features:", ", ".join("{} {:.0%}".format(names[i], gain[i] / gain.sum()) for i in order))
        del xs, ys
        for key in games[test]:
            trk, gt, runs = load(key)
            logp[key], truth[key] = M.log_proba(booster, trk), (gt, runs)
            pred = logp[key].argmax(axis=1).astype(np.int8)
            r = score(gt, pred)
            print("  {} frame argmax: spellF1 {:.3f} macroF1 {:.3f} poss {}".format(
                key, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]), flush=True)
    grids = {key: M.grid(logp[key], *truth[key]) for key in logp}
    print("[{:.0f}s] decoder grids done".format(time.time() - t0), flush=True)
    for chosen_on, test in ((1, 2), (2, 1)):
        decoder, fit = M.best_decoder([grids[k] for k in games[chosen_on]])
        res = []
        for key in games[test]:
            gt, runs = truth[key]
            r = score(gt, M.decode(logp[key], decoder, runs))
            res.append(r)
            print("  decoder chosen on game {} {} (there {:.3f}) -> {}: spellF1 {:.3f} macroF1 {:.3f} poss {} (gt {}) a/d {:.3f} team acc {:.3f}".format(
                chosen_on, decoder, fit, key, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                r["n_gt_spells"], r["alive_dead_f1"], r["team_acc_gt_alive"]), flush=True)
        best = max(np.mean([grids[k][i][1] for k in games[test]]) for i in range(len(grids[games[test][0]])))
        print("  game {}: mean spellF1 {:.3f}   (best possible decoder on this game: {:.3f})".format(
            test, np.mean([r["spell_f1_mean"] for r in res]), best))
    np.savez(os.path.join(M.EXPERIMENTS, "cache", "exp09_metrica_logp.npz"), **logp)


if __name__ == "__main__":
    main()
