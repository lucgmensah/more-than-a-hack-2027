"""Experiment 9, third pass: damage with BAR-ATH's good and bad stretches, team-shape features.

    python experiments/exp09_supervised/run3.py [--tag v3] [--min-s 1.5] [--copies 2]

Changes from run2.py:
  * damage.damage(block_rates=...): every 5-minute block of a DFL match loses the ball as often as a
    randomly drawn 5-minute block of BAR-ATH (12% to 80% of frames). In run.py / run2.py the ball
    was missing on 80% of dead-ball frames everywhere, the model learned "ball visible = play on",
    and missed stoppages in which the BAR-ATH tracker keeps the ball;
  * two damaged copies per match (different draws), so each match is seen in two tracking regimes;
  * features that do not need the ball (team shape, which way the block moves).

Validation, never on BAR-ATH: two DFL matches held out (fit on the other five), and both Metrica
games (never used to fit; decoder chosen on one game, reported on the other). The final decoder is
the best on average over those six validation copies.
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
from methods import period_runs, score  # noqa: E402
from run2 import controlled, metrica_copy  # noqa: E402

CACHE = os.path.join(M.EXPERIMENTS, "cache", "exp09")
OUT = os.path.join(common.OUT_DIR, "exp09")
STEP = 5
HOLD = ("J03WMX", "J03WR9")
BAR_PITCH = (105.3, 68.0)
KEEP = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "alive_dead_f1", "team_acc_gt_alive")
PARAMS = dict(M.PARAMS, learning_rate=0.08, max_bin=127)
ROUNDS = 220
SMOOTH = 17


def prepare(match, seed, booster, cols, rates, tag):
    path = os.path.join(CACHE, "{}_{}_{}.npz".format(tag, match, seed))
    if not os.path.isfile(path):
        clean = dfl_io.load_match(match)
        pitch = tuple(clean["pitch"])
        trk = damage.damage(clean, booster, cols, seed=seed, pitch=pitch, block_rates=rates, smooth=SMOOTH)
        x, _, _ = M.training_rows(trk, clean["label"], step=STEP, pitch=pitch)
        np.savez(path, x=x, players=trk["players"], ball=trk["ball"], team=trk["team"],
                 period=clean["period"], label=clean["label"], pitch=clean["pitch"])
    d = np.load(path)
    return {k: d[k] for k in d.files}


def fit(sets, min_frames):
    n = sum(len(s["x"]) for s in sets)
    x = np.empty((n, sets[0]["x"].shape[1]), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    pos = 0
    for s in sets:
        label = controlled(s["label"], min_frames)
        rows = len(s["x"])
        x[pos: pos + rows] = s["x"]
        y[pos: pos + rows] = np.concatenate([label[::STEP], feats.flip_labels(label)[::STEP]])
        pos += rows
    return lgb.train(PARAMS, lgb.Dataset(x, label=y), num_boost_round=ROUNDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v3")
    parser.add_argument("--min-s", type=float, default=1.5)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--data", default="v3", help="tag of the cached damaged copies (label-independent)")
    args = parser.parse_args()
    min_frames = int(args.min_s * 25)
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bar = common.load_tracking()
    graded = common.graded_mask()
    rates = damage.block_missing_rates(bar, graded)
    os.makedirs(CACHE, exist_ok=True)
    missing_path = os.path.join(CACHE, "missing_model.txt")
    if not os.path.isfile(missing_path):                    # label-free: "is the ball missing?" from player context
        fitted, _, auc = damage.fit_missing_model(bar, graded, pitch=BAR_PITCH)
        print("missing-ball model: held-out AUC {:.3f}".format(auc), flush=True)
        fitted.save_model(missing_path)
    booster = lgb.Booster(model_file=missing_path)
    _, names = feats.build({"players": bar["players"][:400], "ball": bar["ball"][:400], "team": bar["team"]})
    cols = damage.player_columns(names)

    sets = {}
    for match in dfl_io.MATCHES:
        for seed in range(args.copies):
            sets[(match, seed)] = prepare(match, seed, booster, cols, rates, args.data)
            d = sets[(match, seed)]
            obs, lab = np.isfinite(d["ball"][:, 0]), d["label"]
            print("[{:.0f}s] {} copy {}: ball missing {:.2f} (in D {:.2f}, live {:.2f})".format(
                time.time() - t0, match, seed, (~obs).mean(), (~obs[lab == 2]).mean(), (~obs[lab < 2]).mean()), flush=True)

    grids, rows = {}, []
    # -- two DFL matches held out -----------------------------------------------------------------
    path = os.path.join(CACHE, "model_{}_hold.txt".format(args.tag))
    if not os.path.isfile(path):
        fit([s for (m, _), s in sets.items() if m not in HOLD], min_frames).save_model(path)
    hold_model = lgb.Booster(model_file=path)
    held = {}
    for match in HOLD:
        d = sets[(match, 0)]
        truth = controlled(d["label"], min_frames)
        runs = period_runs(d["period"])
        lp = M.log_proba(hold_model, d, pitch=tuple(d["pitch"]))
        held[match] = (lp, truth, runs)
        grids["DFL " + match] = M.grid(lp, truth, runs)
        print("[{:.0f}s] held-out {}: best decoder there {:.3f}".format(
            time.time() - t0, match, max(g[1] for g in grids["DFL " + match])), flush=True)

    # -- final model, Metrica ---------------------------------------------------------------------
    path = os.path.join(CACHE, "model_{}.txt".format(args.tag))
    if not os.path.isfile(path):
        fit(list(sets.values()), min_frames).save_model(path)
    final = lgb.Booster(model_file=path)
    del sets
    gain = final.feature_importance("gain")
    order = np.argsort(gain)[::-1][:20]
    print("[{:.0f}s] final model ready. Top features: {}".format(time.time() - t0, ", ".join(
        "{} {:.0%}".format(final.feature_name()[i], gain[i] / gain.sum()) for i in order)), flush=True)
    _, names = feats.build({"players": bar["players"][:400], "ball": bar["ball"][:400], "team": bar["team"]})
    print("  feature names:", ", ".join("{}={}".format(i, names[i]) for i in order))
    metrica = {}
    for key in ("G1s0", "G1s1", "G2s0", "G2s1"):
        trk, gt, runs = metrica_copy(key, SMOOTH)
        lp = M.log_proba(final, trk)
        metrica[key] = (lp, gt, runs)
        grids["Metrica " + key] = M.grid(lp, gt, runs)
        print("[{:.0f}s] Metrica {}: best decoder there {:.3f}".format(
            time.time() - t0, key, max(g[1] for g in grids["Metrica " + key])), flush=True)

    def report(name, lp, truth, runs, chosen_on):
        decoder, _ = M.best_decoder([grids[k] for k in chosen_on])
        r = score(truth, M.decode(lp, decoder, runs))
        rows.append(dict(validation=name, decoder_from=" + ".join(chosen_on)[:28], decoder=str(decoder),
                         **{k: r[k] for k in KEEP}))

    report("DFL " + HOLD[0], *held[HOLD[0]], ["DFL " + HOLD[1]])
    report("DFL " + HOLD[1], *held[HOLD[1]], ["DFL " + HOLD[0]])
    for key in ("G1s0", "G1s1"):
        report("Metrica " + key, *metrica[key], ["Metrica G2s0", "Metrica G2s1"])
    for key in ("G2s0", "G2s1"):
        report("Metrica " + key, *metrica[key], ["Metrica G1s0", "Metrica G1s1"])
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nValidation (decoder chosen on other data than the row it is reported on)")
    print(table.round(3).to_string(index=False))
    print("mean spellF1: DFL held-out {:.3f}, Metrica {:.3f}".format(
        table.spell_f1_mean[:2].mean(), table.spell_f1_mean[2:].mean()), flush=True)
    table.to_csv(os.path.join(OUT, "validation_{}.csv".format(args.tag)), index=False)
    decoder, fit_score = M.best_decoder(list(grids.values()))
    print("final decoder (mean over the six validation copies): {}  spellF1 there {:.3f}".format(decoder, fit_score))

    # -- BAR-ATH, once ----------------------------------------------------------------------------
    filled, _ = broadcast.fill_ball(bar)
    for name, trk, suffix in (("tactical ball only", bar, "tactical"), ("+ broadcast ball", filled, "broadcast_fill")):
        lp = M.log_proba(final, trk, pitch=BAR_PITCH)
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
