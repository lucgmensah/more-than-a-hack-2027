"""Experiment 9: a supervised model trained on official tracking of OTHER matches.

    python experiments/exp09_supervised/dfl_io.py     # download + parse the 7 DFL matches (2.4 GB)
    python experiments/exp09_supervised/run.py        # about 45 min

Training data: the DFL open dataset (7 Bundesliga matches, TRACAB tracking whose ball frames carry
the possession team and the ball status on every frame), damaged to look like the BAR-ATH automatic
tracking (damage.py, calibrated without labels). Model: gradient-boosted trees on per-frame features
with temporal context (feats.py), then the penalty Viterbi of exp08.

Validation, never on BAR-ATH:
  * leave one DFL match out (7 folds); the decoder of a fold is chosen on the other six matches;
  * the model fitted on all 7 DFL matches, run on the two Metrica games (other provider, other
    labelling convention, exp08's faithful copies).
The BAR-ATH public window is printed at the end as a check.
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
import damage  # noqa: E402
import dfl_io  # noqa: E402
import feats  # noqa: E402
import model as M  # noqa: E402
import common  # noqa: E402
from methods import period_runs, score  # noqa: E402

CACHE = os.path.join(M.EXPERIMENTS, "cache", "exp09")
OUT = os.path.join(common.OUT_DIR, "exp09")
STEP = 5
DAMAGE = {"smooth": 17, "corr_s": 15.0, "margin": 0.3, "scale": 0.85, "min_visible_s": 2.0, "min_gap_s": 1.0}
BAR_PITCH = (105.3, 68.0)
KEEP = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "alive_dead_f1", "team_acc_gt_alive")


def missing_model():
    path = os.path.join(CACHE, "missing_model.txt")
    bar = common.load_tracking()
    _, names = feats.build({"players": bar["players"][:400], "ball": bar["ball"][:400], "team": bar["team"]})
    cols = damage.player_columns(names)
    if not os.path.isfile(path):
        booster, cols, auc = damage.fit_missing_model(bar, common.graded_mask(), pitch=BAR_PITCH)
        print("missing-ball model (player-only context, no labels): held-out AUC {:.3f}".format(auc), flush=True)
        booster.save_model(path)
    return lgb.Booster(model_file=path), cols


def prepare(match, booster, cols):
    """Damaged copy of one DFL match: training rows + the damaged tracking, cached."""
    path = os.path.join(CACHE, "dfl_{}.npz".format(match))
    if not os.path.isfile(path):
        clean = dfl_io.load_match(match)
        trk = damage.damage(clean, booster, cols, seed=0, pitch=tuple(clean["pitch"]), **DAMAGE)
        x, y, _ = M.training_rows(trk, clean["label"], step=STEP, pitch=tuple(clean["pitch"]))
        np.savez(path, x=x, y=y, players=trk["players"], ball=trk["ball"], team=trk["team"],
                 period=clean["period"], label=clean["label"], pitch=clean["pitch"])
    d = np.load(path)
    return {k: d[k] for k in d.files}


def table(rows, title):
    frame = pd.DataFrame(rows)
    print("\n" + title)
    print(frame.round(3).to_string(index=False), flush=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=M.ROUNDS)
    args = parser.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    booster, cols = missing_model()
    data = {}
    for match in dfl_io.MATCHES:
        data[match] = prepare(match, booster, cols)
        obs = np.isfinite(data[match]["ball"][:, 0])
        lab = data[match]["label"]
        print("[{:.0f}s] {} ready: ball missing {:.2f} (in D {:.2f}, live {:.2f})".format(
            time.time() - t0, match, (~obs).mean(), (~obs[lab == 2]).mean(), (~obs[lab < 2]).mean()), flush=True)

    # -- leave one match out --------------------------------------------------------------------
    logp, grids = {}, {}
    for held in dfl_io.MATCHES:
        path = os.path.join(CACHE, "oof_{}.npy".format(held))
        d = data[held]
        if not os.path.isfile(path):
            x = np.vstack([data[m]["x"] for m in dfl_io.MATCHES if m != held])
            y = np.concatenate([data[m]["y"] for m in dfl_io.MATCHES if m != held])
            fold = M.fit(x, y, rounds=args.rounds)
            del x, y
            np.save(path, M.log_proba(fold, d, pitch=tuple(d["pitch"])))
        logp[held] = np.load(path)
        grids[held] = M.grid(logp[held], d["label"], period_runs(d["period"]))
        r = score(d["label"], logp[held].argmax(axis=1).astype(np.int8))
        print("[{:.0f}s] fold {}: frame argmax spellF1 {:.3f} macroF1 {:.3f}; best decoder {:.3f}".format(
            time.time() - t0, held, r["spell_f1_mean"], r["macro_f1"], max(g[1] for g in grids[held])), flush=True)
    rows = []
    for held in dfl_io.MATCHES:
        decoder, fit = M.best_decoder([grids[m] for m in dfl_io.MATCHES if m != held])
        d = data[held]
        r = score(d["label"], M.decode(logp[held], decoder, period_runs(d["period"])))
        rows.append(dict(match=held, decoder=str(decoder), **{k: r[k] for k in KEEP}))
    lomo = table(rows, "DFL, leave one match out (decoder chosen on the other six)")
    print("mean spellF1 {:.3f}  macroF1 {:.3f}".format(lomo.spell_f1_mean.mean(), lomo.macro_f1.mean()))
    lomo.to_csv(os.path.join(OUT, "dfl_lomo.csv"), index=False)
    decoder, fit = M.best_decoder([grids[m] for m in dfl_io.MATCHES])
    print("decoder for the final model: {} (out-of-fold spellF1 {:.3f})".format(decoder, fit))

    # -- final model on the 7 matches -------------------------------------------------------------
    final_path = os.path.join(CACHE, "final_model.txt")
    if not os.path.isfile(final_path):
        x = np.vstack([data[m]["x"] for m in dfl_io.MATCHES])
        y = np.concatenate([data[m]["y"] for m in dfl_io.MATCHES])
        M.fit(x, y, rounds=args.rounds).save_model(final_path)
        del x, y
    final = lgb.Booster(model_file=final_path)
    del data
    print("[{:.0f}s] final model ready".format(time.time() - t0), flush=True)

    # -- other provider, other convention: Metrica ------------------------------------------------
    rows = []
    for key in ("G1s0", "G1s1", "G2s0", "G2s1"):
        d = np.load(os.path.join(M.EXPERIMENTS, "cache", "exp08", key + ".npz"))
        trk = {k: d[k] for k in ("players", "ball", "team")}
        lp = M.log_proba(final, trk)
        runs = [tuple(r) for r in d["runs"].tolist()]
        r = score(d["gt"], M.decode(lp, decoder, runs))
        rows.append(dict(copy=key, **{k: r[k] for k in KEEP}))
    metrica = table(rows, "Metrica (event-proxy labels), model and decoder from DFL only")
    metrica.to_csv(os.path.join(OUT, "metrica_transfer.csv"), index=False)

    # -- BAR-ATH ----------------------------------------------------------------------------------
    bar = common.load_tracking()
    lp = M.log_proba(final, bar, pitch=BAR_PITCH)
    np.save(os.path.join(CACHE, "bar_logp.npy"), lp)
    codes = common.mask_to_graded(M.decode(lp, decoder, common.half_runs()))
    r = common.score_public(codes)
    graded = common.graded_mask()
    share = np.bincount(codes[graded], minlength=4)[:3] / graded.sum()
    print("\nBAR-ATH public window (17 possessions, a check): spellF1 {:.3f}  macroF1 {:.3f}  possessions {} (truth {})  "
          "a/d F1 {:.3f}  team acc {:.3f}".format(r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                                                  r["n_gt_spells"], r["alive_dead_f1"], r["team_acc_gt_alive"]))
    print("whole match: H {:.2f}  A {:.2f}  D {:.2f}".format(*share))
    common.write_submission(os.path.join(OUT, "gbdt_dfl.csv"), codes)
    print("[{:.0f}s] done".format(time.time() - t0))


if __name__ == "__main__":
    main()
