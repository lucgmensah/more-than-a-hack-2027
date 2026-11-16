"""Train the two models on the DFL open data, validate them, and choose the decoder. No BAR-ATH label.

    python solution/metrica.py     # once: Metrica sample games (validation only)
    python solution/dfl.py         # once: DFL open data, 2.4 GB, parsed into solution/cache/dfl/
    python solution/train.py       # about 7 minutes on a laptop CPU (6 threads), a few GB of memory

Steps:
  1. From the BAR-ATH tracking (no label): the 5-minute ball-missing rates, and a model of "is the
     ball missing here?" from player-only context (damage.fit_missing_model).
  2. Two damaged copies of each of the 7 DFL matches (damage.damage), features of every 5th frame
     from both sides.
  3. Team model and dead-ball model, twice: without the two held-out matches J03WMX and J03WR9
     ("hold", for validation) and on all 7 matches ("final", for the submission).
  4. Validation copies: the two held-out DFL matches (hold models) and four damaged Metrica copies
     (final models, which have never seen Metrica). The decoder grid is scored on each copy; every
     reported row uses a decoder chosen on OTHER copies than its own.
  5. The submitted decoder is the best mean spellF1 over the six copies.

Writes solution/cache/train/{team,dead}.txt (final) and {team,dead}_hold.txt. predict.py reads
solution/models/: copy the two final files there to use them.
"""

import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import common
import damage
import decode
import dfl
import features
import metrica
import model

CACHE = os.path.join(common.CACHE_DIR, "train")
HOLD = ("J03WMX", "J03WR9")
COPIES = 2
KEEP = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "alive_dead_f1", "team_acc_gt_alive")


def prepare(match, seed, missing_model, cols, rates):
    """A damaged copy of one DFL match and its training rows, cached."""
    path = os.path.join(CACHE, "dfl_{}_{}.npz".format(match, seed))
    if not os.path.isfile(path):
        clean = dfl.load_match(match)
        pitch = tuple(clean["pitch"])
        trk = damage.damage(clean, missing_model, cols, seed=seed, pitch=pitch, block_rates=rates,
                            smooth=damage.SMOOTH)
        x, _, _ = model.training_rows(trk, clean["label"], step=model.STEP, pitch=pitch)
        np.savez(path, x=x, players=trk["players"], ball=trk["ball"], team=trk["team"],
                 period=clean["period"], label=clean["label"], pitch=clean["pitch"])
    d = np.load(path)
    return {k: d[k] for k in d.files}


def fit_team(sets, min_frames):
    n = sum(len(s["x"]) for s in sets)
    x = np.empty((n, sets[0]["x"].shape[1]), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    pos = 0
    for s in sets:
        label = model.controlled(s["label"], min_frames)
        rows = len(s["x"])
        x[pos: pos + rows] = s["x"]
        y[pos: pos + rows] = np.concatenate([label[::model.STEP], features.flip_labels(label)[::model.STEP]])
        pos += rows
    return lgb.train(model.TEAM_PARAMS, lgb.Dataset(x, label=y), num_boost_round=model.TEAM_ROUNDS)


def fit_dead(sets, cols, min_frames):
    xs, ys = [], []
    for s in sets:
        label = model.controlled(s["label"], min_frames)
        rows = len(label[::model.STEP])
        xs.append(s["x"][:rows][:, cols])               # the un-flipped half: these columns are symmetric
        ys.append((label[::model.STEP] == 2).astype(np.int8))
    return lgb.train(model.DEAD_PARAMS, lgb.Dataset(np.vstack(xs), label=np.concatenate(ys)),
                     num_boost_round=model.DEAD_ROUNDS)


def fitted(path, fit):
    if not os.path.isfile(path):
        fit().save_model(path)
    return lgb.Booster(model_file=path)


def main():
    min_frames = int(model.MIN_POSSESSION_S * common.FPS)
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)

    # 1. BAR-ATH tracking statistics (no label)
    bar = common.load_tracking()
    graded = common.graded_mask()
    rates = damage.block_missing_rates(bar, graded)
    missing_path = os.path.join(CACHE, "missing_model.txt")
    if not os.path.isfile(missing_path):
        missing, _, auc = damage.fit_missing_model(bar, graded, pitch=common.BAR_PITCH)
        print("missing-ball model: held-out AUC {:.3f}".format(auc), flush=True)
        missing.save_model(missing_path)
    missing = lgb.Booster(model_file=missing_path)
    names = model.feature_names(bar)
    player_cols, dead_cols = damage.player_columns(names), model.dead_columns(names)

    # 2. damaged DFL copies
    sets = {}
    for match in dfl.MATCHES:
        for seed in range(COPIES):
            d = sets[(match, seed)] = prepare(match, seed, missing, player_cols, rates)
            obs, lab = np.isfinite(d["ball"][:, 0]), d["label"]
            print("[{:.0f}s] {} copy {}: ball missing {:.2f} (in D {:.2f}, live {:.2f})".format(
                time.time() - t0, match, seed, (~obs).mean(), (~obs[lab == 2]).mean(), (~obs[lab < 2]).mean()),
                flush=True)

    # 3. models
    kept = [s for (m, _), s in sets.items() if m not in HOLD]
    every = list(sets.values())
    models = {
        "hold": (fitted(os.path.join(CACHE, "team_hold.txt"), lambda: fit_team(kept, min_frames)),
                 fitted(os.path.join(CACHE, "dead_hold.txt"), lambda: fit_dead(kept, dead_cols, min_frames))),
        "final": (fitted(os.path.join(CACHE, "team.txt"), lambda: fit_team(every, min_frames)),
                  fitted(os.path.join(CACHE, "dead.txt"), lambda: fit_dead(every, dead_cols, min_frames))),
    }
    print("[{:.0f}s] models ready".format(time.time() - t0), flush=True)

    # 4. validation copies
    copies = {}
    for match in HOLD:
        d = sets[(match, 0)]
        trk = {k: d[k] for k in ("players", "ball", "team")}
        truth, runs = model.controlled(d["label"], min_frames), common.period_runs(d["period"])
        copies["DFL " + match] = (model.combined(*models["hold"], dead_cols, trk, pitch=tuple(d["pitch"])), truth, runs)
    del sets, kept, every
    for key in metrica.KEYS:
        trk, gt, runs = metrica.copy(key)
        copies["Metrica " + key] = (model.combined(*models["final"], dead_cols, trk), gt, runs)
    grids = {}
    for name, (lp, truth, runs) in copies.items():
        grids[name] = decode.grid(lp, truth, runs)
        print("[{:.0f}s] {}: best decoder there {:.3f}".format(time.time() - t0, name, max(g[1] for g in grids[name])),
              flush=True)

    rows = []
    others = {"DFL J03WMX": ["DFL J03WR9"], "DFL J03WR9": ["DFL J03WMX"],
              "Metrica G1s0": ["Metrica G2s0", "Metrica G2s1"], "Metrica G1s1": ["Metrica G2s0", "Metrica G2s1"],
              "Metrica G2s0": ["Metrica G1s0", "Metrica G1s1"], "Metrica G2s1": ["Metrica G1s0", "Metrica G1s1"]}
    for name, chosen_on in others.items():
        decoder, _ = decode.best_decoder([grids[k] for k in chosen_on])
        lp, truth, runs = copies[name]
        r = common.score(truth, decode.decode(lp, decoder, runs))
        rows.append(dict(validation=name, decoder=str(decoder), **{k: r[k] for k in KEEP}))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nValidation (decoder chosen on other copies than the row it is reported on)")
    print(table.round(3).to_string(index=False))
    print("mean spellF1: DFL held-out {:.3f}, Metrica {:.3f}".format(
        table.spell_f1_mean[:2].mean(), table.spell_f1_mean[2:].mean()), flush=True)

    # 5. the submitted decoder
    decoder, fit_score = decode.best_decoder(list(grids.values()))
    print("decoder (best mean over the six validation copies): {}  spellF1 there {:.3f}".format(decoder, fit_score))
    for name in ("team", "dead"):
        state = "identical to" if same_file(os.path.join(CACHE, name + ".txt"),
                                            os.path.join(common.HERE, "models", name + ".txt")) else "differs from"
        print("{}.txt {} the committed solution/models/{}.txt".format(name, state, name))
    print("[{:.0f}s] done".format(time.time() - t0))


def same_file(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


if __name__ == "__main__":
    main()
