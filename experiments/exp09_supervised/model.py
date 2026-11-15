"""Gradient-boosted frame classifier + the penalty Viterbi of exp08, shared by the exp09 scripts."""

import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
for _p in (HERE, os.path.join(EXPERIMENTS, "lib"), os.path.join(EXPERIMENTS, "exp08_friend_learned"),
           os.path.join(EXPERIMENTS, "exp02_metrica_validation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import feats  # noqa: E402
import friend  # noqa: E402
from methods import score  # noqa: E402

PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
          "min_data_in_leaf": 200, "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l2": 5.0, "num_threads": 6, "verbosity": -1, "seed": 0}
ROUNDS = 300
SWITCH = (1, 2, 4, 8, 16, 32, 64)
DEAD = (1, 2, 4, 8, 16, 32, 64)
MIN_ALIVE = (0, 12, 25)


def training_rows(trk, label, step=4, offset=0, pitch=(105.0, 68.0)):
    """Features of every `step`-th labelled frame, seen from both sides (team swap)."""
    xs, ys = [], []
    for flip in (False, True):
        x, names = feats.build(trk, flip=flip, pitch=pitch)
        y = feats.flip_labels(label) if flip else label
        sel = np.flatnonzero(y < 3)[offset::step]
        xs.append(x[sel])
        ys.append(y[sel])
        del x
    return np.vstack(xs), np.concatenate(ys), names


def fit(x, y, rounds=ROUNDS, params=None):
    return lgb.train(params or PARAMS, lgb.Dataset(x, label=y), num_boost_round=rounds)


def log_proba(model, trk, pitch=(105.0, 68.0)):
    """(N, 3) log-probabilities, averaged over the two team orders so the model is exactly symmetric."""
    x, _ = feats.build(trk, pitch=pitch)
    p = model.predict(x)
    x, _ = feats.build(trk, flip=True, pitch=pitch)
    q = model.predict(x)
    p = (p + q[:, [1, 0, 2]]) / 2.0
    return np.log(np.clip(p, 1e-6, 1.0))


def decode(logp, decoder, runs):
    path = friend.viterbi(logp, decoder["switch"], decoder["dead"], runs)
    return friend.merge_short_alive_runs(path, decoder.get("min_alive", 0), runs)


def grid(logp, gt, runs):
    """[(decoder, spellF1, macroF1)] over the decoder grid."""
    out = []
    for switch in SWITCH:
        for dead in DEAD:
            path = friend.viterbi(logp, switch, dead, runs)
            for min_alive in MIN_ALIVE:
                r = score(gt, friend.merge_short_alive_runs(path, min_alive, runs))
                out.append(({"switch": switch, "dead": dead, "min_alive": min_alive},
                            r["spell_f1_mean"], r["macro_f1"]))
    return out


def best_decoder(grids):
    """Decoder with the best mean spellF1 over several grids (one per validation match)."""
    mean = np.mean([[g[1] for g in grid_] for grid_ in grids], axis=0)
    i = int(np.argmax(mean))
    return grids[0][i][0], float(mean[i])
