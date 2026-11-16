"""Shared by the exp11 scripts: the saved validation copies and the official spellF1."""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
for _p in (os.path.join(EXPERIMENTS, "exp08_friend_learned"), os.path.join(EXPERIMENTS, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common  # noqa: E402
import friend  # noqa: E402

warnings.simplefilter("ignore")
pm = common.pm
H, A, D = 0, 1, 2
FPS = 25
SWITCH, DEAD = 8, 64          # the exp09 decoder (predict.DECODER, min_alive 0)
PATH = os.path.join(EXPERIMENTS, "cache", "exp11", "val.npz")


def load():
    """{key: (truth codes, log-probabilities (N, 3), graded runs)} for the eight validation copies."""
    if not os.path.isfile(PATH):
        raise SystemExit("{} not found: run experiments/exp11_spell_metric/prep.py first".format(PATH))
    saved = np.load(PATH)
    keys = sorted({k.rsplit("_", 1)[0] for k in saved.files})
    return {k: (saved[k + "_gt"], saved[k + "_logp"].astype(np.float64), [tuple(r) for r in saved[k + "_runs"]])
            for k in keys}


def decode(logp, runs, switch=SWITCH):
    return friend.viterbi(logp, switch, DEAD, runs)


def spell_scores(gt, pred):
    """The official scorer's spellF1 (X forward-filled first, as the scorer does)."""
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)


def spellf1(gt, pred):
    return spell_scores(gt, pred)["mean_f1"]


def group(keys, prefix):
    return [k for k in keys if k.startswith(prefix)]
