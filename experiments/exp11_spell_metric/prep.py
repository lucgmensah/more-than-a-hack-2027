"""Step 1: the exp09 model's log-probabilities on the eight validation copies, saved once.

    python experiments/exp11_spell_metric/prep.py      # a few minutes; needs exp09's run3.py and run4.py run first

* DFL: the two held-out matches (J03WMX, J03WR9), two damaged copies each, scored with the models
  fitted on the other five matches (cache/exp09/model_v3_hold.txt and dead_v4_hold.txt). Truth: the
  official possession, harmonised to controlled possession exactly as in training (1.5 s).
* Metrica: games 1 and 2, two damaged copies each, scored with the final models of exp09 (they have
  never seen Metrica). Truth: the event-based proxy of exp02.

No BAR-ATH label is read. Output: experiments/cache/exp11/val.npz (gitignored).
"""

import os
import sys
import warnings

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(EXPERIMENTS, "exp09_supervised"), os.path.join(EXPERIMENTS, "lib")]
import feats  # noqa: E402
import run4  # noqa: E402
from methods import period_runs  # noqa: E402
from run2 import controlled, metrica_copy  # noqa: E402
from run3 import CACHE, HOLD, SMOOTH  # noqa: E402

OUT = os.path.join(EXPERIMENTS, "cache", "exp11", "val.npz")
MIN_FRAMES = 37      # 1.5 s, the harmonisation used to train exp09


def main():
    warnings.simplefilter("ignore")
    hold = (lgb.Booster(model_file=os.path.join(CACHE, "model_v3_hold.txt")),
            lgb.Booster(model_file=os.path.join(CACHE, "dead_v4_hold.txt")))
    final = (lgb.Booster(model_file=os.path.join(EXPERIMENTS, "exp09_supervised", "models", "team.txt")),
             lgb.Booster(model_file=os.path.join(EXPERIMENTS, "exp09_supervised", "models", "dead.txt")))
    store, cols = {}, None
    for match in HOLD:
        for seed in (0, 1):
            d = np.load(os.path.join(CACHE, "v3_{}_{}.npz".format(match, seed)))
            trk = {k: d[k] for k in ("players", "ball", "team")}
            if cols is None:
                _, names = feats.build({k: (v[:400] if k != "team" else v) for k, v in trk.items()})
                cols = run4.dead_columns(names)
            key = "DFL_{}_{}".format(match, seed)
            store[key + "_logp"] = run4.combined(*hold, cols, trk, pitch=tuple(d["pitch"])).astype(np.float32)
            store[key + "_gt"] = controlled(d["label"], MIN_FRAMES)
            store[key + "_runs"] = np.array(period_runs(d["period"]))
            print(key, flush=True)
    for key in ("G1s0", "G1s1", "G2s0", "G2s1"):
        trk, gt, runs = metrica_copy(key, SMOOTH)
        store["MET_" + key + "_logp"] = run4.combined(*final, cols, trk).astype(np.float32)
        store["MET_" + key + "_gt"] = gt
        store["MET_" + key + "_runs"] = np.array(runs)
        print("MET_" + key, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **store)
    print("saved", OUT)


if __name__ == "__main__":
    main()
