"""Step 3: post-processing aimed at spellF1, tuned on one validation set and checked on the other.

    python experiments/exp11_spell_metric/gate.py

* insert: take the short possessions that a decode with a lower switch penalty finds and the
  exp09 decode does not, and keep those the model is confident about (mean probability >= theta);
* delete: give the short possessions of the exp09 decode the model doubts (mean probability < theta)
  back to the label before them.
Each grid is chosen on DFL and reported on Metrica, then the other way round. The ceiling of
insertion (keep exactly the extra possessions that raise the score, using the truth) bounds what any
insertion rule could give.
"""

import itertools

import numpy as np

import val

LOW_SWITCHES = (1, 2, 4)
GRID_INSERT = list(itertools.product(LOW_SWITCHES, (0.5, 0.6, 0.7, 0.8, 0.9), (2, 4, 8)))
GRID_DELETE = list(itertools.product((0.3, 0.4, 0.5, 0.6, 0.7), (1, 2, 4, 8)))


def insert(base, low, prob, runs, theta, max_s):
    out = base.copy()
    for s0, e0, team in val.pm.possession_spells(low, runs):
        if (e0 - s0) > max_s * val.FPS or np.mean(base[s0:e0] == team) > 0.5:
            continue
        if prob[s0:e0, team].mean() >= theta:
            out[s0:e0] = team
    return out


def delete(base, prob, runs, theta, max_s):
    out = base.copy()
    for s0, e0, team in val.pm.possession_spells(base, runs):
        if (e0 - s0) <= max_s * val.FPS and prob[s0:e0, team].mean() < theta and s0 > 0:
            out[s0:e0] = out[s0 - 1]
    return out


def ceiling_insert(gt, base, low, runs):
    out = base.copy()
    for s0, e0, team in val.pm.possession_spells(low, runs):
        if np.mean(base[s0:e0] == team) > 0.5:
            continue
        trial = base.copy()
        trial[s0:e0] = team
        if val.spellf1(gt, trial) > val.spellf1(gt, base):
            out[s0:e0] = team
    return out


def main():
    raw = val.load()
    data = {}
    for key, (gt, logp, runs) in raw.items():
        decodes = {s: val.decode(logp, runs, s) for s in LOW_SWITCHES + (val.SWITCH,)}
        data[key] = (gt, np.exp(logp), runs, decodes)
    keys = sorted(data)
    sets = {"DFL": val.group(keys, "DFL"), "Metrica": val.group(keys, "MET")}

    def mean_score(fn, subset):
        return np.mean([val.spellf1(data[k][0], fn(k)) for k in subset])

    def base(k):
        return data[k][3][val.SWITCH]

    def inserted(p):
        return lambda k: insert(base(k), data[k][3][p[0]], data[k][1], data[k][2], p[1], p[2])

    def deleted(p):
        return lambda k: delete(base(k), data[k][1], data[k][2], p[0], p[1])

    print("exp09 decoder: " + "  ".join("{} {:.3f}".format(k, val.spellf1(data[k][0], base(k))) for k in keys))
    for s_low in (2, 4):
        print("ceiling, the useful extra possessions of the switch-{} decode added: mean {:.3f}".format(
            s_low, mean_score(lambda k: ceiling_insert(data[k][0], base(k), data[k][3][s_low], data[k][2]), keys)))
    for train, test in (("DFL", "Metrica"), ("Metrica", "DFL")):
        ref = mean_score(base, sets[test])
        print("\nchosen on {}, reported on {} (exp09 decoder there: {:.3f})".format(train, test, ref))
        for name, grid, make in (("insert", GRID_INSERT, inserted), ("delete", GRID_DELETE, deleted)):
            best = max(grid, key=lambda p: mean_score(make(p), sets[train]))
            got = mean_score(make(best), sets[test])
            print("  {} {}: {:.3f} ({:+.3f})".format(name, best, got, got - ref))


if __name__ == "__main__":
    main()
