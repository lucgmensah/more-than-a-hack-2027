"""From per-frame log-probabilities to whole possessions: a penalty Viterbi over H / A / D.

The recursion is a teammate's (github.com/Romuald-Yapo/barca-hackaton,
scripts/run_temporal_control_model.py), written with Python floats: identical output, about 10x
faster than a per-frame numpy loop. Changing team costs `switch`, entering or leaving dead ball
costs `dead`; an optional post-filter merges alive runs shorter than `min_alive` frames.

The submission uses switch 8, dead 64, min_alive 0, the best mean spellF1 over the six validation
copies of train.py (two held-out DFL matches, four Metrica copies) on the grid below.
"""

import numpy as np

import common

HOME, AWAY, DEAD = 0, 1, 2
SWITCH = (1, 2, 4, 8, 16, 32, 64)
DEAD_PENALTY = (1, 2, 4, 8, 16, 32, 64)
MIN_ALIVE = (0, 12, 25)


def viterbi(em, switch, dead_penalty, runs):
    """Team switches cost `switch`, entering or leaving Dead costs `dead_penalty`; one pass per run."""
    labels = np.full(len(em), DEAD, dtype=np.int8)
    sw, dp = float(switch), float(dead_penalty)
    for start, stop in runs:
        if stop <= start:
            continue
        rows = em[start:stop].tolist()
        n = len(rows)
        back = np.empty((n, 3), dtype=np.int8)
        b = back.tolist()
        h, a, d = rows[0]
        for t in range(1, n):
            # to H from H, A, D (first max wins, as np.argmax)
            c0, c1, c2 = h, a - sw, d - dp
            if c0 >= c1 and c0 >= c2:
                nh, bh = c0, 0
            elif c1 >= c2:
                nh, bh = c1, 1
            else:
                nh, bh = c2, 2
            c0, c1, c2 = h - sw, a, d - dp
            if c0 >= c1 and c0 >= c2:
                na, ba = c0, 0
            elif c1 >= c2:
                na, ba = c1, 1
            else:
                na, ba = c2, 2
            c0, c1, c2 = h - dp, a - dp, d
            if c0 >= c1 and c0 >= c2:
                nd, bd = c0, 0
            elif c1 >= c2:
                nd, bd = c1, 1
            else:
                nd, bd = c2, 2
            row = rows[t]
            h, a, d = nh + row[0], na + row[1], nd + row[2]
            b[t] = (bh, ba, bd)
        current = 0 if (h >= a and h >= d) else (1 if a >= d else 2)
        path = [0] * n
        path[n - 1] = current
        for t in range(n - 1, 0, -1):
            current = b[t][current]
            path[t - 1] = current
        labels[start:stop] = path
    return labels


def merge_short_alive_runs(labels, minimum_length, runs):
    """An H/A run shorter than `minimum_length` between two equal runs takes their label."""
    output = labels.copy()
    if minimum_length <= 0:
        return output
    for start, stop in runs:
        seg = output[start:stop]
        if len(seg) == 0:
            continue
        change = np.flatnonzero(np.diff(seg)) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        values = seg[bounds[:-1]].tolist()
        lengths = np.diff(bounds).tolist()
        for i in range(len(values)):
            if values[i] in (HOME, AWAY) and lengths[i] < minimum_length:
                before = values[i - 1] if i > 0 else DEAD
                after = values[i + 1] if i + 1 < len(values) else DEAD
                if before == after:
                    values[i] = before
        output[start:stop] = np.repeat(np.array(values, dtype=np.int8), lengths)
    return output


def decode(logp, decoder, runs):
    path = viterbi(logp, decoder["switch"], decoder["dead"], runs)
    return merge_short_alive_runs(path, decoder.get("min_alive", 0), runs)


def grid(logp, gt, runs):
    """[(decoder, spellF1, macroF1)] over the decoder grid."""
    out = []
    for switch in SWITCH:
        for dead in DEAD_PENALTY:
            path = viterbi(logp, switch, dead, runs)
            for min_alive in MIN_ALIVE:
                r = common.score(gt, merge_short_alive_runs(path, min_alive, runs))
                out.append(({"switch": switch, "dead": dead, "min_alive": min_alive},
                            r["spell_f1_mean"], r["macro_f1"]))
    return out


def best_decoder(grids):
    """Decoder with the best mean spellF1 over several grids (one per validation copy)."""
    mean = np.mean([[g[1] for g in grid_] for grid_ in grids], axis=0)
    i = int(np.argmax(mean))
    return grids[0][i][0], float(mean[i])
