"""Step 4: would moving one kind of boundary by a fixed delay help? A systematic bias would be free spellF1.

    python experiments/exp11_spell_metric/shift.py
"""

import numpy as np

import val

KINDS = ("team -> team", "play -> stoppage", "stoppage -> play")
SHIFTS = (-12, -6, 0, 6, 12)      # frames; > 0 moves the boundary later


def shift(codes, kind, k):
    out = codes.copy()
    if k == 0:
        return out
    for c in np.flatnonzero(np.diff(codes)) + 1:
        a, b = codes[c - 1], codes[c]
        if kind == "team -> team":
            hit = a < val.D and b < val.D
        elif kind == "play -> stoppage":
            hit = a < val.D and b == val.D
        else:
            hit = a == val.D and b < val.D
        if not hit:
            continue
        if k > 0:
            out[c: c + k] = a
        else:
            out[c + k: c] = b
    return out


def main():
    data = val.load()
    scores = {}
    for key, (gt, logp, runs) in data.items():
        base = val.decode(logp, runs)
        for kind in KINDS:
            for k in SHIFTS:
                scores.setdefault((kind, k), {})[key] = val.spellf1(gt, shift(base, kind, k))
    for kind in KINDS:
        print("{:<17}".format(kind) + "  ".join(
            "{:+.2f} s: DFL {:.3f} Metrica {:.3f}".format(
                k / val.FPS,
                np.mean([v for key, v in scores[(kind, k)].items() if key.startswith("DFL")]),
                np.mean([v for key, v in scores[(kind, k)].items() if key.startswith("MET")]))
            for k in SHIFTS))


if __name__ == "__main__":
    main()
