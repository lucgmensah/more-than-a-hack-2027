"""Step 2: where spellF1 is lost with the exp09 decoder, and what perfect fixes would be worth.

    python experiments/exp11_spell_metric/anatomy.py

Per copy: precision / recall / F1 at each IoU threshold, the missed and invented possessions at
IoU 0.5 by kind, and three oracles that use the truth (they are ceilings, not methods):
  * perfect timing: every predicted possession matched at IoU 0.3 gets the true boundaries;
  * invented possessions removed: every unmatched predicted possession takes the label before it;
  * missed possessions added: every unmatched true possession is written in.
"""

import numpy as np

import val


def matches(gt, pred, tau):
    """The scorer's greedy one-to-one matching (best IoU first, same team only)."""
    runs = val.pm.valid_runs(gt)
    g, p = val.pm.possession_spells(gt, runs), val.pm.possession_spells(pred, runs)
    pairs, used_g, used_p = [], set(), set()
    for iou, i, j in val.pm.spell_overlaps(g, p):
        if iou < tau:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        pairs.append((i, j, iou))
    return g, p, pairs


def missed_kinds(g, p, pairs):
    matched = {i for i, _, _ in pairs}
    kinds = {"matched": [], "no same-team overlap": [], "merged with a neighbour": [], "split in pieces": [],
             "boundaries off": []}
    for i, (s0, e0, team) in enumerate(g):
        over = [q for q in p if q[2] == team and q[1] > s0 and q[0] < e0]
        if i in matched:
            kind = "matched"
        elif not over:
            kind = "no same-team overlap"
        elif len(over) >= 2:
            kind = "split in pieces"
        else:
            ps, pe, _ = over[0]
            spans_other = any(k != i and gg[2] == team and pe > gg[0] and ps < gg[1] for k, gg in enumerate(g))
            kind = "merged with a neighbour" if spans_other else "boundaries off"
        kinds[kind].append((e0 - s0) / val.FPS)
    return kinds


def invented_kinds(g, p, pairs):
    matched = {j for _, j, _ in pairs}
    kinds = {"no same-team overlap": [], "overlaps but not matched": []}
    for j, (s0, e0, team) in enumerate(p):
        if j in matched:
            continue
        over = any(gg[2] == team and gg[1] > s0 and gg[0] < e0 for gg in g)
        kinds["overlaps but not matched" if over else "no same-team overlap"].append((e0 - s0) / val.FPS)
    return kinds


def oracles(gt, pred):
    g, p, pairs = matches(gt, pred, 0.3)
    timing = pred.copy()
    for i, j, _ in pairs:
        lo, hi = min(g[i][0], p[j][0]), max(g[i][1], p[j][1])
        timing[lo:hi] = gt[lo:hi]
    kept = {j for _, j, _ in pairs}
    removed = pred.copy()
    for j, (s0, e0, _) in enumerate(p):
        if j not in kept:
            removed[s0:e0] = removed[s0 - 1] if s0 > 0 else val.D
    found = {i for i, _, _ in pairs}
    added = pred.copy()
    for i, (s0, e0, team) in enumerate(g):
        if i not in found:
            added[s0:e0] = team
    return {"perfect timing": val.spellf1(gt, timing), "invented removed": val.spellf1(gt, removed),
            "missed added": val.spellf1(gt, added)}


def describe(kinds):
    return ", ".join("{} {} (median {:.1f} s)".format(k, len(v), np.median(v) if v else 0) for k, v in kinds.items())


def main():
    data = val.load()
    total = {}
    for key, (gt, logp, runs) in data.items():
        pred = val.decode(logp, runs)
        s = val.spell_scores(gt, pred)
        print("\n{}: spellF1 {:.3f}, {} true possessions, {} predicted".format(key, s["mean_f1"], s["n_gt"], s["n_pred"]))
        print("  " + "  ".join("IoU {:.1f}: P {:.2f} R {:.2f} F1 {:.2f}".format(t, v["precision"], v["recall"], v["f1"])
                               for t, v in s["per_iou"].items()))
        g, p, pairs = matches(gt, pred, 0.5)
        print("  true possessions at IoU 0.5: " + describe(missed_kinds(g, p, pairs)))
        print("  predicted possessions not matched at IoU 0.5: " + describe(invented_kinds(g, p, pairs)))
        row = {"model": s["mean_f1"], **oracles(gt, pred)}
        print("  " + " | ".join("{} {:.3f}".format(k, v) for k, v in row.items()))
        for k, v in row.items():
            total.setdefault(k, []).append(v)
    print("\nmean over the {} validation copies: ".format(len(data))
          + " | ".join("{} {:.3f}".format(k, np.mean(v)) for k, v in total.items()))


if __name__ == "__main__":
    main()
