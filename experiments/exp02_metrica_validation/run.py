"""Experiment 2: validate the methods on Metrica games with realistic tracking damage.

    python experiments/exp02_metrica_validation/download.py      # once
    python experiments/exp02_metrica_validation/run.py           # full grid (about 15-20 min)
    python experiments/exp02_metrica_validation/run.py --quick   # game 1 only, small grid

Why: the BAR-ATH public ground truth has 17 possessions, which cannot rank anything. Two other
games with event data give hundreds of possessions each (RULES.md allows labelled data from other
games). Their tracking is clean, so it is damaged to resemble BAR-ATH (degrade.py), and the labels
are a proxy derived from events (ground_truth.py).

What it can and cannot tell us
  * It compares METHODS (smoothing window, HMM channel sets, tempering alpha) on tracking-only
    evidence. It cannot evaluate the Smart Tagging channels: these games have no Smart Tagging.
  * Absolute scores are not comparable with the BAR-ATH numbers (different ground truth).
  * Protocol: a hyper-parameter is chosen on one game and reported on the OTHER (transfer), so the
    selected value is not scored on the data that selected it.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import common  # noqa: E402
import degrade  # noqa: E402
import features  # noqa: E402
import ground_truth  # noqa: E402
import hmm  # noqa: E402
import metrica_io  # noqa: E402

pm = common.pm
OUT = os.path.join(common.OUT_DIR, "exp02")

CHANNEL_SETS = {
    "nearest+ball_speed": ("nearest", "ball_speed"),
    "nearest+ball_speed+mean_speed": ("nearest", "ball_speed", "mean_speed"),
}
ALPHAS = (0.1, 0.25, 0.5)
POSSESSION_S = (15.0, 30.0)   # declared dwell prior: mean seconds a team keeps the ball
WINDOWS = (0, 25, 50, 125)


def vote_baseline(tracking, window, periods):
    """The repository baseline (nearest player + centred majority vote), per period."""
    nearest = features.nearest_team_codes(tracking)
    out = np.full(len(nearest), common.D, dtype=np.int8)
    for period in np.unique(periods):
        sel = np.flatnonzero(periods == period)
        codes = nearest[sel]
        n = len(codes)
        if window <= 0:
            res = np.where(codes == 0, common.H, np.where(codes == 1, common.A, common.D))
        else:
            home = np.concatenate(([0], np.cumsum(codes == 0)))
            away = np.concatenate(([0], np.cumsum(codes == 1)))
            idx = np.arange(n)
            lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
            h, a = home[hi] - home[lo], away[hi] - away[lo]
            res = np.full(n, common.D, dtype=np.int8)
            seen = (h + a) > 0
            res[seen] = np.where(h[seen] >= a[seen], common.H, common.A)
        out[sel] = res
    return out


def hmm_codes(tracking, names, alpha, possession_s, periods, iterations):
    runs = [(int(np.flatnonzero(periods == p)[0]), int(np.flatnonzero(periods == p)[-1]) + 1)
            for p in np.unique(periods)]
    channels = features.build_channels(tracking, names, events=None)
    codes, _, _ = hmm.decode_match(channels, runs, names, features.N_SYMBOLS, alpha=alpha,
                                   iterations=iterations, verbose=False,
                                   possession_s=possession_s)
    return codes


def score(gt, pred, bootstrap):
    pred = np.array(pred, dtype=np.int8)
    pred[gt == common.X] = common.X
    return pm.evaluate(gt, pred, bootstrap_n=bootstrap)


def record(game, condition, method, param, result):
    return {"game": game, "tracking": condition, "method": method, "param": param,
            "macroF1": result["macro_f1"], "spellF1": result["spell_f1_mean"],
            "poss": result["n_pred_spells"], "gt_poss": result["n_gt_spells"],
            "dur_s": result["mean_spell_duration_s"], "gt_dur_s": result["gt_mean_spell_duration_s"],
            "alive_dead_F1": result["alive_dead_f1"],
            "spell_lo": result["spell_bootstrap_low"], "spell_hi": result["spell_bootstrap_high"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10, help="EM iterations per HMM fit")
    parser.add_argument("--seed", type=int, default=0, help="degradation seed")
    args = parser.parse_args()

    bar = common.load_tracking()
    vis_runs, gap_runs = degrade.reference_runs(bar, common.graded_mask())
    print("BAR-ATH ball runs: visible median {:.0f} frames, missing median {:.0f} frames".format(
        np.median(vis_runs), np.median(gap_runs)))

    games = (1,) if args.quick else (1, 2)
    alphas = (0.5,) if args.quick else ALPHAS
    dwell = (15.0,) if args.quick else POSSESSION_S
    sets = {k: CHANNEL_SETS[k] for k in (("nearest+ball_speed",) if args.quick else CHANNEL_SETS)}

    rows = []
    for game in games:
        clean = metrica_io.load_game(game)
        periods = clean["period"]
        labels = ground_truth.frame_labels(metrica_io.load_events(game), clean)
        gt = ground_truth.with_period_gap(labels, periods)
        graded = gt != common.X
        share = np.bincount(labels[labels != common.X], minlength=3) / max(1, (labels != common.X).sum())
        print("\n== Metrica game {}: {} frames, proxy GT shares H {:.2f} A {:.2f} D {:.2f}".format(
            game, len(labels), *share))
        damaged = degrade.degrade(clean, vis_runs, gap_runs, seed=args.seed)
        seen = ~np.isnan(damaged["ball"][:, 0])
        print("   degraded: ball visible {:.1%}, players tracked per frame {:.1f}".format(
            seen[graded].mean(), (~np.isnan(damaged["players"][graded, :, 0])).sum(axis=1).mean()))

        for condition, tracking in (("clean", clean), ("degraded", damaged)):
            for window in WINDOWS:
                res = score(gt, vote_baseline(tracking, window, periods), args.bootstrap)
                rows.append(record(game, condition, "vote", window, res))
            for set_name, names in sets.items():
                for alpha in alphas:
                    for possession_s in dwell:
                        res = score(gt, hmm_codes(tracking, names, alpha, possession_s, periods,
                                                  args.iterations), args.bootstrap)
                        param = "a={},p={:g}".format(alpha, possession_s)
                        rows.append(record(game, condition, "hmm:" + set_name, param, res))
                        print("   {:<9} hmm {:<32} {:<12} spellF1 {:.3f} macroF1 {:.3f} poss {:>4} (gt {})"
                              .format(condition, set_name, param, rows[-1]["spellF1"],
                                      rows[-1]["macroF1"], rows[-1]["poss"], rows[-1]["gt_poss"]),
                              flush=True)
        os.makedirs(OUT, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "results.csv"), index=False)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    print("\nAll results (degraded tracking is the one that resembles BAR-ATH):")
    print(table[["game", "tracking", "method", "param", "macroF1", "spellF1", "poss", "gt_poss",
                 "dur_s", "gt_dur_s", "alive_dead_F1"]].round(3).to_string(index=False))

    if len(games) == 2:
        print("\nTransfer protocol: choose the parameter on one game's degraded tracking, "
              "report on the other.")
        deg = table[table["tracking"] == "degraded"]
        for method in deg["method"].unique():
            for src, dst in ((1, 2), (2, 1)):
                pick = deg[(deg["method"] == method) & (deg["game"] == src)].sort_values(
                    "spellF1", ascending=False).iloc[0]
                hit = deg[(deg["method"] == method) & (deg["game"] == dst) &
                          (deg["param"] == pick["param"])].iloc[0]
                print("  {:<34} chosen on G{} -> param {:<12} ; on G{}: spellF1 {:.3f} "
                      "macroF1 {:.3f} poss {} (gt {})".format(
                          method, src, pick["param"], dst, hit["spellF1"], hit["macroF1"],
                          hit["poss"], hit["gt_poss"]))
    print("\nWrote", os.path.join(OUT, "results.csv"))


if __name__ == "__main__":
    main()
