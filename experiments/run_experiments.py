"""Run the experiment ladder and print an indicative score table.

    python experiments/run_experiments.py                 # E0..E5 at the declared alpha
    python experiments/run_experiments.py --sweep         # + alpha sensitivity for the full model
    python experiments/run_experiments.py --bootstrap 500 # confidence intervals on the public window

Ladder (each row adds one evidence channel to the previous one):

    E0  baseline_smooth50, recomputed here      sanity check of the loader against the repository
    E1  HMM  nearest                            the baseline's evidence, decoded globally
    E2  HMM  + ball_speed                       dead ball = ball at rest
    E3  HMM  + set_piece                        Smart Tagging restarts
    E4  HMM  + phase_team                       Smart Tagging team of the phase
    E5  HMM  + mean_speed                       collective movement (Lang et al. 2022)

READ THIS BEFORE QUOTING A NUMBER. The score is on the PUBLIC 5-minute window (17 possessions),
which RULES.md allows for checking the pipeline only. Nothing here selects a configuration with
it, and the ladder is a fixed, pre-declared sequence. A gap smaller than the printed bootstrap
interval is not a result. The full-match score is only known to the organisers.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import features  # noqa: E402
import hmm  # noqa: E402

LADDER = (
    ("E1_hmm_nearest", ("nearest",)),
    ("E2_hmm_+ball_speed", ("nearest", "ball_speed")),
    ("E3_hmm_+set_piece", ("nearest", "ball_speed", "set_piece")),
    ("E4_hmm_+phase_team", ("nearest", "ball_speed", "set_piece", "phase_team")),
    ("E5_hmm_+mean_speed", ("nearest", "ball_speed", "set_piece", "phase_team", "mean_speed")),
)
DEFAULT_ALPHA = 0.25
SWEEP_ALPHAS = (0.1, 0.25, 0.5, 1.0)


def baseline_smooth(tracking, window):
    """The repository's baseline (nearest player, centred majority vote), rebuilt from my loader."""
    nearest = features.nearest_team_codes(tracking)
    n = len(nearest)
    home = np.concatenate(([0], np.cumsum(nearest == 0)))
    away = np.concatenate(([0], np.cumsum(nearest == 1)))
    idx = np.arange(n)
    low, high = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    codes = np.full(n, common.D, dtype=np.int8)
    seen = (h + a) > 0
    codes[seen] = np.where(h[seen] >= a[seen], common.H, common.A)
    return codes


def check_loader(tracking):
    """E0: my smooth50 must equal evaluation/results/baseline_smooth50.csv on (almost) every frame."""
    reference = common.pm.label_codes(common.pm.read_label_csv(
        os.path.join(common.ROOT, "evaluation", "results", "baseline_smooth50.csv")))
    mine = baseline_smooth(tracking, 50)
    graded = common.graded_mask()
    agree = float(np.mean(mine[graded] == reference[graded]))
    print("E0 loader check: baseline_smooth50 recomputed == repository file on {:.4%} of graded frames"
          .format(agree))
    if agree < 0.999:
        print("   WARNING: the loader disagrees with the reference baseline; fix before trusting E1+")
    return mine


def run_one(name, names, tracking, events, runs, alpha, bootstrap, verbose=False):
    channels = features.build_channels(tracking, names, events)
    codes, tables, trans = hmm.decode_match(channels, runs, names, features.N_SYMBOLS,
                                            alpha=alpha, verbose=verbose)
    codes = common.mask_to_graded(codes)
    result = common.score_public(codes, bootstrap)
    return codes, result, trans


def row(name, result):
    low, high = result.get("spell_bootstrap_low"), result.get("spell_bootstrap_high")
    ci = "" if low is None or np.isnan(low) else "  [{:.3f}, {:.3f}]".format(low, high)
    return "{:<22} {:>8.4f} {:>8.4f} {:>5d} {:>6.1f} {:>6.3f}{}".format(
        name, result["macro_f1"], result["spell_f1_mean"], result["n_pred_spells"],
        result["mean_spell_duration_s"], result["alive_dead_f1"], ci)


def header(result):
    print("{:<22} {:>8} {:>8} {:>5} {:>6} {:>6}".format(
        "experiment", "macroF1", "spellF1", "poss", "dur", "a/d F1"))
    print("(public window: {} possessions of mean {:.1f} s; indicative only, noise >> 0.04)".format(
        result["n_gt_spells"], result["gt_mean_spell_duration_s"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--sweep", action="store_true", help="alpha sensitivity for E5")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--write", action="store_true",
                        help="also write each prediction to experiments/out/<name>.csv")
    parser.add_argument("--verbose", action="store_true", help="print EM progress")
    args = parser.parse_args()

    tracking = common.load_tracking(use_cache=not args.no_cache)
    events = common.smart_tags()
    runs = common.half_runs()
    ball_seen = float(np.mean(~np.isnan(tracking["ball"][:, 0])[common.graded_mask()]))
    print("ball visible on {:.1%} of graded frames (DATA.md says 59.3%)".format(ball_seen))

    baseline = check_loader(tracking)

    print("\nSmart Tagging: does the Team tag name the side in possession? (tracking-only check)")
    for code, (agree, n) in sorted(features.tag_agreement(tracking, events).items()):
        verdict = "n/a" if np.isnan(agree) else (
            "possession" if agree >= 0.6 else "defending (sign flipped)" if agree <= 0.4 else "ignored")
        print("  {:<30} agree {:>5}  over {:>6} frames  -> {}".format(
            code, "nan" if np.isnan(agree) else "{:.2f}".format(agree), n, verdict))

    print()
    first = common.score_public(common.mask_to_graded(baseline), args.bootstrap)
    header(first)
    print(row("E0_baseline_smooth50", first))

    results = {}
    for name, names in LADDER:
        codes, result, trans = run_one(name, names, tracking, events, runs, args.alpha,
                                       args.bootstrap, verbose=args.verbose)
        results[name] = codes
        print(row(name, result), flush=True)
        if args.write:
            common.write_submission(os.path.join(common.OUT_DIR, name + ".csv"), codes)

    if args.sweep:
        print("\nalpha sensitivity (E5). Not a selection: the declared alpha is {}.".format(
            args.alpha))
        names = LADDER[-1][1]
        for alpha in SWEEP_ALPHAS:
            _, result, trans = run_one("E5", names, tracking, events, runs, alpha, args.bootstrap)
            stays = [1.0 / (1.0 - trans[s, s]) / common.FPS for s in hmm.STATES]
            print(row("E5 alpha={}".format(alpha), result) +
                  "   learned mean stay H/A/D = {:.1f}/{:.1f}/{:.1f} s".format(*stays))


if __name__ == "__main__":
    main()
