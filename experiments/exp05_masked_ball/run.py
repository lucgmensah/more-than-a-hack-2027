"""Experiment 5: hide the ball on BAR-ATH where we are nearly sure who has it, and see who is recovered.

    python experiments/exp05_masked_ball/run.py

Why: the public ground truth is too small to choose between configurations, and Metrica has no
Smart Tagging, so the tagging channels have never been tested anywhere. This builds a check on the
BAR-ATH tracking itself, with no human label involved:

  pseudo-label  a graded frame where the ball is within 1 m of the nearest player and at least 1 m
                closer to him than to the nearest opponent. On clean Metrica tracking this rule is
                right 97% of the time, so it is a reasonable stand-in for "who really had it".
  hiding        in the graded frames, the ball is erased in blocks of 20 s, every other block
                (the other half of the blocks keeps the ball). This copies the real shape of the
                missing-ball problem: long absences, not flicker.
  score         on hidden pseudo-labelled frames: how often the decoded state equals the
                pseudo-label's team (a decoded D counts as wrong).

The pseudo-labels are made by an algorithm from the provided tracking, so RULES.md is respected
(no hand labels on the provided game). They are as good as the 1 m rule and no better: they say
nothing about dead balls and they only cover frames where a player clearly has the ball.
Two hiding phases (blocks starting at 0 and at 20 s) are averaged.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import common  # noqa: E402
import features  # noqa: E402
from methods import hmm_codes  # noqa: E402

BLOCK_S = 20
BLOCK, ITERATIONS, POSSESSION_S = 5, 8, 15.0
TRACKING = ("nearest", "ball_speed", "mean_speed")
CONFIGS = (
    ("tracking          a=0.15", TRACKING, 0.15, 3.0),
    ("tracking          a=0.10", TRACKING, 0.10, 3.0),
    ("+set_piece        a=0.15", TRACKING + ("set_piece",), 0.15, 3.0),
    ("+phase_team       a=0.15", TRACKING + ("phase_team",), 0.15, 3.0),
    ("+both             a=0.15", TRACKING + ("set_piece", "phase_team"), 0.15, 3.0),
    ("+both             a=0.10", TRACKING + ("set_piece", "phase_team"), 0.10, 3.0),
    ("+both             a=0.25", TRACKING + ("set_piece", "phase_team"), 0.25, 3.0),
    ("+team_prior_only  a=0.15", TRACKING + ("phase_team@prior",), 0.15, 3.0),
    ("+team_prior_only  a=0.25", TRACKING + ("phase_team@prior",), 0.25, 3.0),
    ("+both@vis         a=0.15", TRACKING + ("set_piece@vis", "phase_team@vis"), 0.15, 3.0),
    ("+both@vis         a=0.25", TRACKING + ("set_piece@vis", "phase_team@vis"), 0.25, 3.0),
)


def pseudo_labels(tracking, graded):
    """Team (0 home, 1 away) where a player clearly has the ball, else -1."""
    d_home, d_away = features._nearest_by_team(tracking)
    near = np.minimum(d_home, d_away)
    margin = np.abs(d_home - d_away)
    ok = graded & ~np.isnan(near) & (near <= 1.0) & (margin >= 1.0)
    labels = np.full(len(near), -1, dtype=np.int8)
    labels[ok] = (d_away[ok] < d_home[ok]).astype(np.int8)
    return labels


def hide_blocks(tracking, graded, phase_s):
    """Copy of the tracking with the ball erased in every other BLOCK_S block of graded frames."""
    hidden = np.zeros(len(graded), dtype=bool)
    block = BLOCK_S * common.FPS
    for start, stop in common.half_runs():
        for k, lo in enumerate(range(start + phase_s * common.FPS, stop, block)):
            if k % 2 == 0:
                hidden[lo: min(lo + block, stop)] = True
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in tracking.items() if not k.startswith("_")}
    out["ball"][hidden] = np.nan
    return out, hidden & graded


def main():
    tracking = common.load_tracking()
    events = common.smart_tags()
    runs = common.half_runs()
    graded = common.graded_mask()
    truth = pseudo_labels(tracking, graded)
    print("pseudo-labelled frames: {} ({:.1%} of graded)".format(
        int((truth >= 0).sum()), (truth >= 0).sum() / graded.sum()))

    results = {label: [] for label, *_ in CONFIGS}
    results["vote w=50 (repo baseline)"] = []
    for phase in (0, BLOCK_S):
        masked, hidden = hide_blocks(tracking, graded, phase)
        scored = hidden & (truth >= 0)
        print("phase {:>2}s: ball hidden on {:.0%} of graded frames, {} pseudo-labelled hidden frames"
              .format(phase, hidden.sum() / graded.sum(), int(scored.sum())), flush=True)

        # Reference: nearest-player vote on the masked tracking (what the repo baseline would do).
        votes = features.nearest_team_codes(masked)
        window = 50
        h = np.concatenate(([0], np.cumsum(votes == 0)))
        a = np.concatenate(([0], np.cumsum(votes == 1)))
        idx = np.arange(len(votes))
        lo, hi = np.maximum(0, idx - window), np.minimum(len(votes), idx + window + 1)
        hh, aa = h[hi] - h[lo], a[hi] - a[lo]
        vote = np.where(hh + aa == 0, common.D, np.where(hh >= aa, common.H, common.A))
        results["vote w=50 (repo baseline)"].append((vote[scored] == truth[scored]).mean())

        for label, names, alpha, dwell in CONFIGS:
            channels = features.build_channels(masked, names, events)
            codes = hmm_codes(masked, names, alpha, POSSESSION_S, runs, ITERATIONS, BLOCK,
                              channels=channels, min_dwell_s=dwell)
            results[label].append((codes[scored] == truth[scored]).mean())

    print("\nshare of hidden pseudo-labelled frames whose decoded state is the right team "
          "(chance about 0.5 for the team alone, D counts as wrong)")
    for label, values in results.items():
        print("  {:<28} {:.3f}   (phases: {})".format(
            label, np.mean(values), " ".join("{:.3f}".format(v) for v in values)))


if __name__ == "__main__":
    main()
