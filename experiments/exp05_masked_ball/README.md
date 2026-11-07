# Experiment 5: hide the ball where we know who has it

```bash
python experiments/exp05_masked_ball/run.py     # about 3 min
```

A validation set built from the BAR-ATH tracking itself, with no human label. Where a player is
within 1 m of the ball and clearly closer than any opponent, that player's team has it (97% right
on clean Metrica tracking). The ball is erased in alternate 20 s blocks (half of the graded frames,
like the real gaps) and each configuration is scored on the hidden pseudo-labelled frames
(about 12 000 per phase, two phases). A decoded dead ball counts as wrong.

| configuration | right team on hidden frames |
|---|---|
| repository baseline (vote) | 0.16 (it says dead ball when the ball is missing) |
| HMM tracking only, alpha 0.10 / 0.15 | 0.49 / 0.48 |
| + set_piece | 0.64 |
| + phase_team | 0.68 |
| + both, alpha 0.15 / 0.25 | 0.68 / 0.69 |
| + both crossed with ball visibility ("@vis") | 0.59 |

## Interpretation

* With no ball, tracking alone is at chance. **The Smart Tagging carries the whole signal in the
  gaps** (+0.2). Since the real ball is missing on 41% of the graded frames, in 150 long absences
  (65% of the missing frames sit in gaps longer than 20 s, so interpolation is useless), this is
  the strongest lever found so far.
* Crossing the tags with ball visibility, which sounded sensible, made it worse here and on the
  public window. Kept in the code as a recorded negative result.

## Limits

The pseudo-labels cover only frames with a clear holder (15% of graded frames), say nothing about
dead balls, and the tags come from the same tracking, so part of their agreement may be shared
information. It is a consistency check, not a replacement for the private ground truth.
