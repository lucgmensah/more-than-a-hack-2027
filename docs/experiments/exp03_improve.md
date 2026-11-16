# Experiment 3: what improves the score? (development loop on damaged Metrica games)

> Experiment report. The code is in [experiments/exp03_improve](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments/exp03_improve) on branch `lucg` (not on this branch, which holds only the final solution); the commands below run from there.

```bash
python experiments/exp03_improve/run.py                  # stage 1: channels and alpha (about 5 min)
python experiments/exp03_improve/run.py --stage dwell    # stage 2: minimum possession length
```

Six damaged copies (Metrica games 1 and 2, three degradation seeds each); every configuration is
scored on all six and the table gives mean and spread. The HMM runs on 0.2 s blocks (25x faster than
frame by frame: 3 s per fit instead of 75 s).

## Results (mean spellF1 over six copies; spread about 0.015)

| configuration | spellF1 |
|---|---|
| vote, window 50 (repository baseline) | 0.251 |
| HMM nearest + ball_speed + mean_speed, alpha 0.03 | 0.200 |
| same, alpha 0.05 | 0.301 |
| same, **alpha 0.10** | **0.368** |
| same, alpha 0.15 | 0.361 |
| same, alpha 0.20 | 0.305 |
| + minimum possession length 3 s, alpha 0.15 | **0.382** (0.389 with finer distance buckets) |
| + `ttb` (time to ball, Spearman-style), `count_diff` (who outnumbers whom near the ball) | 0.23 to 0.34, no better |

## Interpretation

* The HMM beats the vote by 0.12 to 0.14 on every copy, far beyond the noise.
* **alpha is a sharp knob**: 0.03 -> 0.20, 0.10 -> 0.37. A minimum possession length makes the
  result less dependent on it (alpha 0.25 with 3 s: 0.36 instead of 0.27 without).
* The explicit minimum length is worth about +0.015, inside the noise: the decoder is not the
  bottleneck.
* Finer distance buckets (1/2/4/8 m): +0.007, inside the noise. Where the ball is visible and
  close the evidence is already used up (a player within 1 m of the ball has it 97% of the time;
  beyond 2 m the nearest player is right only 55-60% of the time).
* The two control-aware channels did not help. Not a proof that they cannot: a naive-Bayes HMM
  treats channels as independent, which these are not.
* **Diagnosis of the remaining errors** (best config, game 1): team accuracy is 85% when the ball
  is visible and **62% when it is missing**, and missing-ball frames are a third of the live play.
  Dead ball versus live play accounts for most of the rest. This is where the next gain is.
