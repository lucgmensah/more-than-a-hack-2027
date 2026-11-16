# Experiment 7: a teammate's temporal models, re-tuned on damaged Metrica

> Experiment report. The code is in [experiments/exp07_friend_port](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments/exp07_friend_port) on branch `lucg` (not on this branch, which holds only the final solution); the commands below run from there.

> Superseded by [exp08](exp08_friend_learned.md). exp08 ports his actual code, whose Dead
> score is mostly "ball missing", not speed. It also uses a faithful Metrica damage: the one used
> here is far too harsh.

```bash
python experiments/exp07_friend_port/run.py            # first grid, about 15 min
python experiments/exp07_friend_port/run.py --extend   # past the edges of the first grid
```

A teammate tuned two models on the 17 public possessions (report not in this repository):
a **penalty Viterbi** (continuous Home/Away/Dead scores from distance, time to the ball,
density and team centroids; switch penalty, dead penalty, minimum possession length, final
majority vote; public spellF1 0.520) and **adaptive smoothing** (vote half-width short near
boundaries, long when the ball is missing; public spellF1 0.409, macroF1 0.636). The question:
what do these ideas give once they are chosen on hundreds of possessions instead of 17?

Their code is not available, so this is a re-implementation from the report. Feature weights and
score scales are ours, declared once. Smart Tagging and video are left out (Metrica has neither).
The Dead score uses mean player speed and ball speed (both in the teammate's feature store).

Protocol: choose on the three copies of one Metrica game, report on the other game, both ways.

## Results (six damaged copies, 41% ball missing)

| method | spellF1 chosen on G1 -> G2 | chosen on G2 -> G1 | macroF1 |
|---|---|---|---|
| vote w=50 (repository baseline) | 0.251 | 0.238 | 0.655 |
| vote w=120 (best on the public window) | 0.232 | 0.232 | 0.656 |
| teammate's public configs, as published | 0.21 to 0.23 | 0.18 to 0.19 | 0.34 to 0.36 |
| **penalty Viterbi, re-tuned on Metrica** (switch 16, dead -0.6, min 100 frames) | 0.355 | 0.339 | 0.71 |
| adaptive smoothing, re-tuned on Metrica | 0.339 | 0.310 | 0.70 |
| **HMM, alpha 0.15, dwell 3 s (ours, not re-tuned here)** | **0.394** | **0.372** | **0.715** |

The optimum is inside the grid (dead -0.9 and -1.2 and switch 32 are worse), so the re-tuned
Viterbi is not held back by the grid.

## Interpretation

* **Re-tuned on Metrica, the teammate's Viterbi is a real gain over the baseline (+0.10)**, but stays
  about 0.035 below our HMM (spread 0.015-0.02). The two are close relatives: a Viterbi with
  hand-set scores versus one whose scores are learned by EM. Learning the scores helps.
* **The public-window choices do not transfer.** The published penalties (dead >= 0.3) almost never
  predict a dead ball on Metrica and score below the baseline; the vote window 120 that wins on
  the public window (0.465) loses to 50 here. Part of the gap may come from our score scale not
  being theirs, but the window result does not depend on it.
* **Adaptive smoothing does not beat a single window** once the Viterbi is strong: the decoder
  already places the boundaries.
* Not tested here: the teammate's video ball recovery (optical flow, about 85% of sampled gaps
  recovered). It targets the main remaining error (team accuracy 62% when the ball is missing) and
  can only be checked on BAR-ATH itself, by hiding ATD ball positions where they are known.
