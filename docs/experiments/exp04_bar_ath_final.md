# Experiment 4: the Metrica-chosen configuration, run once on BAR-ATH

> Experiment report. The code is in [experiments/exp04_bar_ath_final](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments/exp04_bar_ath_final) on branch `lucg` (not on this branch, which holds only the final solution); the commands below run from there.

```bash
python experiments/exp04_bar_ath_final/run.py
```

Writes three submissions to `experiments/out/exp04/` (validated with the official validator).
Parameters come from experiment 3 (alpha 0.15, minimum possession 3 s, 0.2 s blocks), never from
BAR-ATH. The public 5-minute window is printed as a pipeline check only; it has 17 possessions.

| run | macroF1 | spellF1 | possessions (truth: 17) | a/d F1 |
|---|---|---|---|---|
| baseline smooth50 (repository) | 0.621 | 0.326 | 26 | 0.899 |
| baseline smooth125 (repository) | 0.581 | 0.424 | 16 | 0.899 |
| T: tracking only | **0.751** | 0.300 | 23 | 0.914 |
| TS: + set_piece + phase_team | 0.677 | 0.311 | 10 | 0.917 |
| TSV: tag channels crossed with ball visibility | 0.602 | 0.280 | 13 | 0.886 |

## How to read it

* **macroF1 +0.13 for the tracking-only HMM** is consistent with the Metrica result (+0.06 to
  +0.08 there). spellF1 is in the noise on 17 possessions; smooth125 is known to look best here
  and to lose on the full match (EVALUATION.md).
* Adding the Smart Tagging lowers macroF1 here, yet experiment 5 shows it is what makes the model
  work when the ball is missing. The two checks disagree; the 5-minute window is mostly
  ball-visible play, where the tags add noise, and has far fewer frames than experiment 5. Neither
  is the full-match score. **Do not pick a configuration on this table.**
* The D share predicted over the whole match is 36-48%, while the public window has 19.6%. The
  declared prior (a stoppage about every 22 s, lasting 12 s) comes from event-coded football and
  probably over-predicts D for this challenge's definition (dead according to corrected tracking).
  This is an open, high-impact question.
