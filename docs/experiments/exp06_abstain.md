# Experiment 6: emit X (abstain) instead of guessing

> Experiment report. The code is in [experiments/exp06_abstain](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments/exp06_abstain) on branch `lucg` (not on this branch, which holds only the final solution); the commands below run from there.

```bash
python experiments/exp06_abstain/run.py     # about 3 min
```

spellF1 forward-fills X with the previous answer, so an X in the middle of a possession could keep
it whole. Tested as pure post-processing of the exp04 configuration (alpha 0.15, dwell 3 s) on the
six damaged Metrica copies.

| policy | spellF1 | macroF1 |
|---|---|---|
| none (guess everywhere) | **0.389** | **0.712** |
| X if ball missing | 0.337 | 0.428 |
| X if ball missing for 20 s or more | 0.346 | 0.504 |
| X if decoded D and ball missing | 0.337 | 0.475 |

## Interpretation

Negative result: every abstention policy loses 0.04 to 0.05 spellF1 (spread 0.014-0.023) and a lot
of macroF1. When the ball is missing the HMM label still changes possession correctly more often than
"repeat the last answer" does, so forward-filling destroys real turnovers. Abstaining would need a
per-frame confidence (posterior), not a ball-visibility rule; not tried yet.

## 6b: abstain on the model's own uncertainty (`confidence.py`)

X on the q share of frames with the lowest smoothed posterior of the decoded state (6 copies):

| q | spellF1 | macroF1 |
|---|---|---|
| 0 | 0.373 | 0.707 |
| 5 % | 0.373 | 0.705 |
| 10 % | 0.373 | 0.700 |
| 20 % | 0.372 | 0.685 |
| 30 % | 0.363 | 0.654 |

No gain either: spellF1 is flat up to 20 % and macroF1 only drops. The Viterbi path is already
the best guess, and the uncertain frames are mostly next to boundaries where the forward-fill
is no better. Conclusion: do not emit X; answer every graded frame.

## 7: semi-Markov decoding (`semimarkov.py`, `hmm.semi_viterbi`)

Same fitted tables; the decoder gives each stay a gamma-shaped duration (shape k, mean per state,
nothing under 3 s) instead of a geometric one. Six damaged copies:

| decoder | spellF1 | macroF1 | possessions (truth ~288/2 games) |
|---|---|---|---|
| Markov, 3 s minimum (reference) | 0.373 | 0.707 | 241 |
| semi-Markov, k=1..4, mean 10-20 s | 0.376 to 0.383 | 0.69 to 0.71 | 211 to 250 |

Best +0.010 (k=2 or 4, mean 20/15 s), spread 0.015-0.027: inside the noise. Not adopted: the decoder
is not the bottleneck, as exp03 already suggested. The gain would have to come from the evidence.
