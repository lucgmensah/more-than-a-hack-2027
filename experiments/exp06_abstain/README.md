# Experiment 6: emit X (abstain) instead of guessing

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
