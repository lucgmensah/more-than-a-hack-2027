# Experiment 11: can the answers be shaped to the ranking metric?

> Experiment report. The code is in [experiments/exp11_spell_metric](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments/exp11_spell_metric) on branch `lucg` (not on this branch, which holds only the final solution); the commands below run from there.

```bash
python experiments/exp11_spell_metric/prep.py      # once, a few minutes (needs exp09's run3.py and run4.py run first)
python experiments/exp11_spell_metric/anatomy.py   # where spellF1 is lost, and oracles
python experiments/exp11_spell_metric/gate.py      # insert / delete short possessions, cross-validated
python experiments/exp11_spell_metric/shift.py     # move one kind of boundary by a fixed delay
```

**Answer: no.** The exp09 decoder is already chosen by spellF1 on the validation data. The three
post-processing ideas below give +0.000 ± 0.003 out of sample. The submission is unchanged.

## How spellF1 works (evaluation/possession_metrics.py)

There is no sliding window. The score is computed in five steps:

1. **X is filled first.** Every X takes the previous label in its half.
2. **Possessions are cut.** A possession is a maximal run of H or a maximal run of A inside the
   graded frames. D separates possessions and is not one itself. This is done on both the truth
   and the prediction.
3. **Pairs are scored by IoU.** Each pair of a true and a predicted possession *of the same team*
   gets IoU = shared frames / frames covered by either.
4. **Pairs are matched one to one,** best IoU first.
5. **F1 is averaged over thresholds.** At each IoU threshold 0.3, 0.4, 0.5, 0.6 and 0.7:
   F1 = 2 × matched / (number predicted + number true). The score is the mean of the five.

The 12- and 25-frame tolerances in the scorer belong to the transition metric, which is reported
but does not rank.

What follows from it:
* every possession counts the same, a 1 s one as much as a 40 s one;
* a missed short possession between two of the other team merges them, so it can cost three
  possessions;
* thresholds 0.6 and 0.7 need boundaries right to a few tenths of a second.

## Data

* Two held-out DFL matches, two damaged copies each, scored with the models fitted on the other
  five matches.
* Metrica games 1 and 2, two damaged copies each, scored with the final models.
* No BAR-ATH label.

## Results (mean over the 8 copies unless stated)

### Where the score is lost (`anatomy.py`)

| IoU threshold | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| precision | 0.82-0.88 | 0.78-0.84 | 0.70-0.79 | 0.62-0.71 | 0.50-0.63 |
| recall | 0.68-0.79 | 0.65-0.75 | 0.55-0.69 | 0.51-0.63 | 0.40-0.57 |

* **Too few possessions.** We predict 251-270 possessions against 281-339 true ones. Precision is
  above recall at every threshold.
* **The missed possessions.** At IoU 0.5 they fall into two groups:
  * short ones that no predicted possession of that team touches at all, 33-64 per copy (median 1.4-2.5 s);
  * possessions merged with a neighbour of the same team, 26-48 per copy (median 4-9 s).

**Oracles.** These use the truth, so they are ceilings, not methods:

| scenario | spellF1 |
|---|---|
| model | 0.679 |
| perfect timing of the possessions it finds | **0.920** |
| every missed possession added | 0.865 |
| every invented possession removed | 0.712 |

### Insert or delete short possessions (`gate.py`)

The grid is chosen on DFL and reported on Metrica, then the other way round.

**Insert.** Short possessions found by a decode with a lower switch penalty are added when the
model is confident. Even with the truth deciding which ones to add, the gain is +0.008 (0.687 vs
0.679). The rules found in the grid:

| chosen on → reported on | rule | change |
|---|---|---|
| DFL → Metrica | switch 1, p ≥ 0.7, ≤ 2 s | +0.000 |
| Metrica → DFL | switch 2, p ≥ 0.6, ≤ 2 s | −0.002 |

**Delete.** Short possessions the model doubts are given back to the label before them.

| chosen on → reported on | rule | change |
|---|---|---|
| DFL → Metrica | p < 0.6, ≤ 2 s | −0.001 |
| Metrica → DFL | p < 0.7, ≤ 1 s | +0.001 |

### Move one kind of boundary by a fixed delay (`shift.py`)

Values are spellF1, DFL / Metrica.

| boundary | −0.48 s | −0.24 s | 0 | +0.24 s | +0.48 s |
|---|---|---|---|---|---|
| team → team | 0.637 / 0.696 | 0.651 / 0.708 | **0.649 / 0.709** | 0.646 / 0.698 | 0.631 / 0.681 |
| play → stoppage | 0.645 / 0.713 | 0.649 / 0.711 | **0.649 / 0.709** | 0.651 / 0.707 | 0.649 / 0.707 |
| stoppage → play | 0.647 / 0.704 | 0.650 / 0.707 | **0.649 / 0.709** | 0.645 / 0.707 | 0.639 / 0.701 |

No delay helps on both datasets, so there is no systematic bias to correct. The boundaries are
imprecise, not shifted.

## Reading

* **The decoder is already at the balance point.** It trades possessions added against
  possessions invented, because its penalties were chosen by spellF1 on these same validation
  copies. The short possessions it does not dare to add are mostly wrong, and the ones it doubts
  are mostly right.
* **The remaining gap is boundary timing (0.68 → 0.92) and short possessions.** Both need better
  evidence at the moment of the touch, mainly where the ball is. That is the purpose of
  [exp10](exp10_video_ball.md), not of post-processing.
