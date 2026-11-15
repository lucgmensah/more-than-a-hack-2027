# Experiment 9: a supervised model trained on official tracking of other matches

```bash
pip install -r experiments/requirements-exp09.txt     # lightgbm, the only extra dependency
python experiments/exp09_supervised/predict.py        # regenerate the submission (no fitting, 1 min)

# how the two models in models/ were produced (about 25 min, needs 3 GB of disk):
python experiments/exp02_metrica_validation/download.py  # Metrica sample games (validation only)
python experiments/exp09_supervised/dfl_io.py            # downloads and parses the DFL open data
python experiments/exp09_supervised/run3.py              # team model, validation
python experiments/exp09_supervised/run4.py              # dead-ball model, validation, BAR-ATH
```

Submission: `experiments/out/exp09/submission.csv`. No BAR-ATH label is used anywhere, for
fitting or for choosing.

## Result

| validation (never BAR-ATH) | spellF1 | macroF1 |
|---|---|---|
| 2 DFL matches held out (fit on the other 5), damaged like BAR-ATH | 0.65 / 0.66 | 0.90 / 0.88 |
| 2 Metrica games (other provider, never used to fit), 4 copies | 0.70 | 0.88 |
| *BAR-ATH public window, 17 possessions (a check, not a choice)* | *0.67* | *0.84* |

For comparison, same checks:

| method | Metrica spellF1 | public spellF1 | public macroF1 |
|---|---|---|---|
| repository baseline, vote w=50 | 0.38 | 0.33 | |
| HMM (exp04 configuration, alpha 0.1) | 0.45 | 0.35 | 0.76 |
| teammate's model, weights learned on Metrica (exp08) | 0.55 | 0.37 | 0.65 |
| **this experiment** | **0.70** | **0.67** | **0.84** |

The exp08 Metrica number was fitted on Metrica itself. This model has never seen Metrica.

## Why the earlier models stopped at 0.55

Oracles on Metrica (exp08's best model):

| scenario | spellF1 |
|---|---|
| model as it is | 0.53-0.59 |
| perfect dead ball, model's team | 0.62-0.64 |
| perfect team, model's dead ball | 0.84-0.87 |

The loss was the team sequence:
* turnovers were dated with a median error of 1 s (a perfect labelling shifted by 1 s scores 0.65);
* possessions shorter than 2 s were never found.

Possession follows touches, not proximity, and no frame-by-frame score sees a touch.

## What is new

1. **Training data with the right kind of label.** The DFL open dataset (Bassek et al. 2025, CC-BY 4.0)
   has 7 Bundesliga matches of official TRACAB tracking. Its ball frames carry the possession team
   and the ball status on every frame. The challenge describes its ground truth as exactly that.
   Loader: [dfl_io.py](dfl_io.py).
2. **Features with touches and context** ([feats.py](feats.py), 159 columns):
   * the last and next moment a player was on the ball or the ball was kicked, and for which team;
   * the same signals 0.25-3 s earlier and later;
   * how long the ball has been and will stay missing;
   * where the ball is relative to the lines;
   * team shape and block movement, usable without a ball.

   Team-swap symmetric by construction. Gradient-boosted trees (LightGBM), then the penalty Viterbi
   of exp08.
3. **A damage model calibrated without labels** ([damage.py](damage.py)). Each part matches a
   statistic of the BAR-ATH tracking:

   | BAR-ATH statistic | BAR-ATH | clean DFL | damaged DFL |
   |---|---|---|---|
   | ball velocity change over 6 frames, 90th percentile | 4.3 m/s | 7.5-8.3 | 4.1-4.5 |
   | kicks detected per minute | 10 | 49-57 | 11-16 |
   | nearest player to ball, median | 1.9 m | 1.3 m (live) | 2.0-2.4 m |
   | ball gaps per hour / median length | 85 / 10.7 s | 0 | 90-92 / 8-12 s |
   | visible ball off the pitch | 4.2% | 31% | 11-17% (not matched) |
   | ball missing per 5-minute block | 12% to 80% | 0% | drawn from BAR-ATH's blocks |

   * The ball is smoothed over 17 frames.
   * Gaps come from a model of "is the ball missing?" fitted on BAR-ATH itself from player-only
     context (no label; held-out AUC 0.86).
   * Every 5-minute block loses the ball as often as a randomly drawn BAR-ATH block.
   * 7% of player observations are dropped.
   * Two damaged copies per match.
4. **Labels harmonised to controlled possession.** DFL switches team at every touch: 9-11 possessions
   per live minute, a third shorter than 2 s. With possessions shorter than 1.5 s given back to the
   team that had the ball, its statistics match the Metrica event labels (5-6 per live minute, about
   15% under 2 s). The threshold was set on those label statistics. 1.0 s ties on Metrica (0.703 vs
   0.701) and 2.0 s is worse (0.681).
5. **Dead ball decided without ball motion** ([run4.py](run4.py)).
   * In DFL the ball is frozen while play is stopped (median speed 0.0 m/s). A three-class model
     learns "a moving ball means play is on".
   * In BAR-ATH the tracker follows the real ball, which rolls away and is carried back while the
     players walk. The three-class model gave P(dead) = 0.00 to such stoppages.
   * The dead/alive model therefore sees only the players, whether the ball is tracked, and where it
     is relative to the lines. Home/away comes from the three-class model, renormalised.

   | check | three-class model | with the dead/alive model |
   |---|---|---|
   | DFL held out, spellF1 | 0.69 | 0.66 |
   | Metrica, spellF1 | 0.70 | 0.70 |
   | Metrica, macroF1 | 0.86 | 0.88 |
   | public check, macroF1 | 0.70 | 0.84 |
   | public check, spellF1 | 0.64 | 0.67 |

   The cost on DFL is expected: there the frozen ball is a valid cue.
6. **The broadcast recording as a second ball sensor** ([broadcast.py](broadcast.py)).
   * Aligned from the data: tactical frame = broadcast frame + 4014 in the first half, + 26918 in
     the second (half-time is cut out of the TV feed).
   * It has a ball on 12.7% of the frames where the tactical tracking has none. Missing ball goes
     from 40.7% to 35.5%, and is halved in fast play.
   * Where both see the ball they are 0.9 m apart in the median. They are more than 5 m apart on 13%
     of frames: balls in the air, seen from two angles.
   * Whether filling helps was decided by a simulation with labels ([fill_sim.py](fill_sim.py)): a
     second sensor as noisy as the broadcast one, on the two held-out DFL matches. Team accuracy
     +0.8 points, macroF1 +0.4, spellF1 +0.015 and -0.003. It is used.

## What the 0.81 on Metrica was

A first model fitted on one Metrica game reached 0.81 on the other
([run_metrica.py](run_metrica.py)). It does not transfer. In the Metrica sample games the ball sits
exactly on a player at each event (within 0.3 m on 47% of frames, against 6% in BAR-ATH), because
the ball track and the labels come from the same event coding. The DFL ball is an optical track,
like BAR-ATH's.

## Caveats

* **Convention.** The BAR-ATH truth is "controlled possession" as its corrected tracking defines
  it. The harmonisation above matches label statistics, not that definition.
* **Bad stretches.** In the worst BAR-ATH blocks the ball is missing on 80% of frames. The DFL
  validation copies contain such blocks; the public window does not (11% missing).
* **Parallax.** The BAR-ATH ball is a single-camera ground projection; DFL gives the true position.
  Visible in the data (2.9% of frames beyond the far touchline against 0.6% beyond the near one),
  not modelled.
* **Remaining errors.** Possessions shorter than 5 s are found about half of the time. Turnovers
  are dated with a median error of 0.3-0.8 s.
* **Tactical video.** Not used in the submission. [exp10](../exp10_video_ball/README.md) has the code to detect
  the ball in it (a GPU job); `predict.py --video-ball` switches it on once that has been run.

## Files

| file | role |
|---|---|
| `dfl_io.py` | download and parse the DFL matches |
| `feats.py` | features |
| `damage.py` | DFL made to look like BAR-ATH |
| `model.py` | boosting and decoder helpers |
| `broadcast.py` | broadcast alignment and ball fill |
| `run_metrica.py`, `run.py`, `run2.py` | the first three passes, kept for the record |
| `run3.py`, `run4.py` | the models used |
| `fill_sim.py` | the second-sensor simulation |
| `predict.py`, `models/` | regenerate the submission |
