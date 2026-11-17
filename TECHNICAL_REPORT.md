# Who has the ball? Possession from automatic tracking, learned on other matches

**More than a Hack 2027, Phase 1: technical document.**

* Repository: [lucgmensah/more-than-a-hack-2027](https://github.com/lucgmensah/more-than-a-hack-2027), branch `submission`.
* Regenerate the submitted file: `python solution/predict.py`, about 1 minute on a laptop CPU (section 11).
* Code of the final method: [solution/](solution/README.md). The whole process, experiment by experiment: [docs/](docs/README.md).

## 1. Summary

We learn possession from matches where it is known, then apply the model to BAR-ATH.

| step | what we do |
|---|---|
| Training data | Official tracking of 7 German professional matches (DFL open data). It gives the team in possession and the ball status (in play or dead) on every frame. |
| Damage | That tracking is degraded until its statistics match the BAR-ATH automatic tracking (ball lost, smoothed, players dropped). |
| Models | Gradient-boosted trees (LightGBM) predict H, A or D per frame from 159 features. The features describe touches, the context before and after the frame, and the shape of the teams. |
| Decoding | A penalty Viterbi turns those frame predictions into whole possessions. |
| Second ball sensor | Where the tactical tracking has lost the ball, it is taken from the broadcast tracking when that one has it. The broadcast tracking is aligned to the tactical timeline from the data itself. |

No BAR-ATH label was used to train or to choose anything. All choices were made on two other kinds of matches:

* two DFL matches held out from training;
* two Metrica sample games, from another provider and never used for training.

| check | spellF1 | macroF1 |
|---|---|---|
| Metrica, 2 games, 4 damaged copies | **0.70** | 0.88 |
| DFL, 2 held-out matches, one damaged copy each | **0.65 / 0.66** | 0.90 / 0.88 |
| BAR-ATH public window, 17 possessions (pipeline check only) | 0.671 | 0.839 |
| *repository baseline (vote ±50 frames) on the same public window* | *0.326* | *0.621* |

The BAR-ATH line was only looked at after each choice had been made.

## 2. How we chose without the answer key

The public 5 minutes contain 17 possessions, which is too few to rank methods. They are also atypical. Only 19.6% of the window is dead ball. Over the whole match, BAR-ATH loses the ball on 41% of the graded frames, and the Metrica games have about 40% dead ball.

Our team learned this the hard way:

* A configuration chosen among 360 on the public window reached 0.52 there.
* Once ported and run on Metrica, the same configuration scored 0.35, below the vote baseline (0.38).
* With its weights learned on Metrica instead, it reached 0.55 (section 9).

**Protocol.** Every choice is made on validation data that the model was not fitted on. The public window is printed as a pipeline check and nothing else:

| validation set | what it is | truth |
|---|---|---|
| DFL held out | matches J03WMX and J03WR9, never used to fit the validation models, one damaged copy each | official possession and ball status |
| Metrica | sample games 1 and 2, two damaged copies each, never used to fit | proxy built from Metrica's event data ([exp02](docs/experiments/exp02_metrica_validation.md)) |

## 3. What the data told us (exploration)

### The BAR-ATH tracking

| finding | number | consequence |
|---|---|---|
| Ball missing on graded frames | 40.7% overall; from 12% to 80% per 5-minute block | Proximity alone fails on 4 frames in 10. The damage model reproduces the bad blocks. |
| Long gaps | 85 gaps per hour, median 10.7 s; 65% of missing frames are in gaps longer than 20 s | Interpolating the ball cannot fill them. |
| Ball speed column of the ATD | present on only 8% of graded frames | Speed is recomputed from positions. |
| Ball smoothness | velocity change over 6 frames, 90th percentile 4.3 m/s (clean DFL: 7.5-8.3) | The track is heavily smoothed, so kicks are blurred. |
| Kicks detected | 10 per minute (clean DFL: 49-57) | Most touches are invisible in the tracking. |
| Nearest player to the ball | median 1.9 m; within 0.3 m on only 6% of frames | Nearest-player rules are noisy. A player within 1 m has the ball 97% of the time; beyond 2 m the nearest player is right only 55-60% of the time. |
| Players | 20.5 tracked per frame; speed median 1.41 m/s, 90th percentile 3.85 m/s | Close to Metrica and DFL, so player features transfer. |
| Ball position bias | 2.9% of frames beyond the far touchline, 0.6% beyond the near one | Single-camera ground projection (parallax). Not modelled. |
| Ball visible during stoppages | the tracker follows the real ball, which rolls away and is carried back | Ball motion cannot be used to detect a dead ball (section 4.5). |

### The other assets

| asset | finding |
|---|---|
| Frame alignment | ATD `frame_id` = video frame + 1; Smart Tagging at `round(timestamp × 25)`. Checked before anything else. |
| Smart Tagging | 638 intervals. The `Team` field of attacking codes agrees with the team nearest the ball 62-76% of the time. For `DEFENDING *` codes it agrees only 24-37%: the field names the defending side. |
| Homography | Valid on 100% of tactical frames and 65.8% of broadcast frames. |
| Broadcast tracking | 164,488 frames on its own timeline, with cuts and replays. It aligns to the tactical timeline at +4,014 frames in the first half and +26,918 in the second. These offsets were found by matching the two ball tracks (FFT cross-correlation), not by hand. |
| Broadcast ball | It has a ball on 12.7% of the frames where the tactical tracking has none. Where both see it, they are 0.9 m apart in the median, and more than 5 m apart on 13% of frames (balls in the air, seen from two angles). |
| Tactical video | The ball is visible on most frames where the tracking lost it: 4-8 pixels, white on green. Simple detectors find it (section 9) but were not accurate enough to help. |

### The external datasets: traps found

* **Metrica sample games.** The ball sits exactly on a player at every event (within 0.3 m on 47% of frames, against 6% in BAR-ATH). The ball track and the labels come from the same event coding. A model fitted on one Metrica game scored 0.81 on the other, but it learned this artefact and does not transfer. We use Metrica for validation only.
* **DFL ball while play is stopped.** The ball is frozen (median speed 0.0 m/s), so a model learns "a still ball means a stoppage". On BAR-ATH that model gave a dead-ball probability of 0.00 to real stoppages. This is why dead ball has its own model that never sees ball motion.
* **DFL possession changes at every touch.** It gives 9-11 possessions per live minute, a third of them shorter than 2 s. Metrica's event labels give 5-6, about 15% shorter than 2 s. We harmonise this (section 5).
* **Clean Metrica already looks like BAR-ATH.** It has the same player speeds, 39-41% ball missing, and the same share of slow frames. Our first damage model (exp02-07) was far too harsh and made every number pessimistic. The "faithful" damage of exp08 fixed it.

## 4. The approach

```
BAR-ATH tactical tracking ──┐
                            ├─ missing ball filled from the aligned broadcast tracking (40.7% -> 35.5% missing)
BAR-ATH broadcast tracking ─┘
        │
        ├─ 159 features per frame (touches, context ±3 s, team shape)
        │        ├─ team model  -> P(H), P(A)        (LightGBM, 3 classes, trained on damaged DFL)
        │        └─ dead model  -> P(D)              (LightGBM, binary, players and lines only)
        │
        └─ log-probabilities -> penalty Viterbi (team switch 8, dead in/out 64) -> H / A / D per graded frame
```

### 4.1 Training data with the right kind of label

The challenge describes its truth as follows:

> a frame is D where the corrected tracking marks the ball dead, and otherwise takes the team the
> tracking names as owning it.

The DFL open dataset (Bassek et al., 2025) carries exactly that on every frame of 7 matches: `BallPossession` and `BallStatus` in its official positional data. That makes about 1 million labelled frames of the right kind. Loader: [dfl.py](solution/dfl.py).

### 4.2 Damage calibrated without labels

[damage.py](solution/damage.py) makes DFL look like BAR-ATH. Each step matches one statistic of the BAR-ATH tracking, measured without any label:

| statistic | BAR-ATH | clean DFL | damaged DFL |
|---|---|---|---|
| ball velocity change over 6 frames, 90th percentile | 4.3 m/s | 7.5-8.3 | 4.1-4.5 |
| kicks detected per minute | 10 | 49-57 | 11-16 |
| nearest player to ball, median | 1.9 m | 1.3 m (live) | 2.0-2.4 m |
| ball gaps per hour / median length | 85 / 10.7 s | 0 | 90-92 / 8-12 s |
| ball missing per 5-minute block | 12% to 80% | 0% | drawn from BAR-ATH's blocks |

How each step works:

* **Smoothing.** The ball is smoothed over 17 frames.
* **Where the ball goes missing.** Gaps are placed by a model of "is the ball missing here?". That model is fitted on BAR-ATH from player-only context; it needs no possession label (held-out AUC 0.86).
* **How much goes missing.** Each 5-minute block loses as much ball as a randomly drawn BAR-ATH block.
* **Players.** 7% of player observations are dropped.
* **Copies.** Each match gets two damaged copies with different random draws.

### 4.3 Features that see touches

[features.py](solution/features.py) builds 159 columns from team, position and speed only. No player identities are used. Our oracle studies (section 9) showed why touches matter: possession follows touches, not proximity. Frame-by-frame scores never see a touch, so the features look for them explicitly:

* **Touches.** The last and next moment a player was on the ball (within 1 m) or the ball was kicked (its velocity changed sharply near a player), and for which team.
* **Context.** The same signals 0.25 to 3 s earlier and later. The labels are produced offline, so the future is available.
* **Ball missing.** How long the ball has been missing and will stay missing.
* **Lines.** Where the ball is relative to the touchlines and goal lines.
* **Team shape.** Centroids, spread and which way each block moves. These work without a ball.

The features are symmetric by construction: swapping the teams negates or swaps the columns. Each training frame is seen from both sides, and at prediction time both views are averaged, so the model cannot prefer a side.

### 4.4 Team model

| setting | value |
|---|---|
| model | LightGBM, 3 classes (H, A, D) |
| rounds / learning rate / leaves | 220 / 0.08 / 63 |
| training rows | every 5th frame of 7 matches × 2 damaged copies × 2 sides (802,128 rows) |

Its home/away output, renormalised over the two teams, gives P(H | alive).

### 4.5 Dead-ball model without ball motion

* **The problem.** In DFL a moving ball means play is on. In BAR-ATH it does not: during a stoppage the ball is chased, kicked back and carried while the players walk.
* **The fix.** A separate binary model decides dead or alive. It sees only the players (speeds, share of slow players, block movement), whether the ball is tracked, and where the ball is relative to the lines. Code: [model.py](solution/model.py).
* **The cost.** It is lower on DFL, where the frozen ball was a valid cue: 0.69 → 0.66.
* **The gain.** It is equal on Metrica spellF1 (0.70) and better on Metrica macroF1: 0.86 → 0.88.

### 4.6 The broadcast tracking as a second ball sensor

The tactical ball is missing on 40.7% of graded frames. Filling it from the aligned broadcast tracking ([broadcast.py](solution/broadcast.py)) brings this to 35.5%, and halves it in fast play.

**Whether to use it was decided with labels, but not BAR-ATH ones** ([fill_sim.py](solution/fill_sim.py)). On the two held-out DFL matches, we simulated a second sensor as noisy as the broadcast one and filling as much:

| change on held-out DFL | value |
|---|---|
| team accuracy | +0.8 points |
| macroF1 | +0.4 points |
| spellF1 | +0.015 and −0.003 |

### 4.7 Decoder

The penalty Viterbi comes from a teammate's model ([decode.py](solution/decode.py)), run on our log-probabilities:

* changing team costs 8;
* entering or leaving dead ball costs 64;
* no minimum length is imposed.

These values were chosen on a grid (switch 1-64 × dead 1-64 × minimum 0/12/25 frames) by mean spellF1 over the six validation copies.

### 4.8 Every number and where it was chosen

| parameter | value | chosen on |
|---|---|---|
| training label harmonisation | 1.5 s | Metrica label statistics; 1.0 s ties (0.703 vs 0.701), 2.0 s is worse (0.681) |
| damage constants | table 4.2 | BAR-ATH tracking statistics (no label) |
| boosting parameters | section 4.4 | set once, checked on DFL held out |
| decoder | switch 8, dead 64, no minimum | DFL held out + Metrica |
| broadcast fill | on | simulation on DFL held out |
| alignment offsets | +4,014 / +26,918 frames | ball tracks of BAR-ATH (no label) |

## 5. The possession rules we implemented

The labels are learned. The rules are the definition of the training labels, plus what the decoder adds.

* **H or A.** The team in possession, as the official tracking of the training matches names it. It is harmonised to *controlled* possession: when one team has the ball, a spell of the other team shorter than 1.5 s is given back to the first team, if that team gets the ball back or play stops right after. In practice:
  * **loose balls and deflections:** a touch by the defending side that does not lead to a controlled spell does not change possession;
  * **challenges:** a won tackle starts a new possession only when the winner keeps the ball for at least 1.5 s;
  * **real turnovers:** a spell shorter than 1.5 s that the other team then keeps is a real turnover and is kept.
* **D.** The ball is out of play, as the official ball status marks it: from the moment the ball leaves the pitch or the referee stops play, until the restart. On BAR-ATH this is decided from the players and the lines, never from ball motion (section 4.5).
* **Restarts.** Possession at a restart is the team that takes it: the first alive frame after D carries the team of the training label. A restart is not treated as a special case.
* **Switches.** A team change must be supported by evidence worth 8 log-probability units, and a switch between play and stoppage by 64 units. There is no fixed minimum possession length; the penalties alone remove flicker.
* **X is never emitted** on a graded frame. Abstaining was tested and costs 0.04-0.05 spellF1 (section 9). Ungraded frames (before kick-off, half-time, after the final whistle, as published in DATA.md) are X.

## 6. Assets: used, tried and dropped

| asset | in the submission | what we tried | why |
|---|---|---|---|
| Tactical tracking (ATD) | **yes**, all features | everything | the main source |
| Broadcast tracking (ATD) | **yes**, ball only | ball fill, aligned from the data | gives back 12.7% of the missing ball; a labelled simulation says it helps |
| Smart Tagging | no | HMM channels (exp01-05), tag priors (teammate) | Recovers the team when the ball is missing (49% → 68% on a pseudo-labelled check, exp05). But it lowers macroF1 on the public window, and no training or validation match has tags, so its weight cannot be learned or checked. Left out rather than set by hand. |
| Tactical video | no | motion, bright blob, template matching, optical flow (teammate); a learned ball detector ([exp10](docs/experiments/exp10_video_ball.md)) | The heuristics found the ball but not precisely enough to raise spellF1. The learned detector is written and smoke-tested but needs a GPU run we could not do in time. Its code is on branch `lucg`, not in this solution. |
| Homography | no (used by exp10 only) | projecting the tracked ball into the image, snapping to the visible ball | only needed for the video |
| Broadcast video and audio | no | not tried | whistles and restarts could mark dead-ball boundaries; no time |
| Public ground truth (5 minutes) | no | pipeline check only | rules, and too few possessions (section 2) |

## 7. Training data

None of it comes from the provided game.

| dataset | source and licence | who labelled it, how | how much | use |
|---|---|---|---|---|
| DFL open data | Bassek, Rein, Weber, Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data. CC-BY 4.0. figshare 10.6084/m9.figshare.28196177, mirror huggingface.co/datasets/pysport/idsse-data | The German league's official data provider. Optical tracking with the possession team and ball status on every frame, as delivered in the official positional data. Not labelled by us. | 7 matches, about 1 million frames | Training. Two matches held out for validation, then all 7 for the final models. |
| Metrica sample games 1 and 2 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), anonymised, free use with acknowledgement | Metrica's event coding. H/A/D per frame is derived from the events by our rules ([metrica.py](solution/metrica.py)): possession starts at a recovery, dead ball runs from a stoppage event to the next played ball. | 2 matches, hundreds of possessions | Validation only |
| BAR-ATH | provided | **no label used** | | The "is the ball missing?" model of the damage step and the alignment use BAR-ATH tracking only. |

No hand labels were made by our team, on any game.

## 8. Results

### Validation (never BAR-ATH)

| method | Metrica spellF1 | DFL held-out spellF1 | public check spellF1 / macroF1 |
|---|---|---|---|
| repository baseline, vote ±50 | 0.38 | | 0.326 / 0.621 |
| teammate's Viterbi, configuration chosen on the public window | 0.35 | | 0.520 / 0.596 |
| our unsupervised HMM (exp04 configuration) | 0.45 | | 0.35 / 0.76 |
| teammate's model, weights learned on Metrica (exp08) | 0.55 | | 0.37 / 0.65 |
| **submission (exp09)** | **0.70** | **0.65-0.66** | **0.671 / 0.839** |

The exp08 Metrica number was fitted on Metrica itself; exp09 has never seen Metrica.

### Where the submission stands (exp11, validation copies)

| IoU threshold | 0.3 | 0.5 | 0.7 |
|---|---|---|---|
| precision | 0.82-0.88 | 0.70-0.79 | 0.50-0.63 |
| recall | 0.68-0.79 | 0.55-0.69 | 0.40-0.57 |

We predict slightly too few possessions: 251-270 against 281-339. The possessions we miss are short (1.4-2.5 s median) or merged into a neighbour.

## 9. How we got here

The whole process is told step by step in [docs/README.md](docs/README.md), with one report per
experiment in [docs/experiments/](docs/experiments). This section is the summary.



### Teammate's exploration (scores on the public window, 17 possessions; [report](docs/experiments/teammate_exploration.md))

| what | result | lesson |
|---|---|---|
| Data QA: assets, frame alignment, homography validity, ball visibility (56.9% of frames) | pipeline facts of section 3 | |
| Official baselines: raw, vote ±25 / ±50 / ±125 | spellF1 0.04 / 0.29 / 0.33 / 0.42 | Temporal persistence is the main lever. |
| 23 tracking variants: ambiguity margin, players near the ball, Smart Tagging prior, time to reach the ball, short ball interpolation, bridging short D, state machines | none better than vote ±50 in a robust way | Frame-level rules are at their limit. |
| Tactical video: global and local motion, bright-blob ball detection, template matching, optical flow | blob detection finds 96% of known balls (median error 19 px); optical flow 1.4 px; public spellF1 ≤ 0.335 | The video carries the ball, but heuristics do not turn it into better possessions. |
| 54-column feature store, KMeans/GMM clustering | clusters separate ball visibility and tracking quality, not teams; cluster gating gives macroF1 0.70 but spellF1 0.31 | |
| Vote-window sweep (25-125) | best 120 on the window (0.465) | Did not transfer: loses to ±50 on Metrica (exp07). |
| Continuous control scores + penalty Viterbi, 360 configurations | **0.520** public | Chosen on 17 possessions: 0.35 on Metrica. The structure was good (exp08). |
| Adaptive smoothing around boundaries, 384 configurations | 0.409 / 0.636 public | Better on both metrics than vote ±50, but not once a strong decoder is used (exp07). |

### Our experiments

| # | question | result | what it changed |
|---|---|---|---|
| [01](docs/experiments/exp01_hmm_unsupervised.md) | Unsupervised HMM (EM + Viterbi) with tracking and Smart Tagging channels | Works once missing-ball symbols are tied across states and transitions fixed to a football prior | The ATD ball speed is unusable; plain EM turns D into "ball not visible". |
| [02](docs/experiments/exp02_metrica_validation.md) | Can damaged Metrica games stand in for BAR-ATH? | The repository baselines keep the same order and size as on BAR-ATH | A validation set with hundreds of possessions |
| [03](docs/experiments/exp03_improve.md) | Which channels and settings help the HMM? | HMM 0.37-0.39 vs vote 0.25 (harsh damage); team accuracy 85% with the ball, 62% without | The missing ball is the bottleneck. |
| [04](docs/experiments/exp04_bar_ath_final.md) | The Metrica-chosen HMM on BAR-ATH | public macroF1 0.751 (baseline 0.621), spellF1 0.30 | It predicts 36-48% D; the window has 19.6%. |
| [05](docs/experiments/exp05_masked_ball.md) | Hide the ball where a player clearly has it: who recovers the team? | tracking only 0.49, + Smart Tagging 0.68 | The tags carry signal in the gaps (section 6). |
| [06](docs/experiments/exp06_abstain.md) | Emit X instead of guessing? Semi-Markov durations? | X: −0.04 to −0.05; semi-Markov: +0.01 (noise) | Never abstain; the decoder is not the bottleneck. |
| [07](docs/experiments/exp07_friend_port.md) | Teammate's Viterbi and adaptive smoothing, re-tuned on Metrica | +0.10 over the baseline; his public configurations below the baseline | The public window misleads. |
| [08](docs/experiments/exp08_friend_learned.md) | His exact model, weights learned by logistic regression on Metrica; damage made faithful | 0.55 on Metrica (his public configuration 0.35, our HMM 0.45) | Learning the scores beats hand-setting them. The oracles show the team sequence is what is lost: a perfect team decision scores 0.84-0.87, a perfect dead ball 0.62-0.64. |
| [09](docs/experiments/exp09_supervised.md) | Supervised model on official tracking of other matches | **0.70 Metrica, 0.65-0.66 DFL, 0.671 public** | **The submission** |
| [10](docs/experiments/exp10_video_ball.md) | Learned ball detector on the tactical video (algorithmic pseudo-labels) | Written; toy run puts the peak on the ball in 92% of crops; full run needs a GPU | Not in the submission |
| [11](docs/experiments/exp11_spell_metric.md) | Can post-processing aimed at spellF1 help? | Insert or delete short possessions, shift boundaries: +0.000 ± 0.003 out of sample | Submission unchanged (section 10) |

exp09 itself was built in passes, each fixing what the previous one showed (section 4 of
[docs/README.md](docs/README.md)):

1. a Metrica-trained model that did not transfer (0.81, an artefact: section 3);
2. DFL with a uniform damage (the model learned "ball visible = play on");
3. labels harmonised to controlled possession;
4. damage by 5-minute blocks drawn from BAR-ATH, plus team-shape features;
5. a dead-ball model without ball motion, and the broadcast ball.

## 10. Limitations, and what we would do with more time

* **Boundary timing.** Turnovers are dated with a median error of 0.3-0.8 s, and possessions shorter than 5 s are found about half of the time. On the validation copies:
  * with perfect timing of the possessions we find, the score would be 0.92 instead of 0.68;
  * adding every missed possession would give 0.87;
  * removing every invented one would give only 0.71.

  Metric-aware post-processing does not close this gap out of sample (exp11). The fix is better evidence at the moment of the touch.
* **The ball in the video.** The most direct way to get that evidence. The detector of [exp10](docs/experiments/exp10_video_ball.md) is written (branch `lucg`). It needs a GPU run, and must pass the label-free checks it prints, before it can be used.
* **Bad stretches.** In the worst 5-minute blocks the ball is missing on 80% of frames. There the team decision rests on team shape. The validation copies contain such blocks; the public window does not (11% missing).
* **Share of dead ball.** The submission labels 43% of the graded frames D. That is close to the
  usual share of stoppage time in professional football, but far from the public window's 19.6%. We
  cannot check it without the full-match truth: the window is the start of the match and is not
  typical.
* **Label convention.** The BAR-ATH truth is controlled possession from its corrected tracking. We match DFL's labels to it through label statistics (the 1.5 s rule), not through the definition itself.
* **Parallax and balls in the air.** The BAR-ATH ball is a single-camera ground projection, and its far-side bias is visible in the data. Neither is modelled.
* **Smart Tagging and audio.** Unused. The tags help in ball gaps (exp05), but without tagged training matches their weight cannot be learned or checked.
* **Validation is a proxy.** DFL and Metrica are damaged to look like BAR-ATH, statistic by statistic, but they are not BAR-ATH. Only the organisers know the full-match score.

## 11. Reproduction

Requirements: Python 3.13 (tested), the pinned packages of [requirements-submission.txt](requirements-submission.txt), and the match bundle unpacked into `data/` as the challenge README says (or `MTAH_DATA_DIR` pointing to it). No GPU, no internet.

```bash
git clone -b submission https://github.com/lucgmensah/more-than-a-hack-2027
cd more-than-a-hack-2027
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-submission.txt
python solution/predict.py                             # about 1 minute, CPU
```

Checked before submission: a fresh clone of this branch, a new virtual environment with only these
packages, and no cache gave the same file, byte for byte, in 1 minute.

| output | value |
|---|---|
| file | `solution/out/submission.csv` |
| validator line | `285 possessions of mean 12.8 s` |
| public check line | spellF1 0.671, macroF1 0.839 |
| sha256 | `147033f58934bdb3e0bda5055156177de25b6428440eb55623a831fc59ca2bce` |

Nothing is fitted at that step. The two models are committed in [solution/models/](solution/models).

How they were trained: about 7 minutes on a laptop CPU once the 3 GB of DFL and Metrica data are downloaded (the scripts fetch them). A retraining from an empty cache gives the same two models, byte for byte:

```bash
python solution/metrica.py     # Metrica sample games (validation)
python solution/dfl.py         # DFL open data, parsed once
python solution/train.py       # damaged copies, both models, validation table, decoder
```

The generated files go to `solution/cache/` and `solution/out/`, which are gitignored. They contain frame-level derived tracking and full-match labels, which must not be published.

## References

* M. Bassek, R. Rein, H. Weber, D. Memmert (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*. CC-BY 4.0.
* Metrica Sports sample data, github.com/metrica-sports/sample-data.
* G. Ke et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS*.
* L. Xie, P. Xu, S.-F. Chang, A. Divakaran, H. Sun (2004). Structure analysis of soccer video with domain knowledge and hidden Markov models. *Pattern Recognition Letters*.
