# How we got to the submission

This folder tells the whole process: what we tried, in which order, what each step gave and what
it changed. The short version is the [technical document](../TECHNICAL_REPORT.md). The code of the
final method is in [solution/](../solution/README.md).

This branch holds only the final solution. The code of every experiment below is on branch
[`lucg`](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments), under
`experiments/`. The teammate's code is in
[Romuald-Yapo/barca-hackaton](https://github.com/Romuald-Yapo/barca-hackaton).

| report | one line |
|---|---|
| [teammate_exploration](experiments/teammate_exploration.md) | Data checks, baselines, 23 tracking heuristics, video detectors, clustering, penalty Viterbi (0.52 on the public window) |
| [exp01](experiments/exp01_hmm_unsupervised.md) | Unsupervised HMM on BAR-ATH: what the data does to EM |
| [exp02](experiments/exp02_metrica_validation.md) | A validation set with hundreds of possessions: Metrica, damaged to look like BAR-ATH |
| [exp03](experiments/exp03_improve.md) | Tuning the HMM on Metrica; the missing ball is the bottleneck |
| [exp04](experiments/exp04_bar_ath_final.md) | The Metrica-chosen HMM on BAR-ATH |
| [exp05](experiments/exp05_masked_ball.md) | Hiding the ball where a player clearly has it: Smart Tagging carries the team in the gaps |
| [exp06](experiments/exp06_abstain.md) | Abstaining (X) and semi-Markov decoding: no gain |
| [exp07](experiments/exp07_friend_port.md) | The teammate's ideas re-tuned on Metrica: the public window misleads |
| [exp08](experiments/exp08_friend_learned.md) | His exact model with learned weights: 0.55 on Metrica; the damage made faithful |
| [exp09](experiments/exp09_supervised.md) | **The submission**: supervised model on official tracking of 7 other matches |
| [exp10](experiments/exp10_video_ball.md) | A learned ball detector on the tactical video (written, needs a GPU) |
| [exp11](experiments/exp11_spell_metric.md) | Can the answers be shaped to spellF1? No |

## The score along the way

The measure is spellF1 on Metrica games the method was not fitted or tuned on. The public-window
column is a pipeline check on 17 possessions.

| step | Metrica spellF1 | public spellF1 |
|---|---|---|
| repository baseline (nearest player, vote ±50) | 0.38 | 0.326 |
| teammate's penalty Viterbi, chosen on the public window | 0.35 | 0.520 |
| our unsupervised HMM | 0.45 | 0.35 |
| teammate's model, weights learned on Metrica (exp08) | 0.55 | 0.37 |
| **supervised model trained on DFL (exp09, submitted)** | **0.70** | **0.671** |

The two columns disagree until exp09. That disagreement shaped the whole process.

## 1. Understanding the data (both of us)

**What we did.** We checked every asset before modelling:
* frame alignment of the tracking, the tags and the video;
* homography validity;
* how often the ball is missing, and where;
* the repository's baselines, reproduced exactly.

**What we found** (details in section 3 of the technical document):
* The ball is missing on 41% of the graded frames, mostly in long gaps. Some 5-minute stretches
  lose 80% of it.
* The ATD's ball-speed column is almost always empty.
* The Smart Tagging `Team` field names the defending side for `DEFENDING *` codes.
* The broadcast tracking is on its own timeline, with cuts and replays.

## 2. Two parallel tracks

**The teammate went for heuristics on the public window**
([teammate_exploration](experiments/teammate_exploration.md)). Frame-level rules did not beat the
vote:
* margins, crowding, time to reach the ball, tag priors, ball interpolation, state machines;
* video detectors that do find the ball;
* clusters that separate tracking quality rather than teams.

What did work was **time**: a long vote, then a penalty Viterbi over continuous control scores. It
reached 0.52 on the public window after a search over 360 configurations.

**We went for a decoder whose parameters are not chosen on the answer key.**
* An HMM learned by EM on the match itself ([exp01](experiments/exp01_hmm_unsupervised.md)).
  Plain EM turned "dead ball" into "ball not visible", so the missing-ball symbols were tied and
  the transitions fixed to a football prior.
* The public window has only 17 possessions, so we built a validation set from the Metrica sample
  games, damaged to look like BAR-ATH ([exp02](experiments/exp02_metrica_validation.md)). The
  repository baselines rank there as they do on BAR-ATH.
* Tuned there ([exp03](experiments/exp03_improve.md)), the HMM beat the vote by 0.12-0.14. Team
  accuracy was 85% with the ball and 62% without: **the missing ball is the bottleneck.**
* On BAR-ATH ([exp04](experiments/exp04_bar_ath_final.md)) it gave macroF1 0.75 (baseline 0.62).
  It also predicted 36-48% of the match as dead ball, against 19.6% in the public window. That
  question stayed open.
* We hid the ball where a player clearly had it ([exp05](experiments/exp05_masked_ball.md)). The
  team was then recovered 49% of the time from the tracking alone, 68% with the Smart Tagging.
* Abstaining with X lost 0.04-0.05 ([exp06](experiments/exp06_abstain.md)), even when based on the
  model's own confidence. Semi-Markov durations gained +0.01, inside the noise. **The decoder was
  not the bottleneck; the evidence was.**

## 3. Merging the two tracks

* We re-implemented the teammate's Viterbi and re-tuned it on Metrica
  ([exp07](experiments/exp07_friend_port.md)). It gave +0.10 over the baseline. His public-window
  configurations, and the vote window 120, scored *below* the baseline there.
* We then ported his exact code and learned its weights by logistic regression on Metrica
  ([exp08](experiments/exp08_friend_learned.md)). It reached **0.55**, ahead of our HMM (0.45) and
  of his published configuration (0.35).
* On the way we found that clean Metrica already looks like BAR-ATH, so our damage had been far too
  harsh. A "faithful" damage replaced it.
* Oracles on this model showed where the score was lost:

  | change | spellF1 |
  |---|---|
  | perfect dead ball | 0.62-0.64 |
  | perfect team | 0.84-0.87 |

  **The team sequence was the loss.** Turnovers were dated about 1 s late, and possessions shorter
  than 2 s were never found. Possession follows touches, and no frame-by-frame score sees a touch.

## 4. The supervised turn (exp09, the submission)

Learning touches needs labels of the right kind on many matches. The DFL open data has the
official possession and ball status on every frame of 7 matches, which is exactly how the challenge
defines its truth. The model was then built in passes, each fixing what the previous one showed
([exp09](experiments/exp09_supervised.md)):

1. **A first model fitted on one Metrica game** scored 0.81 on the other. It does not transfer: the
   Metrica ball sits on a player at every event because the ball track and the labels come from the
   same event coding. Metrica became validation only.
2. **DFL with a uniform damage.** The ball was missing on 80% of dead-ball frames everywhere, so the
   model learned "ball visible = play on".
3. **Labels harmonised to controlled possession** (spells under 1.5 s given back), matched on label
   statistics. On Metrica, 1.0 s and 1.5 s tie (0.703 and 0.701) and 2.0 s is worse (0.681).
4. **Damage by 5-minute blocks drawn from BAR-ATH,** plus team-shape features that need no ball.
5. **A separate dead-ball model that never sees ball motion.** In DFL the ball freezes during
   stoppages; in BAR-ATH the tracker follows it while it is carried back.

   | | three-class model | with the dead-ball model |
   |---|---|---|
   | Metrica spellF1 | 0.70 | 0.70 |
   | Metrica macroF1 | 0.86 | 0.88 |
   | public check macroF1 | 0.70 | 0.84 |

6. **The broadcast tracking as a second ball sensor,** aligned from the data. A labelled simulation
   on held-out DFL matches said it helps a little, so it is used.

Result: **0.70 on Metrica, 0.65-0.66 on held-out DFL matches, 0.671 / 0.839 on the public window.**
It is the first model on which the public check and the validation agree.

## 5. After the submission model

* **[exp10](experiments/exp10_video_ball.md):** a ball detector on the tactical video, trained on
  algorithmic pseudo-labels, to fill the frames where the tracking lost the ball. It is written and
  smoke-tested, but it needs a GPU, so it is not in the submission.
* **[exp11](experiments/exp11_spell_metric.md):** can the answers be shaped to the ranking metric?
  We tried inserting or deleting short possessions and shifting boundaries; they give
  +0.000 ± 0.003 out of sample. The remaining gap is boundary timing: perfect timing would give 0.92
  instead of 0.68. Only better evidence at the moment of the touch will close it.

## 6. Packaging

The final pipeline was rewritten as [solution/](../solution/README.md): the same computations, in
one folder, without the experiment scaffolding. We checked that it regenerates the submitted file
byte for byte from a fresh clone. We also retrained it from scratch and compared the result with
the committed models (see the [solution README](../solution/README.md)).
