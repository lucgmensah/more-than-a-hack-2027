# Teammate's exploration: data checks, heuristics, video, clustering and a penalty Viterbi

> Summary, in English, of the work in [Romuald-Yapo/barca-hackaton](https://github.com/Romuald-Yapo/barca-hackaton).
> His original reports, in French, are in `docs/experiments/` of that repository: `benchmark.md`,
> `atd-protocol.md` and `feature-clustering.md`. His scripts are in `scripts/` there.

## Why it matters for the submission

This work ran in parallel with ours, at the start of the challenge.

* It mapped the data and showed that frame-level rules were at their limit.
* It produced the **penalty Viterbi** that decodes our submission ([solution/decode.py](../../solution/decode.py)).
* It showed, by contrast, how misleading the public window is. That is why we built validation sets
  outside BAR-ATH ([exp07](exp07_friend_port.md), [exp08](exp08_friend_learned.md)).

**Every score below is on the public window: 7,500 frames and 17 possessions.** He used them as a
diagnostic. Where a configuration was *selected* on it, this is said, and the number is optimistic.

## 1. Data checks

| check | result |
|---|---|
| Tactical tracking (ATD) | 199,319 rows, `frame_id` 1 to 199,319; video frame = `frame_id` − 1 |
| Broadcast tracking | 164,488 rows, its own timeline |
| Tactical homography | 199,319 matrices, all valid |
| Broadcast homography | valid on 65.8% of frames |
| Smart Tagging | 638 intervals; frame = `round(timestamp × 25)` |
| Ball visible | 56.9% of tactical frames, 59.3% of graded frames |
| Also measured | players tracked per frame, ball gaps |
| Projection | ATD positions projected to pixels with the homography matrices, to check the alignment |

## 2. Official baselines

Nearest player, then a centred majority vote:

| run | macroF1 | spellF1 | team accuracy |
|---|---|---|---|
| raw (no vote) | 0.597 | 0.038 | 72.8% |
| vote ±25 frames | 0.620 | 0.287 | 79.1% |
| vote ±50 frames (repository reference) | 0.621 | 0.326 | 83.5% |
| vote ±125 frames | 0.581 | 0.424 | 88.5% |

## 3. Frame-level heuristics on the tracking (23 variants)

Each idea was tried raw and with votes of ±25, ±50 and ±125 frames:

* **margin:** no decision when the two teams' nearest players are within 0.75 m of each other;
* **collective:** nearest distance plus the number of players within 5 m of the ball;
* **tags:** a weak prior from the Smart Tagging `Team` field;
* **control:** estimated time for each player to reach the ball, from distance and speed;
* **short interpolation:** the ball interpolated over gaps of at most 5 frames;
* **bridging:** a dead-ball gap of at most 50 frames filled when the same team has the ball on both sides;
* **state machines:** a team change confirmed over 5 observations, the team held for 50 frames without the ball.

| best variants | macroF1 | spellF1 |
|---|---|---|
| vote ±50 frames (reference) | 0.621 | 0.326 |
| short interpolation, vote ±50 | 0.620 | 0.326 |
| tags, vote ±50 | 0.620 | 0.326 |
| collective, vote ±50 | 0.621 | 0.316 |
| bridging, vote ±50 | 0.609 | 0.342 |
| control, raw | 0.522 | 0.019 |

The control score and the state machines lowered the scores. **Conclusion:** the nearest player is
the best frame-level signal, and persistence over time is where the gain is.

## 4. Tactical video heuristics

The ball position was projected into the image with the homography. Four detectors were tried around it:

* global and local motion;
* bright-blob detection;
* template matching between neighbouring frames;
* Lucas-Kanade optical flow.

**How well they find the ball,** on 300 frames where the tracking sees it and 300 tracking gaps:

| detector | known balls found | median error | tracking gaps recovered |
|---|---|---|---|
| bright blob | 96.3% | about 19 px | 74% |
| optical flow | | 1.37 px | about 85% |

**What they give on the public window:**

| run | macroF1 | spellF1 |
|---|---|---|
| bright-blob recovery, vote ±50 | 0.614 | 0.318 |
| optical flow, vote ±50 | 0.615 | 0.335 |
| template matching, vote ±50 | 0.614 | 0.311 |
| motion, vote ±50 | 0.621 | 0.326 |

**Conclusion:** the video does contain the ball, but these detectors were not yet precise enough to
improve possessions. Our [exp10](exp10_video_ball.md) continues this line with a learned detector.

## 5. Feature store and unsupervised clustering

He built a table of 54 columns per frame:

* nearest players, distances and time to reach the ball;
* ball presence, gaps and speed;
* team centroids, spread and speed;
* Smart Tags;
* homography and tracking quality;
* the video detectors' outputs.

The table was standardised and reduced to 12 PCA components. KMeans (8, 12 and 16 clusters) and a
diagonal Gaussian mixture (8 and 12 components) were then fitted on all frames. The public labels
were looked at only afterwards.

| method | purity on the public window | ARI | NMI |
|---|---|---|---|
| GMM 8 | 0.754 | 0.260 | 0.314 |
| GMM 12 | 0.740 | 0.210 | 0.300 |
| KMeans 16 | 0.670 | 0.169 | 0.206 |
| KMeans 12 | 0.638 | 0.129 | 0.220 |
| KMeans 8 | 0.618 | 0.176 | 0.175 |

**Conclusion:** the clusters separate ball visibility, tracking quality and tag activity, not the
team in possession.

Using the clusters to gate unreliable frames:

| run | macroF1 | spellF1 |
|---|---|---|
| vote ±50 (reference) | 0.621 | 0.326 |
| cluster gate 0.55, vote ±50 | 0.701 | 0.318 |
| cluster gate 0.70, vote ±50 | 0.702 | 0.307 |
| centroid blend 0.25, vote ±50 | 0.599 | 0.315 |

Gating helped the frame-level view (dead ball) but not the possessions.

## 6. Vote-window sweep (25 to 125 frames, step 5)

| method | best window for spellF1 | spellF1 | best window for macroF1 | macroF1 |
|---|---|---|---|---|
| baseline | 120 | 0.465 | 45 | 0.623 |
| cluster gate 0.55 | 110 | 0.390 | 45 | 0.706 |
| cluster gate 0.70 | 100 | 0.423 | 35 | 0.708 |

**What it shows:** a long window removes false possessions and raises spellF1, but it absorbs
transitions and lowers macroF1. On Metrica the window 120 loses to 50 ([exp07](exp07_friend_port.md)):
this choice does not transfer.

## 7. Continuous control scores and a penalty Viterbi

`scripts/run_temporal_control_model.py` scores every frame for Home, Away and Dead:

* **Home and Away:** distance, time to reach the ball, player density, team centroids, tags and
  tracking quality;
* **Dead:** mostly "ball missing".

The scores are then decoded with:

* a 3-state Viterbi that has a team-switch penalty and a dead-ball penalty;
* a minimum possession length;
* a final majority vote.

He tested 360 configurations. The best on the public window was switch 0.5, dead 0.3, minimum
50 frames, vote 120.

| run | macroF1 | spellF1 | team accuracy | possessions (truth 17) |
|---|---|---|---|---|
| vote ±50 | 0.621 | 0.326 | 0.835 | 26 |
| vote ±120 | 0.586 | 0.465 | 0.885 | 14 |
| **Viterbi + control scores + vote 120** | 0.597 | **0.520** | 0.897 | 13 |

This was the first clear gain on spellF1, but it was selected on 17 possessions. Ported line by line
([exp08](exp08_friend_learned.md)), the same configuration scores 0.35 on Metrica, below the vote
(0.38). With weights learned on Metrica instead, the same structure reaches 0.55. The structure was
good; the selection on the window was the problem. **The Viterbi recursion is the decoder of our
submission.**

## 8. Adaptive smoothing around boundaries

The vote half-width is:

* short near transitions;
* medium in stable phases;
* long when the ball is missing or the evidence is weak.

He tested 384 configurations. The best balance was switch 1.0, dead 0.75, minimum 50, windows
35 / 15 / 50.

| run | macroF1 | spellF1 |
|---|---|---|
| vote ±50 | 0.621 | 0.326 |
| adaptive smoothing | **0.636** | 0.409 |
| global Viterbi + vote 120 | 0.597 | **0.520** |

It improved both metrics over the vote ±50. Re-tuned on Metrica, it does not beat a single window
once a strong decoder is used ([exp07](exp07_friend_port.md)).

## 9. Broadcast and audio

The broadcast video was not available when this work was done, so the audio was not tested. His plan
was to use the audio for boundaries (whistles, restarts, replays), not for the team. We later used
the broadcast *tracking* as a second ball sensor (in the submission). Audio remains untested.
