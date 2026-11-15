# Experiment 8: the teammate's model, with its numbers learned on Metrica

```bash
python experiments/exp08_friend_learned/run.py            # about 25 min with 6 worker processes
python experiments/exp08_friend_learned/run.py --quick    # one copy per game, tiny grid, ~1 min
```

A teammate's model ([friend.py](friend.py), ported from his repository
`Romuald-Yapo/barca-hackaton`) scores every frame for Home, Away and Dead with weighted sums of
hand-made features:

* **team:** nearest player, time to reach the ball, a "collective" score and team centroids;
* **dead:** ball missing, tracking incomplete, ambiguity.

It then decodes with a 3-state Viterbi (a team-switch penalty and a Dead penalty), merges short
possessions and applies a centred majority vote. He chose its 13 numbers by hand and by a search
on the 17 public possessions. Here they are learned on the annotated Metrica games instead.

**Port check.** His published configuration with his Smart Tagging terms gives public spellF1
**0.520**, exactly his reported number. One term is left out: the unsupervised cluster confidence,
weight 0.20.

## The development set had to be fixed first

Measured on the tracking only, clean Metrica already looks like BAR-ATH:

| | BAR-ATH (graded) | Metrica 1 | Metrica 2 |
|---|---|---|---|
| player speed, median / p90 (m/s) | 1.41 / 3.85 | 1.35 / 3.69 | 1.36 / 3.60 |
| frames with mean player speed < 0.7 m/s | 9.7% | 9.8% | 8.9% |
| ball missing | 40.7% | 39.1% | 41.0% |
| player position jitter (median 2nd difference) | 0.001 m | 0.002 m | 0.003 m |
| tracked players per frame | 20.5 | 22 | 22 |

The degradation used in exp02-07 (`degrade.degrade`) was therefore far too harsh:
* it added 0.25 m and 0.5 m of noise before recomputing speeds, so no frame was slow any more;
* it added BAR-ATH-sized ball gaps on top of Metrica's own, giving 65% missing.

`degrade.degrade_faithful` (used here) changes three things:
* It keeps Metrica's positions, speeds and ball gaps.
* It drops 7% of player observations, giving 20.5 players per frame.
* It adds short open-play ball losses (about 1 s) where BAR-ATH loses the ball more often than
  Metrica at the same player speed. Ball missing is then 42-43% of frames.

Every absolute number in exp02-07 is pessimistic for that reason. Rankings can change too: see the
HMM below.

## What is learned

| variant | team / dead weights | decoder (switch, dead, min length, vote) |
|---|---|---|
| his published config | his, by hand | his, chosen on the public window |
| hand + search | his, by hand | grid on Metrica |
| logreg | multinomial logistic regression on Metrica frames, his features, his symmetric form (H = w.f, A = -w.f, D = v.g + b) | grid on Metrica |
| logreg + speeds | same; Dead may also use mean player speed, ball still, ball fast (in his feature store, not in his emission) | grid on Metrica |
| direct | every number optimised for spellF1 by a (1+14) evolution strategy, 30 generations, from "hand + search" | (same search) |

The logistic-regression emission is a log-probability up to a per-frame constant, so the Viterbi
decodes the most likely sequence under a fixed-penalty Markov chain. The grid uses switch
0.25-512 and dead 0-256; every chosen decoder is inside it.

Protocol: fit and choose on the two copies of one game, report on the two copies of the other, both
ways. Final models are fitted on both games and run once on BAR-ATH.

## Results on Metrica (game not used to fit or choose)

| method | G1 -> G2 | G2 -> G1 | mean spellF1 | macroF1 | possessions (truth 288) |
|---|---|---|---|---|---|
| **logreg + speeds** | 0.586 | 0.523 | **0.554** | **0.832** | 264 |
| logreg (his features only) | 0.582 | 0.515 | 0.549 | 0.806 | 213 |
| direct (all numbers optimised) | 0.554 | 0.514 | 0.534 | 0.809 | 221 |
| hand + search | 0.524 | 0.461 | 0.492 | 0.792 | 190 |
| HMM, ours (alpha chosen: 0.1) | 0.436 | 0.457 | 0.446 | 0.751 | 268 |
| HMM, ours, alpha 0.15 (exp04) | 0.426 | 0.450 | 0.438 | 0.739 | 312 |
| vote w=50 (repository baseline) | 0.388 | 0.373 | 0.381 | 0.765 | 531 |
| his published config | 0.355 | 0.342 | 0.348 | 0.713 | 301 |
| vote w=120 | 0.326 | 0.303 | 0.314 | 0.706 | 397 |

The two copies of a game differ by about 0.01-0.05.

## BAR-ATH public window (17 possessions; a check, never a choice)

Models are fitted on both Metrica games. Each line is one submission in `experiments/out/exp08/`.

| run | public spellF1 | public macroF1 | possessions | D share, whole match |
|---|---|---|---|---|
| his published + tags (chosen on this window) | 0.520 | 0.596 | 13 | 0.27 |
| his published, tracking only (chosen on this window) | 0.480 | 0.591 | 13 | 0.26 |
| direct | 0.422 | 0.656 | 19 | 0.43 |
| logreg | 0.389 | 0.657 | 18 | 0.42 |
| hand + search + his tags | 0.388 | 0.661 | 17 | 0.45 |
| hand + search | 0.376 | 0.659 | 17 | 0.44 |
| logreg + speeds | 0.366 | 0.651 | 18 | 0.44 |
| HMM, ours, alpha 0.1 | 0.347 | 0.760 | 21 | |

## Reading

* **His model structure is good, and his public-window numbers are what hold it back.** With
  weights learned by logistic regression and a decoder chosen on Metrica, it beats his published
  configuration by about +0.20 spellF1 on Metrica. It also beats our HMM by about +0.10. The gap is
  several times the copy-to-copy spread, and it holds in both directions.
* **On the public window the order flips.** His published configuration stays on top (0.52), but it
  was chosen there among 360 configurations, on 17 possessions. The models learned on Metrica score
  0.37-0.42 there, still above our HMM (0.35).
* **The public window is not a typical stretch of the match.** Only 19.6% of its frames are dead
  ball. Metrica has about 40% and BAR-ATH loses the ball on 40.7% of its frames. A model tuned on
  the window predicts little Dead (26% of the match); the models learned on Metrica predict 41-45%.
  The public window cannot tell which is right for the other 85 minutes.
* **What the learned weights say about his features.**
  * The "collective" term carries the team decision (weight about 1.5).
  * Given it, nearest distance and time to the ball add nothing: their weights are slightly
    negative, a redundancy.
  * For Dead, "ball missing" dominates (about +3.5 in log-odds).
  * "Tracking incomplete" and "ambiguity" are close to 0.
  * Slow players (+1.3) and a still ball (+1.4) are the two speed terms that help.
* **Direct optimisation of every number overfits a little.** It is the best on its training game,
  but below logistic regression on the other game. Its chosen penalties also differ a lot between
  the two games (switch 150 vs 1470).
* **Our HMM** keeps the best public macroF1 (0.76). On the faithful Metrica copies it now prefers
  alpha 0.1 to 0.15.

## Caveats

* The Metrica labels are a proxy built from event data. The BAR-ATH truth comes from corrected
  tracking ("controlled possession", dead ball as the tracking marks it). Learned weights carry the
  proxy's definition with them.
* Smart Tagging cannot be learned on Metrica. "+ his tags" adds his hand weights for the tag terms
  unchanged.
