# Experiments: possession as a decoded sequence

Code for testing ideas from the literature review on the real match, with the repository's own
scorer. Numpy and pandas only (the same environment as `evaluation/`).

```bash
python experiments/run_experiments.py                  # ladder E0..E5
python experiments/run_experiments.py --sweep          # + sensitivity to the tempering alpha
python experiments/run_experiments.py --bootstrap 500  # intervals on the public window
python experiments/run_experiments.py --write          # also write each prediction to experiments/out/
```

The first run parses the 186 MB tracking file (about a minute) and caches it in
`experiments/cache/`. That folder and `experiments/out/` are gitignored: RULES.md forbids
publishing frame-level tracking derived from the data.

## What is here

| file | role |
|---|---|
| `common.py` | tracking loader and cache, graded-frame mask, Smart Tagging access, scoring on the public window |
| `features.py` | per-frame categorical evidence channels (nearest player, ball speed, set piece, phase team, mean player speed) |
| `hmm.py` | 3-state (H/A/D) HMM: emission tables fitted by EM without labels, Viterbi decoding |
| `run_experiments.py` | the ladder: each row adds one channel to the one above |

## The idea

The baseline votes over a window; a Viterbi decode decides every frame of a possession together
and trades the evidence against a cost for changing state. This is the non-causal, global version
of the smoothing the challenge already shows to matter (Xie 2004 for the HMM idea, PathCRF 2026 for
the CRF/Viterbi formulation on this problem). Missing-ball frames carry a "missing" symbol instead
of a guess.

## Rules compliance (RULES.md)

* **No tuning on the public ground truth.** It is used to print an indicative score and nothing
  else. The ladder is a fixed sequence, the tempering `alpha` is declared in advance (0.25) and the
  `--sweep` is reported as sensitivity, not used to pick a value.
* **No labels on the provided game.** Emission tables are learned by EM on the match's own
  observations (unsupervised). Transition probabilities are a declared football prior (possession
  about 15 s, about 22 s of play between stoppages, stoppage about 12 s), not learned and not
  taken from the ground truth.
* **Which Smart Tagging codes to trust is decided on tracking only**: for each phase code, how
  often its Team tag equals the team of the player nearest the ball. Attacking codes come out at
  0.62 to 0.76, `DEFENDING *` codes at 0.24 to 0.37 (the Team tag names the defending side, so
  those votes are used with the sign flipped), the rest are ignored.

## Findings that shaped the code (so the next person does not repeat them)

1. **The ATD ball speed column is unusable**: present on 8% of graded frames even though the ball
   position exists on 59%. Speed is recomputed from positions (`features.ball_speed`).
2. **Unconstrained EM turns D into a "ball not visible" detector.** Visibility is the strongest
   structure in the data, so D absorbed the 41% of frames with no ball. The "missing" symbols are
   therefore tied to be equally likely in every state (`hmm._tie_missing`).
3. **Learning the transitions by EM reproduces the evidence's flicker** (possessions of 3 s). They
   are fixed to the dwell-time prior above (`hmm.initial_transitions`).
4. Emissions are tempered (`alpha` < 1) because consecutive frames at 25 fps are far from
   independent; without it the model is overconfident.

## Caveats

The public window has 17 possessions. Differences under the printed bootstrap interval are noise,
and the full-match score is only known to the organisers. `E0` checks that the loader reproduces
`baseline_smooth50` on every graded frame before anything else is trusted.

## Not done yet (ordered by expected value)

* Semi-Markov (explicit duration) decoding instead of geometric dwell times.
* A better "who controls the ball" channel: relative velocity, pitch control (Spearman 2018,
  Fernandez and Bornn 2018).
* Train a supervised model on other games (Metrica sample data, 3 matches, same EPTS format for
  game 3) after degrading them to match this tracking: ball removed on 41% of frames, no player
  identities. Needs a check of the dataset licence first.
* PathCRF (`github.com/hyunsungkim-ds/pathcrf`) and BallRadar (`github.com/pientist/ballradar`) as
  external baselines.
