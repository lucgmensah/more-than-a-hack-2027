# Experiments

Each experiment lives in its own folder with its own `run.py` and `README.md`. They share a small
library and nothing else.

| folder | question | needs |
|---|---|---|
| [exp01_hmm_unsupervised](exp01_hmm_unsupervised/README.md) | Does decoding possession with an HMM (EM without labels, Viterbi) beat the windowed vote on BAR-ATH? | `data/tactical/` |
| [exp02_metrica_validation](exp02_metrica_validation/README.md) | Do the methods hold up on games with hundreds of possessions, once their tracking is damaged to look like BAR-ATH? | Metrica sample games (downloaded by the script) and the BAR-ATH tracking |
| [exp03_improve](exp03_improve/README.md) | Which channels, alpha and minimum possession length help, measured on six damaged Metrica copies? | as exp02 |
| [exp04_bar_ath_final](exp04_bar_ath_final/README.md) | The Metrica-chosen configuration, once on BAR-ATH, with and without Smart Tagging; writes validated submissions | `data/tactical/` |
| [exp05_masked_ball](exp05_masked_ball/README.md) | With the ball erased where a player clearly has it, does Smart Tagging recover the team? | `data/tactical/` |
| [exp06_abstain](exp06_abstain/README.md) | X (abstain) instead of guessing: no (-0.04 to -0.05, flat by confidence); semi-Markov decoder: +0.01, noise | as exp02 |
| [exp07_friend_port](exp07_friend_port/README.md) | A teammate's penalty Viterbi and adaptive smoothing, re-tuned on Metrica instead of the public window: +0.10 over the baseline, 0.035 below our HMM | as exp02 |
| [exp08_friend_learned](exp08_friend_learned/README.md) | The teammate's exact model (ported from his code) with weights learned by logistic regression on Metrica: 0.55 spellF1 across games vs 0.45 for our HMM and 0.35 for his public-window config; public window reverses the order (0.39 vs 0.35 vs 0.52, 17 possessions). Also fixes the Metrica damage (exp02-07 were far too harsh) | faithful damage (degrade_faithful) |
| [lib](lib) | shared code: tracking loader and scoring (`common.py`), evidence channels (`features.py`), HMM (`hmm.py`) | numpy, pandas |

Conventions
* numpy and pandas only, like `evaluation/`.
* `experiments/cache/` and `experiments/out/<experiment>/` hold derived data and are gitignored
  (RULES.md forbids publishing frame-level tracking derived from the data).
* The BAR-ATH public ground truth is only ever used to check the pipeline and print an indicative
  score; nothing is selected or trained on it.

Adding an experiment: create `expNN_name/` with a `run.py` that puts `../lib` on `sys.path`, and a
README that states the question, the rule constraints and the caveats.
