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
| [exp06_abstain](exp06_abstain/README.md) | Does emitting X (forward-filled by spellF1) instead of guessing help? No: -0.04 to -0.05 | as exp02 |
| [lib](lib) | shared code: tracking loader and scoring (`common.py`), evidence channels (`features.py`), HMM (`hmm.py`) | numpy, pandas |

Conventions
* numpy and pandas only, like `evaluation/`.
* `experiments/cache/` and `experiments/out/<experiment>/` hold derived data and are gitignored
  (RULES.md forbids publishing frame-level tracking derived from the data).
* The BAR-ATH public ground truth is only ever used to check the pipeline and print an indicative
  score; nothing is selected or trained on it.

Adding an experiment: create `expNN_name/` with a `run.py` that puts `../lib` on `sys.path`, and a
README that states the question, the rule constraints and the caveats.
