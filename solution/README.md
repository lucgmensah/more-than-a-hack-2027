# solution/: the code of the submission

The method, its validation and its limits are in the [technical document](../TECHNICAL_REPORT.md).
How we got here is in [docs/](../docs/README.md).

## Regenerate the submitted file (about 1 minute, CPU)

```bash
pip install -r requirements-submission.txt        # from the repository root
python solution/predict.py                        # -> solution/out/submission.csv
```

* The match bundle goes in `data/`, as the challenge README says, or `MTAH_DATA_DIR` points to it.
* Nothing is fitted: the two models are committed in [models/](models).
* The script validates the file with `evaluation/validate_submission.py` and prints the
  public-window check (a pipeline check only).

| output | value |
|---|---|
| file | `solution/out/submission.csv` |
| validator line | `285 possessions of mean 12.8 s` |
| public window check | spellF1 0.671, macroF1 0.839 |
| sha256 | `147033f58934bdb3e0bda5055156177de25b6428440eb55623a831fc59ca2bce` |

## Retrain the two models (about 7 minutes after 3 GB of downloads)

```bash
python solution/metrica.py     # Metrica sample games 1 and 2 (validation only)
python solution/dfl.py         # DFL open data (7 matches, 2.4 GB), parsed once
python solution/train.py       # damaged copies, both models, validation table, decoder
python solution/fill_sim.py    # optional: the simulation that decided to use the broadcast ball
```

`train.py` writes to `solution/cache/train/`. It prints:
* the validation table (DFL held out, Metrica);
* the chosen decoder;
* whether the new `team.txt` and `dead.txt` are identical to the committed ones.

To use new models, copy them to `solution/models/`.

## Files

| file | role |
|---|---|
| `predict.py` | the submission: tracking + broadcast ball → features → two models → Viterbi → CSV |
| `train.py` | damaged DFL copies, team and dead-ball models (held-out and final), validation, decoder choice |
| `common.py` | BAR-ATH tracking loader, graded frames, submission writer, scorer wrappers |
| `broadcast.py` | aligns the broadcast tracking to the tactical timeline from the data, fills the missing ball |
| `features.py` | the 159 per-frame features: touches, context ±3 s, team shape; symmetric under a team swap |
| `model.py` | training labels (controlled possession), model parameters, how the two models are combined |
| `decode.py` | penalty Viterbi (teammate's recursion), decoder grid |
| `damage.py` | makes DFL tracking look like BAR-ATH; every constant from label-free BAR-ATH statistics |
| `dfl.py` | downloads and parses the DFL open data (possession and ball status on every frame) |
| `metrica.py` | Metrica sample games: loader, event-based labels, faithful damage (validation only) |
| `fill_sim.py` | is a second, noisier ball sensor worth using? A simulation on held-out DFL matches |
| `models/` | `team.txt` and `dead.txt`, the two LightGBM models of the submission |

`solution/cache/` and `solution/out/` are gitignored. They hold frame-level derived tracking and
full-match labels, which must not be published.

## How this folder was checked

This is the experiment-9 code of branch `lucg` (`experiments/exp09_supervised` and the helpers it
imported from other experiment folders), gathered here without the experiment scaffolding. The
computations were not changed.

* `predict.py` gives the same file, byte for byte, from a fresh clone and a clean environment.
* `train.py`, run from an empty cache (DFL parsed again, everything refitted), rebuilds:
  * damaged training copies identical to the originals;
  * a missing-ball model identical to the original;
  * four models (team and dead-ball, held-out and final) identical byte for byte to the originals,
    including the two committed in `models/`.

  The whole run (DFL parsing, damage, four models, validation) took 6 min 40 s on a laptop CPU.
* Its validation table is the one in the technical document:

  | validation copy | decoder chosen on | spellF1 | macroF1 |
  |---|---|---|---|
  | DFL J03WMX (held out) | DFL J03WR9 | 0.649 | 0.898 |
  | DFL J03WR9 (held out) | DFL J03WMX | 0.661 | 0.883 |
  | Metrica game 1, copies 0 / 1 | Metrica game 2 | 0.704 / 0.697 | 0.880 / 0.879 |
  | Metrica game 2, copies 0 / 1 | Metrica game 1 | 0.702 / 0.702 | 0.882 / 0.882 |

  The decoder chosen over the six copies is switch 8, dead 64, no minimum: the one in `predict.py`.
