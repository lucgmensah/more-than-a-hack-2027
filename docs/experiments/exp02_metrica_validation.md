# Experiment 2: validation on Metrica games with BAR-ATH-like tracking damage

> Experiment report. The code is in [experiments/exp02_metrica_validation](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments/exp02_metrica_validation) on branch `lucg` (not on this branch, which holds only the final solution); the commands below run from there.

The BAR-ATH public ground truth has 17 possessions and cannot rank anything. Metrica's public
sample games 1 and 2 have event data and hundreds of possessions each. Their tracking is clean, so
it is damaged to look like the challenge's tracking and the methods are compared there.

```bash
python experiments/exp02_metrica_validation/download.py      # once: games 1 and 2 into data/external/metrica/
python experiments/exp02_metrica_validation/run.py --quick   # game 1, one HMM config (about 1 min)
python experiments/exp02_metrica_validation/run.py           # full grid, both games, transfer protocol
```

Source: [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) (anonymised;
no licence file, the README asks for responsible use and acknowledgement of the source). Downloaded
data stays in `data/external/` (gitignored) and is not redistributed. Cite Metrica Sports if this
is used in the technical document.

| file | role |
|---|---|
| `metrica_io.py` | reads the CSV tracking and events into the same dict as the BAR-ATH loader; speed derived from positions |
| `ground_truth.py` | frame-level H/A/D proxy from the events (rules in the module docstring) |
| `degrade.py` | damages the tracking: ball missing in bursts resampled from BAR-ATH's own run lengths (41%), players absent in bursts (7%), position noise |
| `run.py` | vote baselines and HMMs on clean and degraded tracking, plus the transfer protocol |

## Why it is a fair stand-in, and where it is not

* **Calibration check (no tuning involved):** on degraded game 1 the repository's own baseline
  scores spellF1 0.202 / 0.239 / 0.246 for windows 25 / 50 / 125 and 0.018 raw. On the full
  BAR-ATH match the published numbers are 0.1996 / 0.2433 / 0.2242 and 0.0486. Same ordering,
  same magnitudes, which is what a good stand-in should show. Degradation constants are declared in
  `degrade.py` and were not adjusted to make this match.
* **The labels are a proxy.** Human-coded events, not corrected tracking: possession starts at a
  RECOVERY, dead ball runs from a stoppage event to the next played ball. Absolute scores are not
  comparable with BAR-ATH.
* **Smart Tagging does not exist for these games**, so the `set_piece` and `phase_team` channels,
  which were the useful ones in experiment 1, cannot be validated here. Only tracking-only evidence
  is compared.
* **Selection protocol:** a parameter is picked on one game and reported on the other, so no number
  in the transfer table was selected on its own data.
