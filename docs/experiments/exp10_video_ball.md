# Experiment 10 (optional): find the ball in the tactical video where the tracking lost it

> Experiment report. The code is in [experiments/exp10_video_ball](https://github.com/lucgmensah/more-than-a-hack-2027/tree/lucg/experiments/exp10_video_ball) on branch `lucg` (not on this branch, which holds only the final solution); the commands below run from there.

**Status: code written and smoke-tested on a CPU. The full run needs a GPU and has not been done.
The submission of exp09 does not depend on it.**

```bash
pip install -r experiments/requirements-exp10.txt      # opencv + torch (install the CUDA build of torch)

python experiments/exp10_video_ball/run.py             # the four steps; stops at once if there is no GPU
python experiments/exp09_supervised/predict.py --video-ball     # use the result (off by default)
```

`predict.py --video-ball` writes `experiments/out/exp09/submission_video.csv`. The default
submission (`submission.csv`) is never overwritten, and is byte-identical with or without this
folder.

## Why

The automatic tracking has no ball on 41% of the graded frames, up to 80% in the worst 5-minute
blocks. The broadcast tracking gives back an eighth of them (exp09). The ball itself is visible in
the tactical video on most of those frames: it is 4 to 8 pixels wide, white on green.

## How (no human label anywhere)

| step | script | what it does | where |
|---|---|---|---|
| 1 | `make_dataset.py` | 6,000 graded frames where the tracking has the ball. The tracked position is projected into the image with the homography and **snapped onto the visible ball** by a classical white-blob search. Positive crops come from there; negative crops come from around players and random pitch spots. | CPU, about 10 min |
| 2 | `train.py` | A small U-Net (0.3 M weights). Input: the colour frame plus the grey frames 2 before and 2 after (the ball moves, the lines do not). Output: a heat-map of the ball. | GPU, minutes |
| 3 | `detect.py` | The net on every graded frame without a tracked ball (about 65,000), whole frame. The 5 strongest peaks are placed on the pitch with the homography. | GPU, of the order of an hour (estimate) |
| 4 | `link.py` | One candidate per frame (or none) is chosen so that the chain is strong and physically possible. The chain is anchored on the tracked ball at both ends of a gap. Chains shorter than 8 frames are dropped. | CPU, a minute |

The snapping in step 1 is needed because the tracked ball is a ground estimate, not the pixel of the
ball. It is typically 7 px away (median), and about 35 px in the example of a goalkeeper holding the ball. It finds one
unambiguous blob on half of the frames; the others are not used as positives.

RULES.md allows this: "pseudo-labels produced by an algorithm, provided no person decided any of
them".

## Deciding whether to use it

`link.py` prints three label-free checks.

1. **Check frames.** 2,000 frames of a held-out period where the tracking has a ball, on which the
   detector ran blind. How often it finds something, and how often that is within 2 m of the tracked
   ball. A gate is declared in `link.py` (found on at least 50% of them, at least 80% of those
   within 2 m). `predict.py --video-ball` refuses a file that failed it.
2. **Agreement with the broadcast ball** on the frames it fills. Reference: the tactical and
   broadcast trackings are 0.91 m apart in the median.
3. **How much is recovered**, by player-speed bucket.

`exp09_supervised/fill_sim.py` measured what a second ball sensor is worth, with labels, on held-out
DFL matches. Filling 12% of the missing frames with a sensor as noisy as the broadcast one gives
+0.8 points of team accuracy. A better sensor that fills more should give more. Change
`TARGET_SHARE`, `PARALLAX` and `SLOW_M` there to the values `link.py` reports before deciding. The
public window must not be used to decide: it has only 11% of its ball missing.

## What has been checked here, and what has not

Checked on this machine (no GPU):

* Seeking in the video returns exactly the frames of a sequential read.
* The snapped positions sit on the ball (looked at 12 of them).
* The four steps run end to end, including with a net that finds nothing.
* A toy run: 197 positive crops, 3 epochs on the CPU. The net put its peak on the ball in 92% of 25
  validation crops. On 30 whole check frames, it found something on 67%, and 70% of those were
  within 2 m of the tracked ball (median 0.65 m).
* CPU detection speed: 2.7 s per frame, which would be 2 days for the match.

Not checked:

* The quality of the detector trained on the full set.
* Its speed on a GPU.
* Whether its detections improve the labels.

## Known limits

* **Balls in the air.** A detection is placed on the pitch as if the ball were on the ground, so a
  high ball lands several metres too far from the camera. The tracking corrects for height; this
  does not.
* **Hard frames.** The frames the tracker lost are the hard ones (occlusion, crowded boxes). The
  check frames are easier than the frames to fill.
* **Reproducibility.** GPU training is not bit-reproducible. If `submission_video.csv` is
  submitted, commit `models/ballnet.pt` and `models/ballnet.json` so the repository regenerates the
  same file (RULES.md, rule 3). Re-run `detect.py`, `link.py` and `predict.py --video-ball` on the
  same machine to confirm.
