"""Step 1: training crops for the ball detector, labelled by an algorithm.

    python experiments/exp10_video_ball/make_dataset.py [--frames 6000]

For a sample of graded frames where the automatic tracking has the ball:
  * positive crop: the tracked position is projected into the image and snapped onto the visible
    ball (vb.snap). Frames where no single blob is found near it are not used as positives;
  * negative crops: around a tracked player and at a random place on the pitch, both far from the
    ball. Players (boots, socks, shirts) and lines are what a ball detector confuses with a ball.
The last part of each half is kept aside as validation. Output: experiments/cache/exp10/dataset.npz
(gitignored: it contains video crops).
"""

import argparse
import os
import time

import numpy as np

import vb

FAR_PX = 90          # a negative crop centre is at least this far from the ball


def crop_at(sample, x0, y0):
    return sample[:, y0: y0 + vb.CROP, x0: x0 + vb.CROP]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=6000, help="frames sampled (about half give a positive)")
    parser.add_argument("--val-share", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    tracking = vb.common.load_tracking()
    matrices, _, valid = vb.load_homography()
    ball = tracking["ball"][:, :2]
    usable = vb.common.graded_mask() & np.isfinite(ball[:, 0]) & valid
    frames = np.sort(rng.choice(np.flatnonzero(usable), min(args.frames, int(usable.sum())), replace=False))
    is_val = np.zeros(len(ball), dtype=bool)
    for start, stop in vb.common.half_runs():
        is_val[int(stop - args.val_share * (stop - start)): stop] = True

    reader = vb.FrameReader()
    width, height = reader.size
    crops, labels, source, split = [], [], [], []
    started = time.time()
    for done, frame in enumerate(frames):
        sample = reader.sample(int(frame))
        if sample is None:
            continue
        centre = np.ascontiguousarray(sample[:3].transpose(1, 2, 0))
        tracked = vb.project(matrices[frame], ball[frame])
        hit = vb.snap(centre, tracked[0], tracked[1])
        avoid = np.array(hit if hit is not None else tracked)
        if hit is not None:
            # the ball anywhere in the crop, at least 16 px from its border
            x0 = int(np.clip(hit[0] - rng.integers(16, vb.CROP - 16), 0, width - vb.CROP))
            y0 = int(np.clip(hit[1] - rng.integers(16, vb.CROP - 16), 0, height - vb.CROP))
            crops.append(crop_at(sample, x0, y0))
            labels.append((hit[0] - x0, hit[1] - y0))
            source.append(frame)
            split.append(is_val[frame])
        players = tracking["players"][frame, :, :2]
        players = vb.project(matrices[frame], players[np.isfinite(players[:, 0])])
        centres = [vb.project(matrices[frame], rng.uniform((2.0, 2.0), vb.PITCH - 2.0))]
        if len(players):
            centres.append(players[rng.integers(len(players))] + rng.normal(0, 12, 2))
        for c in centres:
            if not np.all(np.isfinite(c)) or np.hypot(*(c - avoid)) < FAR_PX or np.hypot(*(c - tracked)) < FAR_PX:
                continue
            x0 = int(np.clip(c[0] - vb.CROP // 2, 0, width - vb.CROP))
            y0 = int(np.clip(c[1] - vb.CROP // 2, 0, height - vb.CROP))
            if x0 <= avoid[0] < x0 + vb.CROP and y0 <= avoid[1] < y0 + vb.CROP:
                continue
            crops.append(crop_at(sample, x0, y0))
            labels.append((-1.0, -1.0))
            source.append(frame)
            split.append(is_val[frame])
        if done % 500 == 499:
            print("  {} / {} frames, {} crops ({:.0f}s)".format(
                done + 1, len(frames), len(crops), time.time() - started), flush=True)

    labels = np.array(labels, dtype=np.float32)
    os.makedirs(vb.CACHE, exist_ok=True)
    path = os.path.join(vb.CACHE, "dataset.npz")
    np.savez(path, x=np.stack(crops), label=labels, frame=np.array(source, dtype=np.int32),
             val=np.array(split, dtype=bool))
    positive = labels[:, 0] >= 0
    print("{} frames sampled: {} positive crops ({:.0%} of the frames gave one), {} negative crops; "
          "{:.0%} kept for validation".format(len(frames), positive.sum(), positive.sum() / len(frames),
                                              (~positive).sum(), np.mean(split)))
    print("written", path)


if __name__ == "__main__":
    main()
