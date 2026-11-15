"""Step 4: chain the detections into ball tracks, write them, and report what they are worth.

    python experiments/exp10_video_ball/link.py [--min-score S] [--min-run 8]

A single frame cannot tell a ball from a white boot; a few seconds can. Inside every stretch where
the tracking has no ball, one candidate per frame (or none) is chosen so that the chain is strong
and physically possible (the ball does not exceed --max-speed), starting from and ending at the
tracked ball when the stretch is bounded by one. Chains shorter than --min-run frames are dropped.

Checks printed, none of them needs a label:
  * on the check frames (tracking has a ball, detector ran blind): how often the strongest
    detection is within 2 m of the tracked ball;
  * on the frames it fills: agreement with the broadcast ball where the broadcast has one;
  * how much of the missing ball is recovered, by player-speed bucket.
Decide from these whether to use it (see README). Output: experiments/cache/exp10/video_ball.npz.
"""

import argparse
import os
import sys

import numpy as np

import vb

sys.path.insert(0, os.path.join(vb.EXPERIMENTS, "exp09_supervised"))
sys.path.insert(0, os.path.join(vb.EXPERIMENTS, "exp02_metrica_validation"))
FPS = 25
MIN_FOUND, MIN_CLOSE = 0.5, 0.8     # the gate on the check frames, see main()


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def chain(frames, cands, threshold, max_speed, slack=1.5, none_cost=3.0):
    """Viterbi over (candidate | none) per frame. cands[i]: (k, 3) x, y, score. Returns the chosen
    candidate index per frame, -1 for none."""
    prev, back = None, []
    for i, c in enumerate(cands):
        k = len(c)
        emit = np.concatenate([np.clip(logit(c[:, 2]) - logit(threshold), -6, 6), [0.0]])   # last state: none
        if prev is None:
            back.append(np.full(k + 1, -1))
            prev = emit
            continue
        before = cands[i - 1]
        trans = np.zeros((len(before) + 1, k + 1))
        if len(before) and k:
            dist = np.linalg.norm(before[:, None, :2] - c[None, :, :2], axis=2)
            reach = max_speed * (frames[i] - frames[i - 1]) / FPS + slack
            trans[: len(before), :k] = np.where(dist <= reach, 0.0, -np.inf)
        trans[: len(before), k] = -none_cost / 2
        trans[len(before), :k] = -none_cost / 2
        total = prev[:, None] + trans
        back.append(total.argmax(axis=0))
        prev = total.max(axis=0) + emit
    state = int(np.argmax(prev))
    chosen = []
    for i in range(len(cands) - 1, -1, -1):
        chosen.append(state if state < len(cands[i]) else -1)
        state = int(back[i][state])
    return chosen[::-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-score", type=float, default=None, help="default: the threshold found at training")
    parser.add_argument("--min-run", type=int, default=8)
    parser.add_argument("--max-speed", type=float, default=40.0)
    args = parser.parse_args()
    data = np.load(os.path.join(vb.CACHE, "candidates.npz"))
    threshold = float(args.min_score if args.min_score is not None else data["threshold"])
    tracking = vb.common.load_tracking()
    ball = tracking["ball"][:, :2]
    has_ball = np.isfinite(ball[:, 0])
    order = np.argsort(data["frame"], kind="stable")
    c_frame, c_xy, c_score = data["frame"][order], data["xy"][order], data["score"][order]
    first = np.searchsorted(c_frame, np.arange(len(ball) + 1))

    def of(frame):
        sl = slice(first[frame], first[frame + 1])
        return np.column_stack([c_xy[sl], c_score[sl]])

    # -- check frames: the detector ran blind where the tracking has a ball ------------------------
    usable, found, dist = False, 0, []
    for frame in data["check"]:
        c = of(int(frame))
        c = c[c[:, 2] >= threshold]
        if len(c):
            found += 1
            dist.append(np.linalg.norm(c[np.argmax(c[:, 2]), :2] - ball[frame]))
    if len(data["check"]):
        dist = np.array(dist)
        print("check frames ({}): a detection above {:.2f} on {:.0%}; of those within 2 m of the tracked ball: {:.0%}, "
              "median distance {:.2f} m".format(len(data["check"]), threshold, found / len(data["check"]),
                                                np.mean(dist <= 2.0) if len(dist) else 0.0,
                                                np.median(dist) if len(dist) else float("nan")))
        usable = bool(found / len(data["check"]) >= MIN_FOUND and len(dist) and np.mean(dist <= 2.0) >= MIN_CLOSE)
        print("gate (declared, not tuned: found on >= {:.0%} of check frames, >= {:.0%} of them within 2 m): {}".format(
            MIN_FOUND, MIN_CLOSE, "PASSED" if usable else "FAILED, predict.py --video-ball will refuse this file"))

    # -- chain inside every stretch without a tracked ball ----------------------------------------
    processed = np.zeros(len(ball), dtype=bool)
    processed[np.setdiff1d(data["processed"], data["check"])] = True
    edges = np.flatnonzero(np.diff(np.concatenate(([0], processed.astype(np.int8), [0]))))
    out_frame, out_xy, out_score = [], [], []
    for start, stop in zip(edges[::2], edges[1::2]):
        frames = list(range(start, stop))
        cands = [of(f) for f in frames]
        if start > 0 and has_ball[start - 1]:                       # anchors: the tracked ball at both ends
            frames.insert(0, start - 1)
            cands.insert(0, np.array([[ball[start - 1, 0], ball[start - 1, 1], 0.999]]))
        if stop < len(ball) and has_ball[stop]:
            frames.append(stop)
            cands.append(np.array([[ball[stop, 0], ball[stop, 1], 0.999]]))
        chosen = chain(frames, cands, threshold, args.max_speed)
        picked = [(f, c[j]) for f, c, j in zip(frames, cands, chosen) if j >= 0 and not has_ball[f]]
        run = []
        for item in picked + [None]:                                # runs of consecutive picks (holes <= 2 frames)
            if item is not None and (not run or item[0] - run[-1][0] <= 3):
                run.append(item)
                continue
            if len(run) >= args.min_run and np.mean([r[1][2] for r in run]) >= threshold:
                out_frame += [r[0] for r in run]
                out_xy += [r[1][:2] for r in run]
                out_score += [r[1][2] for r in run]
            run = [item] if item is not None else []
    out_frame = np.array(out_frame, dtype=np.int32)
    out_xy = np.array(out_xy, dtype=np.float32).reshape(-1, 2)
    path = os.path.join(vb.CACHE, "video_ball.npz")
    np.savez(path, frame=out_frame, xy=out_xy, score=np.array(out_score, dtype=np.float32), threshold=threshold,
             usable=usable)

    # -- what it recovers --------------------------------------------------------------------------
    graded = vb.common.graded_mask()
    missing = graded & ~has_ball
    n_target = int(processed.sum())
    print("{} frames without a tracked ball processed: the ball is placed on {} ({:.1%}); "
          "{:.1%} of all the missing graded frames".format(n_target, len(out_frame), len(out_frame) / max(n_target, 1),
                                                           len(out_frame) / max(int(missing.sum()), 1)))
    import degrade
    filled = dict(tracking)
    filled["ball"] = tracking["ball"].copy()
    filled["ball"][out_frame, :2] = out_xy
    print("ball missing by player-speed bucket, before:", np.round(degrade.missing_by_speed(tracking, graded), 3))
    print("ball missing by player-speed bucket, after: ", np.round(degrade.missing_by_speed(filled, graded), 3))
    try:
        import broadcast
        b = broadcast.load()
        mapping, _ = broadcast.mapping(b, tracking)
        b_ball = np.full((len(ball), 2), np.nan, dtype=np.float32)
        ok = (mapping >= 0) & (mapping < len(ball))
        b_ball[mapping[ok]] = b["ball"][ok, :2]
        both = np.isfinite(b_ball[out_frame, 0])
        if both.sum() >= 20:
            d = np.linalg.norm(out_xy[both] - b_ball[out_frame][both], axis=1)
            print("frames filled where the broadcast also has a ball ({}): distance median {:.2f} m, within 2.4 m {:.0%} "
                  "(the tactical and broadcast trackings between themselves: 0.91 m, 75%)".format(
                      int(both.sum()), np.median(d), np.mean(d <= 2.4)))
    except Exception as error:                                     # the broadcast check is optional
        print("broadcast comparison skipped:", error)
    print("written", path)


if __name__ == "__main__":
    main()
