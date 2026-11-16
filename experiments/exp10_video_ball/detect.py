"""Step 3: run the net on every graded frame where the automatic tracking has no ball.

    python experiments/exp10_video_ball/detect.py [--device auto] [--batch 2] [--check 2000]
    python experiments/exp10_video_ball/detect.py --limit 40 --check 20      # a quick test, CPU is enough

The whole 1920x1080 frame goes through the net; the --top strongest local maxima of the heat-map are
kept as candidates and placed on the pitch with the homography (ground assumption, as the tracking).
About 65,000 frames: of the order of an hour on a recent GPU, days on a CPU.

--check N also runs N frames of the validation period where the tracking HAS a ball. They are not
used to fill anything: link.py compares the detections there with the tracked ball, which says how
good the detector is on whole frames.

Output: experiments/cache/exp10/candidates.npz (gitignored).
"""

import argparse
import os
import time

import numpy as np
import torch

import vb

MARGIN_M = 2.0        # candidates further than this outside the pitch are dropped
FLOOR = 0.05          # candidates below this score are not stored


@torch.no_grad()
def candidates_of(net, batch, top, use_amp):
    """batch (B, 5, H, W) float in [0, 1] -> scores (B, top), pixel x (B, top), pixel y (B, top)."""
    height, width = batch.shape[2:]
    pad_h, pad_w = (-height) % 4, (-width) % 4
    if pad_h or pad_w:
        batch = torch.nn.functional.pad(batch, (0, pad_w, 0, pad_h))
    with torch.autocast(device_type=batch.device.type, enabled=use_amp):
        heat = torch.sigmoid(net(batch).float())[:, :, :height, :width]
    peak = heat == torch.nn.functional.max_pool2d(heat, 9, stride=1, padding=4)     # local maxima only
    scores, where = (heat * peak).flatten(1).topk(top, dim=1)
    return scores.cpu().numpy(), (where % width).cpu().numpy(), (where // width).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--check", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="only the first N target frames (test)")
    parser.add_argument("--val-share", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = vb.pick_device(args.device)
    use_amp = device.type == "cuda"
    # Same input, same output: the repository must regenerate the submission exactly (RULES.md, rule 3).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    net, threshold = vb.load_net(device)
    tracking = vb.common.load_tracking()
    matrices, _, valid = vb.load_homography()
    has_ball = np.isfinite(tracking["ball"][:, 0])
    graded = vb.common.graded_mask() & valid

    targets = np.flatnonzero(graded & ~has_ball)
    if args.limit:
        targets = targets[: args.limit]
    in_val = np.zeros(len(has_ball), dtype=bool)
    for start, stop in vb.common.half_runs():
        in_val[int(stop - args.val_share * (stop - start)): stop] = True
    pool = np.flatnonzero(graded & has_ball & in_val)
    rng = np.random.default_rng(args.seed)
    check = np.sort(rng.choice(pool, min(args.check, len(pool)), replace=False))
    frames = np.union1d(targets, check)
    print("device {}; {} frames without a tracked ball, {} check frames; threshold {:.2f}".format(
        device, len(targets), len(check), threshold), flush=True)

    reader = vb.FrameReader()
    out = {"frame": [], "xy": [], "uv": [], "score": []}
    done, started = [], time.time()

    def flush(pending):
        batch = torch.from_numpy(np.stack([s for _, s in pending])).to(device).float() / 255.0
        scores, px, py = candidates_of(net, batch, args.top, use_amp)
        for (frame, _), s, x, y in zip(pending, scores, px, py):
            uv = np.column_stack([x, y]).astype(np.float64)
            xy = vb.unproject(matrices[frame], uv)
            keep = ((s >= FLOOR) & (xy[:, 0] > -MARGIN_M) & (xy[:, 0] < vb.PITCH[0] + MARGIN_M)
                    & (xy[:, 1] > -MARGIN_M) & (xy[:, 1] < vb.PITCH[1] + MARGIN_M))
            out["frame"].append(np.full(int(keep.sum()), frame, dtype=np.int32))
            out["xy"].append(xy[keep].astype(np.float32))
            out["uv"].append(uv[keep].astype(np.float32))
            out["score"].append(s[keep].astype(np.float32))
            done.append(frame)

    pending = []
    for i, frame in enumerate(frames):
        sample = reader.sample(int(frame))
        if sample is not None:
            pending.append((int(frame), sample))
        if len(pending) == args.batch or (pending and i == len(frames) - 1):
            flush(pending)
            pending = []
        if i % 2000 == 1999:
            rate = (i + 1) / (time.time() - started)
            print("  {} / {} frames, {:.1f} per second, about {:.0f} min left".format(
                i + 1, len(frames), rate, (len(frames) - i - 1) / rate / 60), flush=True)

    os.makedirs(vb.CACHE, exist_ok=True)
    path = os.path.join(vb.CACHE, "candidates.npz")
    np.savez(path, frame=np.concatenate(out["frame"]), xy=np.concatenate(out["xy"]), uv=np.concatenate(out["uv"]),
             score=np.concatenate(out["score"]), processed=np.array(done, dtype=np.int32), check=check,
             threshold=threshold)
    strong = np.concatenate(out["score"]) >= threshold
    print("{} frames processed in {:.0f}s; {} candidates, {} above the threshold".format(
        len(done), time.time() - started, len(strong), int(strong.sum())))
    print("written", path)


if __name__ == "__main__":
    main()
