"""Experiment 10: find the ball in the tactical video where the tracking lost it. A GPU job.

    python experiments/exp10_video_ball/run.py                  # all four steps, stops if there is no GPU
    python experiments/exp10_video_ball/run.py --allow-cpu --frames 400 --epochs 3 --limit 60 --check 30
                                                                # a smoke test of the code on a CPU

Then, to use the result (off by default):

    python experiments/exp09_supervised/predict.py --video-ball

Steps (each is its own script and can be re-run alone):
  1. make_dataset.py  training crops labelled by an algorithm       CPU, about 10 min
  2. train.py         the heat-map net                              GPU, minutes
  3. detect.py        the net on the ~65,000 frames without a ball  GPU, of the order of an hour
  4. link.py          detections chained into tracks, checks        CPU, a minute
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def step(script, *options):
    print("\n=== {} {}".format(script, " ".join(options)), flush=True)
    code = subprocess.call([sys.executable, os.path.join(HERE, script)] + list(options))
    if code:
        raise SystemExit("{} failed (exit {})".format(script, code))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-cpu", action="store_true", help="run even without a GPU (test only)")
    parser.add_argument("--frames", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2, help="full frames per batch at detection")
    parser.add_argument("--check", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="detect on the first N frames only (test)")
    parser.add_argument("--keep-dataset", action="store_true", help="reuse experiments/cache/exp10/dataset.npz")
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
        import torch
    except ImportError as error:
        raise SystemExit("missing dependency ({}): pip install -r experiments/requirements-exp10.txt".format(error))
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU found:", torch.cuda.get_device_name(0))
    elif args.allow_cpu:
        device = "cpu"
        print("no GPU: running on the CPU because --allow-cpu was given (test only)")
    else:
        raise SystemExit("no GPU found. This experiment is optional: the submission of exp09 does not need it.\n"
                         "Run it on a machine with a CUDA GPU, or pass --allow-cpu for a small test.")

    import vb
    if not (args.keep_dataset and os.path.isfile(os.path.join(vb.CACHE, "dataset.npz"))):
        step("make_dataset.py", "--frames", str(args.frames))
    step("train.py", "--epochs", str(args.epochs), "--device", device)
    detect = ["--device", device, "--batch", str(args.batch), "--check", str(args.check)]
    if args.limit:
        detect += ["--limit", str(args.limit)]
    step("detect.py", *detect)
    step("link.py")
    print("\nRead the checks printed by link.py, then, if they are good:\n"
          "    python experiments/exp09_supervised/predict.py --video-ball")


if __name__ == "__main__":
    main()
