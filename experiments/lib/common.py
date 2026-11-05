"""Shared library for the experiments (each experiment lives in its own folder and imports this): load the tracking once, build features, score a file.

Everything here is numpy/pandas only, like the rest of the repository.

Two rules from RULES.md shape this module:

* The public 5-minute ground truth is used ONLY to check the pipeline and to print an indicative
  score. Nothing in experiments/ selects, tunes or trains anything on it.
* Derived frame-level tracking must not be published, so the cache written here lives in
  experiments/cache/ which is gitignored.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # experiments/lib
EXPERIMENTS = os.path.dirname(HERE)                         # experiments/
ROOT = os.path.dirname(EXPERIMENTS)                         # repository root
for _path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(EXPERIMENTS, "cache")
OUT_DIR = os.path.join(EXPERIMENTS, "out")   # one sub-folder per experiment
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]  # (length, width) in metres
N_SLOTS = 46

# Class codes, the same as possession_metrics: H=0, A=1, D=2, X=3.
H, A, D, X = pm.H, pm.A, pm.D, pm.X
LETTERS = np.array(["H", "A", "D", "X"])


# -- tracking -------------------------------------------------------------------------------------

def _parse_atd(raw_path, n_frames):
    """Stream the raw ATD into arrays. One line per frame: 'frame:x,y,s;x,y,s;...:bx,by,bs'."""
    players = np.full((n_frames, N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            if not 0 <= frame < n_frames:
                continue
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            players[frame, : values.size // 3] = values.reshape(-1, 3)[:N_SLOTS]
            ball[frame] = np.array(parts[2].split(","), dtype=np.float32)
    return players, ball


def team_of_slot(meta_path):
    """0 = home, 1 = away for every track slot (re-uses the starter's reading of the metadata)."""
    from baseline import team_of_each_track  # starter/baseline.py
    return team_of_each_track(meta_path)


def load_tracking(use_cache=True):
    """dict with 'players' (N,46,3: x_m, y_m, speed), 'ball' (N,3), 'team' (46,), positions in metres.

    The first call parses 186 MB of text (about a minute); later calls read the npz cache.
    """
    cache = os.path.join(CACHE_DIR, "tracking.npz")
    if use_cache and os.path.isfile(cache):
        data = np.load(cache)
        return {key: data[key] for key in data.files}

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n_frames = source_frames("tactical")
    print("parsing {} (first run only) ...".format(os.path.basename(raw_path)), flush=True)
    players, ball = _parse_atd(raw_path, n_frames)

    scale = np.array([PITCH[0], PITCH[1]], dtype=np.float32)
    players[..., :2] *= scale
    ball[:, :2] *= scale
    out = {"players": players, "ball": ball, "team": team_of_slot(meta_path)}

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def graded_mask(n_frames=None):
    """Boolean mask of the frames that are scored. The boundaries are published in DATA.md."""
    n_frames = n_frames or source_frames("tactical")
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first: last + 1] = True
    return mask


def half_runs():
    """[(start, stop)] half-open video-frame spans of the two halves."""
    return [(first, last + 1) for _, first, last in half_video_blocks(GAME)]


# -- smart tagging --------------------------------------------------------------------------------

TEAM_NAMES = {"barcelona": H, "atletic club": A, "athletic club": A}


def smart_tags():
    """Smart Tagging as a list of dicts (code, start_frame, end_frame, tags). Standard library only."""
    from load_smart_tagging import load
    return load()


def tag_team(event):
    """H / A from the tag's Team group, or None for 'N/A' and anything unknown."""
    return TEAM_NAMES.get((event["tags"].get("Team") or "").strip().lower())


# -- scoring --------------------------------------------------------------------------------------

def to_letters(codes):
    return LETTERS[np.asarray(codes, dtype=np.int64)]


def write_submission(path, codes):
    """frame,label CSV from an int code array (0..3)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    letters = to_letters(codes)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, letter) for i, letter in enumerate(letters))
    return path


def score_public(codes, bootstrap_n=0):
    """Metrics on the PUBLIC 5-minute window. A pipeline check, never a tuning signal."""
    gt = pm.label_codes(pm.read_label_csv(GT_PUBLIC))
    return pm.evaluate(gt, np.asarray(codes, dtype=np.int8), bootstrap_n=bootstrap_n)


def mask_to_graded(codes):
    """Codes with X on every ungraded frame (X there costs nothing)."""
    out = np.full(len(codes), X, dtype=np.int8)
    graded = graded_mask(len(codes))
    out[graded] = np.asarray(codes)[graded]
    return out
