"""BAR-ATH tracking, graded frames, submission file and scoring, shared by every script of the solution.

Two rules from RULES.md shape this module:

* The public 5-minute ground truth is used ONLY to print a pipeline check (score_public). Nothing in
  solution/ selects, tunes or trains anything on it.
* Derived frame-level tracking must not be published, so everything written here goes to
  solution/cache/ and solution/out/, which are gitignored.
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # solution/
ROOT = os.path.dirname(HERE)                                # repository root
for _path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import possession_metrics as pm  # noqa: E402
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
OUT_DIR = os.path.join(HERE, "out")
EXTERNAL = os.path.join(DATA_DIR, "external")    # DFL and Metrica, downloaded by dfl.py and metrica.py
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]  # (length, width) in metres
BAR_PITCH = (105.3, 68.0)
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


def period_runs(periods):
    """[(start, stop)] row spans of each period (DFL and Metrica)."""
    return [(int(np.flatnonzero(periods == p)[0]), int(np.flatnonzero(periods == p)[-1]) + 1)
            for p in np.unique(periods)]


MEAN_SPEED_EDGES = (0.7, 1.5, 2.5, 3.5)   # m/s, mean over tracked players


def speed_bucket(tracking):
    """Bucket 0..4 of the mean tracked-player speed on every frame (4 when no player is tracked)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(tracking["players"][..., 2], axis=1)
    return np.digitize(np.nan_to_num(mean_speed, nan=99.0), MEAN_SPEED_EDGES)


def missing_by_speed(tracking, mask=None):
    """Share of frames with the ball missing in each speed bucket (5 values)."""
    bucket = speed_bucket(tracking)
    missing = np.isnan(tracking["ball"][:, 0])
    if mask is not None:
        bucket, missing = bucket[mask], missing[mask]
    return np.array([missing[bucket == b].mean() if (bucket == b).any() else 0.0
                     for b in range(len(MEAN_SPEED_EDGES) + 1)])


# -- submission and scoring -----------------------------------------------------------------------

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


def mask_to_graded(codes):
    """Codes with X on every ungraded frame (X there costs nothing)."""
    out = np.full(len(codes), X, dtype=np.int8)
    graded = graded_mask(len(codes))
    out[graded] = np.asarray(codes)[graded]
    return out


def score(gt, pred, bootstrap=0):
    """Full metric dict of the official scorer against `gt` (DFL or Metrica validation truth)."""
    pred = np.array(pred, dtype=np.int8)
    pred[gt == X] = X
    return pm.evaluate(gt, pred, bootstrap_n=bootstrap)


def score_public(codes, bootstrap_n=0):
    """Metrics on the PUBLIC 5-minute window. A pipeline check, never a tuning signal."""
    gt = pm.label_codes(pm.read_label_csv(GT_PUBLIC))
    return pm.evaluate(gt, np.asarray(codes, dtype=np.int8), bootstrap_n=bootstrap_n)
