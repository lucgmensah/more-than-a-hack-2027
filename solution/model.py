"""The two gradient-boosted models, the training labels, and how their outputs are combined.

* Training labels: DFL possession harmonised to controlled possession (controlled()).
* Team model: LightGBM, 3 classes (H, A, D), on all 159 features, each frame seen from both sides.
* Dead-ball model: LightGBM, binary, on the columns that do not describe ball motion (DEAD_PREFIXES):
  players, ball tracked or not, ball relative to the lines. In DFL the ball is frozen during
  stoppages, in BAR-ATH the tracker follows it while it is carried back, so ball motion is left out.
* combined(): log P(D) from the dead model, P(H) and P(A) from the team model renormalised over the
  two teams, averaged over both team orders so that the result cannot prefer a side.
"""

import numpy as np

import features

MIN_POSSESSION_S = 1.5     # training labels: a shorter spell interrupting the other team's possession is given back
STEP = 5                   # every 5th frame of the training matches

_BASE = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
         "min_data_in_leaf": 200, "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1,
         "lambda_l2": 5.0, "num_threads": 6, "verbosity": -1, "seed": 0}
TEAM_PARAMS = dict(_BASE, learning_rate=0.08, max_bin=127)
TEAM_ROUNDS = 220

DEAD_PREFIXES = ("mean_speed", "p90_speed", "share_slow", "block_shift", "block_spread", "speed_mean_",
                 "n_tracked", "team_distance", "centroid_gap", "obs", "since_ball", "until_ball",
                 "gap_length", "since_gap", "until_gap", "ball_to_touchline", "ball_to_goalline",
                 "ball_to_boundary", "last_ball_to_boundary", "next_ball_to_boundary", "ball_off_centre")
DEAD_PARAMS = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 200,
               "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
               "max_bin": 127, "num_threads": 6, "verbosity": -1, "seed": 0, "deterministic": True,
               "force_row_wise": True}
DEAD_ROUNDS = 220


# -- training labels ------------------------------------------------------------------------------

def spells(label):
    change = np.flatnonzero(np.diff(label)) + 1
    bounds = np.concatenate(([0], change, [len(label)]))
    return bounds[:-1], bounds[1:], label[bounds[:-1]]


def controlled(label, min_frames):
    """A possession shorter than min_frames that interrupts the other team's possession (or ends it
    just before a stoppage) is given back to that team: a touch is not a possession.

    DFL switches team at every touch (9-11 possessions per live minute, a third shorter than 2 s).
    With 1.5 s, its statistics match the Metrica event labels (5-6 per live minute, about 15% under
    2 s), i.e. controlled possession."""
    out = label.copy()
    if min_frames <= 0:
        return out
    for _ in range(3):
        start, stop, value = spells(out)
        changed = False
        for i in range(len(value)):
            if value[i] < 2 and stop[i] - start[i] < min_frames:
                before = value[i - 1] if i > 0 else 2
                after = value[i + 1] if i + 1 < len(value) else 2
                if before < 2 and before != value[i] and (after == before or after == 2):
                    out[start[i]: stop[i]] = before
                    changed = True
        if not changed:
            break
    return out


def training_rows(trk, label, step=STEP, offset=0, pitch=(105.0, 68.0)):
    """Features of every `step`-th labelled frame, seen from both sides (team swap)."""
    xs, ys = [], []
    for flip in (False, True):
        x, names = features.build(trk, flip=flip, pitch=pitch)
        y = features.flip_labels(label) if flip else label
        sel = np.flatnonzero(y < 3)[offset::step]
        xs.append(x[sel])
        ys.append(y[sel])
        del x
    return np.vstack(xs), np.concatenate(ys), names


# -- prediction -----------------------------------------------------------------------------------

def dead_columns(names):
    return [i for i, n in enumerate(names) if n.startswith(DEAD_PREFIXES)]


def feature_names(trk):
    """The 159 column names (computed on a few hundred frames)."""
    _, names = features.build({"players": trk["players"][:400], "ball": trk["ball"][:400], "team": trk["team"]})
    return names


def combined(team_model, dead_model, cols, trk, pitch=(105.0, 68.0)):
    """(N, 3) log-probabilities: P(D) from the dead model, P(H), P(A) from the team model given alive."""
    x, _ = features.build(trk, pitch=pitch)
    p_dead = np.clip(dead_model.predict(x[:, cols]), 1e-6, 1 - 1e-6)
    p = team_model.predict(x)
    x, _ = features.build(trk, flip=True, pitch=pitch)
    q = team_model.predict(x)
    p = (p + q[:, [1, 0, 2]]) / 2.0
    home = np.clip(p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9), 1e-6, 1 - 1e-6)
    return np.log(np.column_stack([(1 - p_dead) * home, (1 - p_dead) * (1 - home), p_dead]))
