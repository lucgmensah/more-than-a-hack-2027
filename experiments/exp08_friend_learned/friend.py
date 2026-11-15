"""The teammate's score-based Viterbi, ported line by line from his repository, with its weights exposed.

Source: github.com/Romuald-Yapo/barca-hackaton, scripts/run_atd_experiments.py (nearest_features),
scripts/build_feature_store.py (centroids, completeness, tag counts) and
scripts/run_temporal_control_model.py (build_emissions, viterbi, merge_short_alive_runs, the final
majority vote). Differences, all declared:

* `gmm_8_confidence` (an unsupervised 8-cluster mixture over his whole feature store, weight 0.20
  on Dead) is left out: it needs his full store (homography, video) and sklearn.
* The Smart Tagging terms exist only for BAR-ATH (Metrica has no tagging); they are 0 on Metrica.
* The Viterbi is the same recursion written with Python floats (about 10x faster than the
  per-frame numpy version, identical output including ties).

The emission is linear in its weights: H = w . f_own, A = -w . f_own, D = v . f_dead + bias. His
hand-set values are HAND below; exp08 learns them instead.
"""

import warnings

import numpy as np

HOME, AWAY, DEAD = 0, 1, 2

OWN_NAMES = ("nearest", "eta", "collective", "centroid")
DEAD_NAMES = ("ball_missing", "tracking_incomplete", "ambiguity")
EXTRA_DEAD_NAMES = ("slow_players", "ball_still", "ball_fast")   # in his store, not in his emission
TAG_OWN, TAG_SET, TAG_TRANSITION = 0.15, 0.45, -0.20               # his tag weights (BAR-ATH only)

HAND = {"own": np.array([1.25, 0.35, 0.45, 0.20]),
        "dead": np.array([1.35, 0.25, 0.20]),
        "bias": 0.0}
PUBLISHED = {"switch": 0.5, "dead": 0.3, "min_alive": 50, "window": 120}   # his best public spellF1


def _tanh(value, scale):
    return np.tanh(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) / scale)


def _team_min(values, mask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmin(values[:, mask], axis=1)
    return out


def frame_features(tracking, ball_speed=None):
    """Per-frame inputs of his emission. tracking: players (N,P,3) x_m, y_m, speed; ball (N,3); team (P,)."""
    players, ball, team = tracking["players"], tracking["ball"], tracking["team"]
    xy, bxy = players[..., :2], ball[:, :2]
    player_visible = np.isfinite(xy).all(axis=2)
    observed = np.isfinite(bxy).all(axis=1)
    dist = np.sqrt(((xy - bxy[:, None, :]) ** 2).sum(axis=2))
    dist[~player_visible | ~observed[:, None]] = np.nan
    speed = np.nan_to_num(players[..., 2], nan=0.0)
    eta = dist / np.maximum(speed, 0.5)

    best, nearby, best_eta, count, centroid = {}, {}, {}, {}, {}
    for code in (HOME, AWAY):
        sel = team == code
        best[code] = _team_min(dist, sel)
        best_eta[code] = _team_min(eta, sel)
        nearby[code] = (dist[:, sel] <= 5.0).sum(axis=1)
        count[code] = player_visible[:, sel].sum(axis=1)
        positions = np.nan_to_num(xy[:, sel], nan=0.0)
        c = positions.sum(axis=1) / np.maximum(count[code][:, None], 1)
        c[count[code] == 0] = np.nan
        centroid[code] = np.hypot(bxy[:, 0] - c[:, 0], bxy[:, 1] - c[:, 1])

    collective = {code: 1.0 / (1.0 + np.nan_to_num(best[code], nan=50.0)) + 0.05 * nearby[code]
                  for code in (HOME, AWAY)}
    own = np.column_stack([
        _tanh(best[AWAY] - best[HOME], 2.0),
        _tanh(best_eta[AWAY] - best_eta[HOME], 1.5),
        _tanh(collective[HOME] - collective[AWAY], 0.12),
        _tanh(centroid[AWAY] - centroid[HOME], 8.0),
    ]).astype(np.float32)
    complete = (count[HOME] >= 8) & (count[AWAY] >= 8)
    hand_own = own @ HAND["own"]
    ambiguity = np.clip(1.0 - np.abs(hand_own), 0.0, 1.0)
    dead = np.column_stack([1.0 - observed, 1.0 - complete, ambiguity * observed]).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(players[..., 2], axis=1)
    if ball_speed is None:
        ball_speed = np.full(len(ball), np.nan)
    still = observed & (np.nan_to_num(ball_speed, nan=99.0) < 1.0)
    fast = observed & (np.nan_to_num(ball_speed, nan=0.0) > 4.0)
    extra = np.column_stack([np.clip(1.5 - np.nan_to_num(mean_speed, nan=1.5), -1.0, 1.0),
                             still, fast]).astype(np.float32)
    return {"own": own, "dead": dead, "extra": extra, "observed": observed}


def tag_features(events, n_frames):
    """His Smart Tagging inputs: team prior (+1 Barcelona / -1 Athletic, clipped), set-piece and
    transition counts. `events` as common.smart_tags()."""
    prior = np.zeros(n_frames, dtype=np.float32)
    set_piece = np.zeros(n_frames + 1, dtype=np.int32)
    transition = np.zeros(n_frames + 1, dtype=np.int32)
    for event in events:
        start, end = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        if end <= start:
            continue
        team = str(event["tags"].get("Team", "")).strip().lower()
        value = 1.0 if team == "barcelona" else -1.0 if team in {"athletic club", "atletic club"} else 0.0
        prior[start:end] += value
        code = str(event["code"]).upper()
        if code == "SET PIECES":
            set_piece[start] += 1
            set_piece[end] -= 1
        if "TRANSITION" in code:
            transition[start] += 1
            transition[end] -= 1
    return {"prior": np.clip(prior, -2.0, 2.0),
            "set_piece": np.clip(np.cumsum(set_piece[:-1]), 0, 2).astype(np.float32),
            "transition": np.clip(np.cumsum(transition[:-1]), 0, 2).astype(np.float32)}


def emissions(feat, params, extra=False, tags=None):
    """(N, 3) H/A/D scores. params: own (4,), dead (3 or 6,), bias."""
    own = feat["own"] @ params["own"]
    dead_inputs = np.hstack([feat["dead"], feat["extra"]]) if extra else feat["dead"]
    dead = dead_inputs @ params["dead"] + params["bias"]
    if tags is not None:
        own = own + TAG_OWN * tags["prior"] * (0.25 + 0.75 * feat["observed"])
        dead = dead + TAG_SET * tags["set_piece"] + TAG_TRANSITION * tags["transition"]
    return np.column_stack([own, -own, dead]).astype(np.float64)


def viterbi(em, switch, dead_penalty, runs):
    """His recursion: team switches cost `switch`, entering or leaving Dead costs `dead_penalty`."""
    labels = np.full(len(em), DEAD, dtype=np.int8)
    sw, dp = float(switch), float(dead_penalty)
    for start, stop in runs:
        if stop <= start:
            continue
        rows = em[start:stop].tolist()
        n = len(rows)
        back = np.empty((n, 3), dtype=np.int8)
        b = back.tolist()
        h, a, d = rows[0]
        for t in range(1, n):
            # to H from H, A, D (first max wins, as np.argmax)
            c0, c1, c2 = h, a - sw, d - dp
            if c0 >= c1 and c0 >= c2:
                nh, bh = c0, 0
            elif c1 >= c2:
                nh, bh = c1, 1
            else:
                nh, bh = c2, 2
            c0, c1, c2 = h - sw, a, d - dp
            if c0 >= c1 and c0 >= c2:
                na, ba = c0, 0
            elif c1 >= c2:
                na, ba = c1, 1
            else:
                na, ba = c2, 2
            c0, c1, c2 = h - dp, a - dp, d
            if c0 >= c1 and c0 >= c2:
                nd, bd = c0, 0
            elif c1 >= c2:
                nd, bd = c1, 1
            else:
                nd, bd = c2, 2
            row = rows[t]
            h, a, d = nh + row[0], na + row[1], nd + row[2]
            b[t] = (bh, ba, bd)
        current = 0 if (h >= a and h >= d) else (1 if a >= d else 2)
        path = [0] * n
        path[n - 1] = current
        for t in range(n - 1, 0, -1):
            current = b[t][current]
            path[t - 1] = current
        labels[start:stop] = path
    return labels


def merge_short_alive_runs(labels, minimum_length, runs):
    """His post-filter: an H/A run shorter than `minimum_length` between two equal runs takes their label."""
    output = labels.copy()
    if minimum_length <= 0:
        return output
    for start, stop in runs:
        seg = output[start:stop]
        if len(seg) == 0:
            continue
        change = np.flatnonzero(np.diff(seg)) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        values = seg[bounds[:-1]].tolist()
        lengths = np.diff(bounds).tolist()
        for i in range(len(values)):
            if values[i] in (HOME, AWAY) and lengths[i] < minimum_length:
                before = values[i - 1] if i > 0 else DEAD
                after = values[i + 1] if i + 1 < len(values) else DEAD
                if before == after:
                    values[i] = before
        output[start:stop] = np.repeat(np.array(values, dtype=np.int8), lengths)
    return output


def majority(codes, window):
    """The starter's centred vote: H/A by majority of H/A frames within +/-window, D if none."""
    if window <= 0:
        return codes.copy()
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - window), np.minimum(len(codes), index + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    out = np.full(len(codes), DEAD, dtype=np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], HOME, AWAY)
    return out


def post(decoded, min_alive, window, runs):
    return majority(merge_short_alive_runs(decoded, min_alive, runs), window)
