"""Per-frame features with temporal context, computable on any tracking dict (Metrica, DFL, BAR-ATH).

Everything is built from team + position + speed only (no identities), and written so that swapping
the two teams negates the "advantage" columns and swaps the paired ones: build(trk, flip=True) is
the same match seen from the other side, used to train a model that cannot prefer a side.

Three families:
  * now      who is close to the ball, how fast is everything, where is the ball on the pitch;
  * touches  the last / next moment a player was on the ball (within 1 m) or the ball was kicked
             (its velocity changed sharply near a player), and for which team: possession follows
             touches, not proximity;
  * context  the same signals a little earlier and later (the labels are produced offline, so the
             future is available), and how long the ball has been / will stay missing.
"""

import warnings

import numpy as np

FPS = 25
LAGS = (-75, -50, -25, -12, -6, 6, 12, 25, 50, 75)
WINDOWS = (12, 25, 50, 125)
CAP = 500            # frames: "long ago" for every time-since feature
CONTACT_M = 1.0      # a player this close to the ball is "on the ball"
KICK_DV = 4.0        # m/s: velocity change over 6 frames that counts as a kick / touch
KICK_NEAR_M = 3.0


def _shift(a, k, fill=np.nan):
    out = np.full_like(a, fill, dtype=np.float32)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _window_mean(a, lo, hi):
    """Mean of a[t+lo .. t+hi] (inclusive), NaN counted as absent; NaN if nothing in the window."""
    valid = np.isfinite(a)
    c = np.concatenate(([0.0], np.cumsum(np.where(valid, a, 0.0), dtype=np.float64)))
    m = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(a))
    i0, i1 = np.clip(idx + lo, 0, len(a)), np.clip(idx + hi + 1, 0, len(a))
    count = m[i1] - m[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (c[i1] - c[i0]) / count, np.nan).astype(np.float32)


def _since(event, value=None):
    """Frames since the last True in `event` (CAP if none), and the `value` carried from it."""
    idx = np.where(event, np.arange(len(event)), -1)
    last = np.maximum.accumulate(idx)
    since = np.where(last >= 0, np.arange(len(event)) - last, CAP).clip(0, CAP).astype(np.float32)
    carried = None if value is None else np.where(last >= 0, value[np.maximum(last, 0)], 0.0).astype(np.float32)
    return since, carried


def _until(event, value=None):
    since, carried = _since(event[::-1], None if value is None else value[::-1])
    return since[::-1], None if carried is None else carried[::-1]


def build(trk, flip=False, pitch=(105.0, 68.0)):
    """(X float32 (N, F), names). NaN = not available (the model handles it)."""
    players, ball = trk["players"], trk["ball"]
    team = (1 - trk["team"]) if flip else trk["team"]
    n = len(ball)
    obs = np.isfinite(ball[:, 0])
    delta = players[..., :2] - ball[:, None, :2]
    dist = np.sqrt((delta ** 2).sum(axis=2))
    speed = players[..., 2]
    cols, names = [], []

    def add(name, values):
        cols.append(np.asarray(values, dtype=np.float32))
        names.append(name)

    # -- now: proximity ---------------------------------------------------------------------------
    side = {}
    for code, tag in ((0, "h"), (1, "a")):
        d = np.where(np.isnan(dist[:, team == code]), 1e6, dist[:, team == code])
        part = np.partition(d, 1, axis=1)[:, :2] if d.shape[1] > 1 else np.repeat(d, 2, axis=1)
        d1, d2 = part.min(axis=1), part.max(axis=1)
        d1 = np.where((d1 > 1e5) | ~obs, np.nan, d1)
        d2 = np.where((d2 > 1e5) | ~obs, np.nan, d2)
        eta = np.where(np.isnan(dist[:, team == code]), 1e6,
                       dist[:, team == code] / np.maximum(np.nan_to_num(speed[:, team == code]), 0.5)).min(axis=1)
        side[tag] = {"d1": d1, "d2": d2, "eta": np.where((eta > 1e5) | ~obs, np.nan, eta),
                     "n2": np.where(obs, (d <= 2.0).sum(axis=1), np.nan),
                     "n5": np.where(obs, (d <= 5.0).sum(axis=1), np.nan),
                     "n10": np.where(obs, (d <= 10.0).sum(axis=1), np.nan)}
    h, a = side["h"], side["a"]
    adv = a["d1"] - h["d1"]                                    # > 0: home is closer
    add("adv", adv)
    add("adv_tanh", np.tanh(adv / 2.0))
    add("adv2", a["d2"] - h["d2"])
    add("adv_eta", a["eta"] - h["eta"])
    add("adv_n2", h["n2"] - a["n2"])
    add("adv_n5", h["n5"] - a["n5"])
    add("adv_n10", h["n10"] - a["n10"])
    dmin = np.fmin(h["d1"], a["d1"])
    add("d_min", dmin)
    add("d_second_team", np.fmax(h["d1"], a["d1"]))
    add("n5_total", h["n5"] + a["n5"])
    add("n10_total", h["n10"] + a["n10"])

    # -- now: ball motion and place ---------------------------------------------------------------
    bx, by = ball[:, 0], ball[:, 1]
    vel = np.full((n, 2), np.nan, dtype=np.float32)
    vel[2:-2] = (ball[4:, :2] - ball[:-4, :2]) * (FPS / 4.0)
    bspeed = np.sqrt((vel ** 2).sum(axis=1))
    dv = np.full(n, np.nan, dtype=np.float32)
    dv[3:-3] = np.sqrt(((vel[6:] - vel[:-6]) ** 2).sum(axis=1))
    add("obs", obs)
    add("ball_speed", bspeed)
    add("ball_dv", dv)
    to_touch = np.minimum(by, pitch[1] - by)
    to_goal = np.minimum(bx, pitch[0] - bx)
    add("ball_to_touchline", to_touch)
    add("ball_to_goalline", to_goal)
    add("ball_to_boundary", np.minimum(to_touch, to_goal))
    add("ball_off_centre", np.abs(by - pitch[1] / 2))

    # -- now: players -----------------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(speed, axis=1)
        add("mean_speed", mean_speed)
        ordered = np.sort(speed, axis=1)                       # NaN last
        count = np.isfinite(speed).sum(axis=1)
        p90 = ordered[np.arange(n), np.clip(np.round(0.9 * (count - 1)).astype(int), 0, None)]
        add("p90_speed", np.where(count > 0, p90, np.nan))
        add("share_slow", np.nanmean(np.where(np.isnan(speed), np.nan, speed < 1.0), axis=1))
        sh, sa = np.nanmean(speed[:, team == 0], axis=1), np.nanmean(speed[:, team == 1], axis=1)
        add("adv_team_speed", sh - sa)
        cx = np.nanmean(players[:, :, 0], axis=1)
        spread = np.nanstd(players[:, :, 0], axis=1)
    cvel = np.full(n, np.nan, dtype=np.float32)
    cvel[12:-12] = np.abs(cx[24:] - cx[:-24])              # block movement over ~1 s
    add("block_shift", cvel)
    add("block_spread", spread)
    add("n_tracked", np.isfinite(players[..., 0]).sum(axis=1))

    # -- now: team shape, usable when the ball is missing ------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shape = {}
        for code, tag in ((0, "h"), (1, "a")):
            xy = players[:, team == code, :2]
            shape[tag] = (np.nanmean(xy[..., 0], axis=1), np.nanmean(xy[..., 1], axis=1),
                          np.nanstd(xy[..., 0], axis=1), np.nanstd(xy[..., 1], axis=1))
    # which way the home side attacks: the side whose players stand further left attacks to the right
    gap = np.nan_to_num(shape["a"][0] - shape["h"][0])
    direction = np.sign(_window_mean(gap.astype(np.float32), -15000, 15000))
    signed = np.full(n, np.nan, dtype=np.float32)
    signed[12:-12] = (cx[24:] - cx[:-24]) * direction[12:-12]        # > 0: the block moves the way home attacks
    add("adv_block_dir", signed)
    add("centroid_gap", (shape["h"][0] - shape["a"][0]) * direction)     # home pushed up relative to away
    add("adv_spread_x", shape["h"][2] - shape["a"][2])
    add("adv_spread_y", shape["h"][3] - shape["a"][3])
    add("team_distance", np.hypot(shape["h"][0] - shape["a"][0], shape["h"][1] - shape["a"][1]))
    add("ball_progress_home", (bx - pitch[0] / 2) * direction)        # ball in the half home attacks
    for w in (25, 75, 250):
        add("adv_block_dir_past%d" % w, _window_mean(signed, -w, -1))
        add("adv_block_dir_next%d" % w, _window_mean(signed, 1, w))
        add("adv_spread_x_mean%d" % w, _window_mean((shape["h"][2] - shape["a"][2]).astype(np.float32), -w, w))
        add("centroid_gap_mean%d" % w, _window_mean(((shape["h"][0] - shape["a"][0]) * direction).astype(np.float32), -w, w))

    # -- touches ----------------------------------------------------------------------------------
    nearest_sign = np.where(np.isnan(adv), 0.0, np.sign(adv)).astype(np.float32)        # +1 home, -1 away
    contact = obs & (np.nan_to_num(dmin, nan=99.0) <= CONTACT_M)
    is_peak = np.zeros(n, dtype=bool)
    d0 = np.nan_to_num(dv, nan=0.0)
    is_peak[3:-3] = (d0[3:-3] >= KICK_DV) & (d0[3:-3] >= np.maximum.reduce([d0[k: n - 6 + k] for k in range(7)]))
    kick = is_peak & (np.nan_to_num(dmin, nan=99.0) <= KICK_NEAR_M)
    for tag, event in (("contact", contact), ("kick", kick)):
        since, who = _since(event, nearest_sign)
        until, who_next = _until(event, nearest_sign)
        add("since_" + tag, since)
        add("last_%s_team" % tag, who)
        add("until_" + tag, until)
        add("next_%s_team" % tag, who_next)
        # the same, but only counting events of each team
        for sign, side_tag in ((1.0, "h"), (-1.0, "a")):
            s, _ = _since(event & (nearest_sign == sign))
            u, _ = _until(event & (nearest_sign == sign))
            side[side_tag]["since_" + tag], side[side_tag]["until_" + tag] = s, u
        add("adv_since_" + tag, a["since_" + tag] - h["since_" + tag])      # > 0: home touched it more recently
        add("adv_until_" + tag, a["until_" + tag] - h["until_" + tag])
    # how long the nearest team has been the same, and for how long it stays
    change = np.zeros(n, dtype=bool)
    known = nearest_sign != 0
    filled = np.where(known, nearest_sign, 0.0)
    idx = np.where(known, np.arange(n), -1)
    last_known = np.maximum.accumulate(idx)
    carried = np.where(last_known >= 0, filled[np.maximum(last_known, 0)], 0.0)
    change[1:] = (carried[1:] != carried[:-1]) & (carried[1:] != 0) & (carried[:-1] != 0)
    add("since_nearest_change", _since(change)[0])
    add("until_nearest_change", _until(change)[0])
    add("nearest_carried", carried)

    # -- gaps -------------------------------------------------------------------------------------
    since_obs, _ = _since(obs)
    until_obs, _ = _until(obs)
    add("since_ball", since_obs)
    add("until_ball", until_obs)
    add("gap_length", np.where(obs, 0.0, np.minimum(since_obs + until_obs, 2 * CAP)))
    since_gap, _ = _since(~obs)
    until_gap, _ = _until(~obs)
    add("since_gap", since_gap)
    add("until_gap", until_gap)
    for name, values in (("ball_to_boundary", np.minimum(to_touch, to_goal)), ("ball_speed", bspeed),
                         ("adv_tanh", np.tanh(adv / 2.0)), ("d_min", dmin)):
        ok = np.isfinite(values)
        add("last_" + name, _since(ok, np.nan_to_num(values))[1])
        add("next_" + name, _until(ok, np.nan_to_num(values))[1])

    # -- context ----------------------------------------------------------------------------------
    adv_t = np.nan_to_num(np.tanh(adv / 2.0), nan=0.0).astype(np.float32)
    for k in LAGS:
        add("adv_tanh@%+d" % k, _shift(np.tanh(adv / 2.0), k))
        add("ball_speed@%+d" % k, _shift(bspeed, k))
        add("d_min@%+d" % k, _shift(dmin, k))
        add("mean_speed@%+d" % k, _shift(mean_speed, k))
    for w in WINDOWS:
        add("adv_mean_past%d" % w, _window_mean(adv_t, -w, -1))
        add("adv_mean_next%d" % w, _window_mean(adv_t, 1, w))
        add("obs_mean_past%d" % w, _window_mean(obs.astype(np.float32), -w, -1))
        add("obs_mean_next%d" % w, _window_mean(obs.astype(np.float32), 1, w))
        add("speed_mean_past%d" % w, _window_mean(mean_speed, -w, -1))
        add("speed_mean_next%d" % w, _window_mean(mean_speed, 1, w))
        add("ball_speed_past%d" % w, _window_mean(bspeed, -w, -1))
        add("ball_speed_next%d" % w, _window_mean(bspeed, 1, w))
        add("kicks_past%d" % w, _window_mean(kick.astype(np.float32), -w, -1))
        add("kicks_next%d" % w, _window_mean(kick.astype(np.float32), 1, w))
        add("contact_adv_past%d" % w, _window_mean(np.where(contact, nearest_sign, 0.0), -w, -1))
        add("contact_adv_next%d" % w, _window_mean(np.where(contact, nearest_sign, 0.0), 1, w))
    return np.column_stack(cols), names


def flip_labels(label):
    """H <-> A, everything else unchanged."""
    return np.where(label == 0, 1, np.where(label == 1, 0, label)).astype(np.int8)
