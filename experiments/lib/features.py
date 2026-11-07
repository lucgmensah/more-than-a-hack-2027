"""Per-frame observation channels for the HMM, built from the provided data only.

Each channel turns one kind of evidence into a small categorical symbol per frame. Keeping them
categorical (rather than real-valued) is deliberate: the HMM in hmm.py then learns what each
symbol means for each state by EM, with no labels, so no number below is tuned on ground truth.
The bucket edges are physical scales (metres, m/s), not fitted constants.

    channel          symbols                                     literature hook
    ---------------  ------------------------------------------  --------------------------------
    nearest          team of the closest player x how close x    the challenge baseline
                     how clear the margin is, + 'ball missing'
    ball_speed       missing / <1 / 1-4 / 4-10 / >10 m/s          dead ball = ball at rest
    set_piece        inside a SET PIECES tag before the restart   dead-ball prior (Smart Tagging)
    phase_team       team of the overlapping possession-type      tactical prior (Smart Tagging)
                     Smart Tagging phase
    mean_speed       mean speed of all tracked players            Lang et al. 2022: collective
                                                                  movement reveals stoppages
"""

import warnings

import numpy as np

from common import FPS, H, A, N_SLOTS, smart_tags, tag_team

CLOSE_EDGES = (1.0, 2.0, 4.0, 8.0)   # m: player-ball distance buckets. On clean Metrica tracking the
                                      # nearest player is right 97% of the time within 1 m, 68% at
                                      # 1-2 m and about 55-60% beyond 2 m (measured against the proxy GT)
_NEAREST_SYMBOLS = 2 * (len(CLOSE_EDGES) + 1) * 2 + 1   # team x distance bucket x margin, + missing

# Symbol counts per channel (including the 'missing' symbol where there is one).
N_SYMBOLS = {"nearest": _NEAREST_SYMBOLS, "nearest_interp": _NEAREST_SYMBOLS, "ball_speed": 5,
             "set_piece": 2, "phase_team": 3, "mean_speed": 5, "ttb": 13, "count_diff": 6,
             "set_piece@vis": 4, "phase_team@vis": 6}

MARGIN_EDGE = 1.0              # m: how much closer the nearest player is than the nearest opponent
BALL_SPEED_EDGES = (1.0, 4.0, 10.0)   # m/s
MEAN_SPEED_EDGES = (0.7, 1.5, 2.5, 3.5)  # m/s, mean over tracked players (walking ... sprinting)
RESTART_SPEED = 5.0            # m/s: a ball moving this fast is back in play
RESTART_SUSTAIN = 5            # frames the speed must hold
RESTART_SEARCH = 15 * FPS      # look for the restart at most this long after the tag starts


def _nearest_by_team(tracking):
    """Distance from the ball to the closest home / away player, NaN where undefined."""
    players, ball, team = tracking["players"], tracking["ball"], tracking["team"]
    delta = players[..., :2] - ball[:, None, :2]
    dist = np.sqrt(np.einsum("nsc,nsc->ns", delta, delta))     # NaN where player or ball missing
    big = np.float32(1e9)
    home = np.where(np.isnan(dist[:, team == 0]), big, dist[:, team == 0]).min(axis=1)
    away = np.where(np.isnan(dist[:, team == 1]), big, dist[:, team == 1]).min(axis=1)
    # A side with no tracked player gets distance 1e9 (infinitely far), so the other side is the
    # nearest -- as in the reference baseline. Only "ball missing or nobody tracked" is undefined.
    nobody = (home >= big) & (away >= big)
    ball_missing = np.isnan(ball[:, 0])
    home[nobody | ball_missing] = np.nan
    away[nobody | ball_missing] = np.nan
    return home, away


def nearest_channel(tracking):
    """team(2) x distance bucket x margin bucket(2), then one final symbol for ball or players missing."""
    d_home, d_away = _nearest_by_team(tracking)
    missing = np.isnan(d_home) | np.isnan(d_away)
    d_home = np.where(missing, 0, d_home)
    d_away = np.where(missing, 0, d_away)

    team = (d_away < d_home).astype(np.int64)                   # 0 = home nearest, 1 = away nearest
    nearest = np.minimum(d_home, d_away)
    margin = np.abs(d_home - d_away)
    dist_bucket = np.digitize(nearest, CLOSE_EDGES)             # 0,1,2
    margin_bucket = (margin >= MARGIN_EDGE).astype(np.int64)    # 0,1
    per_team = (len(CLOSE_EDGES) + 1) * 2
    symbol = team * per_team + dist_bucket * 2 + margin_bucket
    symbol[missing] = 2 * per_team
    return symbol


INTERP_MAX_GAP_S = 3.0      # s: ball gaps up to this long are bridged by linear interpolation


def interpolated_ball(tracking, max_gap_s=INTERP_MAX_GAP_S):
    """Ball positions with short gaps filled linearly between the visible ends; NaN elsewhere.

    Non-causal on purpose (the whole match is known). Only frames where the ball is MISSING and
    the surrounding gap is short get a value; visible frames stay NaN here, because they are the
    business of the plain `nearest` channel.
    """
    ball = tracking["ball"][:, :2]
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    out = np.full(ball.shape, np.nan, dtype=np.float32)
    if seen.size < 2:
        return out
    index = np.arange(len(ball))
    nxt = np.searchsorted(seen, index, side="left")
    ok = (nxt > 0) & (nxt < seen.size)
    left = seen[np.clip(nxt - 1, 0, seen.size - 1)]
    right = seen[np.clip(nxt, 0, seen.size - 1)]
    fill = ok & np.isnan(ball[:, 0]) & ((right - left) <= max_gap_s * FPS)
    for axis in (0, 1):
        out[fill, axis] = np.interp(index[fill], seen, ball[seen, axis])
    return out


def nearest_interp_channel(tracking):
    """`nearest` evaluated on the interpolated ball; 'missing' wherever the ball is seen or the gap is long."""
    bridged = dict(tracking)
    bridged["ball"] = np.concatenate(
        [interpolated_ball(tracking), np.full((len(tracking["ball"]), 1), np.nan, np.float32)], axis=1)
    return nearest_channel(bridged)


def nearest_team_codes(tracking):
    """Plain nearest-player team (0 home, 1 away, -1 unknown): the challenge baseline's raw vote."""
    d_home, d_away = _nearest_by_team(tracking)
    codes = np.full(len(d_home), -1, dtype=np.int8)
    ok = ~(np.isnan(d_home) | np.isnan(d_away))
    codes[ok] = (d_away[ok] < d_home[ok]).astype(np.int8)
    return codes


SPEED_HALF_WINDOW = 5   # frames: speed = displacement over +/-5 frames (0.4 s), needs both ends visible


def ball_speed(tracking, half_window=SPEED_HALF_WINDOW):
    """Ball speed in m/s from positions, NaN where either end of the window has no ball.

    The speed column of the raw ATD is present on only ~8% of graded frames even where the ball
    position exists (checked), so it cannot be used; positions are on 59% of them.
    """
    key = "_ball_speed_{}".format(half_window)
    if key not in tracking:
        pos = tracking["ball"][:, :2]
        speed = np.full(len(pos), np.nan, dtype=np.float32)
        k = half_window
        delta = pos[2 * k:] - pos[: -2 * k]
        speed[k:-k] = np.sqrt((delta ** 2).sum(axis=1)) / (2.0 * k / FPS)
        tracking[key] = speed
    return tracking[key]


def ball_speed_channel(tracking):
    speed = ball_speed(tracking)
    symbol = np.digitize(np.nan_to_num(speed, nan=0.0), BALL_SPEED_EDGES) + 1   # 1..4
    symbol[np.isnan(speed)] = 0
    return symbol.astype(np.int64)


def mean_speed_channel(tracking):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames with no tracked player at all
        mean = np.nanmean(tracking["players"][..., 2], axis=1)
    symbol = np.digitize(np.nan_to_num(mean, nan=0.0), MEAN_SPEED_EDGES)       # 0..4
    symbol[np.isnan(mean)] = 0
    return symbol.astype(np.int64)


# -- Smart Tagging channels ------------------------------------------------------------------------

def _restart_frame(speed_series, start, stop):
    """First frame in [start, stop) where the ball stays above RESTART_SPEED for RESTART_SUSTAIN frames."""
    fast = np.nan_to_num(speed_series[start:stop], nan=0.0) > RESTART_SPEED
    if fast.size < RESTART_SUSTAIN:
        return None
    run = np.convolve(fast.astype(np.int8), np.ones(RESTART_SUSTAIN, dtype=np.int8), mode="valid")
    hits = np.flatnonzero(run == RESTART_SUSTAIN)
    return start + int(hits[0]) if hits.size else None


def set_piece_windows(tracking, events=None):
    """[(start, stop, type)] dead-ball windows: from the SET PIECES tag start to the detected restart.

    Where the ball never visibly restarts inside the search window, the median delay measured for
    that type over the whole match is used instead (measured on tracking only, no ground truth).
    """
    events = events if events is not None else smart_tags()
    speed_series = ball_speed(tracking)
    n = len(speed_series)
    found, pending = {}, []
    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        kind = (event["tags"].get("Type") or "unknown").strip().lower()
        start = max(0, event["start_frame"])
        restart = _restart_frame(speed_series, start, min(n, start + RESTART_SEARCH))
        if restart is None:
            pending.append((start, kind))
        else:
            found.setdefault(kind, []).append(restart - start)
    overall = int(np.median(np.concatenate([np.array(v) for v in found.values()]))) if found else 5 * FPS
    windows = []
    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        kind = (event["tags"].get("Type") or "unknown").strip().lower()
        start = max(0, event["start_frame"])
        restart = _restart_frame(speed_series, start, min(n, start + RESTART_SEARCH))
        delay = (restart - start) if restart is not None else int(np.median(found.get(kind, [overall])))
        windows.append((start, min(n, start + max(delay, 1)), kind))
    return windows


def set_piece_channel(tracking, events=None):
    symbol = np.zeros(len(tracking["ball"]), dtype=np.int64)
    for start, stop, _ in set_piece_windows(tracking, events):
        symbol[start:stop] = 1
    return symbol


def phase_team_votes(events=None):
    """(votes_home, votes_away, by_code): per-frame overlap counts of team-tagged phases, per code."""
    events = events if events is not None else smart_tags()
    n = max(e["end_frame"] for e in events) + 2
    by_code = {}
    for event in events:
        team = tag_team(event)
        code = (event["code"] or "").strip().lower()
        if team is None or code == "set pieces":
            continue
        diff = by_code.setdefault(code, np.zeros((2, n + 1), dtype=np.int32))
        diff[team, max(0, event["start_frame"])] += 1
        diff[team, min(n, event["end_frame"])] -= 1
    return {code: np.cumsum(diff, axis=1)[:, :n] for code, diff in by_code.items()}


def tag_agreement(tracking, events=None, min_frames=2000):
    """For every phase code: how often the tag's team is also the team of the nearest player.

    Computed against the tracking only (no ground truth). Around 1.0 means the Team tag names the
    side in possession (attacking phases); around 0.0 means it names the defending side; around 0.5
    the tag says nothing about possession. Used to decide which codes to trust, and with which sign.
    """
    votes = phase_team_votes(events)
    nearest = nearest_team_codes(tracking)
    n = len(nearest)
    table = {}
    for code, counts in votes.items():
        k = min(n, counts.shape[1])
        home, away = counts[0, :k], counts[1, :k]
        tagged = (home > 0) != (away > 0)
        tagged_team = np.where(home > 0, 0, 1)
        sel = tagged & (nearest[:k] >= 0)
        if sel.sum() < min_frames:
            table[code] = (np.nan, int(sel.sum()))
            continue
        table[code] = (float(np.mean(tagged_team[sel] == nearest[:k][sel])), int(sel.sum()))
    return table


def phase_team_channel(tracking, events=None, trust=0.6):
    """0 = no usable phase, 1 = phase says home has the ball, 2 = phase says away has the ball.

    Only codes whose Team tag agrees with the nearest-player team at least `trust` of the time are
    used; codes that agree at most 1 - trust (the tag names the defending side) are used with the
    sign flipped. The rest are ignored. `trust` is a plausibility cut on tracking-only statistics.
    """
    table = tag_agreement(tracking, events)
    votes = phase_team_votes(events)
    n = len(tracking["ball"])
    score = np.zeros(n, dtype=np.int32)  # >0 home, <0 away
    for code, counts in votes.items():
        agree = table.get(code, (np.nan, 0))[0]
        if np.isnan(agree):
            continue
        sign = 1 if agree >= trust else (-1 if agree <= 1 - trust else 0)
        if sign == 0:
            continue
        k = min(n, counts.shape[1])
        score[:k] += sign * (counts[0, :k] - counts[1, :k])
    symbol = np.zeros(n, dtype=np.int64)
    symbol[score > 0] = 1
    symbol[score < 0] = 2
    return symbol


# -- control-aware channels (who can actually get to the ball, who is around it) -----------------------

REACTION_S = 0.3            # s: a player keeps his current velocity this long before reacting
MAX_SPEED = 5.0             # m/s: sprint speed used for the arrival-time estimate
TTB_EDGES = (0.5, 1.2)      # s: arrival time buckets
TTB_MARGIN_S = 0.3          # s: margin between the two teams' fastest arrival
ZONE_RADIUS = 5.0           # m: players this close to the ball take part in the duel


def player_velocity(tracking, half_window=SPEED_HALF_WINDOW):
    """(N, P, 2) velocity in m/s from positions over +/- half_window frames, NaN where unknown."""
    key = "_vel_{}".format(half_window)
    if key not in tracking:
        pos = tracking["players"][..., :2]
        vel = np.full(pos.shape, np.nan, dtype=np.float32)
        k = half_window
        vel[k:-k] = (pos[2 * k:] - pos[: -2 * k]) / (2.0 * k / FPS)
        tracking[key] = vel
    return tracking[key]


def _team_min(values, team, which):
    big = np.float32(1e9)
    sub = values[:, team == which]
    out = np.where(np.isnan(sub), big, sub).min(axis=1)
    out[out >= big] = np.nan
    return out


def ttb_channel(tracking):
    """Like `nearest`, but ranks players by estimated time to reach the ball, not by distance.

    A defender running away from the ball is farther in time than his distance suggests, and a
    player sprinting towards it is closer: the simplest physical version of Spearman's pitch
    control. Players with an unknown velocity are assumed to stand still.
    """
    players, ball, team = tracking["players"], tracking["ball"], tracking["team"]
    vel = np.nan_to_num(player_velocity(tracking), nan=0.0)
    future = players[..., :2] + vel * REACTION_S
    delta = future - ball[:, None, :2]
    arrival = REACTION_S + np.sqrt(np.einsum("nsc,nsc->ns", delta, delta)) / MAX_SPEED
    arrival = np.where(np.isnan(players[..., 0]), np.nan, arrival)
    t_home, t_away = _team_min(arrival, team, 0), _team_min(arrival, team, 1)
    missing = np.isnan(t_home) | np.isnan(t_away) | np.isnan(ball[:, 0])
    t_home, t_away = np.where(missing, 0, t_home), np.where(missing, 0, t_away)
    side = (t_away < t_home).astype(np.int64)
    best = np.minimum(t_home, t_away)
    margin = (np.abs(t_home - t_away) >= TTB_MARGIN_S).astype(np.int64)
    symbol = side * 6 + np.digitize(best, TTB_EDGES) * 2 + margin
    symbol[missing] = 12
    return symbol


def count_diff_channel(tracking):
    """Home minus away players within ZONE_RADIUS of the ball, clipped to [-2, 2] -> symbols 0..4; 5 = ball missing."""
    players, ball, team = tracking["players"], tracking["ball"], tracking["team"]
    delta = players[..., :2] - ball[:, None, :2]
    near = np.sqrt(np.einsum("nsc,nsc->ns", delta, delta)) <= ZONE_RADIUS   # NaN compares False
    diff = near[:, team == 0].sum(axis=1) - near[:, team == 1].sum(axis=1)
    symbol = np.clip(diff, -2, 2) + 2
    symbol[np.isnan(ball[:, 0])] = 5
    return symbol.astype(np.int64)


def build_channels(tracking, names, events=None):
    """(N, len(names)) int symbol matrix for the requested channels, in the given order."""
    # `events` stays None for matches without Smart Tagging; only the two tag channels read it, and
    # they load the BAR-ATH tagging themselves when asked for with events=None.
    visible = (~np.isnan(tracking["ball"][:, 0])).astype(np.int64)
    builders = {
        "nearest": lambda: nearest_channel(tracking),
        "ball_speed": lambda: ball_speed_channel(tracking),
        "set_piece": lambda: set_piece_channel(tracking, events),
        "phase_team": lambda: phase_team_channel(tracking, events),
        "mean_speed": lambda: mean_speed_channel(tracking),
        "nearest_interp": lambda: nearest_interp_channel(tracking),
        # "@vis": the same channel crossed with "is the ball visible", so EM can learn that a tag
        # deserves more trust when the tracking has no ball than when it does.
        "set_piece@vis": lambda: set_piece_channel(tracking, events) + 2 * visible,
        "phase_team@prior": lambda: phase_team_channel(tracking, events),   # D row tied in hmm.py
        "phase_team@vis": lambda: phase_team_channel(tracking, events) + 3 * visible,
        "ttb": lambda: ttb_channel(tracking),
        "count_diff": lambda: count_diff_channel(tracking),
    }
    return np.stack([builders[name]() for name in names], axis=1)
