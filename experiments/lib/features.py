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

# Symbol counts per channel (including the 'missing' symbol where there is one).
N_SYMBOLS = {"nearest": 13, "ball_speed": 5, "set_piece": 2, "phase_team": 3, "mean_speed": 5}

CLOSE_EDGES = (1.5, 3.0)       # m: player-ball distance buckets (about arm's reach / a stride / far)
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
    """Symbols 0..11 = team(2) x distance bucket(3) x margin bucket(2); 12 = ball or players missing."""
    d_home, d_away = _nearest_by_team(tracking)
    missing = np.isnan(d_home) | np.isnan(d_away)
    d_home = np.where(missing, 0, d_home)
    d_away = np.where(missing, 0, d_away)

    team = (d_away < d_home).astype(np.int64)                   # 0 = home nearest, 1 = away nearest
    nearest = np.minimum(d_home, d_away)
    margin = np.abs(d_home - d_away)
    dist_bucket = np.digitize(nearest, CLOSE_EDGES)             # 0,1,2
    margin_bucket = (margin >= MARGIN_EDGE).astype(np.int64)    # 0,1
    symbol = team * 6 + dist_bucket * 2 + margin_bucket
    symbol[missing] = 12
    return symbol


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


def build_channels(tracking, names, events=None):
    """(N, len(names)) int symbol matrix for the requested channels, in the given order."""
    # `events` stays None for matches without Smart Tagging; only the two tag channels read it, and
    # they load the BAR-ATH tagging themselves when asked for with events=None.
    builders = {
        "nearest": lambda: nearest_channel(tracking),
        "ball_speed": lambda: ball_speed_channel(tracking),
        "set_piece": lambda: set_piece_channel(tracking, events),
        "phase_team": lambda: phase_team_channel(tracking, events),
        "mean_speed": lambda: mean_speed_channel(tracking),
    }
    return np.stack([builders[name]() for name in names], axis=1)
