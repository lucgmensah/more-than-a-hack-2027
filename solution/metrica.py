"""Metrica Sports sample games 1 and 2: the second validation set (never used to fit anything).

    python solution/metrica.py        # downloads both games into data/external/metrica/ (gitignored)

Source: https://github.com/metrica-sports/sample-data (anonymised; the README asks that use be
responsible and the source acknowledged). Nothing from it is committed.

Three parts:
* load_game: the CSV tracking as the same dict as the BAR-ATH loader (metres, speeds from positions);
* frame_labels: an H/A/D proxy per frame from Metrica's human event data (rules below). It is hand
  data about OTHER games, which RULES.md allows;
* degrade_faithful: makes the clean tracking look like BAR-ATH. Clean Metrica already has BAR-ATH's
  player speeds and 39-41% of ball missing, so only player drop-outs (20.5 tracked of 22) and short
  open-play ball losses (where BAR-ATH loses the ball more often at the same player speed) are added.
Two damaged copies per game (seeds 0 and 1) are cached by copy().
"""

import os
import urllib.request

import numpy as np
import pandas as pd

import common
import damage

FPS = 25
PITCH = (105.0, 68.0)
SPEED_HALF_WINDOW = 5
DATA = os.path.join(common.EXTERNAL, "metrica")
CACHE = os.path.join(common.CACHE_DIR, "metrica")
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
KEYS = ("G1s0", "G1s1", "G2s0", "G2s1")      # game 1 or 2, damage seed 0 or 1


# -- tracking -------------------------------------------------------------------------------------

def paths(game):
    folder = os.path.join(DATA, "Sample_Game_{}".format(game))
    stem = os.path.join(folder, "Sample_Game_{}_".format(game))
    return {"events": stem + "RawEventsData.csv",
            "home": stem + "RawTrackingData_Home_Team.csv",
            "away": stem + "RawTrackingData_Away_Team.csv"}


def download():
    for game in (1, 2):
        for path in paths(game).values():
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = "{}/Sample_Game_{}/{}".format(BASE_URL, game, os.path.basename(path))
            print("downloading", url, flush=True)
            urllib.request.urlretrieve(url, path)
    print("done:", DATA)


def _read_side(path):
    """(frames, periods, players (N, P, 2) normalised, ball (N, 2) normalised) from one team file."""
    raw = pd.read_csv(path, skiprows=2)
    columns = list(raw.columns)
    ball_x = columns.index("Ball")
    # Each player has two columns (x then y); pandas names the second 'Unnamed: k'.
    xy = raw.iloc[:, 3:ball_x].to_numpy(dtype=np.float32).reshape(len(raw), -1, 2)
    ball = raw.iloc[:, ball_x: ball_x + 2].to_numpy(dtype=np.float32)
    return raw["Frame"].to_numpy(), raw["Period"].to_numpy(), xy, ball


def displacement_speed(pos, half_window=SPEED_HALF_WINDOW):
    """m/s from positions over +/- half_window frames; NaN where either end is missing. pos (N, 2) or (N, P, 2)."""
    k = half_window
    speed = np.full(pos.shape[:-1], np.nan, dtype=np.float32)
    delta = pos[2 * k:] - pos[: -2 * k]
    speed[k:-k] = np.sqrt((delta ** 2).sum(axis=-1)) / (2.0 * k / FPS)
    return speed


def load_game(game):
    """dict(players, ball, team, period, frame) for one sample game, positions in metres."""
    p = paths(game)
    frames, periods, home, ball = _read_side(p["home"])
    _, _, away, _ = _read_side(p["away"])
    scale = np.array(PITCH, dtype=np.float32)
    players_xy = np.concatenate([home, away], axis=1) * scale
    ball_xy = ball * scale
    n_players = players_xy.shape[1]

    players = np.full((len(frames), n_players, 3), np.nan, dtype=np.float32)
    players[..., :2] = players_xy
    # Speed is computed inside each period only, so a window never straddles half-time.
    for period in np.unique(periods):
        sel = np.flatnonzero(periods == period)
        players[sel, :, 2] = displacement_speed(players_xy[sel])
    ball_arr = np.full((len(frames), 3), np.nan, dtype=np.float32)
    ball_arr[:, :2] = ball_xy
    team = np.concatenate([np.zeros(home.shape[1], dtype=np.int8),
                           np.ones(away.shape[1], dtype=np.int8)])
    return {"players": players, "ball": ball_arr, "team": team, "period": periods, "frame": frames}


def load_events(game):
    return pd.read_csv(paths(game)["events"])


# -- labels: a proxy from the event data ----------------------------------------------------------
#
# * Alive: the team of the latest on-ball event (PASS, SHOT, RECOVERY, SET PIECE). A BALL LOST or a
#   CHALLENGE does not move possession by itself: in loose phases possession stays with the last team
#   that had the ball, and the other team gets it at its RECOVERY.
# * Dead: from a stoppage event (BALL OUT, FAULT RECEIVED, a goal or a shot out of play, offside, hand
#   ball, end of half) until the next event where the ball is played again (SET PIECE, PASS or SHOT).
#   A SHOT stops play at its END frame, the others at their start frame.
# * Outside the span covered by events of a period: X.

H, A, D, X = 0, 1, 2, 3
TEAM = {"Home": H, "Away": A}
POSSESSION_TYPES = ("PASS", "SHOT", "RECOVERY", "SET PIECE")
PLAYED_TYPES = ("SET PIECE", "PASS", "SHOT")
STOP_BALL_LOST = ("OFFSIDE", "HAND BALL", "END HALF", "REFEREE HIT")


def _is_stop(row):
    kind, sub = row["Type"], str(row["Subtype"])
    if kind in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if kind == "SHOT":
        return "GOAL" in sub or "OUT" in sub
    if kind == "BALL LOST":
        return any(token in sub for token in STOP_BALL_LOST)
    return False


def frame_labels(events, tracking):
    """int8 array over the tracking rows: 0 H, 1 A, 2 D, 3 X (outside the event-covered spans)."""
    frames, periods = tracking["frame"], tracking["period"]
    labels = np.full(len(frames), X, dtype=np.int8)
    index_of = {int(f): i for i, f in enumerate(frames)}
    first_row = {int(p): int(np.flatnonzero(periods == p)[0]) for p in np.unique(periods)}

    for period, block in events.groupby("Period"):
        block = block.sort_values(["Start Frame", "End Frame"], kind="stable")
        start_row = first_row[int(period)]
        stop_row = int(np.flatnonzero(periods == period)[-1]) + 1
        last_event = int(max(block["End Frame"].max(), block["Start Frame"].max()))
        stop_row = min(stop_row, index_of.get(last_event, stop_row - 1) + 1)

        # Timeline of state changes as (row, state) with state 0/1 = alive for that team, 2 = dead.
        changes = [(start_row, D)]
        rows = list(block.to_dict("records"))
        dead = True            # a period starts from a dead ball (kick-off)
        for row in rows:
            at = index_of.get(int(row["End Frame"] if row["Type"] == "SHOT" and _is_stop(row)
                                  else row["Start Frame"]))
            start_at = index_of.get(int(row["Start Frame"]))
            if dead:
                if row["Type"] in PLAYED_TYPES and start_at is not None:
                    dead = False
                    changes.append((start_at, TEAM[row["Team"]]))
                continue
            if _is_stop(row) and at is not None:
                dead = True
                changes.append((at, D))
            elif row["Type"] in POSSESSION_TYPES and start_at is not None:
                changes.append((start_at, TEAM[row["Team"]]))

        changes.sort(key=lambda c: c[0])
        for (row_a, state), nxt in zip(changes, changes[1:] + [(stop_row, None)]):
            labels[row_a: nxt[0]] = state
    return labels


def with_period_gap(array, periods):
    """Copy of a per-row array where the first row of every period after the first is X, so that a
    possession cannot span half-time (the scorer treats X in the truth as 'not graded')."""
    out = np.array(array, copy=True)
    starts = np.flatnonzero(np.diff(periods) != 0) + 1
    out[starts] = X
    return out


# -- damage ---------------------------------------------------------------------------------------

PLAYER_ABSENT_SHARE = 0.07
PLAYER_ABSENT_MEAN_S = 2.0
INPLAY_GAP_MEAN_S = 1.0       # declared: a lost ball in open play is found again in about 1 s


def player_absence_mask(n, n_players, rng):
    """(n, P) True where a player observation is dropped, in bursts of ~2 s, ~7% of the total."""
    mean_absent = PLAYER_ABSENT_MEAN_S * FPS
    mean_present = mean_absent * (1.0 - PLAYER_ABSENT_SHARE) / PLAYER_ABSENT_SHARE
    out = np.zeros((n, n_players), dtype=bool)
    for p in range(n_players):
        pos, absent = 0, bool(rng.random() < PLAYER_ABSENT_SHARE)
        while pos < n:
            length = max(1, int(rng.exponential(mean_absent if absent else mean_present)))
            out[pos: pos + length, p] = absent
            pos += length
            absent = not absent
    return out


def degrade_faithful(tracking, bar_missing, seed=0):
    """Player drop-outs + extra open-play ball losses so the ball is missing as often as in BAR-ATH
    at each player-speed bucket (`bar_missing` = missing_by_speed of BAR-ATH's graded frames)."""
    rng = np.random.default_rng(seed)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in tracking.items() if not k.startswith("_")}
    n, n_players = out["players"].shape[:2]

    bucket = common.speed_bucket(tracking)
    extra = np.clip((bar_missing - common.missing_by_speed(tracking)) / (1.0 - common.missing_by_speed(tracking)), 0, 1)
    mean_len = INPLAY_GAP_MEAN_S * FPS
    visible = ~np.isnan(out["ball"][:, 0])
    starts = np.flatnonzero(visible & (rng.random(n) < extra[bucket] / mean_len))
    for start in starts:
        out["ball"][start: start + max(1, int(rng.exponential(mean_len)))] = np.nan

    out["players"][player_absence_mask(n, n_players, rng)] = np.nan
    return out


# -- validation copies ----------------------------------------------------------------------------

def prepare():
    """Build and cache the four damaged copies (BAR-ATH tracking statistics only, no label)."""
    os.makedirs(CACHE, exist_ok=True)
    bar = common.load_tracking()
    bar_missing = common.missing_by_speed(bar, common.graded_mask())
    for game in (1, 2):
        clean = load_game(game)
        periods = clean["period"]
        gt = with_period_gap(frame_labels(load_events(game), clean), periods)
        for seed in (0, 1):
            path = os.path.join(CACHE, "G{}s{}.npz".format(game, seed))
            if os.path.isfile(path):
                continue
            trk = degrade_faithful(clean, bar_missing, seed=seed)
            np.savez(path, gt=gt, runs=np.array(common.period_runs(periods)),
                     players=trk["players"], ball=trk["ball"], team=trk["team"])


def copy(key, smooth=damage.SMOOTH):
    """(tracking with the ball smoothed like the training data, truth, period runs) of one copy."""
    path = os.path.join(CACHE, key + ".npz")
    if not os.path.isfile(path):
        prepare()
    d = np.load(path)
    trk = {k: d[k] for k in ("players", "ball", "team")}
    trk["ball"] = damage.smooth_ball(trk["ball"], smooth)
    return trk, d["gt"], [tuple(r) for r in d["runs"].tolist()]


if __name__ == "__main__":
    download()
