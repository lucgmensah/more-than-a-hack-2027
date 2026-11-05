"""Metrica Sports public sample games 1 and 2 (CSV format) as the same dict the BAR-ATH loader gives.

Source: https://github.com/metrica-sports/sample-data (anonymised, no licence file; the README asks
that use be responsible and the source acknowledged if used publicly). The files are downloaded
into data/external/metrica/ which is gitignored; nothing from them is committed.

    python experiments/exp02_metrica_validation/download.py     # fetches games 1 and 2

Returned dict matches common.load_tracking(): players (N, P, 3) = x_m, y_m, speed; ball (N, 3);
team (P,) with 0 = home, 1 = away. Speed is derived from positions (the files have none) with the
same +/-5 frame displacement used for the ball.
"""

import os

import numpy as np
import pandas as pd

FPS = 25
PITCH = (105.0, 68.0)
SPEED_HALF_WINDOW = 5

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "external", "metrica")


def paths(game):
    folder = os.path.join(DATA, "Sample_Game_{}".format(game))
    stem = os.path.join(folder, "Sample_Game_{}_".format(game))
    return {"events": stem + "RawEventsData.csv",
            "home": stem + "RawTrackingData_Home_Team.csv",
            "away": stem + "RawTrackingData_Away_Team.csv"}


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
