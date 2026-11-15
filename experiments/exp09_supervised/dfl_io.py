"""DFL open dataset (7 Bundesliga matches, official TRACAB tracking) as the same dict as the other loaders.

Source: Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event data
in elite soccer", Scientific Data, CC-BY 4.0, figshare 10.6084/m9.figshare.28196177 (mirror:
huggingface.co/datasets/pysport/idsse-data). Downloaded by download.py into data/external/dfl/
(gitignored).

Why it matters: the ball frames carry, on every frame, the team in possession and the ball status
(in play / dead) of the official corrected tracking. The BAR-ATH ground truth is described as
exactly that ("a frame is D where the corrected tracking marks the ball dead, and otherwise takes
the team the tracking names as owning it"), so these are labels of the same kind, on seven matches.

    load_match(match_id) -> dict(players (N,P,3) x_m, y_m, speed; ball (N,3) x_m, y_m, z;
                                 team (P,) 0 home / 1 away; period (N,); label (N,) 0 H, 1 A, 2 D)
"""

import mmap
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "external", "dfl")
CACHE = os.path.join(os.path.dirname(HERE), "cache", "dfl")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
FPS = 25
SPEED_HALF_WINDOW = 5

_SET = re.compile(rb'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)">')
_PLAYER = re.compile(rb'<Frame N="(\d+)" T="[^"]*" X="([^"]*)" Y="([^"]*)"')
_BALL = re.compile(rb'<Frame N="(\d+)" T="[^"]*" X="([^"]*)" Y="([^"]*)" Z="([^"]*)"[^>]*'
                   rb'BallPossession="(\d)" BallStatus="(\d)"')
_PITCH = re.compile(rb'<PitchSize X="([\d.]+)" Y="([\d.]+)"')


def file_names(match_id):
    competition = "000001" if match_id in ("J03WMX", "J03WN1") else "000002"
    stem = "DFL-COM-{}_DFL-MAT-{}.xml".format(competition, match_id)
    return {"info": "DFL_02_01_matchinformation_" + stem, "positions": "DFL_04_03_positions_raw_observed_" + stem}


def _teams(match_id):
    root = ET.parse(os.path.join(DATA, file_names(match_id)["info"])).getroot()
    return {team.get("TeamId"): (0 if team.get("Role") == "home" else 1) for team in root.iter("Team")}


def _displacement_speed(pos, k=SPEED_HALF_WINDOW):
    speed = np.full(pos.shape[:-1], np.nan, dtype=np.float32)
    delta = pos[2 * k:] - pos[: -2 * k]
    speed[k:-k] = np.sqrt((delta ** 2).sum(axis=-1)) / (2.0 * k / FPS)
    return speed


def parse(match_id):
    teams = _teams(match_id)
    path = os.path.join(DATA, file_names(match_id)["positions"])
    sets = []
    with open(path, "rb") as handle:
        data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        pitch = tuple(float(v) for v in _PITCH.search(data).groups())
        headers = list(_SET.finditer(data))
        for i, header in enumerate(headers):
            section, team, person = (g.decode() for g in header.groups())
            end = headers[i + 1].start() if i + 1 < len(headers) else len(data)
            chunk = data[header.end(): end]
            if team == "BALL":
                rows = np.array(_BALL.findall(chunk), dtype=np.float64)
            elif team in teams:
                rows = np.array(_PLAYER.findall(chunk), dtype=np.float64)
            else:
                continue                                   # referees
            sets.append((section, team, person, rows))
        data.close()

    persons = sorted({(teams[t], p) for _, t, p, _ in sets if t != "BALL"})
    slot = {p: i for i, (_, p) in enumerate(persons)}
    blocks = []
    for period, section in ((1, "firstHalf"), (2, "secondHalf")):
        ball = next(rows for s, t, _, rows in sets if s == section and t == "BALL")
        first, n = int(ball[0, 0]), len(ball)
        players = np.full((n, len(persons), 3), np.nan, dtype=np.float32)
        for s, t, person, rows in sets:
            if s != section or t == "BALL" or len(rows) == 0:
                continue
            index = rows[:, 0].astype(np.int64) - first
            keep = (index >= 0) & (index < n)
            players[index[keep], slot[person], 0] = rows[keep, 1] + pitch[0] / 2
            players[index[keep], slot[person], 1] = rows[keep, 2] + pitch[1] / 2
        players[..., 2] = _displacement_speed(players[..., :2])
        ball_arr = np.column_stack([ball[:, 1] + pitch[0] / 2, ball[:, 2] + pitch[1] / 2, ball[:, 3]]).astype(np.float32)
        # BallPossession: 1 = home, 2 = away (checked against the nearest player, see README).
        label = np.where(ball[:, 5] == 0, 2, np.where(ball[:, 4] == 1, 0, 1)).astype(np.int8)
        blocks.append((players, ball_arr, np.full(n, period, dtype=np.int8), label))
    return {"players": np.concatenate([b[0] for b in blocks]), "ball": np.concatenate([b[1] for b in blocks]),
            "period": np.concatenate([b[2] for b in blocks]), "label": np.concatenate([b[3] for b in blocks]),
            "team": np.array([t for t, _ in persons], dtype=np.int8), "pitch": np.array(pitch, dtype=np.float32)}


def load_match(match_id):
    cache = os.path.join(CACHE, match_id + ".npz")
    if os.path.isfile(cache):
        data = np.load(cache)
        return {k: data[k] for k in data.files}
    out = parse(match_id)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cache, **out)
    return out


def download():
    import urllib.request
    os.makedirs(DATA, exist_ok=True)
    for match_id in MATCHES:
        for name in file_names(match_id).values():
            target = os.path.join(DATA, name)
            if not os.path.isfile(target):
                print("downloading", name, flush=True)
                urllib.request.urlretrieve(BASE_URL + name, target)


if __name__ == "__main__":
    download()
    for match in MATCHES:
        trk = load_match(match)
        label, ball, players, team = trk["label"], trk["ball"], trk["players"], trk["team"]
        delta = players[..., :2] - ball[:, None, :2]
        dist = np.where(np.isnan(delta[..., 0]), 1e9, np.sqrt((delta ** 2).sum(axis=2)))
        nearest = (dist[:, team == 1].min(axis=1) < dist[:, team == 0].min(axis=1)).astype(np.int8)
        alive = label < 2
        change = np.flatnonzero(np.diff(label)) + 1
        starts = np.concatenate(([0], change))
        n_poss = int((label[starts] < 2).sum())
        print("{}  frames {:>6}  players {:>2}  H/A/D {:.2f}/{:.2f}/{:.2f}  possessions {:>3} (mean {:.1f}s)  "
              "nearest player's team = label on {:.0%} of live frames".format(
                  match, len(label), len(team), *(np.bincount(label, minlength=3) / len(label)),
                  n_poss, alive.sum() / FPS / n_poss, (nearest[alive] == label[alive]).mean()), flush=True)
