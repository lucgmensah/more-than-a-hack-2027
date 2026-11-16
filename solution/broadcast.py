"""The broadcast recording as a second sensor: load its tracking and align it to the tactical timeline.

The TV feed is not synchronised to the tactical video and has cuts and replays. Both trackings give
ball and player positions in the same normalised pitch coordinates, so the alignment is found from
the data: for a chunk of broadcast frames, the offset at which its ball track best matches the
tactical ball track (masked sum of squared differences, computed for every offset at once by FFT).

    python solution/broadcast.py        # prints the offsets and what they recover
"""

import os

import numpy as np

import common
from games import asset_paths

CACHE = os.path.join(common.CACHE_DIR, "broadcast.npz")
PITCH = np.array(common.PITCH, dtype=np.float32)
CHUNK = 1500          # frames of broadcast matched at a time (1 min)
MAX_MATCH_M = 3.0     # median ball distance below which a chunk is considered aligned


def load():
    """dict(players (N, 51, 3) metres, ball (N, 3), team (51,), valid (N,) homography usable)."""
    if os.path.isfile(CACHE):
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    paths = asset_paths(source="broadcast")
    team = common.team_of_slot(paths["atd_meta"])
    players, ball = [], []
    with open(paths["atd_raw"]) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32).reshape(-1, 3)
            row = np.full((len(team), 3), np.nan, dtype=np.float32)
            row[: len(values)] = values[: len(team)]
            players.append(row)
            ball.append(np.array(parts[2].split(","), dtype=np.float32))
    players, ball = np.array(players), np.array(ball)
    players[..., :2] *= PITCH
    ball[:, :2] *= PITCH
    valid = np.loadtxt(paths["homography"], delimiter=",", skiprows=1, usecols=11).astype(bool)
    out = {"players": players, "ball": ball, "team": team, "valid": valid[: len(ball)]}
    np.savez_compressed(CACHE, **out)
    return out


def _masked_ssd(chunk, full):
    """For every offset o: mean squared distance between chunk[i] and full[i + o] where both exist,
    and the number of such frames. chunk (L, 2), full (N, 2), NaN = missing."""
    n = len(full) + len(chunk)
    size = 1 << int(np.ceil(np.log2(n)))
    mc, mf = np.isfinite(chunk[:, 0]), np.isfinite(full[:, 0])
    total = np.zeros(size)
    fm = np.fft.rfft(mf.astype(float), size)
    cm = np.conj(np.fft.rfft(mc.astype(float), size))
    count = np.fft.irfft(fm * cm, size)
    for c in range(2):
        a = np.where(mc, chunk[:, c], 0.0).astype(float)
        b = np.where(mf, full[:, c], 0.0).astype(float)
        fa, fa2 = np.conj(np.fft.rfft(a, size)), np.conj(np.fft.rfft(a * a, size))
        fb, fb2 = np.fft.rfft(b, size), np.fft.rfft(b * b, size)
        total += np.fft.irfft(fa2 * fm, size) + np.fft.irfft(cm * fb2, size) - 2 * np.fft.irfft(fa * fb, size)
    count = np.round(count)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.inf), count


def align(broadcast, tactical, verbose=False):
    """tactical frame of every broadcast frame (-1 where no reliable alignment), chunk by chunk."""
    b_ball, t_ball = broadcast["ball"][:, :2], tactical["ball"][:, :2]
    n_b, n_t = len(b_ball), len(t_ball)
    mapping = np.full(n_b, -1, dtype=np.int64)
    rows = []
    for start in range(0, n_b, CHUNK):
        chunk = b_ball[start: start + CHUNK]
        seen = int(np.isfinite(chunk[:, 0]).sum())
        if seen < 150:
            rows.append((start, None, seen, np.nan))
            continue
        ssd, count = _masked_ssd(chunk, t_ball)
        size = len(ssd)
        offsets = np.arange(size)
        offsets = np.where(offsets >= n_t, offsets - size, offsets)             # negative lags wrap around
        # `position` = tactical frame matched to the first frame of the chunk
        ok = (count >= 0.6 * seen) & (offsets >= 0) & (offsets + len(chunk) <= n_t)
        if not ok.any():
            rows.append((start, None, seen, np.nan))
            continue
        position = int(offsets[int(np.argmin(np.where(ok, ssd, np.inf)))])
        idx = np.arange(len(chunk))
        both = np.isfinite(chunk[:, 0]) & np.isfinite(t_ball[position + idx, 0])
        median = float(np.median(np.linalg.norm(chunk[both] - t_ball[position + idx][both], axis=1)))
        rows.append((start, position - start, seen, median))
        if median <= MAX_MATCH_M:
            mapping[start + idx] = position + idx
    if verbose:
        for start, offset, seen, median in rows:
            print("  broadcast {:>6}..: offset {}  ball frames {:>4}  median distance {:.2f} m".format(
                start, offset, seen, median))
    return mapping, rows


def offsets(broadcast, tactical):
    """The recording has one cut (half-time): (offset before, offset after, first broadcast frame after the cut)."""
    _, rows = align(broadcast, tactical)
    good = np.array([(start, off) for start, off, _, median in rows if off is not None and median <= MAX_MATCH_M])
    first = int(np.bincount(good[good[:, 1] < good[:, 1].mean(), 1]).argmax())
    second = int(np.bincount(good[good[:, 1] > good[:, 1].mean(), 1]).argmax())
    last_first = good[np.abs(good[:, 1] - first) <= 15, 0].max()
    first_second = good[np.abs(good[:, 1] - second) <= 15, 0].min()
    # refine the cut with short blocks between the last chunk of one side and the first of the other
    b_ball, t_ball = broadcast["ball"][:, :2], tactical["ball"][:, :2]
    cut = first_second
    for start in range(last_first, first_second + CHUNK, 125):
        idx = np.arange(start, start + 125)
        cost = []
        for off in (first, second):
            both = np.isfinite(b_ball[idx, 0]) & np.isfinite(t_ball[idx + off, 0])
            cost.append(np.median(np.linalg.norm(b_ball[idx][both] - t_ball[idx + off][both], axis=1))
                        if both.sum() >= 15 else np.nan)
        if np.isfinite(cost[1]) and (not np.isfinite(cost[0]) or cost[1] < cost[0]) and cost[1] < MAX_MATCH_M:
            cut = start
            break
    return first, second, int(cut)


def mapping(broadcast, tactical):
    """Tactical frame of every broadcast frame, and the three constants it is built from."""
    first, second, cut = offsets(broadcast, tactical)
    idx = np.arange(len(broadcast["ball"]))
    return np.where(idx < cut, idx + first, idx + second), (first, second, cut)


def fill_ball(tactical, broadcast=None, max_jump_m=None):
    """Tactical tracking whose missing ball is taken from the aligned broadcast tracking where it has one."""
    broadcast = broadcast or load()
    frames, constants = mapping(broadcast, tactical)
    out = dict(tactical)
    ball = tactical["ball"].copy()
    ok = (frames >= 0) & (frames < len(ball)) & np.isfinite(broadcast["ball"][:, 0])
    target = frames[ok]
    empty = np.isnan(ball[target, 0])
    ball[target[empty], :2] = broadcast["ball"][ok][empty, :2]
    out["ball"] = ball
    return out, constants


def main():
    tactical = common.load_tracking()
    broadcast = load()
    print("broadcast: {} frames, ball on {:.1%}, homography usable on {:.1%}".format(
        len(broadcast["ball"]), np.isfinite(broadcast["ball"][:, 0]).mean(), broadcast["valid"].mean()))
    frames, (first, second, cut) = mapping(broadcast, tactical)
    print("tactical frame = broadcast frame + {} before broadcast frame {}, + {} after (half-time is cut out)".format(
        first, cut, second))
    graded = common.graded_mask()
    t_missing = np.isnan(tactical["ball"][:, 0])
    n = len(t_missing)
    inside = (frames >= 0) & (frames < n)
    covered, usable, have = (np.zeros(n, dtype=bool) for _ in range(3))
    covered[frames[inside]] = True
    usable[frames[inside & broadcast["valid"]]] = True
    b_has = np.isfinite(broadcast["ball"][:, 0])
    have[frames[inside & b_has]] = True
    print("graded tactical frames covered by the broadcast: {:.1%}".format(covered[graded].mean()))
    print("tactical ball missing on graded frames: {:.1%}; the broadcast has a ball on {:.1%} of those".format(
        t_missing[graded].mean(), have[graded & t_missing].mean()))
    print("ball missing after the fill: {:.1%}".format((t_missing & ~have)[graded].mean()))
    print("broadcast shows the pitch (usable homography) on {:.1%} of graded frames: "
          "{:.1%} when the tactical ball is missing, {:.1%} when it is visible".format(
              usable[graded].mean(), usable[graded & t_missing].mean(), usable[graded & ~t_missing].mean()))
    both = inside & b_has
    both[both] = np.isfinite(tactical["ball"][frames[both], 0])
    dist = np.linalg.norm(broadcast["ball"][both, :2] - tactical["ball"][frames[both], :2], axis=1)
    print("where both see the ball: distance median {:.2f} m, 90% below {:.2f} m, above 5 m on {:.1%}".format(
        np.median(dist), np.percentile(dist, 90), (dist > 5).mean()))
    filled, _ = fill_ball(tactical, broadcast)
    print("ball missing by player-speed bucket, before:", np.round(common.missing_by_speed(tactical, graded), 3))
    print("ball missing by player-speed bucket, after: ", np.round(common.missing_by_speed(filled, graded), 3))


if __name__ == "__main__":
    main()
