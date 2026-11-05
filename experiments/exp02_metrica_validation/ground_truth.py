"""A frame-level H/A/D label for the Metrica games, derived from their human event data.

This is a PROXY for the challenge's ground truth (which comes from corrected tracking). It follows
the challenge's own rules as far as events allow:

* Alive: the team of the latest on-ball event (PASS, SHOT, RECOVERY, SET PIECE). A BALL LOST or a
  CHALLENGE does not move possession by itself: in loose phases possession stays with the last team
  that had the ball, and the other team gets it at its RECOVERY.
* Dead: from a stoppage event (BALL OUT, FAULT RECEIVED, a goal or a shot out of play, offside, hand
  ball, end of half) until the next event where the ball is played again (SET PIECE, PASS or SHOT).
  A SHOT stops play at its END frame, the others at their start frame.
* Outside the span covered by events of a period: X.

Absolute scores against it are not comparable with the BAR-ATH numbers, because the definition of
"controlled possession" differs and events are human-coded. It is used to compare METHODS on a
sample of hundreds of possessions instead of 17. This is allowed: it is hand-made data about
OTHER games (RULES.md, "Hand-labelled data from other games is fine").
"""

import numpy as np

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
    """Copy of a per-row array where the first row of every period after the first is X.

    possession_metrics treats X in the ground truth as 'not graded' and encodes runs separately, so
    a single X between the two halves stops a possession from spanning half-time.
    """
    out = np.array(array, copy=True)
    starts = np.flatnonzero(np.diff(periods) != 0) + 1
    out[starts] = X
    return out
