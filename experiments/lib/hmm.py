"""A 3-state HMM (H / A / D) with categorical observation channels, fitted by EM, decoded by Viterbi.

Why this and not a bigger model: the ranking metric rewards whole possessions, and a Viterbi
decode is the global, non-causal version of what the baseline's centred vote approximates. It
decides all the frames of a possession together, trading the evidence against a cost for changing
state. Missing ball frames carry the 'missing' symbol, which the model learns to treat as weakly
informative, so they no longer have to be guessed.

No labels are used anywhere. EM fits the transition matrix and the emission tables on the match's
own observations (unsupervised learning is allowed on the provided game); only the initial tables
encode domain knowledge, and only to anchor which state is which.

Tempering: at 25 fps consecutive frames are strongly autocorrelated, so treating channels and
frames as independent makes the model overconfident. Emission log-likelihoods are multiplied by
`alpha` < 1 in both EM and decoding. alpha is a declared modelling choice (default 0.25), reported
with a sensitivity sweep, never selected on the ground truth.
"""

import numpy as np

from common import D, H, A, FPS

STATES = (H, A, D)          # state index == label code
N_STATES = 3


def _is_team_ball_channel(name):
    """Channels shaped (team x distance bucket x margin) + 'ball missing': nearest, nearest_interp, ttb."""
    return name.startswith(("nearest", "ttb"))


def _initial_tables(n_symbols):
    """Domain-anchored starting tables, one (3, n_symbols) matrix per channel.

    Channel semantics are fixed by features.py. The starting point is deliberately soft; EM moves
    it. Only the *direction* of each effect is encoded: 'home nearest' favours H, 'away nearest'
    favours A, fast ball and fast players favour alive, set-piece window and missing ball favour D.
    """
    tables = {}
    for name, size in n_symbols.items():
        base = np.ones((N_STATES, size))
        if name == "count_diff":
            for symbol in range(5):
                base[H, symbol] += max(0, symbol - 2)
                base[A, symbol] += max(0, 2 - symbol)
        elif _is_team_ball_channel(name):
            per_team = (size - 1) // 2
            for symbol in range(size - 1):
                team, dist_bucket = symbol // per_team, (symbol % per_team) // 2
                closeness = 1.0 / (1 + dist_bucket)
                base[H if team == 0 else A, symbol] += 2.0 * closeness
                base[D, symbol] += 0.5 * closeness
            base[D, size - 1] += 2.0
            base[[H, A], size - 1] += 0.5
        elif name == "ball_speed":
            base[D, 0] += 1.0
            base[D, 1] += 1.0
            base[[H, A], 2:] += 1.0
        elif name.startswith("set_piece"):
            base[D, 1::2] += 3.0                       # symbols 1 (and 3 when crossed with visibility)
        elif name.startswith("phase_team"):
            base[H, 1::3] += 1.0
            base[A, 2::3] += 1.0
        elif name == "mean_speed":
            base[D, 0:2] += 1.0
            base[[H, A], 2:] += 1.0
        tables[name] = base / base.sum(axis=1, keepdims=True)
    return tables


# Declared dwell-time prior, from football common sense and never from the ground truth:
# a team keeps the ball ~15 s before the other takes it; play runs ~22 s between stoppages; a
# stoppage lasts ~12 s. (With those, the ball is dead ~35% of the time, the usual figure for
# top-flight matches.) EM does NOT learn these: on unlabeled data the transition matrix just
# reproduces the flicker of the evidence (it converged to 3 s possessions), which is exactly the
# failure the smoothing is there to remove.
MEAN_POSSESSION_S = 15.0
MEAN_ALIVE_BEFORE_STOP_S = 22.0
MEAN_STOPPAGE_S = 12.0


def initial_transitions(possession_s=MEAN_POSSESSION_S, alive_s=MEAN_ALIVE_BEFORE_STOP_S,
                        stoppage_s=MEAN_STOPPAGE_S):
    """Transition matrix from the three declared dwell times."""
    trans = np.zeros((N_STATES, N_STATES))
    trans[H, A] = trans[A, H] = 1.0 / (possession_s * FPS)
    trans[H, D] = trans[A, D] = 1.0 / (alive_s * FPS)
    trans[D, H] = trans[D, A] = 0.5 / (stoppage_s * FPS)
    for s in range(N_STATES):
        trans[s, s] = 1.0 - trans[s].sum()
    return trans


def _mirror(name, size):
    """Symbol permutation that swaps home and away for a channel (identity if it has no team)."""
    perm = np.arange(size)
    if _is_team_ball_channel(name):
        half = (size - 1) // 2
        perm[: size - 1] = (np.arange(size - 1) + half) % (size - 1)
    elif name == "count_diff":
        perm[:5] = 4 - np.arange(5)
    elif name.startswith("phase_team"):
        for k in range(size // 3):                     # swap "home phase" and "away phase" in each block
            perm[3 * k + 1], perm[3 * k + 2] = 3 * k + 2, 3 * k + 1
    return perm


# Symbols that mean "the ball was not seen". Left free, EM discovers that this is the strongest
# structure in the data and turns state D into a "ball missing" detector (observed: D swallowed the
# 41% of graded frames with no ball, and its table put 0.99 on 'missing'). That fits the
# observations well and answers the wrong question -- whether the ball is VISIBLE, not whether it
# is DEAD. The challenge's own data say occlusion is common in live play, so a missing ball is tied
# to be equally likely in every state: it carries no information and the other channels decide.
def _missing_symbol(name, size):
    """Index of the 'ball not seen' symbol of a channel, or None if the channel has none."""
    if name == "ball_speed":
        return 0
    if _is_team_ball_channel(name) or name == "count_diff":
        return size - 1
    return None


def _tie_missing(table, name, counts=None):
    """Make P(missing | state) the same for every state, rescaling the other symbols to fit."""
    sym = _missing_symbol(name, table.shape[1])
    if sym is None:
        return table
    weight = counts if counts is not None else table
    share = weight[:, sym].sum() / weight.sum()
    rest = 1.0 - table[:, sym]
    table = table * ((1.0 - share) / rest)[:, None]
    table[:, sym] = share
    return table


def _tie_prior_only(table, name, counts):
    """Channels named '...@prior' tell H from A but say nothing about dead ball: D gets the marginal."""
    if name.endswith("@prior"):
        table[D] = counts.sum(axis=0) / counts.sum()
    return table


def _symmetrise(table, name):
    """Force P(symbol | A) to be the mirror image of P(symbol | H).

    Nothing in the physics makes the home side's evidence behave differently from the away side's,
    and tying them halves the parameters and keeps EM from letting one team's state drift into D.
    """
    perm = _mirror(name, table.shape[1])
    mean_h = 0.5 * (table[H] + table[A][perm])        # table[A][perm][k] == table[A][mirror(k)]
    table[H] = mean_h
    table[A] = np.empty_like(mean_h)
    table[A][perm] = mean_h
    return table


def _log_emission(symbols, tables, names, alpha):
    """(T, 3) tempered log-likelihood of each frame under each state (channels multiplied)."""
    out = np.zeros((symbols.shape[0], N_STATES))
    for column, name in enumerate(names):
        out += np.log(tables[name])[:, symbols[:, column]].T
    return alpha * out


def _forward_backward(log_em, trans, start):
    """Scaled forward-backward. Returns gamma (T,3), summed xi (3,3), log-likelihood."""
    T = log_em.shape[0]
    shift = log_em.max(axis=1, keepdims=True)
    em = np.exp(log_em - shift)
    alpha_f = np.empty((T, N_STATES))
    scale = np.empty(T)
    vec = start * em[0]
    scale[0] = vec.sum()
    alpha_f[0] = vec / scale[0]
    for t in range(1, T):
        vec = (alpha_f[t - 1] @ trans) * em[t]
        scale[t] = vec.sum()
        alpha_f[t] = vec / scale[t]
    beta = np.empty((T, N_STATES))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = trans @ (em[t + 1] * beta[t + 1]) / scale[t + 1]
    gamma = alpha_f * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((N_STATES, N_STATES))
    for t in range(T - 1):
        xi += alpha_f[t][:, None] * trans * (em[t + 1] * beta[t + 1] / scale[t + 1])[None, :]
    loglik = float(np.sum(np.log(scale)) + shift.sum())
    return gamma, xi, loglik


def _blocks(T, block):
    """Start index and length of each block of `block` frames covering T frames."""
    starts = np.arange(0, T, block)
    lengths = np.minimum(block, T - starts)
    return starts, lengths


def _block_log_em(log_em, block):
    """Sum the (already tempered) frame log-likelihoods inside blocks. block=1 is the identity.

    A block is one HMM step. With block=5 (0.2 s) the forward-backward and Viterbi loops run 5x
    fewer steps, which is what makes grids of experiments affordable; the state can only change at
    block boundaries, a 0.2 s resolution that is far below the metric's tolerance.
    """
    if block == 1:
        return log_em, None
    starts, lengths = _blocks(log_em.shape[0], block)
    return np.add.reduceat(log_em, starts, axis=0), lengths


def fit(symbols_by_run, names, n_symbols, alpha=0.25, iterations=15, verbose=True,
        learn_transitions=False, possession_s=MEAN_POSSESSION_S, block=1):
    """EM on one or several independent runs (the two halves). Returns (tables, trans).

    Emission tables are learned (H/A tied by mirror symmetry). Transitions stay at the declared
    dwell-time prior unless learn_transitions is set (not supported together with block > 1).
    """
    if learn_transitions and block != 1:
        raise ValueError("learn_transitions needs block=1")
    tables = {n: _tie_missing(t, n) for n, t in _initial_tables(
        {n: n_symbols[n] for n in names}).items()}
    trans = initial_transitions(possession_s=possession_s)
    start = np.array([0.0, 0.0, 1.0]) + 1e-3    # play begins from a kickoff, i.e. a dead ball
    start /= start.sum()

    for step in range(iterations):
        counts = {n: np.full((N_STATES, n_symbols[n]), 1.0) for n in names}   # Dirichlet(1) smoothing
        xi_total = np.full((N_STATES, N_STATES), 1e-3)
        total = 0.0
        trans_step = np.linalg.matrix_power(trans, block) if block > 1 else trans
        for symbols in symbols_by_run:
            log_em, lengths = _block_log_em(_log_emission(symbols, tables, names, alpha), block)
            gamma, xi, loglik = _forward_backward(log_em, trans_step, start)
            if lengths is not None:
                gamma = np.repeat(gamma, lengths, axis=0)      # every frame inherits its block's posterior
            xi_total += xi
            total += loglik
            for column, name in enumerate(names):
                for state in range(N_STATES):
                    counts[name][state] += np.bincount(
                        symbols[:, column], weights=gamma[:, state], minlength=n_symbols[name])
        if learn_transitions:
            trans = xi_total / xi_total.sum(axis=1, keepdims=True)
        tables = {n: _tie_prior_only(_tie_missing(
            _symmetrise(counts[n] / counts[n].sum(axis=1, keepdims=True), n), n, counts[n]),
            n, counts[n]) for n in names}
        if verbose:
            print("    EM {:>2}/{}  loglik {:.0f}  mean stay: H {:.1f}s A {:.1f}s D {:.1f}s".format(
                step + 1, iterations, total,
                *(1.0 / (1.0 - trans[s, s]) / FPS for s in STATES)), flush=True)
    return tables, trans


def _with_minimum_dwell(log_em, log_trans, m):
    """Expand each state into a chain of m sub-states so no stay can be shorter than m steps.

    Sub-state i of state s moves to sub-state i+1 with probability 1; the last one either stays
    (the original self-transition) or leaves to the FIRST sub-state of another state (the original
    switch probabilities). The first m-1 steps of any stay are therefore forced, which is the
    simplest explicit-duration model: a minimum length, then the usual geometric tail.
    """
    n = N_STATES * m
    big_trans = np.full((n, n), -np.inf)
    for s in range(N_STATES):
        for i in range(m - 1):
            big_trans[s * m + i, s * m + i + 1] = 0.0
        last = s * m + m - 1
        big_trans[last, last] = log_trans[s, s]
        for other in range(N_STATES):
            if other != s:
                big_trans[last, other * m] = log_trans[s, other]
    return np.repeat(log_em, m, axis=1), big_trans


def viterbi(symbols, tables, trans, names, alpha=0.25, block=1, min_dwell_s=0.0):
    """Most probable state sequence for one run, as label codes (0=H, 1=A, 2=D).

    min_dwell_s > 0 forbids any stay shorter than that (rounded to whole blocks), i.e. no
    possession and no stoppage shorter than min_dwell_s can appear in the output.
    """
    n_frames = symbols.shape[0]
    log_em, lengths = _block_log_em(_log_emission(symbols, tables, names, alpha), block)
    trans = np.linalg.matrix_power(trans, block) if block > 1 else trans
    log_trans = np.log(trans + 1e-12)
    m = max(1, int(round(min_dwell_s * FPS / block))) if min_dwell_s > 0 else 1
    if m > 1:
        log_em, log_trans = _with_minimum_dwell(log_em, log_trans, m)
        first = np.full(N_STATES * m, -np.inf)
        first[D * m] = 0.0
        first_prior = first
    else:
        first_prior = np.log(np.array([1e-3, 1e-3, 1.0]))
    n_states = log_em.shape[1]
    T = log_em.shape[0]
    delta = first_prior + log_em[0]
    back = np.empty((T, n_states), dtype=np.int16)
    for t in range(1, T):
        scores = delta[:, None] + log_trans
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + log_em[t]
    path = np.empty(T, dtype=np.int16)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    path = (path // m).astype(np.int8)          # sub-state -> state (a no-op when m == 1)
    if lengths is not None:
        path = np.repeat(path, lengths)[:n_frames]
    return path


def semi_viterbi(symbols, tables, trans, names, alpha=0.25, block=5, mean_s=(15.0, 15.0, 12.0),
                 shape=2.0, min_s=3.0, max_s=90.0):
    """Explicit-duration decoding: each stay has a gamma-shaped duration (mean `mean_s` per state,
    zero below `min_s`), instead of the geometric one implied by a Markov self-transition.

    Which state follows is taken from the off-diagonal of the declared transition matrix.
    Emissions are the HMM's (tempered, summed over blocks). Returns label codes per frame.
    """
    n_frames = symbols.shape[0]
    log_em, lengths = _block_log_em(_log_emission(symbols, tables, names, alpha), block)
    T = log_em.shape[0]
    cum = np.vstack([np.zeros((1, N_STATES)), np.cumsum(log_em, axis=0)])
    off = np.array(trans, dtype=float)
    np.fill_diagonal(off, 0.0)
    log_switch = np.log(off / off.sum(axis=1, keepdims=True) + 1e-12)
    np.fill_diagonal(log_switch, -np.inf)

    dmax = int(max_s * FPS / block)
    dmin = max(1, int(round(min_s * FPS / block)))
    x = np.arange(1, dmax + 1) * block / FPS
    log_dur = np.full((N_STATES, dmax), -np.inf)
    for st in range(N_STATES):
        logpdf = (shape - 1.0) * np.log(x) - x * shape / mean_s[st]
        logpdf[:dmin - 1] = -np.inf
        log_dur[st] = logpdf - np.log(np.exp(logpdf).sum())

    V = np.full((T + 1, N_STATES), -np.inf)        # best score of a path whose last segment ends at t
    B = np.full((T + 1, N_STATES), -np.inf)        # best score of starting a segment of state s at t
    Bprev = np.zeros((T + 1, N_STATES), dtype=np.int8)
    best_d = np.zeros((T + 1, N_STATES), dtype=np.int32)
    B[0] = np.log(np.array([1e-3, 1e-3, 1.0]))
    Bprev[0] = -1
    for t in range(1, T + 1):
        for st in range(N_STATES):
            d = np.arange(dmin, min(dmax, t) + 1)
            if len(d) == 0:
                continue
            cand = B[t - d, st] - cum[t - d, st] + log_dur[st, d - 1]
            k = int(cand.argmax())
            V[t, st] = cum[t, st] + cand[k]
            best_d[t, st] = d[k]
        # starting a segment of state s at t follows the best segment of another state ending at t
        sc = V[t][:, None] + log_switch
        B[t] = sc.max(axis=0)
        Bprev[t] = sc.argmax(axis=0)
    path = np.empty(T, dtype=np.int8)
    t, st = T, int(V[T].argmax())
    while t > 0:
        d = int(best_d[t, st])
        path[t - d:t] = st
        t, st = t - d, int(Bprev[t - d, st])
    if lengths is not None:
        path = np.repeat(path, lengths)[:n_frames]
    return path


def confidence(symbols, tables, trans, names, alpha=0.25, block=1, start=None):
    """Per-frame smoothed posterior of the most likely state (no minimum dwell): how sure the model is."""
    n_frames = symbols.shape[0]
    log_em, lengths = _block_log_em(_log_emission(symbols, tables, names, alpha), block)
    trans = np.linalg.matrix_power(trans, block) if block > 1 else trans
    start = np.array([1e-3, 1e-3, 1.0]) if start is None else start
    gamma, _, _ = _forward_backward(log_em, trans, start / start.sum())
    conf = gamma.max(axis=1)
    if lengths is not None:
        conf = np.repeat(conf, lengths)[:n_frames]
    return conf


def decode_match(channels, runs, names, n_symbols, alpha=0.25, iterations=15, verbose=True,
                 learn_transitions=False, possession_s=MEAN_POSSESSION_S, block=1,
                 min_dwell_s=0.0):
    """Fit on the graded halves, decode each. Returns (codes over all frames, tables, trans).

    Ungraded frames get X: they are never scored, and the halves are decoded independently so the
    halftime hole cannot create a phantom state change.
    """
    symbols_by_run = [channels[start:stop] for start, stop in runs]
    tables, trans = fit(symbols_by_run, names, n_symbols, alpha, iterations, verbose,
                        learn_transitions, possession_s, block)
    codes = np.full(channels.shape[0], 3, dtype=np.int8)
    for (start, stop), symbols in zip(runs, symbols_by_run):
        codes[start:stop] = viterbi(symbols, tables, trans, names, alpha, block, min_dwell_s)
    return codes, tables, trans
