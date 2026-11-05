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


def _initial_tables(n_symbols):
    """Domain-anchored starting tables, one (3, n_symbols) matrix per channel.

    Channel semantics are fixed by features.py. The starting point is deliberately soft; EM moves
    it. Only the *direction* of each effect is encoded: 'home nearest' favours H, 'away nearest'
    favours A, fast ball and fast players favour alive, set-piece window and missing ball favour D.
    """
    tables = {}
    for name, size in n_symbols.items():
        base = np.ones((N_STATES, size))
        if name == "nearest":
            for symbol in range(12):
                team, dist_bucket = symbol // 6, (symbol % 6) // 2
                closeness = 1.0 / (1 + dist_bucket)
                base[H if team == 0 else A, symbol] += 2.0 * closeness
                base[D, symbol] += 0.5 * closeness
            base[D, 12] += 2.0
            base[[H, A], 12] += 0.5
        elif name == "ball_speed":
            base[D, 0] += 1.0
            base[D, 1] += 1.0
            base[[H, A], 2:] += 1.0
        elif name == "set_piece":
            base[D, 1] += 3.0
        elif name == "phase_team":
            base[H, 1] += 1.0
            base[A, 2] += 1.0
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
    if name == "nearest":
        perm[:12] = (np.arange(12) + 6) % 12
    elif name == "phase_team":
        perm[1], perm[2] = 2, 1
    return perm


# Symbols that mean "the ball was not seen". Left free, EM discovers that this is the strongest
# structure in the data and turns state D into a "ball missing" detector (observed: D swallowed the
# 41% of graded frames with no ball, and its table put 0.99 on 'missing'). That fits the
# observations well and answers the wrong question -- whether the ball is VISIBLE, not whether it
# is DEAD. The challenge's own data say occlusion is common in live play, so a missing ball is tied
# to be equally likely in every state: it carries no information and the other channels decide.
MISSING_SYMBOL = {"nearest": 12, "ball_speed": 0}


def _tie_missing(table, name, counts=None):
    """Make P(missing | state) the same for every state, rescaling the other symbols to fit."""
    sym = MISSING_SYMBOL.get(name)
    if sym is None:
        return table
    weight = counts if counts is not None else table
    share = weight[:, sym].sum() / weight.sum()
    rest = 1.0 - table[:, sym]
    table = table * ((1.0 - share) / rest)[:, None]
    table[:, sym] = share
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


def fit(symbols_by_run, names, n_symbols, alpha=0.25, iterations=15, verbose=True,
        learn_transitions=False, possession_s=MEAN_POSSESSION_S):
    """EM on one or several independent runs (the two halves). Returns (tables, trans).

    Emission tables are learned (H/A tied by mirror symmetry). Transitions stay at the declared
    dwell-time prior unless learn_transitions is set.
    """
    tables = {n: _tie_missing(t, n) for n, t in _initial_tables(
        {n: n_symbols[n] for n in names}).items()}
    trans = initial_transitions(possession_s=possession_s)
    start = np.array([0.0, 0.0, 1.0]) + 1e-3    # play begins from a kickoff, i.e. a dead ball
    start /= start.sum()

    for step in range(iterations):
        counts = {n: np.full((N_STATES, n_symbols[n]), 1.0) for n in names}   # Dirichlet(1) smoothing
        xi_total = np.full((N_STATES, N_STATES), 1e-3)
        total = 0.0
        for symbols in symbols_by_run:
            gamma, xi, loglik = _forward_backward(
                _log_emission(symbols, tables, names, alpha), trans, start)
            xi_total += xi
            total += loglik
            for column, name in enumerate(names):
                for state in range(N_STATES):
                    counts[name][state] += np.bincount(
                        symbols[:, column], weights=gamma[:, state], minlength=n_symbols[name])
        if learn_transitions:
            trans = xi_total / xi_total.sum(axis=1, keepdims=True)
        tables = {n: _tie_missing(
            _symmetrise(counts[n] / counts[n].sum(axis=1, keepdims=True), n), n, counts[n])
            for n in names}
        if verbose:
            print("    EM {:>2}/{}  loglik {:.0f}  mean stay: H {:.1f}s A {:.1f}s D {:.1f}s".format(
                step + 1, iterations, total,
                *(1.0 / (1.0 - trans[s, s]) / FPS for s in STATES)), flush=True)
    return tables, trans


def viterbi(symbols, tables, trans, names, alpha=0.25):
    """Most probable state sequence for one run, as label codes (0=H, 1=A, 2=D)."""
    log_em = _log_emission(symbols, tables, names, alpha)
    log_trans = np.log(trans + 1e-12)
    T = log_em.shape[0]
    delta = np.log(np.array([1e-3, 1e-3, 1.0])) + log_em[0]
    back = np.empty((T, N_STATES), dtype=np.int8)
    for t in range(1, T):
        scores = delta[:, None] + log_trans
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + log_em[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_match(channels, runs, names, n_symbols, alpha=0.25, iterations=15, verbose=True,
                 learn_transitions=False, possession_s=MEAN_POSSESSION_S):
    """Fit on the graded halves, decode each. Returns (codes over all frames, tables, trans).

    Ungraded frames get X: they are never scored, and the halves are decoded independently so the
    halftime hole cannot create a phantom state change.
    """
    symbols_by_run = [channels[start:stop] for start, stop in runs]
    tables, trans = fit(symbols_by_run, names, n_symbols, alpha, iterations, verbose,
                        learn_transitions, possession_s)
    codes = np.full(channels.shape[0], 3, dtype=np.int8)
    for (start, stop), symbols in zip(runs, symbols_by_run):
        codes[start:stop] = viterbi(symbols, tables, trans, names, alpha)
    return codes, tables, trans
