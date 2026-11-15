"""Experiment 8: learn the teammate's model parameters on annotated Metrica, then test them.

    python experiments/exp08_friend_learned/run.py            # about 30-40 min on 6 worker processes
    python experiments/exp08_friend_learned/run.py --quick    # one copy per game, tiny grid

His model (friend.py, ported from his repository): per-frame Home / Away / Dead scores that are
weighted sums of hand-made features, a 3-state Viterbi with a team-switch penalty and a Dead
penalty, a minimum possession length, then a centred majority vote. He chose the 13 numbers by hand
and by a search on the 17 public possessions. Here they are learned on Metrica (about 290
possessions per game, labels from the event data):

  hand + search     his weights, decoder (switch, dead, min, window) chosen on Metrica
  logreg            weights learned by multinomial logistic regression on Metrica frames (his
                    features), decoder chosen on Metrica
  logreg + speeds   same, Dead may also use player and ball speed (in his feature store, not in his
                    emission)
  direct            every number (weights and decoder) optimised for spellF1 by an evolution strategy,
                    starting from "hand + search"

Development set: Metrica games 1 and 2 with the faithful damage of degrade.degrade_faithful (player
drop-outs, open-play ball losses matched to BAR-ATH by speed bucket, no position noise). Protocol:
fit and choose on one game, report on the other, both ways. The final models are fitted on both
games and run once on BAR-ATH; the public window is printed as a check, never used to choose.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
for _p in (HERE, os.path.join(EXPERIMENTS, "lib"), os.path.join(EXPERIMENTS, "exp02_metrica_validation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common  # noqa: E402
import degrade  # noqa: E402
import features  # noqa: E402
import friend  # noqa: E402
import ground_truth  # noqa: E402
import metrica_io  # noqa: E402
from methods import hmm_codes, period_runs, score, vote_baseline  # noqa: E402

OUT = os.path.join(common.OUT_DIR, "exp08")
CACHE = os.path.join(common.CACHE_DIR, "exp08")
GAMES = (1, 2)
SEEDS = (0, 1)
SWITCH = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DEAD = (0.0, 0.3, 1, 2, 4, 8, 16, 32, 64, 128, 256)
MIN_ALIVE = (0, 25, 50, 100)
WINDOW = (0, 25, 50, 75, 120)
HMM_ALPHAS = (0.1, 0.15, 0.25, 0.5)
HMM_NAMES = ("nearest", "ball_speed", "mean_speed")
ES_GENERATIONS, ES_OFFSPRING = 30, 14
KEEP = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "alive_dead_f1", "team_acc_gt_alive")
PUBLIC_WINDOW = (11185, 18685)

W = {}   # worker-side copies: key -> dict(own, dead, extra, observed, gt, runs)


# -- data -----------------------------------------------------------------------------------------

def prepare(seeds):
    """Damaged copies of both games, cached (gitignored). Returns {key: game}."""
    os.makedirs(CACHE, exist_ok=True)
    bar = common.load_tracking()
    bar_missing = degrade.missing_by_speed(bar, common.graded_mask())
    print("BAR-ATH ball missing by player-speed bucket:", np.round(bar_missing, 3))
    keys = {}
    for game in GAMES:
        clean = metrica_io.load_game(game)
        periods = clean["period"]
        gt = ground_truth.with_period_gap(
            ground_truth.frame_labels(metrica_io.load_events(game), clean), periods)
        print("G{} clean  missing by bucket:".format(game), np.round(degrade.missing_by_speed(clean), 3))
        for seed in seeds:
            key = "G{}s{}".format(game, seed)
            keys[key] = game
            path = os.path.join(CACHE, key + ".npz")
            trk = degrade.degrade_faithful(clean, bar_missing, seed=seed)
            print("{} damaged missing by bucket: {}  overall {:.3f}  players/frame {:.1f}".format(
                key, np.round(degrade.missing_by_speed(trk), 3), np.isnan(trk["ball"][:, 0]).mean(),
                (~np.isnan(trk["players"][..., 0])).sum(axis=1).mean()), flush=True)
            if not os.path.isfile(path):
                feat = friend.frame_features(trk, features.ball_speed(trk))
                np.savez(path, gt=gt, runs=np.array(period_runs(periods)), periods=periods,
                         players=trk["players"], ball=trk["ball"], team=trk["team"],
                         vote50=vote_baseline(trk, 50, periods), vote120=vote_baseline(trk, 120, periods),
                         **{k: feat[k] for k in ("own", "dead", "extra", "observed")})
    return keys


def load_light(key):
    data = np.load(os.path.join(CACHE, key + ".npz"))
    out = {k: data[k] for k in ("own", "dead", "extra", "observed", "gt", "vote50", "vote120")}
    out["runs"] = [tuple(r) for r in data["runs"].tolist()]
    return out


def init_worker(keys):
    for key in keys:
        W[key] = load_light(key)


def metrics_of(gt, pred):
    result = score(gt, pred)
    return {k: float(result[k]) for k in KEEP}


# -- worker tasks ---------------------------------------------------------------------------------

def grid_task(task):
    label, params, extra, key, switch, dead = task
    c = W[key]
    decoded = friend.viterbi(friend.emissions(c, params, extra), switch, dead, c["runs"])
    rows = []
    for min_alive in MIN_ALIVE:
        merged = friend.merge_short_alive_runs(decoded, min_alive, c["runs"])
        for window in WINDOW:
            rows.append(dict(variant=label, key=key, switch=switch, dead=dead, min_alive=min_alive,
                             window=window, **metrics_of(c["gt"], friend.majority(merged, window))))
    return rows


def eval_task(task):
    params, decoder, extra, key = task
    c = W[key]
    decoded = friend.viterbi(friend.emissions(c, params, extra), decoder["switch"], decoder["dead"], c["runs"])
    pred = friend.post(decoded, decoder["min_alive"], decoder["window"], c["runs"])
    return metrics_of(c["gt"], pred)


def hmm_task(task):
    key, alpha = task
    data = np.load(os.path.join(CACHE, key + ".npz"))
    trk = {k: data[k] for k in ("players", "ball", "team")}
    runs = [tuple(r) for r in data["runs"].tolist()]
    codes = hmm_codes(trk, HMM_NAMES, alpha, 15.0, runs, 8, 5, min_dwell_s=3.0)
    return dict(variant="hmm", alpha=alpha, key=key, **metrics_of(data["gt"], codes))


# -- learning -------------------------------------------------------------------------------------

def fit_logreg(keys, extra, step=5, l2=1e-4):
    """Multinomial logistic regression with his symmetric structure: H = w.f, A = -w.f, D = v.g + b."""
    own, dead, y = [], [], []
    for key in keys:
        c = load_light(key)
        sel = np.flatnonzero(c["gt"] < 3)[::step]
        own.append(c["own"][sel])
        dead.append(np.hstack([c["dead"], c["extra"]])[sel] if extra else c["dead"][sel])
        y.append(c["gt"][sel])
    own, dead, y = np.vstack(own).astype(np.float64), np.vstack(dead).astype(np.float64), np.concatenate(y)
    n, k, m = len(y), own.shape[1], dead.shape[1]
    J = np.zeros((n, 3, k + m + 1))
    J[:, 0, :k], J[:, 1, :k] = own, -own
    J[:, 2, k:k + m], J[:, 2, -1] = dead, 1.0
    target = np.eye(3)[y]
    reg = np.full(k + m + 1, l2 * n)
    reg[-1] = 0.0
    theta = np.zeros(k + m + 1)
    for _ in range(100):
        z = np.einsum("nsd,d->ns", J, theta)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        grad = np.einsum("ns,nsd->d", p - target, J) + reg * theta
        jp = np.einsum("ns,nsd->nd", p, J)
        hess = np.einsum("ns,nsd,nse->de", p, J, J) - jp.T @ jp + np.diag(reg)
        delta = np.linalg.solve(hess, grad)
        theta -= delta
        if np.abs(delta).max() < 1e-7:
            break
    z = np.einsum("nsd,d->ns", J, theta)
    acc = float((z.argmax(axis=1) == y).mean())
    return {"own": theta[:k], "dead": theta[k:k + m], "bias": float(theta[-1])}, acc


def choose(grid, variant, keys):
    """Decoder with the best mean spellF1 over `keys` (ties: macroF1)."""
    sub = grid[(grid.variant == variant) & grid.key.isin(keys)]
    mean = sub.groupby(["switch", "dead", "min_alive", "window"])[["spell_f1_mean", "macro_f1"]].mean()
    best = mean.sort_values(["spell_f1_mean", "macro_f1"], ascending=False).index[0]
    decoder = {"switch": float(best[0]), "dead": float(best[1]), "min_alive": int(best[2]), "window": int(best[3])}
    return decoder, float(mean.loc[best, "spell_f1_mean"])


def mutate(params, decoder, rng):
    p = {"own": params["own"].copy(), "dead": params["dead"].copy(), "bias": params["bias"]}
    for name in ("own", "dead"):
        for i in range(len(p[name])):
            if rng.random() < 0.5:
                p[name][i] += rng.normal(0, 0.25) * max(abs(p[name][i]), 0.1)
    if rng.random() < 0.5:
        p["bias"] += rng.normal(0, 0.25) * max(abs(p["bias"]), 0.2)
    d = dict(decoder)
    if rng.random() < 0.5:
        d["switch"] = float(d["switch"] * np.exp(rng.normal(0, 0.4)))
    if rng.random() < 0.5:
        d["dead"] = float(d["dead"] + rng.normal(0, 0.3) * max(abs(d["dead"]), 0.3))
    for name, values in (("min_alive", MIN_ALIVE + (150,)), ("window", WINDOW)):
        if rng.random() < 0.25:
            i = list(values).index(d[name]) + rng.choice((-1, 1))
            d[name] = values[int(np.clip(i, 0, len(values) - 1))]
    return p, d


def evolve(pool, keys, params, decoder, generations, offspring, seed=0):
    """(1 + lambda) evolution strategy on mean spellF1 over `keys`."""
    rng = np.random.default_rng(seed)

    def fitness(candidates):
        tasks = [(p, d, False, k) for p, d in candidates for k in keys]
        res = pool.map(eval_task, tasks)
        return [np.mean([r["spell_f1_mean"] for r in res[i * len(keys):(i + 1) * len(keys)]])
                for i in range(len(candidates))]

    best = (params, decoder)
    best_fit = fitness([best])[0]
    for generation in range(generations):
        candidates = [mutate(best[0], best[1], rng) for _ in range(offspring)]
        fits = fitness(candidates)
        i = int(np.argmax(fits))
        if fits[i] > best_fit:
            best, best_fit = candidates[i], fits[i]
        if generation % 5 == 4 or generation == generations - 1:
            print("    ES gen {:>2}: train spellF1 {:.3f}".format(generation + 1, best_fit), flush=True)
    return best, best_fit


# -- main -----------------------------------------------------------------------------------------

def describe(params):
    return "own {}  dead {}  bias {:.2f}".format(np.round(params["own"], 2), np.round(params["dead"], 2),
                                                 params["bias"])


def main():
    global SWITCH, DEAD, ES_GENERATIONS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    seeds = SEEDS[:1] if args.quick else SEEDS
    if args.quick:
        SWITCH, DEAD, ES_GENERATIONS = (0.5, 4, 32), (0.3, 4, 16), 2
    started = time.time()
    keys = prepare(seeds)
    all_keys = sorted(keys)
    by_game = {g: [k for k in all_keys if keys[k] == g] for g in GAMES}
    directions = {"G1->G2": (by_game[1], by_game[2]), "G2->G1": (by_game[2], by_game[1]),
                  "both": (all_keys, [])}
    os.makedirs(OUT, exist_ok=True)
    pool = Pool(args.workers, initializer=init_worker, initargs=(all_keys,))
    copies = {k: load_light(k) for k in all_keys}

    rows = []   # one row per (variant, direction, key) on the TEST side + train summary
    def record(variant, direction, decoder, params, train_fit, test_keys, test_results, extra_info=""):
        for key, r in zip(test_keys, test_results):
            rows.append(dict(variant=variant, direction=direction, key=key, train_spellF1=train_fit,
                             decoder=str(decoder), params=describe(params) if params else extra_info, **r))

    # 1. fixed methods, nothing to choose: vote baselines and his published configuration.
    for window in (50, 120):
        for key in all_keys:
            rows.append(dict(variant="vote w=%d" % window, direction="fixed", key=key,
                             **metrics_of(copies[key]["gt"], copies[key]["vote%d" % window])))
    res = pool.map(eval_task, [(friend.HAND, friend.PUBLISHED, False, k) for k in all_keys])
    record("his published config", "fixed", friend.PUBLISHED, friend.HAND, np.nan, all_keys, res)
    print("[{:.0f}s] baselines done".format(time.time() - started), flush=True)

    # 2. weights: his, and learned by logistic regression per direction.
    weights = {("hand", d): friend.HAND for d in directions}
    for extra, name in ((False, "logreg"), (True, "logreg+speeds")):
        for direction, (train, _) in directions.items():
            params, acc = fit_logreg(train, extra)
            weights[(name, direction)] = params
            print("  {:<14} {:<7} frame acc {:.3f}  {}".format(name, direction, acc, describe(params)), flush=True)

    # 3. decoder grid on the training copies of each (weights, direction).
    tasks, seen = [], set()
    for (name, direction), params in weights.items():
        train = directions[direction][0]
        for key in train:
            label = name if name == "hand" else "{}|{}".format(name, direction)
            if (label, key) in seen:
                continue
            seen.add((label, key))
            extra = name == "logreg+speeds"
            tasks += [(label, params, extra, key, s, d) for s in SWITCH for d in DEAD]
    grid = pd.DataFrame([r for chunk in pool.imap_unordered(grid_task, tasks, chunksize=2) for r in chunk])
    grid.to_csv(os.path.join(OUT, "grid.csv"), index=False)
    print("[{:.0f}s] grid done ({} decodes)".format(time.time() - started, len(tasks)), flush=True)

    chosen = {}
    for (name, direction), params in weights.items():
        train, test = directions[direction]
        label = name if name == "hand" else "{}|{}".format(name, direction)
        decoder, fit = choose(grid, label, train)
        chosen[(name, direction)] = (params, decoder, fit)
        variant = {"hand": "hand + search"}.get(name, name)
        if test:
            res = pool.map(eval_task, [(params, decoder, name == "logreg+speeds", k) for k in test])
            record(variant, direction, decoder, params, fit, test, res)
        print("  {:<14} {:<7} decoder {}  train {:.3f}".format(name, direction, decoder, fit), flush=True)

    # 4. direct optimisation of every number, starting from hand + search.
    for direction, (train, test) in directions.items():
        params, decoder, _ = chosen[("hand", direction)]
        (params, decoder), fit = evolve(pool, train, params, decoder, ES_GENERATIONS, ES_OFFSPRING)
        chosen[("direct", direction)] = (params, decoder, fit)
        print("  direct {:<7} {}  {}  train {:.3f}".format(direction, describe(params), decoder, fit), flush=True)
        if test:
            record("direct", direction, decoder, params, fit, test,
                   pool.map(eval_task, [(params, decoder, False, k) for k in test]))
    print("[{:.0f}s] ES done".format(time.time() - started), flush=True)

    # 5. our HMM on the same copies, alpha chosen on the training game.
    hmm = pd.DataFrame(pool.map(hmm_task, [(k, a) for k in all_keys for a in HMM_ALPHAS]))
    hmm.to_csv(os.path.join(OUT, "hmm.csv"), index=False)
    hmm_alpha = {}
    for direction, (train, test) in directions.items():
        mean = hmm[hmm.key.isin(train)].groupby("alpha").spell_f1_mean.mean()
        alpha = float(mean.idxmax())
        hmm_alpha[direction] = alpha
        for key in test:
            r = hmm[(hmm.key == key) & (hmm.alpha == alpha)].iloc[0]
            rows.append(dict(variant="HMM (ours)", direction=direction, key=key, train_spellF1=mean.max(),
                             decoder="alpha %g dwell 3s" % alpha, **{k: r[k] for k in KEEP}))
        a015 = hmm[hmm.key.isin(test) & (hmm.alpha == 0.15)]
        for _, r in a015.iterrows():
            rows.append(dict(variant="HMM (ours) alpha 0.15", direction=direction, key=r.key,
                             **{k: r[k] for k in KEEP}))
    pool.close()
    print("[{:.0f}s] HMM done".format(time.time() - started), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "transfer.csv"), index=False)
    summary = summarise(table, by_game)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\nMetrica, spellF1 on the game NOT used to fit/choose (fixed methods: mean over the copies of each game)")
    print(summary.round(3).to_string(index=False))

    final_bar_ath(chosen, hmm_alpha["both"])
    print("[{:.0f}s] done".format(time.time() - started))


def summarise(table, by_game):
    out = []
    for variant, sub in table.groupby("variant", sort=False):
        if (sub.direction == "fixed").all():
            g2 = sub[sub.key.isin(by_game[2])]
            g1 = sub[sub.key.isin(by_game[1])]
        else:
            g2 = sub[sub.direction == "G1->G2"]
            g1 = sub[sub.direction == "G2->G1"]
        out.append({"variant": variant,
                    "G1->G2 spellF1": g2.spell_f1_mean.mean(), "G2->G1 spellF1": g1.spell_f1_mean.mean(),
                    "mean spellF1": (g2.spell_f1_mean.mean() + g1.spell_f1_mean.mean()) / 2,
                    "macroF1": (g2.macro_f1.mean() + g1.macro_f1.mean()) / 2,
                    "poss": (g2.n_pred_spells.mean() + g1.n_pred_spells.mean()) / 2,
                    "gt poss": (g2.n_gt_spells.mean() + g1.n_gt_spells.mean()) / 2,
                    "a/d F1": (g2.alive_dead_f1.mean() + g1.alive_dead_f1.mean()) / 2})
    return pd.DataFrame(out).sort_values("mean spellF1", ascending=False)


def final_bar_ath(chosen, hmm_alpha):
    """Fitted on both Metrica games, run once on BAR-ATH. Public window = report only."""
    bar = common.load_tracking()
    feat = friend.frame_features(bar, features.ball_speed(bar))
    tags = friend.tag_features(common.smart_tags(), len(bar["ball"]))
    runs = common.half_runs()
    gt = common.pm.label_codes(common.pm.read_label_csv(common.GT_PUBLIC))
    lo, hi = PUBLIC_WINDOW
    rows = []

    def run(name, params, decoder, extra=False, use_tags=False, decode_runs=runs, write=True):
        em = friend.emissions(feat, params, extra, tags if use_tags else None)
        if use_tags:   # his ambiguity term includes the tag prior
            amb = np.clip(1 - np.abs(feat["own"] @ friend.HAND["own"] + friend.TAG_OWN * tags["prior"]), 0, 1)
            em[:, 2] += params["dead"][2] * (amb - feat["dead"][:, 2]) * feat["observed"]
        decoded = friend.viterbi(em, decoder["switch"], decoder["dead"], decode_runs)
        codes = common.mask_to_graded(friend.post(decoded, decoder["min_alive"], decoder["window"], decode_runs))
        r = common.score_public(codes)
        sliced = common.pm.evaluate(gt[lo:hi], codes[lo:hi])
        share = np.bincount(codes[common.graded_mask()], minlength=4)[:3] / common.graded_mask().sum()
        rows.append({"run": name, "public spellF1": r["spell_f1_mean"], "public macroF1": r["macro_f1"],
                     "public spellF1 (his slicing)": sliced["spell_f1_mean"], "poss": r["n_pred_spells"],
                     "a/d F1": r["alive_dead_f1"], "H": share[0], "A": share[1], "D": share[2],
                     "decoder": str(decoder)})
        if write:
            common.write_submission(os.path.join(OUT, name.replace(" ", "_").replace("+", "plus") + ".csv"), codes)

    run("port check: his published, tags, public-window decode", friend.HAND, friend.PUBLISHED,
        use_tags=True, decode_runs=[PUBLIC_WINDOW], write=False)
    run("his published + tags", friend.HAND, friend.PUBLISHED, use_tags=True)
    run("his published tracking only", friend.HAND, friend.PUBLISHED)
    for name, label in (("hand", "hand + search"), ("logreg", "logreg"), ("logreg+speeds", "logreg+speeds"),
                        ("direct", "direct")):
        params, decoder, _ = chosen[(name, "both")]
        run(label, params, decoder, extra=name == "logreg+speeds")
    params, decoder, _ = chosen[("hand", "both")]
    run("hand + search + his tags", params, decoder, use_tags=True)

    codes = hmm_codes(bar, HMM_NAMES, hmm_alpha, 15.0, runs, 8, 5, min_dwell_s=3.0)
    codes = common.mask_to_graded(codes)
    r = common.score_public(codes)
    rows.append({"run": "HMM (ours) alpha %g" % hmm_alpha, "public spellF1": r["spell_f1_mean"],
                 "public macroF1": r["macro_f1"], "poss": r["n_pred_spells"], "a/d F1": r["alive_dead_f1"]})
    common.write_submission(os.path.join(OUT, "HMM_alpha%g.csv" % hmm_alpha), codes)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "bar_ath_public.csv"), index=False)
    print("\nBAR-ATH, models fitted on both Metrica games. Public window (17 possessions): a check, not a choice")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
