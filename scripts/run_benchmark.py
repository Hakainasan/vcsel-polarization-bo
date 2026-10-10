"""Multi-seed benchmark: GP-EI / GP-UCB Bayesian optimisation vs random, Sobol and differential evolution.

All worlds are drawn from the *literature-consistent* set (results/anchored_worlds.json, made by scripts/anchored_worlds.py),
i.e. sets of model constants that reproduce the qualitative outcomes reported for fabricated tailored-aperture VCSELs.

Problems
--------
p1  "calibrated, default spec": the constants of the wafer are known (one accepted world), window 1.25-1.95.
p2  "calibrated, strict spec":  same worlds, narrow window 1.50-1.70 (harder: good designs are rarer).
p3  "prior-robust":             constants unknown; objective = mean yield over W accepted worlds (train), evaluated on
                                unseen accepted worlds. This is the problem solved before the first batch is measured.

Every method gets the same evaluation budget. All optimisers see only the smoothed yield on a fixed "training" device
population; every reported quality number is the hard yield of the returned design on a fresh, independent device
population (and, for p3, on unseen worlds) -> no optimism from re-using the population the optimiser saw.

Usage:  python scripts/run_benchmark.py p1 --seeds 8 --budget 48 --jobs 2
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import optimization as opt  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402
from vcsel_bo import problem as PR  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
METHODS = ["random", "sobol", "de", "bo_ei", "bo_ucb"]
STRICT = O.Spec(mu_lo=1.50, mu_hi=1.70)


def world_to_dict(w: P.World):
    return dict(d0=w.d0, b0=w.b0, k_d=w.k_d, k_b=w.k_b)


def dict_to_world(d):
    return P.World(d0=d["d0"], b0=d["b0"], k_d=d["k_d"], k_b=d["k_b"])


def accepted_pool(seed=2026):
    """Accepted worlds in a fixed random order (the split into benchmark / train / test worlds is by position)."""
    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"]
    rng = np.random.default_rng(seed)
    return [acc[i] for i in rng.permutation(len(acc))]


def select_worlds(pool, n_keep=5, n_draw=30, spec=O.Spec()):
    """Take n_draw accepted worlds, drop the infeasible ones (no design of a 128-point Sobol cloud reaches yield 0.6),
    keep n_keep that span the difficulty range (fraction of the cloud with yield > 0.5)."""
    cloud = qmc.Sobol(opt.DIM, scramble=True, seed=1).random(128)
    popn = P.Population.make(16, 0)
    cands = []
    for rec in pool[:n_draw]:
        w = dict_to_world(rec)
        y = np.array([O.evaluate_design(opt.to_design(u), w, popn, spec).hard_yield for u in cloud])
        cands.append(dict(world=rec, best=float(y.max()), frac_good=float(np.mean(y > 0.5))))
    feas = [c for c in cands if c["best"] >= 0.6]
    feas.sort(key=lambda c: c["frac_good"])
    idx = np.unique(np.round(np.linspace(0, len(feas) - 1, min(n_keep, len(feas)))).astype(int))
    return [feas[i] for i in idx], cands


def run_one(problem, widx, world_dict, method, seed, budget, prior_worlds=None, test_worlds=None):
    t0 = time.time()
    spec = STRICT if problem == "p2" else O.Spec()
    if problem == "p3":
        worlds = [dict_to_world(d) for d in prior_worlds]
        f = PR.PriorObjective(worlds, spec=spec)
    else:
        world = dict_to_world(world_dict)
        f = PR.YieldObjective(world, spec=spec)
    tr = opt.METHODS[method](f, budget, seed)
    hard_log = np.array([h for h, s in f.log])[: len(tr.y)]
    ub = tr.best_u
    if problem == "p3":
        tw = [dict_to_world(d) for d in test_worlds]
        pop = P.Population.make(96, 12345)
        held = float(np.mean([O.evaluate_design(opt.to_design(ub), w, pop, spec).hard_yield for w in tw]))
    else:
        held = PR.held_out_yield(ub, world, spec=spec).hard_yield
    return dict(problem=problem, world=widx, method=method, seed=seed, U=tr.U, y=tr.y, hard=hard_log,
                held_out=held, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", choices=["p1", "p2", "p3"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--n_worlds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    pool = accepted_pool()
    wfile = RES / "worlds.json"
    if args.problem in ("p1", "p2"):
        if wfile.exists():
            keep = json.loads(wfile.read_text())["selected"]
        else:
            keep, cands = select_worlds(pool, args.n_worlds)
            wfile.write_text(json.dumps(dict(selected=keep, all_candidates=cands), indent=1))
        print("worlds (frac_good, best):", [(round(k["frac_good"], 3), round(k["best"], 2)) for k in keep], flush=True)
        jobs = [(args.problem, i, k["world"], m, s, args.budget)
                for i, k in enumerate(keep) for s in range(args.seeds) for m in METHODS]
        kw = {}
    else:
        prior_worlds, test_worlds = pool[:8], pool[8:32]
        jobs = [("p3", 0, None, m, s, args.budget) for s in range(args.seeds) for m in METHODS]
        kw = dict(prior_worlds=prior_worlds, test_worlds=test_worlds)
        (RES / "p3_worlds.json").write_text(json.dumps(dict(train=prior_worlds, test=test_worlds), indent=1))

    # Results are checkpointed after every chunk of runs (a restart of the machine must not lose hours of work); a re-run resumes.
    partial = RES / f"benchmark_{args.problem}.partial.pkl"
    out = pickle.load(open(partial, "rb")) if partial.exists() else []
    have = {(r["problem"], r["world"], r["method"], r["seed"]) for r in out}
    todo = [j for j in jobs if (j[0], j[1], j[3], j[4]) not in have]
    print(f"{len(jobs)} runs, budget {args.budget}; {len(out)} already done, {len(todo)} to do", flush=True)
    t0 = time.time()
    chunk = 2 * args.jobs * len(METHODS)                     # about two (world, seed) groups per chunk keeps partial results balanced
    with Parallel(n_jobs=args.jobs) as par:
        for i in range(0, len(todo), chunk):
            out += par(delayed(run_one)(*j, **kw) for j in todo[i: i + chunk])
            tmp = partial.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(out, fh)
            tmp.replace(partial)
            print(f"[{time.time() - t0:.0f}s] {len(out)}/{len(jobs)} runs done", flush=True)
    with open(RES / f"benchmark_{args.problem}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
