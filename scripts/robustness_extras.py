"""Two robustness checks on the designed batch that the main script does not make (model predictions, NOT measurements).

1. BOUNDARY CHECK.  Bayesian optimisation of a reduced model tends to push designs to the walls of the search box, where
   the reduced model is least credible (thin arms, low arm ratio).  This script (a) lists how many designs of the
   portfolio sit on a wall, (b) reports the oxide-aperture area and equivalent diameter of every portfolio design in the
   placeholder process, and (c) re-selects a portfolio from the candidates inside a conservative interior box
   (arm ratio >= 0.6, 4.5 <= w <= 7.5 um, 13 <= L <= 19 um) and scores it on the same unseen test worlds.
2. FRESH DEVICES.  The yield matrices use one fixed device population (common random numbers).  Here the selected sets
   are re-scored on the SAME test worlds but a NEW population of 48 devices (different seed), so the reported gap between
   the portfolio and the reference sets cannot come from over-fitting to particular device draws.

Reads results/batch_yield_matrix.npz, batch_worlds.json, batch_summary.json.  Writes results/robustness_extras.json.
Usage: python scripts/robustness_extras.py [--jobs 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed  # noqa: E402

from vcsel_bo import geometry as G  # noqa: E402
from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import optimization as opt  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402

RES = ROOT / "results"
EPS = 0.02        # a coordinate within 2% of the box width of a wall counts as "on the wall"


def on_wall(x):
    u = opt.to_unit(x)
    u = u[:4]                                             # rotation is periodic: no wall
    return [n for n, v in zip(opt.NAMES[:4], u) if v < EPS or v > 1 - EPS]


def score(Y):
    best = Y.max(0)
    return dict(E_best_yield=float(best.mean()), P_some_design_ge_half=float((best >= 0.5).mean()))


def _row(x, recs, n_dev, seed, spec):
    pop = P.Population.make(n_dev, seed)
    return [O.evaluate_design(x, P.World(d0=r["d0"], b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]), pop, spec).hard_yield for r in recs]


def fresh_matrix(X, recs, n_dev, seed, spec, jobs):
    return np.array(Parallel(n_jobs=jobs)(delayed(_row)(x, recs, n_dev, seed, spec) for x in X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--fresh_dev", type=int, default=48)
    ap.add_argument("--fresh_seed", type=int, default=9999)
    a = ap.parse_args()
    spec = O.Spec()
    Z = np.load(RES / "batch_yield_matrix.npz", allow_pickle=True)
    X, names, Ytr, Yte = Z["X"], [str(n) for n in Z["names"]], Z["Ytr"], Z["Yte"]
    W = json.loads((RES / "batch_worlds.json").read_text())
    test = W["test"]
    rows = np.genfromtxt(RES / "batch_design.csv", delimiter=",", names=True, dtype=None, encoding=None)
    Xp = np.array([[r["L_um"], r["arm_ratio"], r["arm_width_um"], r["corner_rho"], r["rotation_deg"]] for r in rows])
    q = len(Xp)

    # ---- 1a/1b: walls and aperture size of the portfolio ----------------------------------------------------------
    proc = G.Process()
    info = []
    for k, x in enumerate(Xp):
        ap_ = G.build_aperture(G.MaskDesign(*x), proc)
        info.append(dict(id=k, x=[float(v) for v in x], on_wall=on_wall(x), aperture_area_um2=float(ap_.area),
                         equivalent_diameter_um=float(ap_.equivalent_diameter), mesa_area_um2=float(ap_.mesa_area)))
    n_wall = sum(1 for d in info if d["on_wall"])
    # reference apertures for scale
    ref = {}
    for k, d in list(L.DESIGNS.items()) + list(L.MASK_DESIGNS.items()):
        a_ = G.build_aperture(d, proc)
        ref[k] = dict(aperture_area_um2=float(a_.area), equivalent_diameter_um=float(a_.equivalent_diameter))

    # ---- 1c: interior-only portfolio ------------------------------------------------------------------------------
    inside = np.array([(x[1] >= 0.6) and (4.5 <= x[2] <= 7.5) and (13.0 <= x[0] <= 19.0) for x in X])
    idx_in = np.where(inside)[0]
    sel, _ = opt.greedy_portfolio(Ytr[idx_in], q)
    sel = idx_in[np.array(sel)]
    Xi = X[sel]
    seven_i = [names.index(f"anchor:{k}") for k in L.APL126_SEVEN]
    mask_i = [names.index(f"mask:{k}") for k in L.MASK_DESIGNS]
    res = dict(
        note="MODEL PREDICTIONS; see docstring", n_pool=len(X), n_pool_interior=int(inside.sum()), q=q,
        portfolio_designs_on_a_wall=n_wall, portfolio_info=info, reference_apertures=ref,
        interior_box="arm ratio >= 0.6, 4.5 <= w <= 7.5 um, 13 <= L <= 19 um",
        interior_portfolio=dict(designs=[[float(v) for v in x] for x in Xi], origin=[names[i] for i in sel],
                                test_same_devices=score(Yte[sel])),
        portfolio_same_devices=score(np.array([Yte[np.argmin(np.abs(X - x).sum(1))] for x in Xp])),
    )

    # ---- 2: fresh device population --------------------------------------------------------------------------------
    sets = dict(portfolio=Xp, interior_portfolio=Xi, author_mask=X[mask_i], apl126_seven=X[seven_i])
    fresh = {}
    for k, Xs in sets.items():
        Yf = fresh_matrix(Xs, test, a.fresh_dev, a.fresh_seed, spec, a.jobs)
        fresh[k] = score(Yf)
        print(k, fresh[k], flush=True)
    res["test_fresh_devices"] = dict(n_dev=a.fresh_dev, seed=a.fresh_seed, **fresh)
    (RES / "robustness_extras.json").write_text(json.dumps(res, indent=1))
    print(f"portfolio designs on a wall: {n_wall}/{q}")
    print("interior portfolio (same devices):", res["interior_portfolio"]["test_same_devices"])


if __name__ == "__main__":
    main()
