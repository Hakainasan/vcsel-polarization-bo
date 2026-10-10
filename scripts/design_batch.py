"""Design a fabrication batch that hedges over what is not yet known about the wafer (model predictions, NOT measurements).

Worlds = the literature-consistent set (results/anchored_worlds.json): constants of the model that reproduce the
qualitative outcomes reported for fabricated tailored-aperture VCSELs. They are split into disjoint TRAIN and TEST worlds.

1. candidate pool = Sobol points + designs found by prior-robust GP-EI BO (objective = mean smoothed yield over train
                    worlds) + the nine anchor designs + the 16 designs of the author's mask + a rotation family;
2. yield matrix   = hard yield of every candidate in every world (train and test);
3. batch          = greedy portfolio (maximise E_w[max_j yield_j,w]) chosen on TRAIN worlds only;
4. report         = expected best-of-batch yield on unseen TEST worlds, against (i) the seven designs of APL 126,
                    (ii) the author's mask, (iii) a rotation-only family, (iv) random / Sobol picks from the pool;
                    plus the best k designs to ADD to the author's existing mask.

Output: results/batch_design.csv (mask parameters for the layout script), results/batch_summary.json,
        results/batch_yield_matrix.npz
Usage:  python scripts/design_batch.py --q 16 --sobol 128 --train 32 --test 32
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
from scipy.stats import qmc  # noqa: E402

from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import optimization as opt  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402
from vcsel_bo import problem as PR  # noqa: E402

RES = ROOT / "results"


def _row(x, worlds_rec, n_dev, spec):
    pop = P.Population.make(n_dev, 4242)
    return [O.evaluate_design(x, P.World(d0=r["d0"], b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]), pop, spec).hard_yield
            for r in worlds_rec]


def yield_matrix(X, worlds_rec, n_dev, spec, jobs=2):
    rows = Parallel(n_jobs=jobs)(delayed(_row)(x, worlds_rec, n_dev, spec) for x in X)
    return np.array(rows)


def rotation_family(n=8):
    """Rotation-only reference family (one 1:0.7 cross, n orientations)."""
    return np.array([[17.5, 0.7, 6.0, 0.5, th] for th in np.linspace(0, 180, n, endpoint=False)])


def batch_score(Yb):
    """Expected best-of-batch yield, and probability that some design of the batch reaches yield >= 0.5."""
    best = Yb.max(axis=0)
    return dict(E_best_yield=float(best.mean()), P_some_design_ge_half=float((best >= 0.5).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=16)
    ap.add_argument("--sobol", type=int, default=128)
    ap.add_argument("--train", type=int, default=32)
    ap.add_argument("--test", type=int, default=32)
    ap.add_argument("--bo_budget", type=int, default=40)
    ap.add_argument("--bo_runs", type=int, default=2)
    ap.add_argument("--n_dev", type=int, default=24)
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()
    spec = O.Spec()
    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(acc))
    train = [acc[i] for i in perm[: a.train]]
    test = [acc[i] for i in perm[a.train: a.train + a.test]]
    (RES / "batch_worlds.json").write_text(json.dumps(dict(train=train, test=test), indent=1))

    # ---- candidate pool --------------------------------------------------------------------------------
    names, X = [], []
    for u in qmc.Sobol(opt.DIM, scramble=True, seed=a.seed).random(a.sobol):
        names.append("sobol"); X.append(opt.to_design(u))
    tw = [P.World(d0=r["d0"], b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]) for r in train[:12]]
    for s in range(a.bo_runs):
        f = PR.PriorObjective(tw, n_dev=12, pop_seed=s, spec=spec)
        tr = opt.run_bo(f, a.bo_budget, seed=s)
        for i in np.argsort(-tr.y)[:8]:
            names.append(f"bo{s}"); X.append(opt.to_design(tr.U[i]))
        print(f"BO run {s}: best mean soft yield {tr.y.max():.3f}", flush=True)
    for k, d in L.DESIGNS.items():
        names.append(f"anchor:{k}"); X.append(d.as_array())
    for k, d in L.MASK_DESIGNS.items():
        names.append(f"mask:{k}"); X.append(d.as_array())
    for th in np.linspace(0, 180, 8, endpoint=False):
        names.append("rotfam"); X.append(np.array([17.5, 0.7, 6.0, 0.5, th]))
    X = np.array(X)
    print("pool:", len(X), flush=True)

    Ytr = yield_matrix(X, train, a.n_dev, spec)
    Yte = yield_matrix(X, test, a.n_dev, spec)
    np.savez(RES / "batch_yield_matrix.npz", X=X, names=np.array(names), Ytr=Ytr, Yte=Yte)

    idx, sc = opt.greedy_portfolio(Ytr, a.q)
    kind = lambda pref: [i for i, n in enumerate(names) if n.startswith(pref)]
    seven = [names.index(f"anchor:{k}") for k in L.APL126_SEVEN]
    mask16 = [names.index(f"mask:{k}") for k in L.MASK_DESIGNS]
    rot8 = kind("rotfam")

    rand_scores = [batch_score(Yte[np.random.default_rng(r).choice(len(X), a.q, replace=False)])["E_best_yield"]
                   for r in range(50)]
    single = int(np.argmax(Ytr.mean(1)))
    # augmentation of the author's mask: k extra designs chosen on TRAIN worlds
    add, add_sc = opt.greedy_portfolio(Ytr, 8, start=mask16)
    summary = dict(
        note="MODEL PREDICTIONS conditional on literature anchors; worlds = accepted set; not measurements",
        q=a.q, n_pool=len(X), n_train_worlds=len(train), n_test_worlds=len(test), n_dev=a.n_dev,
        greedy_train_score_path=sc,
        test=dict(
            greedy_portfolio=batch_score(Yte[idx]),
            greedy_first8=batch_score(Yte[idx[:8]]),
            apl126_seven_designs=batch_score(Yte[seven]),
            author_mask_16_designs=batch_score(Yte[mask16]),
            rotation_family_8=batch_score(Yte[rot8]),
            random_q_from_pool=dict(mean=float(np.mean(rand_scores)), sd=float(np.std(rand_scores))),
            single_best_train_design=dict(E_yield=float(Yte[single].mean()), P_ge_half=float((Yte[single] >= 0.5).mean()),
                                          x=X[single].tolist()),
            mask_plus_k=[dict(k=k + 1, **batch_score(Yte[mask16 + add[: k + 1]])) for k in range(len(add))],
            n_test_worlds_with_no_working_design=int((Yte.max(axis=0) < 0.3).sum()),
        ),
        mask_additions=[dict(x=X[i].tolist(), origin=names[i]) for i in add],
        per_design_mask=[dict(name=names[i], E_test=float(Yte[i].mean()), P_ge_half=float((Yte[i] >= 0.5).mean()))
                         for i in mask16],
    )
    (RES / "batch_summary.json").write_text(json.dumps(summary, indent=1))
    hdr = "id,origin,L_um,arm_ratio,arm_width_um,corner_rho,corner_radius_um,rotation_deg,E_yield_train,E_yield_test,P_yield_ge_half_test"
    rows = []
    for k, i in enumerate(idx):
        L_, r_, w_, rho, th = X[i]
        rows.append(f"{k},{names[i]},{L_:.3f},{r_:.3f},{w_:.3f},{rho:.3f},{rho * w_ / 2:.3f},{th:.2f},"
                    f"{Ytr[i].mean():.3f},{Yte[i].mean():.3f},{(Yte[i] >= 0.5).mean():.3f}")
    (RES / "batch_design.csv").write_text(hdr + "\n" + "\n".join(rows) + "\n")
    print(json.dumps(summary["test"], indent=1))


if __name__ == "__main__":
    main()
