"""How optimistic is the headline batch score?  Re-score the batches on accepted worlds that were NOT used to pick them.  (MODEL PREDICTIONS.)

design_batch.py splits the accepted worlds into 32 TRAIN worlds (used to choose the portfolio) and 32 TEST worlds (used for the headline
'expected best-of-batch yield').  The other accepted worlds (181 - 64 = 117) were never used, so they are a second, larger
out-of-selection sample.  This script evaluates the same batches on them, with the same device population (24 devices, seed 4242) and the same
switching specification, and reports

  * the expected best-of-batch yield on the 32 test worlds (from the cached matrix), on the 117 other worlds, and on all 149 non-training worlds;
  * the paired differences portfolio - mask, portfolio - seven published designs, mask + 3 interior additions - mask, with a bootstrap over worlds.

'Best-of-batch' assumes that after measurement the best design of the batch can be identified: it is an oracle-selection score, not the yield of a
randomly chosen device.  Writes results/other_worlds_check.json.   Usage: python scripts/other_worlds_check.py [--jobs 2]
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
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import design_batch as D  # noqa: E402
from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import optimization as opt  # noqa: E402

RES = ROOT / "results"


def best_of(Y):
    return Y.max(axis=0)


def boot_diff(a, b, B=4000, seed=0):
    """Mean and 95% percentile bootstrap interval of the paired per-world difference a - b (arrays over worlds)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return dict(mean=float(d.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--n_dev", type=int, default=24)
    a = ap.parse_args()

    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"]
    perm = np.random.default_rng(a.seed).permutation(len(acc))     # same split as design_batch.py
    n_tr = n_te = 32
    other = [acc[i] for i in perm[n_tr + n_te:]]

    Z = np.load(RES / "batch_yield_matrix.npz", allow_pickle=True)
    X, names, Ytr, Yte = Z["X"], [str(n) for n in Z["names"]], Z["Ytr"], Z["Yte"]
    idx, _ = opt.greedy_portfolio(Ytr, 16)
    seven = [names.index(f"anchor:{k}") for k in L.APL126_SEVEN]
    mask16 = [names.index(f"mask:{k}") for k in L.MASK_DESIGNS]
    rot8 = [i for i, n in enumerate(names) if n.startswith("rotfam")]
    add_all, _ = opt.greedy_portfolio(Ytr, 8, start=mask16)              # additions from the whole pool
    inside = np.array([(x[1] >= 0.6) and (4.5 <= x[2] <= 7.5) and (13.0 <= x[0] <= 19.0) for x in X])
    cand = [i for i in range(len(X)) if inside[i] and i not in mask16]
    sub = mask16 + cand
    add_in, _ = opt.greedy_portfolio(Ytr[sub], 8, start=list(range(len(mask16))))
    add_in = [sub[j] for j in add_in]

    sets = {
        "portfolio_16": list(idx),
        "portfolio_first8": list(idx[:8]),
        "mask_16": mask16,
        "seven_published": seven,
        "rotation_family_8": rot8,
        "mask_plus_3_pool": mask16 + list(add_all[:3]),
        "mask_plus_3_interior": mask16 + add_in[:3],
    }
    need = sorted({i for s in sets.values() for i in s})
    print(f"evaluating {len(need)} designs on {len(other)} other accepted worlds ...", flush=True)
    Yo_need = D.yield_matrix(X[need], other, a.n_dev, O.Spec(), jobs=a.jobs)
    Yo = {i: Yo_need[k] for k, i in enumerate(need)}

    out = dict(note="MODEL PREDICTIONS conditional on the literature anchors; oracle 'best-of-batch' score; not measurements",
               n_test_worlds=int(Yte.shape[1]), n_other_worlds=len(other), n_nontrain_worlds=int(Yte.shape[1]) + len(other),
               designs_evaluated=len(need), sets={})
    per_world = {}
    for name, ids in sets.items():
        te = best_of(Yte[ids])
        ot = best_of(np.array([Yo[i] for i in ids]))
        al = np.concatenate([te, ot])
        per_world[name] = al
        out["sets"][name] = dict(n_designs=len(ids), test=float(te.mean()), other=float(ot.mean()), nontrain=float(al.mean()),
                                 other_sd_over_worlds=float(ot.std(ddof=1)))
    out["differences_nontrain"] = {
        "portfolio_minus_mask": boot_diff(per_world["portfolio_16"], per_world["mask_16"]),
        "portfolio_minus_seven": boot_diff(per_world["portfolio_16"], per_world["seven_published"]),
        "mask_plus_3_interior_minus_mask": boot_diff(per_world["mask_plus_3_interior"], per_world["mask_16"]),
        "mask_plus_3_pool_minus_mask": boot_diff(per_world["mask_plus_3_pool"], per_world["mask_16"]),
        "mask_minus_seven": boot_diff(per_world["mask_16"], per_world["seven_published"]),
    }
    out["differences_other_only"] = {
        "portfolio_minus_mask": boot_diff(per_world["portfolio_16"][n_te:], per_world["mask_16"][n_te:]),
    }
    (RES / "other_worlds_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
