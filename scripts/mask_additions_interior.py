"""Which designs would add most to the author's mask if only designs INSIDE the conservative interior box are allowed?  (MODEL PREDICTIONS.)

design_batch.py chooses additions to the 16-design mask from the whole candidate pool; its first addition lies on a wall of the search
box (arm width 3.5 um, arm ratio 0.5), where the reduced model is least credible.  This script repeats the selection with candidates
restricted to the interior box of robustness_extras.py (arm ratio >= 0.6, 4.5 <= w <= 7.5 um, 13 <= L <= 19 um) and scores the mask plus
k additions on the same unseen test worlds.  It needs only the cached yield matrix (results/batch_yield_matrix.npz), so it runs in a second.
Writes results/mask_additions_interior.json.   Usage: python scripts/mask_additions_interior.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import optimization as opt  # noqa: E402

RES = ROOT / "results"


def score(Y):
    best = Y.max(axis=0)
    return dict(E_best_yield=float(best.mean()), P_some_design_ge_half=float((best >= 0.5).mean()))


def main():
    Z = np.load(RES / "batch_yield_matrix.npz", allow_pickle=True)
    X, names, Ytr, Yte = Z["X"], [str(n) for n in Z["names"]], Z["Ytr"], Z["Yte"]
    mask_i = [names.index(f"mask:{k}") for k in L.MASK_DESIGNS]
    inside = np.array([(x[1] >= 0.6) and (4.5 <= x[2] <= 7.5) and (13.0 <= x[0] <= 19.0) for x in X])
    cand = [i for i in range(len(X)) if inside[i] and i not in mask_i]
    sub = mask_i + cand
    add, _ = opt.greedy_portfolio(Ytr[sub], 8, start=list(range(len(mask_i))))
    chosen = [sub[j] for j in add]
    out = dict(note="MODEL PREDICTIONS conditional on the literature anchors; additions restricted to the interior box "
                    "(arm ratio >= 0.6, 4.5 <= w <= 7.5 um, 13 <= L <= 19 um); scored on the unseen test worlds",
               n_candidates=len(cand), mask_only=score(Yte[mask_i]),
               mask_plus_k=[dict(k=k + 1, **score(Yte[mask_i + chosen[: k + 1]])) for k in range(len(chosen))],
               additions=[dict(x=[float(v) for v in X[i]], origin=names[i], E_test=float(Yte[i].mean())) for i in chosen])
    (RES / "mask_additions_interior.json").write_text(json.dumps(out, indent=1))
    print("mask only:", out["mask_only"])
    for r in out["mask_plus_k"]:
        print(f"  +{r['k']}: E_best {r['E_best_yield']:.3f}  P(>=0.5) {r['P_some_design_ge_half']:.3f}")
    for a in out["additions"]:
        print("  add", np.round(a["x"], 2), a["origin"], f"E_test {a['E_test']:.2f}")


if __name__ == "__main__":
    main()
