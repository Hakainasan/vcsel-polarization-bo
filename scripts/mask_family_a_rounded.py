"""What if mask family A is not sharp-cornered?  (MODEL PREDICTIONS; sensitivity of the mask predictions to an assumption.)

The project notes do not record the corner treatment of family A of the author's mask; mask_predictions.py assumes sharp corners (rho = 0).
An earlier note of the author is reported to mention arm tips rounded with a radius of half the arm width (rho = 1.0).  This script recomputes
the eight family-A designs with rho = 1.0 in every accepted world (family B and everything else unchanged, taken from the cache of
mask_predictions.py) and repeats the count of switching devices per group of rotations with the same random numbers, so that the difference to the
main prediction comes from family A only.  Writes results/mask_family_a_rounded.json.   Usage: python scripts/mask_family_a_rounded.py [--rho 1.0]
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
from vcsel_bo import physics as P  # noqa: E402

RES = ROOT / "results"


def one(rec, rho, n_dev=24):
    w = P.World(d0=rec["d0"], b0=rec["b0"], k_d=rec["k_d"], k_b=rec["k_b"])
    pop = P.Population.make(n_dev, 4242)
    fam = L.MASK_FAMILIES["A"]
    out = {}
    for th in L.MASK_ANGLES:
        d = G.MaskDesign(fam["L"], fam["ratio"], fam["w"], rho, th)
        ev = O.evaluate_design(d.as_array(), w, pop)
        f = ev.diag["feats"]
        wts = np.tile(np.array([1 / 6, 2 / 3, 1 / 6])[:, None], (1, pop.n)).ravel() / pop.n
        out[f"A{int(th)}"] = dict(yield_=float(ev.hard_yield), p_switch=float((wts * (f["n_sw"] >= 1)).sum()))
    return out


def group_counts(PS, names, n_worlds, per=2, seed=11):
    groups = {"near 0 deg (0, 30, 150; both families)": [f"{f}{a}" for f in "AB" for a in (0, 30, 150)],
              "45 and 135 deg (both families)": [f"{f}{a}" for f in "AB" for a in (45, 135)],
              "60, 90, 120 deg (both families)": [f"{f}{a}" for f in "AB" for a in (60, 90, 120)]}
    rng, res = np.random.default_rng(seed), {}
    for gname, members in groups.items():
        draws = np.array([sum(rng.binomial(per, PS[m][i]) for m in members) for i in range(n_worlds) for _ in range(20)])
        res[gname] = dict(n_devices=per * len(members), mean=float(draws.mean()), q05=float(np.percentile(draws, 5)),
                          q50=float(np.percentile(draws, 50)), q95=float(np.percentile(draws, 95)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=1.0)
    a = ap.parse_args()
    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"]
    z = np.load(RES / "mask_predictions_per_world.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]
    PS = {n: z["p_switch"][k] for k, n in enumerate(names)}
    Y = {n: z["yield_"][k] for k, n in enumerate(names)}
    sharp = group_counts(PS, names, len(acc))
    res = Parallel(n_jobs=2)(delayed(one)(r, a.rho) for r in acc)
    PS2, Y2 = dict(PS), dict(Y)
    for n in [f"A{int(t)}" for t in L.MASK_ANGLES]:
        PS2[n] = np.array([r[n]["p_switch"] for r in res])
        Y2[n] = np.array([r[n]["yield_"] for r in res])
    rounded = group_counts(PS2, names, len(acc))
    out = dict(note="MODEL PREDICTIONS; family A recomputed with corner fraction rho (arm tips rounded), family B unchanged",
               rho_family_A=a.rho, n_worlds=len(acc), sharp_family_A=sharp, rounded_family_A=rounded,
               per_design=dict(sharp={n: dict(mean_yield=float(Y[n].mean()), mean_p_switch=float(PS[n].mean())) for n in names if n[0] == "A"},
                               rounded={n: dict(mean_yield=float(Y2[n].mean()), mean_p_switch=float(PS2[n].mean())) for n in names if n[0] == "A"}))
    (RES / "mask_family_a_rounded.json").write_text(json.dumps(out, indent=1))
    for g in sharp:
        s, r = sharp[g], rounded[g]
        print(f"{g:45s} sharp {s['mean']:.1f} ({s['q05']:.0f}-{s['q95']:.0f})   rounded {r['mean']:.1f} ({r['q05']:.0f}-{r['q95']:.0f})")
    for n in out["per_design"]["sharp"]:
        print(n, "yield sharp %.2f rounded %.2f | p_switch sharp %.2f rounded %.2f" % (
            out["per_design"]["sharp"][n]["mean_yield"], out["per_design"]["rounded"][n]["mean_yield"],
            out["per_design"]["sharp"][n]["mean_p_switch"], out["per_design"]["rounded"][n]["mean_p_switch"]))


if __name__ == "__main__":
    main()
