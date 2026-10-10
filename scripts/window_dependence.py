"""Does the ranking of rotations depend on what counts as a good switching current?  (MODEL PREDICTIONS, NOT MEASUREMENTS.)

The yield objective requires exactly one polarization switch inside a window of the drive current, mu = I / I_th.  That window is a
specification of the Ising node, not a physical constant.  This script scores a 1:0.7 cross (L = 17.5, w = 6 um, rho = 0.5) at several
rotations in every literature-consistent world (every second world of results/anchored_worlds.json) under four windows:
    default   1.25-1.95 (operating range 1.10-2.10)
    low bias  1.15-1.60 (operating range 1.05-1.70)   Ising nodes favour low-bias switching
    wide      1.15-2.05
    mid       1.50-1.70                                 the narrow window of benchmark problem P2
and reports the mean yield per rotation, the rotation with the highest mean yield, and the share of worlds in which the 60 deg cross
has a higher yield than the 90 deg cross.  The mirror rotations (theta and 180 - theta) are equivalent up to device scatter.
Writes results/window_dependence.json.   Usage: python scripts/window_dependence.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed  # noqa: E402

from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402

RES = ROOT / "results"
WINDOWS = {"default 1.25-1.95": O.Spec(),
           "low bias 1.15-1.60": O.Spec(mu_lo=1.15, mu_hi=1.60, range_lo=1.05, range_hi=1.70),
           "wide 1.15-2.05": O.Spec(mu_lo=1.15, mu_hi=2.05),
           "mid 1.50-1.70": O.Spec(mu_lo=1.50, mu_hi=1.70)}
THETAS = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0]


def one(rec):
    w = P.World(d0=rec["d0"], b0=rec["b0"], k_d=rec["k_d"], k_b=rec["k_b"])
    pop = P.Population.make(24, 777)
    return {name: [O.evaluate_design([17.5, 0.7, 6.0, 0.5, th], w, pop, spec).hard_yield for th in THETAS] for name, spec in WINDOWS.items()}


def main():
    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"][::2]
    rows = Parallel(n_jobs=2)(delayed(one)(r) for r in acc)
    out = dict(note="MODEL PREDICTIONS conditional on the literature anchors; 1:0.7 cross, L=17.5, w=6 um, rho=0.5",
               n_worlds=len(acc), thetas=THETAS, windows={})
    i60, i90 = THETAS.index(60.0), THETAS.index(90.0)
    for name in WINDOWS:
        Y = np.array([r[name] for r in rows])                       # (worlds, thetas)
        out["windows"][name] = dict(mean_yield=Y.mean(0).tolist(), best_theta=float(THETAS[int(np.argmax(Y.mean(0)))]),
                                    share_worlds_60_beats_90=float((Y[:, i60] > Y[:, i90] + 1e-9).mean()),
                                    share_worlds_90_beats_60=float((Y[:, i90] > Y[:, i60] + 1e-9).mean()))
        print(f"{name:20s} mean yield by rotation {np.round(Y.mean(0), 2)} best {out['windows'][name]['best_theta']:.0f}  "
              f"P(60>90) {out['windows'][name]['share_worlds_60_beats_90']:.2f}  P(90>60) {out['windows'][name]['share_worlds_90_beats_60']:.2f}")
    (RES / "window_dependence.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
