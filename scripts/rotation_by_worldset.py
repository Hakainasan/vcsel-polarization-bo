"""Does the best rotation of a 1:0.7 cross depend on how the literature outcomes were translated into constraints?  (MODEL PREDICTIONS.)

The rotation ranking of the main analysis (results/literature_designs.json, results/window_dependence.json) uses the primary
literature-consistent worlds.  Here the same 1:0.7 cross (L = 17.5, w = 6 um, rho = 0.5) is scored on the worlds of every alternative
way of conditioning on the literature:
    primary   results/anchored_worlds.json           (every third world)
    loose     results/anchored_worlds_loose.json     looser translation of "most / seldom / part of the devices"
    strict    results/anchored_worlds_strict.json    stricter translation
    chen      results/anchored_worlds_chen.json      spin-flip constants of the metasurface paper (scored with those constants)
    wide      results/anchored_worlds_wide.json      wider prior (native dichroism from 0.2 1/ns, dichroism coupling from 0.6 1/ns)
    ox10, ox13, chim02, chip02   worlds re-inferred under isotropic / more anisotropic oxidation and thermal dichroism drift
              (each scored with that process assumption)
For each set: mean yield by rotation, the rotation with the highest mean, and the rotations within 90 % of the highest mean.
Writes results/rotation_by_worldset.json.   Usage: python scripts/rotation_by_worldset.py
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

from vcsel_bo import geometry as G  # noqa: E402
from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402
from vcsel_bo.sfm import SFMConstants  # noqa: E402

RES = ROOT / "results"
THETAS = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
DEF = SFMConstants()
SETS = dict(primary=("anchored_worlds.json", 3, DEF, {}),
            loose=("anchored_worlds_loose.json", 1, DEF, {}),
            strict=("anchored_worlds_strict.json", 1, DEF, {}),
            chen=("anchored_worlds_chen.json", 1, SFMConstants(kappa=125.0, alpha=3.0, gamma=0.67, gamma_s=1000.0), {}),
            wide=("anchored_worlds_wide.json", 1, DEF, {}),
            ox10=("anchored_worlds_ox10.json", 1, DEF, dict(process=G.Process(1.0, 1.0))),
            ox13=("anchored_worlds_ox13.json", 1, DEF, dict(process=G.Process(1.0, 1.3))),
            chim02=("anchored_worlds_chim02.json", 1, DEF, dict(chi_th=-0.2)),
            chip02=("anchored_worlds_chip02.json", 1, DEF, dict(chi_th=0.2)))
MAX_WORLDS = 40


def one(rec, c, extra):
    w = P.World(d0=rec["d0"], b0=rec["b0"], k_d=rec["k_d"], k_b=rec["k_b"], **extra)
    pop = P.Population.make(24, 777)
    return [O.evaluate_design([17.5, 0.7, 6.0, 0.5, th], w, pop, O.Spec(), c=c).hard_yield for th in THETAS]


def main():
    out = dict(note="MODEL PREDICTIONS conditional on each set of literature-consistent worlds; 1:0.7 cross, L=17.5, w=6 um, rho=0.5, default window",
               thetas=THETAS, sets={})
    for name, (fname, step, c, extra) in SETS.items():
        f = RES / fname
        if not f.exists():
            print("skip", name, "(missing", fname + ")")
            continue
        acc = L.load_anchored(f)["accepted"][::step][:MAX_WORLDS]
        if not acc:
            print("skip", name, "(no accepted worlds)")
            continue
        Y = np.array(Parallel(n_jobs=2)(delayed(one)(r, c, extra) for r in acc))          # (worlds, thetas)
        m = Y.mean(0)
        plateau = [t for t, v in zip(THETAS, m) if v >= 0.9 * m.max()]
        out["sets"][name] = dict(n_worlds=len(acc), mean_yield=m.tolist(), per_world=Y.round(4).tolist(), best_theta=float(THETAS[int(np.argmax(m))]),
                                 plateau=[min(plateau), max(plateau)], max_mean_yield=float(m.max()),
                                 share_worlds_best_between_70_and_110=float(np.mean([THETAS[int(np.argmax(y))] in (70.0, 80.0, 90.0, 100.0, 110.0) for y in Y if y.max() > 0])))
        print(f"{name:8s} n={len(acc):3d} mean yield by rotation {np.round(m, 2)}  best {out['sets'][name]['best_theta']:.0f}  plateau {plateau[0]:.0f}-{plateau[-1]:.0f}", flush=True)
    # How firm is "the most robust rotation"?  Bootstrap over the worlds of every working set (mean yield of the best rotation >= 0.2);
    # the yield of a rotation is averaged with that of its mirror image (theta and 180 - theta), the share of each set's best is
    # computed in every resample, and the lowest share over the sets is the worst case.
    work = [k for k, v in out["sets"].items() if v["max_mean_yield"] >= 0.2]
    if work:
        th = np.array(THETAS)
        fold_idx = {t: [int(np.argmin(abs(th - t))), int(np.argmin(abs(th - (180.0 - t))))] for t in (40.0, 50.0, 60.0, 70.0, 80.0, 90.0)}
        def folded(Y):        # (worlds, 6): mirror-averaged yield by folded rotation
            return np.stack([Y[:, fold_idx[t]].mean(1) for t in fold_idx], axis=1)
        Yf = {k: folded(np.array(out["sets"][k]["per_world"])) for k in work}
        rng = np.random.default_rng(0)
        B = 2000
        worst = np.zeros((B, len(fold_idx)))
        for b in range(B):
            sh = []
            for k in work:
                Y = Yf[k]
                mm = Y[rng.integers(0, len(Y), len(Y))].mean(0)
                sh.append(mm / mm.max())
            worst[b] = np.min(sh, axis=0)
        ft = list(fold_idx)
        out["worst_case_share_bootstrap"] = dict(
            folded_rotations=ft, sets=work, n_resamples=B,
            mean=worst.mean(0).tolist(), lo=np.percentile(worst, 2.5, axis=0).tolist(), hi=np.percentile(worst, 97.5, axis=0).tolist(),
            p_highest={str(int(t)): float(np.mean(worst.argmax(1) == i)) for i, t in enumerate(ft)},
            p_80_above_90=float(np.mean(worst[:, ft.index(80.0)] > worst[:, ft.index(90.0)])),
            p_80_above_70=float(np.mean(worst[:, ft.index(80.0)] > worst[:, ft.index(70.0)])))
        r = out["worst_case_share_bootstrap"]
        print("worst-case share by folded rotation:", dict(zip([int(t) for t in ft], np.round(r["mean"], 2))))
        print("P(highest worst case):", {k: round(v, 2) for k, v in r["p_highest"].items()}, " P(80>90) %.2f  P(80>70) %.2f" % (r["p_80_above_90"], r["p_80_above_70"]))
    (RES / "rotation_by_worldset.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
