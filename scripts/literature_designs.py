"""Which published (and which of the author's own) aperture designs work best in the literature-consistent worlds?

For every accepted world (results/anchored_worlds.json) evaluate
  * the nine anchor designs (circle, square, symmetric cross, 1:0.7 cruciform at 0/30/60/90 deg, 1:0.6 at 0/90 deg),
  * the 16 designs of the author's mask (two families x eight orientations, `literature.MASK_DESIGNS`),
  * a rotation scan of the cruciform for five arm ratios (L = 17.5 um, w = 6 um, rho = 0.5).
The outputs are yields *predicted by the model*, conditional on the qualitative anchors: they are not measurements.
Writes results/literature_designs.json and results/literature_designs.npz.
"""
from __future__ import annotations

import json
import sys
import time
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
RATIOS = (0.5, 0.6, 0.7, 0.8, 0.9)
THETAS = np.arange(0.0, 180.0, 10.0)
ALL = {**{k: v for k, v in L.DESIGNS.items()}, **{f"mask_{k}": v for k, v in L.MASK_DESIGNS.items()}}


def one(rec, n_dev, want_scan):
    w = P.World(d0=rec["d0"], b0=rec["b0"], k_d=rec["k_d"], k_b=rec["k_b"])
    pop = P.Population.make(n_dev, 4242)
    out = {}
    for name, d in ALL.items():
        x = [d.L, d.ratio, d.w, d.rho, d.theta]
        ev = O.evaluate_design(x, w, pop)
        f = ev.diag["feats"]
        wts = np.tile(np.array([1 / 6, 2 / 3, 1 / 6])[:, None], (1, pop.n)).ravel() / pop.n
        dn = ev.diag["aniso"]["d_norm"][1].mean()
        out[name] = (ev.hard_yield, float((wts * (f["n_sw"] >= 1)).sum()), ev.mu_sw_median, float(dn / w.d0))
    scan = None
    if want_scan:
        pop16 = P.Population.make(16, 4242)
        scan = np.array([[O.evaluate_design([17.5, r, 6.0, 0.5, t], w, pop16).hard_yield for t in THETAS] for r in RATIOS])
    return out, scan


def main():
    R = L.load_anchored(RES / "anchored_worlds.json")
    acc = R["accepted"]
    t0 = time.time()
    res = Parallel(n_jobs=2)(delayed(one)(r, 24, True) for r in acc)
    names = list(ALL)
    Y = np.array([[o[0][n][0] for o in res] for n in names])          # (designs, worlds)
    PS = np.array([[o[0][n][1] for o in res] for n in names])
    MU = np.array([[o[0][n][2] for o in res] for n in names])
    DR = np.array([[o[0][n][3] for o in res] for n in names])
    scan = np.array([o[1] for o in res])                                # (worlds, ratios, thetas)
    np.savez(RES / "literature_designs.npz", names=names, Y=Y, PS=PS, MU=MU, DR=DR, scan=scan, ratios=RATIOS, thetas=THETAS)

    def summ(i):
        y = Y[i]
        return dict(mean_yield=float(y.mean()), q10=float(np.quantile(y, 0.1)), q90=float(np.quantile(y, 0.9)),
                    p_yield_ge_half=float((y >= 0.5).mean()), mean_p_switch=float(PS[i].mean()),
                    median_mu_sw=float(np.nanmedian(MU[i])) if np.isfinite(MU[i]).any() else None,
                    mean_residual_dichroism=float(DR[i].mean()))

    out = dict(note="Model-predicted, conditional on the literature anchors; not measurements.",
               n_worlds=len(acc), designs={n: summ(i) for i, n in enumerate(names)})
    # how often each of the seven APL 126 designs is the best of the seven (ties split)
    seven = [names.index(n) for n in L.APL126_SEVEN]
    Ys = Y[seven]
    top = np.isclose(Ys, Ys.max(0, keepdims=True)) & (Ys.max(0, keepdims=True) > 0.05)
    out["apl126_seven_top_frequency"] = {L.APL126_SEVEN[k]: float((top[k] / np.maximum(top.sum(0), 1)).sum() / len(acc))
                                          for k in range(len(seven))}
    out["apl126_seven_no_working_design_fraction"] = float((Ys.max(0) <= 0.05).mean())
    ms = scan.mean(0)
    out["rotation_scan"] = dict(thetas=THETAS.tolist(), ratios=list(RATIOS), mean_yield=ms.tolist(),
                                best_theta_by_ratio={str(r): float(THETAS[ms[i].argmax()]) for i, r in enumerate(RATIOS)},
                                p_world_reaches_half={str(r): float((scan[:, i].max(1) >= 0.5).mean()) for i, r in enumerate(RATIOS)})
    (RES / "literature_designs.json").write_text(json.dumps(out, indent=1))
    print(f"{time.time() - t0:.0f}s")
    for n in names:
        s = out["designs"][n]
        print(f"{n:12s} yield {s['mean_yield']:.2f} [{s['q10']:.2f},{s['q90']:.2f}]  P(>=.5) {s['p_yield_ge_half']:.2f}  p_sw {s['mean_p_switch']:.2f}  mu_sw {s['median_mu_sw']}")
    print(out["apl126_seven_top_frequency"], out["apl126_seven_no_working_design_fraction"])
    print(out["rotation_scan"]["best_theta_by_ratio"], out["rotation_scan"]["p_world_reaches_half"])


if __name__ == "__main__":
    main()
