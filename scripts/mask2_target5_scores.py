"""Mask-2 candidates scored in the author's process convention.  (MODEL PREDICTIONS.)

Convention: drawn shape = target aperture; mesa = target grown by D = 5 um; isotropic oxidation of depth 5 um with a
device-to-device scatter of 0.25 um; the model constants are those accepted by anchor_target_convention.py (same convention
for the published designs).  Every candidate and both mask-1 families are scored on every accepted world (hard yield,
24 simulated devices, the default specification).  Writes results/mask2_target5_scores.json.
Usage: python scripts/mask2_target5_scores.py [--depth 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed  # noqa: E402

from vcsel_bo import geometry as G  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402

A = dict(L=17.5, ratio=0.7, w=6.0, rho=0.0)          # mask-1 family A (target)
B = dict(L=15.0, ratio=0.7, w=6.0, rho=0.75)         # mask-1 family B (target)
CAND = {}
for th in (0, 30, 45, 60, 70, 80, 90, 100, 110, 120, 135, 150):
    CAND[f"A{th}"] = (A["L"], A["ratio"], A["w"], A["rho"], float(th))
for th in (0, 30, 45, 60, 90, 120, 135, 150):
    CAND[f"B{th}"] = (B["L"], B["ratio"], B["w"], B["rho"], float(th))
CAND.update({
    "M1": (17.374, 0.723, 6.173, 0.824, 70.26), "M2": (18.431, 0.654, 5.061, 0.803, 90.97), "M3": (14.394, 0.746, 5.817, 0.532, 112.11),
    "XL6_0": (30.0, 0.7, 6.0, 0.0, 0.0), "XL6_80": (30.0, 0.7, 6.0, 0.0, 80.0), "XL6_90": (30.0, 0.7, 6.0, 0.0, 90.0),
    "XL6_100": (30.0, 0.7, 6.0, 0.0, 100.0),
    "XL4_80": (28.0, 0.7, 4.0, 0.0, 80.0), "XL4_90": (28.0, 0.7, 4.0, 0.0, 90.0), "XL4_100": (28.0, 0.7, 4.0, 0.0, 100.0),
    "CX0": (17.5, 1.0, 6.0, 0.0, 0.0), "CX45": (17.5, 1.0, 6.0, 0.0, 45.0),
    "CIRC": (13.47, 1.0, 13.47, 1.0, 0.0),            # circle with the area of the family-A target
})


def row(key, worlds, depth, sig_depth, n_dev=24):
    t = G.MaskDesign(*CAND[key])
    m = G.mesa_of_target(t, depth)
    pop = P.Population.make(n_dev, 4242)
    ys = []
    for r in worlds:
        w = replace(P.World(d0=r["d0"], b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]), process=G.Process(depth, 1.0), sig_depth=sig_depth)
        ys.append(O.evaluate_design(m.as_array(), w, pop).hard_yield)
    return key, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=5.0)
    ap.add_argument("--sig-depth", type=float, default=0.25)
    a = ap.parse_args()
    rec = json.loads((ROOT / "results" / f"anchored_worlds_target{a.depth:g}.json").read_text())
    worlds = rec["accepted"]
    res = dict(Parallel(n_jobs=2)(delayed(row)(k, worlds, a.depth, a.sig_depth) for k in CAND))
    Y = {k: np.array(v) for k, v in res.items()}
    out = dict(note="MODEL PREDICTIONS in the target-aperture convention (D = %g um, isotropic); hard yield per world" % a.depth,
               n_worlds=len(worlds), designs={})
    for k, y in Y.items():
        t = CAND[k]
        ap_area = G.build_aperture(G.mesa_of_target(G.MaskDesign(*t), a.depth), G.Process(a.depth, 1.0)).area
        out["designs"][k] = dict(target=dict(L=t[0], ratio=t[1], w=t[2], rho=t[3], theta=t[4]), aperture_area_um2=round(ap_area, 1),
                                 mean=float(y.mean()), q10=float(np.percentile(y, 10)), q90=float(np.percentile(y, 90)),
                                 p_ge_half=float((y >= 0.5).mean()))
        print(f"{k:8s} aperture {ap_area:6.1f} um2   yield {y.mean():.2f} [{np.percentile(y,10):.2f}-{np.percentile(y,90):.2f}]  P(>=0.5) {(y>=0.5).mean():.2f}", flush=True)

    def bob(keys):
        M = np.vstack([Y[k] for k in keys])
        b = M.max(0)
        return dict(E_best=float(b.mean()), P_some_ge_half=float((b >= 0.5).mean()), designs=list(keys))
    mask1 = [f"A{t}" for t in (0, 30, 45, 60, 90, 120, 135, 150)] + [f"B{t}" for t in (0, 30, 45, 60, 90, 120, 135, 150)]
    sweep = [f"A{t}" for t in (70, 80, 90, 100, 110)]
    picks = ["M1", "M2", "M3"]
    xl = ["XL6_80", "XL6_90", "XL6_100", "XL4_90"]
    ctrl = ["A0", "CX0", "CIRC"]
    out["batch"] = dict(mask1=bob(mask1), mask2_core=bob(sweep + picks + ctrl), mask2_core_plus_xl=bob(sweep + picks + ctrl + xl),
                        mask1_plus_mask2=bob(mask1 + sweep + picks + ctrl + xl))
    for k, v in out["batch"].items():
        print(f"{k:22s} E_best {v['E_best']:.3f}  P(some >= 0.5) {v['P_some_ge_half']:.3f}")
    (ROOT / "results" / f"mask2_target{a.depth:g}_scores.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
