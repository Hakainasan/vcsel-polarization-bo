"""Mask-2 controls and the small cross under a stronger native anisotropy (x1, x2, x3).  (MODEL PREDICTIONS.)

Completes the table of native_strength_shift.py for the designs that script does not cover: the smaller cross X7S
(0.75 x the 1:0.7 cross at 90 deg), the circle CIRC and the symmetric cross SYM.  Same worlds, convention and population as
native_strength_shift.py (target grown by 5 um, isotropic oxidation, native dichroism d0 scaled by 1, 2, 3).
Writes results/mask2_extras_native.json.   Usage: python scripts/mask2_extras_native.py
"""
from __future__ import annotations

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

D = 5.0
DES = {
    "X7S": (13.125, 0.7, 4.5, 0.0, 90.0),
    "CIRC": (13.47, 1.0, 13.47, 1.0, 0.0),
    "SYM": (17.5, 1.0, 6.0, 0.0, 0.0),
}


def row(key, worlds, scale, n_dev=24):
    m = G.mesa_of_target(G.MaskDesign(*DES[key]), D)
    pop = P.Population.make(n_dev, 4242)
    ys = []
    for r in worlds:
        w = replace(P.World(d0=r["d0"] * scale, b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]), process=G.Process(D, 1.0), sig_depth=0.25)
        ys.append(O.evaluate_design(m.as_array(), w, pop).hard_yield)
    return key, scale, float(np.mean(ys)), float(np.percentile(ys, 10)), float(np.percentile(ys, 90))


def main():
    worlds = json.loads((ROOT / "results" / "anchored_worlds_target5.json").read_text())["accepted"]
    jobs = [(k, s) for s in (1.0, 2.0, 3.0) for k in DES]
    res = Parallel(n_jobs=2)(delayed(row)(k, worlds, s) for k, s in jobs)
    out = {}
    for k, s, m, lo, hi in res:
        out.setdefault(f"x{s:g}", {})[k] = dict(mean=m, q10=lo, q90=hi)
    for s in out:
        print(s, "  ".join(f"{k}:{v['mean']:.2f}" for k, v in out[s].items()), flush=True)
    (ROOT / "results" / "mask2_extras_native.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
