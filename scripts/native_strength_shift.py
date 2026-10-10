"""How the best elongation at 90 deg moves if the wafer's native anisotropy is stronger than in the published devices.  (MODEL.)

The lab's 940-nm wafer locks every device to [011] (Lu et al., Light Sci. Appl. 14, 344 (2025)), which suggests a stronger native
dichroism than the published 840-nm devices the model is anchored on.  Here the native dichroism d0 of every accepted world is scaled
by 1, 2 and 3, and the arm-ratio series of the family-A cross at 90 deg (plus a stadium series) is scored.  Writes
results/native_strength_shift.json.   Usage: python scripts/native_strength_shift.py
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
DES = {f"r{r:.1f}": (17.5, r, 6.0, 0.0, 90.0) for r in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
DES.update({f"st{a:.1f}": (17.0, (17.0 / a) / 17.0, 17.0 / a, 1.0, 90.0) for a in (1.3, 1.55, 1.8, 2.2)})


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
    (ROOT / "results" / "native_strength_shift.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
