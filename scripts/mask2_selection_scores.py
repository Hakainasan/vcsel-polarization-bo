"""Proposed single-device selection for mask 2 (no arrays), scored in the author's process convention.  (MODEL PREDICTIONS.)

Same convention and constants as mask2_target5_scores.py (target aperture grown by 5 um, isotropic oxidation, worlds accepted by
anchor_target_convention.py).  Blocks: a full orientation sweep of the mask-1 family-A cross in 15 deg steps; a stadium with the
area and aspect ratio of the oxidised family-A aperture (elongation without arms); a large cross whose arms survive; arm-ratio and
size series at 90 deg; controls.  Reports the predicted switching share and the second moments of each oxidised aperture.
Writes results/mask2_selection_scores.json.   Usage: python scripts/mask2_selection_scores.py
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
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed  # noqa: E402

from depth_study import shape_stats  # noqa: E402
from vcsel_bo import geometry as G  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402

D = 5.0
A = (17.5, 0.7, 6.0, 0.0)                    # mask-1 family A target (sharp tips)
CAND = {f"A{t:03d}": A + (float(t),) for t in range(0, 180, 15)}
CAND.update({
    "ST90": (17.0, 11.0 / 17.0, 11.0, 1.0, 90.0), "ST0": (17.0, 11.0 / 17.0, 11.0, 1.0, 0.0),   # stadium 17 x 11 um
    "XL90": (30.0, 0.7, 6.0, 0.0, 90.0), "XL0": (30.0, 0.7, 6.0, 0.0, 0.0),
    "A90_r06": (17.5, 0.6, 6.0, 0.0, 90.0), "A90_r08": (17.5, 0.8, 6.0, 0.0, 90.0),
    "A90_s075": (13.125, 0.7, 4.5, 0.0, 90.0), "A90_s125": (21.875, 0.7, 7.5, 0.0, 90.0),
    "CIRC": (13.47, 1.0, 13.47, 1.0, 0.0), "CX0": (17.5, 1.0, 6.0, 0.0, 0.0), "CX45": (17.5, 1.0, 6.0, 0.0, 45.0),
})


def row(key, worlds, n_dev=24):
    m = G.mesa_of_target(G.MaskDesign(*CAND[key]), D)
    pop = P.Population.make(n_dev, 4242)
    ys = []
    for r in worlds:
        w = replace(P.World(d0=r["d0"], b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]), process=G.Process(D, 1.0), sig_depth=0.25)
        ys.append(O.evaluate_design(m.as_array(), w, pop).hard_yield)
    return key, ys


def main():
    worlds = json.loads((ROOT / "results" / "anchored_worlds_target5.json").read_text())["accepted"]
    res = dict(Parallel(n_jobs=2)(delayed(row)(k, worlds) for k in CAND))
    out = dict(note="MODEL PREDICTIONS (target convention, D = 5 um, isotropic); switching share = hard yield", n_worlds=len(worlds), designs={})
    for k, t in CAND.items():
        y = np.array(res[k])
        ap = G.build_aperture(G.mesa_of_target(G.MaskDesign(*t), D), G.Process(D, 1.0))
        st = shape_stats(ap.loops[0])
        out["designs"][k] = dict(target=list(t), aperture_area=round(st["area"], 1), aspect=round(st["aspect"], 3), axis_deg=round(st["axis_deg"], 1),
                                 solidity=round(st["solidity"], 3), mean=float(y.mean()), q10=float(np.percentile(y, 10)),
                                 q90=float(np.percentile(y, 90)), p_ge_half=float((y >= 0.5).mean()))
        print(f"{k:9s} area {st['area']:6.1f} aspect {st['aspect']:.2f} solidity {st['solidity']:.2f}  share {y.mean():.2f} "
              f"[{np.percentile(y, 10):.2f}-{np.percentile(y, 90):.2f}]", flush=True)
    (ROOT / "results" / "mask2_selection_scores.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
