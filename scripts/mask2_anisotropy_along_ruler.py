"""Net dichroism |gamma_a| and birefringence |gamma_p| along the mask-2 ruler.  (MODEL PREDICTIONS.)

The model's anisotropy is d = d_nat + k_d s_mode (dichroism, the SFM's gamma_a) and b = b_nat + k_b s_stress (birefringence,
the SFM's gamma_p), both in ns^-1 (backend in physics.py).  For every ruler design and native strength (x1, x2, x3: d0 scaled
as in native_strength_shift.py) this reports, over the accepted worlds and the device population at the nominal depth:
  * the median net dichroism |d| and the share of devices whose dichroism favours [01-1] (d_x > 0: y has the lower loss);
  * the median net birefringence |b| and the corresponding frequency splitting 2|b| / 2 pi (GHz) of the two linear modes.
It shows the trade-off an injection-locked Ising machine cares about (Chen et al., arXiv:2609.21331): the elongation that
cancels the dichroism changes the birefringence too.  Writes results/mask2_anisotropy_along_ruler.json.
Usage: python scripts/mask2_anisotropy_along_ruler.py
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

from vcsel_bo import geometry as G  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402

D = 5.0
DES = {f"X{int(round(r * 10))}": (17.5, r, 6.0, 0.0, 90.0) for r in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)}
DES.update({f"S{int(round(a * 100))}": (17.0, (17.0 / a) / 17.0, 17.0 / a, 1.0, 90.0) for a in (1.30, 1.55, 1.80, 2.20)})
DES.update({"CIRC": (13.47, 1.0, 13.47, 1.0, 0.0), "SYM": (17.5, 1.0, 6.0, 0.0, 0.0), "X7S": (13.125, 0.7, 4.5, 0.0, 90.0)})


def main():
    worlds = json.loads((ROOT / "results" / "anchored_worlds_target5.json").read_text())["accepted"]
    pop = P.Population.make(24, 4242)
    backend = P.PhenomenologicalBackend()
    out = dict(note="MODEL PREDICTIONS: net dichroism (gamma_a) and birefringence (gamma_p) in ns^-1 at the nominal depth; "
                    "splitting_GHz = 2 |b| / (2 pi)", n_worlds=len(worlds))
    for s in (1.0, 2.0, 3.0):
        rec = {}
        for key, t in DES.items():
            m = G.mesa_of_target(G.MaskDesign(*t), D)
            dd, bb, fav = [], [], []
            for r in worlds:
                w = replace(P.World(d0=r["d0"] * s, b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]), process=G.Process(D, 1.0), sig_depth=0.25)
                d, b, valid, _ = backend.vectors(tuple(float(v) for v in m.as_array()), w, pop)
                dd.append(np.linalg.norm(d[1], axis=1))
                bb.append(np.linalg.norm(b[1], axis=1))
                fav.append(d[1][:, 0] > 0)
            dd, bb, fav = np.concatenate(dd), np.concatenate(bb), np.concatenate(fav)
            rec[key] = dict(dichroism_median=float(np.median(dd)), birefringence_median=float(np.median(bb)),
                            splitting_GHz_median=float(np.median(bb) / np.pi), share_dichroism_favours_01m1=float(fav.mean()))
        out[f"x{s:g}"] = rec
        print(f"x{s:g}  " + "  ".join(f"{k}: |d| {v['dichroism_median']:.2f} |b| {v['birefringence_median']:.0f} "
                                      f"({v['splitting_GHz_median']:.0f} GHz) fav {v['share_dichroism_favours_01m1']:.2f}"
                                      for k, v in rec.items()), flush=True)
    (ROOT / "results" / "mask2_anisotropy_along_ruler.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
