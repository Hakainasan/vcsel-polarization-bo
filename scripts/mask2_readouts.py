"""Three read-outs of the mask-2 ruler per design and native strength (x1, x2, x3).  (MODEL PREDICTIONS.)

1. spec_switch  : share of devices that pass the full switching spec (objective.Spec: exactly one switch between 1.25 and 1.95 I_th,
                  both states linear and >= 60 deg apart, a stable state over 1.10-2.10 I_th)  = the 'switching share' of the tables;
2. any_switch   : share with exactly one switch anywhere in 1.10-2.10 I_th (no current window, no state-quality checks);
3. start_01m1   : share of lasing devices whose polarization just above threshold (1.10 I_th) lies within 45 deg of mask y,
                  i.e. along [01-1] (the long axis of the ruler designs) instead of the native [011].
Read-out 3 does not need switching: in the model it crosses 1/2 where the aperture balances the native anisotropy.
Same worlds, convention and population as native_strength_shift.py.  Writes results/mask2_readouts.json.
Usage: python scripts/mask2_readouts.py
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
DES = {f"X{int(round(r * 10))}": (17.5, r, 6.0, 0.0, 90.0) for r in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)}
DES.update({f"S{int(round(a * 100))}": (17.0, (17.0 / a) / 17.0, 17.0 / a, 1.0, 90.0) for a in (1.30, 1.55, 1.80, 2.20)})
DES.update({"CIRC": (13.47, 1.0, 13.47, 1.0, 0.0), "SYM": (17.5, 1.0, 6.0, 0.0, 0.0), "X7S": (13.125, 0.7, 4.5, 0.0, 90.0)})


def row(key, worlds, scale, n_dev=24):
    m = G.mesa_of_target(G.MaskDesign(*DES[key]), D)
    pop = P.Population.make(n_dev, 4242)
    out = []
    for r in worlds:
        w = replace(P.World(d0=r["d0"] * scale, b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]), process=G.Process(D, 1.0), sig_depth=0.25)
        ev = O.evaluate_design(m.as_array(), w, pop)
        f = ev.diag["feats"]
        M = pop.n
        wt = P.GH_WEIGHTS[:, None] / M * np.ones((3, M))
        lasing = (f["occ0"].reshape(3, M) >= 0)
        a0 = f["a_start"].reshape(3, M)
        along_y = (np.abs(((a0 - 90.0) + 90.0) % 180.0 - 90.0) < 45.0)
        den = (wt * lasing).sum()
        out.append(dict(spec=ev.hard_yield, any=ev.frac_switch, valid=ev.frac_valid_geometry,
                        start=float((wt * lasing * along_y).sum() / den) if den > 0 else float("nan")))
    return key, scale, out


def main():
    worlds = json.loads((ROOT / "results" / "anchored_worlds_target5.json").read_text())["accepted"]
    jobs = [(k, s) for s in (1.0, 2.0, 3.0) for k in DES]
    res = Parallel(n_jobs=2)(delayed(row)(k, worlds, s) for k, s in jobs)
    out = dict(note="MODEL PREDICTIONS (target convention, D = 5 um, isotropic); means over the accepted worlds", n_worlds=len(worlds))
    for k, s, rows in res:
        rec = {}
        for q in ("spec", "any", "start", "valid"):
            v = np.array([r[q] for r in rows], float)
            v = v[np.isfinite(v)]
            rec[q] = dict(mean=float(v.mean()), q10=float(np.percentile(v, 10)), q90=float(np.percentile(v, 90)))
        out.setdefault(f"x{s:g}", {})[k] = rec
    for s in ("x1", "x2", "x3"):
        print(s)
        for k, rec in out[s].items():
            print(f"  {k:5s} spec {rec['spec']['mean']:.2f}  any {rec['any']['mean']:.2f}  start[01-1] {rec['start']['mean']:.2f}", flush=True)
    (ROOT / "results" / "mask2_readouts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
