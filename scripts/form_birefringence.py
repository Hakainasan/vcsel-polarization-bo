"""Form birefringence of the author's mask designs from the semi-vectorial solver (vector_modes.py).  (MODEL PREDICTIONS.)

For mask families A and B at the eight rotations, with the default process (depth 1.0 um, oxidation-rate ratio 1.15) and with
isotropic oxidation, this computes the splitting of the two linear polarizations caused by the SHAPE of the lateral index guide
alone, and compares it with
  * the phenomenological stress term |k_b| * |s_stress| of the anchored worlds (what the optimiser currently uses), and
  * the measured splittings quoted in physics.py (1.3-9.9 GHz, gamma_p about 4-31 ns^-1) [choquette1995].
A grid-convergence check (h = 0.1 and 0.07 um) is run on one design.

Writes results/form_birefringence.json.   Usage: python scripts/form_birefringence.py [--dn_eff 0.01]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from vcsel_bo import geometry as G  # noqa: E402
from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import vector_modes as V  # noqa: E402
from vcsel_bo.physics import geometry_descriptors  # noqa: E402

RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dn_eff", type=float, default=0.01)
    ap.add_argument("--h", type=float, default=0.1)
    a = ap.parse_args()
    t0 = time.time()
    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"]
    kb = np.abs(np.array([r["k_b"] for r in acc]))
    out = dict(note="MODEL PREDICTIONS: form birefringence of the lateral index guide only (no stress, no thermal or carrier effects)",
               dn_eff=a.dn_eff, h=a.h, n_g=3.6, wavelength_um=0.94, designs={})
    print(f"{'design':8s} {'process':10s} {'dn_eff(1e-6)':>12s} {'axis':>6s} {'split GHz':>9s} {'gp_form':>8s} {'stress term median [5-95%]':>28s}")
    for proc_name, proc in (("aniso1.15", G.Process(1.0, 1.15)), ("isotropic", G.Process(1.0, 1.0))):
        for name, d in L.MASK_DESIGNS.items():
            fb = V.form_birefringence(d, proc, dn_eff=a.dn_eff, h=a.h)
            ds = geometry_descriptors(tuple(d.as_array()), proc.depth, proc.ox_ratio, a.dn_eff)
            st = kb * float(np.hypot(*ds.s_stress))
            rec = dict(process=proc_name, theta=d.theta, valid=fb.valid, dneff=fb.dneff, axis_deg=fb.axis_deg, dnu_ghz=fb.dnu_ghz,
                       gamma_p_form=fb.gamma_p, lam_gap=fb.lam_gap,
                       stress_term_median=float(np.median(st)), stress_term_q05=float(np.percentile(st, 5)),
                       stress_term_q95=float(np.percentile(st, 95)))
            out["designs"][f"{name}|{proc_name}"] = rec
            print(f"{name:8s} {proc_name:10s} {fb.dneff * 1e6:12.2f} {fb.axis_deg:6.1f} {fb.dnu_ghz:9.3f} {fb.gamma_p:8.2f} "
                  f"{rec['stress_term_median']:10.1f} [{rec['stress_term_q05']:.1f}-{rec['stress_term_q95']:.1f}]", flush=True)
    d = L.MASK_DESIGNS["A60"]
    c = V.form_birefringence(d, G.Process(1.0, 1.15), dn_eff=a.dn_eff, h=0.1)
    f = V.form_birefringence(d, G.Process(1.0, 1.15), dn_eff=a.dn_eff, h=0.07)
    out["convergence_A60"] = dict(h01=c.gamma_p, h007=f.gamma_p, axis01=c.axis_deg, axis007=f.axis_deg,
                                 rel_change=abs(c.gamma_p - f.gamma_p) / max(f.gamma_p, 1e-12))
    print("convergence A60:", out["convergence_A60"])
    gp = np.array([v["gamma_p_form"] for v in out["designs"].values() if v["valid"]])
    out["summary"] = dict(gamma_p_form_max=float(gp.max()), gamma_p_form_median=float(np.median(gp)),
                          measured_gamma_p_range=[4.0, 31.0], runtime_s=time.time() - t0)
    print("summary:", out["summary"])
    (RES / "form_birefringence.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
