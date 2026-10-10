"""Re-condition the model on the literature anchors in the TARGET-APERTURE convention of the author's lab.  (MODEL.)

Drawn shapes are target apertures; mesas are the targets grown by the design oxidation depth D; oxidation is isotropic (the
author reports round apertures on circular mesas) with depth D and a device-to-device depth scatter.  The published designs are
read the same way (their sizes are not in the papers; the assumed sizes of literature.py are used as targets).
Writes results/anchored_worlds_target{D}.json.   Usage: python scripts/anchor_target_convention.py [--depth 5] [--draws 8000]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=float, default=5.0)
ap.add_argument("--ox-ratio", type=float, default=1.0)
ap.add_argument("--sig-depth", type=float, default=0.25, help="device-to-device oxidation-depth scatter, 1 sigma (um)")
ap.add_argument("--draws", type=int, default=8000)
ap.add_argument("--jobs", type=int, default=2)
a = ap.parse_args()
os.environ["VCSEL_TARGET_DEPTH"] = str(a.depth)          # read by literature.py at import (also in worker processes)
sys.path.insert(0, str(ROOT / "src"))

from vcsel_bo import geometry as G  # noqa: E402
from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402

base = replace(P.World(), process=G.Process(a.depth, a.ox_ratio), sig_depth=a.sig_depth)
res = L.sample_anchored_worlds(a.draws, seed=1, prior=P.Prior(), thresh=L.PRIMARY, n_jobs=a.jobs, base=base)
res.update(convention="target aperture", target_depth=a.depth, ox_ratio=a.ox_ratio, sig_depth=a.sig_depth)
res["posterior"] = L.posterior_summary(res["accepted"]) if res["accepted"] else {}
out = ROOT / "results" / f"anchored_worlds_target{a.depth:g}.json"
L.save_anchored(out, res)
print(f"target convention D={a.depth}: {res['n_accepted']} / {res['n_draws']} accepted ({100 * res['acceptance']:.2f}%) in {res['seconds']:.0f}s")
print("first failure:", res["first_failure"])
print("posterior:", res["posterior"])
