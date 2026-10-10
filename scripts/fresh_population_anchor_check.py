"""Do the accepted worlds still satisfy the literature anchors on device populations they were NOT selected on?  (MODEL CHECK.)

The rejection sampler accepts a world if it passes every enforced anchor on ONE device population (24 devices, seed 777).  Anchors are yes/no tests
on fractions of devices, so a world near a threshold can pass on one population and fail on another.  Here every accepted world is screened again with
fresh populations (other seeds, same size); the share that still passes tells how much of the acceptance is population luck.  The fresh-device
scoring of the batch designs (Sec. 6 of the supplement) uses such fresh populations, so worlds that no longer pass are slightly outside the conditioning set there.

Writes results/fresh_population_anchors.json.   Usage: python scripts/fresh_population_anchor_check.py [--seeds 1001 1002 1003]
"""
from __future__ import annotations

import argparse
import json
import sys
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


def one(rec, seed, n_dev):
    w = P.World(d0=rec["d0"], b0=rec["b0"], k_d=rec["k_d"], k_b=rec["k_b"])
    ok, failed, _ = L.screen_world(w, P.Population.make(n_dev, seed), O.Spec())
    return bool(ok), failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    ap.add_argument("--n_dev", type=int, default=24)
    a = ap.parse_args()
    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"]
    out = dict(note="MODEL CHECK: share of the accepted worlds that still pass every enforced anchor on a fresh device population of the same size",
               n_worlds=len(acc), n_dev=a.n_dev, per_seed={})
    for s in a.seeds:
        res = Parallel(n_jobs=2)(delayed(one)(r, s, a.n_dev) for r in acc)
        n_ok = sum(ok for ok, _ in res)
        fails = {}
        for ok, f in res:
            if not ok:
                fails[f] = fails.get(f, 0) + 1
        out["per_seed"][str(s)] = dict(n_pass=n_ok, share=n_ok / len(acc), first_failure=fails)
        print(f"seed {s}: {n_ok} of {len(acc)} still pass ({100 * n_ok / len(acc):.0f} %); first failures {fails}", flush=True)
    sh = [v["share"] for v in out["per_seed"].values()]
    out["share_min"], out["share_max"] = float(min(sh)), float(max(sh))
    (RES / "fresh_population_anchors.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
