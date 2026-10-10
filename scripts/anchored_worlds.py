"""Draw literature-consistent worlds by rejection ABC (see src/vcsel_bo/literature.py).

Each draw is a set of model constants (d0, b0, k_d, k_b) from the broad prior in `physics.Prior`; it is kept iff the
simulated device population reproduces every enforced qualitative observation of the tailored-aperture papers.

Usage:
    python scripts/anchored_worlds.py --draws 6000 --thresholds primary  -> results/anchored_worlds.json
    python scripts/anchored_worlds.py --draws 3000 --thresholds loose --out results/anchored_worlds_loose.json
    python scripts/anchored_worlds.py --draws 3000 --thresholds strict --out results/anchored_worlds_strict.json
    python scripts/anchored_worlds.py --draws 4000 --sfm chen --out results/anchored_worlds_chen.json
    python scripts/anchored_worlds.py --draws 4000 --prior wide --out results/anchored_worlds_wide.json
    python scripts/anchored_worlds.py --draws 4000 --ox-ratio 1.0 --out results/anchored_worlds_ox10.json    (re-conditioned)
Also writes per-anchor marginal acceptance (300 exhaustive draws) into the same file for the primary run.
"""
from __future__ import annotations

import argparse
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
from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402
from vcsel_bo.sfm import SFMConstants  # noqa: E402

RES = ROOT / "results"


# SFM constants of the metasurface paper (chen2026meta, Ref. [68] there): kappa 125, gamma 0.67, gamma_s 1000, alpha 3
CHEN_SFM = SFMConstants(kappa=125.0, alpha=3.0, gamma=0.67, gamma_s=1000.0)


def _marginal_one(seed, thresh, c):
    rng = np.random.default_rng(seed)
    w = P.sample_world(rng)
    r = L.anchor_report(w, thresh=thresh, c=c)
    return {k: bool(v[0]) for k, v in r["anchors"].items()}, bool(r["ok_enforced"]), r["metrics"]["x07_30"]["p_switch"]


def marginal_rates(n, thresh, jobs, c):
    out = Parallel(n_jobs=jobs)(delayed(_marginal_one)(10_000 + s, thresh, c) for s in range(n))
    rates = {k: float(np.mean([o[0][k] for o in out])) for k in L.ANCHOR_KEYS}
    return dict(n=n, per_anchor=rates, all_enforced=float(np.mean([o[1] for o in out])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--thresholds", choices=["primary", "loose", "strict"], default="primary")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--sfm", choices=["default", "chen"], default="default",
                    help="spin-flip-model constants: the package default or the set of the metasurface paper")
    ap.add_argument("--prior", choices=["default", "wide"], default="default",
                    help="wide: native dichroism from 0.2 ns^-1 (the default box starts at 1.0, and the accepted worlds pile up against that edge) "
                         "and dichroism coupling from 0.6 ns^-1")
    ap.add_argument("--ox-ratio", type=float, default=None,
                    help="re-condition under another anisotropy of the oxidation rate (default: the World default, 1.15)")
    ap.add_argument("--chi-th", type=float, default=None,
                    help="re-condition with a thermal-dichroism drift coupling chi_th (default 0 = static anisotropy)")
    ap.add_argument("--marginal", type=int, default=300, help="exhaustive draws for per-anchor marginal rates (0 = skip)")
    ap.add_argument("--out", type=str, default=str(RES / "anchored_worlds.json"))
    a = ap.parse_args()
    thresh = dict(primary=L.PRIMARY, loose=L.LOOSE, strict=L.STRICT)[a.thresholds]

    c = CHEN_SFM if a.sfm == "chen" else SFMConstants()
    prior = P.Prior(d0=(0.2, 12.0), k_d_mag=(0.6, 100.0)) if a.prior == "wide" else P.Prior()
    base = P.World()
    if a.ox_ratio is not None:
        base = replace(base, process=G.Process(base.process.depth, a.ox_ratio))
    if a.chi_th is not None:
        base = replace(base, chi_th=a.chi_th)
    res = L.sample_anchored_worlds(a.draws, seed=a.seed, prior=prior, thresh=thresh, n_jobs=a.jobs, c=c, base=base)
    res["prior_set"] = a.prior
    res["threshold_set"] = a.thresholds
    res["sfm_set"] = a.sfm
    res["posterior"] = L.posterior_summary(res["accepted"]) if res["accepted"] else {}
    if a.marginal:
        res["marginal"] = marginal_rates(a.marginal, thresh, a.jobs, c)
    L.save_anchored(a.out, res)
    print(f"{a.thresholds}/{a.sfm}: {res['n_accepted']} / {res['n_draws']} accepted "
          f"({100 * res['acceptance']:.2f}%) in {res['seconds']:.0f}s")
    print("first failure:", res["first_failure"])
    print("posterior:", res["posterior"])
    if a.marginal:
        print("marginal:", res["marginal"])


if __name__ == "__main__":
    main()
