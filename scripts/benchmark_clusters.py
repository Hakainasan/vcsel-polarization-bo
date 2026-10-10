"""Benchmark statistics at the level of the simulated WAFERS (worlds), and the selection-vs-unseen gap of problem P3.  (MODEL PREDICTIONS.)

The paired Wilcoxon tests of run_benchmark.py / make_tables.py pair runs by (world, seed).  Runs on the same wafer share the constants, the device
population and the objective landscape, so they are not independent: the tests overstate the effective sample size.  Here the four (P1) or eight
(P2) seeds of a wafer are first averaged, and the five wafer means are compared:
  * number of wafers in which the Bayesian optimiser is ahead of a baseline (out of 5),
  * mean difference over wafers with a 95 % t-interval (4 degrees of freedom; only five wafers, so the interval is wide).
For P3 (one design for eight worlds, scored on 24 unseen worlds) the hard yield of the returned design on the eight SELECTION worlds is compared with its yield on
the unseen worlds.  Also records how the five wafers of P1/P2 were chosen (see run_benchmark.select_worlds): among 30 accepted worlds those in which
no design of a 128-point Sobol cloud reaches a yield of 0.6 are dropped, and five are kept that span the range of difficulty.

Writes results/benchmark_clusters.json.   Usage: python scripts/benchmark_clusters.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def wafer_level(problem):
    d = pickle.load(open(RES / f"benchmark_{problem}.pkl", "rb"))
    meths = sorted({r["method"] for r in d})
    worlds = sorted({r["world"] for r in d})
    M = {m: np.array([np.mean([r["held_out"] for r in d if r["method"] == m and r["world"] == w]) for w in worlds]) for m in meths}

    def diff(a, b):
        dd = M[a] - M[b]
        n = len(dd)
        se = dd.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        return dict(mean=float(dd.mean()), lo=float(dd.mean() - t * se), hi=float(dd.mean() + t * se), n_ahead=int((dd > 0).sum()), n_wafers=n,
                    per_wafer=[float(x) for x in dd])
    res = {f"{a} - {b}": diff(a, b) for a in meths if a.startswith("bo") for b in meths if not b.startswith("bo")}
    return dict(wafer_means={m: [float(x) for x in M[m]] for m in meths}, comparisons=res,
                runs_per_method=int(len([r for r in d if r["method"] == meths[0]])))


def p3_gap():
    d = pickle.load(open(RES / "benchmark_p3.pkl", "rb"))
    out = {}
    for m in sorted({r["method"] for r in d}):
        rr = [r for r in d if r["method"] == m]
        sel = [float(r["hard"][int(np.argmax(r["y"]))]) for r in rr]      # yield of the returned design on the 8 selection worlds
        held = [float(r["held_out"]) for r in rr]                          # yield on the 24 unseen worlds
        out[m] = dict(selection=float(np.mean(sel)), unseen=float(np.mean(held)), gap=float(np.mean(sel) - np.mean(held)), n_runs=len(rr))
    return out


def main():
    out = dict(note="MODEL PREDICTIONS; wafer-level (cluster) statistics of the optimiser benchmark; not measurements",
               p1=wafer_level("p1"), p2=wafer_level("p2"), p3=p3_gap())
    (RES / "benchmark_clusters.json").write_text(json.dumps(out, indent=1))
    for p in ("p1", "p2"):
        print(p)
        for k, v in out[p]["comparisons"].items():
            print(f"  {k:18s} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  ahead in {v['n_ahead']}/{v['n_wafers']} wafers")
    print("p3", json.dumps(out["p3"], indent=0))


if __name__ == "__main__":
    main()
