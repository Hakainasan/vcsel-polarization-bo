"""Testable predictions for the 16 designs of the author's mask (two families x eight orientations).

For every literature-consistent world and every design: fraction of devices that lase first along the native axis x,
mean azimuth of the low-current state, fraction that switch, median switching current, hard yield.  Aggregated over the
worlds (median and 10-90% interval) they form the table of predictions that the next measurement can test.
Also writes pairwise ordering probabilities, e.g. P(yield at 90 deg > yield at 60 deg).
The azimuth is an AXIAL quantity (0 and 180 degrees are the same direction), so it is summarised by the axial circular mean over
worlds and its resultant length R, not by a linear median.
Writes results/mask_predictions.json.  MODEL PREDICTIONS conditional on the anchors; not measurements.
Usage: python scripts/mask_predictions.py [--from-cache]   (--from-cache rebuilds the summaries from results/mask_predictions_per_world.npz)
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
from joblib import Parallel, delayed  # noqa: E402

from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402

RES = ROOT / "results"


def one(rec, n_dev=24):
    w = P.World(d0=rec["d0"], b0=rec["b0"], k_d=rec["k_d"], k_b=rec["k_b"])
    pop = P.Population.make(n_dev, 4242)
    wts = np.tile(np.array([1 / 6, 2 / 3, 1 / 6])[:, None], (1, pop.n)).ravel() / pop.n
    out = {}
    for name, d in L.MASK_DESIGNS.items():
        ev = O.evaluate_design(d.as_array(), w, pop)
        f = ev.diag["feats"]
        a0 = np.deg2rad(f["a_start"])
        z = (wts * np.exp(2j * a0)).sum()                          # doubled-angle circular mean
        az = float(np.rad2deg(np.angle(z)) / 2 % 180)
        out[name] = dict(yield_=ev.hard_yield, p_x_first=float((wts * (np.cos(2 * a0) > 0)).sum()),
                         p_switch=float((wts * (f["n_sw"] >= 1)).sum()), az_low=az,
                         mu_sw=float(ev.mu_sw_median) if np.isfinite(ev.mu_sw_median) else None)
    return out


def q(v, lo=10, hi=90):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return [float(np.percentile(v, lo)), float(np.median(v)), float(np.percentile(v, hi))]


def axial(v):
    """Axial circular summary of angles in degrees (period 180): [mean direction, resultant length R in 0..1, 10th and 90th percentile of the
    deviation from the mean direction wrapped to [-90, 90)]."""
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    z = np.exp(2j * np.deg2rad(v)).mean()
    c = float((np.rad2deg(np.angle(z)) / 2) % 180)
    dev = (v - c + 90) % 180 - 90
    return dict(mean=c, R=float(np.abs(z)), dev10=float(np.percentile(dev, 10)), dev90=float(np.percentile(dev, 90)))


def from_cache(names):
    z = np.load(RES / "mask_predictions_per_world.npz", allow_pickle=True)
    cn = [str(n) for n in z["names"]]
    assert cn == list(names), "cache was written for another set of mask designs"
    n_w = z["yield_"].shape[1]
    res = []
    for i in range(n_w):
        res.append({n: dict(yield_=float(z["yield_"][k, i]), p_x_first=float(z["p_x_first"][k, i]), p_switch=float(z["p_switch"][k, i]),
                            az_low=float(z["az_low"][k, i]), mu_sw=None if not np.isfinite(z["mu_sw"][k, i]) else float(z["mu_sw"][k, i]))
                    for k, n in enumerate(cn)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true")
    a = ap.parse_args()
    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"]
    t0 = time.time()
    names = list(L.MASK_DESIGNS)
    res = from_cache(names) if a.from_cache else Parallel(n_jobs=2)(delayed(one)(r) for r in acc)
    tab = {}
    for n in names:
        ax = axial([r[n]["az_low"] for r in res])
        tab[n] = dict(yield_=q([r[n]["yield_"] for r in res]), p_x_first=q([r[n]["p_x_first"] for r in res]),
                      p_switch=q([r[n]["p_switch"] for r in res]), az_low_deg=[(ax["mean"] + ax["dev10"]) % 180, ax["mean"], (ax["mean"] + ax["dev90"]) % 180],
                      az_axial=ax, share_worlds_x_like=float(np.mean([r[n]["p_x_first"] > 0.5 for r in res])),
                      mu_sw=q([r[n]["mu_sw"] for r in res]), mean_yield=float(np.mean([r[n]["yield_"] for r in res])))
    Y = {n: np.array([r[n]["yield_"] for r in res]) for n in names}
    PS = {n: np.array([r[n]["p_switch"] for r in res]) for n in names}
    np.savez(RES / "mask_predictions_per_world.npz", names=np.array(names), yield_=np.array([Y[n] for n in names]),
             p_switch=np.array([PS[n] for n in names]), p_x_first=np.array([[r[n]["p_x_first"] for r in res] for n in names]),
             az_low=np.array([[r[n]["az_low"] for r in res] for n in names]),
             mu_sw=np.array([[np.nan if r[n]["mu_sw"] is None else r[n]["mu_sw"] for r in res] for n in names]))
    order = {}

    def gt(a, b):
        return float(np.mean(Y[a] > Y[b] + 1e-9)), float(np.mean(np.abs(Y[a] - Y[b]) <= 0.1))
    for a, b in (("A90", "A60"), ("A90", "A45"), ("A60", "A45"), ("A90", "A30"), ("A60", "A120"), ("A45", "A135"), ("A30", "A150"),
                 ("A90", "B90"), ("A60", "B60"), ("A45", "B45")):
        order[f"{a} > {b}"] = dict(zip(("P_greater", "P_within_0.1"), gt(a, b)))
    # Predictive distribution of the NUMBER of switching devices in groups of designs, for a mask with `per` devices per
    # design (32 devices for 16 designs -> per = 2): draw a world, then Binomial(per, p_switch) per design, sum over the group.
    per, rng = 2, np.random.default_rng(11)
    groups = {"near 0 deg (0, 30, 150; both families)": [f"{f}{a}" for f in "AB" for a in (0, 30, 150)],
              "45 and 135 deg (both families)": [f"{f}{a}" for f in "AB" for a in (45, 135)],
              "60, 90, 120 deg (both families)": [f"{f}{a}" for f in "AB" for a in (60, 90, 120)]}
    counts = {}
    for gname, members in groups.items():
        draws = np.array([sum(rng.binomial(per, PS[m][i]) for m in members) for i in range(len(acc)) for _ in range(20)])
        counts[gname] = dict(n_devices=per * len(members), mean=float(draws.mean()),
                             q05=float(np.percentile(draws, 5)), q50=float(np.percentile(draws, 50)), q95=float(np.percentile(draws, 95)))
    # Same draw, but counting devices that satisfy the WHOLE yield specification (one switch inside the current window, stable states,
    # separation, ellipticity) instead of any polarization flip: this count is smaller.
    rng2, pcounts = np.random.default_rng(12), {}
    for gname, members in groups.items():
        draws = np.array([sum(rng2.binomial(per, Y[m][i]) for m in members) for i in range(len(acc)) for _ in range(20)])
        pcounts[gname] = dict(n_devices=per * len(members), mean=float(draws.mean()),
                              q05=float(np.percentile(draws, 5)), q50=float(np.percentile(draws, 50)), q95=float(np.percentile(draws, 95)))
    out = dict(note="MODEL PREDICTIONS conditional on the literature anchors (interval: 10th, median, 90th percentile over worlds)",
               n_worlds=len(acc), predictions=tab, ordering=order, group_switch_counts=counts, group_pass_counts=pcounts,
               switch_definition="a device counts as switching if at least one polarization flip occurs in the sweep of I/I_th over [1.10, 2.10]",
               pass_definition="a device passes if it meets the whole yield specification (exactly one switch at I/I_th in [1.25, 1.95], stable states, separation >= 60 deg, ellipticity <= 0.45)",
               devices_per_design_assumed=per,
               family_note="A: L=17.5 um, arm ratio 0.7, w=6 um, sharp corners (assumed); B: L=15 um, ratio 0.7, w=6 um, short-arm radius 2.25 um")
    (RES / "mask_predictions.json").write_text(json.dumps(out, indent=1))
    print(f"{time.time() - t0:.0f}s")
    for n in names:
        t = tab[n]
        print(f"{n:5s} yield med {t['yield_'][1]:.2f} [{t['yield_'][0]:.2f},{t['yield_'][2]:.2f}]  x-first med {t['p_x_first'][1]:.2f}  "
              f"az_low med {t['az_low_deg'][1]:.0f}  p_sw {t['p_switch'][1]:.2f}")
    print(json.dumps(order, indent=0))
    print(json.dumps(counts, indent=0))
    print('pass counts:', json.dumps(pcounts, indent=0))


if __name__ == "__main__":
    main()
