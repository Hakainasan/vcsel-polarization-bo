"""Sensitivity of the designed batch to the assumptions (model predictions, NOT measurements).

Metric per scenario and per design set: E_best = mean over TEST worlds of the best hard yield within the set (what
matters for a fabrication run) and P(>=0.5) = fraction of worlds in which some design reaches yield 0.5.
Design sets compared in every scenario: the greedy-portfolio batch (results/batch_design.csv), the author's 16-design mask,
and the seven designs of APL 126. The relative ordering, not the absolute level, is the robustness question.

Scenarios A (perturb the model, same TEST worlds): oxidation depth / anisotropy, aperture index step, thermal dichroism
drift, coupling scale, device scatter x2, mask fabrication tolerance, finer analysis grid, alternative switching-current
windows (Ising nodes favour low bias).
Scenarios B (re-condition on the literature): worlds accepted with LOOSE / STRICT threshold translations, with the spin-flip
constants of the metasurface paper, under a wider prior, and (RECAL) after changing the oxidation anisotropy / thermal drift and
re-inferring the constants (needs results/anchored_worlds_{loose,strict,chen,wide,ox10,ox13,chim02,chip02}.json).
Writes results/sensitivity.json.
"""
import hashlib
import json
import re
import shutil
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
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402
from vcsel_bo.sfm import SFMConstants  # noqa: E402

RES = ROOT / "results"
CHEN = SFMConstants(kappa=125.0, alpha=3.0, gamma=0.67, gamma_s=1000.0)
# Re-conditioned scenarios: the model assumption is changed AND the four anisotropy constants are re-inferred from the same literature
# outcomes (results/anchored_worlds_<tag>.json from `scripts/anchored_worlds.py --ox-ratio/--chi-th`); the designed batch (chosen under the
# baseline assumptions) is then scored on those worlds.  This separates "the ordering needs this assumption to be true" from "the ordering
# needs the constants to stay as they were inferred under the baseline assumption".
RECAL = {"ox10": ("RECONDITIONED: oxidation isotropic (ox_ratio=1.0)", dict(process=(None, 1.0))),
         "ox13": ("RECONDITIONED: oxidation anisotropy ox_ratio=1.3", dict(process=(None, 1.3))),
         "chim02": ("RECONDITIONED: thermal dichroism drift chi_th=-0.2", dict(chi_th=-0.2)),
         "chip02": ("RECONDITIONED: thermal dichroism drift chi_th=+0.2", dict(chi_th=0.2))}


def load_sets():
    rows = np.genfromtxt(RES / "batch_design.csv", delimiter=",", names=True, dtype=None, encoding=None)
    Xb = np.array([[r["L_um"], r["arm_ratio"], r["arm_width_um"], r["corner_rho"], r["rotation_deg"]] for r in rows])
    Xm = np.array([d.as_array() for d in L.MASK_DESIGNS.values()])
    Xs = np.array([L.DESIGNS[k].as_array() for k in L.APL126_SEVEN])
    return dict(portfolio=Xb, author_mask=Xm, apl126_seven=Xs)


def to_worlds(recs):
    return [P.World(d0=r["d0"], b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]) for r in recs]


def score(X, worlds, pop, spec=O.Spec(), c=SFMConstants()):
    Y = np.array([[O.evaluate_design(x, w, pop, spec, c=c).hard_yield for w in worlds] for x in X])
    best = Y.max(0)
    return dict(E_best=float(best.mean()), P_ge_half=float((best >= 0.5).mean()), mean_single=float(Y.mean()))


def scenario(name, kw, sets, test_recs, extra_recs):
    pop = P.Population.make(24, 4242)
    worlds = to_worlds(extra_recs[kw["worlds"]] if "worlds" in kw else test_recs)   # re-conditioned worlds replace the test worlds
    spec, c, X = O.Spec(), SFMConstants(), dict(sets)
    if "process" in kw:
        d, r = kw["process"]
        worlds = [replace(w, process=G.Process(d if d is not None else w.process.depth, r if r is not None else w.process.ox_ratio))
                  for w in worlds]
    if "dn_eff" in kw:
        worlds = [replace(w, dn_eff=kw["dn_eff"]) for w in worlds]
    if "chi_th" in kw:
        worlds = [replace(w, chi_th=kw["chi_th"]) for w in worlds]
    if "coupling" in kw:
        worlds = [replace(w, k_d=w.k_d * kw["coupling"], k_b=w.k_b * kw["coupling"]) for w in worlds]
    if "scatter" in kw:
        s = kw["scatter"]
        worlds = [replace(w, sig_d0=w.sig_d0 * s, sig_b0=w.sig_b0 * s, sig_psi=w.sig_psi * s, sig_depth=w.sig_depth * s) for w in worlds]
    if "mask_tol" in kw:
        r2 = np.random.default_rng(5)
        X = {k: v + r2.normal(0, 1, v.shape) * np.array([0.3, 0.0, 0.3, 0.0, 1.0]) for k, v in X.items()}
    if "grid" in kw:
        G.set_resolution(*kw["grid"])
    if "spec" in kw:
        spec = O.Spec(**kw["spec"])
    if "sfm" in kw:
        c = CHEN if kw["sfm"] == "chen" else c
    out = {k: score(v, worlds, pop, spec, c) for k, v in X.items()}
    if "grid" in kw:
        G.set_resolution(0.25, 0.5)
    return name, out


def main():
    sets = load_sets()
    test_recs = json.loads((RES / "batch_worlds.json").read_text())["test"]
    extra, skipped = {}, {}
    for tag in ("loose", "strict", "chen", "wide") + tuple(RECAL):
        f = RES / f"anchored_worlds_{tag}.json"
        if f.exists():
            recs = json.loads(f.read_text())["accepted"]
            if recs:
                extra[tag] = recs[:32]
            else:
                skipped[tag] = "no world accepted (0 of %d draws)" % json.loads(f.read_text())["n_draws"]
    scen = [("baseline", {}),
            ("process depth=0.7", dict(process=(0.7, None))), ("process depth=1.3", dict(process=(1.3, None))),
            ("process ox_ratio=1.0", dict(process=(None, 1.0))), ("process ox_ratio=1.3", dict(process=(None, 1.3))),
            ("dn_eff=0.005", dict(dn_eff=0.005)), ("dn_eff=0.02", dict(dn_eff=0.02)),
            ("chi_th=-0.2", dict(chi_th=-0.2)), ("chi_th=-0.05", dict(chi_th=-0.05)),
            ("chi_th=+0.05", dict(chi_th=0.05)), ("chi_th=+0.2", dict(chi_th=0.2)),
            ("coupling x0.5", dict(coupling=0.5)), ("coupling x2", dict(coupling=2.0)),
            ("device scatter x2", dict(scatter=2.0)),
            ("mask tolerance (rot 1 deg, L and w 0.3 um)", dict(mask_tol=True)),
            ("fine geometry grid", dict(grid=(0.125, 0.25))),
            ("window 1.15-1.60 (low bias)", dict(spec=dict(mu_lo=1.15, mu_hi=1.60, range_lo=1.05, range_hi=1.70))),
            ("window 1.15-2.05 (wide)", dict(spec=dict(mu_lo=1.15, mu_hi=2.05))),
            ("SFM constants of the metasurface paper (same worlds)", dict(sfm="chen"))]
    for tag, label in (("loose", "worlds accepted with LOOSE thresholds"), ("strict", "worlds accepted with STRICT thresholds"),
                       ("chen", "worlds accepted with the metasurface-paper SFM constants"),
                       ("wide", "worlds accepted under a wider prior (native dichroism from 0.2 1/ns)")):
        if tag in extra:
            kw = dict(worlds=tag)
            if tag == "chen":
                kw["sfm"] = "chen"
            scen.append((label, kw))
    for tag, (label, mod) in RECAL.items():          # re-conditioned on the literature under a changed process assumption
        if tag in extra:
            scen.append((label, dict(worlds=tag, **mod)))
    # every scenario is cached (a restart must not lose the work); the cache is dropped if any input file changed
    parts = RES / "sensitivity_parts"
    h = hashlib.sha1()
    for f in ("batch_design.csv", "batch_worlds.json", "anchored_worlds_loose.json", "anchored_worlds_strict.json", "anchored_worlds_chen.json"):
        if (RES / f).exists():
            h.update((RES / f).read_bytes())
    stamp = parts / "_inputs.sha1"
    if parts.exists() and (not stamp.exists() or stamp.read_text() != h.hexdigest()):
        shutil.rmtree(parts)
    parts.mkdir(exist_ok=True)
    stamp.write_text(h.hexdigest())

    late = {}      # world files that are not part of the global stamp: adding them later must not discard the other scenarios
    for tag in ("wide",) + tuple(RECAL):
        fl = RES / f"anchored_worlds_{tag}.json"
        if fl.exists():
            late[tag] = hashlib.sha1(fl.read_bytes()).hexdigest()[:8]

    def cache_file(name, kw):
        # "chi_th=-0.2" and "chi_th=+0.2" must not share a file (they did in the first version: the sign was lost when the name was cleaned)
        sign = ("_" + hashlib.sha1(name.encode()).hexdigest()[:6]) if re.search(r"[=x][+-]", name) else ""
        return parts / (re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:80] + sign
                        + (("_" + late[kw["worlds"]]) if kw.get("worlds") in late else "") + ".json")

    def cached(name, kw):
        f = cache_file(name, kw)
        if f.exists():
            return name, json.loads(f.read_text())
        n_, o_ = scenario(name, kw, sets, test_recs, extra)
        f.write_text(json.dumps(o_))
        return n_, o_
    res = Parallel(n_jobs=2, verbose=5)(delayed(cached)(n, kw) for n, kw in scen)
    out = {"note": "MODEL PREDICTIONS; scenarios A perturb the model on the same test worlds, B re-condition on the literature",
           "n_test_worlds": len(test_recs), "sets": {k: len(v) for k, v in sets.items()}, "skipped_scenarios": skipped}
    for n, o in res:
        out[n] = o
    (RES / "sensitivity.json").write_text(json.dumps(out, indent=1))
    for n, o in res:
        print(f"{n:62s}", {k: round(v["E_best"], 2) for k, v in o.items()})


if __name__ == "__main__":
    main()
