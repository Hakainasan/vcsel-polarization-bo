"""Published fabricated-device observations, encoded as consistency constraints on the model.

WHAT THIS IS (AND IS NOT)
-------------------------
The experimental papers on tailored oxide apertures report *qualitative* outcomes per aperture design (which
polarization lases first, whether the device switches, which design needed the least injection power). They do not
report dichroism/birefringence values, switching currents or yields, so there is nothing to *fit* quantitatively.
What can be done honestly is to require that a candidate "world" (a set of model constants) reproduces each reported
outcome, and to discard worlds that do not (approximate Bayesian computation with qualitative summary statistics).
The accepted worlds are the literature-consistent prior used for design; they are still uncalibrated in magnitude.

Every anchor below names the paper it comes from (keys refer to `docs/references.bib`, source-by-source notes are in
`docs/LITERATURE_ANCHORS.md`). The wording of the papers ("most", "seldom", "part of the fabricated VCSELs") is
translated into numbers by the thresholds in `THRESH`. Those numbers are *choices*; `scripts/anchored_worlds.py`
repeats the sampling with looser and stricter thresholds to show what they change.

Axis convention: x = natively favoured ("horizontal") polarization axis = wafer flat; theta = angle of the long arm
from x (see `physics`).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path

import numpy as np

from . import geometry as G
from . import objective as O
from .physics import Population, Prior, World, sample_world
from .sfm import SFMConstants

# Representative mask designs (um). Cross: L = long arm, w = arm width, rho = corner rounding fraction.
# The absolute sizes of the fabricated devices are not given in the papers; these are typical of the design box.
L0, W0, RHO0 = 17.5, 6.0, 0.5
AREA0 = 2 * L0 * W0 - W0 * W0
_SQ = float(np.sqrt(AREA0))
_CIRC = float(2 * np.sqrt(AREA0 / np.pi))

DESIGNS = {
    "circle": G.MaskDesign(_CIRC, 1.0, _CIRC, 1.0, 0.0),
    "square": G.MaskDesign(_SQ, 1.0, _SQ, 0.0, 0.0),
    "sym_cross": G.MaskDesign(L0, 1.0, W0, RHO0, 0.0),
    "x07_0": G.MaskDesign(L0, 0.7, W0, RHO0, 0.0),
    "x07_30": G.MaskDesign(L0, 0.7, W0, RHO0, 30.0),
    "x07_60": G.MaskDesign(L0, 0.7, W0, RHO0, 60.0),
    "x07_90": G.MaskDesign(L0, 0.7, W0, RHO0, 90.0),
    "x06_0": G.MaskDesign(L0, 0.6, W0, RHO0, 0.0),
    "x06_90": G.MaskDesign(L0, 0.6, W0, RHO0, 90.0),
}

# The polarization-control mask designed in the lab of the author (notes of the project, mask v8): two aperture
# families of arm ratio 0.7 and arm width 6 um, each at eight orientations. The corner treatment of family A is not
# recorded in the notes and is assumed sharp; family B has a round short-arm radius of 2.25 um (rho = 2.25 / (w / 2)).
# Optional convention of the author's lab: the drawn shape is the TARGET APERTURE and the mesa is that shape grown by the
# design oxidation depth.  If VCSEL_TARGET_DEPTH (um) is set, the published designs above are read as targets and their
# mesas are built with geometry.mesa_of_target (used by scripts/anchor_target_convention.py; the default is unchanged).
_TARGET_DEPTH = os.environ.get("VCSEL_TARGET_DEPTH")
if _TARGET_DEPTH:
    DESIGNS = {k: G.mesa_of_target(v, float(_TARGET_DEPTH)) for k, v in DESIGNS.items()}

MASK_ANGLES = (0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0)
MASK_FAMILIES = {"A": dict(L=17.5, ratio=0.7, w=6.0, rho=0.0), "B": dict(L=15.0, ratio=0.7, w=6.0, rho=0.75)}
MASK_DESIGNS = {f"{fam}{int(th)}": G.MaskDesign(p["L"], p["ratio"], p["w"], p["rho"], th)
                for fam, p in MASK_FAMILIES.items() for th in MASK_ANGLES}

# The seven aperture designs of the APL 126 experiment (Fig. 2(a)-(g)), drawn with the typical sizes above.
APL126_SEVEN = ("square", "x07_0", "x07_30", "x07_60", "x07_90", "x06_0", "x06_90")

# Translation of the wording of the papers into numbers. PRIMARY is used for the main results; LOOSE / STRICT are used
# to check how much the conclusions depend on these (arbitrary) translations.
PRIMARY = dict(most=0.60,        # "most devices ..."                    -> fraction >= 0.60
               seldom=0.30,      # "seldom switch"                        -> switching fraction <= 0.30
               part=0.10,        # "part of the fabricated VCSELs switch" -> switching fraction >= 0.10
               more=0.05)        # "greater proportion ..."               -> larger by >= 0.05
LOOSE = dict(most=0.50, seldom=0.40, part=0.05, more=0.02)
STRICT = dict(most=0.75, seldom=0.20, part=0.20, more=0.10)
THRESH = PRIMARY


def design_metrics(x_design: G.MaskDesign, world: World, pop: Population, spec: O.Spec = O.Spec(),
                   c: SFMConstants = SFMConstants()) -> dict:
    """Population-level observables of one design in one world: fraction of devices that lase first along x (native
    axis), fraction that switch at least once inside the operating range, fraction switching from x to the other state.
    Raw states are used (geometry validity is handled by the yield, not by the anchors)."""
    x = [x_design.L, x_design.ratio, x_design.w, x_design.rho, x_design.theta]
    ev = O.evaluate_design(x, world, pop, spec, c=c)
    f = ev.diag["feats"]
    a0 = f["a_start"]
    x_first = np.cos(np.deg2rad(2 * a0)) > 0
    sw = f["n_sw"] >= 1
    wts = np.tile(np.array([1 / 6, 2 / 3, 1 / 6])[:, None], (1, pop.n)).ravel() / pop.n
    return dict(p_x_first=float((wts * x_first).sum()), p_switch=float((wts * sw).sum()),
                p_x_to_y=float((wts * (sw & x_first)).sum()), p_y_to_x=float((wts * (sw & ~x_first)).sum()),
                mu_sw=ev.mu_sw_median, valid_geometry=ev.frac_valid_geometry, hard_yield=ev.hard_yield)


@dataclass(frozen=True)
class Anchor:
    key: str
    source: str            # citation key(s) (see docs/references.bib)
    claim: str             # what the paper reports (paraphrase; short quoted fragments are marked with quotes)
    designs: tuple         # designs whose metrics the check needs
    check: object          # callable (metrics, thresholds) -> (ok, explanation)
    enforced: bool = True  # False: reported only, does not reject worlds


def _both(m, T, keys):
    return all(m[k]["p_x_first"] >= T["most"] and m[k]["p_switch"] <= T["seldom"] for k in keys)


ANCHORS = [
    Anchor("A1_circle_square_native_x", "yuan2025clea; yuan2025apl126",
           "Circular apertures mostly lase in the x (native) polarization and seldom switch; the square aperture "
           "is a non-switchable reference.",
           ("circle", "square"),
           lambda m, T: (_both(m, T, ("circle", "square")),
                         f"x-first {m['circle']['p_x_first']:.2f}/{m['square']['p_x_first']:.2f}, "
                         f"switch {m['circle']['p_switch']:.2f}/{m['square']['p_switch']:.2f}")),
    Anchor("A2_long_arm_along_native_axis_strengthens", "yuan2025apl126",
           "A cruciform whose longer arm lies along the natively favoured (horizontal) axis strengthens the built-in "
           "anisotropy, i.e. behaves like the circle: x-first, seldom switching.",
           ("x07_0",),
           lambda m, T: (_both(m, T, ("x07_0",)),
                         f"x-first {m['x07_0']['p_x_first']:.2f}, switch {m['x07_0']['p_switch']:.2f}")),
    Anchor("A3_rotation_by_90_flips_initial_polarization", "yuan2025clea; yuan2025mesa",
           "For the 1:0.7 cruciform most unrotated devices lase x-polarized while 90-degree-rotated devices lase "
           "y-polarized: the initial polarization follows the elongated arm.",
           ("x07_0", "x07_90"),
           lambda m, T: (m["x07_0"]["p_x_first"] >= T["most"] and (1 - m["x07_90"]["p_x_first"]) >= T["most"],
                         f"x-first {m['x07_0']['p_x_first']:.2f} (0 deg) / {m['x07_90']['p_x_first']:.2f} (90 deg)")),
    Anchor("A5_y_elongated_switches_y_to_x", "yuan2025mesa",
           "The y-elongated (90-degree) cruciform switches from initial y to x polarization; the x-elongated one keeps "
           "a stable x polarization.",
           ("x07_90", "x07_0"),
           lambda m, T: (m["x07_90"]["p_y_to_x"] > m["x07_90"]["p_x_to_y"] and m["x07_0"]["p_switch"] <= T["seldom"],
                         f"y->x {m['x07_90']['p_y_to_x']:.2f} vs x->y {m['x07_90']['p_x_to_y']:.2f}")),
    Anchor("A4_rotated_cruciforms_switch", "yuan2025apl126",
           "Rotated 1:0.7 cruciforms (30, 60 and 90 degrees in the reported set) show polarization switching in part of "
           "the fabricated devices, and those devices have better optical-injection-locking performance. Only the "
           "60- and 90-degree members are enforced; 30 degrees is reported.",
           ("x07_60", "x07_90", "x07_0"),
           lambda m, T: (all(m[k]["p_switch"] >= T["part"] for k in ("x07_60", "x07_90"))
                         and m["x07_90"]["p_switch"] > m["x07_0"]["p_switch"],
                         f"switch 60/90 deg: {m['x07_60']['p_switch']:.2f}/{m['x07_90']['p_switch']:.2f} "
                         f"vs 0 deg {m['x07_0']['p_switch']:.2f}")),
    Anchor("A6_symmetric_cross_switches_more_than_circle", "yuan2025clea",
           "Symmetric cruciform VCSELs show a greater proportion of devices with polarization switching than circular ones "
           "(not enforced: in a second-rank model a C4 cross has no shape anisotropy of its own, so this can only come from "
           "anisotropic oxidation).",
           ("sym_cross", "circle"),
           lambda m, T: (m["sym_cross"]["p_switch"] >= m["circle"]["p_switch"] + T["more"],
                         f"switch sym cross {m['sym_cross']['p_switch']:.2f} vs circle {m['circle']['p_switch']:.2f}"),
           enforced=False),
]
ANCHOR_KEYS = [a.key for a in ANCHORS]
# cheap-to-expensive order for early rejection
_ORDER = ["A1_circle_square_native_x", "A2_long_arm_along_native_axis_strengthens",
          "A3_rotation_by_90_flips_initial_polarization", "A5_y_elongated_switches_y_to_x",
          "A4_rotated_cruciforms_switch", "A6_symmetric_cross_switches_more_than_circle"]
_BY_KEY = {a.key: a for a in ANCHORS}
REPORT_ONLY_DESIGNS = ("x07_30",)


def anchor_report(world: World, pop: Population | None = None, spec: O.Spec = O.Spec(),
                  thresh: dict = PRIMARY, extra_designs: tuple = REPORT_ONLY_DESIGNS,
                  c: SFMConstants = SFMConstants()) -> dict:
    """Evaluate every anchor in a world. Returns dict(anchors={key: (ok, text)}, metrics={design: {...}}, ok_enforced)."""
    pop = pop or Population.make(24, 777)
    need = sorted({d for a in ANCHORS for d in a.designs} | set(extra_designs))
    metrics = {k: design_metrics(DESIGNS[k], world, pop, spec, c) for k in need}
    res = {a.key: a.check(metrics, thresh) for a in ANCHORS}
    ok = all(res[a.key][0] for a in ANCHORS if a.enforced)
    return dict(anchors=res, metrics=metrics, ok_enforced=ok)


def screen_world(world: World, pop: Population, spec: O.Spec = O.Spec(), thresh: dict = PRIMARY,
                 c: SFMConstants = SFMConstants()):
    """Staged test used for sampling: anchors are checked cheapest-first and evaluation stops at the first enforced
    failure. Returns (accepted, failed_key or None, n_designs_evaluated)."""
    metrics: dict = {}
    for key in _ORDER:
        a = _BY_KEY[key]
        if not a.enforced:
            continue
        for d in a.designs:
            if d not in metrics:
                metrics[d] = design_metrics(DESIGNS[d], world, pop, spec, c)
        ok, _ = a.check(metrics, thresh)
        if not ok:
            return False, key, len(metrics)
    return True, None, len(metrics)


# ----------------------------------------------------------------------------------------------
# Literature-consistent worlds by rejection ABC
# ----------------------------------------------------------------------------------------------

CONSTS = ("d0", "b0", "k_d", "k_b")


def _chunk(seed: int, n: int, prior: Prior, thresh: dict, spec: O.Spec, pop_seed: int, n_dev: int, c: SFMConstants,
           base: World | None = None):
    rng = np.random.default_rng(seed)
    pop = Population.make(n_dev, pop_seed)
    accepted, fails, n_eval = [], {}, 0
    for _ in range(n):
        w = sample_world(rng, prior, base)
        ok, failed, nd = screen_world(w, pop, spec, thresh, c)
        n_eval += nd
        if ok:
            accepted.append(dict(d0=w.d0, b0=w.b0, k_d=w.k_d, k_b=w.k_b))
        else:
            fails[failed] = fails.get(failed, 0) + 1
    return accepted, fails, n_eval


def sample_anchored_worlds(n_draws: int, seed: int = 0, prior: Prior = Prior(), thresh: dict = PRIMARY,
                           spec: O.Spec = O.Spec(), n_jobs: int = 2, chunk: int = 250, pop_seed: int = 777,
                           n_dev: int = 24, c: SFMConstants = SFMConstants(), base: World | None = None) -> dict:
    """Rejection ABC: draw `n_draws` worlds from the prior and keep those that satisfy every enforced anchor.

    `base` fixes everything that is not a sampled constant (oxidation process, thermal-dichroism coupling, index step):
    the default `World()` is the main analysis; a different `base` re-conditions the model on the same literature
    outcomes under another process assumption (the "recalibrated" robustness scenarios of `scripts/sensitivity.py`).

    Returns dict(accepted=[{d0,b0,k_d,k_b}...], n_draws, n_accepted, acceptance, first_failure={anchor: count}).
    Deterministic given (seed, n_draws, chunk); each chunk has its own generator so jobs are independent.
    """
    from joblib import Parallel, delayed

    sizes = [chunk] * (n_draws // chunk) + ([n_draws % chunk] if n_draws % chunk else [])
    t0 = time.time()
    outs = Parallel(n_jobs=n_jobs)(delayed(_chunk)(seed * 100003 + i, s, prior, thresh, spec, pop_seed, n_dev, c, base)
                                   for i, s in enumerate(sizes))
    accepted, fails = [], {}
    for acc, f, _ in outs:
        accepted += acc
        for k, v in f.items():
            fails[k] = fails.get(k, 0) + v
    b = base or World()
    return dict(accepted=accepted, n_draws=int(n_draws), n_accepted=len(accepted),
                acceptance=len(accepted) / max(n_draws, 1), first_failure=fails, seed=seed,
                thresholds=dict(thresh), sfm_constants=asdict(c), seconds=time.time() - t0,
                base=dict(process_depth=b.process.depth, process_ox_ratio=b.process.ox_ratio, chi_th=b.chi_th,
                          dn_eff=b.dn_eff))


def worlds_from_records(records) -> list[World]:
    return [World(d0=r["d0"], b0=r["b0"], k_d=r["k_d"], k_b=r["k_b"]) for r in records]


def save_anchored(path, result: dict):
    Path(path).write_text(json.dumps(result, indent=1))


def load_anchored(path) -> dict:
    return json.loads(Path(path).read_text())


def posterior_summary(records) -> dict:
    """Quantiles and sign fractions of the world constants among accepted worlds."""
    out = {}
    for c in CONSTS:
        v = np.array([r[c] for r in records], float)
        out[c] = dict(median=float(np.median(v)), q05=float(np.quantile(v, 0.05)), q95=float(np.quantile(v, 0.95)),
                      frac_positive=float(np.mean(v > 0)))
    return out
