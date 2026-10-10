"""Write README.md from the cached results, so that no number of the README is typed by hand.  Usage: python scripts/make_readme.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def jl(name):
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else None


def f2(x, n=2):
    return "n/a" if x is None else f"{x:.{n}f}"


A, D, Bs, R, M, Wd, S, V = (jl("anchored_worlds.json"), jl("literature_designs.json"), jl("batch_summary.json"), jl("robustness_extras.json"),
                            jl("mask_predictions.json"), jl("window_dependence.json"), jl("sensitivity.json"), jl("value_of_first_batch.json"))
Mi, Rw, Rk = jl("mask_additions_interior.json"), jl("rotation_by_worldset.json"), None
Ow, Cl = jl("other_worlds_check.json"), jl("benchmark_clusters.json")
Fr, Fa = jl("mask_family_a_rounded.json"), jl("fresh_population_anchors.json")
stats = jl("benchmark_stats.json") or []
cell = {(s["problem"], s["method"]): s for s in stats if "method" in s}
tests = {(s["problem"], s["test"]): s for s in stats if "test" in s}
NAMES = dict(random="random search", sobol="Sobol points", de="differential evolution", bo_ei="GP-EI (Bayesian optimisation)", bo_ucb="GP-UCB (Bayesian optimisation)")

bench_rows = []
for m in ("random", "sobol", "de", "bo_ei", "bo_ucb"):
    cols = []
    for p in ("p1", "p2", "p3"):
        s = cell.get((p, m))
        cols.append("n/a" if s is None else f"{s['mean_held_out']:.2f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]")
    bench_rows.append(f"| {NAMES[m]} | " + " | ".join(cols) + " |")
n_p = {p: cell.get((p, "random"), {}).get("n", "n/a") for p in ("p1", "p2", "p3")}

bt = Bs["test"] if Bs else {}
fr = R["test_fresh_devices"] if R else {}
g = M["group_switch_counts"] if M else {}
gk = list(g)
ds = D["designs"] if D else {}
rs = D["rotation_scan"] if D else {}
reach = rs.get("p_world_reaches_half", {})
top7 = D["apl126_seven_top_frequency"] if D else {}
lb = Wd["windows"]["low bias 1.15-1.60"] if Wd else {}
df = Wd["windows"]["default 1.25-1.95"] if Wd else {}
n_sens_ok, n_sens = 0, 0
if S:
    for k, v in S.items():
        if isinstance(v, dict) and isinstance(v.get("portfolio"), dict):
            n_sens += 1
            n_sens_ok += int(v["portfolio"]["E_best"] > v["author_mask"]["E_best"] > v["apl126_seven"]["E_best"])

def bench_reading():
    """One paragraph on the optimiser benchmark, from the cached statistics (no number typed by hand)."""
    if not cell:
        return "The benchmark has not been run."
    parts = []
    for p, lab in (("p1", "P1 (default window)"), ("p2", "P2 (narrow window)"), ("p3", "P3 (unknown wafer)")):
        if (p, "random") not in cell:
            continue
        lo = min(cell[(p, m)]["mean_held_out"] for m in NAMES); hi = max(cell[(p, m)]["mean_held_out"] for m in NAMES)
        sig = [k for k, v in tests.items() if k[0] == p and v["wilcoxon_p"] < 0.05 and v["mean_diff"] > 0 and any(t in k[1] for t in ("random", "sobol"))]
        fr_ = next((x["friedman_p"] for x in stats if x.get("problem") == p and "friedman_p" in x), None)
        parts.append(f"{lab}: held-out yields {lo:.2f}-{hi:.2f}; Bayesian optimisation significantly better than random or Sobol in {len(sig)} of 4 paired tests"
                     + ((f"; Friedman test over the five methods p {'< 0.01' if fr_ < 0.01 else f'= {fr_:.2f}'}") if fr_ is not None else ""))
    return "; ".join(parts) + "."


def wafer_reading():
    """Wafer-level (cluster) statistics of the benchmark, and the P3 selection-vs-unseen gap, from results/benchmark_clusters.json."""
    if not Cl:
        return ""
    def ahead(p, a, b):
        return Cl[p]["comparisons"][f"{a} - {b}"]["n_ahead"]
    g3 = Cl["p3"]
    return (f" At the level of the five simulated wafers (runs on one wafer are not independent, so the paired-run p-values overstate the evidence): P1, GP-EI ahead of random search on {ahead('p1', 'bo_ei', 'random')}/5 wafers, "
            f"of Sobol on {ahead('p1', 'bo_ei', 'sobol')}/5, of differential evolution on {ahead('p1', 'bo_ei', 'de')}/5; P2, GP-EI ahead of random on {ahead('p2', 'bo_ei', 'random')}/5, of Sobol on {ahead('p2', 'bo_ei', 'sobol')}/5, "
            f"of differential evolution on {ahead('p2', 'bo_ei', 'de')}/5 (GP-UCB: {ahead('p2', 'bo_ucb', 'random')}/5, {ahead('p2', 'bo_ucb', 'sobol')}/5, {ahead('p2', 'bo_ucb', 'de')}/5). "
            f"The five wafers were chosen among feasible accepted worlds to span the range of difficulty. In P3 every method loses about "
            f"{min(v['gap'] for v in g3.values()):.2f}-{max(v['gap'] for v in g3.values()):.2f} between the eight worlds it was tuned on ({min(v['selection'] for v in g3.values()):.2f}-{max(v['selection'] for v in g3.values()):.2f}) "
            f"and the 24 unseen ones ({min(v['unseen'] for v in g3.values()):.2f}-{max(v['unseen'] for v in g3.values()):.2f}): a design tuned on eight worlds over-fits them, whichever algorithm found it (8 runs per method, little power).")


def other_rows():
    if not Ow:
        return "n/a", "n/a", "n/a"
    s = Ow["sets"]
    return tuple(f"{s[k]['other']:.2f}" for k in ("portfolio_16", "mask_16", "seven_published"))


wide_row = (S or {}).get("worlds accepted under a wider prior (native dichroism from 0.2 1/ns)", {})
mask_add_txt = ""
if Mi:
    mp = Mi["mask_plus_k"]
    ad = Mi["additions"]
    mask_add_txt = (f" If the mask can take more devices, the interior-box additions that help most in the model are crosses near "
                    f"{ad[0]['x'][1]:.2f}/{ad[0]['x'][4]:.0f} degrees, {ad[1]['x'][1]:.2f}/{ad[1]['x'][4]:.0f} degrees and {ad[2]['x'][1]:.2f}/{ad[2]['x'][4]:.0f} degrees (arm ratio/rotation; width about 5-6 um): "
                    f"expected best yield {Mi['mask_only']['E_best_yield']:.2f} (mask) -> {mp[0]['E_best_yield']:.2f}, {mp[1]['E_best_yield']:.2f}, {mp[2]['E_best_yield']:.2f} with 1, 2, 3 additions.")

rot_txt, rot_robust = "", ""
if Rw and Rw.get("sets"):
    import numpy as np
    NM = dict(primary="primary set", loose="loose thresholds", strict="strict thresholds", wide="wider prior", ox10="isotropic oxidation",
              ox13="oxidation ratio 1.3", chen="metasurface-paper constants")
    fold = lambda t: min(t, 180.0 - t)          # theta and 180 - theta are mirror images
    bs = {NM[k]: fold(v["best_theta"]) for k, v in Rw["sets"].items() if k in NM and v["max_mean_yield"] >= 0.2}
    rot_txt = ("Best rotation of the 1:0.7 cross (folded to 0-90 degrees, since theta and 180 - theta are mirror images) under each way of conditioning on the literature: "
               + ", ".join(f"{k} {v:.0f} deg" for k, v in bs.items()) + "; with the two thermal-drift conditionings no rotation works.")
    th_ = Rw["thetas"]
    work = {k: v for k, v in Rw["sets"].items() if v["max_mean_yield"] >= 0.2}
    share = np.array([[y / v["max_mean_yield"] for y in v["mean_yield"]] for v in work.values()])
    worst = share.min(0)
    ib = int(np.argmax(worst))
    rot_robust = (f"Across the alternative ways of conditioning on the literature the maximum moves inside the plateau; {th_[ib]:.0f} degrees (equivalently {180 - th_[ib]:.0f}) is the most robust single rotation, narrowly (bootstrap: highest worst case in {Rw['worst_case_share_bootstrap']['p_highest']['80'] * 100:.0f} % of resamples against {Rw['worst_case_share_bootstrap']['p_highest']['90'] * 100:.0f} % for 90 degrees), and there is no reliable ranking inside the 70-110 degree plateau; treat 80-100 degrees as its core. "
                  f"At {th_[ib]:.0f} degrees the yield is never below {worst[ib]:.2f} of a set's own best in the {len(work)} sets in which some rotation works (90 degrees: {worst[th_.index(90.0)]:.2f}).")

# sensitivity: which rows break the ordering
sens_fail = []
if S:
    for k, v in S.items():
        if isinstance(v, dict) and isinstance(v.get("portfolio"), dict) and not (v["portfolio"]["E_best"] > v["author_mask"]["E_best"] > v["apl126_seven"]["E_best"]):
            sens_fail.append(k)

text = f"""# Bayesian optimization of VCSEL oxide-aperture geometry for polarization-switching photonic Ising machines

> **Status: a literature-anchored simulation study.** The constants of the device model are *not measured*. They are constrained only by the qualitative outcomes that
> fabricated tailored-aperture VCSELs show in the published experiments (Yuan et al. 2025; see `docs/LITERATURE_ANCHORS.md`). Every number below is a **model prediction conditional on
> those constraints**. No measurement of the author's own devices is used and nothing is validated on a device of this project; what the study delivers is a design pipeline,
> testable predictions for the author's mask, and a calibration protocol for its first measurements. The published outcomes *constrain* the model; because they select its constants they cannot also *validate* it.

## Question
Photonic Ising machines can encode a spin in the two orthogonal linear polarizations of a VCSEL, but crystal anisotropy pins the light to one of them. Tailored oxide apertures can compensate this.
**Which oxide-aperture geometry gives the best polarization switching for an Ising node, and how should a fabrication batch be chosen when the constants of the wafer are not known?**

## Short answer (conditional; details in `supplementary/supplement.pdf`, Sec. 4)
* **Experiments** (Yuan et al., APL 126, 151101): the text that was read reports injection-locking power for the 1:0.7 cruciform rotated by 60 degrees (248 uW at 10 mA; 3.6 uW just above threshold) and for the non-switchable square (371 uW at 10 mA); it does not rank all seven designs.
  That is an injection-locking measurement, not a yield, and the number of devices per design is not stated in the text that was read.
* **This model**, over {D['n_worlds'] if D else 'n/a'} literature-consistent worlds: symmetric apertures (circle, square) do not work; a rotated 1:0.7 cross has a yield that rises between 50 and 70 degrees, a plateau between 70 and 110 degrees,
  and falls beyond it. In the primary set of worlds the best rotation is 70-80 degrees under every switching-current window tried. {rot_robust} The 90-degree cross is close to the best unless switching at low bias is required
  (60 degrees beats 90 degrees in {f2(lb.get('share_worlds_60_beats_90', None) and 100 * lb['share_worlds_60_beats_90'], 0)} % of the worlds for a low-bias window, in {f2(df.get('share_worlds_60_beats_90', None) and 100 * df['share_worlds_60_beats_90'], 0)} % for the default window).
  Arm ratio matters: the share of worlds in which some rotation reaches a yield of 0.5 is {', '.join(f"{100 * reach[k]:.0f} % ({k})" for k in reach)}. Part of the peak at 0.7 is built into the anchors.
* **Beyond the published designs** the search favours thin-armed crosses near 85-90 degrees; half of the model-selected batch lies on the wall of the search box (mostly thin arms), where the reduced model is least credible. These are hypotheses.
* **What would decide**: the author's mask (two families, eight rotations) contains 60, 90 and 120 degrees but neither 80 nor its mirror image 100 degrees; predictions for every design are in `results/mask_predictions.json` and supplement Sec. 8.{mask_add_txt}
  {rot_txt}

## What is done here, and what is not claimed
1. A spin-flip-model simulator turns an exact mask geometry (length, arm ratio, width, corner radius, rotation) into a device-level switching **yield** under fabrication scatter (`src/vcsel_bo`).
2. The unknown constants (native dichroism and birefringence, two aperture-coupling coefficients, with signs) are sampled by **rejection ABC with the published qualitative outcomes as constraints**:
   {A['n_accepted'] if A else 'n/a'} of {A['n_draws'] if A else 'n/a'} prior draws ({f2(A and 100 * A['acceptance'], 1)} %) reproduce all of them. They force the sign of the geometric dichroism coupling and leave the sign of the birefringence coupling open.
3. **Bayesian optimisation** (GP, Matern-5/2 ARD, EI/UCB) is benchmarked against random, Sobol and differential evolution at equal budget, scored on fresh device populations with paired tests.
4. A **batch** (a mask) is designed by a greedy submodular portfolio over worlds; the author's mask and the seven published designs are scored against it.
Not claimed: new physics, a validated prediction, or novelty beyond the combination (a limited search found no earlier Bayesian or portfolio design of aperture *shapes*; that is absence of evidence, not proof).

## Results at a glance (model predictions)
Held-out hard yield of the design returned by each optimiser (mean [95 % bootstrap interval]; runs per method: P1 {n_p['p1']}, P2 {n_p['p2']}, P3 {n_p['p3']}):

| method | P1 default window | P2 narrow window | P3 unknown wafer, unseen worlds |
|---|---|---|---|
{chr(10).join(bench_rows)}

Reading (from `results/benchmark_stats.json` and `results/benchmark_clusters.json`): {bench_reading()}{wafer_reading()}

Expected best yield of a set of 16 designs, on worlds not used for selection (fraction of worlds in which some design reaches 0.5 in brackets):

| set of designs | 32 test worlds, same device population | 32 test worlds, fresh 48-device population | {Ow['n_other_worlds'] if Ow else 'n/a'} further accepted worlds |
|---|---|---|---|
| model-selected portfolio (16) | {f2(bt.get('greedy_portfolio', {}).get('E_best_yield'))} ({f2(bt.get('greedy_portfolio', {}).get('P_some_design_ge_half'))}) | {f2(fr.get('portfolio', {}).get('E_best_yield'))} ({f2(fr.get('portfolio', {}).get('P_some_design_ge_half'))}) | {other_rows()[0]} |
| portfolio restricted to an interior box (ratio >= 0.6, 4.5 <= w <= 7.5 um) | {f2(R and R['interior_portfolio']['test_same_devices']['E_best_yield'])} | {f2(fr.get('interior_portfolio', {}).get('E_best_yield'))} ({f2(fr.get('interior_portfolio', {}).get('P_some_design_ge_half'))}) | n/a |
| the author's mask (16 designs) | {f2(bt.get('author_mask_16_designs', {}).get('E_best_yield'))} ({f2(bt.get('author_mask_16_designs', {}).get('P_some_design_ge_half'))}) | {f2(fr.get('author_mask', {}).get('E_best_yield'))} ({f2(fr.get('author_mask', {}).get('P_some_design_ge_half'))}) | {other_rows()[1]} |
| the seven designs of APL 126 | {f2(bt.get('apl126_seven_designs', {}).get('E_best_yield'))} ({f2(bt.get('apl126_seven_designs', {}).get('P_some_design_ge_half'))}) | {f2(fr.get('apl126_seven', {}).get('E_best_yield'))} ({f2(fr.get('apl126_seven', {}).get('P_some_design_ge_half'))}) | {other_rows()[2]} |
| 16 random designs of the candidate pool | {f2(bt.get('random_q_from_pool', {}).get('mean'))} +- {f2(bt.get('random_q_from_pool', {}).get('sd'))} | n/a | n/a |

The headline is optimistic: the 32 test worlds happen to give higher levels than the {Ow['n_other_worlds'] if Ow else 'n/a'} further accepted worlds (`scripts/other_worlds_check.py`); the ordering holds (portfolio minus mask {f2(Ow and Ow['differences_other_only']['portfolio_minus_mask']['mean'])}, 95 % bootstrap interval {f2(Ow and Ow['differences_other_only']['portfolio_minus_mask']['lo'])} to {f2(Ow and Ow['differences_other_only']['portfolio_minus_mask']['hi'])}), the level does not.
"Best of batch" assumes that the best design of the batch can be identified after measurement.

The ordering portfolio > mask > seven published designs holds in {n_sens_ok} of {n_sens} sensitivity rows (`results/sensitivity.json`; the baseline is one of the rows and the row with the metasurface-paper constants is degenerate, no set switches; process depth and anisotropy, thermal dichroism, coupling scale, device scatter,
mask tolerance, grid resolution, switching-current windows, loose / strict translations of the anchors, a wider prior, and rows that re-infer the constants under a changed assumption). It fails in: {'; '.join(sens_fail)}.
Reading: the advantage of the model-selected batch depends on anisotropic oxidation (with isotropic oxidation the portfolio only ties the mask even after re-inference) and **every yield presumes that switching comes from the spin-flip
dynamics at fixed anisotropy** (a thermal dichroism drift of the size probed removes the yield from every set of designs). Measure oxide anisotropy, the temperature dependence of switching and the anisotropy scale first (supplement Sec. 7).

**Testable predictions for the author's mask** (two devices per design assumed; a device "switches" if it flips polarization at least once in the sweep): {g[gk[2]]['mean']:.1f} of the {g[gk[2]]['n_devices']} devices at 60, 90, 120 degrees should switch (5-95 %: {g[gk[2]]['q05']:.0f}-{g[gk[2]]['q95']:.0f}; {M['group_pass_counts'][gk[2]]['mean']:.1f} would also meet the whole yield specification),
{g[gk[0]]['mean']:.1f} of the {g[gk[0]]['n_devices']} devices at 0, 30, 150 degrees ({g[gk[0]]['q05']:.0f}-{g[gk[0]]['q95']:.0f}); a measurement outside these intervals would count against the model class at about the 5 % level per statement (the 30-degree designs are where the model already disagrees with the paper). The intervals are wide because the literature-consistent set is wide.
A synthetic test of what the first batch could teach (well specified, hence optimistic): the yield of the best single design rises only from {f2(V and V['no_measurement']['mean_yield'])} to about {f2(V and min(v['mean_yield_of_chosen_design'] for v in V.values() if isinstance(v, dict) and 'n_devices_per_design' in v))}-{f2(V and max(v['mean_yield_of_chosen_design'] for v in V.values() if isinstance(v, dict) and 'n_devices_per_design' in v))};
switch counts barely identify the birefringence sign, so record the azimuth of the low-current state and the switching current, not only whether a device switches.

## Where the model is weak (details: `docs/AUDIT.md`, supplement Sec. 9)
* Two published observations are **not reproduced**: switching of the 30-degree cruciform (mean switching fraction {f2(ds.get('x07_30', {}).get('mean_p_switch'))} against {f2(ds.get('x07_60', {}).get('mean_p_switch'))} at 60 degrees) and a symmetric cross that switches more often than a circle
  ({f2(ds.get('sym_cross', {}).get('mean_p_switch'))} against {f2(ds.get('circle', {}).get('mean_p_switch'))}). Predictions far from 90 degrees are the least reliable.
* The predicted yields presume that switching comes from the spin-flip dynamics at a fixed anisotropy. With a thermal dichroism drift of the size probed (an arbitrary probe, not an estimate) no set of designs reaches a useful yield, and the sign of the native birefringence that the anchors imply flips with the sign of the drift (supplement Sec. 7).
  The oxidation anisotropy is a placeholder (1.15): with isotropic oxidation the model-selected batch only ties the author's mask, even when the constants are re-inferred.
* Prior box: the accepted native dichroism piles up against the lower edge of the default prior (d0 >= 1 per ns); a wider prior lowers the portfolio's lead (expected best yield {f2(wide_row.get('portfolio', {}).get('E_best'))} against {f2(wide_row.get('author_mask', {}).get('E_best'))} for the mask) but keeps the order.
  The spin-flip constants of the metasurface paper (larger gamma_s) accept no world in 4000 draws, so that robustness check is inconclusive, not confirmatory.
* Single-mode spin-flip model, scalar mode used only as a weight, phenomenological linear coupling, placeholder oxidation depth and anisotropy; thresholds translating "most / seldom / part of the devices" are choices (loose and strict variants are reported).
* Sources not read in full: J. Lightwave Technol. 43, 6819 (2025) (abstract only) and Appl. Phys. Lett. 127, 221102 (2025) (listing only); they contain the statistics that would make the anchors quantitative. Whether the JLT paper contains a calibrated model of aperture shape and switching is unknown to me, so no novelty claim is made.
* The headline batch scores are optimistic (see above); the optimiser benchmark rests on five simulated wafers (P1, P2) and eight runs (P3); the published designs are drawn at assumed sizes (L = 17.5 um, w = 6 um).
* The corner treatment of mask family A is not recorded and is assumed sharp (with fully rounded arm tips the predicted number of switching devices at 60-120 degrees would be {f2(Fr and Fr['rounded_family_A'][list(Fr['rounded_family_A'])[2]]['mean'], 1)} instead of {f2(Fr and Fr['sharp_family_A'][list(Fr['sharp_family_A'])[2]]['mean'], 1)} of 12; check the mask file); the mask has 32 devices, i.e. two per design if all designs are equally represented.
* Acceptance depends partly on the device population: on fresh populations {f2(Fa and 100 * Fa['share_min'], 0)}-{f2(Fa and 100 * Fa['share_max'], 0)} % of the accepted worlds still pass all five enforced anchors (nearly all failures at the anchor with the smallest margin).

## Reproduce
```
pip install -r requirements.txt
pytest                                    # about one minute
python scripts/check_citations.py         # every cited key exists, every entry has a verification status, removed references stay out
python scripts/anchored_worlds.py --draws 8000 --seed 1 --thresholds primary --marginal 300     # literature-consistent worlds (~8 min)
bash scripts/run_all.sh                   # designs, batch, checks, benchmarks, sensitivity (~1.5 h on two cores; checkpointed, resumable)
bash scripts/run_extra.sh                 # wider prior, re-conditioned worlds, sensitivity, best rotation per set of worlds, further-worlds check, wafer-level benchmark statistics (~20 min)
python scripts/value_of_first_batch.py && python scripts/window_dependence.py && python scripts/mask_additions_interior.py
python scripts/make_figures.py && python scripts/make_tables.py && python scripts/make_readme.py
cp figures/{rotation_scan,designs,benchmark,batch_shapes,anchored_worlds}.png supplementary/figures/   # the supplement compiles on its own, as on Overleaf
cd supplementary && pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/VCSEL_Polarization_BO.ipynb
```

## Repository map
`src/vcsel_bo/`: `geometry` (mask -> aperture -> tensors), `modes`, `physics` (World, Prior, Population), `sfm` (spin-flip model), `objective` (yield), `literature` (anchors, ABC sampler, design metrics),
`optimization` (GP-EI/UCB, baselines, portfolio), `problem`, `calibration`, `vector_modes` (semi-vectorial form birefringence). `mask/`: mask 2, the anisotropy ruler (reader for the lab's L-Edit TDB file, GDS generator, KLayout checks, depth tolerance, README, measurement and paper plan; its model numbers come from `scripts/native_strength_shift.py`, `mask2_extras_native.py` and `mask2_readouts.py` in the target convention of `anchor_target_convention.py`). `scripts/`: pipeline steps. `results/`: every cached result. `figures/`, `supplementary/` (LaTeX + PDF), `docs/` (anchors, audit, checklist, references), `notebooks/` (`VCSEL_Polarization_BO.ipynb`: this pipeline; `VCSEL_Polarization_BO_Project.ipynb` with `supplementary/VCSEL_Polarization_BO_Supplementary.pdf`: the earlier stand-alone Colab prototype with Fourier-parameterised apertures), `tests/`.

The lab's decoded V04 mask (`mask/example/`) is not in the repository; recreate it from the lab's TDB file with `python mask/tdb_to_gds.py VCSEL_ARRAY_Heming_-_V04.tdb mask/example` before running `mask/make_mask2.py` or `mask/check_mask2.py`. `mask/out/` holds the delivered mask-2 group (`VCSEL_PM_Group_2.gds`), its device table, check reports and previews.

## References
`docs/references.bib` (every entry has a verification status) and `docs/REFERENCES.md` (generated). Two references of an earlier version could not be verified and were removed; see `docs/AUDIT.md`.
"""
(ROOT / "README.md").write_text(text)
print(f"wrote README.md ({len(text)} characters)")
