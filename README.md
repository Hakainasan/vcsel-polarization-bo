# Bayesian optimization of VCSEL oxide-aperture geometry for polarization-switching photonic Ising machines

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
* **This model**, over 181 literature-consistent worlds: symmetric apertures (circle, square) do not work; a rotated 1:0.7 cross has a yield that rises between 50 and 70 degrees, a plateau between 70 and 110 degrees,
  and falls beyond it. In the primary set of worlds the best rotation is 70-80 degrees under every switching-current window tried. Across the alternative ways of conditioning on the literature the maximum moves inside the plateau; 80 degrees (equivalently 100) is the most robust single rotation, narrowly (bootstrap: highest worst case in 80 % of resamples against 7 % for 90 degrees), and there is no reliable ranking inside the 70-110 degree plateau; treat 80-100 degrees as its core. At 80 degrees the yield is never below 0.91 of a set's own best in the 6 sets in which some rotation works (90 degrees: 0.87). The 90-degree cross is close to the best unless switching at low bias is required
  (60 degrees beats 90 degrees in 53 % of the worlds for a low-bias window, in 31 % for the default window).
  Arm ratio matters: the share of worlds in which some rotation reaches a yield of 0.5 is 9 % (0.5), 34 % (0.6), 74 % (0.7), 24 % (0.8), 1 % (0.9). Part of the peak at 0.7 is built into the anchors.
* **Beyond the published designs** the search favours thin-armed crosses near 85-90 degrees; half of the model-selected batch lies on the wall of the search box (mostly thin arms), where the reduced model is least credible. These are hypotheses.
* **What would decide**: the author's mask (two families, eight rotations) contains 60, 90 and 120 degrees but neither 80 nor its mirror image 100 degrees; predictions for every design are in `results/mask_predictions.json` and supplement Sec. 8. If the mask can take more devices, the interior-box additions that help most in the model are crosses near 0.72/70 degrees, 0.78/80 degrees and 0.65/91 degrees (arm ratio/rotation; width about 5-6 um): expected best yield 0.68 (mask) -> 0.78, 0.80, 0.83 with 1, 2, 3 additions.
  Best rotation of the 1:0.7 cross (folded to 0-90 degrees, since theta and 180 - theta are mirror images) under each way of conditioning on the literature: primary set 70 deg, loose thresholds 70 deg, strict thresholds 80 deg, wider prior 80 deg, isotropic oxidation 90 deg, oxidation ratio 1.3 80 deg; with the two thermal-drift conditionings no rotation works.

## What is done here, and what is not claimed
1. A spin-flip-model simulator turns an exact mask geometry (length, arm ratio, width, corner radius, rotation) into a device-level switching **yield** under fabrication scatter (`src/vcsel_bo`).
2. The unknown constants (native dichroism and birefringence, two aperture-coupling coefficients, with signs) are sampled by **rejection ABC with the published qualitative outcomes as constraints**:
   181 of 8000 prior draws (2.3 %) reproduce all of them. They force the sign of the geometric dichroism coupling and leave the sign of the birefringence coupling open.
3. **Bayesian optimisation** (GP, Matern-5/2 ARD, EI/UCB) is benchmarked against random, Sobol and differential evolution at equal budget, scored on fresh device populations with paired tests.
4. A **batch** (a mask) is designed by a greedy submodular portfolio over worlds; the author's mask and the seven published designs are scored against it.
Not claimed: new physics, a validated prediction, or novelty beyond the combination (a limited search found no earlier Bayesian or portfolio design of aperture *shapes*; that is absence of evidence, not proof).

## Results at a glance (model predictions)
Held-out hard yield of the design returned by each optimiser (mean [95 % bootstrap interval]; runs per method: P1 20, P2 40, P3 8):

| method | P1 default window | P2 narrow window | P3 unknown wafer, unseen worlds |
|---|---|---|---|
| random search | 0.81 [0.74, 0.88] | 0.33 [0.29, 0.37] | 0.39 [0.36, 0.42] |
| Sobol points | 0.87 [0.80, 0.92] | 0.35 [0.30, 0.39] | 0.42 [0.38, 0.46] |
| differential evolution | 0.89 [0.84, 0.94] | 0.38 [0.34, 0.42] | 0.37 [0.32, 0.41] |
| GP-EI (Bayesian optimisation) | 0.96 [0.92, 0.98] | 0.43 [0.38, 0.47] | 0.41 [0.36, 0.45] |
| GP-UCB (Bayesian optimisation) | 0.86 [0.74, 0.96] | 0.43 [0.38, 0.48] | 0.40 [0.38, 0.43] |

Reading (from `results/benchmark_stats.json` and `results/benchmark_clusters.json`): P1 (default window): held-out yields 0.81-0.96; Bayesian optimisation significantly better than random or Sobol in 2 of 4 paired tests; Friedman test over the five methods p < 0.01; P2 (narrow window): held-out yields 0.33-0.43; Bayesian optimisation significantly better than random or Sobol in 4 of 4 paired tests; Friedman test over the five methods p < 0.01; P3 (unknown wafer): held-out yields 0.37-0.42; Bayesian optimisation significantly better than random or Sobol in 0 of 4 paired tests; Friedman test over the five methods p = 0.18. At the level of the five simulated wafers (runs on one wafer are not independent, so the paired-run p-values overstate the evidence): P1, GP-EI ahead of random search on 5/5 wafers, of Sobol on 5/5, of differential evolution on 4/5; P2, GP-EI ahead of random on 4/5, of Sobol on 5/5, of differential evolution on 4/5 (GP-UCB: 5/5, 5/5, 4/5). The five wafers were chosen among feasible accepted worlds to span the range of difficulty. In P3 every method loses about 0.16-0.24 between the eight worlds it was tuned on (0.58-0.63) and the 24 unseen ones (0.37-0.42): a design tuned on eight worlds over-fits them, whichever algorithm found it (8 runs per method, little power).

Expected best yield of a set of 16 designs, on worlds not used for selection (fraction of worlds in which some design reaches 0.5 in brackets):

| set of designs | 32 test worlds, same device population | 32 test worlds, fresh 48-device population | 117 further accepted worlds |
|---|---|---|---|
| model-selected portfolio (16) | 0.86 (0.94) | 0.86 (0.94) | 0.74 |
| portfolio restricted to an interior box (ratio >= 0.6, 4.5 <= w <= 7.5 um) | 0.83 | 0.81 (0.91) | n/a |
| the author's mask (16 designs) | 0.68 (0.72) | 0.67 (0.69) | 0.61 |
| the seven designs of APL 126 | 0.57 (0.59) | 0.58 (0.59) | 0.56 |
| 16 random designs of the candidate pool | 0.69 +- 0.12 | n/a | n/a |

The headline is optimistic: the 32 test worlds happen to give higher levels than the 117 further accepted worlds (`scripts/other_worlds_check.py`); the ordering holds (portfolio minus mask 0.13, 95 % bootstrap interval 0.09 to 0.16), the level does not.
"Best of batch" assumes that the best design of the batch can be identified after measurement.

The ordering portfolio > mask > seven published designs holds in 19 of 26 sensitivity rows (`results/sensitivity.json`; the baseline is one of the rows and the row with the metasurface-paper constants is degenerate, no set switches; process depth and anisotropy, thermal dichroism, coupling scale, device scatter,
mask tolerance, grid resolution, switching-current windows, loose / strict translations of the anchors, a wider prior, and rows that re-infer the constants under a changed assumption). It fails in: process ox_ratio=1.0; chi_th=-0.2; chi_th=+0.2; coupling x0.5; SFM constants of the metasurface paper (same worlds); RECONDITIONED: oxidation isotropic (ox_ratio=1.0); RECONDITIONED: thermal dichroism drift chi_th=-0.2.
Reading: the advantage of the model-selected batch depends on anisotropic oxidation (with isotropic oxidation the portfolio only ties the mask even after re-inference) and **every yield presumes that switching comes from the spin-flip
dynamics at fixed anisotropy** (a thermal dichroism drift of the size probed removes the yield from every set of designs). Measure oxide anisotropy, the temperature dependence of switching and the anisotropy scale first (supplement Sec. 7).

**Testable predictions for the author's mask** (two devices per design assumed; a device "switches" if it flips polarization at least once in the sweep): 5.5 of the 12 devices at 60, 90, 120 degrees should switch (5-95 %: 1-11; 3.4 would also meet the whole yield specification),
0.5 of the 12 devices at 0, 30, 150 degrees (0-4); a measurement outside these intervals would count against the model class at about the 5 % level per statement (the 30-degree designs are where the model already disagrees with the paper). The intervals are wide because the literature-consistent set is wide.
A synthetic test of what the first batch could teach (well specified, hence optimistic): the yield of the best single design rises only from 0.64 to about 0.68-0.70;
switch counts barely identify the birefringence sign, so record the azimuth of the low-current state and the switching current, not only whether a device switches.

## Where the model is weak (details: `docs/AUDIT.md`, supplement Sec. 10)
* Two published observations are **not reproduced**: switching of the 30-degree cruciform (mean switching fraction 0.07 against 0.55 at 60 degrees) and a symmetric cross that switches more often than a circle
  (0.02 against 0.00). Predictions far from 90 degrees are the least reliable.
* The predicted yields presume that switching comes from the spin-flip dynamics at a fixed anisotropy. With a thermal dichroism drift of the size probed (an arbitrary probe, not an estimate) no set of designs reaches a useful yield, and the sign of the native birefringence that the anchors imply flips with the sign of the drift (supplement Sec. 7).
  The oxidation anisotropy is a placeholder (1.15): with isotropic oxidation the model-selected batch only ties the author's mask, even when the constants are re-inferred.
* Prior box: the accepted native dichroism piles up against the lower edge of the default prior (d0 >= 1 per ns); a wider prior lowers the portfolio's lead (expected best yield 0.72 against 0.63 for the mask) but keeps the order.
  The spin-flip constants of the metasurface paper (larger gamma_s) accept no world in 4000 draws, so that robustness check is inconclusive, not confirmatory.
* Single-mode spin-flip model, scalar mode used only as a weight, phenomenological linear coupling, placeholder oxidation depth and anisotropy; thresholds translating "most / seldom / part of the devices" are choices (loose and strict variants are reported).
* Sources not read in full: J. Lightwave Technol. 43, 6819 (2025) (abstract only) and Appl. Phys. Lett. 127, 221102 (2025) (listing only); they contain the statistics that would make the anchors quantitative. Whether the JLT paper contains a calibrated model of aperture shape and switching is unknown to me, so no novelty claim is made.
* The headline batch scores are optimistic (see above); the optimiser benchmark rests on five simulated wafers (P1, P2) and eight runs (P3); the published designs are drawn at assumed sizes (L = 17.5 um, w = 6 um).
* The corner treatment of mask family A is not recorded and is assumed sharp (with fully rounded arm tips the predicted number of switching devices at 60-120 degrees would be 5.0 instead of 5.5 of 12; check the mask file); the mask has 32 devices, i.e. two per design if all designs are equally represented.
* Acceptance depends partly on the device population: on fresh populations 83-92 % of the accepted worlds still pass all five enforced anchors (nearly all failures at the anchor with the smallest margin).

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
cd supplementary && pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/VCSEL_Polarization_BO.ipynb
```

## Repository map
`src/vcsel_bo/`: `geometry` (mask -> aperture -> tensors), `modes`, `physics` (World, Prior, Population), `sfm` (spin-flip model), `objective` (yield), `literature` (anchors, ABC sampler, design metrics),
`optimization` (GP-EI/UCB, baselines, portfolio), `problem`, `calibration`, `vector_modes` (semi-vectorial form birefringence). `mask/`: mask 2, the anisotropy ruler (reader for the lab's L-Edit TDB file, GDS generator, KLayout checks, depth tolerance, README, measurement and paper plan; its model numbers come from `scripts/native_strength_shift.py`, `mask2_extras_native.py` and `mask2_readouts.py` in the target convention of `anchor_target_convention.py`). `scripts/`: pipeline steps. `results/`: every cached result. `figures/`, `supplementary/` (LaTeX + PDF), `docs/` (anchors, audit, checklist, references), `notebooks/` (`VCSEL_Polarization_BO.ipynb`: this pipeline; `VCSEL_Polarization_BO_Project.ipynb` with `supplementary/VCSEL_Polarization_BO_Supplementary.pdf`: the earlier stand-alone Colab prototype with Fourier-parameterised apertures), `tests/`.

The lab's decoded V04 mask (`mask/example/`) is not in the repository; recreate it from the lab's TDB file with `python mask/tdb_to_gds.py VCSEL_ARRAY_Heming_-_V04.tdb mask/example` before running `mask/make_mask2.py` or `mask/check_mask2.py`. `mask/out/` holds the delivered mask-2 group (`VCSEL_PM_Group_2.gds`), its device table, check reports and previews.

## References
`docs/references.bib` (every entry has a verification status) and `docs/REFERENCES.md` (generated). Two references of an earlier version could not be verified and were removed; see `docs/AUDIT.md`.
