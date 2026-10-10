# Final checklist (version 0.5.0, 2026-09-30)

Everything computed in this repository is a **model prediction** conditional on the qualitative outcomes reported for fabricated devices; nothing is measured on the author's own devices.

## Complete (done, and verified by running the code)
- Spin-flip simulator: fixed points and linear stability; 33 of 36 sweep points agree with time-domain integration (the three disagreements are at I/I_th = 2.2, outside the range used, I/I_th <= 2.1).
- Geometry layer (exact cruciform, anisotropic oxidation as erosion, second-rank tensors) and the yield objective (hard yield for reporting, smoothed for optimisation).
- Literature anchoring by rejection ABC: 181 of 8000 draws accepted; loose / strict thresholds; wider prior; four re-conditioned variants; a run with the spin-flip constants of the metasurface paper (0 accepted).
- Optimiser benchmark P1, P2, P3 with paired tests **and** wafer-level statistics (five wafers); selection-versus-unseen gap of P3.
- Batch design: portfolio, additions to the author's mask, interior-only variant, fresh device populations, and 117 further accepted worlds not used for selection.
- Sensitivity table (26 rows) and the best rotation under every conditioning, with a bootstrap over worlds.
- Predictions for the author's mask (switching counts, pass counts, axial azimuth, sensitivity to rounded arm tips of family A); calibration module and a synthetic first-batch test.
- 61 unit / regression / citation tests pass; the citation-consistency check passes; the notebook executes without errors; the supplement compiles (20 pages).
- Reference list: all 28 entries were confirmed to exist by an independent check; six corrections applied.
- Application texts: abstract (4,937 characters), motivation letter, keywords and TL;DR, lecture notes.

- Semi-vectorial mode solver (`src/vcsel_bo/vector_modes.py`, 2026-10-01): matches the exact TE/TM slab splitting to 0.1 %, gives zero splitting for a circle, follows the long axis of an ellipse under rotation, and changes by 1.3 % between grids of 0.1 and 0.07 um. Result (`scripts/form_birefringence.py`): the form birefringence of your mask designs is 0.07-0.15 ns^-1 (20-50 MHz) at an index step of 0.01, and at most 0.4 ns^-1 at 0.05. That is 25-450 times below the measured splittings at the assumed step (at least 10 times at 0.05) (4-31 ns^-1) and below the stress term the anchored worlds need, so the shape acts on the birefringence through strain, not through the optical guide. 67 tests pass.

## Uncertain (not resolved; treat as open)
- The spin-flip constants are typical magnitudes, not fitted. The constants of the metasurface paper accept no world, so that check is inconclusive.
- Two published observations are not reproduced: switching of the 30-degree cross, and a symmetric cross switching more than a circle.
- Sources not read in full: J. Lightwave Technol. 43, 6819 (abstract only) and Appl. Phys. Lett. 127, 221102 (listing only). Please upload them; they would make the anchors quantitative.
- Quoted fragments come from arXiv HTML text and were not compared character by character with the PDFs; the numbers taken from the metasurface paper were not re-verified after the first pass.
- The number of devices per design and the mask sizes of the published experiments are not in the text that was read; published designs are drawn at assumed sizes.
- Assumptions about your mask: two devices per design (32 devices / 16 designs); family A with sharp corners (rounded tips: 5.0 instead of 5.5 of 12 switching devices at 60-120 degrees).
- 83-92 % of the accepted worlds still pass all five anchors on fresh device populations (acceptance is partly population luck).
- The headline best-of-batch yield 0.86 is optimistic: 0.74 on 117 further worlds; "best of batch" assumes the best design can be identified after measurement.
- Benchmark evidence rests on five simulated wafers (P1, P2, chosen among feasible worlds) and eight P3 runs.
- References: the full author list of APL 127 is missing; the venue of the mesa-orientation paper comes from an independent check.
- Facts in the letter that come from your own notes and could not be checked: the image series across currents and temperatures, the spatial-entropy analysis of near- and far-field images, and the description of the photomask (two aperture families at eight rotations). The OIST high-school story is in your own draft, and the course project is confirmed by your recommendation request.
- Degree status: the letter says PhD student, as your own draft and your request email do. An earlier version said "at ITMO University" (taken from a profile note, not from your text); it was removed. Add your university only if you want it named, and keep it identical to your CV.
- Recommendation letter: the application page does not say whether you upload it or the referee sends it; the reference is due on 10 October 2026 (a Saturday).
- The poster-abstract deadline (30 September) has no time or time zone on the page; if it means the end of the day in Japan, that is 17:59 in Riyadh.

## Needs experimental validation
- Oxide anisotropy and depth (cross-sections or infrared images of the oxide front): the model-selected batch beats the mask only if the oxidation is anisotropic.
- Whether switching follows current (spin-flip) or temperature (thermal drift): polarization-resolved L-I curves at several heat-sink temperatures. Every yield presumes spin-flip switching at fixed anisotropy.
- The scale of dichroism and birefringence produced by a given shape: polarization-resolved spectra below threshold.
- The predictions for your mask: about 5.5 of 12 devices switching at 60-120 degrees (1-11), about 0.5 of 12 near 0 degrees (0-4); the low-current polarization of each design.
- The 70-110 degree plateau of the 1:0.7 cross and its 80-100 degree core (your mask has 60, 90 and 120 degrees, but not 70, 80 or 100).
- Whether the thin-armed designs on the wall of the search box work at all (the reduced model is least credible there).
- The sign of the birefringence coupling: record the azimuth of the low-current state and the switching current, not only whether a device switches.
