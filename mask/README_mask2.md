# VCSEL_PM_Group_2: "anisotropy ruler" group for the lab's PM_Full frame (version 2.1, 2026-10-09)

**What it is.** `VCSEL_PM_Group_2.gds` holds one flat GDSII cell, `PM_Group_2`, of 2000 × 2000 µm. It is a drop-in replacement for `PM_Group_1` of `VCSEL_ARRAY_Heming_-_V04.tdb`, where `PM_Full` steps that group 6 × 6 over 12 × 12 mm. L-Edit opens it with File → Import → GDSII.

**Contents.** 42 single devices on a 6 × 7 grid (290 × 280 µm pitch), to the right of the lab's alignment strip. The strip is copied from the example:
- Its n-Metal, p-MetalPad, first-p-metal and Oxide/Nitride layers are identical to the example (XOR 0).
- The Mesa layer agrees to within 20 nm. The raw XOR is 6.5 µm², all of it in slivers thinner than 12 nm where circles were re-polygonised.

**Changed from version 1.**
- The pairs and arrays are removed.
- The selection was re-decided around one question: how much aperture elongation this wafer needs.
- Mask 1 goes on the same wafer in the same run and already covers the rotation angles 0–150°, so mask 2 spends every slot on elongation.

**Changed in 2.1, after an independent review of 2.0:**
- **Layout:** the replicates are placed by an exact integer program; see *Layout*.
- **Ring opening:** the contact-ring opening is a 4.6 µm parallel slot instead of a ±10° wedge, which was only 1.8–3.3 µm wide at these small ring radii.
- **Corners:** every convex corner is rounded with r ≥ 1.05 µm. The notch where the pad lead meets the pad ring gets a 2 µm fillet, so nothing on any layer is narrower than 2 µm.
- **Labels:** the glyphs no longer have pixels touching only at a corner, and they are mirrored like the lab's own strip text.
- **Clean geometry:** vertex noise of a few nm from boolean operations is removed; no edge is shorter than 5 nm.
- **Decoded example:** `example_decoded.gds` drew the full p-pad and passivation rings as filled disks; it is fixed (the mask itself was not affected).

**Before exposure: the mask x-axis must be aligned to the native polarization axis [011].** On the lab's 940-nm wafer every device lased along [011] (Lu et al. 2025). Every elongated device here has its long axis at 90° to mask x, i.e. along [01-1], and that is what makes it compensate.
- **Small errors don't matter.** The model's angle window is broad: at the published native strength, the switching share stays above half its peak from about 56° to 124°. Its maxima are near 75° and 105°, with a shallow dip at 90° that is within the model's uncertainty.
- **A 90° error does.** If x ends up along [01-1], the whole ruler pins instead of switching.
- Use the same alignment as mask 1.

**Layers** (all datatype 0):

| Number | Layer | What is drawn on it |
|---|---|---|
| 1 | Mesa | the mesa |
| 2 | n-Metal | n-contact, pads and labels |
| 3 | p-MetalPad | p-pad metal |
| 4 | Grid Layer | drawn target aperture; reference only, not printed |
| 5 | N Well | Ø 20 µm circle, as in the example |
| 6 | first-p-metal | contact ring |
| 7 | Oxide/Nitride | passivation |

**Process assumptions.**
- The drawn shape is the target aperture.
- The mesa is the target grown by 5 µm (round offset).
- Lateral oxidation is isotropic, 5 µm deep.

## Content

All elongated designs have their long axis at 90° (along mask y). "Aspect" is the aperture's second-moment aspect ratio (√ of the ratio of the principal second moments; 1 = no elongation). The model columns give the predicted share of switching devices (target convention, 5 µm, isotropic oxidation), with the wafer's native anisotropy taken as in the published 840-nm devices (×1) or two or three times stronger (×2, ×3).

| ID | Design (target) | Per group | Aperture after 5 µm: area, aspect | Model ×1 / ×2 / ×3 |
|---|---|---|---|---|
| X9 | cross 17.5 µm, arms 6 µm, arm ratio 0.9 | 4 | 184.9 µm², 1.13 | 0.00 / 0.00 / 0.00 |
| X8 | arm ratio 0.8 | 4 | 172.7, 1.29 | 0.10 / 0.00 / 0.00 |
| X7 | arm ratio 0.7 (= mask-1 A90) | 4 | 158.5, 1.50 | **0.58** / 0.16 / 0.02 |
| X6 | arm ratio 0.6 | 4 | 143.1, 1.77 | 0.17 / **0.39** / 0.20 |
| X5 | arm ratio 0.5 | 4 | 127.4, 2.15 | 0.03 / 0.30 / **0.29** |
| X4 | arm ratio 0.4 | 4 | 112.4, 2.63 | 0.01 / 0.11 / 0.27 |
| S130 | stadium, long axis 17 µm, length/width 1.30 | 3 | 185.2, 1.27 | **0.58** / 0.08 / 0.00 |
| S155 | stadium, length/width 1.55 | 3 | 160.3, 1.50 | 0.23 / **0.41** / 0.14 |
| S180 | stadium, length/width 1.80 | 3 | 141.1, 1.74 | 0.04 / 0.32 / **0.28** |
| S220 | stadium, length/width 2.20 | 3 | 118.3, 2.13 | 0.01 / 0.16 / 0.27 |
| CIRC | circle Ø 13.47 µm (the cross's target area), control | 2 | 142.2, 1.00 | 0 / 0 / 0 |
| SYM | symmetric cross 17.5 µm, arms 6 µm, control | 2 | 195.4, 1.00 | 0 / 0 / 0 |
| X7S | 0.75 × X7 (13.1 µm, arms 4.5 µm), smaller aperture | 2 | 91.8, 1.49 | 0.45 / 0.27 / 0.04 |

Bold marks the predicted peak of each series at each native strength.

**How to read the model.** These are model predictions, not measurements.
- **Why a peak:** switching needs balance, so the predicted share peaks where the aperture's elongation cancels the wafer's native pull. The stronger the pull, the further the peak moves towards stronger elongation.
- **Why the ruler is wide:** Lu et al. saw every device on this wafer lock to [011], which points to a stronger pull than in the published devices. The ruler therefore spans arm ratios 0.9 to 0.4 (aspect 1.13 to 2.63).
- **More predictions:** the model also predicts two read-outs that need no switching, the start axis just above threshold and the mode splitting. They are in `MEASUREMENT_AND_PAPER_PLAN.md` and in the CSV.

## Built-in tests

**Cross vs stadium, as matched pairs.** At 5 µm oxidation each stadium has nearly the aperture aspect of one cross (within 0.035) and an area within 8 % of it:

| Pair | Aspect (stadium / cross) | Area, µm² (stadium / cross) |
|---|---|---|
| S130 ↔ X8 | 1.27 / 1.29 | 185.2 / 172.7 |
| S155 ↔ X7 | 1.50 / 1.50 | 160.3 / 158.5 |
| S180 ↔ X6 | 1.74 / 1.77 | 141.1 / 143.1 |
| S220 ↔ X5 | 2.13 / 2.15 | 118.3 / 127.4 |

- **What the model predicts:** its anisotropy comes from the aperture boundary, and a cross's arms add boundary in both directions, so the model treats a stadium as the stronger source at equal aspect. Its stadium curve peaks one pair earlier than its cross curve at every native strength (×1: S130 vs X7; ×2: S155 vs X6; ×3: S180 vs X5, where both curves are flat).
- **What a different result would mean:** if the measured curves peak at the same pair, elongation of the area is all that counts and the boundary picture is wrong.
- Either result is a design rule.

**Size at fixed shape.** X7S has the aspect of X7 (1.49 vs 1.50) with 0.58 × its area. It tests whether the ruler reading carries over to smaller emitters (and to array elements).

**Controls.** CIRC and SYM have no elongation. They measure the native statistics: the azimuth distribution, the native mode splitting, and whether anything switches without help.

## Layout

The 42 slots are assigned by an exact integer program (`slot_plan` in the generator; scipy/HiGHS). It is deterministic, so a rebuild reproduces the layout.
- The copies of every design sit in different rows and different columns, and no two copies are diagonal neighbours.
- **Centring:** every cross, control and small-cross design has its mean position exactly at the centre of the device grid, (1108.6, 975.5) µm. Note that this is the grid's centre, not the 2 mm group's centre.
- **Stadiums:** the 3-copy stadiums cannot be centred exactly in x. Each sits 1/6 column (48 µm) off, with signs +, −, −, + along S130 → S220, so the offsets have no linear trend along the series.
- **Consequence:** a linear gradient of oxidation depth or exposure across the group cannot line up with the ratio or aspect series.

Over the 36 groups of the field this gives:
- 144 devices per cross,
- 108 per stadium,
- 72 per control and for the small cross.

## If the real oxidation depth is not 5 µm (`depth_tolerance.json`, geometry only)

The drawn mesas were eroded by real depths from 4.0 to 6.5 µm:
- **Apertures:** every device keeps exactly one aperture over the whole range. Nothing pinches off.
- **Metal clearance:** the contact ring stays D − 3.5 µm clear of the aperture: 0.5 µm at 4.0 µm and 1.5 µm at 5 µm (measured 1.49–1.50). Below about 3.5 µm the ring reaches the aperture edge, before any overlay error.
- **Area:** around 5 µm, ±0.5 µm changes the area by 13–22 %. Across 4.0–6.5 µm, each 0.5 µm step changes it by 11–27 %.
- **Aspect:** grows with depth. ±0.5 µm moves it by under a quarter of the spacing between neighbouring designs for X9–X6, S130 and S155, by about a third for X5 and S180, and by about half for X4 and S220.

| Aspect at depth (µm) | X9 | X8 | X7 | X6 | X5 | X4 | S130 | S155 | S180 | S220 | X7S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.5 | 1.12 | 1.28 | 1.47 | 1.72 | 2.04 | 2.42 | 1.25 | 1.46 | 1.67 | 1.99 | 1.45 |
| 5.0 | 1.13 | 1.29 | 1.50 | 1.77 | 2.15 | 2.63 | 1.27 | 1.50 | 1.74 | 2.13 | 1.49 |
| 5.5 | 1.13 | 1.30 | 1.52 | 1.84 | 2.28 | 2.90 | 1.29 | 1.55 | 1.83 | 2.30 | 1.53 |

**Consequence for the analysis:** plot the results against the *measured* aperture (infrared image of the oxide front), not the drawn ratio. The drawn-ratio reading is only as good as the depth control. The measured-aperture reading is not, and it is the one that transfers to other masks.

## Template (measured from the example) and its 5 µm adaptation

See `example_decoded/template_measurements.json`. With a 5 µm rim, the in-mesa dimensions are scaled down (**process decision, please confirm**):

| Feature | Example (10 µm rim) | This mask (5 µm rim) |
|---|---|---|
| first-p-metal ring | 6 µm wide | 0.75–3.5 µm inside the mesa edge (2.75 µm wide) |
| Ring clearance to the aperture | 1.6–3.0 µm | 1.5 µm |
| Ring opening towards the pads | ±10° wedge, 4.6–6.7 µm wide at the ring | 4.6 µm parallel slot |
| Passivation opening | about 4 µm inside the mesa edge | 2 µm inside |
| Contact strip | 3.0–4.5 µm | 1.5 µm |

Everything outside the mesa (p-pad ring and lead, passivation ring and field, pads, n-Metal "D") is the example's geometry, with corners rounded as above. The n-Metal polygons are reused from `PM_circle_R_20`.

## Checks (`code/check_mask2.py`, KLayout, independent of the generator)

**Results**
- **Structure:** one flat cell, layers 1–7 / datatype 0 only, everything inside the 2 × 2 mm frame.
- **Records and vertices:** the largest GDS record is 4.4 kB. The largest polygon has 552 vertices (GDSII allows 8190).
- **Data hygiene:**
  - No polygon intersects itself, and no edge is shorter than 5 nm.
  - The only direction reversals are the six zero-width cut lines with which GDSII represents holes, all in label glyphs (e.g. the loop of a "9").
- **Widths and gaps** (device area), checked two ways:
  - **Morphological:** opening and closing at 2 µm leave no residue on any layer.
  - **Edge-based:** KLayout width and space checks with projection metrics find nothing below 2 µm, and no features touch only at a corner.
  - A Euclidian edge check also flags the rounded convex corners (r = 1.05 µm) at 1.35–1.42 µm. That is the metric measuring around a polygonised arc, not a narrow feature.
- **Apertures:** exactly one predicted aperture per device (mesa eroded by a 5 µm disk), and no parasitic islands. KLayout's areas agree with the generator's within 0.1 µm².
- **Contact ring:**
  - It lies on the mesa, 1.49–1.50 µm from the aperture.
  - Its opening is 4.60 µm wide.
  - Every ring has a contact opening touched by the p-pad.
- **Aperture coverage:** no p-pad or passivation sits over an aperture.
- **n-Metal vs p-MetalPad:** at least 10 µm apart in the device area (in fact at least 24 µm). In the lab's alignment strip the two metals overlap by design.

`code/depth_tolerance.py` reads the written GDS (not the generator's shapes) and gives the depth table above.

## Still to confirm before the mask shop

1. **Mask x ∥ [011]:** check the wafer flat specification and use the same alignment as mask 1.
2. **Compensation direction:** the choice of 90° comes from the model's anchoring on the 840-nm GaAs-well devices. Mask 1's angle series on the same wafer tests it. If mask 1 switches at 0° rather than 90°, mask 2 will pin, and mask 3 should turn the ruler by 90°.
3. **Ring and passivation dimensions:** confirm them at the 5 µm rim (including the 4.6 µm ring opening), given your aligner's overlay tolerance.
4. **N Well:** what does it do in your process? It is copied from the example.
5. **Labels:**
   - Design IDs are drawn in n-Metal (2 µm pixels), 7.5 µm above each device's n pad. The lab's own strip text is on p-MetalPad.
   - `LABEL_LAYER` moves the labels to another layer and `LABEL_MIRROR` sets their handedness. Delete them if the lab doesn't label in metal.
6. **Mirrored text:**
   - The lab's strip text (KAUST, the numbers) is drawn mirrored (x → −x), so the device labels are mirrored the same way. Both then read correctly if your mask-making chain mirrors the layout.
   - If it does mirror, every angle θ in the CAD becomes 180° − θ on the wafer. That changes nothing on mask 2, where every design is symmetric about both axes, but read mask 1's angle series accordingly.
7. **Wafer cross-section:** the wafer comparison uses the lab's published 940-nm wafer. Please confirm it with your own cross-section.
8. **Vertex limit:** if the mask writer needs ≤ 200 vertices per polygon, split on export (KLayout: Save As → GDS2 options; or L-Edit's export settings). Not done here, because splitting in gdstk leaves nanometre slivers along the cuts.

## Files

- `VCSEL_PM_Group_2.gds`
- `VCSEL_PM_Group_2_devices.csv`: one row per slot, with:
  - position, design, target and mesa sizes;
  - the predicted aperture area and aspect;
  - the model's switching share and start-axis share at ×1, ×2 and ×3, and its mode splitting.
- `MEASUREMENT_AND_PAPER_PLAN.md`: the read-outs, what to measure, what you can conclude, the paper.
- `check_report.json`, `build_report.json`, `depth_tolerance.json`
- `previews/`: the group and four device close-ups (green: predicted aperture). `preview_with_apertures.gds` is the group with the predicted apertures on layer 9, for viewing only; do not send it to the mask shop.
- `example_decoded/`: V04 as GDSII (rings fixed), renders, template measurements.
- `code/`: the TDB reader and converter, the generator (all dimensions are named parameters at the top), the checks and the depth tolerance.
  - `python3 code/make_mask2.py VCSEL_ARRAY_Heming_-_V04.tdb out/` rebuilds the GDS exactly (XOR with the delivered file: 0 µm² on every layer).
- `model/`: the model numbers in the tables and the scripts that produced them. The scripts need the `vcsel-polarization-bo` repository.
