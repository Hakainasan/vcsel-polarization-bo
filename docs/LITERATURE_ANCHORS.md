# What the fabricated-device papers say, and how the model uses it

Keys refer to `docs/references.bib` (statuses there: READ-FULL / READ-ABSTRACT / LISTING). Quoted fragments below were
extracted from the arXiv HTML text of each paper by the fetch tool in a second, dedicated pass and are given as found;
they were not compared character by character against the PDFs; check any quotation you reuse against the PDF. Two phrases quoted in the first version of this project
("seldom switch polarization", "maintained stable x-polarization") were **not** found in that second pass and were removed. An independent reference check later
located "seldom switch polarization" in the Fig. 2(a) caption of the CLEO paper (I have not re-verified this), so that quotation was probably right; the anchors use the wording that was found in the second pass.
The numbers attributed to the metasurface paper below (0.0025, 0.07, 28 times, 62 % to 30 %) come from the PDF you supplied and were not re-verified in the last pass.

## 1. Facts reported by the papers

| Source | What it reports (quoted fragment) | Numbers? |
|---|---|---|
| CLEO 2025 `yuan2025clea` (full text) | "For the circular VCSEL, the polarization remains x-polarized without switching as the current increases"; Fig. 2: "Most circular VCSELs lase x-polarized, while symmetrical cruciform VCSELs exhibit higher average polarization azimuths"; "cruciform VCSELs exhibit higher average polarization azimuths and a greater proportion of devices with polarization switching"; "Most unrotated VCSELs lase x-polarized, while 90°-rotated VCSELs lase y-polarized"; "Rotating the orientation angle of the cruciform leads to predictable shifts in polarization states" | none (no percentages, device counts or currents in the text) |
| APL 126 `yuan2025apl126` (full text, theory table not re-read) | Seven designs: "square, cruciform (1:0.7 aspect ratio), cruciform (1:0.7 aspect ratio) rotated by 30°, 60°, and 90°, and cruciform (1:0.6 aspect ratio) without and with 90° rotation"; cruciforms "can either strengthen (e.g., Fig. 2(b), with a ratio of 1:0.7 and a longer horizontal arm) or mitigate (e.g., Fig. 2(e), with a ratio of 1:0.7 rotated by 90°, where the vertical arm is longer) the built-in amplitude anisotropy"; "greater anisotropy leads to more stable polarization states that are harder to switch"; "VCSELs exhibiting polarization switching properties (Fig. 2(c)-(e)) have better OIL performance" | **yes**: minimum injection power at 10 mA (HC): 248 µW for the switchable device of Fig. 2(d) (1:0.7 cruciform rotated **60°**) vs 371 µW for the non-switchable square (Fig. 2(a)); at low current (LC, just above threshold) 3.6 µW for Fig. 2(d). Devices per design: not stated in the text read |
| Mesa orientation `yuan2025mesa` (full text) | "the initial polarization states (just above threshold) of the two VCSELs are orthogonal and aligned with the direction of the elongated mesa arm"; "the VCSEL with the 90° rotated mesa (elongated along the y-direction) exhibits polarization switching - from initial y to x-polarization"; "By elongating the aperture along the y-direction, this anisotropy is compensated"; x is "parallel to the flat edge of the wafer" | none |
| JLT 43, 6819 `yuan2025jlt` (**abstract only**) | "a method of tailoring oxide apertures to compensate for inherent gain anisotropy and yield the desired instability"; "hundreds of VCSELs with diverse oxide aperture shapes and sizes ... clear trends linking aperture designs to polarization tendencies" | the statistics are in the full text, which was **not** read |
| APL 127 `yuan2025apl127` (**listing only**) | polarization-switchable VCSEL arrays for photonic Ising computing | not read |
| Metasurface `chen2026meta` (full text, simulations only) | free-running VCSEL with gamma_a, gamma_p > 0 "predominantly emits in Y polarization", both < 0 gives X; injection ratio needed to flip polarization: 0.0025 (isotropic) vs 0.07 (gamma_a = ±0.67 ns^-1, gamma_p = ±192 ns^-1), "28 times higher"; 3-bit Ising accuracy over 100 random interaction matrices "decreased from 62% to 30% as the anisotropy increases" (gamma_a, gamma_p) = (0.67, 192), (0.67, 500), (6.7, 1920) ns^-1 | yes, but from simulations, not devices |

**Answer to "which aperture is best?" as far as the papers go.** The injection-locking numbers in the APL 126 text I read concern two devices:
the **1:0.7 cruciform rotated by 60°** (Fig. 2(d): 3.6 µW just above threshold, 248 µW at 10 mA) and the non-switchable square
(Fig. 2(a): 371 µW at 10 mA). I found no ranking of all seven designs by injection power, so the paper does not show that the 60° device is the best of them.
It groups Fig. 2(c)-(e) (30°, 60°, 90°) as switching devices with better injection-locking performance. The text I could read does not say
how many devices were measured per design, so it does not establish that 60° is intrinsically better than 30° or 90°. The metasurface paper adds, by simulation,
that lower net anisotropy makes injection flipping about 28 times easier, and that the accuracy of a 3-bit Ising test fell from 62 % to 30 % as the anisotropy increased.
The JLT paper (abstract only) reports statistics over hundreds of devices that would answer the question quantitatively; it was not read in full.

## 2. What is encoded (`src/vcsel_bo/literature.py`)

Each anchor is a *qualitative* outcome turned into a yes/no test on a simulated device population (24 devices x 3
oxidation-depth nodes) in one candidate "world" (native dichroism d0, native birefringence b0, aperture coupling k_d, k_b,
each with unknown sign). Worlds that fail an enforced anchor are rejected (approximate Bayesian computation with
qualitative summaries). The words of the papers are turned into numbers by thresholds that are **choices**:

| threshold | meaning | primary | loose | strict |
|---|---|---|---|---|
| most | "most devices lase x" | >= 0.60 | 0.50 | 0.75 |
| seldom | "does not switch" / non-switchable | <= 0.30 | 0.40 | 0.20 |
| part | "part of the devices switch" | >= 0.10 | 0.05 | 0.20 |
| more | "a greater proportion" | +0.05 | +0.02 | +0.10 |

| anchor | design(s) | enforced | source |
|---|---|---|---|
| A1 circle and square lase x-first and (nearly) never switch | circle, square | yes | `yuan2025clea`, `yuan2025apl126` |
| A2 1:0.7 cruciform with the long arm along x behaves like the circle | 1:0.7 at 0° | yes | `yuan2025apl126` |
| A3 initial polarization follows the long arm (x at 0°, y at 90°) | 1:0.7 at 0°, 90° | yes | `yuan2025clea`, `yuan2025mesa` |
| A5 the 90° cruciform switches y -> x (more than x -> y) | 1:0.7 at 90° | yes | `yuan2025mesa` |
| A4 rotated 1:0.7 cruciforms switch in part of the devices | 60°, 90° (30° reported) | yes for 60° and 90° | `yuan2025apl126` |
| A6 symmetric cross switches more than the circle | symmetric cross | no (reported) | `yuan2025clea` |

Mask sizes of the fabricated devices are not in the text read; the encoded designs use L = 17.5 µm, w = 6 µm,
corner fraction 0.5 (a typical point of the design box). The 1:0.6 designs of APL 126 (Fig. 2(f), (g)) are not encoded
because the text read makes no statement about their switching.

## 3. What is *not* encoded and why

* The injection-power numbers (248, 371, 3.6 µW) and Chen et al.'s flip ratios (0.0025 vs 0.07) need an optical-injection
  model and the unknown injection coupling; they are used only as the *direction* of the Ising figure of merit (switching
  devices with lower net anisotropy are better), not as calibration data.
* The device statistics behind "clear trends" in the JLT paper. They would turn the qualitative anchors into quantitative
  ones; the model has a calibration module (`calibration.py`) ready for them.
* Any measured dichroism, birefringence or switching current: none appears in the sources read.

## 4. Limits of this anchoring

* Anchors are necessary, not sufficient: many different worlds reproduce them, which is why the design step optimises over
  the accepted set instead of over one fitted world.
* The model is *constrained by* these outcomes, not *validated against* them: the outcomes are used to select the worlds, so they cannot also test the model.
  No comparison with held-out fabricated-device data has been made; the mask predictions (`results/mask_predictions.json`) are what can test it.
* Model structure: the aperture enters as a second-rank (C2) tensor. Devices rotated by 30° or 60° induce dichroism and
  birefringence vectors that are not parallel to the native ones, giving elliptical eigenstates; 30° switching is therefore
  reported, not enforced.
* Thresholds are choices; the sampling is repeated with loose and strict sets (`results/anchored_worlds_*.json`).
* The prior box is a choice too. With the default box (native dichroism from 1 ns^-1) the accepted d0 piles up against the lower edge;
  `anchored_worlds_wide.json` repeats the sampling with d0 from 0.2 ns^-1 and |k_d| from 0.6 ns^-1. In the model the lower bound of d0 comes
  from the device-to-device scatter of the native dichroism (0.25 ns^-1), not from the papers.
* The spin-flip constants of the metasurface paper (gamma_s = 1000 ns^-1 instead of 50) accept **no** world in 4000 draws
  (`anchored_worlds_chen.json`): initial polarization (A1-A3) is reproduced, but the 60- and 90-degree crosses never switch. The robustness
  check with those constants is therefore *inconclusive*, not confirmatory; why the model needs different magnitudes there was not pursued.
* Re-conditioning under another process assumption (`anchored_worlds_ox10.json`, `_ox13.json`, `_chim02.json`, `_chip02.json`) shows how much
  of a conclusion is tied to the placeholder oxidation anisotropy (1.15) and to the absence of thermal dichroism drift.
