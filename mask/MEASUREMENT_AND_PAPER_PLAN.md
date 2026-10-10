# Measurement and paper plan for mask 2 (the anisotropy ruler)

## The gap

**Shape has long been used to pin VCSEL polarization.** Elongated, eye-shaped and asymmetric oxide apertures do it (Yoshikawa et al. 1998; J. Zhang et al. 2018, 2021).

**The NUS group (Yuan, Danner) turned shape into compensation.** Their cruciform apertures cancel the wafer's native anisotropy instead of adding to it:
- the cruciform itself (APL 126, 2025);
- large-scale statistics (JLT 43, 2025);
- arrays for Ising computing (APL 127, 2025);
- mesa orientation (IPC 2025);
- GHz-rate binary updates by polarization switching (Laser Photonics Rev. 2026).

All of this is on ≈ 840-nm GaAs-well wafers, with one cruciform ratio, 1:0.7, which the authors call "intuitively chosen".

**Their newest preprint makes the goal explicit** (Chen, Yuan, …, Danner, Chen, arXiv:2609.21331, Sept 2026). An injection-locked Ising machine needs *low* anisotropy:
- **Locking:** in their spin-flip simulations an isotropic VCSEL locks to an injected field 28 times weaker than an anisotropic one needs.
- **Accuracy:** a 3-spin machine's accuracy falls from 62 % to 30 % as the anisotropy grows.
- **Their proposal:** a metasurface etched into the top GaAs layer to cancel the mirror's polarization-dependent reflectivity.
- **Its status:** it has been fabricated only on bare GaAs so far. Its dichroism and birefringence (0.67 and 192 ns⁻¹) are taken from 1550-nm VCSELs, not measured.
- **How they see aperture shaping:** as a way to *lock* polarization, with finer control still needed.

**What nobody has measured:**
1. **How much aperture elongation a given wafer needs** to bring its dichroism (the gain difference between the two polarizations) through zero. The published ratio is one point, not a calibration.
2. **What that costs in birefringence** (the frequency splitting of the two polarization modes). The model predicts that the elongation that cancels the dichroism also raises the splitting, and the metasurface preprint shows that a larger birefringence also makes injection locking harder.
3. **What does the work: the aperture's edge or its overall elongation.** The edge carries the oxide stress and the cross's arms; the elongation shapes the mode.

Mask 2 answers all three with no new process step. That is the "finer control" the metasurface preprint asks for, available on a standard VCSEL process today.

**Our wafer is different in the ways that matter:**
- **Active region:** 940-nm wells, which must be strained InGaAs.
- **Top mirror:** 20 pairs instead of 28.
- **Native lock:** in the lab's own measurements every device locks to [011], whatever its shape.

## The wafers

| | NUS (cruciform papers) | Lab 940-nm wafer (Lu et al., Light Sci. Appl. 14, 344, 2025) |
|---|---|---|
| Emission | ≈ 838 nm | ≈ 940 nm |
| Active region | GaAs quantum wells | 5 wells, λ/2 cavity (InGaAs, inferred) |
| Bottom DBR | 40 pairs, Si-doped | 38 pairs, n-AlGaAs |
| Top DBR | 28 pairs, C-doped | 20 pairs, p-AlGaAs, 20 nm p⁺⁺ GaAs cap |
| Oxide layer | "high-Al" layer in the top DBR (no thickness or position given) | Al₀.₉₈ in the top DBR (no thickness or position given) |
| Substrate | GaAs (100) | n-GaAs (100) |
| Mesa / oxidation | ICP; 450 °C at 0.1 µm/min; uniform in all directions | ICP-RIE ≈ 3.85 µm; 25 µm oxidation in that paper; 5 µm planned here |
| Native polarization | squares lase along the primary flat | every device along [011] |

Not available to me: the NUS JLT (large statistics) and APL 127 (arrays) full texts, which are paywalled. Please check the right-hand column against your own wafer's cross-section.

## Three read-outs (and an optional fourth)

1. **Start axis.** Does the device start lasing, just above threshold, along [011] or along [01-1]?
   - **Cost:** one polarizer measurement at one current.
   - **Works without switching:** the start axis flips where the net dichroism changes sign, so the design at which half the devices start along [01-1] is the **balance point**: the design with the smallest dichroism.
2. **Switching share.** Does the device switch once, between about 1.25 and 1.95 × I_th, between two linear states at least 60° apart?
   - This is the model's definition, and what a current-driven Ising node needs.
   - In the model it peaks at the first design past the balance point, where most devices already start along [01-1].
3. **Anisotropy constants, from spectra just below threshold.** These are continuous values for every device, so they give the strongest statistics:
   - **Signed dichroism:** from the intensity ratio of the two polarization modes.
   - **Birefringence:** from their frequency splitting.
   - Together they are the γa and γp of the spin-flip model, which the metasurface preprint had to borrow from 1550-nm devices.
4. **Optional, injection-locking threshold.** The weakest frequency-matched injection from a tunable 940-nm master laser that flips the polarization. Measure it on a few devices of the controls, the balance design and its neighbours. It is the Ising figure of merit of the metasurface preprint, measured instead of simulated.

**Model predictions** (not measurements). The native anisotropy is ×1, ×2 or ×3 that of the 840-nm devices; "aspect" is the aperture's second-moment aspect ratio.

| Cross series (aspect) | X9 (1.13) | X8 (1.29) | X7 (1.50) | X6 (1.77) | X5 (2.15) | X4 (2.63) |
|---|---|---|---|---|---|---|
| Start along [01-1], ×1 / ×2 / ×3 | 0.01 / 0 / 0 | 0.26 / 0 / 0 | 0.96 / 0.26 / 0.03 | 1 / 0.74 / 0.30 | 1 / 0.97 / 0.56 | 1 / 1 / 0.74 |
| Switching share, ×1 / ×2 / ×3 | 0 / 0 / 0 | 0.10 / 0 / 0 | **0.58** / 0.16 / 0.02 | 0.17 / **0.39** / 0.20 | 0.03 / 0.30 / **0.29** | 0.01 / 0.11 / 0.27 |
| Median net dichroism, ns⁻¹, ×1 / ×2 / ×3 | 1.2 / 3.0 / 4.9 | **0.5** / 2.0 / 3.9 | 1.6 / **1.0** / 2.4 | 3.6 / 1.6 / **1.4** | 4.9 / 2.7 / 1.6 | 6.0 / 3.9 / 2.2 |
| Mode splitting, GHz (any strength) | 9.9 | 11.9 | 14.6 | 17.3 | 20.6 | 22.5 |

| Stadium series (aspect) | S130 (1.27) | S155 (1.50) | S180 (1.74) | S220 (2.13) |
|---|---|---|---|---|
| Start along [01-1], ×1 / ×2 / ×3 | 0.83 / 0.14 / 0.01 | 1 / 0.65 / 0.22 | 1 / 0.94 / 0.51 | 1 / 1 / 0.70 |
| Switching share, ×1 / ×2 / ×3 | **0.58** / 0.08 / 0 | 0.23 / **0.41** / 0.14 | 0.04 / 0.32 / **0.28** | 0.01 / 0.16 / 0.27 |
| Median net dichroism, ns⁻¹, ×1 / ×2 / ×3 | **1.1** / **1.1** / 2.7 | 3.1 / 1.3 / **1.5** | 4.8 / 2.6 / 1.6 | 5.8 / 3.6 / 2.0 |
| Mode splitting, GHz (any strength) | 13.9 | 17.0 | 19.7 | 22.0 |

**Controls and the small cross:**
- **Controls (CIRC, SYM):** they start along [011], never switch, keep the native dichroism (1.9 / 3.8 / 5.7 ns⁻¹) and split by 8.7 GHz.
- **Small cross (X7S):** starts along [01-1] with a share of 0.99 / 0.39 / 0.07, switches with 0.45 / 0.27 / 0.04, and splits by 15.3 GHz.

**What the model is saying:**
- **Balance point:** the start axis reaches ½ (interpolated) at aspect ≈ 1.36 / 1.64 / 2.06 for crosses and < 1.27 / 1.43 / 1.73 for stadiums. The stadium always balances earlier; that is the matched-pair prediction.
- **Birefringence:** it does not cancel. The elongation that brings the dichroism to zero raises the splitting from 9 GHz (circle) to about 13, 16 or 20 GHz at ×1, ×2 or ×3.

## What to measure, per device

1. **Polarization-resolved L-I curves**, current swept up and down. Record:
   - the start axis and azimuth just above threshold;
   - whether the device switches, and at what current I_sw;
   - the hysteresis width;
   - whether both states are linear.

   Keep the raw data, so that both "any switch" and the model's switching definition can be applied afterwards. In the model, 0.79 of X7 devices switch somewhere at ×1, but only 0.58 meet the full definition.
2. **Polarization-resolved spectra just below threshold**, for every measured device:
   - **Splitting:** fit the peak of each polarization separately; the difference is the splitting.
   - **Signed dichroism:** take the intensity ratio of the two modes at a fixed fraction of I_th.
   - **Resolution:** at 940 nm, 10 GHz is 0.029 nm. Use an OSA with ≤ 0.02 nm resolution and peak fitting, or a scanning Fabry–Pérot.
3. **The same at a second heat-sink temperature** (e.g. 20 and 40 °C) for a subset. Spin-flip switching and thermally driven switching (gain–cavity detuning) shift differently with temperature.
4. **Near-field image:** single- or multi-mode, and whether the mode follows the aperture.
5. **Infrared image of the oxide front** for a few devices per design and group: the real aperture area and aspect, and the real oxidation depth.
   - **This is the x-axis of the analysis.** A ±0.5 µm depth error moves the aspect of X5 or X4 by a third to a half of a ruler step (`depth_tolerance.json`).
   - Plotted against the measured aspect, the result no longer depends on the depth control.
6. **Optional, injection locking** on a few devices per design around the balance point and on the controls (read-out 4).
7. **Bookkeeping:** group position on the wafer, slot ID, and the angle between the mask x-axis and the wafer flat.

## How many, in two passes

- **Pass 1 (screen), about 80 devices:** 6 devices of each design, taken from groups spread over the wafer. This gives the start-axis curve, the spectra and a rough position of the switching peak.
- **Pass 2 (resolve), about 200–260 devices:**
  - 30–40 devices of the three designs around the peak in each series;
  - 10 of each control.
- **Model update in between:** update the model with pass 1 and let it choose pass 2. This is the first loop of the Bayesian-optimization workflow, on real data.

**Why these numbers.** Statistical power of Fisher's exact test at α = 0.05, with the model's predicted shares:

| Comparison | n = 20 | n = 30 | n = 40 |
|---|---|---|---|
| Switching, X7 vs X6 at ×1 (0.58 vs 0.17) | 0.70 | 0.88 | 0.96 |
| Start axis, X7 vs S155 at ×2 (0.26 vs 0.65) | 0.57 | 0.83 | 0.93 |
| Switching, X7 vs S155 at ×1 (0.58 vs 0.23) | 0.52 | 0.73 | 0.86 |

- **The binary shares need these counts.** The continuous read-out 3 locates the balance point with far fewer devices.
- **Supply is not the limit:** the field holds 144 devices of each cross and 108 of each stadium.
- **Time:** about 300 devices is a few days on an automated probe station.

## What you can conclude

| What you see | What it means | Next step |
|---|---|---|
| The start axis crosses ½ inside the ruler, and switching peaks just past it | The balance point and the best shapes for this wafer (below). The model converts the balance point into the wafer's native dichroism. | Mask 3: finer steps around the balance point; array elements with that shape |
| The start axis crosses ½, but (almost) nothing switches | The dichroism is balanced, but in the spin-flip picture the birefringence has the sign that forbids switching. Fine for injection locking, not for current-driven switching. | Change the birefringence: size (compare X7S), stress, or a birefringence-engineering approach |
| Every device stays on [011], even X4 and S220 | The native pull is stronger than ×3; you get a **lower bound** on the native anisotropy | Stronger elongation (ratio < 0.4), or another lever (mesa orientation, surface grating, metasurface) |
| Every device starts on [01-1], even X9 | The native pull is weaker than in the 840-nm devices; you get an **upper bound** | Weaker elongation (ratio 0.9–1.0) |
| The controls switch | Switching happens without help on this wafer; check whether it is thermal | Re-read the ruler as a shift of an existing switching population |

**The best shape depends on how the Ising machine drives its spins, and the ruler gives both:**
- **Current-driven switching** (the route of the NUS arrays and GHz updates): the design at the switching-share peak, with I_sw in the window.
- **Injection locking** (the metasurface preprint's architecture): the design closest to zero dichroism, i.e. at the start-axis crossover. The exception is if the birefringence it adds costs more injection power than the dichroism it removes; read-out 4 settles that.

**More results come from the same data, whatever the outcome above:**
- **The anisotropy map.** Measured dichroism and birefringence versus measured aperture shape, for a 940-nm wafer. This is the input the Ising simulations of the metasurface preprint had to borrow from 1550-nm devices. It also shows directly whether shape compensation trades dichroism for birefringence, as the model predicts.
- **Edge or area (the matched pairs).** If the stadium curve balances and peaks about one pair earlier than the cross curve (S130 vs X8, S155 vs X7, …), the aperture edge carries the anisotropy, as the model assumes. If the curves line up pair by pair, the overall elongation is all that counts. Then the model's boundary descriptor must change, and arrays can use simple stadiums or ellipses.
- **Uniformity.** The steepness of the start-axis curve, and the spread of the measured dichroism at one design, measure the device-to-device scatter of the anisotropy. That is the second requirement the metasurface preprint sets for Ising nodes ("all VCSELs behave the same"). It tells you how precisely array elements must be made.
- **Size (X7S vs X7).** These have the same aspect at 0.58 × the area. If they behave alike, the ruler reading carries over to small array elements.

**Mechanism and alignment, from the extra measurements:**
- The temperature dependence of I_sw and of the splitting shows whether the switching is spin-flip or thermal, and how much of the birefringence the aperture sets.
- Mask 1's rotation data on the same wafer gives the angle window, i.e. how precisely array elements must be aligned to the crystal.
- Mask 1 also tests the compensation direction: whether the long axis must be ⊥ [011], as assumed.

## The paper

**Working title:** *Tuning VCSEL anisotropy through zero with the oxide-aperture shape: an anisotropy ruler for Ising-machine wafers at 940 nm.*

**The story:**
- **The problem.** Every VCSEL wafer is born with a preferred polarization, set by its crystal and its strain. That preference is exactly what a photonic Ising machine cannot afford: the latest preprint from the group that pioneered cruciform apertures shows that an isotropic VCSEL locks to an injected field 28 times weaker, and proposes a metasurface to get there. That metasurface still has to be integrated with a laser.
- **The alternative.** The aperture shape can do the same job today, on a standard process, with nothing more than a drawing on the mask. What is missing is how much shape a given wafer needs: the one published ratio was chosen by intuition.
- **The measurement.** This paper turns the intuition into a measurement. One exposure of a graded ruler reads the wafer's native anisotropy off in units of aperture elongation. It maps both spin-flip constants against the measured aperture shape. Matched crosses and stadiums show whether the aperture's edge or its overall shape does the work.
- **The result.** A one-mask calibration recipe that any lab can apply to a new wafer, plus the best shapes for this 940-nm wafer, one for current-driven and one for injection-locked Ising nodes.

**Where to publish**
- **Full article:** Optics Express or J. Lightwave Technol.
- **Letter:** if the matched-pair or the trade-off result is crisp, Optics Letters, IEEE PTL, or APL (where the NUS work appeared).

**Figures**
1. **Concept and process.** The wafer and process, and drawn versus real apertures (IR images): 5 µm oxidation turns a 17.5 µm cross into an elongated polygon. The matched-pair design.
2. **Native statistics,** from the circles and symmetric crosses: azimuth histogram, splitting, dichroism scatter.
3. **The ruler.** Start-axis share and switching share against the **measured** aperture aspect, for crosses and stadiums, with confidence intervals and wafer-position maps.
4. **The anisotropy map.** Signed dichroism and birefringence against the measured aspect: zero crossing, the trade-off, and their temperature dependence. The angle window from mask 1.
5. **Ising relevance and the model.**
   - The injection-locking threshold across the balance point (if measured).
   - The model calibrated on these data: from a literature prior to a wafer-specific posterior. It predicts the best shape for each architecture with an uncertainty and proposes mask 3.

Figure 5 is the machine-learning contribution behind the MLSS poster. A follow-up paper can close the loop with a model-designed mask 3, from a two-pass measurement to a model-chosen design.

## Sources

- W. Chen, Z. Yuan, H.-L. Lin, L. Qi, J. Chu, A. Danner, Y. Chen, "Metasurface-integrated VCSEL designed for polarization control in optical Ising machines," arXiv:2609.21331 (2026).
- Z. Yuan, D. Zhang, L. Shi, Y. Liu, A. Danner, "Enhanced polarization locking in VCSELs," Appl. Phys. Lett. 126, 151101 (2025).
- Z. Yuan et al., "Large-scale fabrication and analysis of polarization behavior in VCSELs with tailored apertures," J. Lightwave Technol. 43, 6819 (2025). Abstract only.
- Z. Yuan et al., "Tailored polarization-switchable VCSEL arrays for photonic Ising computing," Appl. Phys. Lett. 127, 221102 (2025). Abstract only.
- Z. Yuan, W. Shan, T. Chen, B. Lin, A. Danner, "Mesa orientation engineering for polarization locking in VCSELs," IEEE Photonics Conference (2025).
- Z. Yuan et al., "Drive-efficient GHz-rate binary-state updates enabled by VCSEL polarization switching," Laser Photonics Rev., e71890 (2026). As cited by Chen et al.; not read.
- T. Yoshikawa et al., IEEE J. Quantum Electron. 34, 1009 (1998); J. Zhang et al., Jpn. J. Appl. Phys. 57, 120309 (2018); J. Zhang et al., Opt. Laser Technol. 139, 106948 (2021). Aperture-shape polarization control, as cited by Chen et al.; not read.
- Lu et al., Light Sci. Appl. 14, 344 (2025) (the lab's 940-nm wafer).
