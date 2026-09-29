# ML-Guided Discovery of Polarization-Switching VCSEL Apertures

This repository contains the computational prototype for a machine-learning-guided search for oxide-aperture geometries that support current-driven polarization switching in VCSELs, motivated by polarization-encoded photonic Ising computing.

## Core idea

The workflow is:

**continuous aperture geometry -> reduced VCSEL polarization model -> switching utility -> GPR (Matérn 5/2 ARD) -> Expected Improvement -> next geometry**

Machine learning is used as the search engine. The Gaussian-process surrogate predicts both the expected switching utility and its uncertainty, and Expected Improvement uses both quantities to decide which expensive physics evaluation should be run next.

## Contents

- `notebooks/VCSEL_Polarization_BO_Project.ipynb` - Colab/Jupyter implementation.
- `supplementary/VCSEL_Polarization_BO_Supplementary.tex` - LaTeX supplementary methodology.
- `figures/vcsel_bo_workflow.png` - workflow figure used by the supplementary document.

## Scope

The current physics backend is a transparent reduced-order Spin-Flip Model (SFM) prototype with a phenomenological aperture-to-anisotropy mapping. It is not presented as a validated full-wave model of a fabricated 940-nm VCSEL. The ML layer is deliberately modular so geometry-specific data from a validated full-wave solver or experiment can later replace the reduced-order backend.

## Reproducibility

Run the notebook in Google Colab. The notebook generates a continuous Fourier-parameterized candidate pool, evaluates selected geometries with the reduced-order simulator, fits a GPR surrogate, selects candidates with Expected Improvement, and compares the discovery trajectory with random search.

## Citation

The supplementary methodology contains the current references and should be cited together with this repository when the work is shared.
