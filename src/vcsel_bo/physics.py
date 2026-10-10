"""From aperture geometry to VCSEL polarization anisotropy (the *uncalibrated* hypothesis layer).

What is literature-grounded and what is assumed (keys refer to docs/references.bib)
-----------------------------------------------------------------------------------
Grounded:
  * The lasing polarization of a VCSEL is set by the linear anisotropies of the cavity: a dichroism
    (gain/loss difference) and a birefringence (frequency splitting) between two orthogonal linear
    modes, which enter the SFM as (gamma_a, gamma_p)  [sanmiguel1995; martinregalado1997].
  * VCSELs carry a *native* anisotropy; tailored oxide apertures are used to *compensate* it, which
    yields devices with controllable, easily switchable polarization: the JLT abstract states "a method of
    tailoring oxide apertures to compensate for inherent gain anisotropy" [yuan2025jlt]; cruciform apertures
    strengthen or mitigate the built-in amplitude anisotropy depending on the orientation of the long arm
    [yuan2025apl126]; the initial polarization follows the elongated arm [yuan2025mesa, yuan2025clea].
  * Polarization switching in a VCSEL happens when the gain sweeps through the polarization resonances, and the
    splitting between them is strongly influenced by local strain [choquette1995]. Observed splittings range from
    below 1.3 GHz to 9.9 GHz in that paper (gamma_p = pi * splitting -> about 4 to 31 ns^-1); the metasurface paper
    uses |gamma_p| = 192 ns^-1 and |gamma_a| = 0.67 ns^-1 as an "anisotropic" example [chen2026meta].
  * The oxide aperture follows the mesa shape [yuan2025mesa], and lateral oxidation of AlGaAs is anisotropic in the
    plane, hence the eroded-mesa aperture model in `geometry` (an assumption of this work, with a placeholder rate ratio).

NOT grounded (assumptions, listed so that nobody mistakes them for results):
  * that the *birefringence* induced by the aperture is a stress effect proportional to a second-rank boundary tensor
    (`s_stress`); the strain sensitivity of the splitting is documented [choquette1995], the aperture-stress tensor is not;
  * the coupling coefficients k_d, k_b and the native constants d0, b0: they are constrained only qualitatively by the
    published outcomes (`literature.py`), no measured values exist in the sources read.

Assumed (phenomenological, to be calibrated -> `calibration.py`):
  * The dichroism/birefringence induced by an aperture is linear in a *second-rank* (C2) boundary
    tensor of the aperture: dichroism follows the mode-weighted tensor `s_mode` (optical overlap channel),
    birefringence the stress-weighted tensor `s_stress` (long-range oxide-stress channel).
  * Second-rank means: a C4-symmetric aperture (symmetric cross, circle) induces *no* anisotropy by
    itself. Non-zero anisotropy of nominally symmetric crosses comes only from anisotropic oxidation
    (which our process model captures) and from native scatter.

Vectors: an anisotropy of magnitude m and principal axis psi is the spin-2 vector m (cos 2psi, sin 2psi).
The SFM matrix is  M = (d1 + i b1) sigma_z + (d2 + i b2) sigma_x  with d = dichroism vector, b = birefringence
vector, both in the crystal frame.

AXIS CONVENTION (matches the experimental papers): x = the natively favoured polarization axis (the "horizontal"
axis of Yuan et al., along the wafer flat). Native dichroism is therefore d_nat = -d0 (cos 2psi, sin 2psi) with
d0 > 0 (d1 < 0 means the x mode has the LOWER loss). The mask rotation theta is the angle of the long arm from x.
The shape vector s points along the elongation direction, so k_d < 0 means "elongation favours the long axis",
which is what the fabricated devices show [yuan2025apl126]: a long arm along the native (horizontal) axis strengthens
the built-in anisotropy, a long arm along the orthogonal axis mitigates it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import geometry as G
from .sfm import SFMConstants


# ----------------------------------------------------------------------------------------------
# World (epistemic constants) and device scatter (aleatoric)
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """Unknown-but-fixed constants of a wafer + process (the things a measured batch calibrates).

    The default values are ONE example world that reproduces every enforced qualitative observation of the published
    tailored-aperture papers (`literature.anchor_report`): the accepted world closest to the median of the literature-
    consistent set in log-magnitude, with the most frequent sign pattern (b0 > 0, k_b < 0). It is an illustration, not a
    fitted value; many other worlds are equally consistent with the literature (`results/anchored_worlds.json`).
    """

    d0: float = 2.5            # native dichroism magnitude (ns^-1); the native mode is x (lower loss)
    b0: float = 27.0           # native birefringence along x (ns^-1), signed; b0 > 0 = class opposite to d_nat: the native device cannot switch
    k_d: float = -21.0         # dichroism per unit s_mode (ns^-1); < 0: elongation favours the long axis
    k_b: float = -57.0         # birefringence per unit s_stress (ns^-1), signed
    chi_th: float = 0.0        # thermal dichroism drift coupling (0 = static anisotropy, SFM-only switching)
    mu0: float = 1.6           # current where cavity-gain detuning vanishes (used only if chi_th != 0)
    process: G.Process = field(default_factory=lambda: G.Process(depth=1.0, ox_ratio=1.15))
    dn_eff: float = 0.01       # effective index step of the oxide aperture
    # aleatoric device-to-device scatter
    sig_d0: float = 0.25       # ns^-1
    sig_b0: float = 2.5        # ns^-1
    sig_psi: float = 3.0       # deg, native-axis misorientation
    sig_depth: float = 0.12    # um, oxidation-depth scatter (1 sigma)


@dataclass(frozen=True)
class Prior:
    """Broad prior over the epistemic constants. Magnitudes are log-uniform over wide ranges (an order of magnitude or
    more) around the values found in the sources read: |gamma_p| of 4-31 ns^-1 from splittings of 1.3-9.9 GHz
    [choquette1995] and 192 ns^-1 as an "anisotropic" example [chen2026meta]; |gamma_a| of 0.67 ns^-1 [chen2026meta].
    The ranges are choices, not measurements. Signs are unknown a priori: the literature anchors in
    `literature.py` decide them."""

    d0: tuple = (1.0, 12.0)
    b0_mag: tuple = (5.0, 100.0)
    k_d_mag: tuple = (3.0, 100.0)
    k_b_mag: tuple = (5.0, 200.0)
    log_uniform: bool = True
    ox_ratio: tuple | None = None   # process (depth, rate ratio) is *measured*, not inferred: keep None to hold it fixed
    depth: tuple | None = None
    p_pos_b0: float = 0.5      # probability b0 > 0 (opposite class to native dichroism -> no switching)
    p_pos_kd: float = 0.5      # probability that elongation *disfavours* the long axis (k_d > 0)
    p_pos_kb: float = 0.5


def sample_world(rng, prior: Prior = Prior(), base: World | None = None) -> World:
    base = base or World()

    def u(lohi):
        lo, hi = lohi
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if prior.log_uniform else float(rng.uniform(lo, hi))

    sgn = lambda p: 1.0 if rng.random() < p else -1.0
    return replace(
        base,
        d0=u(prior.d0),
        b0=sgn(prior.p_pos_b0) * u(prior.b0_mag),
        k_d=sgn(prior.p_pos_kd) * u(prior.k_d_mag),
        k_b=sgn(prior.p_pos_kb) * u(prior.k_b_mag),
        process=G.Process(depth=rng.uniform(*prior.depth) if prior.depth else base.process.depth,
                          ox_ratio=rng.uniform(*prior.ox_ratio) if prior.ox_ratio else base.process.ox_ratio),
    )


# ----------------------------------------------------------------------------------------------
# Device population (common random numbers)
# ----------------------------------------------------------------------------------------------

# 3-point Gauss-Hermite quadrature for the oxidation-depth scatter: nodes (-sqrt3, 0, sqrt3) sigma
GH_NODES = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
GH_WEIGHTS = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])


@dataclass(frozen=True)
class Population:
    """Fixed standard-normal draws shared by every design evaluated with it (common random numbers)."""

    z_d: np.ndarray
    z_b: np.ndarray
    z_psi: np.ndarray

    @staticmethod
    def make(n_dev: int, seed: int) -> "Population":
        rng = np.random.default_rng(seed)
        # antithetic pairs halve the sampling variance of the yield estimate
        h = n_dev // 2
        z = rng.standard_normal((3, h))
        z = np.concatenate([z, -z], axis=1)
        return Population(z[0], z[1], z[2])

    @property
    def n(self):
        return len(self.z_d)


# ----------------------------------------------------------------------------------------------
# Geometry cache
# ----------------------------------------------------------------------------------------------

_GEO_CACHE: dict = {}
_GEO_CACHE_MAX = 4096


def geometry_descriptors(design_key: tuple, depth: float, ox_ratio: float, dn_eff: float):
    key = (tuple(round(v, 9) for v in design_key), round(depth, 6), round(ox_ratio, 6), dn_eff, G.H_GRID, G.H_MODE)
    hit = _GEO_CACHE.get(key)
    if hit is not None:
        return hit
    design = G.MaskDesign(*design_key)
    ap = G.build_aperture(design, G.Process(depth, ox_ratio))
    d = G.descriptors(ap, dn_eff=dn_eff)
    if len(_GEO_CACHE) > _GEO_CACHE_MAX:
        _GEO_CACHE.clear()
    _GEO_CACHE[key] = d
    return d


# ----------------------------------------------------------------------------------------------
# Anisotropy backend interface
# ----------------------------------------------------------------------------------------------


class AnisotropyBackend:
    """Interface: aperture geometry -> (dichroism vector, birefringence vector, validity).

    `PhenomenologicalBackend` below is the default. To use a validated model (full-vector eigenmode
    solver, or a regression fitted to measured polarization data) implement `vectors` and pass it to
    `objective.evaluate_design(..., backend=...)`. Nothing else in the optimisation stack changes.
    """

    def vectors(self, design_key, world: World, pop: Population):
        raise NotImplementedError


class PhenomenologicalBackend(AnisotropyBackend):
    """d = d_nat + k_d s_mode,  b = b_nat + k_b s_stress   (see module docstring)."""

    min_area = 20.0   # um^2; smaller apertures are treated as pinched off (failed device)

    def vectors(self, design_key, world: World, pop: Population):
        """Returns d (3, M, 2), b (3, M, 2), valid (3,), diag dict; the leading axis is the depth-quadrature node."""
        M = pop.n
        d = np.zeros((3, M, 2))
        b = np.zeros((3, M, 2))
        valid = np.zeros(3, bool)
        diag = dict(area=np.zeros(3), s_mode=np.zeros((3, 2)), s_stress=np.zeros((3, 2)), gap=np.zeros(3))
        psi = np.deg2rad(world.sig_psi) * pop.z_psi
        cs = np.stack([np.cos(2 * psi), np.sin(2 * psi)], axis=-1)          # (M, 2) native axis direction
        d_nat = -(world.d0 + world.sig_d0 * pop.z_d)[:, None] * cs      # native mode = x (lower loss)
        b_nat = (world.b0 + world.sig_b0 * pop.z_b)[:, None] * cs
        for j in range(3):
            depth = max(world.process.depth + GH_NODES[j] * world.sig_depth, 0.0)
            ds = geometry_descriptors(design_key, depth, world.process.ox_ratio, world.dn_eff)
            valid[j] = ds.valid and ds.n_components == 1 and ds.area >= self.min_area
            diag["area"][j] = ds.area
            diag["s_mode"][j] = ds.s_mode
            diag["s_stress"][j] = ds.s_stress
            diag["gap"][j] = ds.mode_gap
            d[j] = d_nat + world.k_d * ds.s_mode[None, :]
            b[j] = b_nat + world.k_b * ds.s_stress[None, :]
        return d, b, valid, diag
