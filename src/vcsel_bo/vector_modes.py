"""Polarization-resolved transverse modes of the oxide aperture: FORM birefringence from geometry.

What this adds to the package
-----------------------------
`modes.py` solves a *scalar* mode, so the package cannot compute any polarization splitting from the shape;
the birefringence enters the spin-flip model only through the phenomenological term k_b * s_stress with an
unknown coefficient. This module computes one physical contribution from first principles: the form
(geometric) birefringence of the lateral index guide formed by the aperture. It does NOT compute the stress
(elasto-optic) birefringence, which the literature points to as the stronger effect [choquette1995]; see the
note at the end.

Model (semi-vectorial finite differences, effective-index picture)
-----------------------------------------------------------------
For the field polarized along x the transverse equation is

    d/dx[ (1/eps) d(eps Ex)/dx ] + d2Ex/dy2 + k0^2 (eps - eps_ref) Ex = lam Ex,     lam = beta^2 - k0^2 eps_ref,

i.e. the normal component eps*Ex is continuous across an x-interface (and Ex across a y-interface); for Ey the
roles of x and y swap. Interfaces use the arithmetic mean of eps at half-nodes; a 5-point stencil; Dirichlet
walls far from the mode. The two polarizations are solved on the SAME grid, so the scalar discretisation error
cancels in their difference.

A rotated anisotropy has an arbitrary principal axis, but a semi-vectorial solver only knows grid x and y.
Because birefringence is a second-rank quantity, beta(alpha) - beta(alpha + 90) = D cos 2(alpha - alpha0);
solving on a grid aligned with alpha = 0 and on one aligned with alpha = 45 deg gives D (the splitting) and
alpha0 (the polarization with the larger propagation constant = the lower optical frequency).

Conversion to the spin-flip model: Dn_eff = Dbeta / k0, splitting Dnu = nu * Dn_eff / n_g, gamma_p = pi * Dnu
(same convention as `physics.py`). The group index n_g is an assumption (default 3.6).

Assumptions (state them on the poster if you use the numbers)
  * effective-index step dn_eff of the aperture (default 0.01, the same value the rest of the package uses);
  * weak guidance; one transverse mode (the fundamental) per polarization;
  * no stress, no thermal lens, no carrier-induced index, no gain guiding;
  * the sign with which the result enters the SFM birefringence vector (see `FormBirefringenceBackend`) depends on
    the field-envelope convention and is NOT verified; the magnitude and axis are what this module establishes.

Stress birefringence (not here): needs a plane-strain or 3-D mechanical FEM of the oxidised layer (oxide volume
shrinkage as the load), followed by the photoelastic tensor of GaAs. COMSOL (Structural Mechanics), or the open-source
scikit-fem / FEniCSx, can do the mechanical part; its output plugs into the same backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage as ndi

from . import geometry as G

C_UM_PER_NS = 2.99792458e5      # speed of light in um/ns


# ----------------------------------------------------------------------------------------------
# Operator and eigen-solver
# ----------------------------------------------------------------------------------------------


def semivector_operator(eps: np.ndarray, h: float, k0: float, pol: str, eps_ref: float | None = None):
    """Sparse operator A with A E = lam E for polarization 'x' or 'y'. eps: (Ny, Nx), rows = y, cols = x."""
    if pol not in ("x", "y"):
        raise ValueError("pol must be 'x' or 'y'")
    if eps_ref is None:
        eps_ref = float(eps.min())
    ep = eps if pol == "x" else eps.T            # make the corrected direction the column (x) axis
    ny, nx = ep.shape
    epR = np.concatenate([ep[:, 1:], ep[:, -1:]], axis=1)
    epL = np.concatenate([ep[:, :1], ep[:, :-1]], axis=1)
    hR, hL = 0.5 * (ep + epR), 0.5 * (ep + epL)
    cR = epR / hR / h ** 2
    cL = epL / hL / h ** 2
    dX = -ep * (1.0 / hR + 1.0 / hL) / h ** 2
    cR[:, -1] = 0.0                               # Dirichlet: no neighbour beyond the wall
    cL[:, 0] = 0.0
    N = nx * ny
    main = dX.ravel() - 2.0 / h ** 2 + k0 ** 2 * (ep.ravel() - eps_ref)
    up = cR.ravel()[:-1]                          # A[k, k+1]
    lo = cL.ravel()[1:]                           # A[k, k-1]
    ones = np.full(N - nx, 1.0 / h ** 2)          # plain second difference along the other axis
    A = sp.diags([main, up, lo, ones, ones], [0, 1, -1, nx, -nx], shape=(N, N), format="csc")
    if pol == "y":
        # the operator was built for the transposed array; permute back to the row-major order of eps
        orig_of_t = np.arange(N).reshape(nx, ny).T.ravel()   # original row-major index of each transposed node
        P = sp.csr_matrix((np.ones(N), (np.arange(N), orig_of_t)), shape=(N, N))   # x_t = P x_orig
        A = (P.T @ A @ P).tocsc()
    return A, eps_ref


def fundamental(eps: np.ndarray, h: float, k0: float, pol: str, eps_ref: float | None = None, k: int = 2):
    """Largest eigenvalues lam (= beta^2 - k0^2 eps_ref) for one polarization; returns (lam sorted desc, vecs)."""
    A, eps_ref = semivector_operator(eps, h, k0, pol, eps_ref)
    top = k0 ** 2 * (float(eps.max()) - eps_ref)
    sigma = 1.05 * top + 0.05                      # just above the guided spectrum -> nearest = fundamental
    vals, vecs = spla.eigs(A, k=k, sigma=sigma, which="LM", tol=1e-12)
    order = np.argsort(-vals.real)
    return vals.real[order], vecs[:, order].real, eps_ref


# ----------------------------------------------------------------------------------------------
# Aperture -> index map -> splitting
# ----------------------------------------------------------------------------------------------


@dataclass
class FormBirefringence:
    dneff: float          # n_eff(alpha0) - n_eff(alpha0 + 90), >= 0
    axis_deg: float       # alpha0 in [0, 180): polarization with the larger n_eff (lower frequency)
    dnu_ghz: float        # frequency splitting
    gamma_p: float        # pi * splitting, ns^-1 (SFM units)
    lam_gap: float        # (lam_1 - lam_2) / lam_1 of the x-polarized set: small -> nearly degenerate / multimode
    n_eff: float          # mean effective index of the fundamental
    h: float
    valid: bool


def index_map_rotated(ap: G.Aperture, alpha_deg: float, h: float, margin: float, n_core: float, dn_eff: float):
    """eps on a square grid whose x axis points along physical angle alpha (crystal frame)."""
    if not ap.loops:
        raise ValueError("aperture is pinched off")
    loop = ap.loops[0]
    c = loop.mean(axis=0)
    half = np.max(np.hypot(*(loop - c).T)) + margin
    n = int(np.ceil(2 * half / h)) | 1                       # odd -> a node at the centre
    q = (np.arange(n) - n // 2) * h
    QX, QY = np.meshgrid(q, q)
    a = np.deg2rad(alpha_deg)
    PX = c[0] + np.cos(a) * QX - np.sin(a) * QY
    PY = c[1] + np.sin(a) * QX + np.cos(a) * QY
    ax0, hg = ap.axes[0], ap.axes[1] - ap.axes[0]
    F = ndi.map_coordinates(ap.F, [(PY - ax0) / hg, (PX - ax0) / hg], order=3, mode="nearest")
    depth = ap.process.depth if ap.process.depth > 0 else 1.0
    chi = np.clip(0.5 + F * depth / h, 0.0, 1.0)              # anti-aliased coverage from the signed distance
    n_clad = n_core - dn_eff
    return n_clad ** 2 + (n_core ** 2 - n_clad ** 2) * chi, n_clad ** 2


def form_birefringence(design: G.MaskDesign, process: G.Process, dn_eff: float = 0.01, n_core: float = 3.3,
                       wavelength_um: float = 0.94, n_g: float = 3.6, h: float = 0.1, margin: float = 3.0,
                       aperture: G.Aperture | None = None) -> FormBirefringence:
    ap = aperture if aperture is not None else G.build_aperture(design, process)
    if (not ap.loops) or ap.area < 4.0:
        return FormBirefringence(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, h, False)
    k0 = 2 * np.pi / wavelength_um
    dbeta, gaps, betas = [], [], []
    for alpha in (0.0, 45.0):
        eps, eps_ref = index_map_rotated(ap, alpha, h, margin, n_core, dn_eff)
        lx, _, _ = fundamental(eps, h, k0, "x", eps_ref)
        ly, _, _ = fundamental(eps, h, k0, "y", eps_ref)
        bx = np.sqrt(lx[0] + k0 ** 2 * eps_ref)
        by = np.sqrt(ly[0] + k0 ** 2 * eps_ref)
        dbeta.append((lx[0] - ly[0]) / (bx + by))             # beta_x - beta_y without cancellation error
        betas.append(0.5 * (bx + by))
        gaps.append((lx[0] - lx[1]) / lx[0] if lx[0] > 0 else 0.0)
    D = float(np.hypot(*dbeta))
    axis = float(np.rad2deg(0.5 * np.arctan2(dbeta[1], dbeta[0])) % 180.0)
    dneff = D / k0
    nu = C_UM_PER_NS / wavelength_um                          # GHz (1/ns)
    dnu = nu * dneff / n_g
    guided = all(g > 0 for g in gaps)
    return FormBirefringence(dneff, axis, dnu, np.pi * dnu, float(min(gaps)), float(np.mean(betas) / k0), h, guided)


# ----------------------------------------------------------------------------------------------
# Optional backend for the optimiser
# ----------------------------------------------------------------------------------------------


def make_backend(sign: float = 1.0, **kw):
    """Backend that adds the computed form birefringence to the phenomenological one:

        b = b_nat + k_b * s_stress + sign * gamma_p_form * (cos 2 alpha0, sin 2 alpha0).

    The stress term stays phenomenological (k_b from the anchored worlds). `sign` maps "larger n_eff along alpha0"
    onto the SFM birefringence vector; its correct value depends on the field-envelope convention of `sfm.py` and
    has not been verified, so run the optimiser with both signs before trusting a ranking.
    """
    from .physics import GH_NODES, PhenomenologicalBackend

    cache: dict = {}

    class FormBirefringenceBackend(PhenomenologicalBackend):
        def vectors(self, design_key, world, pop):
            d, b, valid, diag = super().vectors(design_key, world, pop)
            diag["gamma_p_form"] = np.zeros(3)
            for j in range(3):
                depth = max(world.process.depth + GH_NODES[j] * world.sig_depth, 0.0)
                key = (tuple(round(v, 9) for v in design_key), round(depth, 6), round(world.process.ox_ratio, 6), world.dn_eff)
                if key not in cache:
                    proc = G.Process(depth=depth, ox_ratio=world.process.ox_ratio)
                    cache[key] = form_birefringence(G.MaskDesign(*design_key), proc, dn_eff=world.dn_eff, **kw)
                fb = cache[key]
                if fb.valid:
                    a = np.deg2rad(fb.axis_deg)
                    b[j] = b[j] + sign * fb.gamma_p * np.array([np.cos(2 * a), np.sin(2 * a)])[None, :]
                    diag["gamma_p_form"][j] = fb.gamma_p
            return d, b, valid, diag

    return FormBirefringenceBackend()
