"""Switching objective: the *yield* of clean polarization-switching devices.

Why yield
---------
The previous prototype scored a hand-weighted mix of a transient polarization swing, a Gaussian
"centre" term, a span term and a complexity penalty - four arbitrary weights and no physical meaning.
Here the objective is a single, physical, experimentally measurable quantity:

    Y(design) = P( a fabricated device of this design is a clean, in-window polarization switch ),

the fraction of devices that switch - exactly what Yuan et al. report per aperture design (JLT 2025;
CLEO 2025) and what a fabrication batch produces. "Clean" is a checklist (`Spec`), not a weighting:

  1. exactly one polarization switch occurs while the drive current is swept up through the operating range;
  2. it occurs inside the target window  mu_lo <= mu_sw <= mu_hi   (mu = I / I_threshold);
  3. a stable lasing state exists at every current of the operating range (no polarization chaos /
     unresolved dynamics), judged from the SFM fixed-point stability;
  4. the two states are (nearly) orthogonal linear polarizations (azimuth separation, low ellipticity);
  5. the aperture survived oxidation as one connected region of sufficient area.

Each condition is a *margin* in its own physical unit divided by a tolerance; a device passes iff the
smallest margin is positive. The optimiser uses the smoothed yield  mean sigma(2 m)  (logistic of the
worst margin) to give it gradient information; every reported number is the hard yield  mean 1[m > 0].
Device-to-device scatter of the native anisotropy, the axis misorientation and the oxidation depth
(aleatoric) is integrated with common random numbers, so evaluations are deterministic and comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sfm
from .physics import (GH_WEIGHTS, AnisotropyBackend, PhenomenologicalBackend, Population, World)
from .sfm import SFMConstants


@dataclass(frozen=True)
class Spec:
    """What counts as a good polarization-switching Ising node."""

    mu_lo: float = 1.25          # earliest acceptable switching current (I/I_th)
    mu_hi: float = 1.95          # latest acceptable switching current
    range_lo: float = 1.10       # operating range over which stable, clean states are required
    range_hi: float = 2.10
    min_sep_deg: float = 60.0    # min azimuth separation between the states before/after the switch
    max_ellip: float = 0.45      # max |S3| of either state (0 = linear, 1 = circular)
    min_stab: float = 0.02       # stable iff Re(lambda_max) < -min_stab  (ns^-1)
    tol_mu: float = 0.08         # tolerance scales for the margins
    tol_az: float = 15.0
    tol_ell: float = 0.15
    tol_stab: float = 0.10
    soft_sharpness: float = 2.0  # sigma(sharpness * margin) used by the optimiser only


MUS = np.linspace(1.05, 2.25, 17)


@dataclass
class Evaluation:
    hard_yield: float            # fraction of devices passing all checks
    soft_yield: float            # smoothed yield used as the BO objective
    mu_sw_median: float          # median switching current over devices that switch once in range
    frac_switch: float           # fraction of devices with exactly one switch in the operating range
    frac_valid_geometry: float   # depth-quadrature-weighted fraction of survivable apertures
    margins: np.ndarray          # (3, M) worst margins
    diag: dict


# ----------------------------------------------------------------------------------------------
# Sweep of both lasing branches
# ----------------------------------------------------------------------------------------------


def branch_states(m11, m12, mus, c: SFMConstants):
    """Both lasing branches of the SFM at every (device, current). m11, m12: (N,) or (N, nmu)."""
    N = np.shape(m11)[0]
    nmu = len(mus)
    MU = np.broadcast_to(mus[None, :], (N, nmu))
    m11 = np.broadcast_to(np.asarray(m11, complex).reshape(N, -1), (N, nmu))
    m12 = np.broadcast_to(np.asarray(m12, complex).reshape(N, -1), (N, nmu))
    out = []
    for which in (0, 1):
        u0 = sfm.linear_mode_guess(MU, m11, m12, c, which)
        u, ok = sfm.newton_fixed_point(u0, MU, m11, m12, c)
        lam = sfm.stability(u, MU, m11, m12, c)
        st = sfm.stokes(u)
        phys = ok & (st["intensity"] > 1e-3) & (u[..., 4] > 0.5)
        out.append(dict(u=u, lam=lam, phys=phys, az=st["azimuth"], ell=np.abs(st["S3"]), I=st["intensity"]))
    # discard a second branch that converged to the same fixed point as the first
    daz = np.abs(out[0]["az"] - out[1]["az"])
    daz = np.minimum(daz, 180.0 - daz)
    dup = (daz < 5.0) & (np.abs(out[0]["ell"] - out[1]["ell"]) < 0.05) & \
          (np.abs(out[0]["I"] - out[1]["I"]) < 0.01 * out[0]["I"])
    out[1]["phys"] = out[1]["phys"] & ~dup
    return out


def _ang_diff(a, b):
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def classify(br, mus, spec: Spec):
    """Margins and switching features for every device from the two branch tables."""
    inR = (mus >= spec.range_lo - 1e-9) & (mus <= spec.range_hi + 1e-9)
    jr = np.where(inR)[0]
    mu_r = mus[jr]
    N = br[0]["lam"].shape[0]
    stab = [b["phys"][:, jr] & (b["lam"][:, jr] < -spec.min_stab) for b in br]
    lam = [np.where(b["phys"][:, jr], b["lam"][:, jr], 1.0) for b in br]   # non-existent -> unstable
    az = [b["az"][:, jr] for b in br]
    ell = [b["ell"][:, jr] for b in br]

    occ = np.where(stab[0][:, 0], 0, np.where(stab[1][:, 0], 1, -1))
    occ0 = occ.copy()
    n_sw = np.zeros(N, int)
    mu_sw = np.full(N, np.nan)
    ar = np.arange(N)
    for j in range(1, len(jr)):
        cur = occ
        stab_cur = np.where(cur == 0, stab[0][:, j], np.where(cur == 1, stab[1][:, j], False))
        need = (cur >= 0) & ~stab_cur
        other_stab = np.where(cur == 0, stab[1][:, j], stab[0][:, j])
        sw = need & other_stab
        lost = need & ~other_stab
        # interpolated crossing of the departing branch's eigenvalue
        lc_prev = np.where(cur == 0, lam[0][:, j - 1], lam[1][:, j - 1])
        lc_now = np.where(cur == 0, lam[0][:, j], lam[1][:, j])
        t = np.where((lc_prev < 0) & (lc_now > 0), -lc_prev / np.where(lc_now - lc_prev == 0, 1, lc_now - lc_prev), 0.5)
        mu_here = mu_r[j - 1] + np.clip(t, 0, 1) * (mu_r[j] - mu_r[j - 1])
        first = sw & (n_sw == 0)
        mu_sw = np.where(first, mu_here, mu_sw)
        n_sw = n_sw + sw
        new_occ = np.where(sw, 1 - cur, np.where(lost, -1, cur))
        reacq = (cur == -1)
        new_occ = np.where(reacq & stab[0][:, j], 0, np.where(reacq & stab[1][:, j], 1, new_occ))
        occ = new_occ
    occ_end = occ

    # ---- margins -------------------------------------------------------------------------
    # (a) switching current
    m_mu = np.full(N, -6.0)
    has = (n_sw == 1) & np.isfinite(mu_sw)
    m_mu[has] = np.minimum(mu_sw[has] - spec.mu_lo, spec.mu_hi - mu_sw[has]) / spec.tol_mu
    # extrapolated crossing for devices that do not switch inside the range (gives the optimiser a gradient)
    l0 = lam[0]
    nos = (n_sw == 0)
    dmu = mu_r[1] - mu_r[0]
    slope_hi = (l0[:, -1] - l0[:, -2]) / dmu
    slope_lo = (l0[:, 1] - l0[:, 0]) / dmu
    above = nos & (occ0 == 0) & (occ_end == 0) & (slope_hi > 1e-6) & (np.abs(l0[:, -1]) < 5)
    mu_x_hi = mu_r[-1] - l0[:, -1] / np.where(slope_hi > 1e-6, slope_hi, 1.0)
    m_mu = np.where(above, np.clip(np.minimum(mu_x_hi - spec.mu_lo, spec.mu_hi - mu_x_hi) / spec.tol_mu, -6, 6), m_mu)
    below = nos & (occ0 == 1) & (slope_lo > 1e-6) & (np.abs(l0[:, 0]) < 5)
    mu_x_lo = mu_r[0] - l0[:, 0] / np.where(slope_lo > 1e-6, slope_lo, 1.0)
    m_mu = np.where(below, np.clip(np.minimum(mu_x_lo - spec.mu_lo, spec.mu_hi - mu_x_lo) / spec.tol_mu, -6, 6), m_mu)
    m_mu = np.minimum(m_mu, 6.0)
    # (b) exactly one switch
    m_one = np.where(n_sw == 1, 2.0, -1.0)
    # (c) stable state exists at every current of the operating range
    best = np.max(np.stack([-lam[0], -lam[1]], 0), axis=0)            # (N, nR): best stability margin per current
    m_stab = (best.min(axis=1) - spec.min_stab) / spec.tol_stab
    # (d) orthogonality and linearity of the two states (only meaningful when a switch happens)
    a_start = np.where(occ0 == 0, az[0][:, 0], az[1][:, 0])
    a_end = np.where(occ_end == 0, az[0][:, -1], az[1][:, -1])
    e_start = np.where(occ0 == 0, ell[0][:, 0], ell[1][:, 0])
    e_end = np.where(occ_end == 0, ell[0][:, -1], ell[1][:, -1])
    m_sep = np.where(n_sw == 1, (_ang_diff(a_start, a_end) - spec.min_sep_deg) / spec.tol_az, 2.0)
    m_ell = np.where(n_sw == 1, (spec.max_ellip - np.maximum(e_start, e_end)) / spec.tol_ell, 2.0)
    m_all = np.minimum.reduce([m_mu, m_one, m_stab, m_sep, m_ell])
    feats = dict(mu_sw=mu_sw, n_sw=n_sw, m_mu=m_mu, m_one=m_one, m_stab=m_stab, m_sep=m_sep, m_ell=m_ell,
                 sep=_ang_diff(a_start, a_end), a_start=a_start, a_end=a_end, e_start=e_start, e_end=e_end,
                 occ0=occ0, occ_end=occ_end)
    return m_all, feats


# ----------------------------------------------------------------------------------------------
# Design evaluation
# ----------------------------------------------------------------------------------------------

_DEFAULT_BACKEND = PhenomenologicalBackend()


def evaluate_design(x, world: World, pop: Population, spec: Spec = Spec(),
                    backend: AnisotropyBackend | None = None, c: SFMConstants = SFMConstants(),
                    mus: np.ndarray = MUS) -> Evaluation:
    """Yield of a mask design x = (L, ratio, w, rho, theta) in a given `World`, over a device population."""
    backend = backend or _DEFAULT_BACKEND
    key = tuple(float(v) for v in x)
    d, b, valid, diag = backend.vectors(key, world, pop)          # (3, M, 2)
    M = pop.n
    d = d.reshape(3 * M, 2)
    b = b.reshape(3 * M, 2)
    aniso = dict(d_norm=np.linalg.norm(d, axis=1).reshape(3, M), b_norm=np.linalg.norm(b, axis=1).reshape(3, M))
    if world.chi_th != 0.0:                                        # optional thermal dichroism drift
        d = d[:, None, :] + world.chi_th * (mus[None, :, None] - world.mu0) * b[:, None, :]
        m11 = d[..., 0] + 1j * b[:, None, 0]
        m12 = d[..., 1] + 1j * b[:, None, 1]
    else:
        m11 = d[:, 0] + 1j * b[:, 0]
        m12 = d[:, 1] + 1j * b[:, 1]
    br = branch_states(m11, m12, mus, c)
    m_all, feats = classify(br, mus, spec)
    m_all = m_all.reshape(3, M)
    m_all = np.where(valid[:, None], m_all, -6.0)
    hard = (m_all > 0).astype(float)
    soft = 1.0 / (1.0 + np.exp(-spec.soft_sharpness * np.clip(m_all, -30, 30)))
    w = GH_WEIGHTS[:, None] / M
    sw_ok = (feats["n_sw"].reshape(3, M) == 1) & valid[:, None]
    mu_sw = feats["mu_sw"].reshape(3, M)
    med = float(np.nanmedian(mu_sw[sw_ok])) if sw_ok.any() else float("nan")
    return Evaluation(
        hard_yield=float((hard * w).sum()),
        soft_yield=float((soft * w).sum()),
        mu_sw_median=med,
        frac_switch=float((sw_ok * w).sum()),
        frac_valid_geometry=float((valid * GH_WEIGHTS).sum()),
        margins=m_all,
        diag=dict(geometry=diag, feats=feats, aniso=aniso),
    )
