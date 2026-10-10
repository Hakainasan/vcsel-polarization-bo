"""Spin-Flip Model (SFM) of a single-transverse-mode VCSEL: steady states and stability.

Model (linear-polarization basis; San Miguel, Feng & Moloney 1995; Martin-Regalado et al. 1997)
--------------------------------------------------------------------------------------------
    dEx/dt = -(kappa + i kappa alpha) Ex + kappa (1 + i alpha) (D Ex + i n Ey) - (m11 Ex + m12 Ey)
    dEy/dt = -(kappa + i kappa alpha) Ey + kappa (1 + i alpha) (D Ey - i n Ex) - (m12 Ex - m11 Ey)
    dD/dt  = -gamma [ D (1 + |Ex|^2 + |Ey|^2) - mu + i n (Ey Ex* - Ex Ey*) ]
    dn/dt  = -gamma_s n - gamma [ n (|Ex|^2 + |Ey|^2) + i D (Ey Ex* - Ex Ey*) ]

with the complex-symmetric, traceless linear-anisotropy matrix

    M = (gamma_a1 + i gamma_p1) sigma_z + (gamma_a2 + i gamma_p2) sigma_x,
    m11 = gamma_a1 + i gamma_p1,   m12 = gamma_a2 + i gamma_p2.

(gamma_a1, gamma_p1) are the dichroism / birefringence along the x-y axes;
(gamma_a2, gamma_p2) the same along the 45-degree diagonals. For gamma_a2 = gamma_p2 = 0 this is
exactly the textbook aligned SFM (x-mode has extra loss gamma_a, phase offset gamma_p).

Why steady states instead of time integration
---------------------------------------------
The earlier prototype integrated each current point for 0.22 ns while the carrier lifetime is
1/gamma = 1 ns, i.e. it scored a *transient*. Here every current point is instead analysed at its
fixed points: Newton solve in the frame co-rotating with the lasing frequency, stability from the
eigenvalues of the 6x6 Jacobian (the trivial global-phase mode removed). Time-domain integration
(`integrate`) is kept only to validate the fixed-point analysis (see tests).

Units: rates in ns^-1, time in ns, fields normalised to the saturation intensity, mu = I / I_th.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SFMConstants:
    """Material/cavity constants of the SFM. Defaults: typical 850-980 nm GaAs VCSEL magnitudes
    used in the SFM literature (Martin-Regalado 1997). They are illustrative, not fitted to any
    wafer of yours."""

    kappa: float = 300.0     # field decay rate (ns^-1)
    alpha: float = 3.0       # linewidth-enhancement factor
    gamma: float = 1.0       # carrier decay rate (ns^-1)
    gamma_s: float = 50.0    # spin-flip rate (ns^-1)


# state layout: y = [Exr, Exi, Eyr, Eyi, D, n]


def rhs(y, mu, m11, m12, c: SFMConstants):
    """Right-hand side of the SFM. y: (..., 6); mu, m11, m12 broadcast against y[..., 0]."""
    y = np.asarray(y, dtype=float)
    Ex = y[..., 0] + 1j * y[..., 1]
    Ey = y[..., 2] + 1j * y[..., 3]
    D = y[..., 4]
    n = y[..., 5]
    kap, alp = c.kappa, c.alpha
    g = kap * (1 + 1j * alp)
    lin = -kap * (1 + 1j * alp)
    dEx = lin * Ex + g * (D * Ex + 1j * n * Ey) - (m11 * Ex + m12 * Ey)
    dEy = lin * Ey + g * (D * Ey - 1j * n * Ex) - (m12 * Ex - m11 * Ey)
    I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    im = (Ey * np.conj(Ex)).imag           # Im(Ey Ex*)
    # i n (Ey Ex* - Ex Ey*) = i n (2 i Im(Ey Ex*)) = -2 n Im(Ey Ex*)
    dD = -c.gamma * (D * (1 + I) - mu - 2 * n * im)
    dn = -c.gamma_s * n - c.gamma * (n * I - 2 * D * im)
    out = np.stack([dEx.real, dEx.imag, dEy.real, dEy.imag, dD, dn], axis=-1)
    return out


def rhs_circular_reference(y, mu, m11, m12, c: SFMConstants):
    """Independent implementation in the circular basis, used only as a test oracle.

    E+ = (Ex - i Ey)/sqrt2, E- = (Ex + i Ey)/sqrt2; carriers N+/N- with D=(N+ + N-)/2, n=(N- - N+)/2.
    Returned in the linear basis so it can be compared to `rhs`.
    """
    y = np.asarray(y, dtype=float)
    Ex = y[..., 0] + 1j * y[..., 1]
    Ey = y[..., 2] + 1j * y[..., 3]
    D = y[..., 4]
    n = y[..., 5]
    Ep = (Ex - 1j * Ey) / np.sqrt(2)
    Em = (Ex + 1j * Ey) / np.sqrt(2)
    kap, alp = c.kappa, c.alpha
    g = kap * (1 + 1j * alp)
    # linear-anisotropy matrix M in the circular basis: M_circ = U^dagger M U
    # For E = (Ex,Ey), Ep=(Ex-iEy)/r2, Em=(Ex+iEy)/r2 :
    #   (M E)_x = m11 Ex + m12 Ey ; (M E)_y = m12 Ex - m11 Ey
    MEx = m11 * Ex + m12 * Ey
    MEy = m12 * Ex - m11 * Ey
    MEp = (MEx - 1j * MEy) / np.sqrt(2)
    MEm = (MEx + 1j * MEy) / np.sqrt(2)
    Np = D - n     # carrier populations coupled to E+ and E-  (n_here = -n_std of San Miguel 1995)
    Nm = D + n
    dEp = -kap * (1 + 1j * alp) * Ep + g * Np * Ep - MEp
    dEm = -kap * (1 + 1j * alp) * Em + g * Nm * Em - MEm
    Ip, Im_ = np.abs(Ep) ** 2, np.abs(Em) ** 2
    # standard SFM carrier equations rewritten with n_std = -n:
    dD = -c.gamma * (D * (1 + Ip + Im_) - mu - n * (Ip - Im_))
    dn = -c.gamma_s * n - c.gamma * (n * (Ip + Im_) - D * (Ip - Im_))
    dEx = (dEp + dEm) / np.sqrt(2)
    dEy = 1j * (dEp - dEm) / np.sqrt(2)
    return np.stack([dEx.real, dEx.imag, dEy.real, dEy.imag, dD, dn], axis=-1)


# ----------------------------------------------------------------------------------------------
# Fixed points
# ----------------------------------------------------------------------------------------------


def _rot_rhs(u, mu, m11, m12, c):
    """RHS in the frame rotating at Omega (u[...,6] = Omega); returns F (..., 6)."""
    y = u[..., :6]
    Om = u[..., 6]
    F = rhs(y, mu, m11, m12, c)
    Ex = y[..., 0] + 1j * y[..., 1]
    Ey = y[..., 2] + 1j * y[..., 3]
    dEx = -1j * Om * Ex
    dEy = -1j * Om * Ey
    F[..., 0] += dEx.real
    F[..., 1] += dEx.imag
    F[..., 2] += dEy.real
    F[..., 3] += dEy.imag
    return F


def _cblock(z):
    """Real 2x2 block of multiplication by complex z: [[Re z, -Im z], [Im z, Re z]]."""
    return z.real, -z.imag, z.imag, z.real


def jacobian(u, mu, m11, m12, c: SFMConstants):
    """Analytic Jacobian dF/d(Exr,Exi,Eyr,Eyi,D,n,Omega) of the rotating-frame RHS, (..., 6, 7)."""
    y = u[..., :6]
    Om = u[..., 6]
    x1, x2, y1, y2, D, n = (y[..., k] for k in range(6))
    Ex = x1 + 1j * x2
    Ey = y1 + 1j * y2
    g = c.kappa * (1 + 1j * c.alpha)
    I = x1 ** 2 + x2 ** 2 + y1 ** 2 + y2 ** 2
    im = y2 * x1 - y1 * x2
    J = np.zeros(u.shape[:-1] + (6, 7))
    # field rows: complex partials
    dxx = -g + g * D - m11 - 1j * Om
    dxy = 1j * g * n - m12
    dyx = -1j * g * n - m12
    dyy = -g + g * D + m11 - 1j * Om
    for (r0, c0, z) in ((0, 0, dxx), (0, 2, dxy), (2, 0, dyx), (2, 2, dyy)):
        a, b, cc, d = _cblock(z)
        J[..., r0, c0] = a
        J[..., r0, c0 + 1] = b
        J[..., r0 + 1, c0] = cc
        J[..., r0 + 1, c0 + 1] = d
    # d/dD, d/dn of field rows
    for (r0, dz_D, dz_n) in ((0, g * Ex, 1j * g * Ey), (2, g * Ey, -1j * g * Ex)):
        J[..., r0, 4] = dz_D.real
        J[..., r0 + 1, 4] = dz_D.imag
        J[..., r0, 5] = dz_n.real
        J[..., r0 + 1, 5] = dz_n.imag
    # d/dOmega of field rows: -i E
    J[..., 0, 6] = x2
    J[..., 1, 6] = -x1
    J[..., 2, 6] = y2
    J[..., 3, 6] = -y1
    ga, gs = c.gamma, c.gamma_s
    # carrier rows
    J[..., 4, 0] = -ga * (2 * D * x1 - 2 * n * y2)
    J[..., 4, 1] = -ga * (2 * D * x2 + 2 * n * y1)
    J[..., 4, 2] = -ga * (2 * D * y1 + 2 * n * x2)
    J[..., 4, 3] = -ga * (2 * D * y2 - 2 * n * x1)
    J[..., 4, 4] = -ga * (1 + I)
    J[..., 4, 5] = 2 * ga * im
    J[..., 5, 0] = -ga * (2 * n * x1 - 2 * D * y2)
    J[..., 5, 1] = -ga * (2 * n * x2 + 2 * D * y1)
    J[..., 5, 2] = -ga * (2 * n * y1 + 2 * D * x2)
    J[..., 5, 3] = -ga * (2 * n * y2 - 2 * D * x1)
    J[..., 5, 4] = 2 * ga * im
    J[..., 5, 5] = -gs - ga * I
    return J


def linear_mode_guess(mu, m11, m12, c: SFMConstants, which: int):
    """Initial guess for the two lasing branches from the eigenvectors of the anisotropy matrix.

    Near threshold each mode is an eigenvector of M (complex symmetric 2x2). `which` = 0/1 picks the
    eigen-mode with the smaller/larger loss (Re lambda). Returns u (..., 7) with n = 0.
    """
    m11 = np.asarray(m11, dtype=complex)
    m12 = np.asarray(m12, dtype=complex)
    shp = np.broadcast_shapes(np.shape(mu), m11.shape, m12.shape)
    mu = np.broadcast_to(mu, shp)
    m11 = np.broadcast_to(m11, shp)
    m12 = np.broadcast_to(m12, shp)
    # eigenvalues of [[m11, m12],[m12, -m11]] : +-sqrt(m11^2 + m12^2)
    lam = np.sqrt(m11 ** 2 + m12 ** 2 + 0j)
    # choose sign so that lam is the eigenvalue with Re>=0 (larger loss)
    sgn = np.where(lam.real < 0, -1.0, 1.0)
    lam = lam * sgn
    lam_lo = -lam                 # lower-loss (gain-favoured) eigenvalue
    lam_hi = lam
    lam_sel = lam_lo if which == 0 else lam_hi
    # eigenvector (m12, lam - m11) for eigenvalue lam of [[m11, m12],[m12, -m11]] (first row: m11 v1 + m12 v2 = lam v1)
    v1 = m12
    v2 = lam_sel - m11
    # if m12 ~ 0 fall back to the axis-aligned eigenvector
    small = np.abs(m12) < 1e-12
    v1 = np.where(small, np.where(np.abs(lam_sel - m11) < 1e-12, 1.0, 0.0), v1)
    v2 = np.where(small, np.where(np.abs(lam_sel - m11) < 1e-12, 0.0, 1.0), v2)
    nrm = np.sqrt(np.abs(v1) ** 2 + np.abs(v2) ** 2)
    v1, v2 = v1 / nrm, v2 / nrm
    Dth = 1.0 + lam_sel.real / c.kappa
    Iexc = np.maximum(mu / Dth - 1.0, 1e-3)
    amp = np.sqrt(Iexc)
    Ex = amp * v1
    Ey = amp * v2
    Om = (c.alpha * c.kappa * (Dth - 1.0) - lam_sel.imag)
    u = np.stack([Ex.real, Ex.imag, Ey.real, Ey.imag, Dth * np.ones(shp), np.zeros(shp), Om], axis=-1)
    return u


def newton_fixed_point(u0, mu, m11, m12, c: SFMConstants, max_iter=30, tol=1e-8):
    """Damped Newton solve of the fixed-point equations with a phase gauge condition.

    Unknowns u = (Exr,Exi,Eyr,Eyi,D,n,Omega). The 6 equations F(u)=0 are augmented by the gauge
    condition  g . du = 0  with g the direction of the global-phase rotation, giving a square 7x7
    system. Only unconverged points are iterated. Returns (u, converged mask).
    """
    u = np.array(u0, dtype=float, copy=True)
    shp = u.shape[:-1]
    mu = np.broadcast_to(mu, shp)
    m11 = np.broadcast_to(np.asarray(m11, dtype=complex), shp)
    m12 = np.broadcast_to(np.asarray(m12, dtype=complex), shp)
    flat = int(np.prod(shp))
    U = u.reshape(flat, 7)
    MU = np.ascontiguousarray(mu).reshape(flat)
    M11 = np.ascontiguousarray(m11).reshape(flat)
    M12 = np.ascontiguousarray(m12).reshape(flat)
    scale = tol * max(1.0, c.kappa)
    active = np.arange(flat)
    for _ in range(max_iter):
        Ua = U[active]
        F = _rot_rhs(Ua, MU[active], M11[active], M12[active], c)
        done = np.max(np.abs(F), axis=-1) < scale
        if done.all():
            active = active[:0]
            break
        keep = ~done
        active = active[keep]
        Ua, F = Ua[keep], F[keep]
        J = jacobian(Ua, MU[active], M11[active], M12[active], c)
        A = np.zeros((len(active), 7, 7))
        A[:, :6, :] = J
        A[:, 6, 0] = -Ua[:, 1]
        A[:, 6, 1] = Ua[:, 0]
        A[:, 6, 2] = -Ua[:, 3]
        A[:, 6, 3] = Ua[:, 2]
        rhs_v = np.zeros((len(active), 7))
        rhs_v[:, :6] = -F
        try:
            step = np.linalg.solve(A, rhs_v[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = np.zeros_like(rhs_v)
            for k in range(len(active)):
                try:
                    step[k] = np.linalg.lstsq(A[k], rhs_v[k], rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        smax = np.max(np.abs(step[:, :6]), axis=-1)
        damp = np.minimum(1.0, 0.5 / np.maximum(smax, 1e-12))
        Un = Ua + damp[:, None] * step
        Un[:, 4] = np.clip(Un[:, 4], 0.0, 5.0)
        U[active] = Un
    Fall = _rot_rhs(U, MU, M11, M12, c)
    conv = np.max(np.abs(Fall), axis=-1) < 1e3 * scale
    return U.reshape(u.shape), conv.reshape(shp)


def stability(u, mu, m11, m12, c: SFMConstants):
    """Leading non-trivial Jacobian eigenvalue real part at fixed points u (..., 7).

    The Jacobian always has one exact zero eigenvalue (global-phase symmetry); it is removed by
    dropping the eigenvalue of smallest modulus. Stable iff the result is < 0.
    """
    shp = u.shape[:-1]
    mu = np.broadcast_to(mu, shp)
    m11 = np.broadcast_to(np.asarray(m11, dtype=complex), shp)
    m12 = np.broadcast_to(np.asarray(m12, dtype=complex), shp)
    J = jacobian(u, mu, m11, m12, c)[..., :6, :6]
    lam = np.linalg.eigvals(J)
    idx = np.argmin(np.abs(lam), axis=-1)
    re = lam.real.copy()
    np.put_along_axis(re, idx[..., None], -np.inf, axis=-1)
    return re.max(axis=-1)


def stokes(u):
    """Normalised polarization descriptors of fixed points u (..., >=4).

    Returns dict: intensity; azimuth (deg, 0..180); ellipticity chi = S3 (in [-1,1], 0 = linear);
    P = S1 = (Ix - Iy)/(Ix + Iy).
    """
    Ex = u[..., 0] + 1j * u[..., 1]
    Ey = u[..., 2] + 1j * u[..., 3]
    Ix, Iy = np.abs(Ex) ** 2, np.abs(Ey) ** 2
    I = Ix + Iy + 1e-30
    S1 = (Ix - Iy) / I
    S2 = 2 * (Ex * np.conj(Ey)).real / I
    S3 = 2 * (Ex * np.conj(Ey)).imag / I
    az = (0.5 * np.degrees(np.arctan2(S2, S1))) % 180.0
    return dict(intensity=Ix + Iy, azimuth=az, ellipticity=S3, P=S1, S1=S1, S2=S2, S3=S3)


# ----------------------------------------------------------------------------------------------
# Time-domain integration (validation only)
# ----------------------------------------------------------------------------------------------


def integrate(y0, mu, m11, m12, c: SFMConstants, t_end=30.0, rtol=1e-7, atol=1e-9):
    """Reference time-domain integration (implicit LSODA/Radau) used to validate fixed points."""
    from scipy.integrate import solve_ivp

    sol = solve_ivp(lambda t, y: rhs(y, mu, m11, m12, c), (0.0, t_end), np.asarray(y0, float),
                    method="LSODA", rtol=rtol, atol=atol)
    return sol
