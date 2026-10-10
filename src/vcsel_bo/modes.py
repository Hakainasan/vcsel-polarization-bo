"""Scalar fundamental transverse mode of the oxide-defined index guide.

Weakly-guiding scalar (Helmholtz) problem in the effective-index picture,

    [ -laplacian - U0 * chi(x, y) ] psi = E psi ,      U0 = k0^2 (n_core^2 - n_clad^2) ~ k0^2 * 2 n Dn_eff,

with chi the (anti-aliased) aperture indicator. The fundamental mode is the lowest eigenpair (E < 0
means guided). This is a *scalar* model: it says where the optical field sits relative to the aperture
boundary and nothing about polarization by itself. Polarization enters only through the
second-rank descriptors in `geometry.descriptors` and the SFM in `sfm`. A vector solver
(FDTD/FEM/eigenmode) can replace `geometry.descriptors` through the `AnisotropyBackend` interface in
`physics`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _laplacian_2d(n, h):
    """Negative 2-D Laplacian, 9-point (Mehrstellen) stencil, Dirichlet boundary.

    L9 = (T x I + I x T) - (h^2/6) T x T with T = tridiag(-1, 2, -1)/h^2. The leading truncation error
    of this stencil is isotropic (proportional to the biharmonic operator), unlike the 5-point stencil
    whose O(h^2) error carries a 4-fold grid anisotropy that would leak into the rotation dependence
    of the polarization descriptors.
    """
    e = np.ones(n)
    T = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format="csr") / h ** 2
    I = sp.identity(n, format="csr")
    return sp.kron(I, T) + sp.kron(T, I) - (h ** 2 / 6.0) * sp.kron(T, T)


def fundamental_mode(chi: np.ndarray, h: float, U0: float):
    """Lowest scalar mode of the guide defined by indicator `chi` (rows = y, cols = x).

    Returns (psi, E0, gap): the normalised nodeless field on the grid, its eigenvalue and the gap to
    the next mode. Shift-invert about a value below the spectrum for speed.
    """
    n = chi.shape[0]
    A = _laplacian_2d(n, h) - sp.diags((U0 * chi).ravel(), 0, format="csr")
    k = 3
    try:
        vals, vecs = spla.eigsh(A, k=k, sigma=-U0 - 1.0, which="LM", tol=1e-6)
    except Exception:  # very rare ARPACK failure: dense fallback on a cropped problem
        vals, vecs = np.linalg.eigh(A.toarray())
        vals, vecs = vals[:k], vecs[:, :k]
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    psi = vecs[:, 0].reshape(n, n)
    psi = psi * np.sign(psi.sum() if psi.sum() != 0 else 1.0)
    psi = psi / np.sqrt((psi ** 2).sum())
    return psi, float(vals[0]), float(vals[1] - vals[0])
