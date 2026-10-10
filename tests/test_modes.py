"""Scalar mode solver vs the exact LP01 solution of a circular step-index guide."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, k0, k1

from vcsel_bo import modes as M


def lp01_energy(a, U0):
    V = a * np.sqrt(U0)
    def g(u):
        w = np.sqrt(V ** 2 - u ** 2)
        return u * j1(u) / j0(u) - w * k1(w) / k0(w)
    u = brentq(g, 1e-6, min(2.4048, V * 0.999999) - 1e-6)
    return -((np.sqrt(V ** 2 - u ** 2)) / a) ** 2


def _disk_chi(a, h, half=10.0, ss=8):
    n = int(2 * half / h) + 1
    ax = (np.arange(n) - (n - 1) / 2) * h
    X, Y = np.meshgrid(ax, ax)
    sub = (np.arange(ss) + 0.5) / ss - 0.5
    chi = np.zeros((n, n))
    for dx in sub:
        for dy in sub:
            chi += np.hypot(X + dx * h, Y + dy * h) < a
    return chi / ss ** 2


def test_lp01_eigenvalue_matches_analytic():
    a, U0, h = 3.0, 2.95, 0.5
    psi, E0, gap = M.fundamental_mode(_disk_chi(a, h), h, U0)
    assert abs(E0 - lp01_energy(a, U0)) / abs(lp01_energy(a, U0)) < 0.005


def test_fundamental_mode_is_nodeless_and_normalised():
    a, U0, h = 3.0, 2.95, 0.5
    psi, E0, gap = M.fundamental_mode(_disk_chi(a, h), h, U0)
    assert (psi >= -1e-9).all()
    assert abs((psi ** 2).sum() - 1.0) < 1e-9
    assert gap > 0
