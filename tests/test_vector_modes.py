"""Checks of the semi-vectorial solver (vector_modes.py) against exact results and symmetry."""
import numpy as np
import pytest
from scipy.optimize import brentq

from vcsel_bo import geometry as G
from vcsel_bo import vector_modes as V

WL = 0.94
K0 = 2 * np.pi / WL


def slab_beta(n1, n2, t, pol):
    """Exact fundamental (even) mode of a symmetric slab of thickness t: TE or TM."""
    r = 1.0 if pol == "TE" else (n1 / n2) ** 2

    def f(b):
        kap = np.sqrt(K0 ** 2 * n1 ** 2 - b ** 2)
        gam = np.sqrt(b ** 2 - K0 ** 2 * n2 ** 2)
        return kap * np.tan(kap * t / 2) - r * gam

    # first branch of tan: kappa * t / 2 < pi / 2  <=>  beta > sqrt(k0^2 n1^2 - (pi/t)^2)
    lo = max(K0 * n2, np.sqrt(max(K0 ** 2 * n1 ** 2 - (np.pi / t) ** 2, 0.0))) * (1 + 1e-12)
    hi = K0 * n1 * (1 - 1e-12)
    return brentq(f, lo, hi)


@pytest.mark.parametrize("h", [0.025])
def test_slab_te_tm_against_exact(h):
    """Index varies along y only; Ex is TE, Ey is TM. The x-walls add the same -kx^2 to both, so the difference
    of the eigenvalues equals beta_TE^2 - beta_TM^2 of the slab."""
    n1, n2, t = 3.4, 3.0, 0.5
    ny, nx = 160, 21
    y = (np.arange(ny) - (ny - 1) / 2) * h          # even count: interfaces fall exactly on half-nodes (t/h = 20 inside)
    eps1d = np.where(np.abs(y) < t / 2, n1 ** 2, n2 ** 2)
    eps = np.repeat(eps1d[:, None], nx, axis=1)
    lx, _, ref = V.fundamental(eps, h, K0, "x")
    ly, _, _ = V.fundamental(eps, h, K0, "y")
    num = lx[0] - ly[0]
    bte, btm = slab_beta(n1, n2, t, "TE"), slab_beta(n1, n2, t, "TM")
    exact = bte ** 2 - btm ** 2
    assert exact > 0 and num > 0                      # TE above TM
    assert abs(num - exact) / exact < 0.01          # 0.1 % at h = 0.025; second-order convergence (0.45 % at 0.05)


def test_y_operator_is_transpose_of_x():
    rng = np.random.default_rng(0)
    eps = 9.0 + 0.3 * ndi_smooth(rng.random((31, 37)))
    lx, _, _ = V.fundamental(eps.T, 0.1, K0, "x")
    ly, _, _ = V.fundamental(eps, 0.1, K0, "y")
    assert np.allclose(lx[0], ly[0], rtol=1e-9)


def ndi_smooth(a):
    from scipy import ndimage
    return ndimage.gaussian_filter(a, 3)


def _ellipse_aperture(ax_len, ay_len, theta_deg=0.0):
    """An Aperture object with an elliptical signed-distance-like field (for tests only)."""
    axes = np.arange(-12, 12.0001, 0.25)
    X, Y = np.meshgrid(axes, axes)
    t = np.deg2rad(theta_deg)
    u = np.cos(t) * X + np.sin(t) * Y
    v = -np.sin(t) * X + np.cos(t) * Y
    r = np.sqrt((u / ax_len) ** 2 + (v / ay_len) ** 2)
    F = (1.0 - r) * min(ax_len, ay_len)               # ~ signed distance near the edge (depth = 1)
    ang = np.linspace(0, 2 * np.pi, 400)
    loop = np.c_[ax_len * np.cos(ang) * np.cos(t) - ay_len * np.sin(ang) * np.sin(t),
                 ax_len * np.cos(ang) * np.sin(t) + ay_len * np.sin(ang) * np.cos(t)]
    d = G.MaskDesign(10.0, 1.0, 4.0, 1.0, 0.0)
    return G.Aperture(d, G.Process(depth=1.0, ox_ratio=1.0), axes, F, [loop], np.pi * ax_len * ay_len, 1, 1.0)


KW = dict(dn_eff=0.03, h=0.1, margin=3.0)


def test_circle_has_no_splitting_ellipse_has():
    circ = V.form_birefringence(None, None, aperture=_ellipse_aperture(3.0, 3.0), **KW)
    ell = V.form_birefringence(None, None, aperture=_ellipse_aperture(3.6, 2.5), **KW)
    assert ell.valid and circ.valid
    assert circ.dneff < 0.02 * ell.dneff


def test_ellipse_axis_is_long_axis_and_rotates():
    """The polarization parallel to the long axis (TE-like on the long edges) has the larger n_eff."""
    for th in (0.0, 30.0, 100.0):
        r = V.form_birefringence(None, None, aperture=_ellipse_aperture(3.6, 2.5, th), **KW)
        diff = (r.axis_deg - th + 90) % 180 - 90
        assert abs(diff) < 2.0, (th, r.axis_deg)


def test_rotation_invariance_of_magnitude():
    a = V.form_birefringence(None, None, aperture=_ellipse_aperture(3.6, 2.5, 0.0), **KW)
    b = V.form_birefringence(None, None, aperture=_ellipse_aperture(3.6, 2.5, 30.0), **KW)
    assert abs(a.dneff - b.dneff) / a.dneff < 0.05


def test_grid_convergence():
    ap = _ellipse_aperture(3.6, 2.5, 20.0)
    c = V.form_birefringence(None, None, aperture=ap, dn_eff=0.03, h=0.1)
    f = V.form_birefringence(None, None, aperture=ap, dn_eff=0.03, h=0.07)
    assert abs(c.dneff - f.dneff) / f.dneff < 0.1


def test_mesa_of_target_round_trip():
    """Target-aperture convention: growing a target by D and oxidising D isotropically returns the target for a circle,
    and for a cross keeps the target's outer extent while filling the inner corners (aperture area >= target area)."""
    circ = G.MaskDesign(12.0, 1.0, 12.0, 1.0, 0.0)
    m = G.mesa_of_target(circ, 5.0)
    assert abs(m.L - 22.0) < 1e-9 and abs(m.w - 22.0) < 1e-9 and abs(m.rho - 1.0) < 1e-9
    ap = G.build_aperture(m, G.Process(5.0, 1.0))
    assert abs(ap.area - np.pi * 36.0) / (np.pi * 36.0) < 0.02
    cross = G.MaskDesign(17.5, 0.7, 6.0, 0.0, 30.0)
    ap2 = G.build_aperture(G.mesa_of_target(cross, 5.0), G.Process(5.0, 1.0))
    tgt = G.build_aperture(cross, G.Process(0.0, 1.0))
    assert ap2.area >= tgt.area * 0.99 and ap2.n_components == 1
