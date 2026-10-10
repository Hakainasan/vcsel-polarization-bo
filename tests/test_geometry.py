"""Geometry, oxidation and second-rank descriptors."""
import numpy as np
import pytest
from contourpy import contour_generator

from vcsel_bo import geometry as G

ISO = G.Process(depth=1.0, ox_ratio=1.0)


def _ang(v):
    return (0.5 * np.degrees(np.arctan2(v[1], v[0]))) % 180.0


def _desc(design, process=ISO):
    return G.descriptors(G.build_aperture(design, process))


def _polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _disk_loop(R, process, h=0.125):
    """Eroded-disk outline from the erosion field (validation of the oxidation model against exact geometry)."""
    ax = np.arange(-int(9 / h), int(9 / h) + 1) * h
    X, Y = np.meshgrid(ax, ax)
    t = np.linspace(0, 2 * np.pi, int(2 * np.pi * R / 0.02), endpoint=False)
    bnd = np.column_stack([R * np.cos(t), R * np.sin(t)])
    F = G.erosion_field(bnd, np.hypot(X, Y) < R, X, Y, process)
    return np.asarray(contour_generator(x=ax, y=ax, z=F).lines(0.0)[0])


def test_mesa_area_matches_analytic_cross():
    d = G.MaskDesign(L=17.5, ratio=0.7, w=6.0, rho=0.0, theta=33.0)
    ap = G.build_aperture(d, G.Process(depth=0.0))
    exact = d.L * d.w + d.ratio * d.L * d.w - d.w ** 2
    assert abs(ap.area - exact) / exact < 0.01


def test_rounding_removes_corner_area():
    a0 = G.build_aperture(G.MaskDesign(17.5, 0.7, 6.0, 0.0, 0.0), G.Process(0.0)).area
    a1 = G.build_aperture(G.MaskDesign(17.5, 0.7, 6.0, 1.0, 0.0), G.Process(0.0)).area
    assert a1 < a0
    # four fully rounded tips of radius w/2 each lose (1 - pi/4) r^2 per corner, 8 corners
    assert abs((a0 - a1) - 8 * (1 - np.pi / 4) * 3.0 ** 2) < 0.05 * (a0 - a1)


def test_isotropic_erosion_of_disk_is_exact():
    loop = _disk_loop(5.0, G.Process(1.0, 1.0))
    r = np.hypot(loop[:, 0], loop[:, 1])
    assert abs(r.mean() - 4.0) < 0.01 and r.std() < 0.01
    assert abs(_polygon_area(loop) - np.pi * 16.0) / (np.pi * 16.0) < 0.005


def test_anisotropic_erosion_axes_exact():
    loop = _disk_loop(5.0, G.Process(1.0, 1.44))
    # d_x = 1/1.2, d_y = 1.2  (y oxidises faster -> smaller extent along y)
    assert abs(loop[:, 0].max() - (5 - 1 / 1.2)) < 0.03
    assert abs(loop[:, 1].max() - (5 - 1.2)) < 0.03


def test_narrow_arms_pinch_off():
    ap = G.build_aperture(G.MaskDesign(L=14.0, ratio=0.7, w=3.5, rho=0.5, theta=0.0), G.Process(depth=2.0, ox_ratio=1.0))
    assert ap.area < 20.0 or ap.n_components != 1


@pytest.mark.parametrize("theta", [0.0, 17.0, 45.0, 73.0])
def test_symmetric_cross_has_no_second_rank_anisotropy(theta):
    """A C4-symmetric aperture carries no second-rank tensor: cannot itself induce dichroism/birefringence."""
    ds = _desc(G.MaskDesign(17.5, 1.0, 6.0, 0.5, theta), ISO)
    assert np.linalg.norm(ds.s_mode) < 0.01
    assert np.linalg.norm(ds.s_stress) < 0.01


def test_anisotropic_oxidation_breaks_c4():
    """Symmetric cross + anisotropic oxidation -> non-zero tensor: the fabrication-induced route to switching."""
    ds = _desc(G.MaskDesign(17.5, 1.0, 6.0, 0.5, 0.0), G.Process(1.0, 1.15))
    assert np.linalg.norm(ds.s_mode) > 0.02


def test_tensor_follows_rotation():
    """Orientation of the descriptor equals the design rotation; magnitude is rotation-invariant."""
    mags, smags, errs = [], [], []
    for th in (0, 20, 40, 60, 80, 100, 130, 160):
        ds = _desc(G.MaskDesign(17.5, 0.7, 6.0, 0.5, th))
        mags.append(np.linalg.norm(ds.s_mode))
        smags.append(np.linalg.norm(ds.s_stress))
        e = abs(_ang(ds.s_mode) - th) % 180
        errs.append(min(e, 180 - e))
    mags, smags = np.array(mags), np.array(smags)
    assert max(errs) < 1.0
    assert (mags.max() - mags.min()) / mags.mean() < 0.08          # mode-grid ripple (pk-pk ~6% at h_mode=0.5; shrinks with set_resolution)
    assert (smags.max() - smags.min()) / smags.mean() < 0.01       # pure geometry: exact


def test_quarter_turn_flips_sign():
    """For a C2 shape, rotating by 90 deg flips the elongation vector."""
    a = _desc(G.MaskDesign(17.5, 0.7, 6.0, 0.5, 20.0)).s_mode
    b = _desc(G.MaskDesign(17.5, 0.7, 6.0, 0.5, 110.0)).s_mode
    assert np.dot(a, b) < 0
    assert abs(np.linalg.norm(a) - np.linalg.norm(b)) / np.linalg.norm(a) < 0.05


def test_x_elongated_aperture_gives_positive_x_vector():
    ds = _desc(G.MaskDesign(17.5, 0.7, 6.0, 0.5, 0.0))
    assert ds.s_mode[0] > 0.05 and abs(ds.s_mode[1]) < 0.01


def test_descriptors_are_smooth_in_design_parameters():
    """No pixel-quantisation staircase: second differences in w are tiny compared to the overall trend."""
    ws = np.linspace(5.8, 6.2, 9)
    v = np.array([np.linalg.norm(_desc(G.MaskDesign(17.5, 0.7, w, 0.5, 0.0)).s_stress) for w in ws])
    assert np.max(np.abs(np.diff(v, 2))) < 0.15 * np.abs(v[-1] - v[0]) + 1e-4


def test_resolution_convergence_of_mean_tensor():
    """Mean tensor magnitude agrees between (0.25, 0.5) and (0.125, 0.25) um resolution within 5 %."""
    orig = (G.H_GRID, G.H_MODE)
    vals = {}
    try:
        for hg, hm in ((0.25, 0.5), (0.125, 0.25)):
            G.set_resolution(hg, hm)
            vals[hg] = np.mean([np.linalg.norm(_desc(G.MaskDesign(17.5, 0.7, 6.0, 0.5, th)).s_mode) for th in (0, 30, 60)])
    finally:
        G.set_resolution(*orig)
    assert abs(vals[0.25] - vals[0.125]) / vals[0.125] < 0.05
