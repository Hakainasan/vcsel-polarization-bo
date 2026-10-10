"""Yield objective: sanity, invariances and margin consistency (reference world = one literature-consistent world)."""
import numpy as np

from vcsel_bo import objective as O
from vcsel_bo import physics as P
from vcsel_bo import sfm

C = sfm.SFMConstants()
POP = P.Population.make(16, 0)
WORLD = P.World()          # default world: reproduces the enforced anchors of `literature` (see tests/test_literature.py)


def test_symmetric_designs_do_not_switch():
    """C4-symmetric apertures carry only the native anisotropy -> no switching, as observed experimentally."""
    ev = O.evaluate_design([17.5, 1.0, 6.0, 0.5, 0.0], WORLD, POP)
    assert ev.hard_yield == 0.0
    assert ev.frac_switch == 0.0


def test_yield_is_deterministic_common_random_numbers():
    x = [17.5, 0.7, 6.0, 0.5, 20.0]
    assert O.evaluate_design(x, WORLD, POP).soft_yield == O.evaluate_design(x, WORLD, POP).soft_yield


def test_180_degree_periodicity():
    a = O.evaluate_design([17.5, 0.7, 6.0, 0.5, 80.0], WORLD, POP)
    b = O.evaluate_design([17.5, 0.7, 6.0, 0.5, 260.0], WORLD, POP)
    assert abs(a.hard_yield - b.hard_yield) < 0.1


def test_long_arm_along_native_axis_reinforces_and_orthogonal_arm_compensates():
    """Axis convention of the papers: x = natively favoured axis; a long arm along x strengthens the built-in
    anisotropy (no switching), a long arm along y compensates it (switching)."""
    reinforcing = O.evaluate_design([17.5, 0.7, 6.0, 0.5, 0.0], WORLD, POP).hard_yield
    compensating = O.evaluate_design([17.5, 0.7, 6.0, 0.5, 90.0], WORLD, POP).hard_yield
    assert reinforcing < 0.02 and compensating > 0.6


def test_mirror_symmetry_about_the_native_axis():
    """Reflecting the mask about the native axis (theta -> 180 - theta) must not change the physics (up to the finite
    device population)."""
    for th in (60.0, 75.0):
        a = O.evaluate_design([17.5, 0.7, 6.0, 0.5, th], WORLD, POP).hard_yield
        b = O.evaluate_design([17.5, 0.7, 6.0, 0.5, 180.0 - th], WORLD, POP).hard_yield
        assert abs(a - b) < 0.2


def test_margins_and_hard_yield_are_consistent():
    ev = O.evaluate_design([17.5, 0.7, 6.0, 0.5, 20.0], WORLD, POP)
    w = P.GH_WEIGHTS[:, None] / POP.n
    assert abs(((ev.margins > 0) * w).sum() - ev.hard_yield) < 1e-12
    assert 0.0 <= ev.soft_yield <= 1.0


def test_anisotropy_diagnostics_are_reported_and_shape_compensates_native_dichroism():
    """Long arm along the native axis strengthens the net dichroism, across it mitigates it (Yuan et al.): the net
    normalised dichroism must fall monotonically with the rotation and cross the circle's value on the way."""
    circle = O.evaluate_design([17.0, 1.0, 17.0, 1.0, 0.0], WORLD, POP).diag["aniso"]["d_norm"].mean()
    vals = [O.evaluate_design([17.5, 0.7, 6.0, 0.5, th], WORLD, POP).diag["aniso"]["d_norm"].mean() for th in (0.0, 30.0, 60.0, 90.0)]
    assert vals[0] > circle > vals[-1]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def _classify_aligned(ga, gp, n=1):
    m11 = np.full(n, ga + 1j * gp, complex)
    m12 = np.zeros(n, complex)
    br = O.branch_states(m11, m12, O.MUS, C)
    return O.classify(br, O.MUS, O.Spec())


def test_partially_compensated_device_switches_inside_window():
    m, f = _classify_aligned(1.2, 20.0)
    assert f["n_sw"][0] == 1
    assert 1.4 < f["mu_sw"][0] < 1.7
    assert m[0] > 0


def test_native_strong_dichroism_does_not_switch_in_window():
    m, f = _classify_aligned(3.5, 20.0)
    assert f["n_sw"][0] == 0 and m[0] < 0


def test_overcompensation_leaves_switching_class():
    """Dichroism flipped past zero (opposite sign to birefringence): no clean switch."""
    m, f = _classify_aligned(-1.2, 20.0)
    assert f["n_sw"][0] == 0 and m[0] < 0


def test_low_birefringence_does_not_switch():
    m, f = _classify_aligned(0.8, 4.0)
    assert f["n_sw"][0] == 0 and m[0] < 0
