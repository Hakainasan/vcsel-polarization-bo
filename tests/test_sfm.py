"""Spin-flip model: formulation consistency, analytic Jacobian, fixed points vs time domain, symmetry."""
import numpy as np
import pytest

from vcsel_bo import sfm

C = sfm.SFMConstants()


def _rand_state(rng):
    return rng.normal(size=6) * np.array([.5, .5, .5, .5, .3, .2]) + np.array([0, 0, 0, 0, 1, 0])


def test_linear_and_circular_basis_formulations_agree():
    rng = np.random.default_rng(0)
    for _ in range(100):
        y = _rand_state(rng)
        mu = rng.uniform(1, 2)
        m11 = rng.normal() + 1j * rng.normal() * 20
        m12 = rng.normal() * .5 + 1j * rng.normal() * 10
        assert np.allclose(sfm.rhs(y, mu, m11, m12, C), sfm.rhs_circular_reference(y, mu, m11, m12, C), atol=1e-8)


def test_analytic_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    for _ in range(30):
        u = np.concatenate([_rand_state(rng)[:4] * 1.2, [rng.uniform(.8, 1.3), rng.normal() * .2, rng.normal() * 5]])[None]
        mu = np.array([rng.uniform(1, 2)])
        m11 = np.array([rng.normal() + 1j * rng.normal() * 20])
        m12 = np.array([rng.normal() * .5 + 1j * rng.normal() * 8])
        Ja = sfm.jacobian(u, mu, m11, m12, C)[0]
        f = lambda uu: sfm._rot_rhs(uu, mu, m11, m12, C)
        Jn = np.array([((f(u + np.eye(7)[j] * 1e-6) - f(u - np.eye(7)[j] * 1e-6)) / 2e-6)[0] for j in range(7)]).T
        assert np.max(np.abs(Ja - Jn)) / max(1, np.max(np.abs(Jn))) < 1e-6


def _branch(mu, m11, m12, which):
    mu, m11, m12 = np.array([mu]), np.array([m11], complex), np.array([m12], complex)
    u0 = sfm.linear_mode_guess(mu, m11, m12, C, which)
    u, ok = sfm.newton_fixed_point(u0, mu, m11, m12, C)
    lam = sfm.stability(u, mu, m11, m12, C)
    return u, bool(ok[0]), float(lam[0]), sfm.stokes(u)


def test_aligned_lp_steady_state_is_analytic():
    """x-mode: D = 1 + gamma_a/kappa, I = mu/D - 1."""
    ga, gp, mu = 0.5, 20.0, 1.6
    u, ok, lam, st = _branch(mu, ga + 1j * gp, 0.0, 1)      # branch 1 = higher-loss = x mode
    assert ok
    assert abs(u[0, 4] - (1 + ga / C.kappa)) < 1e-9
    assert abs(st["intensity"][0] - (mu / (1 + ga / C.kappa) - 1)) < 1e-9
    assert abs(st["P"][0] - 1.0) < 1e-6


def test_switching_requires_same_sign_of_dichroism_and_birefringence():
    """Literature-known SFM asymmetry: switch to the gain-disfavoured mode only for one relative sign."""
    mu = 1.6
    _, _, lam_fav_same, _ = _branch(mu, 0.5 + 20j, 0, 0)
    _, _, lam_dis_same, _ = _branch(mu, 0.5 + 20j, 0, 1)
    _, _, lam_fav_opp, _ = _branch(mu, 0.5 - 20j, 0, 0)
    assert lam_fav_same > 0 and lam_dis_same < 0            # favoured unstable, disfavoured stable -> switched
    assert lam_fav_opp < 0                                  # opposite class: favoured mode stays stable


def test_switching_current_grows_with_dichroism():
    mus = np.linspace(1.02, 2.4, 60)
    def mu_sw(ga):
        u0 = sfm.linear_mode_guess(mus, np.full(mus.shape, ga + 20j, dtype=complex), np.zeros_like(mus, complex), C, 0)
        u, ok = sfm.newton_fixed_point(u0, mus, np.full(mus.shape, ga + 20j, dtype=complex), np.zeros_like(mus, complex), C)
        lam = sfm.stability(u, mus, np.full(mus.shape, ga + 20j, dtype=complex), np.zeros_like(mus, complex), C)
        i = np.where(~(ok & (lam < 0)))[0]
        return mus[i[0]] if len(i) else np.inf
    vals = [mu_sw(g) for g in (0.2, 0.5, 1.0, 2.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_axis_swap_symmetry():
    """Swapping x<->y maps (m11, m12) -> (-m11, m12); the gain-favoured branch is invariant."""
    mus = np.array([1.3, 1.7, 2.0])
    for m11, m12 in ((0.5 + 20j, 0.3 + 4j), (1.2 + 35j, -0.4 + 2j)):
        r = []
        for s in (1.0, -1.0):
            M11 = np.full(3, s * m11, complex)
            M12 = np.full(3, m12, complex)
            u0 = sfm.linear_mode_guess(mus, M11, M12, C, 0)
            u, ok = sfm.newton_fixed_point(u0, mus, M11, M12, C)
            r.append(sfm.stability(u, mus, M11, M12, C))
        assert np.allclose(r[0], r[1], atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("mu", [1.2, 1.6, 2.0])
def test_time_domain_converges_to_predicted_stable_state(mu):
    m11, m12 = 0.5 + 20j, 0.0
    stable = []
    for which in (0, 1):
        u, ok, lam, st = _branch(mu, m11, m12, which)
        if ok and lam < -1e-3:
            stable.append(st["P"][0])
    sol = sfm.integrate([1e-2, 0, 1e-2, 0, 1, 0], mu, m11, m12, C, t_end=60)
    st = sfm.stokes(np.concatenate([sol.y[:, -1], [0.0]])[None])
    assert any(abs(st["P"][0] - p) < 0.02 for p in stable)
