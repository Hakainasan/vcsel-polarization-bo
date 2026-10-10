"""Acquisition functions, GP surrogate, baselines' budget accounting, portfolio selection."""
import numpy as np
import pytest

from vcsel_bo import optimization as opt


def test_unit_design_roundtrip():
    u = np.random.default_rng(0).random((10, 5))
    assert np.allclose(opt.to_unit(opt.to_design(u)), u)


def test_features_are_periodic_in_theta():
    u0 = np.array([[0.3, 0.2, 0.9, 0.4, 0.0]])
    u1 = u0.copy(); u1[0, 4] = 1.0          # theta = 180 deg
    assert np.allclose(opt.features(u0), opt.features(u1), atol=1e-12)


def test_ei_properties():
    mu = np.linspace(-1, 2, 50)
    sd = np.full(50, 0.3)
    ei = opt.expected_improvement(mu, sd, best=0.5)
    assert (ei >= 0).all()
    assert (np.diff(ei) > 0).all()                                 # increasing in the mean
    sds = np.linspace(0.01, 2, 50)
    assert (np.diff(opt.expected_improvement(np.full(50, 0.0), sds, best=0.5)) > 0).all()   # increasing in sd
    # zero variance limit
    assert abs(opt.expected_improvement(np.array([1.0]), np.array([1e-14]), 0.4, xi=0.0)[0] - 0.6) < 1e-6
    assert opt.expected_improvement(np.array([0.0]), np.array([1e-14]), 0.4, xi=0.0)[0] < 1e-9


def _smooth(u):
    u = np.asarray(u)
    return float(np.exp(-3 * np.sum((u[:4] - 0.5) ** 2)) * (0.6 + 0.4 * np.cos(2 * np.pi * u[4])))


def test_gp_generalises_on_smooth_function():
    rng = np.random.default_rng(0)
    U = rng.random((80, 5))
    y = np.array([_smooth(u) for u in U])
    gp = opt.fit_gp(U, y, seed=0, n_restarts=2)
    Ut = rng.random((200, 5))
    yt = np.array([_smooth(u) for u in Ut])
    pred = gp.predict(opt.features(Ut))
    r2 = 1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)
    assert r2 > 0.85


@pytest.mark.parametrize("name", ["random", "sobol", "de", "bo_ei", "bo_ucb"])
def test_every_method_respects_the_budget(name):
    calls = []
    def f(u):
        calls.append(1)
        return _smooth(u)
    tr = opt.METHODS[name](f, 16, 0)
    assert len(tr.y) == 16 and len(calls) == 16
    assert tr.U.shape == (16, 5) and ((tr.U >= 0) & (tr.U <= 1)).all()
    assert (np.diff(tr.best_so_far) >= 0).all()


def test_bo_beats_random_on_a_smooth_problem():
    b = np.mean([opt.run_bo(_smooth, 30, s, n_init=10).best_so_far[-1] for s in range(3)])
    r = np.mean([opt.run_random(_smooth, 30, s).best_so_far[-1] for s in range(3)])
    assert b > r


def test_greedy_portfolio_hedges():
    Y = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6]])
    idx, scores = opt.greedy_portfolio(Y, 2)
    assert len(set(idx)) == 2
    assert scores[1] >= scores[0]
    assert scores[1] >= Y.mean(axis=1).max() - 1e-12
    assert abs(scores[1] - 0.8) < 1e-12          # 0.6-design + one specialist: mean of [1,0.6] or [0.6,1]
