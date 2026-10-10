"""Calibration is exercised on SYNTHETIC data only (a known world generates the 'measurements')."""
import numpy as np

from vcsel_bo import calibration as C
from vcsel_bo import objective as O
from vcsel_bo import optimization as opt
from vcsel_bo.physics import World


def _obs(world, designs, n_dev=48):
    out = []
    for x in designs:
        s = C.simulate_summary(x, world, n_dev=n_dev, pop_seed=7)
        n = s["n"]
        k = int(round(s["p_switch"] * n_dev))
        out.append(C.DesignObservation(x=x, n_dev=n_dev, n_switch=k,
                                       mu_sw=None if not np.isfinite(s["mu_sw"]) else float(s["mu_sw"])))
    return out


def test_summary_is_deterministic_and_bounded():
    x = np.array([17.0, 0.7, 6.0, 0.5, 0.0])
    a = C.simulate_summary(x, World(), n_dev=16, pop_seed=1)
    b = C.simulate_summary(x, World(), n_dev=16, pop_seed=1)
    assert a["p_switch"] == b["p_switch"] and 0 <= a["p_switch"] <= 1


def test_abc_weights_normalised_and_truth_scores_better_than_average():
    truth = World()
    designs = [opt.to_design(u) for u in np.random.default_rng(3).random((6, 5))]
    obs = _obs(truth, designs)
    post = C.abc_posterior(obs, n_samples=24, seed=1, n_dev_sim=24)
    assert abs(post["weights"].sum() - 1) < 1e-9 and post["ess"] >= 1
    d_truth = C._summary_distance(obs, [C.simulate_summary(o.x, truth, 24, 1000 + j) for j, o in enumerate(obs)])
    assert d_truth <= np.median(post["dist2"])
    s = C.posterior_summary(post)
    assert 0 <= s["P(k_d>0)"] <= 1


def test_abc_accepts_an_explicit_world_set():
    """The literature-consistent worlds can be passed as the prior of the update (a measured batch reweights them)."""
    truth = World(d0=2.0, b0=25.0, k_d=-20.0, k_b=-30.0)
    others = [World(d0=2.0 * f, b0=25.0, k_d=-20.0 * f, k_b=-30.0) for f in (0.5, 0.75, 1.5, 2.0)]
    designs = [np.array([17.5, 0.7, 6.0, 0.5, th]) for th in (0.0, 60.0, 90.0)]
    obs = _obs(truth, designs, n_dev=24)
    post = C.abc_posterior(obs, worlds=[truth] + others, seed=0, n_dev_sim=24)
    assert len(post["worlds"]) == 5 and abs(post["weights"].sum() - 1) < 1e-9
    assert post["weights"][0] >= post["weights"].min()
