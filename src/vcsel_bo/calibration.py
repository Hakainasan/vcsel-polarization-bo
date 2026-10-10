"""Calibrating the wafer constants from a measured batch (approximate Bayesian computation).

The design study is only as good as the constants in `physics.World` (native dichroism/birefringence and the
two aperture-coupling coefficients, including their *signs*). Those cannot be derived from first principles
here; they are identified from the first fabricated batch.

Measurement protocol (per fabricated design, N devices; polarization-resolved L-I sweeps):
    * n_switch  number of devices with exactly one polarization switch in the operating range
    * mu_sw     median switching current in units of the threshold current (over the switchers), if any
    * az_low    circular-mean azimuth of the low-current polarization state (deg, in the crystal frame)

`abc_posterior` importance-weights draws from the prior by how well their *simulated* summaries reproduce the
measured ones (Gaussian-kernel ABC; Beaumont et al. 2002). It returns weights, an effective sample size and
posterior summaries. Nothing here is fitted to real data yet: the module is exercised on synthetic data in
`tests/test_calibration.py` to prove that the identification works *in principle*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import objective as O
from . import optimization as opt
from . import sfm
from .physics import Population, Prior, World, sample_world


@dataclass
class DesignObservation:
    x: np.ndarray                 # mask design (L, ratio, w, rho, theta)
    n_dev: int
    n_switch: int
    mu_sw: float | None = None    # median switching current of the switchers (I/I_th)
    az_low: float | None = None   # mean azimuth of the low-current state (deg), or None if not measured


def simulate_summary(x, world: World, n_dev: int = 48, pop_seed: int = 0, spec: O.Spec = O.Spec()):
    """Model prediction of the measured summary statistics for a design in a world (with device scatter)."""
    pop = Population.make(n_dev, pop_seed)
    ev = O.evaluate_design(x, world, pop, spec)
    n = 3 * pop.n
    p = float(ev.frac_switch)
    return dict(p_switch=p, mu_sw=ev.mu_sw_median, n=n)


def _summary_distance(obs: list[DesignObservation], sims: list[dict], sigma_mu=0.08, sigma_p_floor=0.05):
    d2 = 0.0
    for o, s in zip(obs, sims):
        p_obs = o.n_switch / max(o.n_dev, 1)
        sp = np.sqrt(max(p_obs * (1 - p_obs), 0.02) / o.n_dev + sigma_p_floor ** 2)
        d2 += ((s["p_switch"] - p_obs) / sp) ** 2
        if o.mu_sw is not None and np.isfinite(s["mu_sw"]) and o.n_switch > 0:
            d2 += ((s["mu_sw"] - o.mu_sw) / sigma_mu) ** 2
        elif o.mu_sw is not None and o.n_switch > 0 and not np.isfinite(s["mu_sw"]):
            d2 += 9.0    # observed switchers but simulated none: penalise like a 3-sigma miss
    return d2


def abc_posterior(obs: list[DesignObservation], n_samples: int = 300, prior: Prior = Prior(), base: World | None = None,
                  seed: int = 0, temperature: float = 1.0, spec: O.Spec = O.Spec(), n_dev_sim: int = 48,
                  worlds: list[World] | None = None):
    """Importance-weighted ABC posterior over worlds. Returns dict(worlds, weights, ess, dist2).

    `worlds`: candidate worlds to weight. Default: `n_samples` draws from `prior`. To update the literature-anchored belief
    with a measured batch, pass the accepted worlds of `results/anchored_worlds.json` (`literature.worlds_from_records`):
    the literature-consistent set then acts as the prior and far fewer simulations are wasted on worlds that the
    published experiments already exclude.
    """
    rng = np.random.default_rng(seed)
    if worlds is None:
        worlds = [sample_world(rng, prior, base) for _ in range(n_samples)]
    else:
        worlds = list(worlds)
        n_samples = len(worlds)
    d2 = np.zeros(n_samples)
    for i, w in enumerate(worlds):
        sims = [simulate_summary(o.x, w, n_dev=n_dev_sim, pop_seed=1000 + j, spec=spec) for j, o in enumerate(obs)]
        d2[i] = _summary_distance(obs, sims)
    logw = -0.5 * d2 / temperature
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    return dict(worlds=worlds, weights=w, ess=float(ess), dist2=d2)


def posterior_summary(post):
    """Weighted posterior means and sign probabilities of the world constants."""
    W = post["weights"]
    arr = lambda f: np.array([f(w) for w in post["worlds"]])
    out = {}
    for name, f in (("d0", lambda w: w.d0), ("b0", lambda w: w.b0), ("k_d", lambda w: w.k_d), ("k_b", lambda w: w.k_b)):
        v = arr(f)
        m = float((W * v).sum())
        sd = float(np.sqrt((W * (v - m) ** 2).sum()))
        out[name] = (m, sd)
    out["P(k_d>0)"] = float((W * (arr(lambda w: w.k_d) > 0)).sum())
    out["P(b0>0)"] = float((W * (arr(lambda w: w.b0) > 0)).sum())
    out["P(k_b>0)"] = float((W * (arr(lambda w: w.k_b) > 0)).sum())
    out["ESS"] = post["ess"]
    return out
