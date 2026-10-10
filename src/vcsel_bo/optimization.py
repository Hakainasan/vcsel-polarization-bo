"""Bayesian optimisation of the mask design, and the baselines it is benchmarked against.

Design space (unit cube u in [0,1]^5  <->  mask parameters)
    L      in [12, 20] um        long-arm length
    ratio  in [0.5, 1.0]         short/long arm ratio
    w      in [3.5, 8] um        arm width
    rho    in [0, 1]             corner rounding fraction (radius = rho * w / 2)
    theta  in [0, 180) deg       rotation from the crystal axis (period 180 deg; the GP sees (cos 2theta, sin 2theta))

The GP is a Matern-5/2 ARD process (one length scale per input feature) with a white-noise term.
ARD is what makes the surrogate *interpretable*: a long length scale means the yield is insensitive
to that design variable. Acquisition: Expected Improvement (Jones, Schonlau & Welch 1998) or UCB,
maximised over the continuous box by multi-start L-BFGS-B seeded from a Sobol candidate cloud.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

LOWER = np.array([12.0, 0.5, 3.5, 0.0, 0.0])
UPPER = np.array([20.0, 1.0, 8.0, 1.0, 180.0])
NAMES = ["L (um)", "arm ratio", "arm width w (um)", "corner rounding rho", "rotation theta (deg)"]
DIM = 5


def to_design(u):
    """Unit cube -> mask parameters (L, ratio, w, rho, theta)."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return LOWER + u * (UPPER - LOWER)


def to_unit(x):
    return (np.asarray(x, float) - LOWER) / (UPPER - LOWER)


def features(u):
    """GP input features: 4 linear coordinates + (cos, sin) of 2*theta mapped to [0,1]."""
    u = np.atleast_2d(u)
    th = np.pi * u[:, 4]                      # 2*theta with theta = 180*u -> 2*theta = 2*pi*u ; theta in rad = pi*u
    return np.column_stack([u[:, :4], 0.5 * (1 + np.cos(2 * th)), 0.5 * (1 + np.sin(2 * th))])


# ----------------------------------------------------------------------------------------------
# GP + acquisition
# ----------------------------------------------------------------------------------------------


def make_gp(seed=0, n_restarts=3):
    k = ConstantKernel(0.1, (1e-3, 10.0)) * Matern(length_scale=np.ones(6) * 0.5, length_scale_bounds=(0.05, 20.0), nu=2.5) \
        + WhiteKernel(1e-3, (1e-6, 1e-1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=n_restarts, random_state=seed)


def fit_gp(U, y, seed=0, n_restarts=3):
    gp = make_gp(seed, n_restarts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        gp.fit(features(U), np.asarray(y, float))
    return gp


def expected_improvement(mu, sd, best, xi=0.01):
    """EI for maximisation. EI >= 0, and EI -> max(mu - best - xi, 0) as sd -> 0."""
    sd = np.maximum(sd, 1e-12)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def ucb(mu, sd, beta=2.0):
    return mu + np.sqrt(beta) * sd


def _acq_value(gp, U, kind, best, xi=0.01, beta=2.0):
    mu, sd = gp.predict(features(U), return_std=True)
    return expected_improvement(mu, sd, best, xi) if kind == "ei" else ucb(mu, sd, beta)


def maximise_acquisition(gp, kind, best, rng, n_cand=2048, n_start=8, exclude=None):
    """Multi-start maximisation of the acquisition over the unit cube."""
    cand = qmc.Sobol(DIM, scramble=True, seed=int(rng.integers(1 << 31))).random(n_cand)
    val = _acq_value(gp, cand, kind, best)
    starts = cand[np.argsort(-val)[:n_start]]
    best_u, best_v = cand[np.argmax(val)], val.max()
    for s in starts:
        res = minimize(lambda u: -_acq_value(gp, u[None, :], kind, best)[0], s, method="L-BFGS-B",
                       bounds=[(0, 1)] * DIM, options=dict(maxiter=40))
        if -res.fun > best_v:
            best_v, best_u = -res.fun, np.clip(res.x, 0, 1)
    return best_u


# ----------------------------------------------------------------------------------------------
# Optimisers: all share  run_*(f, budget, seed) -> Trace
# ----------------------------------------------------------------------------------------------


@dataclass
class Trace:
    U: np.ndarray            # (n, 5) evaluated points, unit cube
    y: np.ndarray            # (n,) objective values (the BO objective, i.e. smoothed yield)
    method: str
    seed: int

    @property
    def best_so_far(self):
        return np.maximum.accumulate(self.y)

    @property
    def best_u(self):
        return self.U[int(np.argmax(self.y))]


def run_random(f: Callable, budget: int, seed: int, **_) -> Trace:
    rng = np.random.default_rng(seed)
    U = rng.random((budget, DIM))
    return Trace(U, np.array([f(u) for u in U]), "random", seed)


def run_sobol(f: Callable, budget: int, seed: int, **_) -> Trace:
    U = qmc.Sobol(DIM, scramble=True, seed=seed).random(budget)
    return Trace(U, np.array([f(u) for u in U]), "sobol", seed)


class _Stop(Exception):
    pass


def run_de(f: Callable, budget: int, seed: int, **_) -> Trace:
    """Differential evolution (population 10, hard budget cut-off): a standard derivative-free evolutionary baseline."""
    from scipy.optimize import differential_evolution

    U, y = [], []

    def wrapped(u):
        if len(y) >= budget:
            raise _Stop
        v = f(np.asarray(u))
        U.append(np.array(u, float))
        y.append(v)
        return -v

    try:
        differential_evolution(wrapped, [(0, 1)] * DIM, popsize=2, maxiter=10_000, tol=0, seed=seed,
                               init="sobol", polish=False, updating="immediate", mutation=(0.5, 1.0),
                               recombination=0.8)
    except _Stop:
        pass
    return Trace(np.array(U), np.array(y), "de", seed)


def run_bo(f: Callable, budget: int, seed: int, n_init: int = 12, kind: str = "ei", n_restarts: int = 2,
           refit_every: int = 1, **_) -> Trace:
    """GP-EI (or GP-UCB) Bayesian optimisation seeded with a scrambled Sobol design."""
    rng = np.random.default_rng(seed)
    U = list(qmc.Sobol(DIM, scramble=True, seed=seed).random(n_init))
    y = [f(u) for u in U]
    gp = None
    while len(y) < budget:
        if gp is None or (len(y) - n_init) % refit_every == 0:
            gp = fit_gp(np.array(U), y, seed=seed + len(y), n_restarts=n_restarts)
        u = maximise_acquisition(gp, kind, max(y), rng)
        U.append(u)
        y.append(f(u))
    return Trace(np.array(U), np.array(y), f"bo_{kind}", seed)


METHODS = {"random": run_random, "sobol": run_sobol, "de": run_de,
           "bo_ei": lambda f, b, s, **k: run_bo(f, b, s, kind="ei", **k),
           "bo_ucb": lambda f, b, s, **k: run_bo(f, b, s, kind="ucb", **k)}


# ----------------------------------------------------------------------------------------------
# Batch (portfolio) selection for a fabrication run
# ----------------------------------------------------------------------------------------------


def greedy_portfolio(Y: np.ndarray, q: int, start=()):
    """Choose q designs maximising  E_w[ max_{j in batch} Y[j, w] ]  (greedy; monotone submodular).

    Y: (n_candidates, n_worlds) yield of each candidate under each *epistemic* sample of the unknown
    constants. This picks a batch that hedges over what is not yet known about the wafer/process, instead of q
    near-copies of one best guess. Greedy selection of a monotone submodular objective is within (1 - 1/e) of the
    optimum (Nemhauser, Wolsey & Fisher 1978). `start` lists candidates that are already in the batch (e.g. the designs
    of an existing mask): the q new designs are then chosen to complement them.
    Returns (indices of the q new designs, expected batch score after each addition).
    """
    n, W = Y.shape
    chosen_all = list(start)
    cur = np.max(Y[list(start)], axis=0) if len(start) else np.zeros(W)
    chosen, scores = [], []
    for _ in range(q):
        gain = np.maximum(Y, cur[None, :]).mean(axis=1) - cur.mean()
        gain[chosen_all] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        chosen_all.append(j)
        cur = np.maximum(cur, Y[j])
        scores.append(float(cur.mean()))
    return chosen, scores
