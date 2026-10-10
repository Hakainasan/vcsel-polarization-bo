"""Glue: objective callables over the unit cube for a given World, and held-out evaluation."""
from __future__ import annotations

import numpy as np

from . import objective as O
from . import optimization as opt
from .physics import Population, World


class YieldObjective:
    """f(u) -> smoothed yield of design u in `world`, over a fixed device population (common random numbers).

    Keeps a log of hard yields so reports can distinguish the optimiser's surrogate objective from the
    physical (hard) yield.
    """

    def __init__(self, world: World, n_dev: int = 16, pop_seed: int = 0, spec: O.Spec = O.Spec(), backend=None):
        self.world, self.spec, self.backend = world, spec, backend
        self.pop = Population.make(n_dev, pop_seed)
        self.n_calls = 0
        self.log = []

    def __call__(self, u) -> float:
        ev = O.evaluate_design(opt.to_design(u), self.world, self.pop, self.spec, backend=self.backend)
        self.n_calls += 1
        self.log.append((ev.hard_yield, ev.soft_yield))
        return ev.soft_yield


def held_out_yield(u, world: World, n_dev: int = 96, pop_seed: int = 12345, spec: O.Spec = O.Spec(), backend=None):
    """Hard yield of design u on a fresh device population (independent of the one the optimiser saw)."""
    pop = Population.make(n_dev, pop_seed)
    return O.evaluate_design(opt.to_design(u), world, pop, spec, backend=backend)


class PriorObjective:
    """Prior-robust objective: mean smoothed yield over W worlds drawn from the prior (epistemic uncertainty).

    This is the right objective *before* the batch is measured: the constants (native anisotropy, coupling
    coefficients, their signs) are not known, so we ask for designs that work across what is plausible.
    """

    def __init__(self, worlds, n_dev: int = 16, pop_seed: int = 0, spec: O.Spec = O.Spec(), backend=None):
        self.worlds, self.spec, self.backend = list(worlds), spec, backend
        self.pop = Population.make(n_dev, pop_seed)
        self.n_calls = 0
        self.log = []

    def per_world(self, u, hard=False):
        x = opt.to_design(u)
        out = []
        for w in self.worlds:
            ev = O.evaluate_design(x, w, self.pop, self.spec, backend=self.backend)
            out.append(ev.hard_yield if hard else ev.soft_yield)
        return np.array(out)

    def __call__(self, u) -> float:
        x = opt.to_design(u)
        soft, hard = [], []
        for w in self.worlds:
            ev = O.evaluate_design(x, w, self.pop, self.spec, backend=self.backend)
            soft.append(ev.soft_yield)
            hard.append(ev.hard_yield)
        self.n_calls += 1
        self.log.append((float(np.mean(hard)), float(np.mean(soft))))
        return float(np.mean(soft))
