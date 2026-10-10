"""Literature anchors: the encoded outcomes, the staged screening, the sampler and the citation bookkeeping."""
from pathlib import Path

import numpy as np

from vcsel_bo import literature as L
from vcsel_bo import physics as P

ROOT = Path(__file__).resolve().parents[1]


def test_default_world_reproduces_every_enforced_anchor():
    rep = L.anchor_report(P.World())
    failed = [k for k, (ok, _) in rep["anchors"].items() if not ok and k != "A6_symmetric_cross_switches_more_than_circle"]
    assert rep["ok_enforced"] and not failed


def test_designs_reported_by_the_papers_are_present():
    assert set(L.APL126_SEVEN) <= set(L.DESIGNS)
    assert L.DESIGNS["x07_60"].theta == 60.0 and L.DESIGNS["x07_60"].ratio == 0.7
    assert len(L.MASK_DESIGNS) == 16
    assert abs(L.MASK_DESIGNS["B45"].corner_radius - 2.25) < 1e-12        # family B: round short-arm radius 2.25 um


def test_world_without_shape_coupling_fails_the_rotation_anchor():
    """If the aperture shape cannot move the polarization (k_d ~ 0), rotating by 90 deg cannot flip the initial state."""
    rep = L.anchor_report(P.World(k_d=-0.1, k_b=-0.1))
    assert not rep["anchors"]["A3_rotation_by_90_flips_initial_polarization"][0]


def test_world_with_wrong_sign_coupling_fails():
    """Elongation that DISfavours the long axis contradicts 'the initial polarization follows the elongated arm'."""
    rep = L.anchor_report(P.World(k_d=+21.0))
    assert not rep["ok_enforced"]


def test_staged_screen_agrees_with_full_report():
    rng = np.random.default_rng(11)
    pop = P.Population.make(24, 777)
    worlds = [P.sample_world(rng) for _ in range(4)] + [P.World()]
    for w in worlds:
        ok, failed, n_design = L.screen_world(w, pop)
        assert ok == L.anchor_report(w, pop)["ok_enforced"]
        need = {d for a in L.ANCHORS if a.enforced for d in a.designs}
        # the cheapest anchor (two designs) is checked first; an accepted world had to be evaluated on every design
        assert (failed is None) == ok and n_design >= 2 and (not ok or n_design == len(need))


def test_sampler_is_reproducible_and_accepted_worlds_pass_the_report():
    a = L.sample_anchored_worlds(24, seed=5, n_jobs=1, chunk=12)
    b = L.sample_anchored_worlds(24, seed=5, n_jobs=1, chunk=12)
    assert a["accepted"] == b["accepted"] and a["n_draws"] == 24
    for w in L.worlds_from_records(a["accepted"]):
        assert L.anchor_report(w)["ok_enforced"]


def test_thresholds_are_ordered_from_loose_to_strict():
    assert L.LOOSE["most"] < L.PRIMARY["most"] < L.STRICT["most"]
    assert L.LOOSE["seldom"] > L.PRIMARY["seldom"] > L.STRICT["seldom"]
    assert L.LOOSE["part"] < L.PRIMARY["part"] < L.STRICT["part"]


def test_every_anchor_names_sources_that_exist_in_the_bibliography():
    bib = (ROOT / "docs" / "references.bib").read_text()
    for a in L.ANCHORS:
        for key in a.source.replace(";", " ").split():
            assert f"{{{key}," in bib, (a.key, key)


def test_sampler_with_another_base_world_keeps_the_process_and_stays_reproducible():
    """Re-conditioning under a changed process assumption: the base world's oxidation ratio must reach the screening
    and be recorded; the default base must reproduce the earlier behaviour."""
    from dataclasses import replace

    from vcsel_bo import geometry as G
    base = replace(P.World(), process=G.Process(1.0, 1.0))
    a = L.sample_anchored_worlds(8, seed=3, n_jobs=1, chunk=8, base=base)
    b = L.sample_anchored_worlds(8, seed=3, n_jobs=1, chunk=8, base=base)
    assert a["accepted"] == b["accepted"] and a["base"]["process_ox_ratio"] == 1.0
    d = L.sample_anchored_worlds(8, seed=3, n_jobs=1, chunk=8)
    assert d["base"]["process_ox_ratio"] == P.World().process.ox_ratio
