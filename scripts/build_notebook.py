"""Builds notebooks/VCSEL_Polarization_BO.ipynb (then execute it with nbconvert; see README).

The notebook reads the cached results of the pipeline (results/*.json, figures/*.png) and runs a few live calls (seconds).
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    M("""# Bayesian optimization of VCSEL oxide-aperture geometry for polarization-switching photonic Ising machines
**Status: a literature-anchored simulation study.** The constants of the model are *not measured*: they are constrained only by the qualitative outcomes
that the fabricated-device papers report (Yuan et al. 2025), by approximate Bayesian computation. Every number below is a **model prediction conditional on those constraints**;
nothing is validated on a device of this project. The notebook reads the cached results of `scripts/run_all.sh` and runs a few live calls that take seconds.

Pipeline: mask parameters (L, arm ratio, w, corner rounding, rotation) -> exact drawn cross -> anisotropic lateral oxidation (eroded aperture) -> second-rank boundary tensors
-> dichroism / birefringence = native + coupling x tensor -> spin-flip model (fixed points + stability) -> yield (checklist) -> Bayesian optimisation and batch (portfolio) design."""),
    C("""import sys, json, warnings
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from IPython.display import Image, display
warnings.filterwarnings('ignore')
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
from vcsel_bo import geometry as G, physics as P, objective as O, optimization as opt, sfm, literature as L, problem as PR
np.set_printoptions(precision=3, suppress=True)
RES = ROOT / 'results'
def J(name): return json.loads((RES / name).read_text())"""),
    M("## 1. Geometry: drawn mesa, oxide aperture, second-rank descriptors"),
    C("""fig, axs = plt.subplots(1, 4, figsize=(11, 3))
proc = G.Process()   # placeholder oxidation depth / anisotropy
for ax, th in zip(axs, (0, 30, 60, 90)):
    d = G.MaskDesign(17.5, 0.7, 6.0, 0.5, th)
    ap = G.build_aperture(d, proc)
    m = G.mesa_boundary(d); ax.plot(m[:, 0], m[:, 1], '.', c='0.75', ms=0.5)
    for lp in ap.loops: ax.plot(lp[:, 0], lp[:, 1], 'k')
    ds = G.descriptors(ap)
    ax.set_aspect('equal'); ax.set_title(f'rot {th} deg, s_mode=({ds.s_mode[0]:.2f},{ds.s_mode[1]:.2f})', fontsize=8)
plt.show()
sym = G.descriptors(G.build_aperture(G.MaskDesign(17.5, 1.0, 6.0, 0.5, 0.0), G.Process(1.0, 1.0)))
print('C4 symmetric cross, isotropic oxidation: |s_mode| =', np.linalg.norm(sym.s_mode), '(exactly 0 by symmetry)')"""),
    M("## 2. Spin-flip model: stable branches, and validation against time-domain integration"),
    C("""Cst = sfm.SFMConstants()
mus = np.linspace(1.05, 2.25, 49)
fig, ax = plt.subplots(figsize=(5, 3))
for ga in (1.0, 2.0, 3.0):
    for which, ls in ((0, '-'), (1, '--')):
        m11 = np.full(mus.shape, ga + 20j, dtype=complex); m12 = np.zeros(mus.shape, complex)
        u0 = sfm.linear_mode_guess(mus, m11, m12, Cst, which)
        u, ok = sfm.newton_fixed_point(u0, mus, m11, m12, Cst)
        lam = sfm.stability(u, mus, m11, m12, Cst)
        ax.plot(mus[lam < -1e-3], sfm.stokes(u)['P'][lam < -1e-3], ls, marker='.', ms=2, label=f'g_a={ga} branch {which}' if which == 0 else None)
ax.set_xlabel('I/I_th'); ax.set_ylabel('P = S1 of stable states'); ax.legend(fontsize=7); plt.show()
import csv
rows = list(csv.DictReader(open(RES / 'sfm_validation.csv')))
print(len(rows), 'sweep points; columns:', list(rows[0].keys()))"""),
    M("""## 3. Anchoring the model to the published devices
The papers report outcomes per design, not numbers. Each outcome is a yes/no test on a simulated device population; worlds (constants of the model, including signs) are drawn from a broad prior and kept
only if they pass all enforced tests (rejection ABC). Details and the sources of each anchor: `docs/LITERATURE_ANCHORS.md`, supplement Sec. 3."""),
    C("""A = J('anchored_worlds.json')
print(f"{A['n_accepted']} of {A['n_draws']} prior draws accepted ({100*A['acceptance']:.1f} %)")
for k, v in A['posterior'].items():
    print(f"{k:4s} median {v['median']:8.2f}   5-95 %: [{v['q05']:.2f}, {v['q95']:.2f}]   share > 0: {v['frac_positive']:.2f}")
print('share of prior draws passing each anchor alone:', {k.split('_')[0]: round(v, 2) for k, v in A['marginal']['per_anchor'].items()})
pop = P.Population.make(24, 777)
rep = L.anchor_report(P.World(), pop)
print('reference world passes all enforced anchors:', rep['ok_enforced'])
display(Image(str(ROOT / 'figures' / 'anchored_worlds.png')))"""),
    M("""Not reproduced by the model (reported, not tuned away): switching of the 30 degree cruciform and a symmetric cross that switches more often than a circle; see supplement Sec. 3."""),
    M("## 4. Predicted yield of published and own designs (mean over literature-consistent worlds)"),
    C("""D = J('literature_designs.json')
print(f"{D['n_worlds']} worlds")
for k, s in D['designs'].items():
    print(f"{k:10s} mean yield {s['mean_yield']:.2f}  [10%,90%] [{s['q10']:.2f},{s['q90']:.2f}]  P(yield>=0.5) {s['p_yield_ge_half']:.2f}  mean P(switch) {s['mean_p_switch']:.2f}")
print('best of the seven designs of APL 126, share of worlds:', {k: round(v, 2) for k, v in D['apl126_seven_top_frequency'].items()},
      ' no working design among the seven:', round(D['apl126_seven_no_working_design_fraction'], 2))
display(Image(str(ROOT / 'figures' / 'rotation_scan.png'))); display(Image(str(ROOT / 'figures' / 'designs.png')))"""),
    C("""W = J('window_dependence.json')
print('1:0.7 cross, mean yield at rotations', W['thetas'], 'for different switching-current windows')
for k, v in W['windows'].items():
    print(f"{k:20s}", np.round(v['mean_yield'], 2), ' best', v['best_theta'], ' P(60>90)', round(v['share_worlds_60_beats_90'], 2), ' P(90>60)', round(v['share_worlds_90_beats_60'], 2))"""),
    M("## 5. Optimiser benchmark (cached; `scripts/run_benchmark.py`)"),
    C("""S = J('benchmark_stats.json')
for r in S:
    if 'method' in r: print(f"{r['problem']} {r['method']:7s} held-out yield {r['mean_held_out']:.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  n={r['n']}")
print()
for r in S:
    if 'test' in r and r['test'] in ('bo_ei - random','bo_ei - sobol','bo_ei - de','bo_ucb - random','bo_ucb - sobol','bo_ucb - de'):
        print(f"{r['problem']} {r['test']:16s} mean diff {r['mean_diff']:+.3f}  Wilcoxon p {r['wilcoxon_p']:.3f}  pairs {r['n_pairs']}")
display(Image(str(ROOT / 'figures' / 'benchmark.png')))"""),
    M("## 6. Fabrication batch (portfolio) and its checks"),
    C("""B = J('batch_summary.json')['test']
print({k: (round(v['E_best_yield'], 3), round(v['P_some_design_ge_half'], 3)) for k, v in B.items() if isinstance(v, dict) and 'E_best_yield' in v})
R = J('robustness_extras.json')
print('designs of the portfolio on a wall of the search box:', sum(1 for d in R['portfolio_info'] if d['on_wall']), 'of', R['q'])
print('fresh device population:', {k: round(v['E_best_yield'], 3) for k, v in R['test_fresh_devices'].items() if isinstance(v, dict) and 'E_best_yield' in v})
O2 = J('other_worlds_check.json')
print('expected best-of-batch yield, 32 test worlds vs', O2['n_other_worlds'], 'further accepted worlds (never used for selection):',
      {k: (round(v['test'], 2), round(v['other'], 2)) for k, v in O2['sets'].items()})
print(open(RES / 'batch_design.csv').read())
display(Image(str(ROOT / 'figures' / 'batch_shapes.png')))"""),
    M("## 7. Sensitivity to the assumptions"),
    C("""s = J('sensitivity.json')
for k, v in s.items():
    if isinstance(v, dict) and isinstance(v.get('portfolio'), dict): print(f'{k:62s}', {a: round(b['E_best'], 2) for a, b in v.items()})"""),
    M("## 8. Testable predictions for the author's mask (2 families x 8 rotations)"),
    C("""Mk = J('mask_predictions.json')
for k, t in Mk['predictions'].items():
    y = t['yield_']; print(f"{k:5s} yield median {y[1]:.2f} [{y[0]:.2f}, {y[2]:.2f}]   share lasing first on x {t['p_x_first'][1]:.2f}   share that switch {t['p_switch'][1]:.2f}")
print('devices that SWITCH (at least one flip in the sweep):', json.dumps(Mk['group_switch_counts'], indent=1))
print('devices that also meet the whole yield specification:', json.dumps(Mk['group_pass_counts'], indent=1))
Fr = J('mask_family_a_rounded.json')
print('if family A had fully rounded arm tips (rho = 1):', {g: round(v['mean'], 1) for g, v in Fr['rounded_family_A'].items()})"""),
    M("""## 9. Live end-to-end calls (seconds)
The mini optimisation below (16 evaluations, one seed) only demonstrates the API. It is far smaller than the benchmark of Sec. 5 (48 evaluations, 40 world-seed pairs), and a design picked by hand from the literature
(rotation 90 degrees) scores higher than what 16 evaluations find: Bayesian optimisation is a search tool inside the model, it does not replace the physical prior knowledge that the literature anchors provide."""),
    C("""pop = P.Population.make(16, 0)
w = P.World()   # reference world: one literature-consistent set of constants (not a fit)
for th in (0, 60, 90):
    ev = O.evaluate_design(np.array([17.5, 0.7, 6.0, 0.5, th]), w, pop)
    print(f'rotation {th:3d}: hard yield {ev.hard_yield:.2f}  soft {ev.soft_yield:.2f}  median mu_sw {ev.mu_sw_median}  valid geometry {ev.frac_valid_geometry:.2f}')
f = PR.YieldObjective(w, n_dev=12)
tr = opt.METHODS['bo_ei'](f, 16, 0)
print('mini Bayesian optimisation (16 evaluations): best smoothed yield', round(float(tr.y.max()), 3), 'at', np.round(opt.to_design(tr.best_u), 2))"""),
    M("""## 10. What would change the conclusions
The constants of the wafer (including signs), the oxidation depth and anisotropy, single-mode versus multimode apertures, thermal effects, the switching-current window that defines a good node,
and the two published observations that the model does not reproduce. The measurements that calibrate them are listed in supplement Sec. 8 (`calibration.py`); the two sources that were not read in full
(J. Lightwave Technol. 43, 6819; Appl. Phys. Lett. 127, 221102) would turn the qualitative anchors into quantitative ones."""),
]
out = ROOT / "notebooks" / "VCSEL_Polarization_BO.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
