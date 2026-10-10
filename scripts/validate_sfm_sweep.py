"""Validate the fixed-point/Jacobian machinery against time-domain SFM integration over a (gamma_a, mu) sweep.

For each (dichroism g_a, current mu) with birefringence g_p = +20 ns^-1 (same sign class, switching allowed):
  * fixed-point method: which of the two linear-polarization branches is stable;
  * time-domain method: integrate from a random small-field start and record the final polarization P = S1.
Agreement = the time-domain final state matches one of the fixed-point branches that the Jacobian calls stable.
Writes results/sfm_validation.csv and figures/sfm_validation.png. Synthetic constants (SFMConstants defaults).
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from vcsel_bo import sfm  # noqa: E402

C = sfm.SFMConstants()
rows = []
rng = np.random.default_rng(0)
GA = [1.0, 2.0, 3.0]
MUS = np.linspace(1.1, 2.2, 12)
for ga in GA:
    for mu in MUS:
        m11, m12 = complex(ga, 20.0), 0j
        stable = []
        for which in (0, 1):
            u0 = sfm.linear_mode_guess(np.array([mu]), np.array([m11]), np.array([m12]), C, which)
            u, ok = sfm.newton_fixed_point(u0, np.array([mu]), np.array([m11]), np.array([m12]), C)
            lam = sfm.stability(u, np.array([mu]), np.array([m11]), np.array([m12]), C)[0]
            I = sfm.stokes(u)["intensity"][0]
            if ok[0] and I > 1e-3 and lam < -1e-3:
                stable.append(float(sfm.stokes(u)["P"][0]))
        # time domain from 3 random starts
        finals = []
        for _ in range(3):
            y0 = np.array([*rng.normal(0, 0.05, 4), 1.0, 0.0])
            sol = sfm.integrate(y0, mu, m11, m12, C, t_end=80)
            st = sfm.stokes(np.concatenate([sol.y[:, -1], [0.0]])[None])
            finals.append(float(st["P"][0]))
        agree = all(any(abs(f - p) < 0.03 for p in stable) for f in finals) if stable else False
        rows.append((ga, mu, len(stable), ";".join(f"{p:.3f}" for p in stable), ";".join(f"{f:.3f}" for f in finals), int(agree)))
out = ROOT / "results" / "sfm_validation.csv"
out.write_text("gamma_a,mu,n_stable_fixed_points,P_stable,P_time_domain,agree\n" +
               "\n".join(",".join(map(lambda v: f"{v:.4g}" if isinstance(v, float) else str(v), r)) for r in rows) + "\n")
n_agree = sum(r[-1] for r in rows)
print(f"{n_agree}/{len(rows)} (gamma_a, mu) points: time-domain endpoint matches a Jacobian-stable fixed point")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(5, 3.2))
for ga in GA:
    sel = [r for r in rows if r[0] == ga]
    ax.plot([r[1] for r in sel], [float(r[4].split(";")[0]) for r in sel], "o", ms=4, label=f"time domain, $\\gamma_a$={ga:g}")
    ps = [(r[1], float(p)) for r in sel if r[3] for p in r[3].split(";")]
    ax.plot(*zip(*ps), "k_", ms=8)
ax.plot([], [], "k_", label="stable fixed point (Jacobian)")
ax.set_xlabel("$\\mu = I/I_{th}$"); ax.set_ylabel("$P=S_1$"); ax.legend(fontsize=7); ax.set_title("SFM: fixed-point vs time-domain (synthetic constants)", fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "sfm_validation.png", dpi=160)
