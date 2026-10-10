"""Candidate designs for the second fabrication mask: model scores, oxidation robustness, and a shape sheet.  (MODEL PREDICTIONS.)

Scores every candidate on the 32 unseen test worlds of design_batch.py (same 24-device population, same specification), and on the
other 117 accepted worlds; checks the aperture at oxidation depths 0.7 / 1.0 / 1.3 um (area, pinch-off); draws mesa and aperture.
Writes results/mask2_candidates.json and figures/mask2_candidates.png.   Usage: python scripts/mask2_candidates.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from design_batch import yield_matrix  # noqa: E402
from vcsel_bo import geometry as G  # noqa: E402
from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"

# name, group, (L, ratio, w, rho, theta), why
CANDIDATES = [
    ("M1", "model pick", (17.374, 0.723, 6.173, 0.824, 70.26), "best interior design of the batch search"),
    ("M2", "model pick", (18.431, 0.654, 5.061, 0.803, 90.97), "second interior pick"),
    ("M3", "model pick", (14.394, 0.746, 5.817, 0.532, 112.11), "smaller cross, hedges other worlds"),
    ("R70", "rotation sweep", (17.5, 0.7, 6.0, 0.5, 70.0), "plateau edge"),
    ("R80", "rotation sweep", (17.5, 0.7, 6.0, 0.5, 80.0), "most robust rotation in the model"),
    ("R90", "rotation sweep", (17.5, 0.7, 6.0, 0.5, 90.0), "plateau core; also on mask 1"),
    ("R100", "rotation sweep", (17.5, 0.7, 6.0, 0.5, 100.0), "mirror of 80"),
    ("R110", "rotation sweep", (17.5, 0.7, 6.0, 0.5, 110.0), "mirror of 70"),
    ("W85", "thin-arm (risky)", (19.571, 0.5, 3.5, 1.0, 84.91), "best single design in the model; on the wall of the search box"),
    ("W90", "thin-arm (risky)", (17.709, 0.5, 3.5, 0.0, 89.99), "sharp-cornered thin-arm variant"),
    ("C0", "control", (17.5, 0.7, 6.0, 0.5, 0.0), "long arm on the native axis: predicted locked"),
    ("CX0", "control", (17.5, 1.0, 6.0, 0.5, 0.0), "symmetric cross"),
    ("CX45", "control", (17.5, 1.0, 6.0, 0.5, 45.0), "symmetric cross on the diagonal"),
    # circle = MaskDesign(D, 1, D, 1, 0); D = 12.6 um gives about the aperture area of R90 at 1 um oxidation depth
    ("CIRC", "control", (12.6, 1.0, 12.6, 1.0, 0.0), "circular mesa with about the aperture area of R90 (no shape anisotropy)"),
]


def best_of_batch(Y):
    """Expected best-of-batch yield over worlds, and probability that some design reaches yield >= 0.5."""
    b = Y.max(axis=0)
    return dict(E_best=float(b.mean()), P_some_ge_half=float((b >= 0.5).mean()))


def main():
    bw = json.load(open(RES / "batch_worlds.json"))
    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"]
    used = {(r["d0"], r["b0"], r["k_d"], r["k_b"]) for r in bw["train"] + bw["test"]}
    other = [r for r in acc if (r["d0"], r["b0"], r["k_d"], r["k_b"]) not in used]
    rows = [c for c in CANDIDATES if c[2] is not None]
    X = np.array([c[2] for c in rows])
    spec = O.Spec()
    Yt = yield_matrix(X, bw["test"], 24, spec)
    Yo = yield_matrix(X, other, 24, spec)
    out = dict(note="MODEL PREDICTIONS (hard yield = share of simulated devices that switch inside the specification); not measurements",
               n_test_worlds=len(bw["test"]), n_other_worlds=len(other), designs=[])
    for k, (name, grp, x, why) in enumerate(rows):
        ox = {}
        for depth in (0.7, 1.0, 1.3):
            ap = G.build_aperture(G.MaskDesign(*x), G.Process(depth=depth, ox_ratio=1.15))
            ox[str(depth)] = dict(area_um2=round(ap.area, 1), components=ap.n_components, d_eq_um=round(ap.equivalent_diameter, 2))
        out["designs"].append(dict(name=name, group=grp, L=x[0], ratio=x[1], w=x[2], rho=x[3], theta=x[4],
                                   corner_radius_um=round(x[3] * x[2] / 2, 3), why=why,
                                   yield_test_mean=float(Yt[k].mean()), yield_test_q10=float(np.percentile(Yt[k], 10)),
                                   yield_test_q90=float(np.percentile(Yt[k], 90)), yield_other_mean=float(Yo[k].mean()),
                                   p_works_test=float((Yt[k] >= 0.5).mean()), oxidation=ox))
        print(f"{name:5s} {grp:16s} test {Yt[k].mean():.2f} [{np.percentile(Yt[k],10):.2f}-{np.percentile(Yt[k],90):.2f}] other {Yo[k].mean():.2f} "
              f"area(0.7/1.0/1.3) {ox['0.7']['area_um2']}/{ox['1.0']['area_um2']}/{ox['1.3']['area_um2']} comps {ox['1.3']['components']}", flush=True)
    # batch scores on the same 32 test worlds (same device population as design_batch.py), alone and together with mask 1
    z = np.load(RES / "batch_yield_matrix.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]
    Y1 = z["Yte"][[i for i, n in enumerate(names) if n.startswith("mask:")]]
    core = [k for k, c in enumerate(rows) if c[1] in ("model pick", "rotation sweep")]
    no_thin = [k for k, c in enumerate(rows) if c[1] != "thin-arm (risky)"]
    out["batch"] = dict(mask1=best_of_batch(Y1), mask2_all=best_of_batch(Yt), mask2_without_thin_arms=best_of_batch(Yt[no_thin]),
                        mask2_core_only=best_of_batch(Yt[core]), mask1_plus_mask2=best_of_batch(np.vstack([Y1, Yt])),
                        mask1_plus_mask2_without_thin_arms=best_of_batch(np.vstack([Y1, Yt[no_thin]])))
    for k, v in out["batch"].items():
        print(f"{k:38s} E_best {v['E_best']:.3f}  P(some >= 0.5) {v['P_some_ge_half']:.3f}")
    np.savez(RES / "mask2_candidates_yields.npz", names=np.array([c[0] for c in rows]), Y_test=Yt, Y_other=Yo)
    (RES / "mask2_candidates.json").write_text(json.dumps(out, indent=1))

    fig, axs = plt.subplots(2, 7, figsize=(14, 4.6))
    for ax, d in zip(axs.ravel(), out["designs"]):
        ax.set_aspect("equal"); ax.axis("off")
        des = G.MaskDesign(d["L"], d["ratio"], d["w"], d["rho"], d["theta"])
        mb = G.mesa_boundary(des)
        ax.fill(mb[:, 0], mb[:, 1], color="#d9d9d9", lw=0)
        ap = G.build_aperture(des, G.Process(1.0, 1.15))
        for lp in ap.loops:
            ax.plot(lp[:, 0], lp[:, 1], color="k", lw=1)
        ax.set_xlim(-11, 11); ax.set_ylim(-11, 11)
        ax.set_title(f"{d['name']}  {d['theta']:.0f}°\nyield {d['yield_test_mean']:.2f}", fontsize=8)
    fig.suptitle("Mask-2 candidates: mesa (grey) and oxide aperture at 1.0 µm depth (black); model yield on 32 test worlds", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "mask2_candidates.png", dpi=150)


if __name__ == "__main__":
    main()
