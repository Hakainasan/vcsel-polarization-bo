"""What does a deep oxidation do to a cruciform aperture?  (GEOMETRY ONLY; no device physics, no measurement.)

Convention of the author's masks: the drawn cruciform is the TARGET APERTURE and the mesa is that shape grown by the design
oxidation depth D (offset with round joins).  Dilation distributes over a union, so the mesa of a cruciform target
(L, ratio, w, rho) is again a union of two rounded rectangles:

    L' = L + 2 D,   short' = ratio L + 2 D,   w' = w + 2 D,   tip radius' = rho w / 2 + D      (function `mesa_of_target`).

The oxide front then removes a depth D from every mesa edge.  With isotropic oxidation of exactly D the aperture is the
morphological CLOSING of the target by a disk of radius D: convex corners come back unchanged, but every concave corner is
filled with a fillet of radius D.  When D exceeds the length by which an arm sticks out, the fillet bridges neighbouring arm tips
and the cross turns into an elongated octagon.  Anisotropic oxidation (rate ratio rho_ox) removes different depths along the two
crystal axes, and the difference grows in proportion to D.

For each target and each D this script reports: aperture area relative to the target, solidity (area / convex-hull area; 1 for an
octagon, lower for a cross), the aspect ratio and axis of the second moments, with isotropic and with anisotropic oxidation, and
the area change for a depth error of +-10 %.  Writes results/depth_study.json and figures/depth_study.png.
Usage: python scripts/depth_study.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from vcsel_bo import geometry as G  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
DEPTHS = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
OX = 1.15

TARGETS = {  # name: (L, ratio, w, rho, theta) of the TARGET APERTURE
    "A60 (mask 1, family A)": (17.5, 0.7, 6.0, 0.0, 60.0),
    "B90 (mask 1, family B)": (15.0, 0.7, 6.0, 0.75, 90.0),
    "R80 (1:0.7 cross)": (17.5, 0.7, 6.0, 0.5, 80.0),
    "M2 (model pick)": (18.431, 0.654, 5.061, 0.803, 90.97),
    "W85 (thin arm)": (19.571, 0.5, 3.5, 1.0, 84.91),
    "CX0 (symmetric cross)": (17.5, 1.0, 6.0, 0.5, 0.0),
}


def mesa_of_target(t, D):
    """Mesa = target aperture grown by D (exact for a union of rounded rectangles)."""
    L, ratio, w, rho, th = t
    L2, w2 = L + 2 * D, w + 2 * D
    return G.MaskDesign(L2, (ratio * L + 2 * D) / L2, w2, (rho * w + 2 * D) / w2, th)


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def shape_stats(loop):
    """Area, solidity, second-moment aspect ratio and long-axis angle (deg, 0..180) of a closed polygon."""
    a = poly_area(loop)
    hull = ConvexHull(loop)
    sol = a / hull.volume
    # second moments from a dense interior sample (robust and simple)
    lo, hi = loop.min(0), loop.max(0)
    g = np.stack(np.meshgrid(np.arange(lo[0], hi[0], 0.05), np.arange(lo[1], hi[1], 0.05)), -1).reshape(-1, 2)
    from matplotlib.path import Path as MPath
    pts = g[MPath(loop).contains_points(g)]
    c = pts - pts.mean(0)
    cov = c.T @ c / len(c)
    ev, evec = np.linalg.eigh(cov)
    ang = float(np.rad2deg(np.arctan2(evec[1, 1], evec[0, 1])) % 180.0)
    return dict(area=float(a), solidity=float(sol), aspect=float(np.sqrt(ev[1] / ev[0])), axis_deg=ang)


def realised(t, D, ox, depth_scale=1.0):
    ap = G.build_aperture(mesa_of_target(t, D), G.Process(depth=D * depth_scale, ox_ratio=ox))
    return ap


def main():
    out = dict(note="GEOMETRY ONLY: target aperture -> mesa (target grown by D) -> aperture after oxidation of depth D; "
                    "anisotropic = rate ratio %.2f (placeholder, not measured)" % OX, depths=list(DEPTHS), targets={})
    for name, t in TARGETS.items():
        tgt = G.mesa_boundary(G.MaskDesign(*t))
        # order the boundary samples of the target into a polygon through a zero-depth aperture contour
        tgt_loop = G.build_aperture(G.MaskDesign(*t), G.Process(depth=0.0, ox_ratio=1.0)).loops[0]
        st = shape_stats(tgt_loop)
        rec = dict(target=dict(L=t[0], short=t[0] * t[1], w=t[2], tip_radius=t[3] * t[2] / 2, theta=t[4], **st), by_depth={})
        for D in DEPTHS:
            m = mesa_of_target(t, D)
            row = dict(mesa=dict(L=m.L, short=m.L * m.ratio, w=m.w, tip_radius=m.corner_radius))
            for label, ox in (("isotropic", 1.0), ("anisotropic", OX)):
                ap = realised(t, D, ox)
                if not ap.loops:
                    row[label] = dict(valid=False)
                    continue
                s = shape_stats(ap.loops[0])
                s["area_rel_target"] = s["area"] / st["area"]
                s["components"] = ap.n_components
                lo = realised(t, D, ox, 0.9)
                hi = realised(t, D, ox, 1.1)
                s["area_at_depth_minus10pct"] = float(lo.area)
                s["area_at_depth_plus10pct"] = float(hi.area)
                row[label] = s
            rec["by_depth"][str(D)] = row
            a, i = row["anisotropic"], row["isotropic"]
            print(f"{name:26s} D={D:4.1f}  iso: area x{i['area_rel_target']:.2f} solidity {i['solidity']:.2f} aspect {i['aspect']:.2f} | "
                  f"aniso: area x{a['area_rel_target']:.2f} aspect {a['aspect']:.2f} axis {a['axis_deg']:.0f}  "
                  f"+-10% depth: {a['area_at_depth_minus10pct']:.0f}..{a['area_at_depth_plus10pct']:.0f} um2", flush=True)
        print(f"{'':26s} target: area {st['area']:.0f} um2, solidity {st['solidity']:.2f}, aspect {st['aspect']:.2f}, axis {st['axis_deg']:.0f}")
        out["targets"][name] = rec
    (RES / "depth_study.json").write_text(json.dumps(out, indent=1))

    show = ["A60 (mask 1, family A)", "R80 (1:0.7 cross)", "W85 (thin arm)"]
    fig, axs = plt.subplots(len(show), len(DEPTHS), figsize=(2.6 * len(DEPTHS), 2.7 * len(show)))
    for r, name in enumerate(show):
        t = TARGETS[name]
        tgt_loop = G.build_aperture(G.MaskDesign(*t), G.Process(depth=0.0, ox_ratio=1.0)).loops[0]
        for c, D in enumerate(DEPTHS):
            ax = axs[r, c]
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            mb = G.build_aperture(mesa_of_target(t, D), G.Process(depth=0.0, ox_ratio=1.0)).loops[0]
            ax.fill(mb[:, 0], mb[:, 1], color="#dcdcdc", lw=0)
            ap = realised(t, D, OX)
            for lp in ap.loops:
                ax.fill(lp[:, 0], lp[:, 1], color="#2a78d6", alpha=0.35, lw=0)
                ax.plot(lp[:, 0], lp[:, 1], color="#1c5cab", lw=1.2)
            ax.plot(tgt_loop[:, 0], tgt_loop[:, 1], color="#111111", lw=0.9, ls=(0, (3, 2)))
            lim = t[0] / 2 + 11.5
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            if r == 0:
                ax.set_title(f"oxidation depth {D:g} µm", fontsize=9)
            if c == 0:
                ax.text(-0.02, 0.5, name, rotation=90, va="center", ha="right", transform=ax.transAxes, fontsize=8.5)
    fig.suptitle("Grey: mesa (target grown by the depth).  Dashed: drawn target aperture.  Blue: aperture after oxidation, rate ratio 1.15 (geometry model)",
                 fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    fig.savefig(FIG / "depth_study.png", dpi=140)


if __name__ == "__main__":
    main()
