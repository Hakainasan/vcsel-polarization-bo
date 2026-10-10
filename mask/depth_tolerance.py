"""What happens to every device of PM_Group_2 if the real oxidation depth differs from the design value (5 um).  GEOMETRY ONLY.

Reads the GDS written by make_mask2.py (not the generator's internal shapes) and, for each device and each real depth D, erodes the
drawn mesa by D (isotropic oxidation, round offset) and reports:
  * the number of oxide apertures (pinch-off would split one aperture into two or more);
  * the aperture area, its second-moment aspect ratio and long-axis angle;
  * the clearance between the first-p-metal ring and the aperture (metal over the emitting area if it reaches zero).
Writes <out_dir>/depth_tolerance.json and prints a per-design summary (replicates of a design are identical, so one is shown).

Usage: python -I mask/depth_tolerance.py mask/out
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import gdstk
import numpy as np
from scipy.spatial import cKDTree

DEPTHS = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5)
PREC = 1e-3
MIN_AREA = 0.5            # um^2; smaller fragments are numerical dust, not apertures
STEP = 0.02               # um; boundary resampling for the clearance measurement


def poly_moments(pts):
    """Area, centroid and central second moments of a simple polygon (shoelace formulas)."""
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2.0
    cx = ((x + x1) * cr).sum() / (6 * a)
    cy = ((y + y1) * cr).sum() / (6 * a)
    ixx = ((x * x + x * x1 + x1 * x1) * cr).sum() / 12.0        # integral of x^2
    iyy = ((y * y + y * y1 + y1 * y1) * cr).sum() / 12.0        # integral of y^2
    ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / 24.0
    return a, cx, cy, ixx, iyy, ixy


def shape(polys):
    """Total area, aspect ratio (sqrt of the eigenvalue ratio of the area covariance) and long-axis angle of a set of polygons."""
    tot = np.zeros(6)
    for p in polys:
        a, cx, cy, ixx, iyy, ixy = poly_moments(p.points)
        s = np.sign(a)                                            # orientation-independent
        tot += s * np.array([a, a * cx, a * cy, ixx, iyy, ixy])
    a = tot[0]
    cx, cy = tot[1] / a, tot[2] / a
    sxx, syy, sxy = tot[3] / a - cx * cx, tot[4] / a - cy * cy, tot[5] / a - cx * cy
    ev, evec = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
    ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180.0)
    return float(a), float(np.sqrt(ev[1] / ev[0])), ang


def dense_boundary(polys):
    out = []
    for p in polys:
        q = p.points
        for a, b in zip(q, np.roll(q, -1, axis=0)):
            n = max(int(np.ceil(np.hypot(*(b - a)) / STEP)), 1)
            t = np.arange(n)[:, None] / n
            out.append(a + t * (b - a))
    return np.vstack(out) if out else np.zeros((0, 2))


def main():
    out_dir = Path(sys.argv[1])
    lib = gdstk.read_gds(str(out_dir / "VCSEL_PM_Group_2.gds"))
    top = lib.top_level()[0]
    by_layer = {}
    for p in top.polygons:
        by_layer.setdefault(p.layer, []).append(p)
    devices = list(csv.DictReader(open(out_dir / "VCSEL_PM_Group_2_devices.csv")))

    def near(layer, x, y, r):
        sel = []
        for p in by_layer.get(layer, []):
            (x0, y0), (x1, y1) = p.bounding_box()
            if x0 - r <= x <= x1 + r and y0 - r <= y <= y1 + r:
                sel.append(p)
        return sel

    res, summary = {}, {}
    for d in devices:
        x, y = float(d["x_um"]), float(d["y_um"])
        mesa = [p for p in near(1, x, y, 0.0) if p.contain((x, y))]
        ring = near(6, x, y, 30.0)
        assert len(mesa) == 1, (d["slot"], len(mesa))
        rec = {}
        for D in DEPTHS:
            ap = [p for p in gdstk.offset(mesa, -D, join="round", tolerance=0.01, precision=PREC) if abs(p.area()) > MIN_AREA]
            if not ap:
                rec[f"{D:.1f}"] = dict(apertures=0)
                continue
            area, aspect, axis = shape(ap)
            clr = float(cKDTree(dense_boundary(ring)).query(dense_boundary(ap))[0].min()) if ring else None
            overlap = bool(ring) and bool(gdstk.boolean(ap, ring, "and", precision=PREC))
            rec[f"{D:.1f}"] = dict(apertures=len(ap), area=round(area, 1), aspect=round(aspect, 3), axis_deg=round(axis, 1),
                                   ring_clearance_um=None if clr is None else round(0.0 if overlap else clr, 2))
        res[d["slot"]] = dict(id=d["id"], **rec)
        summary.setdefault(d["id"], rec)
    out = dict(note="GEOMETRY ONLY: drawn mesa eroded isotropically by the real oxidation depth D (design D = 5 um); "
                    "aspect = sqrt of the eigenvalue ratio of the aperture's area covariance; axis 90 deg = along mask y",
               depths_um=list(DEPTHS), by_design=summary, by_slot=res)
    (out_dir / "depth_tolerance.json").write_text(json.dumps(out, indent=1))
    print(f"{'design':6s} " + " ".join(f"{'D=' + format(D, '.1f'):>22s}" for D in DEPTHS))
    print(f"{'':6s} " + " ".join(f"{'n  area aspect clr':>22s}" for _ in DEPTHS))
    for sid, rec in summary.items():
        cells = []
        for D in DEPTHS:
            r = rec[f"{D:.1f}"]
            cells.append(f"{r['apertures']} {r.get('area', 0):6.1f} {r.get('aspect', 0):5.2f} {r.get('ring_clearance_um') or 0:4.2f}")
        print(f"{sid:6s} " + " ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
