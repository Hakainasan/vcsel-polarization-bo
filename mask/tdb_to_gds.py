"""Convert the decoded L-Edit .tdb (tdb_reader.py) into GDSII and measure the device template.

Layer map (the numbering of the author's masks; L-Edit imports it with File -> Import Mask Data -> GDSII against the reference TDB):
    Mesa 1, n-Metal 2, p-MetalPad 3, Grid Layer 4, N Well 5, first-p-metal 6, Oxide/Nitride 7   (datatype 0 everywhere)

Usage: python -I mask/tdb_to_gds.py <file.tdb> <out_dir>
Writes <out_dir>/example_decoded.gds (hierarchical, top cell PM_Full), example_report.json (per-cell check against the stored bounding
boxes, sub-micron specks), template_measurements.json (single-device template rules).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gdstk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tdb_reader as T  # noqa: E402

TECH = {3: (1, "Mesa"), 4: (2, "n-Metal"), 5: (3, "p-MetalPad"), 7: (4, "Grid Layer"), 18: (5, "N Well"),
        19: (6, "first-p-metal"), 30: (7, "Oxide/Nitride")}
TOL = 0.005          # um, maximum deviation of a polygonised arc from the true circle


def to_gds(cells, path):
    lib = gdstk.Library("EXAMPLE_DECODED", unit=1e-6, precision=1e-9)
    gcells = {c.index: lib.new_cell(c.name) for c in cells}
    specks = []
    for c in cells:
        gc = gcells[c.index]
        for s in c.shapes:
            if s.layer not in TECH:
                continue
            lay = TECH[s.layer][0]
            pts = np.array(T.shape_points(s, TOL))
            if len(pts) < 3:
                continue
            ext = (pts.max(0) - pts.min(0)).max()
            if ext < 0.5:
                specks.append(dict(cell=c.name, layer=TECH[s.layer][1], centre=[round(float(v), 3) for v in pts.mean(0)],
                                   size_um=round(float(ext), 3)))
            gc.add(gdstk.Polygon(pts, layer=lay, datatype=0))
        for inst in c.instances:
            if 0 < inst.cell_index <= len(cells):
                gc.add(gdstk.Reference(gcells[inst.cell_index], origin=(inst.x, inst.y)))
    lib.write_gds(path)
    return specks


def ring_of(s):
    p = np.array(T.shape_points(s, TOL))
    r = np.hypot(p[:, 0], p[:, 1])
    return float(r.min()), float(r.max())


def measure_single(cell):
    """Template numbers of a single circular device (mesa radius R)."""
    out = dict(cell=cell.name)
    for s in cell.shapes:
        name = TECH.get(s.layer, (0, "?"))[1]
        pts = np.array(T.shape_points(s, TOL))
        if (pts.max(0) - pts.min(0)).max() < 0.5:
            continue
        if s.kind == "circle":
            x0, y0, x1, y1 = s.data
            out.setdefault(name, []).append(dict(circle_centre=[(x0 + x1) / 2, (y0 + y1) / 2], r=(x1 - x0) / 2))
        elif s.kind == "box":
            out.setdefault(name, []).append(dict(box=[round(v, 3) for v in s.data]))
        else:
            arcs = sum(1 for it in s.data if it[0] == "c")
            rec = dict(polygon_vertices=[[round(it[1], 3), round(it[2], 3)] for it in s.data if it[0] == "v"], arcs=arcs)
            if arcs:
                rmin, rmax = ring_of(s)
                rec.update(r_inner=round(rmin, 3), r_outer=round(rmax, 3))
                v = [it for it in s.data if it[0] == "v"]
                ang = [np.degrees(np.arctan2(it[2], it[1])) for it in v]
                rec["vertex_angles_deg"] = [round(a, 2) for a in ang]
            out.setdefault(name, []).append(rec)
    return out


def main():
    src, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    cells = T.read(src)
    specks = to_gds(cells, str(outdir / "example_decoded.gds"))
    rep = []
    for c in cells:
        bb = T.cell_bbox(cells, c)
        ok = bb is not None and all(abs(a - b) < 0.02 for a, b in zip(bb, c.stored_bbox))
        rep.append(dict(cell=c.name, index=c.index, shapes=len(c.shapes), instances=len(c.instances), bbox_matches_file=ok,
                        decoded_bbox=[round(float(v), 3) for v in bb] if bb else None, stored_bbox=[round(v, 3) for v in c.stored_bbox]))
    (outdir / "example_report.json").write_text(json.dumps(dict(cells=rep, specks=specks), indent=1))
    singles = [c for c in cells if c.name.startswith("PM_circle_R_") and "array" not in c.name]
    meas = {c.name: measure_single(c) for c in singles}
    (outdir / "template_measurements.json").write_text(json.dumps(meas, indent=1))
    print(f"{sum(r['bbox_matches_file'] for r in rep)} of {len(rep)} cells match the stored bounding box; {len(specks)} sub-0.5 um specks")


if __name__ == "__main__":
    main()
