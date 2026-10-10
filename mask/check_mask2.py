"""Checks of the mask-2 GDSII with KLayout (independent of the generator's own geometry code).

  * file structure: one flat cell, layers 1-7 / datatype 0 only, everything inside the 2 x 2 mm frame, GDSII records < 32 kB,
    vertices per polygon;
  * data hygiene: edges shorter than 5 nm, spikes (direction reversals), self-intersecting polygons;
  * minimum width and spacing per layer in the device area, two ways: morphologically (opening / closing residues) and with
    KLayout's edge-based width and space checks (projection metrics, on the merged layer smoothed by 10 nm); features that touch
    only at a corner (zero-width necks) are counted separately;
  * predicted oxide apertures = Mesa shrunk by the oxidation depth (round corners): number per device, no parasitic islands;
  * contact rules: first-p-metal inside the mesa and >= 1 um clear of the aperture; no p-metal or passivation over an aperture;
    a passivation opening on every contact ring; p-MetalPad touching every ring; n-Metal >= 10 um from p-MetalPad (device area;
    the lab's alignment marks overlap the two metals by design);
  * the alignment strip against the example's PM_alignment (XOR with the decoded example: raw, and ignoring < 20 nm slivers);
  * neighbouring devices do not touch.
Writes check_report.json and preview PNGs.  Usage: python -I mask/check_mask2.py <out_dir> <example_decoded.gds>
"""
from __future__ import annotations

import csv
import json
import math
import struct
import sys
from pathlib import Path

import klayout.db as db
import klayout.lay as lay

D_OX = 5.0
NAMES = {1: "Mesa", 2: "n-Metal", 3: "p-MetalPad", 4: "Grid Layer", 5: "N Well", 6: "first-p-metal", 7: "Oxide/Nitride"}
MIN_W = {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0, 6: 2.0, 7: 2.0}
MIN_S = {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0, 6: 2.0, 7: 2.0}


def gds_records(path):
    """Largest GDSII record (bytes) and the number of records, from the raw stream."""
    b = Path(path).read_bytes()
    i, n, mx = 0, 0, 0
    while i + 4 <= len(b):
        L = struct.unpack(">H", b[i:i + 2])[0]
        if L == 0:
            break
        n, mx, i = n + 1, max(mx, L), i + L
    return mx, n


def disk(r_um, ly, n=64):
    """Polygon circumscribing a disk of radius r (contains the true disk, so distances measured with it are conservative)."""
    rr = r_um / math.cos(math.pi / n) / ly.dbu
    return db.Polygon([db.Point(round(rr * math.cos(2 * math.pi * k / n)), round(rr * math.sin(2 * math.pi * k / n))) for k in range(n)])


def clearance(a, b, ly, hi=5.0, tol=0.002):
    """Smallest distance between regions a and b (um), by bisection on the Minkowski dilation of b; 0 if they touch."""
    if not (a & b).is_empty():
        return 0.0
    if not (a & b.minkowski_sum(disk(hi, ly))).is_empty():
        lo, up = 0.0, hi
        while up - lo > tol:
            mid = (lo + up) / 2
            if (a & b.minkowski_sum(disk(mid, ly))).is_empty():
                lo = mid
            else:
                up = mid
        return round(lo, 3)
    return None                                                # farther than hi


def hygiene(top, ly, L, area_box):
    """Edges shorter than 5 nm, spikes (turns > 170 deg) and the largest vertex count, over the raw polygons of one layer."""
    short = spikes = maxv = npoly = 0
    for sh in top.shapes(ly.layer(L, 0)).each():
        if not sh.is_polygon() and not sh.is_box():
            continue
        poly = sh.polygon
        if not area_box.contains(poly.bbox().center()):
            continue
        npoly += 1
        pts = [(p.x, p.y) for p in poly.each_point_hull()]
        maxv = max(maxv, len(pts))
        n = len(pts)
        for k in range(n):
            ax, ay = pts[k - 1]
            bx, by = pts[k]
            cx, cy = pts[(k + 1) % n]
            e1, e2 = (bx - ax, by - ay), (cx - bx, cy - by)
            l1, l2 = math.hypot(*e1), math.hypot(*e2)
            if l2 * ly.dbu < 0.005:
                short += 1
            if l1 > 0 and l2 > 0 and (e1[0] * e2[0] + e1[1] * e2[1]) / (l1 * l2) < math.cos(math.radians(170)):
                spikes += 1
    return dict(polygons=npoly, max_vertices=maxv, edges_below_5nm=short, spikes_over_170deg=spikes)


def um(v, ly):
    return int(round(v / ly.dbu))


def main():
    outdir, example = Path(sys.argv[1]), sys.argv[2]
    gds = outdir / "VCSEL_PM_Group_2.gds"
    ly = db.Layout()
    ly.read(str(gds))
    rep = dict(file=str(gds.name))
    tops = ly.top_cells()
    rep["top_cells"] = [c.name for c in tops]
    top = tops[0]
    rep["flat"] = all(c.child_cells() == 0 for c in tops) and ly.cells() == 1
    rep["layers"] = sorted(f"{ly.get_info(li).layer}/{ly.get_info(li).datatype}" for li in ly.layer_indexes()
                           if not top.bbox_per_layer(li).empty())
    bb = top.dbbox()
    rep["bbox_um"] = [bb.left, bb.bottom, bb.right, bb.top]
    rep["inside_frame"] = bb.left >= 0 and bb.bottom >= 0 and bb.right <= 2000 and bb.top <= 2000
    rep["largest_record_bytes"], rep["records"] = gds_records(gds)

    R = {L: db.Region(top.begin_shapes_rec(ly.layer(L, 0))) for L in NAMES}
    for r in R.values():
        r.merge()
    dev_area = db.Region(db.Box(um(220, ly), 0, um(2000, ly), um(2000, ly)))
    strip_area = db.Region(db.Box(0, 0, um(220, ly), um(2000, ly)))
    dev_box = db.Box(um(220, ly), 0, um(2000, ly), um(2000, ly))

    # data hygiene (raw polygons as written) and self-intersections
    rep["hygiene_device_area"] = {NAMES[L]: hygiene(top, ly, L, dev_box) for L in NAMES}
    rep["self_intersecting_polygons"] = {NAMES[L]: db.Region(top.begin_shapes_rec(ly.layer(L, 0))).strange_polygon_check().count()
                                         for L in NAMES}

    # edge-based width / space (projection metrics: parallel-facing edges, so rounded corners of polygonised arcs are not
    # flagged), merged layer smoothed by 10 nm, device area; corner-only contacts (Euclidian, < 10 nm) separately
    E, Pm = db.Region.Euclidian, db.Region.Projection
    rep["drc_width_space_2um"] = {}
    for L in NAMES:
        reg = (R[L] & dev_area).smoothed(um(0.01, ly))
        if reg.is_empty():
            continue
        nd = lambda ep: [e for e in ep.each() if e.first.length() > 0 and e.second.length() > 0]  # noqa: E731
        w = db.EdgePairs(nd(reg.width_check(um(MIN_W[L], ly), False, Pm)))
        s = db.EdgePairs(nd(reg.space_check(um(MIN_S[L], ly), False, Pm)))
        touch = (R[L] & dev_area).space_check(um(0.01, ly), False, E)
        where = lambda ep: sorted({(round(e.bbox().center().x * ly.dbu, 1), round(e.bbox().center().y * ly.dbu, 1))  # noqa: E731
                                   for e in ep.each()})[:5]
        rep["drc_width_space_2um"][NAMES[L]] = dict(
            width_violations=w.count(), min_width_um=min((e.distance() * ly.dbu for e in w.each()), default=None), where_w=where(w),
            space_violations=s.count(), min_space_um=min((e.distance() * ly.dbu for e in s.each()), default=None), where_s=where(s),
            corner_contacts=touch.count())

    # widths / spaces in the device area, checked morphologically: a feature narrower than w disappears in an opening with
    # radius w/2, a gap narrower than w fills in a closing with radius w/2.  Residues of rounded corners are < 0.5 um2 and are
    # not violations; anything larger is reported with its position.  (Edge-pair width checks flag every polygonised arc.)
    rep["narrow_features"], rep["narrow_gaps"] = {}, {}
    for L in MIN_W:
        reg = R[L] & dev_area
        rw, rs = um(MIN_W[L] / 2 - 0.05, ly), um(MIN_S[L] / 2 - 0.05, ly)
        thin = reg - reg.sized(-rw, -rw, 2).sized(rw, rw, 2)
        notch = reg.sized(rs, rs, 2).sized(-rs, -rs, 2) - reg
        thin.merge()
        notch.merge()
        rep["narrow_features"][NAMES[L]] = [[round(p.bbox().center().x * ly.dbu, 1), round(p.bbox().center().y * ly.dbu, 1),
                                             round(p.area() * ly.dbu ** 2, 2)] for p in thin.each() if p.area() * ly.dbu ** 2 > 0.5]
        rep["narrow_gaps"][NAMES[L]] = [[round(p.bbox().center().x * ly.dbu, 1), round(p.bbox().center().y * ly.dbu, 1),
                                         round(p.area() * ly.dbu ** 2, 2)] for p in notch.each() if p.area() * ly.dbu ** 2 > 0.5]

    # predicted apertures: Mesa eroded by a 128-gon disk of radius D_OX (exact Minkowski erosion = isotropic oxidation front)
    mesa_dev = R[1] & dev_area
    frame = db.Region(db.Box(um(150, ly), -um(50, ly), um(2050, ly), um(2050, ly)))
    disk = db.Polygon([db.Point(round(um(D_OX, ly) * math.cos(2 * math.pi * k / 128)),
                                round(um(D_OX, ly) * math.sin(2 * math.pi * k / 128))) for k in range(128)])
    ap = frame - (frame - mesa_dev).minkowski_sum(disk)
    ap = ap & dev_area
    ap.merge()
    devices = list(csv.DictReader(open(outdir / "VCSEL_PM_Group_2_devices.csv")))
    per = []
    for d in devices:
        x, y = float(d["x_um"]), float(d["y_um"])
        win = db.Region(db.Box(um(x - 70, ly), um(y - 70, ly), um(x + 70, ly), um(y + 70, ly)))
        a = ap & win
        polys = [p for p in a.each()]
        areas = sorted(round(p.area() * ly.dbu ** 2, 1) for p in polys)
        pm = R[6] & win
        clear = (pm & a.sized(um(1.0, ly))).is_empty()
        on_mesa = (pm - (R[1] & win)).is_empty()
        no_ppad = (R[3] & win & a).is_empty()
        no_ox = (R[7] & win & a).is_empty()
        contact = pm - (R[7] & win)                           # metal on semiconductor (passivation opened)
        ppad_touch = not (contact & (R[3] & win)).is_empty()
        exp = 1 if d["kind"] == "single" else None
        gap = min((e.distance() * ly.dbu for e in pm.space_check(um(10.0, ly), False, db.Region.Projection).each()
                   if e.first.length() > 0 and e.second.length() > 0), default=None)       # slot width of the C ring
        clr = clearance(pm, a, ly)
        per.append(dict(slot=d["slot"], id=d["id"], apertures=len(polys), areas=areas, parasitic=sum(1 for v in areas if v < 20.0),
                        ring_clear_of_aperture_1um=clear, ring_to_aperture_um=None if clr is None else round(clr, 3),
                        ring_gap_um=None if gap is None else round(gap, 3), ring_on_mesa=on_mesa, no_ppad_on_aperture=no_ppad,
                        no_passivation_on_aperture=no_ox, contact_opening=not contact.is_empty(), ppad_touches_contact=ppad_touch,
                        expected_apertures=exp))
    rep["devices"] = per
    rep["devices_failing"] = [p["slot"] + " " + p["id"] for p in per
                              if p["parasitic"] or (p["ring_gap_um"] is not None and p["ring_gap_um"] < 2.0)
                              or not (p["ring_clear_of_aperture_1um"] and p["ring_on_mesa"] and p["no_ppad_on_aperture"]
                                                        and p["no_passivation_on_aperture"] and p["contact_opening"] and p["ppad_touches_contact"])
                              or (p["expected_apertures"] and p["apertures"] != p["expected_apertures"])]
    # n-Metal to p-MetalPad separation (shorts)
    rep["nmetal_ppad_closer_than_10um"] = (R[2] & dev_area).separation_check(R[3] & dev_area, um(10.0, ly)).count()
    rep["nmetal_overlaps_ppad"] = not (R[2] & R[3] & dev_area).is_empty()
    # neighbouring devices: n-Metal of different devices must not touch (merged polygon count = devices + labels)
    nm = R[2] & dev_area
    rep["nmetal_islands_device_area"] = nm.count()

    # alignment strip identical to the example
    ex = db.Layout()
    ex.read(example)
    pa = ex.cell("PM_alignment")
    diff, raw = {}, {}
    for L in NAMES:
        li_ex = ex.find_layer(L, 0)
        e = db.Region(pa.begin_shapes_rec(li_ex)) if li_ex is not None else db.Region()
        e.merge()
        mine = R[L] & strip_area
        x = (mine ^ e)
        raw[NAMES[L]] = round(x.area() * ly.dbu ** 2, 3)
        x = x.sized(-um(0.02, ly))                            # ignore sub-20 nm differences (arc polygonisation)
        diff[NAMES[L]] = round(x.area() * ly.dbu ** 2, 3)
    rep["alignment_strip_xor_area_um2"] = diff
    rep["alignment_strip_xor_area_um2_raw"] = raw
    (outdir / "check_report.json").write_text(json.dumps(rep, indent=1))

    # previews
    prev = outdir / "preview_with_apertures.gds"
    l2 = db.Layout()
    l2.read(str(gds))
    t2 = l2.top_cells()[0]
    t2.shapes(l2.layer(9, 0)).insert(ap)
    l2.write(str(prev))
    cols = {1: 0x9e9e9e, 2: 0x2f7ed8, 3: 0xe8a33d, 4: 0x111111, 5: 0xb05fd6, 6: 0xd94b4b, 7: 0x3fbfbf, 9: 0x14a514}

    def render(box, out, w, h):
        lv = lay.LayoutView()
        lv.load_layout(str(prev), True)
        lv.set_config("background-color", "#ffffff")
        lv.set_config("grid-visible", "false")
        lv.clear_layers()
        for L, c in cols.items():
            lp = lay.LayerPropertiesNode()
            lp.source = f"{L}/0@1"
            lp.fill_color = c
            lp.frame_color = c
            lp.dither_pattern = 0 if L == 9 else (5 if L in (1, 7) else 2)
            lv.insert_layer(lv.end_layers(), lp)
        lv.max_hier()
        lv.zoom_box(db.DBox(*box))
        lv.save_image(str(outdir / out), w, h)

    render((-10, -10, 2010, 2010), "preview_group.png", 1500, 1500)
    first = {d["id"]: d for d in devices}
    for key in ("X7", "X4", "S155", "SYM"):
        d = first[key]
        x, y = float(d["x_um"]), float(d["y_um"])
        render((x - 60, y - 45, x + 60, y + 45), f"preview_{key}.png", 1200, 900)
    print(json.dumps({k: v for k, v in rep.items() if k != "devices"}, indent=1))


if __name__ == "__main__":
    main()
