"""Mask 2: cruciform-aperture VCSELs in the lab's 2 x 2 mm group format (drop-in replacement for PM_Group_1 of
VCSEL_ARRAY_Heming_-_V04.tdb).  Writes one flat GDSII cell, a CSV device map, a JSON check report and PNG previews.

Process assumptions (from the author, 2026-10-08): drawn shape = TARGET aperture; mesa = target grown by the oxidation depth
D = 5 um (offset with round joins); oxidation isotropic (round apertures on circular mesas).

Device template: measured from the example (PM_circle_R_20 and R_24.14, see example/template_measurements.json) and adapted to a
5 um oxide rim.  The example's in-mesa dimensions (6 um contact ring ending 1-2.4 um inside the mesa edge, passivation opening
about 4 um inside it) assume a rim of 8-10 um and would put metal over a 5 um-rim aperture, so they are scaled down
(RING_*, OX_IN below; THIS IS A PROCESS DECISION TO CONFIRM).  Everything outside the mesa (p-pad ring, lead and pad,
passivation ring and field, n-metal 'D' and pads) follows the example; single devices reuse the example's n-Metal polygons verbatim.

Layers (datatype 0): 1 Mesa, 2 n-Metal, 3 p-MetalPad, 4 Grid Layer (non-printing reference: drawn target aperture),
5 N Well (as in the example: a 20 um circle at each emitter centre), 6 first-p-metal, 7 Oxide/Nitride.
Rules kept from mask 1: one flat cell, layers 1-7 only, cruciform layers rotate with the device, pads axis-aligned,
explicit construction with round-join offsets and unions (no XOR).

Usage: python -I mask/make_mask2.py <example.tdb> <out_dir>
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import gdstk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tdb_reader as T  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Parameters (um)
# ----------------------------------------------------------------------------------------------
D_OX = 5.0            # design oxidation depth: mesa = target grown by D_OX
TOL = 0.01            # arc tolerance (max deviation of a polygonised arc), well below a mask writer's address grid
PREC = 1e-3           # boolean/offset precision = the 1 nm GDSII database unit
CLEAN_R = 1.05        # clean-up radius: notches and slivers narrower than 2 * CLEAN_R are removed (closing + opening), so every
                      # convex corner and every concave junction is rounded with r >= 1.05 um (no sub-2 um width or space)
MAX_POINTS = 4000     # vertices per GDSII polygon: keeps every record below 32 kB and every polygon whole (GDSII allows 8190).
                      # A writer limited to 200 vertices: split on export (KLayout or L-Edit), not here, because the splitting in
                      # gdstk leaves nanometre slivers along the cuts
JUNCTION_R = 2.0      # closing radius of the p-pad and passivation outlines: the acute notch where the tapered lead meets the
                      # round pad band gets a 2 um fillet, so no part of it is narrower than 2 um
TIDY_EPS = 0.005      # um: vertices that move the outline by less than this are dropped (nm zigzags left by boolean snapping);
                      # arc vertices (40 nm from their neighbours' chord at TOL = 10 nm) are kept
SPIKE_LEN = 0.1       # um: a vertex where the outline turns by more than SPIKE_TURN_DEG next to an edge shorter than this is an
SPIKE_TURN_DEG = 100.0  # offset artefact (a tooth or a reversal tens of nm high), not a feature: it is dropped.  Real corners in
                      # this layout are either gentle (arc vertices, < 10 deg) or between edges longer than 1 um
RING_OUT = 0.75       # first-p-metal outer edge, inside the mesa edge      (example: 1.0-2.4 um; rim 10 um)
RING_IN = 3.5         # first-p-metal inner edge, inside the mesa edge      (example: 7-8.4 um) -> 1.5 um clear of the aperture
OX_IN = 2.0           # Oxide/Nitride inner edge, inside the mesa edge      (example: ~4 um)  -> 1.5 um contact strip
PPAD_OUT = 8.0        # p-MetalPad band outer edge, outside the mesa edge   (example: R + 8)
OX_OUT = 22.0         # Oxide/Nitride band outer edge, outside the mesa edge (example: R + 22)
GAP_WIDTH = 4.6       # opening of each first-p-metal C ring, towards the pads: a parallel slot this wide.  The example cuts a
                      # +-10 deg wedge, 4.6-6.7 um wide at its 13-19 um ring radii; at the 5-10 um radii here the same angle would
                      # leave only 1.8-3.3 um, so the example's narrowest width is kept instead of its angle
NWELL_R = 10.0        # N Well circle radius (example: 10 um)
LABEL_LAYER = "nmetal"  # design-ID labels (the lab's own strip text is on p-MetalPad; n-Metal keeps 7.5 um from the n pads)
LABEL_MIRROR = True   # draw the labels mirrored (x -> -x) like the lab's strip text (KAUST, numbers), so both read the same way
                      # on the wafer; set False for right-reading labels
N_CLEAR = 10.0        # minimum clearance between Oxide/Nitride band and the n-metal 'D' of multi-element structures
BRIDGE_LEN_OVERLAP = 3.0   # bridges overlap each mesa by this length
LAY = dict(mesa=1, nmetal=2, ppad=3, grid=4, nwell=5, pmetal=6, ox=7)

GROUP = 2000.0                                   # group frame, as PM_Group_1 (alignment strip PM_alignment at (0, 0))
COLS = [383.6 + 290.0 * k for k in range(6)]     # device grid: 6 columns x 7 rows in the area right of the strip
ROWS = [135.5 + 280.0 * k for k in range(7)]


# ----------------------------------------------------------------------------------------------
# Geometry primitives
# ----------------------------------------------------------------------------------------------

def rounded_rect(hx, hy, r):
    p = gdstk.rectangle((-hx, -hy), (hx, hy))
    if r > 1e-6:
        p.fillet(min(r, hx - 1e-4, hy - 1e-4), tolerance=TOL)
    return p


def cross_target(L, ratio, w, rho, theta_deg, centre=(0.0, 0.0)):
    """Target aperture: union of two rounded rectangles (long arm along theta); tip corner radius rho * w / 2."""
    r = rho * w / 2.0
    if abs(ratio - 1.0) < 1e-9 and abs(w - L) < 1e-9 and rho >= 1.0:          # circle
        return [gdstk.ellipse(centre, L / 2.0, tolerance=TOL)]
    if ratio * L <= w + 1e-9 and rho >= 1.0:
        # stadium: the second arm is a circle of diameter w inside the first one, tangent to its sides; a union of tangent
        # outlines leaves slivers, so the stadium is drawn directly as one rounded rectangle
        u = [rounded_rect(L / 2.0, w / 2.0, r)]
    else:
        u = gdstk.boolean(rounded_rect(L / 2.0, w / 2.0, r), rounded_rect(w / 2.0, ratio * L / 2.0, r), "or", precision=PREC)
    for p in u:
        p.rotate(math.radians(theta_deg))
        p.translate(*centre)
    return u


def grow(polys, dist):
    """Offset with round joins (dist > 0 grows; dist < 0 = an isotropic oxidation front moving in by |dist|)."""
    return gdstk.offset(polys, dist, join="round", tolerance=TOL, precision=PREC)


def union(*groups):
    allp = [p for g in groups for p in g]
    return gdstk.boolean(allp, [], "or", precision=PREC)


def clean(polys, r=CLEAN_R, r_close=None):
    """Closing (radius r_close, default r) then opening (radius r), round joins: fills slits and notches narrower than 2 r_close
    and removes slivers narrower than 2 r; leaves corners with radius > r, gaps and features wider than that unchanged."""
    if not polys:
        return polys
    rc = r if r_close is None else r_close
    closed = grow(grow(polys, rc), -rc)
    return grow(grow(closed, -r), r)


def band(polys, d_in, d_out):
    """Region between offset(polys, d_in) and offset(polys, d_out), d_in < d_out (negative = inside the outline)."""
    return gdstk.boolean(grow(polys, d_out), grow(polys, d_in), "not", precision=PREC)


def left_edge_at_y0(polys):
    xs = []
    for p in polys:
        pts = p.points
        for i in range(len(pts)):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
            if (y0 <= 0 < y1) or (y1 <= 0 < y0):
                xs.append(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
    return min(xs) if xs else min(p.points[:, 0].min() for p in polys)


def wedge(half_deg, centre, length=22.0):
    a = math.radians(half_deg)
    cx, cy = centre
    return gdstk.Polygon([(cx, cy), (cx - length * math.cos(a), cy + length * math.sin(a)), (cx - length * math.cos(a), cy - length * math.sin(a))])


def slot(width, centre, length=40.0):
    """Parallel-sided opening from the device centre towards the pads (-x), `width` wide."""
    cx, cy = centre
    return gdstk.rectangle((cx - length, cy - width / 2.0), (cx, cy + width / 2.0))


def tidy_points(pts, eps=TIDY_EPS):
    """Drop vertices that move the outline by less than eps (duplicates, collinear points, the nanometre zigzags that boolean
    snapping leaves) and the sharp teeth tens of nm high that round-join offsets leave where an arc meets a straight edge
    (SPIKE_LEN, SPIKE_TURN_DEG).  Repeats on local minima of the deviation, so neighbours are never judged on stale geometry."""
    p = np.asarray(pts, float)
    while len(p) > 3:
        a, c = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
        ac = c - a
        L2 = (ac ** 2).sum(1)
        t = np.clip(((p - a) * ac).sum(1) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        dev = np.hypot(*(p - (a + t[:, None] * ac)).T)                 # distance of each vertex from its neighbours' chord
        e_in, e_out = p - a, c - p
        l_in, l_out = np.hypot(*e_in.T), np.hypot(*e_out.T)
        cos_turn = (e_in * e_out).sum(1) / np.maximum(l_in * l_out, 1e-18)
        spike = (cos_turn < math.cos(math.radians(SPIKE_TURN_DEG))) & (np.minimum(l_in, l_out) < SPIKE_LEN)
        dev = np.where(spike, 0.0, dev)                                 # artefacts count as removable at once
        cand = (dev < eps) & (dev <= np.roll(dev, 1)) & (dev < np.roll(dev, -1))      # never two neighbours at once
        if not cand.any():                                                           # flat runs of equal deviation
            weak = np.flatnonzero((dev < eps) & (dev <= np.roll(dev, 1)) & (dev <= np.roll(dev, -1)))
            if len(weak) == 0:
                break
            cand = np.zeros(len(p), bool)
            cand[weak[0]] = True
        p = p[~cand]
    return p


def tidy(polys):
    out = []
    for q in polys:
        pts = tidy_points(q.points)
        if len(pts) >= 3 and abs(gdstk.Polygon(pts).area()) > 1e-6:
            out.append(gdstk.Polygon(pts, layer=q.layer, datatype=q.datatype))
    return out


def aperture_of(mesa):
    return grow(mesa, -D_OX)


def aspect_of(polys):
    """Second-moment aspect ratio of a set of polygons: sqrt of the eigenvalue ratio of the area covariance (1 = isotropic)."""
    m = np.zeros(6)
    for p in polys:
        x, y = p.points[:, 0], p.points[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        s = np.sign(cr.sum())
        m += s * np.array([cr.sum() / 2, ((x + x1) * cr).sum() / 6, ((y + y1) * cr).sum() / 6,
                           ((x * x + x * x1 + x1 * x1) * cr).sum() / 12, ((y * y + y * y1 + y1 * y1) * cr).sum() / 12,
                           ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / 24])
    a, cx, cy = m[0], m[1] / m[0], m[2] / m[0]
    cov = np.array([[m[3] / a - cx * cx, m[5] / a - cx * cy], [m[5] / a - cx * cy, m[4] / a - cy * cy]])
    ev = np.linalg.eigvalsh(cov)
    return float(np.sqrt(ev[1] / ev[0]))


def components_and_barrier(apertures):
    n = len(apertures)
    if n < 2:
        return n, 0.0
    best = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            a, b = apertures[i].points, apertures[j].points
            best = min(best, float(np.min(np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]))))
    return n, best


# ----------------------------------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------------------------------

def single(target, nmetal_example):
    """One cruciform VCSEL with the example's single-device template (n-Metal copied from PM_circle_R_20)."""
    mesa = grow(target, D_OX)
    ring = gdstk.boolean(band(mesa, -RING_IN, -RING_OUT), [slot(GAP_WIDTH, (0, 0))], "not", precision=PREC)
    ppad = pad_and_ring(mesa, [mesa])
    ox = passivation(mesa)
    out = dict(mesa=mesa, grid=list(target), nwell=[gdstk.ellipse((0, 0), NWELL_R, tolerance=TOL)], pmetal=clean(ring),
               ppad=ppad, ox=ox, nmetal=clean(union([gdstk.Polygon(p.points) for p in nmetal_example])))
    return out


def pad_and_ring(mesa, element_mesas, keep_off=None):
    """p-MetalPad: outer outline (element bands out to PPAD_OUT, lead, pad) cleaned BEFORE the inner hole is cut, so that the
    clean-up never sees a keyholed polygon; the hole is the region inside RING_IN of each element (and keep_off, if given)."""
    outer = union(*[grow(m, PPAD_OUT) for m in element_mesas], grow(mesa, PPAD_OUT))
    xl = left_edge_at_y0(outer)
    lead = gdstk.Polygon([(-42.5, -31.0), (-42.5, 31.0), (xl + 4.0, 6.0), (xl + 4.0, -6.0)])
    outer = clean(union(outer, [lead, gdstk.rectangle((-130.0, -31.0), (-42.5, 31.0))]), r_close=JUNCTION_R)
    holes = union(*[grow(m, -RING_IN) for m in element_mesas])     # per element only: no slit along closed bridges
    res = gdstk.boolean(outer, holes, "not", precision=PREC)
    if keep_off:
        res = gdstk.boolean(res, keep_off, "not", precision=PREC)
    return res


def passivation(mesa):
    """Oxide/Nitride: band from OX_IN inside to OX_OUT outside the mesa, trapezoid and field under the pad (example)."""
    outer = grow(mesa, OX_OUT)
    xo = left_edge_at_y0(outer)
    trap = gdstk.Polygon([(-42.0, -42.0), (-42.0, 42.0), (xo + 3.0, 14.0), (xo + 3.0, -14.0)])
    outer = clean(union(outer, [trap, gdstk.rectangle((-138.6, -42.0), (-42.0, 42.0))]), r_close=JUNCTION_R)
    return gdstk.boolean(outer, grow(mesa, -OX_IN), "not", precision=PREC)


def bridge(c0, c1, width):
    """Rectangle along the line c0 -> c1, `width` wide, reaching BRIDGE_LEN_OVERLAP into each mesa (mesa edges found later)."""
    (x0, y0), (x1, y1) = c0, c1
    L = math.hypot(x1 - x0, y1 - y0)
    p = gdstk.rectangle((-L / 2, -width / 2), (L / 2, width / 2))
    p.rotate(math.atan2(y1 - y0, x1 - x0))
    p.translate((x0 + x1) / 2, (y0 + y1) / 2)
    return p


def n_metal_generic(r_in, r_out):
    """n-Metal 'D' sized to clear a larger structure: annular sector (+-118 deg, opening towards the pads), the example's two
    pads, and convex fillers joining each pad to the sector end."""
    ang = math.radians(118.0)
    sector = gdstk.ellipse((0, 0), r_out, inner_radius=r_in, initial_angle=-ang, final_angle=ang, tolerance=TOL)
    parts = [sector]
    for sgn in (1, -1):
        pad = gdstk.rectangle((-138.4, sgn * 55.5), (-41.0, sgn * 125.5))
        e_in = (r_in * math.cos(ang), sgn * r_in * math.sin(ang))
        e_out = (r_out * math.cos(ang), sgn * r_out * math.sin(ang))
        pts = np.array([(-41.0, sgn * 55.5), (-41.0, sgn * 125.5), e_out, e_in])
        hull = pts[_convex_hull(pts)]
        parts += [pad, gdstk.Polygon(hull)]
    return union(parts)


def _convex_hull(pts):
    from scipy.spatial import ConvexHull
    return ConvexHull(pts).vertices


def multi(target_fn, centres, bridge_width, nwell=True):
    """Coupled structure (pair or hex array): one mesa per element, bridges between nearest neighbours (width = bridge_width),
    one C ring per element, shared p-pad and n-metal (as the example's pairs and arrays)."""
    targets = []
    for c in centres:
        targets += target_fn(c)
    mesas = grow(targets, D_OX)
    # nearest-neighbour bridges
    cs = np.array(centres)
    dmin = min(np.hypot(*(cs[i] - cs[j])) for i in range(len(cs)) for j in range(i + 1, len(cs)))
    bridges = []
    if bridge_width > 0:
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if np.hypot(*(cs[i] - cs[j])) < 1.05 * dmin:
                    bridges.append(bridge(tuple(cs[i]), tuple(cs[j]), bridge_width))
        # trim bridges to the part between the element centres' mesas plus BRIDGE_LEN_OVERLAP inside each
        bridges = gdstk.boolean(bridges, grow(mesas, -BRIDGE_LEN_OVERLAP), "not", precision=PREC)
    mesa = union(mesas, bridges)
    rings = []
    for k, c in enumerate(centres):
        own = grow(target_fn(c), D_OX)
        rings += gdstk.boolean(band(own, -RING_IN, -RING_OUT), [slot(GAP_WIDTH, c)], "not", precision=PREC)
    pband = union(*[band(grow(target_fn(c), D_OX), -RING_IN, PPAD_OUT) for c in centres], band(mesa, -RING_IN, PPAD_OUT))
    xl = left_edge_at_y0(pband)
    lead = gdstk.Polygon([(-42.5, -31.0), (-42.5, 31.0), (xl + 4.0, 6.0), (xl + 4.0, -6.0)])
    ppad = union(pband, [lead, gdstk.rectangle((-130.0, -31.0), (-42.5, 31.0))])
    oband = band(mesa, -OX_IN, OX_OUT)
    xo = left_edge_at_y0(oband)
    trap = gdstk.Polygon([(-42.0, -42.0), (-42.0, 42.0), (xo + 3.0, 14.0), (xo + 3.0, -14.0)])
    ox = union(oband, [trap, gdstk.rectangle((-138.6, -42.0), (-42.0, 42.0))])
    r_ext = max(float(np.hypot(p.points[:, 0], p.points[:, 1]).max()) for p in oband)
    r_in = max(60.0, r_ext + N_CLEAR)
    out = dict(mesa=mesa, grid=targets, pmetal=rings, ppad=ppad, ox=ox, nmetal=n_metal_generic(r_in, r_in + 50.0),
               nwell=[gdstk.ellipse(c, NWELL_R, tolerance=TOL) for c in centres] if nwell else [])
    return out, dict(n_rin=r_in, bridge_width=bridge_width)


def spacing_for(target_fn, centres_fn, n_expected, mesa_gap):
    """Smallest scale s at which neighbouring MESAS are separated by at least `mesa_gap` um (bridges then set the coupling)."""
    lo, hi = 1.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        t = []
        for c in centres_fn(mid):
            t.append(grow(target_fn(c), D_OX))
        ok = True
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                a = np.vstack([p.points for p in t[i]])
                b = np.vstack([p.points for p in t[j]])
                inter = gdstk.boolean(t[i], grow(t[j], mesa_gap), "and", precision=PREC)
                if inter:
                    ok = False
                    break
            if not ok:
                break
        lo, hi = (lo, mid) if ok else (mid, hi)
        if hi - lo < 0.01:
            break
    return hi


def pair_centres(s):
    return [(0.0, -s / 2), (0.0, s / 2)]


def hex7_centres(p):
    return [(0.0, 0.0)] + [(p * math.cos(math.radians(30 + 60 * k)), p * math.sin(math.radians(30 + 60 * k))) for k in range(6)]


# ----------------------------------------------------------------------------------------------
# Alignment strip copied from the example; labels
# ----------------------------------------------------------------------------------------------

TDB2GDS = {3: 1, 4: 2, 5: 3, 7: 4, 18: 5, 19: 6, 30: 7}


def flatten(cells, cell, dx=0.0, dy=0.0, out=None):
    out = [] if out is None else out
    for s in cell.shapes:
        if s.layer not in TDB2GDS:
            continue
        pts = np.array(T.shape_points(s, TOL))
        if len(pts) < 3 or (pts.max(0) - pts.min(0)).max() < 0.5:       # drop the sub-0.5 um specks of the example
            continue
        out.append((TDB2GDS[s.layer], pts + [dx, dy]))
    for inst in cell.instances:
        flatten(cells, cells[inst.cell_index - 1], dx + inst.x, dy + inst.y, out)
    return out


FONT = {   # 5 x 7 pixel glyphs (rows top to bottom); drawn as explicit rectangles, one per horizontal run of pixels
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"], "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"], "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"], "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"], "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"], "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"], "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"], "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"], "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"], "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"], "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"], "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"], " ": ["00000"] * 7,
}


def solid_glyph(rows):
    """Glyph bitmap without pixels that touch only at a corner (a zero-width neck on the mask): wherever two pixels meet
    diagonally with both side pixels empty, the side pixel in the upper row is filled.  Repeats until none is left."""
    g = [list(r) for r in rows]
    changed = True
    while changed:
        changed = False
        for r in range(len(g) - 1):
            for c in range(len(g[0]) - 1):
                a, b, d, e = g[r][c], g[r][c + 1], g[r + 1][c], g[r + 1][c + 1]
                if a == "1" and e == "1" and b == "0" and d == "0":
                    g[r][c + 1] = "1"
                    changed = True
                elif b == "1" and d == "1" and a == "0" and e == "0":
                    g[r][c] = "1"
                    changed = True
    return ["".join(r) for r in g]


def label(text, x, y, pix=2.0, mirror=None):
    """Text from 2 um pixels (strokes >= 2 um wide, no corner-only contacts); (x, y) is the lower-left corner.  With mirror
    (default LABEL_MIRROR) the text is reflected x -> -x about its own centre, like the lab's strip text."""
    mirror = LABEL_MIRROR if mirror is None else mirror
    rects = []
    for k, ch in enumerate(text.upper()):
        g = solid_glyph(FONT[ch])
        x0 = x + k * 6 * pix
        for r, row in enumerate(g):
            yy = y + (6 - r) * pix
            c = 0
            while c < 5:
                if row[c] == "1":
                    c1 = c
                    while c1 + 1 < 5 and row[c1 + 1] == "1":
                        c1 += 1
                    rects.append(gdstk.rectangle((x0 + c * pix, yy), (x0 + (c1 + 1) * pix, yy + pix)))
                    c = c1 + 1
                else:
                    c += 1
    merged = gdstk.boolean(rects, [], "or", precision=PREC)
    if mirror and merged:
        xc = x + (len(text) * 6 - 1) * pix / 2.0                       # centre of the text block (5 of every 6 columns inked)
        for p in merged:
            p.mirror((xc, 0.0), (xc, 1.0))
    for p in merged:
        p.layer = LAY[LABEL_LAYER]
    return merged


# ----------------------------------------------------------------------------------------------
# Content of the group (targets: L, ratio, w, rho, theta; MODEL = yield predicted in the 5 um target convention,
# results/mask2_target5_scores.json; 'n' = devices per group)
# ----------------------------------------------------------------------------------------------
FA = (17.5, 0.7, 6.0, 0.0)          # mask-1 family A target (sharp tips): the comparable shape
# Selection decided with Polina on 2026-10-09: an "anisotropy ruler" on the same wafer and run as mask 1 (mask 1 already
# covers the rotation angles 0-150 deg with families A and B).  All devices at 90 deg = long axis perpendicular to the
# mask x axis, which must be aligned to the native polarization axis [011] of the wafer.  No arrays or pairs.
SINGLES = [
    # id,     role,                                   target (L, ratio, w, rho, theta),                n
    ("X9",   "ruler: cross 1:0.9 at 90 deg",          (17.5, 0.9, 6.0, 0.0, 90.0), 4),
    ("X8",   "ruler: cross 1:0.8 at 90 deg",          (17.5, 0.8, 6.0, 0.0, 90.0), 4),
    ("X7",   "ruler: cross 1:0.7 at 90 deg (= mask-1 A90)", (17.5, 0.7, 6.0, 0.0, 90.0), 4),
    ("X6",   "ruler: cross 1:0.6 at 90 deg",          (17.5, 0.6, 6.0, 0.0, 90.0), 4),
    ("X5",   "ruler: cross 1:0.5 at 90 deg",          (17.5, 0.5, 6.0, 0.0, 90.0), 4),
    ("X4",   "ruler: cross 1:0.4 at 90 deg",          (17.5, 0.4, 6.0, 0.0, 90.0), 4),
    ("S130", "stadium, aspect 1.30, at 90 deg",       (17.0, (17.0 / 1.30) / 17.0, 17.0 / 1.30, 1.0, 90.0), 3),
    ("S155", "stadium, aspect 1.55, at 90 deg",       (17.0, (17.0 / 1.55) / 17.0, 17.0 / 1.55, 1.0, 90.0), 3),
    ("S180", "stadium, aspect 1.80, at 90 deg",       (17.0, (17.0 / 1.80) / 17.0, 17.0 / 1.80, 1.0, 90.0), 3),
    ("S220", "stadium, aspect 2.20, at 90 deg",       (17.0, (17.0 / 2.20) / 17.0, 17.0 / 2.20, 1.0, 90.0), 3),
    ("CIRC", "control: circle (native axis)",         (13.47, 1.0, 13.47, 1.0, 0.0), 2),
    ("SYM",  "control: symmetric cross",              (17.5, 1.0, 6.0, 0.0, 0.0), 2),
    ("X7S",  "smaller aperture: 0.75 x cross 1:0.7 at 90 deg", (13.125, 0.7, 4.5, 0.0, 90.0), 2),
]
MULTIS = []                         # arrays and pairs removed (2026-10-09)
MULTI_TARGET = FA + (80.0,)         # the coupled structures use R80
MESA_GAP = 2.0                      # um between neighbouring mesas in pairs/arrays (as the example's array)


def spoke_bridges_only(centres):
    """Bridges from the centre element to each ring element only (ring-to-ring bridges enclose islands that stay unoxidised)."""
    return [(0, j) for j in range(1, len(centres))]


def build_multi(kind, wb, nmetal_example=None):
    tf = lambda c=(0.0, 0.0): cross_target(*MULTI_TARGET, centre=c)  # noqa: E731
    fn = pair_centres if kind == "pair" else hex7_centres
    n = 2 if kind == "pair" else 7
    s = spacing_for(tf, fn, n, MESA_GAP)
    centres = fn(s)
    targets = []
    for c in centres:
        targets += tf(c)
    mesas = grow(targets, D_OX)
    links = [(0, 1)] if kind == "pair" else spoke_bridges_only(centres)
    br = [bridge(centres[i], centres[j], wb) for i, j in links]
    br = gdstk.boolean(br, grow(mesas, -BRIDGE_LEN_OVERLAP), "not", precision=PREC)
    mesa = union(mesas, br)
    mesa = clean(mesa)
    ap = aperture_of(mesa)
    keep_off = grow(ap, 1.5)                    # metal stays >= 1.5 um from the predicted aperture, also at necks and junctions
    rings = []
    for c in centres:
        rings += gdstk.boolean(band(grow(tf(c), D_OX), -RING_IN, -RING_OUT), [slot(GAP_WIDTH, c)], "not", precision=PREC)
    rings = gdstk.boolean(clean(rings), keep_off, "not", precision=PREC)
    rings = grow(grow(rings, -CLEAN_R), CLEAN_R)  # opening: drop ring fragments narrower than 2 * CLEAN_R left by the cut
    element_mesas = [grow(tf(c), D_OX) for c in centres]
    ppad = pad_and_ring(mesa, element_mesas, keep_off)
    ox = passivation(mesa)
    r_ext = max(float(np.hypot(p.points[:, 0], p.points[:, 1]).max()) for p in grow(mesa, OX_OUT))   # outer edge of the band
    r_in = max(60.0, r_ext + N_CLEAR)
    out = dict(mesa=mesa, grid=targets, pmetal=rings, ppad=ppad, ox=ox,
               nmetal=clean(n_metal_generic(r_in, r_in + 50.0)), nwell=[gdstk.ellipse(c, NWELL_R, tolerance=TOL) for c in centres])
    ap = aperture_of(out["mesa"])
    ncomp, barr = components_and_barrier(ap)
    return out, dict(spacing=s, centres=[[round(x, 3), round(y, 3)] for x, y in centres], n_rin=r_in, n_rout=r_in + 50.0,
                     apertures=ncomp, barrier_um=round(barr, 2), aperture_areas=[round(a.area(), 1) for a in ap],
                     neck_um=round(max(wb - 2 * D_OX, 0.0), 2))


SERIES = [["X9", "X8", "X7", "X6", "X5", "X4"], ["S130", "S155", "S180", "S220"]]


def _centroids(assign, nc):
    pos = {}
    for k, sid in enumerate(assign):
        pos.setdefault(sid, []).append(divmod(k, nc))
    return pos


def _layout_cost(assign, nr, nc):
    """Penalty of a slot assignment (list of design IDs, row-major):
    * copies of a design sharing a row or a column (hard);
    * the distance of each design's centroid from the device-grid centre (weighted by the number of copies);
    * a linear trend of the centroids along each design series (X9..X4, S130..S220), which a gradient across the group would
      turn into a bias along the series."""
    pos = _centroids(assign, nc)
    pen = 0.0
    cen = {}
    for sid, p in pos.items():
        rows = [q[0] for q in p]
        cols = [q[1] for q in p]
        pen += 100.0 * (len(rows) - len(set(rows)) + len(cols) - len(set(cols)))
        mr, mc = sum(rows) / len(p) - (nr - 1) / 2, sum(cols) / len(p) - (nc - 1) / 2
        cen[sid] = (mr, mc)
        pen += len(p) * (mr * mr + mc * mc)
    for ser in SERIES:
        k = np.arange(len(ser)) - (len(ser) - 1) / 2
        for ax in (0, 1):
            v = np.array([cen[s][ax] for s in ser if s in cen])
            if len(v) == len(ser):
                pen += 10.0 * (float((k * v).sum() / (k * k).sum()) * len(ser)) ** 2
    return pen


def _exact_layout(ids_n, nr, nc):
    """Integer program (scipy/HiGHS) for the slot assignment.  Every slot holds one device; every design has its n copies in
    different rows and different columns, never diagonally adjacent; every design's mean row is exactly the grid's middle
    row; designs with an even number
    of copies sit exactly on the middle column, and the 3-copy stadiums (whose column sums cannot be a half-integer) sit 1/6
    column off it with signs +, -, -, + along S130..S220, which leaves no linear trend along the series.
    Returns the row-major list of design IDs, or None if HiGHS is unavailable or finds no solution."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        return None
    names = [s for s, _ in ids_n]
    nd, ncell = len(names), nr * nc
    nv = nd * ncell
    idx = lambda d, r, c: d * ncell + r * nc + c          # noqa: E731
    A, lo, hi = [], [], []

    def add(coef, a, b):
        row = np.zeros(nv)
        for k, v in coef:
            row[k] = v
        A.append(row)
        lo.append(a)
        hi.append(b)

    for r in range(nr):
        for c in range(nc):
            add([(idx(d, r, c), 1) for d in range(nd)], 1, 1)
    stad_sign = {s: sg for s, sg in zip(SERIES[1], (+1, -1, -1, +1))}
    for d, (sid, n) in enumerate(ids_n):
        add([(idx(d, r, c), 1) for r in range(nr) for c in range(nc)], n, n)
        for r in range(nr):
            add([(idx(d, r, c), 1) for c in range(nc)], 0, 1)
        for c in range(nc):
            add([(idx(d, r, c), 1) for r in range(nr)], 0, 1)
        for r in range(nr - 1):                                    # no two copies diagonally adjacent either
            for c in range(nc):
                for c2 in (c - 1, c + 1):
                    if 0 <= c2 < nc:
                        add([(idx(d, r, c), 1), (idx(d, r + 1, c2), 1)], 0, 1)
        add([(idx(d, r, c), r) for r in range(nr) for c in range(nc)], n * (nr - 1) / 2, n * (nr - 1) / 2)
        col_target = n * (nc - 1) / 2
        if abs(col_target - round(col_target)) > 1e-9:            # odd n with an even number of columns: +-1/2 in the sum
            col_target += 0.5 * stad_sign.get(sid, 1)
        add([(idx(d, r, c), c) for r in range(nr) for c in range(nc)], col_target, col_target)
    res = milp(c=np.zeros(nv), constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options=dict(disp=False, time_limit=120))
    if res.x is None:
        return None
    x = np.round(res.x).astype(int).reshape(nd, nr, nc)
    return [names[int(np.argmax(x[:, r, c]))] for r in range(nr) for c in range(nc)]


def slot_plan(seed=2026, iters=60000, restarts=8):
    """42 slots (6 columns x 7 rows), single devices only.  The copies of a design go to different rows AND different columns,
    each design's copies are centred on the device grid, and the small offsets that cannot be avoided (3 copies over 6 columns)
    cancel along each series, so that a gradient of oxidation depth or exposure across the group is not confounded with the
    ratio or aspect series.  Solved exactly as an integer program (_exact_layout); a seeded swap search (simulated annealing)
    is the fallback.  Both are deterministic, so a rebuild reproduces the layout; the constraints are asserted below."""
    ids, meta = [], {}
    for sid, role, tgt, n in SINGLES:
        ids += [sid] * n
        meta[sid] = (role, tgt)
    nr, nc = len(ROWS), len(COLS)
    assert len(ids) == nr * nc, len(ids)
    best = _exact_layout([(sid, n) for sid, _, _, n in SINGLES], nr, nc)
    rng = np.random.default_rng(seed)
    for _ in range(0 if best is not None else restarts):
        cur = list(rng.permutation(ids))
        cur_cost = _layout_cost(cur, nr, nc)
        best_cost = _layout_cost(best, nr, nc) if best is not None else float("inf")
        for it in range(iters):
            temp = 3.0 * (1.0 - it / iters) + 1e-3
            i, j = rng.integers(0, len(cur), 2)
            if cur[i] == cur[j]:
                continue
            cur[i], cur[j] = cur[j], cur[i]
            c = _layout_cost(cur, nr, nc)
            if c <= cur_cost or rng.random() < math.exp((cur_cost - c) / temp):
                cur_cost = c
                if c < best_cost:
                    best, best_cost = cur[:], c
            else:
                cur[i], cur[j] = cur[j], cur[i]
    plan = []
    for k, sid in enumerate(best):
        r, c = divmod(k, nc)
        role, tgt = meta[sid]
        plan.append(dict(slot=f"r{r + 1}c{c + 1}", col=c, row=r, id=sid, role=role, target=tgt, kind="single"))
    for sid in meta:
        mine = [d for d in plan if d["id"] == sid]
        assert len({d["row"] for d in mine}) == len(mine), f"{sid}: two copies in one row"
        assert len({d["col"] for d in mine}) == len(mine), f"{sid}: two copies in one column"
    return plan


def main():
    src, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    cells = T.read(src)
    byname = {c.name: c for c in cells}
    nmetal_example = [gdstk.Polygon(np.array(T.shape_points(s, TOL))) for s in byname["PM_circle_R_20"].shapes
                      if s.layer == 4 and np.ptp(np.array(T.shape_points(s, TOL)), 0).max() > 0.5]
    lib = gdstk.Library("PM_GROUP_2", unit=1e-6, precision=1e-9)
    top = lib.new_cell("PM_Group_2")
    # alignment strip, exactly as in the example (PM_alignment instance at (0, 0) of PM_Group_1)
    for lay, pts in flatten(cells, byname["PM_alignment"]):
        top.add(gdstk.Polygon(pts, layer=lay))
    res = Path(__file__).resolve().parents[1] / "results"            # in the repository
    if not res.exists():
        res = Path(__file__).resolve().parents[1] / "model"          # in the delivered package (code/ next to model/)
    scores = {}
    ns = json.loads((res / "native_strength_shift.json").read_text()) if (res / "native_strength_shift.json").exists() else {}
    sel = json.loads((res / "mask2_selection_scores.json").read_text())["designs"] if (res / "mask2_selection_scores.json").exists() else {}
    key_ns = {"X9": "r0.9", "X8": "r0.8", "X7": "r0.7", "X6": "r0.6", "X5": "r0.5", "X4": "r0.4",
              "S130": "st1.3", "S155": "st1.6", "S180": "st1.8", "S220": "st2.2"}
    key_sel = {"CIRC": "CIRC", "SYM": "CX0", "X7S": "A90_s075"}
    ex = json.loads((res / "mask2_extras_native.json").read_text()) if (res / "mask2_extras_native.json").exists() else {}
    ro = json.loads((res / "mask2_readouts.json").read_text()) if (res / "mask2_readouts.json").exists() else {}
    an = json.loads((res / "mask2_anisotropy_along_ruler.json").read_text()) if (res / "mask2_anisotropy_along_ruler.json").exists() else {}
    start = {sid: {f"x{m}": ro[f"x{m}"][sid]["start"]["mean"] for m in (1, 2, 3)}
             for sid, *_ in SINGLES if ro and all(sid in ro.get(f"x{m}", {}) for m in (1, 2, 3))}
    for sid, *_ in SINGLES:
        if sid in key_ns and ns:
            scores[sid] = {f"x{m}": ns[f"x{m}"][key_ns[sid]]["mean"] for m in (1, 2, 3)}
        elif ex and all(sid in ex.get(f"x{m}", {}) for m in (1, 2, 3)):
            scores[sid] = {f"x{m}": ex[f"x{m}"][sid]["mean"] for m in (1, 2, 3)}
        elif sid in key_sel and sel:
            scores[sid] = {"x1": sel[key_sel[sid]]["mean"]}
    rows, report = [], dict(devices=[], parameters=dict(D_OX=D_OX, RING_OUT=RING_OUT, RING_IN=RING_IN, OX_IN=OX_IN, PPAD_OUT=PPAD_OUT,
                                                        OX_OUT=OX_OUT, GAP_WIDTH=GAP_WIDTH, NWELL_R=NWELL_R, CLEAN_R=CLEAN_R,
                                                        TIDY_EPS=TIDY_EPS, MAX_POINTS=MAX_POINTS, JUNCTION_R=JUNCTION_R, LABEL_LAYER=LABEL_LAYER,
                                                        LABEL_MIRROR=LABEL_MIRROR, MESA_GAP=MESA_GAP))
    for d in slot_plan():
        x, y = COLS[d["col"]], ROWS[d["row"]]
        if d["kind"] == "single":
            geo = single(cross_target(*d["target"]), nmetal_example)
        else:
            geo, info = build_multi(d["kind"], d["bridge"])
        # place, then tidy (on the final coordinates, so the 1 nm rounding of the writer cannot undo it)
        geo = {k: tidy([gdstk.Polygon(p.points + [x, y], layer=LAY[k]) for p in v]) for k, v in geo.items()}
        for polys in geo.values():
            for p in polys:
                top.add(p)
        if d["kind"] == "single":                     # apertures of the mesa exactly as written
            aps = aperture_of(geo["mesa"])
            info = dict(apertures=len(aps), aperture_areas=[round(a.area(), 1) for a in aps],
                        aperture_aspect=round(aspect_of(aps), 3))
        for p in label(d["id"], x - 136.0, y + 133.0):
            top.add(p)
        sc = scores.get(d["id"])
        L, ratio, w, rho, th = d["target"]
        rows.append(dict(slot=d["slot"], x_um=round(x, 1), y_um=round(y, 1), id=d["id"], kind=d["kind"], role=d["role"],
                         target_L=L, target_short=round(L * ratio, 3), target_w=w, target_tip_radius=round(rho * w / 2, 3), rotation_deg=th,
                         mesa_L=L + 2 * D_OX, mesa_w=w + 2 * D_OX,
                         predicted_aperture_um2="; ".join(str(a) for a in info["aperture_areas"]),
                         predicted_aperture_aspect=info.get("aperture_aspect", ""),
                         model_share_native_x1=round(sc["x1"], 2) if sc and "x1" in sc else "",
                         model_share_native_x2=round(sc["x2"], 2) if sc and "x2" in sc else "",
                         model_share_native_x3=round(sc["x3"], 2) if sc and "x3" in sc else "",
                         **{f"model_start_01m1_native_x{m}": round(start[d["id"]][f"x{m}"], 2) if d["id"] in start else ""
                            for m in (1, 2, 3)},
                         model_mode_splitting_GHz=round(an["x1"][d["id"]]["splitting_GHz_median"], 1)
                         if an and d["id"] in an.get("x1", {}) else ""))
        report["devices"].append(dict(slot=d["slot"], id=d["id"], **{k: v for k, v in info.items()}))
    lib.write_gds(str(outdir / "VCSEL_PM_Group_2.gds"), max_points=MAX_POINTS)
    with open(outdir / "VCSEL_PM_Group_2_devices.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (outdir / "build_report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(rows)} devices written to {outdir / 'VCSEL_PM_Group_2.gds'}")


if __name__ == "__main__":
    main()
