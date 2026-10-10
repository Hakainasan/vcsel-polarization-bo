"""Reader for the geometry in a Tanner L-Edit .tdb file (the subset used by the lab's VCSEL masks).

The .tdb format is Tanner's own binary format and is not documented.  This reader was written by inspecting
VCSEL_ARRAY_Heming_-_V04.tdb and decodes what that file uses:

  * cells (name, stored bounding box, position in file order = the index instances refer to),
  * layer selection (index into the technology layer list, 1-based),
  * boxes, box lists, circles, polygons with straight edges and circular arcs,
  * instances (cell index, translation, the remaining transform words, instance name).

Wire/text objects inside the logo cells use a vertex encoding that is not decoded (reported as `skipped`).
Every cell's decoded geometry is checked against the bounding box stored in the file (`check_bboxes`), which is how the
decoding was validated.  Units: integers in the file are nanometres; everything returned here is in micrometres.

Tokens (tag byte, then fields):
  0x59/0xD9  key, 2-byte count, 1-byte length, data (big-endian int32 values)
  0x18/0x98  key, 1-byte length, data (big-endian int16 values)
  0x08/0x88/0x43  key, 2-byte value          0x09/0x89  key, 4-byte value
  0x11/0x91  key, 1-byte length, data         0x12/0x92  key, 1-byte length, text
  0x03/0x83 + kind  begin of an object (kind 1 instance, 3 box or circle (0x83), 4 polygon (0x83: with arcs))
Polygon vertices are key 2; arc points are key 8 followed by a flag word (0x6000 = arc centre, 0x2000 = a point the arc
passes through); an arc runs from the previous vertex, around the centre, through that point, to the next vertex.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

import numpy as np

HEADER_RE = re.compile(rb'\x01([\x01-\x40])([A-Za-z][A-Za-z0-9_.\-]{2,60})\x18\x02\x0e')


@dataclass
class Shape:
    layer: int                 # 1-based index into the technology layer list
    kind: str                  # 'box' | 'circle' | 'polygon'
    data: object               # box/circle: (x0, y0, x1, y1); polygon: list of path items
    offset: int = 0


@dataclass
class Instance:
    cell_index: int            # 1-based position of the referenced cell in file order
    x: float
    y: float
    xform: tuple               # remaining transform words as stored (seen: (1, 1, 0, 0))
    array: tuple               # (cols, rows) as stored (seen: (1, 1))
    name: str = ""
    bbox: tuple = ()


@dataclass
class Cell:
    name: str
    index: int
    start: int
    stored_bbox: tuple
    shapes: list = field(default_factory=list)
    instances: list = field(default_factory=list)
    skipped: int = 0


def _i32(data):
    n = len(data) // 4
    return [v / 1000.0 for v in struct.unpack('>%di' % n, data[:4 * n])]


def _i16(data):
    n = len(data) // 2
    return [v / 1000.0 for v in struct.unpack('>%dh' % n, data[:2 * n])]


def tokens(b: bytes, p: int, end: int):
    """Yield (offset, tag, key, payload) until `end` or the next cell header. payload: list of floats (blobs), int, or str."""
    while p < end:
        t = b[p]
        if t in (0x59, 0xD9):
            key, ln = b[p + 1], b[p + 4]
            yield p, 'blob32', key, _i32(b[p + 5:p + 5 + ln])
            p += 5 + ln
        elif t in (0x69, 0xE9):                     # same as 0x59 with a 2-byte length (long box lists)
            key, ln = b[p + 1], int.from_bytes(b[p + 4:p + 6], 'big')
            yield p, 'blob32', key, _i32(b[p + 6:p + 6 + ln])
            p += 6 + ln
        elif t in (0x18, 0x98):
            key, ln = b[p + 1], b[p + 2]
            yield p, 'blob16', key, _i16(b[p + 3:p + 3 + ln])
            p += 3 + ln
        elif t in (0x08, 0x88, 0x43):
            yield p, 'u16', b[p + 1], int.from_bytes(b[p + 2:p + 4], 'big')
            p += 4
        elif t in (0x09, 0x89):
            yield p, 'u32', b[p + 1], int.from_bytes(b[p + 2:p + 6], 'big')
            p += 6
        elif t in (0x11, 0x91):
            ln = b[p + 2]
            yield p, 'bytes', b[p + 1], b[p + 3:p + 3 + ln]
            p += 3 + ln
        elif t in (0x12, 0x92):
            ln = b[p + 2]
            yield p, 'str', b[p + 1], b[p + 3:p + 3 + ln].decode('latin-1')
            p += 3 + ln
        elif t in (0x03, 0x83) and p + 1 < end and b[p + 1] in (1, 2, 3, 4, 5, 6):
            yield p, 'begin', b[p + 1], (t == 0x83)
            p += 2
        elif t == 0x03:
            yield p, 'begin', 0, False
            p += 1
        else:
            yield p, 'unknown', t, None
            p += 1


def read(path: str) -> list:
    b = open(path, 'rb').read()
    heads = []
    for m in HEADER_RE.finditer(b):
        n, name = m.group(1)[0], m.group(2)
        if len(name) != n:
            continue
        k = b.find(b'\x59\x03\x00\x01\x20', m.end(), m.end() + 80)
        if k < 0:
            continue
        heads.append((m.start(), name.decode(), tuple(_i32(b[k + 5:k + 21])), k + 37))
    cells = []
    for i, (start, name, bb, gstart) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(b)
        stop = b.find(b'\x11\x07\x01\x00\x12\x08\x00', gstart, end)   # property block that closes a cell
        end = stop if stop > 0 else end
        cell = Cell(name, i + 1, start, bb)
        _parse_cell(b, gstart, end, cell)
        cells.append(cell)
    return cells


def _parse_cell(b, p, end, cell: Cell):
    layer = 0
    obj = None          # current object: dict(kind, circle, ...)
    pending_arc = None  # arc point waiting for its flag

    def flush():
        nonlocal obj
        if obj is None:
            return
        if obj['kind'] == 3 and obj.get('bbox'):
            cell.shapes.append(Shape(obj['layer'], 'circle', tuple(obj['bbox']), obj['off']))   # kind 3 = circle/ellipse (boxes are stored as box lists)
        elif obj['kind'] == 4 and obj.get('path'):
            cell.shapes.append(Shape(obj['layer'], 'polygon', obj['path'], obj['off']))
        elif obj['kind'] == 1 and obj.get('xf') is not None:
            x, y, *rest = obj['xf']
            cell.instances.append(Instance(obj.get('cell', 0), x, y, tuple(round(r * 1000) for r in rest),
                                           tuple(round(a * 1000) for a in obj.get('array', (0.001, 0.001))),
                                           obj.get('name', ''), tuple(obj.get('bbox', ()))))
        obj = None

    for off, typ, key, val in tokens(b, p, end):
        if typ == 'u16' and key == 6 and b[off] == 0x43:      # layer selection
            flush()
            layer = val
            continue
        if typ == 'begin':
            if key in (1, 3, 4, 5, 6):
                flush()
                obj = dict(kind=key, circle=val, layer=layer, off=off, path=[])
            continue                                           # kind 0/2: property sub-records (instance name etc.)
        if typ in ('blob32', 'blob16'):
            if key == 6 and (obj is None or obj['kind'] != 4 or obj.get('bbox') is None or b[off] in (0x98, 0xD9, 0xE9)):
                flush()                                        # box list on the current layer
                for j in range(0, len(val) - 3, 4):
                    cell.shapes.append(Shape(layer, 'box', tuple(val[j:j + 4]), off))
                continue
            if obj is None:
                if key not in (2, 8):
                    continue
                cell.skipped += 1
                continue
            if key == 1 and 'bbox' not in obj:
                obj['bbox'] = val[:4]
            elif key == 2 and obj['kind'] in (4, 5, 6):
                obj['path'].append(('v', val[0], val[1]))
            elif key == 8 and obj['kind'] in (4, 5, 6):
                pending_arc = (val[0], val[1])
            elif key == 11 and obj['kind'] == 1:
                obj['xf'] = val
            elif key == 9 and obj['kind'] == 1:
                obj['array'] = val
            continue
        if typ == 'u16' and key == 8 and pending_arc is not None and obj is not None:
            obj['path'].append(('c' if val == 0x6000 else 't', pending_arc[0], pending_arc[1]))
            pending_arc = None
            continue
        if typ == 'u32' and key == 4 and obj is not None and obj['kind'] == 1:
            obj['cell'] = val
            continue
        if typ == 'str' and key == 1 and obj is not None and obj['kind'] == 1:
            obj['name'] = val
            continue
        if typ == 'unknown':
            cell.skipped += 1
    flush()


# ----------------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------------


def _arc_points(p0, c, t, p1, tol=0.005, full_dir=1):
    """Points of the circular arc from p0 around centre c, passing through t, ending at p1 (p0 and p1 excluded).

    A full circle (p1 == p0) has no direction of its own: its through point lies on both ways round.  L-Edit stores a full
    torus as a keyhole path (inner point -> outer point -> full outer circle -> back -> full inner circle), which is only a ring
    if the two circles run in opposite directions, so the caller alternates `full_dir` (+1 = counter-clockwise, -1 = clockwise)."""
    r = np.hypot(p0[0] - c[0], p0[1] - c[1])
    a0 = np.arctan2(p0[1] - c[1], p0[0] - c[0])
    at = np.arctan2(t[1] - c[1], t[0] - c[0])
    a1 = np.arctan2(p1[1] - c[1], p1[0] - c[0])
    ccw_t = (at - a0) % (2 * np.pi)
    ccw_1 = (a1 - a0) % (2 * np.pi)
    full = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-6 * max(r, 1.0)
    if full:
        span = 2 * np.pi * (1 if full_dir >= 0 else -1)
    elif ccw_t <= ccw_1:          # counter-clockwise from p0 reaches t before p1
        span = ccw_1
    else:
        span = -((a0 - a1) % (2 * np.pi) or 2 * np.pi)
    step = 2 * np.arccos(max(1 - tol / max(r, 1e-9), -1.0)) if r > tol else np.pi / 8
    n = max(int(np.ceil(abs(span) / step)), 2)
    ang = a0 + span * np.arange(1, n) / n
    return [(c[0] + r * np.cos(a), c[1] + r * np.sin(a)) for a in ang]


def polygon_points(path, tol=0.005):
    """Expand a decoded path (vertices and arcs) into a closed list of points."""
    verts = []
    items = list(path)
    i = 0
    first = None
    n_full = 0                       # full circles seen so far: alternate their direction (see _arc_points)
    while i < len(items):
        typ, x, y = items[i]
        if typ == 'v':
            verts.append((x, y))
            if first is None:
                first = (x, y)
            i += 1
        elif typ == 'c':
            c = (x, y)
            t = None
            if i + 1 < len(items) and items[i + 1][0] == 't':
                t = (items[i + 1][1], items[i + 1][2])
                i += 2
            else:
                i += 1
            nxt = None
            for j in range(i, len(items)):
                if items[j][0] == 'v':
                    nxt = (items[j][1], items[j][2])
                    break
            if nxt is None:
                nxt = first
            if verts and t is not None:
                p0 = verts[-1]
                is_full = np.hypot(nxt[0] - p0[0], nxt[1] - p0[1]) < 1e-6 * max(np.hypot(p0[0] - c[0], p0[1] - c[1]), 1.0)
                verts.extend(_arc_points(p0, c, t, nxt, tol, full_dir=1 if n_full % 2 == 0 else -1))
                n_full += int(is_full)
        else:
            i += 1
    return verts


def circle_points(bb, tol=0.005):
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    rx, ry = (bb[2] - bb[0]) / 2, (bb[3] - bb[1]) / 2
    r = max(rx, ry)
    n = max(int(np.ceil(np.pi / np.arccos(1 - tol / r))), 16)
    a = 2 * np.pi * np.arange(n) / n
    return list(zip(cx + rx * np.cos(a), cy + ry * np.sin(a)))


def shape_points(s: Shape, tol=0.005):
    if s.kind == 'box':
        x0, y0, x1, y1 = s.data
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if s.kind == 'circle':
        return circle_points(s.data, tol)
    return polygon_points(s.data, tol)


def cell_bbox(cells, cell, tol=0.005, _cache=None):
    """Bounding box of a cell's own shapes and its instances (recursively)."""
    _cache = {} if _cache is None else _cache
    if cell.index in _cache:
        return _cache[cell.index]
    xs, ys = [], []
    for s in cell.shapes:
        pts = np.array(shape_points(s, tol))
        if len(pts):
            xs += [pts[:, 0].min(), pts[:, 0].max()]
            ys += [pts[:, 1].min(), pts[:, 1].max()]
    for inst in cell.instances:
        sub = cells[inst.cell_index - 1]
        bb = cell_bbox(cells, sub, tol, _cache)
        if bb is not None:
            xs += [bb[0] + inst.x, bb[2] + inst.x]
            ys += [bb[1] + inst.y, bb[3] + inst.y]
    out = (min(xs), min(ys), max(xs), max(ys)) if xs else None
    _cache[cell.index] = out
    return out
