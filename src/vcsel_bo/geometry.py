"""Mask geometry -> oxide aperture -> polarization-relevant descriptors.

Design space (matches the cruciform mask generator: rotation / arm ratio / arm width / corner radius)
----------------------------------------------------------------------------------------------------
    L      long-arm tip-to-tip length (um)
    ratio  short-arm length / long-arm length, in (0, 1]        (1 -> symmetric cross)
    w      arm width (um)
    rho    corner rounding fraction in [0, 1]; corner radius = rho * w / 2 on all arm-tip corners
           (rho = 1 -> fully rounded arm tips)
    theta  rotation of the long arm from the crystal x axis (deg); the shape has C2 symmetry for
           ratio < 1 (period 180 deg) and C4 symmetry for ratio = 1 (period 90 deg)

The *drawn* cruciform is the mesa. Lateral wet oxidation removes a depth d from every mesa edge, at
rates that differ along the two crystal axes ([110] vs [1-10]). The aperture is therefore the mesa
**eroded** by an ellipse with semi-axes (d_x, d_y). This is a Minkowski erosion: it rounds concave
(inner) corners with the oxidation depth, and it pinches off arms narrower than 2 d - effects seen on real
cruciform devices that the previous prototype ignored.

Numerics (no raster quantisation)
---------------------------------
The mesa boundary is generated analytically (line and arc samples every ~0.02 um). The erosion field
F(p) = (distance to the nearest boundary sample in *scaled* coordinates (x/d_x, y/d_y)) - 1 is evaluated on a
0.25 um grid; F is smooth, so its zero contour (marching squares) reproduces the aperture edge to a small
fraction of a pixel. Consequently every descriptor is a continuous function of (L, ratio, w, rho, theta) -
a binary raster would make arm widths jump in pixel-sized steps, which corrupts the surrogate model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

H_GRID = 0.25        # analysis-grid pitch (um); the scalar mode is solved on the stride-2 sub-grid
H_MODE = 0.5         # mode-solver pitch (um)
PAD = 4.0            # margin (um) around the mesa
SPACING = 0.02       # boundary sampling (um)


def set_resolution(h_grid=0.25, h_mode=0.5):
    """Switch numerical resolution (e.g. (0.125, 0.25) to verify final designs). h_mode must be a multiple of h_grid."""
    global H_GRID, H_MODE
    assert abs(h_mode / h_grid - round(h_mode / h_grid)) < 1e-9
    H_GRID, H_MODE = float(h_grid), float(h_mode)


@dataclass(frozen=True)
class MaskDesign:
    L: float
    ratio: float
    w: float
    rho: float
    theta: float

    def as_array(self):
        return np.array([self.L, self.ratio, self.w, self.rho, self.theta], float)

    @staticmethod
    def from_array(x):
        return MaskDesign(*(float(v) for v in x))

    @property
    def corner_radius(self):
        return self.rho * self.w / 2.0


def mesa_of_target(target: "MaskDesign", depth: float) -> "MaskDesign":
    """Mesa for a drawn TARGET APERTURE: the target grown by the design oxidation depth (offset with round joins).

    Dilation distributes over a union, so a cross of two rounded rectangles stays a cross of two rounded rectangles:
    L' = L + 2d, short' = ratio L + 2d, w' = w + 2d, tip radius' = rho w / 2 + d.  With isotropic oxidation of exactly d the
    aperture is then the morphological closing of the target (inner corners filled with fillets of radius d).
    """
    L2, w2 = target.L + 2 * depth, target.w + 2 * depth
    return MaskDesign(L2, (target.ratio * target.L + 2 * depth) / L2, w2, (target.rho * target.w + 2 * depth) / w2, target.theta)


@dataclass(frozen=True)
class Process:
    """Lateral-oxidation parameters of the fabrication run (SET THESE FROM YOUR PROCESS).

    depth       nominal oxidation depth d (um): how far the oxide front advances from each mesa edge.
    ox_ratio    rate_y / rate_x, the crystal-direction oxidation-rate ratio (1 = isotropic).
                Reported in-plane anisotropies of AlGaAs oxidation are of order 10-30 %.
    """
    depth: float = 1.0
    ox_ratio: float = 1.15


# ----------------------------------------------------------------------------------------------
# Exact mesa geometry
# ----------------------------------------------------------------------------------------------


def _sdf_rounded_box(px, py, half_x, half_y, r):
    qx = np.abs(px) - (half_x - r)
    qy = np.abs(py) - (half_y - r)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - r


def _rounded_box_boundary(hx, hy, r, spacing):
    """Points on the boundary of a rounded box centred at the origin (axes-aligned)."""
    pts = []
    def seg(p0, p1):
        n = max(int(np.ceil(np.hypot(*(np.subtract(p1, p0))) / spacing)), 1)
        t = np.linspace(0, 1, n + 1)[:, None]
        pts.append(np.asarray(p0)[None] * (1 - t) + np.asarray(p1)[None] * t)
    def arc(cx, cy, a0):
        n = max(int(np.ceil(r * (np.pi / 2) / spacing)), 1)
        a = a0 + np.linspace(0, np.pi / 2, n + 1)
        pts.append(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]))
    ix, iy = hx - r, hy - r
    seg((ix, -hy), (-ix, -hy)) if False else None
    # counter-clockwise edges
    seg((-ix, -hy), (ix, -hy)); 
    if r > 0: arc(ix, -iy, -np.pi / 2)
    seg((hx, -iy), (hx, iy))
    if r > 0: arc(ix, iy, 0.0)
    seg((ix, hy), (-ix, hy))
    if r > 0: arc(-ix, iy, np.pi / 2)
    seg((-hx, iy), (-hx, -iy))
    if r > 0: arc(-ix, -iy, np.pi)
    return np.vstack(pts)


def _rot(p, th):
    c, s = np.cos(th), np.sin(th)
    return np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]])


def _to_local(X, Y, th):
    c, s = np.cos(th), np.sin(th)
    return c * X + s * Y, -s * X + c * Y


def mesa_boundary(design: MaskDesign, spacing=SPACING):
    """Dense samples of the exact boundary of the drawn cruciform (union of two rounded rectangles)."""
    th = np.deg2rad(design.theta)
    r = float(np.clip(design.corner_radius, 0.0, design.w / 2.0 - 1e-9))
    a = (design.L / 2.0, design.w / 2.0)
    b = (design.w / 2.0, design.ratio * design.L / 2.0)
    pa = _rounded_box_boundary(a[0], a[1], r, spacing)
    pb = _rounded_box_boundary(b[0], b[1], r, spacing)
    # keep the part of each outline that is not strictly inside the other arm
    tol = 1e-7
    keep_a = _sdf_rounded_box(pa[:, 0], pa[:, 1], b[0], b[1], r) >= -tol
    keep_b = _sdf_rounded_box(pb[:, 0], pb[:, 1], a[0], a[1], r) >= -tol
    return _rot(np.vstack([pa[keep_a], pb[keep_b]]), th)


def inside_mesa(design: MaskDesign, X, Y):
    th = np.deg2rad(design.theta)
    u, v = _to_local(X, Y, th)
    r = float(np.clip(design.corner_radius, 0.0, design.w / 2.0 - 1e-9))
    s1 = _sdf_rounded_box(u, v, design.L / 2.0, design.w / 2.0, r)
    s2 = _sdf_rounded_box(u, v, design.w / 2.0, design.ratio * design.L / 2.0, r)
    return np.minimum(s1, s2) < 0.0


def grid_axes(design: MaskDesign, pad=PAD):
    """Symmetric analysis-grid axis with the origin on a node and an integer number of mode-grid cells."""
    m = int(np.ceil((design.L / 2.0 + pad) / H_MODE))
    n_side = m * int(round(H_MODE / H_GRID))
    return np.arange(-n_side, n_side + 1) * H_GRID


# ----------------------------------------------------------------------------------------------
# Anisotropic lateral oxidation as an exact erosion field
# ----------------------------------------------------------------------------------------------


def erosion_field(boundary_pts, inside, X, Y, process: Process):
    """Smooth field F with F > 0 inside the aperture and F = 0 on its boundary.

    Aperture = mesa eroded by the ellipse with semi-axes d_x = d / sqrt(ox_ratio), d_y = d sqrt(ox_ratio)
    (geometric-mean depth d). In scaled coordinates (x / d_x, y / d_y) the ellipse is a unit disk, so
    F = (distance to the nearest mesa-boundary sample, scaled) - 1 inside the mesa, and -(distance) - 1
    outside. If depth == 0 the aperture is the mesa itself and F is the (scaled-by-1um) distance.
    """
    d = process.depth
    if d <= 0:
        dx = dy = 1.0
        shift = 0.0
    else:
        dx = d / np.sqrt(process.ox_ratio)
        dy = d * np.sqrt(process.ox_ratio)
        shift = 1.0
    tree = cKDTree(np.column_stack([boundary_pts[:, 0] / dx, boundary_pts[:, 1] / dy]))
    q = np.column_stack([X.ravel() / dx, Y.ravel() / dy])
    dist, _ = tree.query(q)
    dist = dist.reshape(X.shape)
    if d <= 0:
        return np.where(inside, dist, -dist)
    return np.where(inside, dist - shift, -dist - shift)


@dataclass
class Aperture:
    design: MaskDesign
    process: Process
    axes: np.ndarray            # analysis-grid axis (um)
    F: np.ndarray               # erosion field, > 0 inside the aperture (rows = y, cols = x)
    loops: list                 # closed boundary polylines (K_i, 2) in um
    area: float                 # um^2 (polygon area of the kept loop)
    n_components: int
    mesa_area: float

    @property
    def equivalent_diameter(self):
        return 2.0 * np.sqrt(self.area / np.pi) if self.area > 0 else 0.0


def _polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def build_aperture(design: MaskDesign, process: Process) -> Aperture:
    from contourpy import contour_generator

    ax = grid_axes(design)
    X, Y = np.meshgrid(ax, ax)
    bnd = mesa_boundary(design)
    ins = inside_mesa(design, X, Y)
    F = erosion_field(bnd, ins, X, Y, process)
    loops = [np.asarray(l) for l in contour_generator(x=ax, y=ax, z=F).lines(0.0) if len(l) >= 4]
    areas = [_polygon_area(l) for l in loops]
    keep = [i for i, a in enumerate(areas) if a > 1.0]
    n_comp = len(keep)
    if keep:
        main = keep[int(np.argmax([areas[i] for i in keep]))]
        loops, area = [loops[main]], areas[main]
    else:
        loops, area = [], 0.0
    return Aperture(design, process, ax, F, loops, float(area), n_comp, float(ins.sum() * H_GRID ** 2))


# ----------------------------------------------------------------------------------------------
# Boundary segments and second-rank descriptors
# ----------------------------------------------------------------------------------------------


def boundary_segments(ap: Aperture):
    """Segments of the aperture outline: midpoints (K,2), normal angle phi_n (K), lengths (K)."""
    if not ap.loops:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    poly = ap.loops[0]
    p0, p1 = poly[:-1], poly[1:]
    t = p1 - p0
    length = np.hypot(t[:, 0], t[:, 1])
    good = length > 1e-9
    t, length, p0, p1 = t[good], length[good], p0[good], p1[good]
    # normal = tangent rotated by 90 deg; only n n^T (2*angle) matters, so orientation is irrelevant
    return 0.5 * (p0 + p1), np.arctan2(t[:, 0], -t[:, 1]), length


def second_rank_vector(phi_n, weights):
    """Weighted mean of (cos 2 phi_n, sin 2 phi_n), negated so that the vector points along the
    direction of *elongation* (an aperture elongated along x gives (+, 0))."""
    wsum = weights.sum()
    if wsum <= 0:
        return np.zeros(2)
    c = (weights * np.cos(2 * phi_n)).sum() / wsum
    s = (weights * np.sin(2 * phi_n)).sum() / wsum
    return np.array([-c, -s])


@dataclass
class Descriptors:
    s_mode: np.ndarray       # mode-weighted second-rank boundary vector (optical/gain overlap channel)
    s_stress: np.ndarray     # stress-kernel-weighted boundary vector (strain/birefringence channel)
    area: float
    equivalent_diameter: float
    n_components: int
    area_fraction: float     # aperture area / mesa area
    mode_energy: float       # fundamental-mode eigenvalue (um^-2), < 0 if guided
    mode_gap: float          # spacing to the second scalar mode (um^-2); small -> multimode risk
    valid: bool


def descriptors(ap: Aperture, dn_eff=0.01, wavelength_um=0.94, n_core=3.3, rho0=1.0) -> Descriptors:
    """Second-rank geometric descriptors of an aperture (see `modes.fundamental_mode`)."""
    from .modes import fundamental_mode

    frac = ap.area / max(ap.mesa_area, 1e-9)
    if (not ap.loops) or ap.area < 4.0:
        return Descriptors(np.zeros(2), np.zeros(2), ap.area, ap.equivalent_diameter, ap.n_components, frac, 0.0, 0.0, False)
    stride = int(round(H_MODE / H_GRID))
    ax_c = ap.axes[::stride]
    depth = ap.process.depth if ap.process.depth > 0 else 1.0
    # anti-aliased index profile at the mode nodes: coverage ~ clip(0.5 + s/h), s = signed distance to the edge
    chi = np.clip(0.5 + ap.F[::stride, ::stride] * depth / H_MODE, 0.0, 1.0)
    U0 = (2 * np.pi / wavelength_um) ** 2 * 2 * n_core * dn_eff
    psi, E0, gap = fundamental_mode(chi, H_MODE, U0)
    mids, phin, ln = boundary_segments(ap)
    fx = (mids[:, 0] - ax_c[0]) / H_MODE
    fy = (mids[:, 1] - ax_c[0]) / H_MODE
    # cubic interpolation: |psi|^2 decays over ~0.6 um near the edge, comparable to the mode-grid pitch
    inten = np.clip(ndi.map_coordinates(psi ** 2, [fy, fx], order=3, mode="nearest"), 0.0, None)
    s_mode = second_rank_vector(phin, inten * ln)
    # stress channel: long-range 1/(r^2 + rho0^2) kernel from the intensity-weighted centroid
    Xc, Yc = np.meshgrid(ax_c, ax_c)
    w2 = psi ** 2
    cx, cy = (w2 * Xc).sum() / w2.sum(), (w2 * Yc).sum() / w2.sum()
    r2 = (mids[:, 0] - cx) ** 2 + (mids[:, 1] - cy) ** 2
    s_str = second_rank_vector(phin, ln / (r2 + rho0 ** 2))
    return Descriptors(s_mode, s_str, ap.area, ap.equivalent_diameter, ap.n_components, frac, E0, gap, True)


def design_descriptors(design: MaskDesign, process: Process, dn_eff=0.01):
    return descriptors(build_aperture(design, process), dn_eff=dn_eff)
