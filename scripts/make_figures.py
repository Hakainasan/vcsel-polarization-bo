"""Figures + statistics from the cached results (results/*).  Every figure is a MODEL PREDICTION conditional on the
literature anchors, not a measurement.  Usage: python scripts/make_figures.py"""
import json
import pickle
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
from scipy.stats import friedmanchisquare, wilcoxon  # noqa: E402

from vcsel_bo import geometry as G  # noqa: E402
from vcsel_bo import literature as L  # noqa: E402
from vcsel_bo import objective as O  # noqa: E402
from vcsel_bo import physics as P  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
METHODS = ["random", "sobol", "de", "bo_ei", "bo_ucb"]
COL = dict(random="#888888", sobol="#4c78a8", de="#f58518", bo_ei="#e45756", bo_ucb="#72b7b2")
TITLES = dict(p1="P1 known wafer, default window", p2="P2 known wafer, narrow window (1.5-1.7)",
              p3="P3 unknown wafer: one design for all worlds")
SYN = "model prediction, conditional on literature anchors"


def load(p):
    """Final results if present, else the checkpoint of a run that is still going (partial results are balanced over methods)."""
    for name in (f"benchmark_{p}.pkl", f"benchmark_{p}.partial.pkl"):
        f = RES / name
        if f.exists():
            return pickle.load(open(f, "rb"))
    return None


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = np.array([r.choice(x, len(x)).mean() for _ in range(n)])
    return np.percentile(m, [2.5, 97.5])


# ----------------------------------------------------------------------------------------------
def regret_and_heldout():
    stats = []
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
    for k, p in enumerate(["p1", "p2", "p3"]):
        R = load(p)
        if R is None:
            continue
        worlds = sorted({r["world"] for r in R})
        ref = {w: max(r["y"].max() for r in R if r["world"] == w) for w in worlds}
        budget = max(len(r["y"]) for r in R)
        ax = axs[0, k]
        for m in METHODS:
            curves = []
            for r in R:
                if r["method"] != m:
                    continue
                c = np.maximum.accumulate(r["y"])
                c = np.concatenate([c, np.full(budget - len(c), c[-1])])[:budget]
                curves.append(ref[r["world"]] - c)
            C = np.array(curves)
            mean = C.mean(0)
            lo, hi = np.array([boot_ci(C[:, i], 300) for i in range(C.shape[1])]).T
            ax.plot(np.arange(1, budget + 1), np.maximum(mean, 1e-4), color=COL[m], label=m)
            ax.fill_between(np.arange(1, budget + 1), np.maximum(lo, 1e-4), np.maximum(hi, 1e-4), color=COL[m], alpha=0.15)
        ax.set_yscale("log"); ax.set_title(TITLES[p], fontsize=9); ax.set_xlabel("evaluations")
        if k == 0:
            ax.set_ylabel("simple regret (smoothed yield)"); ax.legend(fontsize=7)
        ax = axs[1, k]
        data = []
        for m in METHODS:
            v = np.array([r["held_out"] for r in R if r["method"] == m])
            data.append(v)
            lo, hi = boot_ci(v)
            stats.append(dict(problem=p, method=m, mean_held_out=float(v.mean()), ci_lo=float(lo), ci_hi=float(hi), n=len(v)))
        bp = ax.boxplot(data, labels=METHODS, patch_artist=True)
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COL[m]); patch.set_alpha(0.6)
        ax.set_ylabel("held-out hard yield"); ax.tick_params(axis="x", labelsize=8)
        groups = {}                                       # Friedman test over the five methods (paired by world and seed), per-world means
        for r in R:
            groups.setdefault((r["world"], r["seed"]), {})[r["method"]] = r["held_out"]
        full = {k_: g for k_, g in groups.items() if len(g) == len(METHODS)}
        if len(full) >= 3:
            Am = np.array([[g[m] for m in METHODS] for g in full.values()])
            try:
                fp = float(friedmanchisquare(*[Am[:, j] for j in range(len(METHODS))]).pvalue)
            except ValueError:
                fp = float("nan")
            perw = {}
            for (w_, sd_), g in full.items():
                perw.setdefault(w_, []).append([g[m] for m in METHODS])
            stats.append(dict(problem=p, friedman_p=fp, n_groups=len(full), methods=METHODS,
                              per_world_means={str(w_): np.mean(v_, 0).tolist() for w_, v_ in perw.items()}))
        for a in ("bo_ei", "bo_ucb"):
            for b in ("random", "sobol", "de"):
                pa = {(r["world"], r["seed"]): r["held_out"] for r in R if r["method"] == a}
                pb = {(r["world"], r["seed"]): r["held_out"] for r in R if r["method"] == b}
                keys = sorted(set(pa) & set(pb))                 # pairs present for both methods (matters for partial results)
                d = np.array([pa[q] - pb[q] for q in keys])
                try:
                    pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                except ValueError:
                    pv = 1.0
                stats.append(dict(problem=p, test=f"{a} - {b}", mean_diff=float(d.mean()), wilcoxon_p=pv, n_pairs=len(d)))
    fig.suptitle(f"Optimiser benchmark ({SYN}); equal budget", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "benchmark.png", dpi=160); plt.close(fig)
    (RES / "benchmark_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# ----------------------------------------------------------------------------------------------
def anchors_figure():
    f = RES / "anchored_worlds.json"
    if not f.exists():
        return
    R = json.loads(f.read_text())
    acc = R["accepted"]
    d0 = np.array([a["d0"] for a in acc]); b0 = np.array([a["b0"] for a in acc])
    kd = np.array([a["k_d"] for a in acc]); kb = np.array([a["k_b"] for a in acc])
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.3))
    ax = axs[0]
    ax.scatter(d0, -kd, s=10, c="#e45756")
    pr = P.Prior()
    ax.plot([pr.d0[0], pr.d0[1], pr.d0[1], pr.d0[0], pr.d0[0]],
            [pr.k_d_mag[0], pr.k_d_mag[0], pr.k_d_mag[1], pr.k_d_mag[1], pr.k_d_mag[0]], "k:", lw=0.8)
    fw = RES / "anchored_worlds_wide.json"
    ttl = "compensation ridge (dotted: prior box)"
    if fw.exists():        # the same anchors under a wider prior box: the ridge continues below the default edge d0 = 1
        aw = json.loads(fw.read_text())["accepted"]
        if aw:
            ax.scatter([a["d0"] for a in aw], [-a["k_d"] for a in aw], s=14, facecolors="none", edgecolors="#4c78a8", lw=0.8)
            pw = P.Prior(d0=(0.2, 12.0), k_d_mag=(0.6, 100.0))
            ax.plot([pw.d0[0], pw.d0[1], pw.d0[1], pw.d0[0], pw.d0[0]],
                    [pw.k_d_mag[0], pw.k_d_mag[0], pw.k_d_mag[1], pw.k_d_mag[1], pw.k_d_mag[0]], "--", color="#4c78a8", lw=0.8)
            ttl = "compensation ridge (red: primary, dotted box;\nblue circles: wider prior, dashed box)"
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("native dichroism $d_0$ (ns$^{-1}$)"); ax.set_ylabel("$|k_d|$ (ns$^{-1}$)")
    ax.set_title(ttl, fontsize=8)
    for ax, v, lab in ((axs[1], b0, "native birefringence $b_0$ (ns$^{-1}$)"), (axs[2], kb, "birefringence coupling $k_b$ (ns$^{-1}$)")):
        pos, neg = v[v > 0], -v[v < 0]
        bins = np.logspace(np.log10(4), np.log10(220), 16)
        ax.hist(pos, bins=bins, color="#4c78a8", alpha=0.8, label=f"> 0 ({len(pos)})")
        ax.hist(neg, bins=bins, color="#f58518", alpha=0.8, label=f"< 0 ({len(neg)}), plotted as |.|")
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.legend(fontsize=7)
    ax = axs[3]
    m = R.get("marginal", {}).get("per_anchor", {})
    if m:
        keys = list(m)
        ax.barh(range(len(keys)), [m[k] for k in keys], color="#72b7b2")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels([k.split("_")[0] for k in keys], fontsize=8)
        ax.axvline(R["acceptance"], color="k", ls="--", lw=0.8)
        ax.set_xlabel("share of prior draws passing alone", fontsize=8); ax.set_title("dashed: all enforced together", fontsize=8)
    fig.suptitle(f"Literature-consistent worlds: {R['n_accepted']} of {R['n_draws']} prior draws pass every enforced anchor", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "anchored_worlds.png", dpi=160); plt.close(fig)


def rotation_and_designs_figures():
    f = RES / "literature_designs.npz"
    if not f.exists():
        return
    Z = np.load(f, allow_pickle=True)
    names = list(Z["names"]); Y = Z["Y"]; scan = Z["scan"]; ratios = Z["ratios"]; th = Z["thetas"]
    # rotation scan
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cm = plt.get_cmap("viridis")
    for i, r in enumerate(ratios):
        m = scan[:, i].mean(0)
        ax.plot(th, m, "-o", ms=3, color=cm(i / (len(ratios) - 1)), label=f"arm ratio {r:.1f}")
        if abs(r - 0.7) < 1e-9:
            ax.fill_between(th, np.quantile(scan[:, i], 0.1, axis=0), np.quantile(scan[:, i], 0.9, axis=0), color=cm(i / 4), alpha=0.18)
    for a in (30, 60, 90):
        ax.axvline(a, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("rotation of the long arm from the native axis (deg)"); ax.set_ylabel("mean hard yield over literature-consistent worlds")
    ax.set_title("L = 17.5 um, w = 6 um; band = 10-90% of worlds (ratio 0.7)", fontsize=8); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "rotation_scan.png", dpi=160); plt.close(fig)
    # designs
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1.4]))
    ax = axs[0]
    order = list(L.APL126_SEVEN) + ["circle", "sym_cross"]
    lab = dict(square="square", x07_0="1:0.7 0", x07_30="1:0.7 30", x07_60="1:0.7 60", x07_90="1:0.7 90", x06_0="1:0.6 0", x06_90="1:0.6 90",
               circle="circle", sym_cross="sym. cross")
    data = [Y[names.index(k)] for k in order]
    bp = ax.boxplot(data, labels=[lab[k] for k in order], patch_artist=True, whis=(10, 90), showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4c78a8"); patch.set_alpha(0.6)
    ax.plot(range(1, len(order) + 1), [d.mean() for d in data], "k_", ms=12)
    ax.set_ylabel("hard yield"); ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_title("designs of the published experiments (bar: mean)", fontsize=8)
    ax = axs[1]
    angs = np.array(L.MASK_ANGLES)
    for fam, col in (("A", "#e45756"), ("B", "#72b7b2")):
        M = np.array([Y[names.index(f"mask_{fam}{int(a)}")] for a in angs])
        ax.plot(angs, M.mean(1), "-o", color=col, label=f"family {fam}")
        ax.fill_between(angs, np.quantile(M, 0.1, axis=1), np.quantile(M, 0.9, axis=1), color=col, alpha=0.15)
    ax.set_xlabel("rotation (deg)"); ax.set_ylabel("hard yield"); ax.legend(fontsize=7)
    ax.set_title("the 16 designs of the author's mask (band: 10-90% of worlds)", fontsize=8)
    fig.suptitle(f"Predicted yield of published and own designs ({SYN})", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "designs.png", dpi=160); plt.close(fig)


def landscape():
    """Yield over (arm ratio, rotation) for the default reference World; other parameters fixed."""
    w = P.World()
    pop = P.Population.make(16, 0)
    rs = np.linspace(0.5, 1.0, 21)
    ths = np.linspace(0, 180, 37)
    Y = np.array([[O.evaluate_design([17.5, r, 6.0, 0.5, t], w, pop).hard_yield for t in ths] for r in rs])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = ax.pcolormesh(ths, rs, Y, shading="auto", vmin=0, vmax=1)
    ax.set_xlabel("rotation $\\theta$ (deg)"); ax.set_ylabel("arm ratio")
    ax.set_title("Yield slice in the reference world (L=17.5, w=6, rho=0.5)", fontsize=8)
    fig.colorbar(im, label="hard yield"); fig.tight_layout(); fig.savefig(FIG / "landscape.png", dpi=160); plt.close(fig)


def batch_figure():
    f = RES / "batch_design.csv"
    if not f.exists():
        return
    rows = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)
    fig, axs = plt.subplots(2, 8, figsize=(13, 3.6))
    w = P.World()
    for k, row in enumerate(rows[:16]):
        ax = axs[k // 8, k % 8]
        design = G.MaskDesign(row["L_um"], row["arm_ratio"], row["arm_width_um"], row["corner_rho"], row["rotation_deg"])
        ap = G.build_aperture(design, w.process)
        for loop in ap.loops:
            ax.plot(loop[:, 0], loop[:, 1], "k", lw=1)
        m = G.mesa_boundary(design)
        ax.plot(m[:, 0], m[:, 1], ".", color="0.7", ms=0.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"#{int(row['id'])} r={row['arm_ratio']:.2f} th={row['rotation_deg']:.0f}\nE={row['E_yield_test']:.2f}", fontsize=6)
    fig.suptitle(f"Batch designs: mesa (grey) and eroded oxide aperture (black) in the placeholder process ({SYN})", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "batch_shapes.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    for fn in (anchors_figure, rotation_and_designs_figures, landscape, batch_figure):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:  # keep going; report
            print(fn.__name__, "skipped:", repr(e))
    print(json.dumps(regret_and_heldout(), indent=1)[:2500])
