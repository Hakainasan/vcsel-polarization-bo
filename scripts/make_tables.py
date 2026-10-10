"""Write supplementary/tables.tex (LaTeX macros and tables) and results/key_numbers.json from the cached results, so that
no number in the supplement is typed by hand.  Usage: python scripts/make_tables.py"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from vcsel_bo import literature as L  # noqa: E402

RES, SUP = ROOT / "results", ROOT / "supplementary"
METHODS = ["random", "sobol", "de", "bo_ei", "bo_ucb"]
METHOD_NAMES = dict(random="random search", sobol="Sobol", de="differential evolution", bo_ei="GP-EI", bo_ucb="GP-UCB")
LAB = dict(circle="circle", square="square", sym_cross="symmetric cross", x07_0="1:0.7, 0$^\\circ$", x07_30="1:0.7, 30$^\\circ$",
           x07_60="1:0.7, 60$^\\circ$", x07_90="1:0.7, 90$^\\circ$", x06_0="1:0.6, 0$^\\circ$", x06_90="1:0.6, 90$^\\circ$")
out, key = [], {}


def jl(name):
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else None


def fmt(v, n=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


_DIG = "zero one two three four five six seven eight nine".split()


def macro(name, val):
    name = "".join(_DIG[int(ch)].capitalize() if ch.isdigit() else ch for ch in name.replace("_", ""))   # TeX names: letters only
    out.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    key[name] = val


# ---------------------------------------------------------------- anchored worlds
A = jl("anchored_worlds.json")
if A:
    macro("nDraws", f"{A['n_draws']:,}".replace(",", "\\,"))
    macro("nAccepted", A["n_accepted"])
    macro("accPct", fmt(100 * A["acceptance"], 1))
    post = A["posterior"]
    for c, nm in (("d0", "dzero"), ("b0", "bzero"), ("k_d", "kd"), ("k_b", "kb")):
        macro(f"post{nm}", f"{post[c]['median']:.1f}\\;[{post[c]['q05']:.1f},\\,{post[c]['q95']:.1f}]")
    macro("fracBzeroPos", fmt(100 * post["b0"]["frac_positive"], 0))
    macro("fracKbNeg", fmt(100 * (1 - post["k_b"]["frac_positive"]), 0))
    macro("nMarginalDraws", A["marginal"]["n"])
    macro("marginalAllEnforced", fmt(100 * A["marginal"]["all_enforced"], 1))
    for k, v in A["marginal"]["per_anchor"].items():
        macro("marg" + k.split("_")[0], fmt(100 * v, 0))
    ff = A["first_failure"]
    macro("failAone", ff.get("A1_circle_square_native_x", 0))
    macro("failAthree", ff.get("A3_rotation_by_90_flips_initial_polarization", 0))
    macro("secAnchored", f"{A['seconds']:.0f}")

for tag in ("loose", "strict", "chen"):
    B = jl(f"anchored_worlds_{tag}.json")
    if B:
        macro(f"nAcc{tag.capitalize()}", B["n_accepted"])
        macro(f"nDraws{tag.capitalize()}", B["n_draws"])
        macro(f"accPct{tag.capitalize()}", fmt(100 * B["acceptance"], 1))

# ---------------------------------------------------------------- designs
D = jl("literature_designs.json")
if D:
    macro("nDesignWorlds", D["n_worlds"])
    rows = []
    for k in list(L.DESIGNS) + [f"mask_{m}" for m in L.MASK_DESIGNS]:
        s = D["designs"][k]
        name = LAB.get(k) or f"mask {k.split('_')[1][0]}, {k.split('_')[1][1:]}$^\\circ$"
        mu = "--" if s["median_mu_sw"] is None else fmt(s["median_mu_sw"], 2)
        rows.append(f"{name} & {fmt(s['mean_yield'])} & [{fmt(s['q10'])}, {fmt(s['q90'])}] & {fmt(s['p_yield_ge_half'])} & "
                    f"{fmt(s['mean_p_switch'])} & {mu} \\\\")
        key[f"design_{k}"] = s
    out.append("\\newcommand{\\tabDesigns}{\\begin{tabular}{lccccc}\\toprule design & mean yield & [10\\%,90\\% of worlds] & "
               "$P(\\text{yield}\\ge0.5)$ & mean $P(\\text{switch})$ & median $\\mu_{sw}$\\\\\\midrule\n"
               + "\n".join(rows[:9]) + "\n\\midrule\n" + "\n".join(rows[9:]) + "\n\\bottomrule\\end{tabular}}")
    top = D["apl126_seven_top_frequency"]
    macro("noWorkingSeven", fmt(100 * D["apl126_seven_no_working_design_fraction"], 0))
    out.append("\\newcommand{\\tabTopSeven}{\\begin{tabular}{lc}\\toprule design of APL 126 & share of worlds in which it is the best of the seven\\\\\\midrule\n"
               + "\n".join(f"{LAB[k]} & {fmt(100 * v, 0)}\\% \\\\" for k, v in top.items()) + "\n\\bottomrule\\end{tabular}}")
    rs = D["rotation_scan"]
    macro("bestThetaSeven", fmt(rs["best_theta_by_ratio"]["0.7"], 0))
    ms = np.array(rs["mean_yield"]); th = np.array(rs["thetas"]); i7 = rs["ratios"].index(0.7)
    plateau = th[ms[i7] >= 0.9 * ms[i7].max()]
    macro("plateauLo", fmt(plateau.min(), 0)); macro("plateauHi", fmt(plateau.max(), 0))
    macro("meanYieldSevenMax", fmt(ms[i7].max()))
    for r, w in zip(rs["ratios"], ("Five", "Six", "Seven", "Eight", "Nine")):
        macro("reachRatio" + w, fmt(100 * rs["p_world_reaches_half"][str(r)], 0))
    for k, w in (("x07_0", "Zero"), ("x07_30", "Thirty"), ("x07_60", "Sixty"), ("x07_90", "Ninety"), ("sym_cross", "Sym"),
                 ("circle", "Circle"), ("square", "Square")):
        macro("psw" + w, fmt(D["designs"][k]["mean_p_switch"]))
        macro("yld" + w, fmt(D["designs"][k]["mean_yield"]))
        macro("ge" + w, fmt(D["designs"][k]["p_yield_ge_half"]))

# ---------------------------------------------------------------- dependence on the switching-current window
Wd = jl("window_dependence.json")
if Wd:
    th = Wd["thetas"]
    wl = []
    for name, v in Wd["windows"].items():
        wl.append(f"{name} & " + " & ".join(fmt(x) for x in v["mean_yield"]) + f" & {v['best_theta']:.0f}$^\\circ$ & {fmt(v['share_worlds_60_beats_90'], 2)} & {fmt(v['share_worlds_90_beats_60'], 2)} \\\\")
    out.append("\\newcommand{\\tabWindow}{\\begin{tabular}{l" + "c" * len(th) + "ccc}\\toprule window ($\\mu_{sw}$) & " + " & ".join(f"{t:.0f}$^\\circ$" for t in th)
               + " & best & $P(60^\\circ>90^\\circ)$ & $P(90^\\circ>60^\\circ)$\\\\\\midrule\n" + "\n".join(wl) + "\n\\bottomrule\\end{tabular}}")
    macro("nWindowWorlds", Wd["n_worlds"])
    lb = Wd["windows"]["low bias 1.15-1.60"]; df = Wd["windows"]["default 1.25-1.95"]
    macro("lowBiasSixtyBeatsNinety", fmt(100 * lb["share_worlds_60_beats_90"], 0)); macro("lowBiasNinetyBeatsSixty", fmt(100 * lb["share_worlds_90_beats_60"], 0))
    macro("defSixtyBeatsNinety", fmt(100 * df["share_worlds_60_beats_90"], 0)); macro("defNinetyBeatsSixty", fmt(100 * df["share_worlds_90_beats_60"], 0))
    key["window_dependence"] = Wd["windows"]

# ---------------------------------------------------------------- best rotation under alternative ways of conditioning on the literature
Rw = jl("rotation_by_worldset.json")
if Rw and Rw["sets"]:
    th = Rw["thetas"]
    LABS = dict(primary="primary (every third world)", loose="loose thresholds", strict="strict thresholds",
                chen="metasurface-paper SFM constants", wide="wider prior",
                ox10="re-inferred, isotropic oxidation ($\\eta=1.0$)", ox13="re-inferred, $\\eta=1.3$",
                chim02="re-inferred, thermal drift $\\chi_{th}=-0.2$", chip02="re-inferred, thermal drift $\\chi_{th}=+0.2$")
    rl = []
    for k, v in Rw["sets"].items():
        rl.append(f"{LABS.get(k, k)} & {v['n_worlds']} & " + " & ".join(fmt(x) for x in v["mean_yield"])
                  + f" & {v['best_theta']:.0f}$^\\circ$ & {v['plateau'][0]:.0f}--{v['plateau'][1]:.0f}$^\\circ$ \\\\")
    # worst case over the sets in which some rotation works at all (mean yield of the best rotation >= 0.2): share of each set's best
    work = {k: v for k, v in Rw["sets"].items() if v["max_mean_yield"] >= 0.2}
    share = np.array([[y / v["max_mean_yield"] for y in v["mean_yield"]] for v in work.values()])
    worst = share.min(0)
    rl.append("worst case over the " + str(len(work)) + " working sets (share of best) & & " + " & ".join(fmt(x, 2) for x in worst) + " & "
              + f"{th[int(np.argmax(worst))]:.0f}$^\\circ$ & \\\\")
    macro("rotRobustBest", f"{th[int(np.argmax(worst))]:.0f}"); macro("rotRobustBestShare", fmt(float(worst.max()), 2))
    macro("nRotWorking", len(work))
    for t_, nm in ((60.0, "Sixty"), (120.0, "OneTwenty")):
        if t_ in th:
            col = share[:, th.index(t_)]
            macro(f"share{nm}Lo", fmt(100 * float(col.min()), 0)); macro(f"share{nm}Hi", fmt(100 * float(col.max()), 0))
    for t_ in (70, 80, 90):
        if float(t_) in th:
            macro(f"rotWorst{t_}", fmt(float(worst[th.index(float(t_))]), 2))
    key["rotation_worst_case_share"] = dict(zip([f"{t:.0f}" for t in th], [float(x) for x in worst]))
    out.append("\\newcommand{\\tabRotSets}{\\begin{tabular}{lc" + "c" * len(th) + "cc}\\toprule world set & worlds & "
               + " & ".join(f"{t:.0f}$^\\circ$" for t in th) + " & best & within 90\\,\\% of best\\\\\\midrule\n"
               + "\n".join(rl) + "\n\\bottomrule\\end{tabular}}")
    bests = sorted({int(v["best_theta"]) for v in Rw["sets"].values()})
    macro("rotSetsBestLo", str(bests[0])); macro("rotSetsBestHi", str(bests[-1])); macro("nRotSets", len(Rw["sets"]))
    key["rotation_by_worldset"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_world"} for k, v in Rw["sets"].items()}
    Rb = Rw.get("worst_case_share_bootstrap")
    if Rb:
        ft = Rb["folded_rotations"]
        key["rotation_worst_case_bootstrap"] = {k: v for k, v in Rb.items()}
        for t_, nm in ((70.0, "Seventy"), (80.0, "Eighty"), (90.0, "Ninety")):
            i_ = ft.index(t_)
            macro(f"pHighest{nm}", fmt(100 * Rb["p_highest"][str(int(t_))], 0))
            macro(f"worstMean{nm}", fmt(Rb["mean"][i_], 2)); macro(f"worstLo{nm}", fmt(Rb["lo"][i_], 2)); macro(f"worstHi{nm}", fmt(Rb["hi"][i_], 2))
        macro("pEightyAboveNinety", fmt(100 * Rb["p_80_above_90"], 0)); macro("pEightyAboveSeventy", fmt(100 * Rb["p_80_above_70"], 0))
        macro("nBootRot", Rb["n_resamples"])
    if "wide" in Rw["sets"]:
        macro("wideBest", f"{Rw['sets']['wide']['best_theta']:.0f}")
        macro("wideMax", fmt(Rw["sets"]["wide"]["max_mean_yield"]))

# ---------------------------------------------------------------- the prior box (accepted worlds against its edges)
for tag, mname in (("", "Prim"), ("_wide", "Wide")):
    f = jl(f"anchored_worlds{tag}.json")
    if f and f.get("accepted"):
        d0 = np.array([r["d0"] for r in f["accepted"]])
        macro("nAcc" + mname, len(d0))
        macro("dzeroMed" + mname, fmt(float(np.median(d0)), 2))
        macro("dzeroQfive" + mname, fmt(float(np.percentile(d0, 5)), 2)); macro("dzeroQninetyfive" + mname, fmt(float(np.percentile(d0, 95)), 2))
        lo = 1.0 if tag == "" else 0.2
        macro("dzeroNearEdge" + mname, fmt(100 * float(np.mean(d0 < 2 * lo)), 0))
        if tag == "_wide":
            macro("accPctWide", fmt(100 * f["acceptance"], 1))
            macro("dzeroBelowOneWide", fmt(100 * float(np.mean(d0 < 1.0)), 0))
            macro("dzeroBelowHalfWide", fmt(100 * float(np.mean(d0 < 0.5)), 0))
            macro("nDrawsWide", f["n_draws"])

# re-conditioned runs: how many worlds pass the same anchors when an assumption is changed, and the sign of the native birefringence
for tag, name in (("ox10", "OxTen"), ("ox13", "OxThirteen"), ("chim02", "ChiMinus"), ("chip02", "ChiPlus")):
    f = jl(f"anchored_worlds_{tag}.json")
    if f:
        macro(f"nAcc{name}", f["n_accepted"]); macro(f"nDraws{name}", f["n_draws"])
        if f["accepted"]:
            macro(f"fracBzeroPos{name}", fmt(100 * float(np.mean([r["b0"] > 0 for r in f["accepted"]])), 0))

# ---------------------------------------------------------------- benchmark
stats = jl("benchmark_stats.json")
if stats:
    cell = {}
    for s in stats:
        if "method" in s:
            cell[(s["problem"], s["method"])] = s
    rows = []
    for m in METHODS:
        c = []
        for p in ("p1", "p2", "p3"):
            s = cell.get((p, m))
            c.append("--" if s is None else f"{fmt(s['mean_held_out'])} [{fmt(s['ci_lo'])}, {fmt(s['ci_hi'])}]")
        rows.append(f"{METHOD_NAMES[m]} & " + " & ".join(c) + " \\\\")
    n1 = cell.get(("p1", "random"), {}).get("n"); n3 = cell.get(("p3", "random"), {}).get("n")
    macro("nRunsPone", n1 if n1 else "--"); macro("nRunsPthree", n3 if n3 else "--")
    out.append("\\newcommand{\\tabBench}{\\begin{tabular}{lccc}\\toprule & P1 (default window) & P2 (narrow window) & P3 (one design, unseen worlds)\\\\\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}")
    tests = [s for s in stats if "test" in s]
    tr = []
    for s in tests:
        if s["test"] in ("bo_ei - random", "bo_ei - sobol", "bo_ei - de", "bo_ucb - random", "bo_ucb - sobol", "bo_ucb - de"):
            tr.append(f"{s['problem'].upper()} & {s['test'].replace('bo_ei', 'GP-EI').replace('bo_ucb', 'GP-UCB').replace('de', 'DE').replace('sobol', 'Sobol').replace('random', 'random')}"
                      f" & {s['mean_diff']:+.3f} & {s['wilcoxon_p']:.3f} & {s['n_pairs']} \\\\")
    out.append("\\newcommand{\\tabTests}{\\begin{tabular}{llccc}\\toprule problem & paired difference & mean diff. & Wilcoxon $p$ & pairs\\\\\\midrule\n"
               + "\n".join(tr) + "\n\\bottomrule\\end{tabular}}")
    for (p, m), s in cell.items():
        key[f"bench_{p}_{m}"] = s
    PN = dict(p1="One", p2="Two", p3="Three")
    MN = dict(random="Random", sobol="Sobol", de="De", bo_ei="BoEi", bo_ucb="BoUcb")
    for (p, m), s in cell.items():
        macro(f"bench{PN[p]}{MN[m]}", fmt(s["mean_held_out"]))
    tdict = {(s["problem"], s["test"]): s for s in tests}
    for p in PN:
        for a in ("bo_ei", "bo_ucb"):
            for b in ("random", "sobol", "de"):
                s = tdict.get((p, f"{a} - {b}"))
                if s:
                    macro(f"diff{PN[p]}{MN[a]}{MN[b]}", f"{s['mean_diff']:+.2f}")
                    macro(f"pval{PN[p]}{MN[a]}{MN[b]}", f"{s['wilcoxon_p']:.3f}")
    for s_ in stats:
        if "friedman_p" in s_:
            macro(f"fried{PN[s_['problem']]}", f"{s_['friedman_p']:.2f}")
            wm = s_["per_world_means"]
            i_ei, i_de, i_rn = METHODS.index("bo_ei"), METHODS.index("de"), METHODS.index("random")
            best_bo = {w_: max(v_[METHODS.index("bo_ei")], v_[METHODS.index("bo_ucb")]) for w_, v_ in wm.items()}
            macro(f"nWorlds{PN[s_['problem']]}", len(wm))
            macro(f"worldsBoAheadDe{PN[s_['problem']]}", sum(best_bo[w_] > wm[w_][i_de] + 0.02 for w_ in wm))
            macro(f"worldsDeAheadBo{PN[s_['problem']]}", sum(wm[w_][i_de] > best_bo[w_] + 0.02 for w_ in wm))
            macro(f"worldsTieBoDe{PN[s_['problem']]}", sum(abs(wm[w_][i_de] - best_bo[w_]) <= 0.02 for w_ in wm))
    for p in PN:
        v = [cell[(p, m)]["mean_held_out"] for m in METHODS if (p, m) in cell]
        if v:
            macro(f"bench{PN[p]}Min", fmt(min(v))); macro(f"bench{PN[p]}Max", fmt(max(v)))
        n_ = cell.get((p, "random"), {}).get("n")
        macro(f"nRuns{PN[p]}", n_ if n_ else "--")
    # runs whose returned design is poor (held-out yield below 0.5) on P1: a mean can hide an unreliable optimiser
    import pickle  # noqa: E402
    pk = RES / "benchmark_p1.pkl"
    if pk.exists():
        runs1 = pickle.load(open(pk, "rb"))
        for m in METHODS:
            h_ = [r["held_out"] for r in runs1 if r["method"] == m]
            macro(f"lowRunsOne{MN[m]}", sum(x < 0.5 for x in h_))
        macro("nRunsOnePerMethod", len([r for r in runs1 if r["method"] == "random"]))
        bad = [max(r["y"]) for r in runs1 if r["method"] == "bo_ucb" and r["held_out"] < 0.5]     # best smoothed yield the optimiser itself saw
        if bad:
            macro("ucbLowBestMin", fmt(min(bad))); macro("ucbLowBestMax", fmt(max(bad)))
        else:
            macro("ucbLowBestMin", "--"); macro("ucbLowBestMax", "--")

# wafer-level statistics (runs on one wafer are not independent) and the selection-vs-unseen gap of P3
Cl = jl("benchmark_clusters.json")
if Cl:
    PN2 = dict(p1="One", p2="Two")
    MN2 = dict(random="Random", sobol="Sobol", de="De", bo_ei="BoEi", bo_ucb="BoUcb")
    rows_c = []
    for p_, nmp in PN2.items():
        for cmp_, v_ in Cl[p_]["comparisons"].items():
            a_, b_ = [s.strip() for s in cmp_.split(" - ")]
            macro(f"wafer{nmp}{MN2[a_]}{MN2[b_]}", f"{v_['mean']:+.2f}\\;[{v_['lo']:+.2f},\\,{v_['hi']:+.2f}]")
            macro(f"waferAhead{nmp}{MN2[a_]}{MN2[b_]}", v_["n_ahead"])
            rows_c.append(f"{p_.upper()} & {METHOD_NAMES[a_]} $-$ {METHOD_NAMES[b_]} & {v_['mean']:+.3f} & [{v_['lo']:+.3f}, {v_['hi']:+.3f}] & {v_['n_ahead']} of {v_['n_wafers']} \\\\")
    out.append("\\newcommand{\\tabWafer}{\\begin{tabular}{llccc}\\toprule problem & comparison & mean over wafers & 95\\,\\% $t$-interval (4 d.f.) & wafers ahead\\\\\\midrule\n"
               + "\n".join(rows_c) + "\n\\bottomrule\\end{tabular}}")
    key["benchmark_clusters"] = {p_: Cl[p_]["comparisons"] for p_ in PN2}
    g3 = Cl["p3"]
    macro("pThreeSelLo", fmt(min(v["selection"] for v in g3.values()))); macro("pThreeSelHi", fmt(max(v["selection"] for v in g3.values())))
    macro("pThreeUnseenLo", fmt(min(v["unseen"] for v in g3.values()))); macro("pThreeUnseenHi", fmt(max(v["unseen"] for v in g3.values())))
    macro("pThreeGapLo", fmt(min(v["gap"] for v in g3.values()))); macro("pThreeGapHi", fmt(max(v["gap"] for v in g3.values())))
    key["p3_selection_vs_unseen"] = g3

W = jl("worlds.json")
if W:
    macro("fracGoodRange", " -- ".join(fmt(x, 2) for x in (min(k["frac_good"] for k in W["selected"]), max(k["frac_good"] for k in W["selected"]))))

# ---------------------------------------------------------------- batch
Bs = jl("batch_summary.json")
if Bs:
    t = Bs["test"]
    macro("nTrainWorlds", Bs["n_train_worlds"]); macro("nTestWorlds", Bs["n_test_worlds"]); macro("nPool", Bs["n_pool"])
    lines = []
    for lab, k in (("greedy portfolio, 16 designs", "greedy_portfolio"), ("first 8 designs of the portfolio", "greedy_first8"),
                   ("the seven designs of APL 126", "apl126_seven_designs"), ("author's mask (16 designs)", "author_mask_16_designs"),
                   ("rotation-only family (8, ratio 0.7)", "rotation_family_8")):
        lines.append(f"{lab} & {fmt(t[k]['E_best_yield'])} & {fmt(t[k]['P_some_design_ge_half'])} \\\\")
        key[f"batch_{k}"] = t[k]
    r = t["random_q_from_pool"]
    lines.append(f"16 random designs from the pool & {fmt(r['mean'])} $\\pm$ {fmt(r['sd'])} & -- \\\\")
    s1 = t["single_best_train_design"]
    macro("singleBestYield", fmt(s1["E_yield"]))
    lines.append(f"best single design (mean yield) & {fmt(s1['E_yield'])} (mean) & {fmt(s1['P_ge_half'])} \\\\")
    out.append("\\newcommand{\\tabBatch}{\\begin{tabular}{lcc}\\toprule set of designs & $E[\\text{best yield}]$ & $P(\\text{some design}\\ge0.5)$\\\\\\midrule\n"
               + "\n".join(lines) + "\n\\bottomrule\\end{tabular}}")
    macro("noWorkingTest", t["n_test_worlds_with_no_working_design"])
    aug = t["mask_plus_k"]
    out.append("\\newcommand{\\tabMaskPlus}{\\begin{tabular}{lcc}\\toprule & $E[\\text{best yield}]$ & $P(\\text{some design}\\ge0.5)$\\\\\\midrule\n"
               f"author's mask (16) & {fmt(t['author_mask_16_designs']['E_best_yield'])} & {fmt(t['author_mask_16_designs']['P_some_design_ge_half'])} \\\\\n"
               + "\n".join(f"+ {a['k']} added design{'s' if a['k'] > 1 else ''} & {fmt(a['E_best_yield'])} & {fmt(a['P_some_design_ge_half'])} \\\\" for a in aug)
               + "\n\\bottomrule\\end{tabular}}")
    add_rows = []
    for j, a in enumerate(Bs["mask_additions"]):
        x = a["x"]
        add_rows.append(f"{j + 1} & {x[0]:.1f} & {x[1]:.2f} & {x[2]:.1f} & {x[3]:.2f} & {x[4]:.0f} \\\\")
    out.append("\\newcommand{\\tabAdd}{\\begin{tabular}{cccccc}\\toprule \\# & $L$ ($\\mu$m) & arm ratio & $w$ ($\\mu$m) & $\\rho$ & rotation ($^\\circ$)\\\\\\midrule\n"
               + "\n".join(add_rows) + "\n\\bottomrule\\end{tabular}}")
    mrows = []
    for a in Bs["per_design_mask"]:
        mrows.append(f"{a['name'].replace('mask:', '')} & {fmt(a['E_test'])} & {fmt(a['P_ge_half'])}")
f = RES / "batch_design.csv"
if f.exists():
    rows = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)
    lines = []
    for r in rows:
        origin = str(r["origin"]).replace("_", "\\_")
        lines.append(f"{int(r['id'])} & {origin} & {r['L_um']:.1f} & {r['arm_ratio']:.2f} & {r['arm_width_um']:.1f} & "
                     f"{r['corner_rho']:.2f} & {r['rotation_deg']:.0f} & {r['E_yield_test']:.2f} & {r['P_yield_ge_half_test']:.2f} \\\\")
    out.append("\\newcommand{\\tabBatchList}{\\begin{tabular}{clcccccccc}\\toprule id & origin & $L$ & ratio & $w$ & $\\rho$ & $\\theta$ & $E$(yield) & $P(\\ge0.5)$\\\\\\midrule\n"
               + "\n".join(lines) + "\n\\bottomrule\\end{tabular}}")

# ---------------------------------------------------------------- accepted worlds re-screened on fresh device populations
Fa = jl("fresh_population_anchors.json")
if Fa:
    macro("freshAnchorLo", fmt(100 * Fa["share_min"], 0)); macro("freshAnchorHi", fmt(100 * Fa["share_max"], 0))
    macro("nFreshAnchorPops", len(Fa["per_seed"]))
    key["fresh_population_anchors"] = Fa

# ---------------------------------------------------------------- the batch scores on accepted worlds that were used neither for selection nor for the headline test
Ow = jl("other_worlds_check.json")
if Ow:
    macro("nOtherWorlds", Ow["n_other_worlds"]); macro("nNontrainWorlds", Ow["n_nontrain_worlds"])
    AB = dict(portfolio_16="Port", portfolio_first8="PortEight", mask_16="Mask", seven_published="Seven", rotation_family_8="Rot",
              mask_plus_3_interior="MaskInt", mask_plus_3_pool="MaskPool")
    for k_, ab in AB.items():
        s_ = Ow["sets"][k_]
        macro("oth" + ab + "Test", fmt(s_["test"])); macro("oth" + ab + "Other", fmt(s_["other"])); macro("oth" + ab + "Non", fmt(s_["nontrain"]))
    d_ = Ow["differences_nontrain"]["portfolio_minus_mask"]
    macro("othDiffPortMaskNon", f"{d_['mean']:+.2f}\\;[{d_['lo']:+.2f},\\,{d_['hi']:+.2f}]")
    d_ = Ow["differences_other_only"]["portfolio_minus_mask"]
    macro("othDiffPortMaskOther", f"{d_['mean']:+.2f}\\;[{d_['lo']:+.2f},\\,{d_['hi']:+.2f}]")
    d_ = Ow["differences_nontrain"]["portfolio_minus_seven"]
    macro("othDiffPortSevenNon", f"{d_['mean']:+.2f}\\;[{d_['lo']:+.2f},\\,{d_['hi']:+.2f}]")
    rows_o = []
    for lab, k_ in (("greedy portfolio (16)", "portfolio_16"), ("author's mask (16)", "mask_16"), ("seven designs of APL 126", "seven_published"),
                    ("rotation-only family (8)", "rotation_family_8"), ("author's mask + 3 interior additions", "mask_plus_3_interior")):
        s_ = Ow["sets"][k_]
        rows_o.append(f"{lab} & {fmt(s_['test'])} & {fmt(s_['other'])} & {fmt(s_['nontrain'])} \\\\")
    out.append("\\newcommand{\\tabOther}{\\begin{tabular}{lccc}\\toprule set of designs & " + f"{Ow['n_test_worlds']} test worlds (headline) & {Ow['n_other_worlds']} other accepted worlds & all {Ow['n_nontrain_worlds']}"
               + "\\\\\\midrule\n" + "\n".join(rows_o) + "\n\\bottomrule\\end{tabular}}")
    key["other_worlds_check"] = Ow

# ---------------------------------------------------------------- additions to the author's mask restricted to the interior box
Mi = jl("mask_additions_interior.json")
if Mi:
    mp = Mi["mask_plus_k"]
    macro("intMaskOnly", fmt(Mi["mask_only"]["E_best_yield"]))
    for k_, nm in ((1, "One"), (2, "Two"), (3, "Three"), (4, "Four")):
        macro("intPlus" + nm, fmt(mp[k_ - 1]["E_best_yield"]))
    rows_ = []
    for j, a in enumerate(Mi["additions"][:4]):
        x = a["x"]
        rows_.append(f"{j + 1} & {x[0]:.1f} & {x[1]:.2f} & {x[2]:.1f} & {x[3]:.2f} & {x[4]:.0f} & {fmt(a['E_test'])} & {fmt(mp[j]['E_best_yield'])} \\\\")
    out.append("\\newcommand{\\tabAddInterior}{\\begin{tabular}{cccccccc}\\toprule \\# & $L$ ($\\mu$m) & arm ratio & $w$ ($\\mu$m) & $\\rho$ & rotation ($^\\circ$) & "
               "$E$(yield) alone & $E[\\text{best}]$ of mask + \\#\\\\\\midrule\n" + "\n".join(rows_) + "\n\\bottomrule\\end{tabular}}")
    key["mask_additions_interior"] = Mi

# ---------------------------------------------------------------- robustness extras (boundary check, fresh devices)
R = jl("robustness_extras.json")
if R:
    wall = [d for d in R["portfolio_info"] if d["on_wall"]]
    macro("nWall", len(wall)); macro("nQ", R["q"])
    macro("nPoolInterior", R["n_pool_interior"]); macro("nPoolAll", R["n_pool"])
    fr = R["test_fresh_devices"]
    macro("freshDev", fr["n_dev"])
    for k, nm in (("portfolio", "Port"), ("interior_portfolio", "Int"), ("author_mask", "Mask"), ("apl126_seven", "Seven")):
        macro("fresh" + nm, fmt(fr[k]["E_best_yield"])); macro("freshP" + nm, fmt(fr[k]["P_some_design_ge_half"]))
    macro("interiorSame", fmt(R["interior_portfolio"]["test_same_devices"]["E_best_yield"]))
    macro("interiorSameP", fmt(R["interior_portfolio"]["test_same_devices"]["P_some_design_ge_half"]))
    if wall:
        macro("wallAreaLo", fmt(min(d["aperture_area_um2"] for d in wall), 0)); macro("wallAreaHi", fmt(max(d["aperture_area_um2"] for d in wall), 0))
    ref = R["reference_apertures"]
    macro("refAreaLo", fmt(min(ref[k]["aperture_area_um2"] for k in ("x07_0", "x07_90", "x06_0", "x06_90")), 0))
    macro("refAreaHi", fmt(max(ref[k]["aperture_area_um2"] for k in ("x07_0", "x07_90", "x06_0", "x06_90")), 0))
    macro("circleArea", fmt(ref["circle"]["aperture_area_um2"], 0))
    ilines = []
    for d in R["portfolio_info"]:
        x = d["x"]
        ilines.append(f"{d['id']} & {x[0]:.1f} & {x[1]:.2f} & {x[2]:.1f} & {x[3]:.2f} & {x[4]:.0f} & {d['aperture_area_um2']:.0f} & "
                      f"{'yes' if d['on_wall'] else ''} \\\\")
    out.append("\\newcommand{\\tabWalls}{\\begin{tabular}{cccccccc}\\toprule \\# & $L$ & ratio & $w$ & $\\rho$ & $\\theta$ & aperture area ($\\mu$m$^2$) & on a wall of the search box\\\\\\midrule\n"
               + "\n".join(ilines) + "\n\\bottomrule\\end{tabular}}")
    itab = R["interior_portfolio"]
    out.append("\\newcommand{\\tabInterior}{\\begin{tabular}{lcc}\\toprule set of designs (16) & $E[\\text{best yield}]$, fresh devices & $P(\\text{some design}\\ge0.5)$\\\\\\midrule\n"
               f"greedy portfolio, whole search box & {fmt(fr['portfolio']['E_best_yield'])} & {fmt(fr['portfolio']['P_some_design_ge_half'])} \\\\\n"
               f"greedy portfolio, interior box only & {fmt(fr['interior_portfolio']['E_best_yield'])} & {fmt(fr['interior_portfolio']['P_some_design_ge_half'])} \\\\\n"
               f"author's mask & {fmt(fr['author_mask']['E_best_yield'])} & {fmt(fr['author_mask']['P_some_design_ge_half'])} \\\\\n"
               f"seven designs of APL 126 & {fmt(fr['apl126_seven']['E_best_yield'])} & {fmt(fr['apl126_seven']['P_some_design_ge_half'])} \\\\\n"
               "\\bottomrule\\end{tabular}}")

# ---------------------------------------------------------------- predictions for the author's mask
M = jl("mask_predictions.json")
if M:
    macro("nMaskWorlds", M["n_worlds"])
    rows = []
    for k, t in M["predictions"].items():
        y, xf, sw, mu = t["yield_"], t["p_x_first"], t["p_switch"], t["mu_sw"]
        ax = t["az_axial"]
        rows.append(f"{k[0]}, {k[1:]}$^\\circ$ & {fmt(y[1])} [{fmt(y[0])}, {fmt(y[2])}] & {fmt(xf[1])} & {ax['mean']:.0f}$^\\circ$ ({ax['R']:.2f}) & "
                    f"{fmt(sw[1])} & {'--' if mu is None else fmt(mu[1])} \\\\")
    out.append("\\newcommand{\\tabMaskPred}{\\begin{tabular}{lccccc}\\toprule design & yield: median [10\\%, 90\\%] & lases first on $x$ (median share) & "
               "azimuth: axial mean ($R$) & switching share (median) & median $\\mu_{sw}$\\\\\\midrule\n"
               + "\n".join(rows[:8]) + "\n\\midrule\n" + "\n".join(rows[8:]) + "\n\\bottomrule\\end{tabular}}")
    orows = [f"{k.replace('>', '$>$')} & {fmt(v['P_greater'])} & {fmt(v['P_within_0.1'])} \\\\" for k, v in M["ordering"].items()]
    out.append("\\newcommand{\\tabOrder}{\\begin{tabular}{lcc}\\toprule comparison (yield) & $P$(first larger) & $P$(within 0.1)\\\\\\midrule\n"
               + "\n".join(orows) + "\n\\bottomrule\\end{tabular}}")
    key["mask_predictions"] = M["predictions"]

# ---------------------------------------------------------------- what the first measured batch can teach (synthetic)
V = jl("value_of_first_batch.json")
if V:
    macro("voiNone", fmt(V["no_measurement"]["mean_yield"])); macro("voiOracle", fmt(V["oracle_mean_yield"]))
    macro("voiPriorSign", fmt(1 - V["prior_sign_kb_positive_fraction"])); macro("voiPriorStd", fmt(V["prior_std_log10_d0"]))
    macro("voiNWorlds", V["n_worlds"])
    rows_v = [v for v in V.values() if isinstance(v, dict) and "n_devices_per_design" in v]
    lines = [f"{v['observation_model']} & {v['temperature']:g} & {v['n_devices_per_design']} & {fmt(v['mean_yield_of_chosen_design'])} & "
             f"{v['mean_ess']:.1f} & {fmt(v['posterior_prob_of_true_sign_kb'])} & {fmt(v['mean_abs_error_log10_d0'])} \\\\" for v in rows_v]
    macro("voiLo", fmt(min(v["mean_yield_of_chosen_design"] for v in rows_v))); macro("voiHi", fmt(max(v["mean_yield_of_chosen_design"] for v in rows_v)))
    macro("voiSignHi", fmt(max(v["posterior_prob_of_true_sign_kb"] for v in rows_v)))
    macro("voiEssLo", f"{min(v['mean_ess'] for v in rows_v):.1f}"); macro("voiEssHi", f"{max(v['mean_ess'] for v in rows_v):.1f}")
    out.append("\\newcommand{\\tabVOI}{\\begin{tabular}{lccccc}\\toprule observation model & tempering $T$ & devices per design & yield of the chosen design & "
               "effective sample size & $P$(true sign of $k_b$)\\\\\\midrule\n" + "\n".join(l.rsplit(" & ", 1)[0] + " \\\\" for l in lines)
               + "\n\\bottomrule\\end{tabular}}")
if M:
    g = M["group_switch_counts"]
    for (k, v), nm in zip(g.items(), ("NearZero", "Mid", "High")):
        macro("cnt" + nm + "N", v["n_devices"]); macro("cnt" + nm + "Mean", f"{v['mean']:.1f}")
        macro("cnt" + nm + "Lo", f"{v['q05']:.0f}"); macro("cnt" + nm + "Hi", f"{v['q95']:.0f}")
    for (k, v), nm in zip(M["group_pass_counts"].items(), ("NearZero", "Mid", "High")):
        macro("cnt" + nm + "PassMean", f"{v['mean']:.1f}"); macro("cnt" + nm + "PassLo", f"{v['q05']:.0f}"); macro("cnt" + nm + "PassHi", f"{v['q95']:.0f}")
    xl = {k: v["share_worlds_x_like"] for k, v in M["predictions"].items()}
    macro("xLikeAsixty", fmt(100 * xl["A60"], 0)); macro("xLikeAonetwenty", fmt(100 * xl["A120"], 0))
    macro("xLikeBsixty", fmt(100 * xl["B60"], 0)); macro("xLikeBonetwenty", fmt(100 * xl["B120"], 0))

# ---------------------------------------------------------------- mask family A with rounded arm tips (the corner treatment of family A is an assumption)
Fr = jl("mask_family_a_rounded.json")
if Fr:
    macro("rhoFamilyA", f"{Fr['rho_family_A']:g}")
    for (gs, gr), nm in zip(zip(Fr["sharp_family_A"].values(), Fr["rounded_family_A"].values()), ("NearZero", "Mid", "High")):
        macro("cnt" + nm + "RoundMean", f"{gr['mean']:.1f}"); macro("cnt" + nm + "RoundLo", f"{gr['q05']:.0f}"); macro("cnt" + nm + "RoundHi", f"{gr['q95']:.0f}")
    key["mask_family_a_rounded"] = Fr

# ---------------------------------------------------------------- sensitivity
S = jl("sensitivity.json")
if S:
    lines = []
    for k, v in S.items():
        if isinstance(v, dict) and isinstance(v.get("portfolio"), dict):
            kk = k.replace("%", "\\%").replace("_", "\\_")
            lines.append(f"{kk} & {fmt(v['portfolio']['E_best'])} & {fmt(v['author_mask']['E_best'])} & "
                         f"{fmt(v['apl126_seven']['E_best'])} \\\\")
    out.append("\\newcommand{\\tabSens}{\\begin{tabular}{lccc}\\toprule scenario & portfolio (16) & author's mask (16) & APL 126 seven\\\\\\midrule\n"
               + "\n".join(lines) + "\n\\bottomrule\\end{tabular}}")
    key["sensitivity"] = {k: v for k, v in S.items() if isinstance(v, dict)}
    sc = {k: v for k, v in S.items() if isinstance(v, dict) and isinstance(v.get("portfolio"), dict)}
    ok = {k: v["portfolio"]["E_best"] > v["author_mask"]["E_best"] > v["apl126_seven"]["E_best"] for k, v in sc.items()}
    macro("sensN", len(sc)); macro("sensOk", sum(ok.values()))
    macro("sensPortAboveMask", sum(v["portfolio"]["E_best"] > v["author_mask"]["E_best"] for v in sc.values()))
    macro("sensPortAboveSeven", sum(v["portfolio"]["E_best"] > v["apl126_seven"]["E_best"] for v in sc.values()))
    macro("sensPortMin", fmt(min(v["portfolio"]["E_best"] for v in sc.values()))); macro("sensPortMax", fmt(max(v["portfolio"]["E_best"] for v in sc.values())))
    key["sensitivity_ordering_ok"] = ok
    key["sensitivity_skipped"] = S.get("skipped_scenarios", {})

import re  # noqa: E402
bib = (ROOT / "docs" / "references.bib").read_text()
rows = []
for m in re.finditer(r"@\w+\{([^,\s]+),(.*?)\n\}", bib, flags=re.S):
    st = re.search(r"STATUS:\s*([A-Z-]+)", m.group(2))
    rows.append(f"\\texttt{{{m.group(1)}}} & {st.group(1) if st else '?'} \\\\")
out.append("\\newcommand{\\tabRefStatus}{\\begin{tabular}{ll}\\toprule key & verification status\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}")

(SUP / "tables.tex").write_text("% generated by scripts/make_tables.py - do not edit\n" + "\n".join(out) + "\n")
(RES / "key_numbers.json").write_text(json.dumps(key, indent=1, default=str))
print(f"wrote supplementary/tables.tex ({len(out)} items) and results/key_numbers.json")
