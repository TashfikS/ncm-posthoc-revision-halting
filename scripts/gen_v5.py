"""Generate numbers_v5.tex, tab_v5.tex, tab_dagaug.tex from the scaled (v5) Kaggle result files."""
import json, glob, re, sys, numpy as np
from scipy import stats
OUT = sys.argv[1] if len(sys.argv) > 1 else "tmlr_sub"
J = lambda f: json.load(open(f))
def merge(*fs):
    R = J(fs[0]); R = dict(R); R["seeds"] = dict(R["seeds"])
    for f in fs[1:]: R["seeds"].update(J(f)["seeds"])
    return R
V5 = {"sudoku-easy": merge("kout/v5_sudoku-easy/results/sudoku-easy/results.json"),
      "sudoku-medium": merge("kout/v5_sudoku-medium/results/sudoku-medium/results.json"),
      "sudoku-hard": merge("kout/v5_sudoku-hard/results/sudoku-hard/results.json"),
      "wmt14": merge("kout/v5_wmt14/results/wmt14/results.json"),
      "xsum": merge("kout/v5_xsum/results/xsum/results.json", "kout/v5b_xsum/results/xsum-s3/results.json")}
PK = {"sudoku-easy": ("exact", "exact"), "sudoku-medium": ("exact", "exact"), "sudoku-hard": ("exact", "exact"),
      "wmt14": ("bleu", "sbleu"), "xsum": ("rougeL", "rougeL")}
NAME = {"sudoku-easy": "Sudoku-easy", "sudoku-medium": "Sudoku-medium", "sudoku-hard": "Sudoku-hard", "wmt14": "WMT14 (BLEU)", "xsum": "XSum (R-L)"}
MAC = {"sudoku-easy": "Easy", "sudoku-medium": "Med", "sudoku-hard": "Hard", "wmt14": "Wmt", "xsum": "Xsum"}
def mcn(a, b):
    a = np.array(a) > .5; b = np.array(b) > .5; n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
def wil(a, b):
    d = np.array(b) - np.array(a)
    return 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(a, b).pvalue)
def fp(p):
    if p >= 0.995: return "1.00"
    if p < 1e-4: return r"$<\!10^{-4}$"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
mac = []; rows = []; summ = {}
VARS = [("loop", "In-loop, no selection"), ("loop_sel", "In-loop, val.-selected $s$"), ("proj", "Projection, no selection"), ("proj_sel", "Projection, val.-selected $s$")]
for t, R in V5.items():
    pk, ik = PK[t]; b = R["base_test"]; base = b["scores"][pk]; S = R["seeds"]; sd = sorted(S, key=lambda x: int(x))
    summ[t] = {"base": base, "n_train": R["n"]["train"] if isinstance(R["n"], dict) else None}
    mac.append(rf"\newcommand{{\VfBase{MAC[t]}}}{{{base:.2f}}}")
    cells = [NAME[t], f"{base:.2f}"]; sub = ["", ""]
    for v, _ in VARS:
        x = np.array([S[s][v]["scores"][pk] for s in sd])
        tp = stats.ttest_1samp(x, base).pvalue if x.std() > 1e-9 else 1.0
        pp = [mcn(b["items"][ik], S[s][v]["items"][ik]) if pk == "exact" else wil(b["items"][ik], S[s][v]["items"][ik]) for s in sd]
        nsig = sum((p < 0.05) and (S[s][v]["scores"][pk] > base) for p, s in zip(pp, sd))
        nneg = sum((p < 0.05) and (S[s][v]["scores"][pk] < base) for p, s in zip(pp, sd))
        summ[t][v] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "t_p": float(tp), "per_seed": x.tolist(), "paired_p": pp, "n_sig_pos": nsig, "n_sig_neg": nneg}
        tag = "".join(c for c in v.title() if c.isalpha())
        mac += [rf"\newcommand{{\Vf{tag}{MAC[t]}}}{{{x.mean():.2f}}}", rf"\newcommand{{\Vf{tag}{MAC[t]}Std}}{{{x.std(ddof=1):.2f}}}",
                rf"\newcommand{{\Vf{tag}{MAC[t]}P}}{{{fp(tp)}}}", rf"\newcommand{{\Vf{tag}{MAC[t]}Min}}{{{x.min():.2f}}}", rf"\newcommand{{\Vf{tag}{MAC[t]}Max}}{{{x.max():.2f}}}"]
        cells.append(f"{x.mean():.2f}$\\pm${x.std(ddof=1):.2f}"); sub.append(f"({fp(tp)}; {nsig}/{nneg})")
    sl = [S[s]["s_loop"] for s in sd]; spj = [S[s]["s_proj"] for s in sd]
    summ[t]["s_loop"] = sl; summ[t]["s_proj"] = spj
    mac.append(rf"\newcommand{{\VfSzero{MAC[t]}}}{{{sum(1 for q in sl if q == 0)}}}")
    cells.append(",".join(f"{q:g}" for q in sl)); sub.append("")
    rows.append(" & ".join(cells) + r" \\" + "\n" + " & ".join(r"{\scriptsize " + c + "}" if c else "" for c in sub) + r" \\[2pt]")
tab = r"""\begin{table*}[t]
\centering
\caption{Scaled re-evaluation on the full official test sets (Sudoku: 1{,}500 held-out boards per difficulty; WMT14: 3{,}003 sentences; XSum: first 1{,}500 articles) with correctors trained on externally sourced, test-disjoint splits (Section~\ref{sec:scaled}). Mean$\pm$std over five seeds; second line, in parentheses: one-sample $t$-test of the seed means against NCM, and the number of seeds significantly better/worse than NCM in a paired test (McNemar for Sudoku, Wilcoxon over sentence-level scores for text; $p<0.05$). Last column: closed-loop step-size scale $s$ selected on validation for each seed (in-loop).}
\label{tab:v5}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccccc}
\toprule
Task & NCM & In-loop, no sel. & In-loop, val.-sel. & Projection, no sel. & Projection, val.-sel. & selected $s$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
open(f"{OUT}/tab_v5.tex", "w").write(tab)
# ---- DAgger on augmented data
drows = []
for d in ("hard", "medium"):
    base = None; arms = {}
    for arm in ("dagger", "dagger_c"):
        ex = []; pp = []; sel = 0
        for f in sorted(glob.glob(f"kout/dagaug4_{d}/results/sudoku-{d}-{arm}-s*/results.json")):
            R = J(f); b = R["base_test"]; base = b["scores"]["exact"]
            for s, v in R["arms"][arm].items():
                ex.append(v["test"]["scores"]["exact"]); pp.append(mcn(b["items"]["exact"], v["test"]["items"]["exact"])); sel += v["best"] == -1
        ex = np.array(ex); tp = stats.ttest_1samp(ex, base).pvalue if ex.std() > 1e-9 else 1.0
        arms[arm] = (ex, tp, pp, sel)
        tag = {"dagger": "Dag", "dagger_c": "DagC"}[arm]; D = d.title()
        mac += [rf"\newcommand{{\Va{tag}{D}}}{{{ex.mean():.2f}}}", rf"\newcommand{{\Va{tag}{D}Std}}{{{ex.std(ddof=1):.2f}}}", rf"\newcommand{{\Va{tag}{D}P}}{{{fp(tp)}}}",
                rf"\newcommand{{\Va{tag}{D}Max}}{{{ex.max():.2f}}}", rf"\newcommand{{\Va{tag}{D}MinP}}{{{fp(min(pp))}}}", rf"\newcommand{{\Va{tag}{D}Kept}}{{{sel}}}"]
        summ.setdefault("dagaug", {})[f"{d}_{arm}"] = {"mean": float(ex.mean()), "sd": float(ex.std(ddof=1)), "t_p": float(tp), "per_seed": ex.tolist(), "mcnemar_p": pp, "kept_official": int(sel), "n": len(ex)}
    drows.append(f"Sudoku-{d} & {base:.2f} & " + " & ".join(f"{arms[a][0].mean():.2f}$\\pm${arms[a][0].std(ddof=1):.2f} & {fp(arms[a][1])} & {fp(min(arms[a][2]))} & {arms[a][3]}/{len(arms[a][0])}" for a in ("dagger", "dagger_c")) + r" \\")
open(f"{OUT}/tab_dagaug.tex", "w").write(r"""\begin{table*}[t]
\centering
\caption{Generator co-adaptation with 12{,}000 symmetry-augmented training boards (4 DAgger rounds of 1{,}000 fresh boards; checkpoint and corrector selected on 1{,}200 augmented validation boards; 1{,}500 held-out test boards; five seeds). Columns per arm: mean$\pm$std solve rate, one-sample $t$-test of seed means against NCM, smallest per-seed McNemar $p$, and the number of seeds for which validation selection kept the official generator.}
\label{tab:dagaug}
\small
\resizebox{\textwidth}{!}{\begin{tabular}{lc cccc cccc}
\toprule
& & \multicolumn{4}{c}{DAgger} & \multicolumn{4}{c}{DAgger + $C_\omega$}\\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
Task & NCM & solved & $t$-test $p$ & min.\ McN.\ $p$ & kept & solved & $t$-test $p$ & min.\ McN.\ $p$ & kept\\
\midrule
""" + "\n".join(drows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
""")
# ---- probe
for t, f in [("Hard", "kout/v5_sudoku-hard/results/sudoku-hard-probe/probe.json"), ("Med", "kout/v5_sudoku-medium/results/sudoku-medium-probe/probe.json"),
             ("Easy", "kout/v5_sudoku-easy/results/sudoku-easy-probe/probe.json"), ("Xsum", "kout/v5b_xsum/results/xsum-probe/probe.json")]:
    Q = J(f); pk = "rougeL" if t == "Xsum" else "exact"
    lr = Q["label_rate_by_level_test"][:-1]
    mac += [rf"\newcommand{{\Pr{t}Lab}}{{{100*max(lr):.1f}}}", rf"\newcommand{{\Pr{t}Lev}}{{{Q['test']['levels']:.2f}}}",
            rf"\newcommand{{\Pr{t}Q}}{{{Q['test']['scores'][pk]:.2f}}}", rf"\newcommand{{\Pr{t}Full}}{{{Q['test_full']['scores'][pk]:.2f}}}"]
    mac.append(rf"\newcommand{{\Pr{t}Save}}{{{Q['test_full']['levels']-Q['test']['levels']:.2f}}}")
    if Q["auroc_test_pooled"] is not None: mac.append(rf"\newcommand{{\Pr{t}Auc}}{{{Q['auroc_test_pooled']:.2f}}}")
    summ.setdefault("probe", {})[t] = {"label_rate_by_level": lr, "auroc": Q["auroc_test_pooled"], "test": Q["test"], "full": Q["test_full"]}
# ---- perturbation pilot
P = J("kout/perturb/results/perturb/results.json"); bv = P["official"]["base_val"]["scores"]["exact"]
log = open("kout/perturb/results/perturb/log.txt").read()
best = {}
for sg in ("0.05", "0.1"):
    v = [float(m) for m in re.findall(rf"\[sigma {re.escape(sg)}\] ep\d+ loss [\d.]+ val \{{'exact': ([\d.]+)", log)]
    best[sg] = (max(v), len(v))
allv = [float(m) for m in re.findall(r"ep\d+ loss [\d.]+ val \{'exact': ([\d.]+)", log)]
mac += [rf"\newcommand{{\PtBaseVal}}{{{bv:.1f}}}", rf"\newcommand{{\PtFiveMax}}{{{best['0.05'][0]:.1f}}}", rf"\newcommand{{\PtTenMax}}{{{best['0.1'][0]:.1f}}}",
        rf"\newcommand{{\PtMin}}{{{min(allv):.1f}}}"]
os_ = P["official"]["seeds"]; ls = [os_[s]["loop_sel"]["scores"]["exact"] for s in os_]; l1 = [os_[s]["loop_1.0"]["scores"]["exact"] for s in os_]
mac += [rf"\newcommand{{\PtLoopSelMin}}{{{min(ls):.2f}}}", rf"\newcommand{{\PtLoopSelMax}}{{{max(ls):.2f}}}", rf"\newcommand{{\PtLoopOneMin}}{{{min(l1):.1f}}}", rf"\newcommand{{\PtLoopOneMax}}{{{max(l1):.1f}}}"]
summ["perturb"] = {"base_val": bv, "best_val_by_sigma": best, "official_loop_sel": ls, "official_loop_1": l1}
open(f"{OUT}/numbers_v5.tex", "w").write("\n".join(mac) + "\n")
json.dump(summ, open("summary_v5.json", "w"), indent=1, default=float)
print(len(mac), "macros")
