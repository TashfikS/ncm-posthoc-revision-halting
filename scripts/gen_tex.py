"""Generate numbers.tex and table fragments from summary.json / raw results (no hand-typed numbers)."""
import json, re, numpy as np
S = json.load(open("summary.json")); F = json.load(open("flux_gt.json"))
P = "paper/"
TASKS = [t for t in ["sudoku-easy", "sudoku-medium", "sudoku-hard", "wmt14", "xsum"] if t in S]
PK = {"sudoku-easy": "exact", "sudoku-medium": "exact", "sudoku-hard": "exact", "wmt14": "bleu", "xsum": "rougeL"}
NAME = {"sudoku-easy": "Sudoku-easy", "sudoku-medium": "Sudoku-medium", "sudoku-hard": "Sudoku-hard", "wmt14": "WMT14 De--En", "xsum": "XSum"}
MET = {"exact": "solved \\%", "bleu": "BLEU", "rougeL": "R-L"}


def f(x, d=2): return f"{x:.{d}f}"
def pv(p):
    if p >= 0.995: return "1.0"
    if p < 1e-4: return "$<$$10^{-4}$"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


M = {}
def off(t, pat):
    m = re.findall(pat, S[t]["official_stdout"]); return m
for t, key in (("sudoku-easy", "Easy"), ("sudoku-medium", "Medium"), ("sudoku-hard", "Hard")):
    M["Off" + key] = off(t, r"\n\s+([\d.]+)\s+\d+/\d+")[0]
    M["Base" + key] = f(S[t]["baseline"]["exact"])
    M["LoopVone" + key] = f(S[t]["loop_v1"]["pc"]["exact"][0])
M["OffWmt"] = off("wmt14", r"\n\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+")[0]
if "xsum" in S:
    m = off("xsum", r"\n\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")[0]
    M["OffXsumRone"], M["OffXsumRtwo"], M["OffXsumRL"] = m[0], m[1], m[2]
    M["BaseXsum"] = f(S["xsum"]["baseline"]["rougeL"]); M["LoopVoneXsum"] = f(S["xsum"]["loop_v1"]["pc"]["rougeL"][0])
M["BaseWmt"] = f(S["wmt14"]["baseline"]["bleu"]); M["LoopVoneWmt"] = f(S["wmt14"]["loop_v1"]["pc"]["bleu"][0])
M["LDSud"] = f(max(F["sudoku-hard"]["LD2_hat_max"]), 3); M["LDWmt"] = f(max(F["wmt14"]["LD2_hat_max"]), 3)
M["RhoGtOneSud"] = str(sum(r > 1 for r in F["sudoku-hard"]["rho_median"])); M["RhoGtOneWmt"] = str(sum(r > 1 for r in F["wmt14"]["rho_median"]))
M["FluxLastSud"] = f(F["sudoku-hard"]["rel_flux"][-2], 2); M["FluxLastWmt"] = f(F["wmt14"]["rel_flux"][-2], 2)
M["CorrParams"] = f(S["sudoku-hard"]["loop_v1"]["corr_params"] / 1e6, 2)
for t, key in (("sudoku-hard", "Hard"), ("sudoku-medium", "Medium")):
    z = S[t]["zs_full"]
    M["ZsFluxFive" + key] = f(z["flux_0.05"]["exact"], 1); M["ZsFluxFiveLev" + key] = f(z["flux_0.05"]["levels"], 2)
    M["ZsStabTen" + key] = f(z["stab_0.1"]["exact"], 1); M["ZsStabTenLev" + key] = f(z["stab_0.1"]["levels"], 2)
    M["ZsBase" + key] = f(z["baseline"]["exact"], 1)
M["ZsFluxFifteenMedium"] = f(S["sudoku-medium"]["zs_full"]["flux_0.15"]["exact"], 1)
M["ZsFluxFifteenLevMedium"] = f(S["sudoku-medium"]["zs_full"]["flux_0.15"]["levels"], 2)
M["LoopVtwoHard"] = f(S["sudoku-hard"]["loop_v2"]["pc"]["exact"][0]); M["LoopVtwoHardStd"] = f(S["sudoku-hard"]["loop_v2"]["pc"]["exact"][1])
M["LoopVtwoHardP"] = pv(S["sudoku-hard"]["loop_v2"]["pc"]["mcnemar_p_min"])
M["LoopVtwoHardT"] = pv(S["sudoku-hard"]["loop_v2"]["pc"]["exact_t_p"])
open(P + "numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))

# ------------------------------------------------ Table I
rows = []
for t in TASKS:
    s, pk = S[t], PK[t]; b = s["baseline"]
    def line(name, sc, sd, lev, flops, ms, mb, p1, p2):
        pm = ("$\\pm$" + sd) if sd is not None else ""
        return f" & {name} & {sc}{pm} & {lev} & {flops} & {ms} & {mb} & {p1} & {p2}\\\\"
    gf = lambda x: f"{x/1e9:.1f}"
    nb = s["n_test"]
    blk = [line("NCM (official)", f(b[pk]), None, f(b["levels"], 1), gf(b["flops"] / s_bs(t) if False else b["flops"]), f(b["ms"], 0), f(b["peak_mb"], 0), "--", "--")]
    h = s["halt_only"]
    blk.append(line("+ halting", f(h[pk]), None, f(h["levels"], 1), gf(h["flops"]), f(h["ms"], 0), f(h["peak_mb"], 0), pv(h["wilcoxon_p"]), "--"))
    for tag, nm in (("loop_v1", "in-loop, no sel."), ("loop_v2", "in-loop, val.-sel."), ("decode_v3", "decode-path")):
        if tag not in s: continue
        for part, pre in (("pc", "$C_\\omega$"), ("full", "$C_\\omega$+halt")):
            x = s[tag][part]
            p2 = pv(x["mcnemar_p_min"]) if "mcnemar_p_min" in x else pv(x[pk + "_t_p"])
            blk.append(line(f"{pre} ({nm})", f(x[pk][0]), f(x[pk][1]), f(x["levels"][0], 2), gf(x.get("flops", 0)), f(x["ms"][0], 0), f(x["peak_mb"], 0), pv(x["wilcoxon_p_min"]), p2))
    blk[0] = f"\\multirow{{{len(blk)}}}{{*}}{{\\rotatebox{{90}}{{\\scriptsize {NAME[t]}}}}}" + blk[0]
    rows.append("\n".join(blk))
def s_bs(t): return 1
tab = r"""\begin{table*}[t]
\centering
\caption{Main comparison on official NCM checkpoints (single Tesla T4). Quality: Sudoku boards solved (\%), WMT14 corpus BLEU, XSum ROUGE-L, on held-out test items (Sudoku 1{,}500, WMT14 2{,}003, XSum 1{,}000). Learned rows: mean$\pm$std over seeds \{42,0,1,2,3\}. $\bar k$: mean levels executed. GFLOPs: per evaluation batch, measured. ms: wall-clock per batch (Sudoku 128, WMT14 64, XSum 32 sequences). MB: peak allocated GPU memory. $p_W$: smallest paired Wilcoxon $p$ over seeds vs.\ NCM (per-item cell acc./sentence BLEU/ROUGE-L). $p_2$: smallest exact McNemar $p$ on solved boards (Sudoku) or one-sample $t$-test of seed means (text). ``no sel.'': trained without validation checkpoint and step-size selection.}
\label{tab:main}
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{llccccccc}
\toprule
 & Method & Quality & $\bar k$ & GFLOPs & ms & MB & $p_W$ & $p_2$\\
\midrule
""" + "\n\\midrule\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(P + "tab_main.tex", "w").write(tab)

# ------------------------------------------------ Table II: level-by-level
lines = []
for t, m, lab in (("sudoku-hard", "cell", "Sudoku-hard cell acc."), ("sudoku-hard", "exact", "Sudoku-hard solved"), ("wmt14", "bleu", "WMT14 BLEU"), ("xsum", "rougeL", "XSum R-L")):
    if t not in S: continue
    s = S[t]; K = len(s["per_level_base"])
    for tag, nm in (("base", "NCM"), ("loop_v1", "in-loop $C_\\omega$ (no sel.)"), ("decode_v3", "decode-path $C_\\omega$")):
        pl = s["per_level_base"] if tag == "base" else (s.get(tag, {}) or {}).get("per_level_pc42")
        if not pl: continue
        vals = [f(x[m], 1) for x in pl] + ["--"] * (12 - K)
        lines.append(f"{lab if tag=='base' else ''} & {nm} & " + " & ".join(vals) + "\\\\")
    lines.append("\\midrule")
tab2 = r"""\begin{table*}[t]
\centering
\caption{Quality of the decoded intermediate latent $D_\psi(\zh^{(k)})$ (or $D_\psi(\zt^{(k)})$) at every level $k$, seed 42, test items. Fine-drop makes every level decodable, but quality rises steeply only over the last third of the hierarchy. WMT14 has 11 levels and XSum 9.}
\label{tab:levels}
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{ll""" + "c" * 12 + r"""}
\toprule
 & & \multicolumn{12}{c}{level $k$}\\
 & & """ + " & ".join(str(i) for i in range(1, 13)) + r"""\\
\midrule
""" + "\n".join(lines[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(P + "tab_levels.tex", "w").write(tab2)

# ------------------------------------------------ Table III: sensitivity
R1 = json.load(open("kout/sudoku_v1/results/sudoku-hard/results.json")); R2 = json.load(open("kout/sudoku_v2/results/sudoku-hard/results.json"))
L = []
z = S["sudoku-hard"]["tau_test_base"]
for tau in ("0.01", "0.03", "0.05", "0.1", "0.15"):
    a, b = z[f"flux_{tau}"], z[f"stab_{tau}"]
    L.append(f"$\\tau={tau}$ & {f(a['exact'],1)} / {f(a['levels'],2)} & {f(b['exact'],1)} / {f(b['levels'],2)} & -- & -- & --\\\\")
L.append("\\midrule")
for nm, R in (("no sel.", R1), ("val.-sel.", R2)):
    for eta, x in R["sens_eta"].items():
        L.append(f"$\\eta_k\\equiv{eta}$ ({nm}) & {f(x['scores']['exact'],1)} & {f(x['scores']['cell'],1)} & {x['params']/1e6:.2f}M & {x['train_s']:.0f} & {x['train_peak_mb']:.0f}\\\\")
L.append("\\midrule")
for nm, R in (("no sel.", R1), ("val.-sel.", R2)):
    for tg, x in R["sens_size"].items():
        L.append(f"size {x['params']/1e6:.2f}M ({nm}) & {f(x['scores']['exact'],1)} & {f(x['scores']['cell'],1)} & {x['params']/1e6:.2f}M & {x['train_s']:.0f} & {x['train_peak_mb']:.0f}\\\\")
tab3 = r"""\begin{table}[t]
\centering
\caption{Sensitivity on Sudoku-hard (1{,}500 test boards; NCM: """ + f(S["sudoku-hard"]["baseline"]["exact"], 1) + r"""\% solved, """ + f(S["sudoku-hard"]["baseline"]["cell"], 1) + r"""\% cells). Top: halting threshold on uncorrected latents, cells give solved\,\% / $\bar k$ for the flux and stability rules. Middle and bottom: in-loop corrector with fixed $\eta_k$ (no step-size calibration) and different sizes (learned $\eta_k$, initialized at 0.5), seed 42, without and with validation checkpoint selection; columns give solved\,\%, cell\,\%, parameters, training time (s, 40 epochs on T4) and peak training memory (MB).}
\label{tab:sens}
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lccccc}
\toprule
Setting & flux / solved & stab / cell & params & s & MB\\
\midrule
""" + "\n".join(L) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(P + "tab_sens.tex", "w").write(tab3)

# ------------------------------------------------ Table IV: hyper-parameters
hp = r"""\begin{table}[t]
\centering
\caption{Hyper-parameters. Generator and tokenizer settings are those of the official evaluation configs.}
\label{tab:hp}
\footnotesize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccc}
\toprule
 & Sudoku & WMT14 & XSum\\
\midrule
levels $K$ & 12 & 11 & 9\\
$L$, $d$ & 89, 256 & 64, 768 & 48, 768\\
generator params & """ + f"{S['sudoku-hard']['gen_params']/1e6:.1f}M & {S['wmt14']['gen_params']/1e6:.1f}M & " + (f"{S['xsum']['gen_params']/1e6:.1f}M" if 'xsum' in S else '--') + r"""\\
$C_\omega$ params (width) & """ + f"{S['sudoku-hard']['loop_v1']['corr_params']/1e6:.2f}M & {S['wmt14']['loop_v1']['corr_params']/1e6:.2f}M & " + (f"{S['xsum']['loop_v1']['corr_params']/1e6:.2f}M" if 'xsum' in S else '--') + r"""\\
train / test items & 500 / 1500 & 1000 / 2003 & 500 / 1000\\
epochs & 40 & 20 & 30\\
batch (train / eval) & 32 / 128 & 32 / 64 & 32 / 32\\
optimizer & \multicolumn{3}{c}{AdamW~\cite{loshchilov2019adamw}, lr $3{\times}10^{-4}$, wd 0.01, one-cycle, clip 1.0}\\
$\alpha$ (KL weight) & \multicolumn{3}{c}{0.1}\\
$\eta_k$ init / scale grid & \multicolumn{3}{c}{$\sigma(\beta_k)=0.5$ / $s\in\{0,0.25,0.5,1\}$}\\
rollout / corruption mix & \multicolumn{3}{c}{0.75 / 0.25 (2 corrupted copies, $u\sim\mathcal{U}(0,1)$)}\\
$\tau$ grid, $k_{\min}$ & \multicolumn{3}{c}{\{0.01,0.03,0.05,0.10,0.15\}, 2}\\
$\tau_{\mathrm{tok}}$ grid & \multicolumn{3}{c}{\{0.005,0.01,0.02,0.05\}}\\
selection tolerance & \multicolumn{3}{c}{0.5 points of full-depth quality}\\
seeds & \multicolumn{3}{c}{42, 0, 1, 2, 3}\\
\bottomrule
\end{tabular}
\end{table}
"""
open(P + "tab_hp.tex", "w").write(hp)

# ------------------------------------------------ qualitative boards
R = R1; b = R["baseline"]["per_level_hyps"]; c = R["pc"]["42"]["per_level_hyps"]
import pickle
refs = R1["baseline"]["items"]["exact"]
idx = next(i for i in range(len(refs)) if refs[i] > 0.5 and R["pc"]["42"]["items"]["exact"][i] < 0.5)
def board(s):
    s = "".join(ch if ch.isdigit() else "." for ch in s if ch.isdigit() or ch == "_")[:81].ljust(81, "."); return "\\\\".join(" ".join(s[r*9:(r+1)*9]) for r in range(9))
q = [r"\begin{figure*}[t]\centering\scriptsize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule & level 3 & level 6 & level 9 & level 12\\\midrule"]
for nm, H in (("NCM", b), ("in-loop $C_\\omega$ (no sel.)", c)):
    cells = [r"\texttt{\begin{tabular}{@{}c@{}}" + board(H[k][idx]) + r"\end{tabular}}" for k in (2, 5, 8, 11)]
    q.append(nm + " & " + " & ".join(cells) + r"\\\midrule")
q[-1] = q[-1].replace(r"\midrule", "")
q += [r"\bottomrule\end{tabular}", r"\caption{Intermediate decodings of Sudoku-hard test board " + str(idx + 500) + r" (seed 42). Both samplers produce near-identical coarse boards; the in-loop corrector perturbs the generator input from level 6 on and the final board is wrong, while NCM solves it. A dot marks a position whose decoded token is not a digit.}", r"\label{fig:qual}\end{figure*}"]
open(P + "fig_qual.tex", "w").write("\n".join(q))
print(M)

# ------------------------------------------------ Table V: DAgger co-adaptation + v4 diagnostics macros
from scipy import stats as _st
DG = {}
rows = []
for d in ("easy", "medium", "hard"):
    Rd = json.load(open(f"kout/dagger_{d}/results/sudoku-{d}/results.json")); b = Rd["base_test"]
    cells = [f"Sudoku-{d}", f(b["scores"]["exact"])]
    for arm in ("tf", "dagger", "dagger_c"):
        Sa = Rd["arms"][arm]; ex = [Sa[s]["test"]["scores"]["exact"] for s in Sa]
        tp = _st.ttest_1samp(ex, b["scores"]["exact"]).pvalue if np.std(ex) > 0 else 1.0
        pm = []
        for s in Sa:
            x0 = np.array(b["items"]["exact"]) > .5; x1 = np.array(Sa[s]["test"]["items"]["exact"]) > .5
            n01, n10 = int((~x0 & x1).sum()), int((x0 & ~x1).sum()); pm.append(_st.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0)
        nsel = sum(1 for s in Sa if Sa[s]["best"] == -1)
        cells.append(f"{np.mean(ex):.2f}$\\pm${np.std(ex, ddof=1):.2f} ({pv(tp)}; {nsel}/5)")
        DG[(d, arm)] = (np.mean(ex), tp)
    ts = np.mean([Rd["arms"][a_]["42"]["train_s"] for a_ in ("tf", "dagger", "dagger_c")])
    rows.append(" & ".join(cells) + "\\\\")
tab5 = r"""\begin{table*}[t]
\centering
\caption{Closed-loop co-adaptation of the generator on Sudoku (1{,}500 held-out boards; 5 seeds). All arms fine-tune the 25.3M non-embedding generator parameters on the 400 training boards of $N_{\mathrm{tr}}$ for 3 rounds $\times$ 4 epochs (AdamW, lr $3{\times}10^{-5}$) and select the checkpoint by closed-loop solve rate on the 100 validation boards (the official model is a candidate). \emph{Control}: teacher-forced reference states. \emph{DAgger}: 50/50 reference and on-policy rollout states, labels given by greedy residual quantization of $z-\text{state}$. \emph{DAgger}+$C_\omega$: rollout states revised by a jointly trained in-loop corrector. Cells: solved \% mean$\pm$std (seed-mean $t$-test $p$ vs.\ NCM; number of seeds for which the official model was selected).}
\label{tab:dagger}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
 & NCM & Control (TF fine-tune) & DAgger & DAgger + $C_\omega$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(P + "tab_dagger.tex", "w").write(tab5)
D4 = json.load(open("diag_v4.json"))
M2 = {"ProjNCM": f(D4["proj"]["ncm"]["exact"], 1), "ProjNoSelOff": f(D4["proj"]["v1_proj0"]["exact"], 1),
      "ProjNoSelOn": f(D4["proj"]["v1_proj1"]["exact"], 1), "ProjSelOff": f(D4["proj"]["v2_proj0"]["exact"], 1),
      "ProjSelOn": f(D4["proj"]["v2_proj1"]["exact"], 1)}
mg = D4["margin"]
M2["MarginEqMax"] = f(max(r["eq_final_pct"] for t in mg for r in mg[t][:-1]), 0)
M2["MarginMedMax"] = f(max(mg[t][-2]["mu_median"] for t in mg), 3)
fl = [mg[t][-1]["flips_vs_prev"] for t in mg]; M2["FlipsMin"] = f(min(fl), 1); M2["FlipsMax"] = f(max(fl), 1)
for (d, arm), (m_, p_) in DG.items():
    key = {"tf": "Tf", "dagger": "Dag", "dagger_c": "DagC"}[arm] + d.capitalize()
    M2["Dg" + key] = f(m_); M2["DgP" + key] = pv(p_)
WM = json.load(open("kout/wmt14_v1/results/wmt14/results.json"))
vm = [WM["corr_info"][s]["hist"][-1]["val_mse"] for s in WM["corr_info"]]; vu = [WM["corr_info"][s]["hist"][0]["val_mse_uncorrected"] for s in WM["corr_info"]]
M2["WmtMseLo"], M2["WmtMseHi"], M2["WmtUncLo"], M2["WmtUncHi"] = f(min(vm)), f(max(vm)), f(min(vu)), f(max(vu))
open(P + "numbers.tex", "a").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M2.items()))
print(M2)
