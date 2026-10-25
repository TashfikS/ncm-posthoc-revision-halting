import json, os, numpy as np
import matplotlib.pyplot as plt

SUM = json.load(open("summary.json"))
FLUX = json.load(open("flux_gt.json"))
os.makedirs("paper/figs", exist_ok=True)
C_BASE, C_APC, C_GREY = "#4d4d4d", "#d95f02", "#9e9e9e"
TT = [("sudoku-hard", "cell", "Sudoku-hard cell acc. (%)"), ("wmt14", "bleu", "WMT14 BLEU"), ("xsum", "rougeL", "XSum ROUGE-L")]
TT = [t for t in TT if t[0] in SUM]

# ---------------- Fig 2: level-by-level decoding
fig, axs = plt.subplots(1, len(TT), figsize=(7.16, 2.2))
for ax, (t, m, lab) in zip(np.atleast_1d(axs), TT):
    S = SUM[t]; b = [x[m] for x in S["per_level_base"]]; k = np.arange(1, len(b) + 1)
    ax.plot(k, b, "-o", color=C_BASE, ms=3, lw=1.2, label="NCM (official)")
    for tag, col, mk, name in (("decode_v3", C_APC, "s", "decode-path $C_\\omega$"), ("loop_v1", C_GREY, "^", "in-loop $C_\\omega$, no selection")):
        if tag in S and S[tag]["per_level_pc42"]:
            ax.plot(k, [x[m] for x in S[tag]["per_level_pc42"]], "--" if tag == "loop_v1" else "-", marker=mk, color=col, ms=3, lw=1.0, label=name)
    ax.set_xlabel("level decoded $k$"); ax.set_ylabel(lab); ax.set_xticks(k[::2] if len(k) > 9 else k); ax.margins(0.04)
np.atleast_1d(axs)[0].legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig("paper/figs/fig_levels.pdf"); fig.savefig("paper/figs/fig_levels.png", dpi=300)

# ---------------- Fig 3: closed-loop dose-response of the in-loop corrector
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for t, col, mk in (("sudoku-easy", "#1b9e77", "o"), ("sudoku-medium", "#7570b3", "s"), ("sudoku-hard", "#e7298a", "^"), ("wmt14", "#d95f02", "D"), ("xsum", "#66a61e", "v")):
    if t not in SUM or "loop_v2" not in SUM[t]:
        continue
    ci = SUM[t]["loop_v2"]["corr_info"]
    sc = [0.0, 0.25, 0.5, 1.0]
    vals = np.array([[ci[s]["calib"][str(x)] for x in sc] for s in ci])
    rel = vals / vals[:, :1].clip(1e-9) * 100
    ax.errorbar(sc, rel.mean(0), yerr=rel.std(0, ddof=1), color=col, marker=mk, ms=4, lw=1.2, capsize=2, label=t)
ax.set_xlabel("step-size scale $s$ (in-loop $C_\\omega$)"); ax.set_ylabel("validation quality\n(% of $s{=}0$)")
ax.legend(frameon=False, fontsize=6.5, loc="lower left"); ax.margins(0.04)
fig.tight_layout(); fig.savefig("paper/figs/fig_dose.pdf"); fig.savefig("paper/figs/fig_dose.png", dpi=300)

# ---------------- Fig 4: halting trade-off (test split, baseline latents)
fig, axs = plt.subplots(1, len(TT), figsize=(7.16, 2.2))
for ax, (t, m, lab) in zip(np.atleast_1d(axs), TT):
    S = SUM[t]; m2 = "exact" if t.startswith("sudoku") else m
    for mode, col, mk in (("flux", C_APC, "o"), ("stab", "#1f78b4", "s")):
        pts = sorted([(v["levels"], v[m2]) for k_, v in S["tau_test_base"].items() if k_.startswith(mode)])
        if pts:
            ax.plot(*zip(*pts), marker=mk, color=col, lw=1.0, ms=3.5, label=f"{mode} rule, $\\tau\\in$[0.01,0.15]")
    ax.plot([S["baseline"]["levels"]], [S["baseline"][m2]], "*", color="k", ms=8, label="full depth")
    ax.set_xlabel("mean levels executed"); ax.set_ylabel(lab.replace("cell acc.", "solved") if t.startswith("sudoku") else lab); ax.margins(0.06)
np.atleast_1d(axs)[0].legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig("paper/figs/fig_halt.pdf"); fig.savefig("paper/figs/fig_halt.png", dpi=300)

# ---------------- Fig 5: relative flux on reference trajectories
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for t, col, mk in (("sudoku-hard", "#e7298a", "^"), ("wmt14", "#d95f02", "D")):
    r = FLUX[t]["rel_flux"]; ax.plot(np.arange(2, len(r) + 2), r, marker=mk, color=col, ms=3.5, lw=1.2, label=t)
for tau in (0.01, 0.05, 0.15):
    ax.axhline(tau, color=C_GREY, lw=0.6, ls=":")
ax.text(2.1, 0.17, "$\\tau$ grid (0.01, 0.05, 0.15)", fontsize=6.5, color=C_GREY)
ax.set_yscale("log"); ax.set_xlabel("level $k$"); ax.set_ylabel("relative flux $\\Delta_k$\n(reference latents)")
ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.margins(0.04)
fig.tight_layout(); fig.savefig("paper/figs/fig_flux.pdf"); fig.savefig("paper/figs/fig_flux.png", dpi=300)
print("figs ok")
