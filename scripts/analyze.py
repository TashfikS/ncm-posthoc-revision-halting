"""Aggregate Kaggle T4 runs into tables (JSON + LaTeX fragments) and figures."""
import json, os, glob, numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = "kout"
TASKS = ["sudoku-easy", "sudoku-medium", "sudoku-hard", "wmt14", "xsum"]
SEEDS = ["42", "0", "1", "2", "3"]
FAM = {"sudoku-easy": "sudoku", "sudoku-medium": "sudoku", "sudoku-hard": "sudoku", "wmt14": "wmt14", "xsum": "xsum"}
PRIM = {"sudoku": ("exact", "cell"), "wmt14": ("bleu", "sbleu"), "xsum": ("rougeL", "rougeL")}


def load(task, ver):
    p = f"{K}/{FAM[task]}_{ver}/results/{task}/results.json"
    return json.load(open(p)) if os.path.exists(p) else None


def item_key(task):
    return PRIM[FAM[task]][1]


def mcnemar(a, b):
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def wilcox(a, b):
    d = np.asarray(b) - np.asarray(a)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def seed_block(R, name, task, base):
    if R is None or name not in R or not R[name]:
        return None
    f, ik = FAM[task], item_key(task)
    rows = [R[name][s] for s in SEEDS if s in R[name]]
    out = {"n_seeds": len(rows)}
    for m in rows[0]["scores"]:
        v = np.array([r["scores"][m] for r in rows])
        out[m] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        bv = base["scores"][m]
        out[m + "_t_p"] = float(stats.ttest_1samp(v, bv).pvalue) if len(v) > 1 and v.std() > 0 else (1.0 if np.allclose(v, bv) else 0.0)
    out["levels"] = (float(np.mean([r["levels"] for r in rows])), float(np.std([r["levels"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0)
    out["ms"] = (float(np.mean([r["ms_per_batch"] for r in rows])), float(np.std([r["ms_per_batch"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0)
    out["peak_mb"] = float(np.max([r["peak_mb"] for r in rows]))
    ps = [wilcox(base["items"][ik], r["items"][ik]) for r in rows]
    out["wilcoxon_p_max"] = float(np.max(ps)); out["wilcoxon_p_min"] = float(np.min(ps))
    if f == "sudoku":
        pm = [mcnemar(base["items"]["exact"], r["items"]["exact"]) for r in rows]
        out["mcnemar_p_max"] = float(np.max(pm)); out["mcnemar_p_min"] = float(np.min(pm))
    return out


def flops_for(R, levels, key="flops_base"):
    fb = {int(k): v for k, v in R[key].items()}
    lv = int(round(levels))
    return fb.get(lv, fb[max(fb)])


SUM = {}
for task in TASKS:
    v1, v2, v3 = load(task, "v1"), load(task, "v2"), load(task, "v3")
    if v1 is None:
        continue
    base = v1["baseline"]
    S = {"official_stdout": v1.get("official", {}).get("stdout", "")[-700:], "levels": v1["levels"],
         "gen_params": v1["gen_params"], "tok_params": v1["tok_params"], "n_test": len(base["items"][item_key(task)]),
         "baseline": {**base["scores"], "levels": base["levels"], "ms": base["ms_per_batch"], "ms_std": base["ms_per_batch_std"],
                      "peak_mb": base["peak_mb"], "flops": flops_for(v1, base["levels"])},
         "halt_sel": v1["halt_selected"],
         "halt_only": {**v1["halt_only"]["scores"], "levels": v1["halt_only"]["levels"], "ms": v1["halt_only"]["ms_per_batch"],
                       "peak_mb": v1["halt_only"]["peak_mb"], "flops": flops_for(v1, v1["halt_only"]["levels"]),
                       "wilcoxon_p": wilcox(base["items"][item_key(task)], v1["halt_only"]["items"][item_key(task)])},
         "per_level_base": base["per_level"], "flops_base": v1["flops_base"],
         "tau_test_base": {k: {**r["scores"], "levels": r["levels"]} for k, r in v1["tau_test_base"].items()},
         "freeze_base": {k: {**r["scores"], "frozen": r["frozen_frac"]} for k, r in v1.get("freeze", {}).items() if k.startswith("base_")},
         "zs_full": {k: {**r["scores"], "levels": r["levels"]} for k, r in v1.get("zs_full", {}).items()}}
    for tag, R in (("loop_v1", v1), ("loop_v2", v2), ("decode_v3", v3)):
        if R is None:
            continue
        S[tag] = {"pc": seed_block(R, "pc", task, base), "full": seed_block(R, "full", task, base),
                  "halt_sel": R.get("halt_selected_pc"), "corr_params": R.get("corr_params_1.2M"),
                  "flops_pc": R.get("flops_pc"),
                  "per_level_pc42": R["pc"]["42"].get("per_level") if "42" in R.get("pc", {}) else None,
                  "corr_info": {s: {k: v for k, v in i.items() if k != "hist"} for s, i in R.get("corr_info", {}).items()},
                  "hist42": R.get("corr_info", {}).get("42", {}).get("hist"),
                  "tau_test_pc": {k: {**r["scores"], "levels": r["levels"]} for k, r in R.get("tau_test_pc", {}).items()},
                  "freeze_pc": {k: {**r["scores"], "frozen": r["frozen_frac"]} for k, r in R.get("freeze", {}).items() if not k.startswith("base_")},
                  "sens_eta": R.get("sens_eta"), "sens_size": R.get("sens_size")}
        if S[tag]["full"] and R.get("flops_pc"):
            S[tag]["full"]["flops"] = flops_for(R, S[tag]["full"]["levels"][0], "flops_pc")
            S[tag]["pc"]["flops"] = flops_for(R, S[tag]["pc"]["levels"][0], "flops_pc")
    SUM[task] = S
json.dump(SUM, open("summary.json", "w"), indent=1, default=float)
print("tasks summarised:", list(SUM))
