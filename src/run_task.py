"""Experiment driver for APC-NCM on official HazyResearch/ncm checkpoints (single T4)."""
import os, sys, json, time, math, pickle, subprocess, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apc_lib as A

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--ck", required=True); ap.add_argument("--dd", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_train", type=int, required=True); ap.add_argument("--n_test", type=int, required=True)
ap.add_argument("--bs", type=int, required=True)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--official_n", type=int, default=0)
ap.add_argument("--sens", action="store_true")
ap.add_argument("--corr_mode", default="loop", choices=["loop", "decode"])
ap.add_argument("--skip_sweeps", action="store_true", help="skip baseline-only sweeps (reuse v1)")
ap.add_argument("--zs_full", action="store_true", help="zero-shot halting curve on all items")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SEEDS = [42, 0, 1, 2, 3]
TAUS = [0.01, 0.03, 0.05, 0.10, 0.15]
if os.environ.get("QUICK"):
    SEEDS, TAUS = [42], [0.05]
LOG = open(os.path.join(a.out, "log.txt"), "a")


def log(*s):
    m = " ".join(str(x) for x in s); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


R = {"task": a.task, "args": vars(a), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
     "torch": torch.__version__}


def dump():
    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)


# ------------------------------------------------------------------ 1. official reproduction
if a.official_n >= 0 and not a.skip_sweeps:
    cfgp = os.path.join(A.NCM_ROOT, "config", "eval", f"{a.task}.yaml")
    cmd = [sys.executable, os.path.join(A.NCM_ROOT, "evaluate.py"), "--config", cfgp, "--checkpoint-dir", a.ck,
           "--data-dir", a.dd, "--out", os.path.join(a.out, "official"), "--gpus", "0"]
    if a.official_n > 0:
        cmd += ["--n", str(a.official_n)]
    log("OFFICIAL:", " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=A.NCM_ROOT, capture_output=True, text=True, env={**os.environ, "NCM_ROOT": A.NCM_ROOT})
    R["official"] = {"cmd": " ".join(cmd), "returncode": p.returncode, "stdout": p.stdout[-6000:], "stderr": p.stderr[-3000:],
                     "seconds": time.time() - t0}
    log(p.stdout[-3000:]); log(p.stderr[-1500:]); dump()

# ------------------------------------------------------------------ 2. load
T = A.Task(a.task, a.ck, a.dd)
K = len(T.levels)
tr = list(range(a.n_train)); te = list(range(a.n_train, a.n_train + a.n_test))
R["levels"] = T.levels; R["d_vq"] = T.d
R["gen_params"] = A.n_params(T.net); R["tok_params"] = A.n_params(T.tk.vqvae)
log("levels", T.levels, "d", T.d, "gen params", R["gen_params"])


def prim(sc):
    return {"exact": "exact", "bleu": "bleu", "rouge": "rougeL"}[T.metric]


def evaluate(idxs, C=None, halt=None, freeze=None, record=False, keep_hyps=False):
    hyps, levels_w, times, frozen = [], [], [], []
    per_level = [[] for _ in range(K)] if record else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    refs = [T.ref_str(T.refs[i]) for i in idxs]
    first = True
    for s in range(0, len(idxs), a.bs):
        b = idxs[s:s + a.bs]
        pf = torch.tensor([T.prefix(T.items[i]) for i in b], device=T.dev)
        if first:  # warm-up (not timed)
            A.apc_generate(T, pf, corrector=C, halt=halt, freeze_tau=freeze, corr_mode=a.corr_mode); first = False
        S = A.apc_generate(T, pf, corrector=C, halt=halt, freeze_tau=freeze, record=record, corr_mode=a.corr_mode)
        hyps += A.decode_latent(T, S["z"])
        levels_w += [S["levels_run"]] * len(b)
        times.append(S["time"] * 1000)
        if S["frozen_frac"]:
            frozen.append(S["frozen_frac"][-1])
        if record:
            for k, zk in enumerate(S["zs"]):
                per_level[k] += A.decode_latent(T, zk)
    cs, items = A.corpus_scores(T, hyps, refs)
    out = {"scores": cs, "levels": float(np.mean(levels_w)), "ms_per_batch": float(np.mean(times)),
           "ms_per_batch_std": float(np.std(times)), "total_s": float(np.sum(times) / 1000),
           "peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0,
           "frozen_frac": float(np.mean(frozen)) if frozen else 0.0,
           "items": {k: v.tolist() for k, v in items.items()}}
    if record:
        out["per_level"] = [A.corpus_scores(T, h, refs)[0] for h in per_level]
        out["per_level_hyps"] = [h[:12] for h in per_level]
    if keep_hyps:
        out["hyps"] = hyps
    return out


def short(r):
    return {k: v for k, v in r.items() if k not in ("items", "per_level_hyps", "hyps")}


# ------------------------------------------------------------------ 3. FLOPs per level count
from torch.utils.flop_counter import FlopCounterMode


def flops_by_levels(C=None):
    pf = torch.tensor([T.prefix(T.items[i]) for i in te[:a.bs]], device=T.dev)
    res = {}
    for k in range(2, K + 1):
        halt = None if k == K else {"mode": "flux", "tau": float("inf"), "kmin": k - 1}
        fc = FlopCounterMode(display=False)
        with fc:
            S = A.apc_generate(T, pf, corrector=C, halt=halt, corr_mode=a.corr_mode)
            A.decode_latent(T, S["z"])
        assert S["levels_run"] == k, (S["levels_run"], k)
        res[k] = fc.get_total_flops() / len(pf)
    return res


def flops_of(run, table):
    # items were halted batch-wise; levels_run per item recorded through mean -> use per-batch exact levels
    return run["levels_hist_flops"] if "levels_hist_flops" in run else None


# ------------------------------------------------------------------ 4. baseline + zero-shot halting
log("== baseline (test)")
base = evaluate(te, record=True, keep_hyps=True); R["baseline"] = base; log(short(base)); dump()
FB = flops_by_levels(); R["flops_base"] = FB; log("flops/sample by levels", FB); dump()

log("== tau sweep on TRAIN split (baseline, flux & stability)")
R["tau_train"] = {}
for mode in ("flux", "stab"):
    for tau in (TAUS if not a.skip_sweeps else []):
        r = evaluate(tr, halt={"mode": mode, "tau": tau, "kmin": 2})
        R["tau_train"][f"{mode}_{tau}"] = short(r); log(mode, tau, short(r)["scores"], r["levels"])
rb_tr = evaluate(tr); R["baseline_train"] = short(rb_tr)
pk = prim(None)


def pick_tau(sweep, base_score, tol):
    best = None
    for key, r in sweep.items():
        if r["scores"][pk] >= base_score - tol and (best is None or r["levels"] < best[1]["levels"]):
            best = (key, r)
    if best is None:  # nothing within tolerance: most accurate setting, ties -> fewer levels
        best = max(sweep.items(), key=lambda kv: (kv[1]["scores"][pk], -kv[1]["levels"]))
    return best


TOL = {"exact": 0.5, "bleu": 0.5, "rouge": 0.5}[T.metric]
sel = pick_tau(R["tau_train"], rb_tr["scores"][pk], TOL) if R["tau_train"] else ("flux_0.01", None)
mode_s, tau_s = sel[0].split("_")[0], float(sel[0].split("_")[1])
HALT = {"mode": mode_s, "tau": tau_s, "kmin": 2}
R["halt_selected"] = HALT; log("selected halting", HALT); dump()

log("== halting-only (test)")
ho = evaluate(te, halt=HALT, keep_hyps=True); R["halt_only"] = ho; log(short(ho)); dump()

log("== test tau sweep, baseline")
R["tau_test_base"] = {}
for mode in ("flux", "stab"):
    for tau in (TAUS if not a.skip_sweeps else []):
        R["tau_test_base"][f"{mode}_{tau}"] = short(evaluate(te, halt={"mode": mode, "tau": tau, "kmin": 2}))
dump()
if a.zs_full and not a.skip_sweeps:
    allidx = list(range(len(T.items)))
    R["zs_full"] = {"baseline": short(evaluate(allidx))}
    for tau in TAUS:
        R["zs_full"][f"flux_{tau}"] = short(evaluate(allidx, halt={"mode": "flux", "tau": tau, "kmin": 2}))
        R["zs_full"][f"stab_{tau}"] = short(evaluate(allidx, halt={"mode": "stab", "tau": tau, "kmin": 2}))
    dump()

# ------------------------------------------------------------------ 5. corrector training data
log("== building corrector training set from TRAIN split rollouts + code corruption")
t0 = time.time()
DATA = A.build_train_set(T, tr, a.bs, n_corrupt=2, seed=0, all_levels=(a.corr_mode == "decode"))
NLC = DATA["zg"].shape[1]
R["train_set_build_s"] = time.time() - t0
log("train set", {k: tuple(v.shape) for k, v in DATA.items()}, f"{time.time() - t0:.1f}s"); dump()


def hidden_for(target):
    h = 32
    while A.n_params(A.LatentCorrector(T.d, NLC, hidden=h)) < target:
        h += 4
    return h


H12 = hidden_for(1.2e6)
R["hidden_1.2M"] = H12; R["corr_params_1.2M"] = A.n_params(A.LatentCorrector(T.d, NLC, hidden=H12))


def train(seed, hidden=H12, eta_init=0.5, learn_eta=True, calibrate=True):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    C, hist, secs = A.train_corrector(T, DATA, hidden=hidden, epochs=a.epochs, seed=seed, eta_init=eta_init,
                                      learn_eta=learn_eta, log=log)
    C.eval()
    val_items = [tr[i] for i in C.val_idx]
    cal = {}
    for sc in ((0.0, 0.25, 0.5, 1.0) if calibrate else (1.0,)):
        C.eta_scale = sc
        cal[sc] = evaluate(val_items, C=C)["scores"][pk]
    best_sc = max(cal, key=lambda x: (round(cal[x], 6), -x))
    C.eta_scale = best_sc
    if a.corr_mode == "decode" and calibrate:  # decode path: levels are independent -> per-level scale
        pl = {}
        for sc in (0.0, 0.25, 0.5, 1.0):
            C.eta_scale = sc
            ck_ = "cell" if T.metric == "exact" else pk
            pl[sc] = [x[ck_] for x in evaluate(val_items, C=C, record=True)["per_level"]]
        vec = [max(pl, key=lambda x: (round(pl[x][k], 6), -x)) for k in range(K)]
        vec = vec[:NLC] + [0.0] * (NLC - len(vec))
        C.eta_scale = torch.tensor(vec)
        best_sc = vec
        log(f"  per-level eta scale (decode path): {vec}")
    log(f"  closed-loop eta-scale calibration on {len(val_items)} val items: {cal} -> {best_sc}")
    info = {"hist": hist, "best_epoch": C.best_epoch, "calib": {str(k): v for k, v in cal.items()}, "eta_scale": best_sc, "train_s": secs, "params": A.n_params(C),
            "train_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0,
            "eta": C.eta(torch.arange(NLC, device=C.eta_logit.device)).tolist()}
    return C, info


R["pc"], R["full"], R["corr_info"] = {}, {}, {}
for seed in SEEDS:
    log(f"== seed {seed}")
    C, info = train(seed)
    R["corr_info"][str(seed)] = info
    if seed == 42:
        torch.save(C.state_dict(), os.path.join(a.out, "corrector_seed42.pt"))
        R["flops_pc"] = flops_by_levels(C)
        # re-select tau for full model on TRAIN split
        sw = {}
        for mode in ("flux", "stab"):
            for tau in TAUS:
                sw[f"{mode}_{tau}"] = short(evaluate(tr, C=C, halt={"mode": mode, "tau": tau, "kmin": 2}))
        R["tau_train_pc"] = sw
        pc_tr = short(evaluate(tr, C=C)); R["pc_train"] = pc_tr
        s2 = pick_tau(sw, max(pc_tr["scores"][pk], rb_tr["scores"][pk]) if False else pc_tr["scores"][pk], TOL)
        HALT_PC = {"mode": s2[0].split("_")[0], "tau": float(s2[0].split("_")[1]), "kmin": 2}
        R["halt_selected_pc"] = HALT_PC; log("selected halting (APC)", HALT_PC)
    rec = seed == 42
    r = evaluate(te, C=C, record=rec, keep_hyps=rec); R["pc"][str(seed)] = r if rec else {**short(r), "items": r["items"]}
    log("PC", seed, short(r)["scores"], r["levels"])
    f = evaluate(te, C=C, halt=HALT_PC, keep_hyps=rec); R["full"][str(seed)] = f if rec else {**short(f), "items": f["items"]}
    log("FULL", seed, short(f)["scores"], f["levels"])
    if seed == 42:
        R["tau_test_pc"] = {}
        for mode in ("flux", "stab"):
            for tau in TAUS:
                R["tau_test_pc"][f"{mode}_{tau}"] = short(evaluate(te, C=C, halt={"mode": mode, "tau": tau, "kmin": 2}))
        R["freeze"] = {}
        for ft in ((0.01,) if os.environ.get("QUICK") else (0.005, 0.01, 0.02, 0.05)):
            R["freeze"][str(ft)] = short(evaluate(te, C=C, freeze=ft))
            R["freeze"][f"base_{ft}"] = short(evaluate(te, freeze=ft))
        C42 = C
    dump()

# ------------------------------------------------------------------ 6. sensitivity (eta, size)
if a.sens:
    R["sens_eta"], R["sens_size"] = {}, {}
    for eta in (0.25, 0.5, 1.0):
        C, info = train(42, eta_init=min(eta, 0.999), learn_eta=False, calibrate=False)
        r = short(evaluate(te, C=C)); r.update({k: info[k] for k in ("train_s", "params", "train_peak_mb")})
        R["sens_eta"][str(eta)] = r; log("eta", eta, r["scores"]); dump()
    for tgt in (0.3e6, 1.2e6, 3.0e6):
        h = hidden_for(tgt)
        C, info = train(42, hidden=h, calibrate=False)
        r = short(evaluate(te, C=C)); r.update({k: info[k] for k in ("train_s", "params", "train_peak_mb")}); r["hidden"] = h
        R["sens_size"][str(tgt)] = r; log("size", tgt, h, r["scores"]); dump()

R["done"] = True
dump()
log("ALL DONE")
