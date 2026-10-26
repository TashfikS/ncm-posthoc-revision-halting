"""v5: correctors trained on external / augmented training data, evaluated on the official test sets.

Train/val: Sudoku = symmetry-augmented boards from official items 0-399 / 400-499; WMT14 = WMT14 train shard /
newstest2013; XSum = XSum train / validation. Test: Sudoku official items 500-1999 (unchanged), WMT14 all 3,003,
XSum first 1,500 (the official evaluation subset used throughout). Variants per seed (42,0,1,2,3):
  loop      : in-loop C_omega, no selection (s = 1)
  loop_sel  : in-loop, step-size scale s in {0,.25,.5,1} selected on val
  proj      : in-loop + codebook projection, s = 1
  proj_sel  : in-loop + projection, s selected on val
  sel_halt  : loop_sel + halting rule selected on val
Selection ties on the primary metric are broken by lower mean validation token cross-entropy of the decoder.
"""
import os, sys, json, time, math, pickle, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apc_lib as A, apc_v4 as V

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--ck", required=True); ap.add_argument("--dd", required=True)
ap.add_argument("--ext", required=True, help="augmented / external split pickle")
ap.add_argument("--out", required=True)
ap.add_argument("--n_train", type=int, default=4000); ap.add_argument("--n_val", type=int, default=1000)
ap.add_argument("--n_test", type=int, default=0, help="0 = full official test (XSum: 1500)")
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--seeds", default="42,0,1,2,3")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "log.txt"), "a")
def log(*s):
    m = " ".join(str(x) for x in s); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

T = A.Task(a.task, a.ck, a.dd); K = len(T.levels)
X = pickle.load(open(a.ext, "rb"))
if a.task.startswith("sudoku"):
    trI, trR, vaI, vaR = X["train_puzzles"], X["train_solutions"], X["val_puzzles"], X["val_solutions"]
    end = 500 + a.n_test if a.n_test else len(T.items)
    teI, teR = T.items[500:end], T.refs[500:end]
else:
    trI, trR, vaI, vaR = X["train_src"], X["train_ref"], X["val_src"], X["val_ref"]
    nt = a.n_test or (1500 if a.task == "xsum" else len(T.items))
    teI, teR = T.items[:nt], T.refs[:nt]
trI, trR = trI[:a.n_train], trR[:a.n_train]; vaI, vaR = vaI[:a.n_val], vaR[:a.n_val]
# concatenate so that apc_lib index-based helpers work: [train | val | test]
T.items = list(trI) + list(vaI) + list(teI); T.refs = list(trR) + list(vaR) + list(teR)
TR = list(range(len(trI))); VA = list(range(len(trI), len(trI) + len(vaI))); TE = list(range(len(trI) + len(vaI), len(T.items)))
PK = {"exact": "exact", "bleu": "bleu", "rouge": "rougeL"}[T.metric]
R = {"args": vars(a), "n": {"train": len(TR), "val": len(VA), "test": len(TE)}, "levels": T.levels,
     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
def dump(): json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
log("splits", R["n"])


@torch.no_grad()
def evaluate(idxs, C=None, project=False, halt=None, items=False):
    hyps, ce, lev, ms = [], [], [], []
    for s in range(0, len(idxs), a.bs):
        b = idxs[s:s + a.bs]
        pf = torch.tensor([T.prefix(T.items[i]) for i in b], device=T.dev)
        if project:
            r = V.generate_v4(T, pf, corrector=C, project=True); z = r["z"]; lv = [K] * len(b); t = r["time"]
        else:
            S = A.apc_generate(T, pf, corrector=C, halt=halt); z = S["z"]; lv = [S["levels_run"]] * len(b); t = S["time"]
        hyps += A.decode_latent(T, z); lev += lv; ms.append(t * 1000)
        y = torch.tensor([T.target_ids(T.refs[i]) for i in b], device=T.dev)
        lg = T.tk.vqvae.decoder(z).float()
        ce.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="mean").item() * len(b))
    refs = [T.ref_str(T.refs[i]) for i in idxs]
    sc, it = A.corpus_scores(T, hyps, refs)
    out = {"scores": sc, "ce": float(np.sum(ce) / len(idxs)), "levels": float(np.mean(lev)), "ms_per_batch": float(np.mean(ms))}
    if items:
        out["items"] = {k: [float(x) for x in v] for k, v in it.items()}
    return out


def better(x, y):  # primary metric, tie -> lower CE
    return (round(x["scores"][PK], 6), -x["ce"]) > (round(y["scores"][PK], 6), -y["ce"])


base_val = evaluate(VA); base_test = evaluate(TE, items=True)
R["base_val"], R["base_test"] = base_val, base_test
log("NCM val", base_val["scores"], "test", base_test["scores"]); dump()

t0 = time.time()
DATA = A.build_train_set(T, TR, a.bs, n_corrupt=2, seed=0)
R["train_set_s"] = time.time() - t0; log("train set", {k: tuple(v.shape) for k, v in DATA.items()})
h = 32
while A.n_params(A.LatentCorrector(T.d, K - 1, hidden=h)) < 1.2e6:
    h += 4
R["hidden"] = h; R["corr_params"] = A.n_params(A.LatentCorrector(T.d, K - 1, hidden=h))

TAUS = [0.01, 0.03, 0.05, 0.10, 0.15]
R["seeds"] = {}
for seed in [int(s) for s in a.seeds.split(",")]:
    C, hist, secs = A.train_corrector(T, DATA, hidden=h, epochs=a.epochs, seed=seed, bs=32, log=log)
    C.eval(); rec = {"train_s": secs, "best_epoch": getattr(C, "best_epoch", None), "hist": hist}
    # selection on val
    for proj in (False, True):
        cal = {}
        for s in (0.0, 0.25, 0.5, 1.0):
            C.eta_scale = s; cal[s] = evaluate(VA, C, project=proj)
        best = 0.0
        for s in (0.25, 0.5, 1.0):
            if better(cal[s], cal[best]):
                best = s
        rec["cal_proj" if proj else "cal_loop"] = {str(k): {"score": v["scores"][PK], "ce": v["ce"]} for k, v in cal.items()}
        rec["s_proj" if proj else "s_loop"] = best
    # halting on val for loop_sel
    C.eta_scale = rec["s_loop"]; nohalt = cal_val = evaluate(VA, C)
    sw = {}
    for mode in ("flux", "stab"):
        for tau in TAUS:
            sw[f"{mode}_{tau}"] = evaluate(VA, C, halt={"mode": mode, "tau": tau, "kmin": 2})
    cand = [(k, v) for k, v in sw.items() if v["scores"][PK] >= nohalt["scores"][PK] - 0.5]
    hsel = min(cand, key=lambda kv: kv[1]["levels"])[0] if cand else None
    rec["halt_sel"] = hsel; rec["halt_val"] = {k: {"score": v["scores"][PK], "levels": v["levels"]} for k, v in sw.items()}
    # test
    C.eta_scale = 1.0; rec["loop"] = evaluate(TE, C, items=True)
    rec["proj"] = evaluate(TE, C, project=True, items=True)
    C.eta_scale = rec["s_loop"]; rec["loop_sel"] = evaluate(TE, C, items=True)
    if hsel:
        md, tu = hsel.split("_"); rec["sel_halt"] = evaluate(TE, C, halt={"mode": md, "tau": float(tu), "kmin": 2}, items=True)
    else:
        rec["sel_halt"] = rec["loop_sel"]
    C.eta_scale = rec["s_proj"]; rec["proj_sel"] = evaluate(TE, C, project=True, items=True)
    R["seeds"][str(seed)] = rec
    log(f"== seed {seed} s_loop {rec['s_loop']} s_proj {rec['s_proj']} halt {hsel} | " +
        " ".join(f"{v} {rec[v]['scores'][PK]:.2f}" for v in ("loop", "loop_sel", "proj", "proj_sel", "sel_halt")) +
        f" | NCM {base_test['scores'][PK]:.2f}")
    if seed == 42:
        torch.save(C.state_dict(), os.path.join(a.out, "corrector_seed42.pt"))
    dump()
R["done"] = True; dump(); log("ALL DONE")
