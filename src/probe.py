"""Learned halting probe baseline.

For every sequence and level k of the official greedy sampler, features = [mean-pooled z^(k), max-pooled z^(k),
level one-hot, decoder margin statistics (min / mean / 10% quantile of p1-p2), fraction of decoded tokens changed
from level k-1]. Label = 1 iff the level-k decoding equals the final-level decoding (whole sequence).
A 2-layer MLP is trained on training-split rollouts; at test time a sequence halts at the first k >= kmin with
probe probability > threshold (threshold chosen on the validation split as the most aggressive value whose quality
is within 0.5 points of full depth). Because halting only truncates the official trajectory, the halted output is
exactly D(z^(k*)) and is computed post hoc from recorded trajectories (equivalent to per-sequence compaction).
"""
import os, sys, json, pickle, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apc_lib as A

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--ck", required=True); ap.add_argument("--dd", required=True)
ap.add_argument("--ext", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_train", type=int, default=3000); ap.add_argument("--n_val", type=int, default=1000)
ap.add_argument("--n_test", type=int, default=0); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--kmin", type=int, default=2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
T = A.Task(a.task, a.ck, a.dd); K = len(T.levels)
X = pickle.load(open(a.ext, "rb"))
if a.task.startswith("sudoku"):
    trI, trR, vaI, vaR = X["train_puzzles"], X["train_solutions"], X["val_puzzles"], X["val_solutions"]
    end = 500 + a.n_test if a.n_test else len(T.items); teI, teR = T.items[500:end], T.refs[500:end]
else:
    trI, trR, vaI, vaR = X["train_src"], X["train_ref"], X["val_src"], X["val_ref"]
    nt = a.n_test or (1500 if a.task == "xsum" else len(T.items)); teI, teR = T.items[:nt], T.refs[:nt]
PK = {"exact": "exact", "bleu": "bleu", "rouge": "rougeL"}[T.metric]


@torch.no_grad()
def collect(items, refs):
    feats, labels, hyps_by_level = [], [], [[] for _ in range(K)]
    for s in range(0, len(items), a.bs):
        pf = torch.tensor([T.prefix(x) for x in items[s:s + a.bs]], device=T.dev)
        S = A.apc_generate(T, pf, record=True)
        zs = S["zs"]; B = pf.shape[0]
        lg = [T.tk.vqvae.decoder(z).float() for z in zs]
        args = [l.argmax(-1) for l in lg]; fin = args[-1]
        f_b = []
        for k, (z, l) in enumerate(zip(zs, lg)):
            p = l.softmax(-1); t2 = p.topk(2, -1).values; mu = t2[..., 0] - t2[..., 1]
            ch = (args[k] != args[k - 1]).float().mean(1) if k else torch.ones(B, device=T.dev)
            oh = torch.zeros(B, K, device=T.dev); oh[:, k] = 1
            f_b.append(torch.cat([z.mean(1), z.amax(1), oh, mu.min(1, keepdim=True).values, mu.mean(1, keepdim=True),
                                  mu.quantile(0.1, dim=1, keepdim=True), ch[:, None]], 1))
            labels.append((args[k] == fin).all(1).float().cpu())
            hyps_by_level[k] += A.decode_latent(T, z)
        feats.append(torch.stack(f_b, 1).cpu())
    F_ = torch.cat(feats)  # N K F
    Y = torch.stack(labels, 0).view(-1, K, B) if False else None
    lab = torch.zeros(F_.shape[0], K)
    # rebuild labels in N,K order
    idx = 0
    for s in range(0, len(items), a.bs):
        B = min(a.bs, len(items) - s)
        for k in range(K):
            lab[s:s + B, k] = labels[idx]; idx += 1
    return F_, lab, hyps_by_level


R = {"task": a.task}
Ftr, Ytr, _ = collect(trI[:a.n_train], trR[:a.n_train])
Fva, Yva, Hva = collect(vaI[:a.n_val], vaR[:a.n_val])
Fte, Yte, Hte = collect(teI, teR)
R["label_rate_by_level_train"] = Ytr.mean(0).tolist(); R["label_rate_by_level_test"] = Yte.mean(0).tolist()
mu_, sd_ = Ftr.reshape(-1, Ftr.shape[-1]).mean(0), Ftr.reshape(-1, Ftr.shape[-1]).std(0) + 1e-6
nrm = lambda x: (x - mu_) / sd_
torch.manual_seed(42)
mlp = nn.Sequential(nn.Linear(Ftr.shape[-1], 128), nn.GELU(), nn.Linear(128, 1))
opt = torch.optim.AdamW(mlp.parameters(), 1e-3, weight_decay=0.01)
Xtr, ytr = nrm(Ftr[:, :-1].reshape(-1, Ftr.shape[-1])), Ytr[:, :-1].reshape(-1)  # exclude final level (label trivially 1)
pos = ytr.mean().clamp(1e-4, 1 - 1e-4)
for ep in range(30):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 512):
        b = perm[i:i + 512]
        loss = F.binary_cross_entropy_with_logits(mlp(Xtr[b]).squeeze(1), ytr[b], pos_weight=(1 - pos) / pos)
        opt.zero_grad(); loss.backward(); opt.step()


def probs(Fx):
    with torch.no_grad():
        return torch.sigmoid(mlp(nrm(Fx.reshape(-1, Fx.shape[-1]))).view(Fx.shape[0], Fx.shape[1]))


def auroc(p, y):
    p, y = p.numpy(), y.numpy()
    if y.min() == y.max(): return None
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def halted(Pp, H, refs, thr):
    N = Pp.shape[0]; kstar = np.full(N, K - 1)
    for i in range(N):
        for k in range(a.kmin - 1, K - 1):
            if Pp[i, k] > thr:
                kstar[i] = k; break
    hyps = [H[kstar[i]][i] for i in range(N)]
    sc, it = A.corpus_scores(T, hyps, [T.ref_str(r) for r in refs])
    return sc, float((kstar + 1).mean()), it


Pva, Pte = probs(Fva), probs(Fte)
R["auroc_test_by_level"] = [auroc(Pte[:, k], Yte[:, k]) for k in range(K - 1)]
R["auroc_test_pooled"] = auroc(Pte[:, :-1].reshape(-1), Yte[:, :-1].reshape(-1))
full_val, _, _ = halted(Pva, Hva, vaR[:a.n_val], 2.0)
sweep = {}
for thr in (0.5, 0.7, 0.9, 0.95, 0.99):
    sc, lv, _ = halted(Pva, Hva, vaR[:a.n_val], thr); sweep[str(thr)] = {"score": sc[PK], "levels": lv}
ok = [t for t, v in sweep.items() if v["score"] >= full_val[PK] - 0.5]
thr = float(min(ok, key=lambda t: sweep[t]["levels"])) if ok else 2.0
R["val_sweep"] = sweep; R["thr"] = thr; R["full_val"] = full_val
sc, lv, it = halted(Pte, Hte, teR, thr); full_te, _, it0 = halted(Pte, Hte, teR, 2.0)
R["test"] = {"scores": sc, "levels": lv}; R["test_full"] = {"scores": full_te, "levels": float(K)}
R["test_sweep"] = {str(t): dict(zip(("scores", "levels"), halted(Pte, Hte, teR, t)[:2])) for t in (0.5, 0.7, 0.9, 0.95, 0.99)}
json.dump(R, open(os.path.join(a.out, "probe.json"), "w"), indent=1, default=float)
print("PROBE", a.task, "label_rate(test,k<K)", [round(x, 3) for x in R["label_rate_by_level_test"][:-1]], "AUROC", R["auroc_test_pooled"],
      "thr", thr, "test", sc[PK], "levels", lv, "full", full_te[PK])
