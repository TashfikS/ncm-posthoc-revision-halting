"""DAgger-style closed-loop co-adaptation of the NCM generator (and optionally C_omega) on Sudoku.

Arms (all start from the official generator.pt, all use only the first N_tr items of the official test file):
  tf      : fine-tune G on teacher-forced reference states  (control: extra data, no closed-loop states)
  dagger  : fine-tune G on a 50/50 mix of reference states and its own greedy rollout states,
            with the DAgger expert label  r_{k+1} = Q_{k+1}(z - state)  (greedy residual quantisation)
  dagger_c: as dagger, but states are revised by C_omega (in-loop) and (G, C_omega) are trained jointly
            with CE + L_APC (MSE part); C_omega is used in-loop at test time.
Rollouts are re-collected with the current model at the start of every round. Checkpoint selection:
best closed-loop solve rate on the 20% validation part of N_tr (round 0 = official model included).
"""
import os, sys, json, time, copy, argparse, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apc_lib as A, apc_v4 as V

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--ck", required=True); ap.add_argument("--dd", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_train", type=int, default=500); ap.add_argument("--n_test", type=int, default=1500)
ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--ebs", type=int, default=250)
ap.add_argument("--arms", default="tf,dagger,dagger_c"); ap.add_argument("--seeds", default="42,0,1,2,3")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "log.txt"), "a")
def log(*s):
    m = " ".join(str(x) for x in s); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

T = A.Task(a.task, a.ck, a.dd)
DEV = T.dev; K = len(T.levels); net = T.net
BASE_SD = copy.deepcopy(net.state_dict())
TRAIN_PARAMS = [n for n, p in net.named_parameters() if n.startswith(("transformer.blocks", "transformer.ln_f", "lm_heads"))]
log("trainable generator tensors", len(TRAIN_PARAMS), "params", sum(dict(net.named_parameters())[n].numel() for n in TRAIN_PARAMS))
ntr = int(0.8 * a.n_train)
TR = list(range(ntr)); VA = list(range(ntr, a.n_train)); TE = list(range(a.n_train, a.n_train + a.n_test))
R = {"args": vars(a), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "levels": T.levels}
def dump(): json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)


@torch.no_grad()
def evaluate(idxs, C=None, rec_items=False):
    hyps, ms = [], []
    for s in range(0, len(idxs), a.ebs):
        b = idxs[s:s + a.ebs]
        pf = torch.tensor([T.prefix(T.items[i]) for i in b], device=DEV)
        r = V.generate_v4(T, pf, corrector=C)
        hyps += A.decode_latent(T, r["z"]); ms.append(r["time"] * 1000)
    refs = [T.ref_str(T.refs[i]) for i in idxs]
    sc, items = A.corpus_scores(T, hyps, refs)
    out = {"scores": sc, "ms_per_batch": float(np.mean(ms))}
    if rec_items:
        out["items"] = {k: [float(x) for x in v] for k, v in items.items()}
    return out


@torch.no_grad()
def ref_data(idxs):
    y = torch.tensor([T.target_ids(T.refs[i]) for i in idxs], device=DEV)
    z = T.tk.vqvae.encode_z(y).float()
    idx, zh = T.vq.z_to_idx(z, return_zhats=True)
    return z, idx, torch.stack(zh[:-1], 1)  # z (B,L,d), codes (B,sum l), states after levels 0..K-2


@torch.no_grad()
def expert_codes(z, state, li_next):
    """Greedy residual quantisation of (z - state) at level li_next (the DAgger expert)."""
    r = z - state
    if li_next != K - 1:
        flat = T.vq._down_for_quant(r, li_next, T.levels[li_next]).reshape(-1, r.shape[-1])
    else:
        flat = r.reshape(-1, r.shape[-1])
    return torch.argmin(T.vq._distances(flat, T.vq.embeddings[li_next].weight), 1).view(r.shape[0], -1)


@torch.no_grad()
def rollout_states(pf, C=None):
    r = V.generate_v4(T, pf, corrector=C, record="pre")
    return torch.stack([zz for _, zz in r["zs"][:K - 1]], 1)  # (B,K-1,L,d), all sequences active


def level_loss(pf, states, z, C=None, ctx=None, backward=True):
    """states: (B,K-1,L,d) input states after levels 0..K-2 (pre-correction). Returns CE (+ mse on C)."""
    st = V.capture(T, pf)
    prefix_emb, pbias, lpe, acc = st["prefix_emb"], st["pbias"], st["lpe"], st["acc"]
    ce, mse, n = 0.0, 0.0, 0
    for li in range(K - 1):
        s = states[:, li]
        if C is not None:
            kk = torch.full((s.shape[0],), li, device=DEV, dtype=torch.long)
            s = C.correct(s, ctx, kk, None)
        tgt = expert_codes(z, s.detach(), li + 1)
        x = F.interpolate(s.transpose(-1, -2), size=(T.levels[li + 1],), mode="area").transpose(-1, -2)
        x = net.transformer.wte(x) + lpe[:, acc:acc + T.levels[li + 1], :]
        logits = net._decode_level_logits(x, prefix_emb, li, pbias, hidx=li + (1 - net._lvl_first))
        l_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)) / (K - 1)
        l = l_ce + (((s - REFST[:, li]).pow(2).mean() / LAM[li]) / (K - 1) if C is not None else 0.0)
        if backward:
            l.backward()  # per-level backward: levels are independent given their input states
        ce = ce + float(l_ce); mse = mse + (float(l) - float(l_ce))
        acc += T.levels[li + 1]; n += 1
    return ce, mse


PF_TR = torch.tensor([T.prefix(T.items[i]) for i in TR], device=DEV)
Z_TR, IDX_TR, ST_TR = ref_data(TR)
LAM_ALL = ST_TR.float().pow(2).mean(dim=(0, 2, 3)).clamp_min(1e-6)

base_val = evaluate(VA)["scores"]; R["base_val"] = base_val
base_test = evaluate(TE, rec_items=True); R["base_test"] = base_test
log("NCM val", base_val, "test", base_test["scores"]); dump()

R["arms"] = {}
for arm in a.arms.split(","):
    R["arms"][arm] = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        A.set_seed(seed); net.load_state_dict(BASE_SD)
        for n_, p in net.named_parameters():
            p.requires_grad_(n_ in TRAIN_PARAMS)
        C = None; params = [p for n_, p in net.named_parameters() if n_ in TRAIN_PARAMS]
        if arm == "dagger_c":
            hid = 316
            C = A.LatentCorrector(T.d, K - 1, hidden=hid).to(DEV); params = params + list(C.parameters())
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        best = (base_val["exact"], -1, copy.deepcopy(net.state_dict()), None); hist = []
        t0 = time.time()
        for rnd in range(a.rounds):
            net.eval()
            if C is not None: C.eval()
            if arm == "tf":
                states = ST_TR
            else:
                roll = torch.cat([rollout_states(PF_TR[s:s + a.ebs], C) for s in range(0, len(TR), a.ebs)])
                states = torch.cat([ST_TR, roll])  # 50/50 reference + on-policy states
            Z = Z_TR if arm == "tf" else torch.cat([Z_TR, Z_TR]); PF = PF_TR if arm == "tf" else torch.cat([PF_TR, PF_TR])
            REFS_ = ST_TR if arm == "tf" else torch.cat([ST_TR, ST_TR])
            N = states.shape[0]
            for ep in range(a.epochs):
                net.train()
                if C is not None: C.train()
                perm = torch.randperm(N, device=DEV); tl = []
                for s in range(0, N, a.bs):
                    b = perm[s:s + a.bs]
                    REFST = REFS_[b]; LAM = LAM_ALL
                    ctx = T.prefix_ctx(PF[b]) if C is not None else None
                    opt.zero_grad(set_to_none=True)
                    ce, mse = level_loss(PF[b], states[b], Z[b], C, ctx)
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tl.append(ce + mse)
                net.eval()
                if C is not None: C.eval()
                v = evaluate(VA, C)["scores"]
                hist.append({"round": rnd, "epoch": ep, "loss": float(np.mean(tl)), "val": v})
                log(f"[{arm} s{seed}] r{rnd} ep{ep} loss {np.mean(tl):.4f} val {v}")
                if v["exact"] > best[0] or (v["exact"] == best[0] and v["cell"] > (best[3] or {}).get("cell", base_val["cell"])):
                    best = (v["exact"], (rnd, ep), copy.deepcopy(net.state_dict()), v, copy.deepcopy(C.state_dict()) if C is not None else None)
        net.load_state_dict(best[2])
        if C is not None and len(best) > 4 and best[4] is not None:
            C.load_state_dict(best[4])
        elif C is not None and best[1] == -1:
            C = None  # official model selected: no corrector
        net.eval()
        te = evaluate(TE, C, rec_items=True)
        R["arms"][arm][str(seed)] = {"best": best[1], "val_best": best[3], "hist": hist, "train_s": time.time() - t0,
                                     "test": te, "peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0}
        log(f"== {arm} seed {seed} best {best[1]} TEST {te['scores']} (NCM {base_test['scores']})"); dump()
net.load_state_dict(BASE_SD)
R["done"] = True; dump(); log("ALL DONE")
