"""Pilot of prescription (ii): continued Stage-2 training with continuous latent perturbation (Sudoku-hard).

For each sigma in --sigmas: start from the official generator, fine-tune all generator parameters on symmetry-
augmented training boards. Input states per level k are the reference cumulative latents zGT^(k) (50%) or
NCM-style code-corrupted latents (50%), plus isotropic Gaussian noise sigma * rms_k * N(0, I) (rms_k = per-level RMS
of zGT^(k)). Labels = greedy residual quantisation of (z - state) at level k+1 (the expert for that state).
Checkpoint selection: val solve rate, ties -> lower val CE; the official model is a candidate.
Then for the selected perturbation-trained generator: (a) its own test baseline, (b) one-step off-manifold
sensitivity L_G (same protocol as theory_consts.py), (c) C_omega retrained against it (3 seeds) and evaluated in
loop at s in {0.25, 0.5, 1}, val-selected s, and with projection.
"""
import os, sys, json, time, copy, pickle, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apc_lib as A, apc_v4 as V

ap = argparse.ArgumentParser()
ap.add_argument("--ck", required=True); ap.add_argument("--dd", required=True); ap.add_argument("--ext", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--sigmas", default="0.05,0.1,0.2")
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--per_epoch", type=int, default=2000)
ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--ebs", type=int, default=250)
ap.add_argument("--n_val", type=int, default=1200); ap.add_argument("--n_test", type=int, default=0)
ap.add_argument("--corr_train", type=int, default=3000); ap.add_argument("--corr_epochs", type=int, default=10)
ap.add_argument("--corr_seeds", default="42,0,1")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "log.txt"), "a")
def log(*s):
    m = " ".join(str(x) for x in s); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

T = A.Task("sudoku-hard", a.ck, a.dd); DEV = T.dev; K = len(T.levels); net, vq = T.net, T.vq
X = pickle.load(open(a.ext, "rb"))
end = 500 + a.n_test if a.n_test else len(T.items)
te_i, te_r = T.items[500:end], T.refs[500:end]
T.items = X["train_puzzles"] + X["val_puzzles"][:a.n_val] + te_i; T.refs = X["train_solutions"] + X["val_solutions"][:a.n_val] + te_r
ntr = len(X["train_puzzles"]); TR = list(range(ntr)); VA = list(range(ntr, ntr + a.n_val)); TE = list(range(ntr + a.n_val, len(T.items)))
BASE = copy.deepcopy(net.state_dict())
R = {"args": vars(a), "n": {"train": ntr, "val": len(VA), "test": len(TE)}}
def dump(): json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)


@torch.no_grad()
def evaluate(idxs, C=None, project=False, items=False):
    hyps, ces = [], []
    for s in range(0, len(idxs), a.ebs):
        b = idxs[s:s + a.ebs]; pf = torch.tensor([T.prefix(T.items[i]) for i in b], device=DEV)
        r = V.generate_v4(T, pf, corrector=C, project=project); hyps += A.decode_latent(T, r["z"])
        y = torch.tensor([T.target_ids(T.refs[i]) for i in b], device=DEV); lg = T.tk.vqvae.decoder(r["z"]).float()
        ces.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1)).item() * len(b))
    sc, it = A.corpus_scores(T, hyps, [T.ref_str(T.refs[i]) for i in idxs]); sc = dict(sc); sc["ce"] = float(np.sum(ces) / len(idxs))
    return {"scores": sc, **({"items": {k: [float(x) for x in v] for k, v in it.items()}} if items else {})}


def better(x, y):
    return (x["exact"], -x["ce"]) > (y["exact"], -y["ce"])


@torch.no_grad()
def ref_states(idxs, gen):
    y = torch.tensor([T.target_ids(T.refs[i]) for i in idxs], device=DEV)
    z = T.tk.vqvae.encode_z(y).float(); idx, zh = vq.z_to_idx(z, return_zhats=True)
    st = torch.stack(zh[:-1], 1)
    cor = A.corrupted_trajectory(T, idx, gen).float()
    return z, st, cor


def expert(z, state, li):
    r = z - state
    flat = vq._down_for_quant(r, li, T.levels[li]).reshape(-1, T.d) if li != K - 1 else r.reshape(-1, T.d)
    return torch.argmin(vq._distances(flat, vq.embeddings[li].weight), 1).view(r.shape[0], -1)


def train_step(pf, z, states, opt, params):
    st = V.capture(T, pf); pe, pb, lpe, acc = st["prefix_emb"], st["pbias"], st["lpe"], st["acc"]
    opt.zero_grad(set_to_none=True); tot = 0.0
    for li in range(K - 1):
        s = states[:, li]
        with torch.no_grad():
            tgt = expert(z, s, li + 1)
        x = F.interpolate(s.transpose(-1, -2), size=(T.levels[li + 1],), mode="area").transpose(-1, -2)
        x = net.transformer.wte(x) + lpe[:, acc:acc + T.levels[li + 1], :]
        lg = net._decode_level_logits(x, pe, li, pb, hidx=li + (1 - net._lvl_first))
        l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tgt.reshape(-1)) / (K - 1)
        l.backward(); tot += l.item(); acc += T.levels[li + 1]
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); return tot


@torch.no_grad()
def lg_proxy(C):
    """Within-level slope of next-level code-error norm vs lattice distance (64 val boards)."""
    ii = VA[:64]; pf = torch.tensor([T.prefix(T.items[i]) for i in ii], device=DEV)
    y = torch.tensor([T.target_ids(T.refs[i]) for i in ii], device=DEV); z = T.tk.vqvae.encode_z(y).float()
    r = V.generate_v4(T, pf, record=True); zs = [q for _, q in r["zs"]]
    st = V.capture(T, pf); pe, pb, lpe = st["prefix_emb"], st["pbias"], st["lpe"]; accs = [st["acc"]]
    for li in range(K - 1): accs.append(accs[-1] + T.levels[li + 1])
    ctx = T.prefix_ctx(pf); pts = []; old = C.eta_scale
    for sc in (0.0, 0.25, 0.5, 1.0):
        C.eta_scale = sc
        for li in range(K - 2):
            prev = zs[li - 1] if li > 0 else torch.zeros_like(zs[li])
            zt = C.correct(zs[li], ctx, torch.full((len(ii),), li, device=DEV, dtype=torch.long), None)
            zp, _ = V.project_level(T, prev, zt, li); d = (zt - zp).flatten(1).norm(dim=1)
            x = F.interpolate(zt.transpose(-1, -2), size=(T.levels[li + 1],), mode="area").transpose(-1, -2)
            x = net.transformer.wte(x) + lpe[:, accs[li]:accs[li] + T.levels[li + 1], :]
            rh = net._decode_level_logits(x, pe, li, pb, hidx=li + (1 - net._lvl_first)).argmax(-1)
            rs = expert(z, zt, li + 1)
            def emb(q):
                e = vq.embeddings[li + 1](q); e = vq._up_to_full(e, li + 1, T.L) if li + 1 != K - 1 else e
                return vq._apply_quant_resi(e, li + 1)
            g = (emb(rh) - emb(rs)).flatten(1).norm(dim=1)
            pts += [(sc, li, u, v) for u, v in zip(d.tolist(), g.tolist())]
    C.eta_scale = old; P = np.array(pts)
    sl = [float(np.polyfit(P[P[:, 1] == li][:, 2], P[P[:, 1] == li][:, 3], 1)[0]) for li in range(K - 2) if P[P[:, 1] == li][:, 2].std() > 1e-6]
    return {"LG_median": float(np.median(sl)), "sigma_by_scale": {str(s): float(P[P[:, 0] == s][:, 3].mean()) for s in (0.0, .25, .5, 1.0)},
            "dist_by_scale": {str(s): float(P[P[:, 0] == s][:, 2].mean()) for s in (0.0, .25, .5, 1.0)}}


def corrector_eval(tag):
    out = {"base_test": evaluate(TE, items=True), "base_val": evaluate(VA)}
    h = 32
    while A.n_params(A.LatentCorrector(T.d, K - 1, hidden=h)) < 1.2e6: h += 4
    DATA = A.build_train_set(T, TR[:a.corr_train], 64, n_corrupt=2, seed=0)
    out["seeds"] = {}
    for seed in [int(s) for s in a.corr_seeds.split(",")]:
        C, hist, secs = A.train_corrector(T, DATA, hidden=h, epochs=a.corr_epochs, seed=seed, bs=32, log=log); C.eval()
        rec = {"train_s": secs}
        cal = {}
        for s in (0.0, 0.25, 0.5, 1.0):
            C.eta_scale = s; cal[s] = evaluate(VA, C)["scores"]
        best = 0.0
        for s in (0.25, 0.5, 1.0):
            if better(cal[s], cal[best]): best = s
        rec["cal"] = {str(k): v for k, v in cal.items()}; rec["s_sel"] = best
        for s in (0.25, 0.5, 1.0):
            C.eta_scale = s; rec[f"loop_{s}"] = evaluate(TE, C, items=True)
        C.eta_scale = best; rec["loop_sel"] = evaluate(TE, C, items=True)
        C.eta_scale = 1.0; rec["proj"] = evaluate(TE, C, project=True, items=True)
        if seed == 42:
            rec["LG"] = lg_proxy(C)
        out["seeds"][str(seed)] = rec
        log(f"[{tag} C s{seed}] s_sel {best} " + " ".join(f"{k} {rec[k]['scores']['exact']:.2f}" for k in ("loop_0.25", "loop_0.5", "loop_1.0", "loop_sel", "proj")) +
            f" | gen-base {out['base_test']['scores']['exact']:.2f}")
        dump()
    return out


# ---- reference: official generator
net.load_state_dict(BASE); net.eval()
R["official"] = corrector_eval("official"); dump()
# ---- perturbation-trained generators
gen = torch.Generator(device=DEV); gen.manual_seed(0)
R["sigma"] = {}
_, st_all, _ = ref_states(TR[:500], gen); RMS = st_all.pow(2).mean(dim=(0, 2, 3)).sqrt()
for sig in [float(x) for x in a.sigmas.split(",")]:
    A.set_seed(42); net.load_state_dict(BASE)
    params = [p for n, p in net.named_parameters() if n.startswith(("transformer.blocks", "transformer.ln_f", "lm_heads"))]
    for n, p in net.named_parameters(): p.requires_grad_(any(p is q for q in params))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    v0 = evaluate(VA)["scores"]; best = (v0, -1, copy.deepcopy(net.state_dict())); hist = []
    for ep in range(a.epochs):
        sub = TR[(ep * a.per_epoch) % ntr:][:a.per_epoch]; net.train(); tl = []
        perm = np.random.permutation(len(sub))
        for i in range(0, len(sub), a.bs):
            b = [sub[j] for j in perm[i:i + a.bs]]
            pf = torch.tensor([T.prefix(T.items[q]) for q in b], device=DEV)
            z, st, cor = ref_states(b, gen)
            pick = (torch.rand(len(b), 1, 1, 1, device=DEV) < 0.5).float()
            states = pick * st + (1 - pick) * cor
            states = states + sig * RMS.to(DEV)[None, :, None, None] * torch.randn_like(states)
            tl.append(train_step(pf, z, states, opt, params))
        net.eval(); v = evaluate(VA)["scores"]; hist.append({"epoch": ep, "loss": float(np.mean(tl)), "val": v})
        log(f"[sigma {sig}] ep{ep} loss {np.mean(tl):.4f} val {v}")
        if better(v, best[0]): best = (v, ep, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[2]); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    R["sigma"][str(sig)] = {"best_epoch": best[1], "hist": hist}
    R["sigma"][str(sig)].update(corrector_eval(f"sigma{sig}")); dump()
net.load_state_dict(BASE)
R["done"] = True; dump(); log("ALL DONE")
