"""Measure drift-coherence c0 and an off-manifold sensitivity proxy L_G on released checkpoints.

c0: for greedy rollouts, e^(k) = zhat^(k) - zGT^(k), eps_k = e^(k) - e^(k-1);
    c0_k = E[<u_{k-1}, eps_k>] / E||eps_k||,  u = e/||e||.
L_G: perturb rollout states off the codeword lattice with the seed-42 in-loop corrector at scale s,
     z~ = C_s(zhat^(k)); measure lattice distance d = ||z~ - Pi(z~)|| (re-quantised level residual)
     and next-level code error sigma_{k+1} = ||A_{k+1}(E(r_hat) - E(r*))|| where r* is the greedy
     residual-quantisation expert of (z - z~). Slope of sigma vs d (pooled OLS over s, k) = L_G proxy.
"""
import sys, json, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "."); import apc_lib as A, apc_v4 as V

def run(task, n, ck_v1):
    T = A.Task(task, f"ck/{task}", f"data/{task}"); K = len(T.levels); vq, net = T.vq, T.net
    ii = list(range(n)); pf = torch.tensor([T.prefix(T.items[i]) for i in ii])
    with torch.no_grad():
        y = torch.tensor([T.target_ids(T.refs[i]) for i in ii]); z = T.tk.vqvae.encode_z(y).float()
        _, zg = vq.z_to_idx(z, return_zhats=True)
        r = V.generate_v4(T, pf, record=True); zs = [zz for _, zz in r["zs"]]
        c0, sig = [], []
        e_prev = zs[0] - zg[0]
        for k in range(1, len(zs)):
            e = zs[k] - zg[k]; eps = (e - e_prev).flatten(1); u = e_prev.flatten(1)
            nu = u.norm(dim=1, keepdim=True); m = nu.squeeze(1) > 1e-6
            if m.sum() > 0:
                proj = ((u[m] / nu[m]) * eps[m]).sum(1)
                c0.append(float(proj.mean() / eps[m].norm(dim=1).mean().clamp_min(1e-9)))
                sig.append(float(eps.norm(dim=1).mean()))
            e_prev = e
        # L_G proxy
        h = json.load(open(ck_v1.replace("corrector_seed42.pt", "results.json")))["hidden_1.2M"]
        C = A.LatentCorrector(T.d, K - 1, hidden=h); C.load_state_dict(torch.load(ck_v1, map_location="cpu")); C.eval()
        st = V.capture(T, pf); pe, pb, lpe = st["prefix_emb"], st["pbias"], st["lpe"]
        ctx = T.prefix_ctx(pf); pts = []
        acc0 = st["acc"]; accs = [acc0]
        for li in range(K - 1): accs.append(accs[-1] + T.levels[li + 1])
        for s in (0.0, 0.25, 0.5, 1.0):
            C.eta_scale = s
            for li in range(K - 2):
                zk = zs[li]; prev = zs[li - 1] if li > 0 else torch.zeros_like(zk)
                kk = torch.full((n,), li, dtype=torch.long)
                zt = C.correct(zk, ctx, kk, None)
                zp, _ = V.project_level(T, prev, zt, li)
                d = (zt - zp).flatten(1).norm(dim=1)
                # generator codes at level li+1 from input zt
                x = F.interpolate(zt.transpose(-1, -2), size=(T.levels[li + 1],), mode="area").transpose(-1, -2)
                x = net.transformer.wte(x) + lpe[:, accs[li]:accs[li] + T.levels[li + 1], :]
                lg = net._decode_level_logits(x, pe, li, pb, hidx=li + (1 - net._lvl_first))
                rh = lg.argmax(-1)
                res = z - zt
                flat = vq._down_for_quant(res, li + 1, T.levels[li + 1]).reshape(-1, T.d) if li + 1 != K - 1 else res.reshape(-1, T.d)
                rs = torch.argmin(vq._distances(flat, vq.embeddings[li + 1].weight), 1).view(n, -1)
                def emb(r_):
                    q = vq.embeddings[li + 1](r_)
                    q = vq._up_to_full(q, li + 1, T.L) if li + 1 != K - 1 else q
                    return vq._apply_quant_resi(q, li + 1)
                sg = (emb(rh) - emb(rs)).flatten(1).norm(dim=1)
                for a, b in zip(d.tolist(), sg.tolist()):
                    pts.append((s, li, a, b))
    P = np.array(pts)
    on = P[P[:, 0] == 0]
    slope = float(np.polyfit(P[:, 2], P[:, 3], 1)[0])
    # per-level slopes (within-level to remove level confound)
    pl = []
    for li in range(K - 2):
        q = P[P[:, 1] == li]
        if q[:, 2].std() > 1e-6: pl.append(float(np.polyfit(q[:, 2], q[:, 3], 1)[0]))
    return {"c0_per_level": c0, "c0_mean": float(np.mean(c0)), "c0_min": float(np.min(c0)), "sigma_per_level": sig,
            "LG_pooled": slope, "LG_per_level": pl, "LG_median": float(np.median(pl)),
            "sigma_on_mean": float(on[:, 3].mean()), "dist_by_scale": {str(s): float(P[P[:, 0] == s][:, 2].mean()) for s in (0.0, .25, .5, 1.0)},
            "sigma_by_scale": {str(s): float(P[P[:, 0] == s][:, 3].mean()) for s in (0.0, .25, .5, 1.0)}, "n": n}

out = {"sudoku-hard": run("sudoku-hard", 64, "kout/sudoku_v1/results/sudoku-hard/corrector_seed42.pt")}
print(json.dumps(out["sudoku-hard"])[:800])
out["wmt14"] = run("wmt14", 12, "kout/wmt14_v1/results/wmt14/corrector_seed42.pt")
json.dump(out, open("theory_constants.json", "w"), indent=1)
print(json.dumps(out["wmt14"])[:800])
