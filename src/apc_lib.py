"""APC-NCM: Adaptive Predictor-Corrector Next-Concept Modeling.

Runs on top of the *unmodified* official HazyResearch/ncm code. The official
NextConceptTransformer.generate() loop is executed as-is; the predictor-corrector
revision, halting and token freezing are injected by wrapping
MultiscaleResidualVectorQuantizer.get_next_autoregressive_input (models/quant.py),
which generate() calls once per level. Halting raises an exception that unwinds the
official loop, so no level after k* is executed.
"""
import os, sys, math, time, pickle, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from omegaconf import OmegaConf

NCM_ROOT = os.environ.get("NCM_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncm"))
sys.path.insert(0, NCM_ROOT)
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- loading
class Task:
    """Mirror of scripts/eval_shard.py model/data loading (device-agnostic)."""

    def __init__(self, task, ck, dd, dev=DEV):
        self.name = task
        self.CFG = CFG = OmegaConf.load(os.path.join(NCM_ROOT, "config", "eval", f"{task}.yaml"))
        G = CFG.generation
        CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
        for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
            if CFG.checkpoints.get(name):
                CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
        self.tk = VQVAETokenizer.from_config(CFG.tokenizer)
        self.tk.vqvae = self.tk.vqvae.to(dev).eval()
        self.net = NextConceptTransformer(vqvae_local=self.tk.vqvae, config=CFG.model).to(dev)
        sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
        self.net.load_state_dict(sd["model"], strict=True)
        self.net.eval()
        for p in list(self.net.parameters()) + list(self.tk.vqvae.parameters()):
            p.requires_grad_(False)
        self.vq = self.tk.vqvae.vq
        self.dev = dev
        self.P = int(CFG.model.prefix_len)
        self.EOS, self.PAD = self.tk.pre_tokenizer.eos_token_id, self.tk.pre_tokenizer.pad_token_id
        self.temp = float(G.get("temperature", 1.0)); self.top_k = int(G.get("top_k", 1))
        self.top_p = float(G.get("top_p", 0.0)); self.cfg = float(G.get("cfg_scale", 0.0))
        self.freq = float(G.get("frequency_penalty", 0.0)); self.mu = float(G.get("frequency_deficit", 0.0))
        self.CUT = bool(G.get("cut_at_eos", True))
        self.ALLOW = None
        if G.get("frequency_allowance"):
            path, ref = str(G.frequency_allowance).split(":", 1)
            obj = sd if path == CFG.checkpoints.generator else torch.load(os.path.join(ck, path), map_location="cpu", weights_only=False)
            for k in ref.split("."):
                obj = obj[k]
            self.ALLOW = obj.double().to(dev)
        del sd
        self.metric = str(CFG.eval.metric)
        self.items = self.refs = None
        if CFG.data and CFG.data.get("test"):
            self.items = pickle.load(open(os.path.join(dd, CFG.data.test), "rb"))
            self.refs = pickle.load(open(os.path.join(dd, CFG.data.test_refs), "rb"))
        self.src_tok = None
        if self.items is not None and isinstance(self.items[0], str):
            from transformers import AutoTokenizer
            self.src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))
        self.levels = list(self.vq.levels)
        self.L = self.levels[-1]
        self.d = self.tk.vqvae.embed_dim
        self.ctx_len = self.tk.vqvae.context_length

    # --- identical to eval_shard.prefix / dec / pick_tokens
    def prefix(self, item):
        ids = self.src_tok(item)["input_ids"] if self.src_tok is not None else list(item)
        return (ids[:self.P] + [self.PAD] * self.P)[:self.P]

    def dec(self, ids):
        ids = list(ids)
        if self.CUT and self.EOS in ids:
            ids = ids[:ids.index(self.EOS)]
        return self.tk.pre_tokenizer.decode([t for t in ids if t not in (self.EOS, self.PAD)], skip_special_tokens=True)

    def ref_str(self, r):
        return r if isinstance(r, str) else self.dec(r)

    def pick_tokens(self, logits):
        if self.freq <= 0:
            return logits.argmax(-1)
        B, T, V = logits.shape
        lg = logits.double()
        allow = (self.ALLOW[:V] * T)[None, :] if self.ALLOW is not None else None
        cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
        out = torch.empty(B, T, dtype=torch.long, device=lg.device)
        for t in range(T):
            step = lg[:, t, :].clone()
            excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
            step -= self.freq * excess
            if self.mu > 0 and allow is not None:
                step += self.mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
            p = step.argmax(-1)
            out[:, t] = p
            cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
        return out

    def prefix_ctx(self, prefix_tokens):
        """Frozen raw-token prefix embedding (same op generate() uses for raw_token_prefix)."""
        return self.tk.vqvae.encoder.wte(prefix_tokens)[:, :self.P, :].float()

    def target_ids(self, ref):
        """Pre-tokenised ground-truth sequence padded/cut to the VQ-VAE context length (training only)."""
        if isinstance(ref, str):
            ids = self.tk.pre_tokenizer.encode(ref)
            if self.EOS is not None:
                ids = ids + [self.EOS]
        else:
            ids = list(ref)
        pad = self.PAD if self.PAD is not None else (self.EOS if self.EOS is not None else 0)
        return (ids + [pad] * self.ctx_len)[:self.ctx_len]


# ----------------------------------------------------------------------------- corrector
class DSConv(nn.Module):
    def __init__(self, h, k=5):
        super().__init__()
        self.dw = nn.Conv1d(h, h, k, padding=k // 2, groups=h)
        self.pw = nn.Conv1d(h, h, 1)
        self.norm = nn.LayerNorm(h)

    def forward(self, x):  # x: B L h
        y = self.norm(x).transpose(1, 2)
        return x + self.pw(F.gelu(self.dw(y))).transpose(1, 2)


class LatentCorrector(nn.Module):
    """C_omega(z_pred^(k), h_prefix, k): 2 depthwise-separable conv blocks + cross-attention,
    zero-initialised output projection, learnable per-level step size eta_k in (0,1]."""

    def __init__(self, d_vq, n_levels, hidden=192, heads=4, eta_init=0.5, learn_eta=True):
        super().__init__()
        self.inp = nn.Linear(d_vq, hidden)
        self.lvl = nn.Embedding(n_levels, hidden)
        self.ctx = nn.Linear(d_vq, hidden)
        self.conv1, self.conv2 = DSConv(hidden), DSConv(hidden)
        self.q_norm, self.kv_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.xattn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.out = nn.Linear(hidden, d_vq)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        e = torch.full((n_levels,), float(eta_init)).clamp(1e-3, 1 - 1e-3)
        self.eta_logit = nn.Parameter(torch.log(e / (1 - e)), requires_grad=learn_eta)

    eta_scale = 1.0

    def eta(self, k):
        sc = self.eta_scale
        if torch.is_tensor(sc):
            sc = sc.to(self.eta_logit.device)[k]
        return sc * torch.sigmoid(self.eta_logit[k])

    def forward(self, z, ctx, k, ctx_pad=None):
        # z: B L d_vq ; ctx: B P d_vq ; k: LongTensor B
        scale = z.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-6).sqrt()
        h = self.inp(z / scale) + self.lvl(k)[:, None, :]
        h = self.conv2(self.conv1(h))
        c = self.kv_norm(self.ctx(ctx))
        a, _ = self.xattn(self.q_norm(h), c, c, key_padding_mask=ctx_pad, need_weights=False)
        h = h + a
        h = h + self.mlp(h)
        return self.out(h) * scale  # residual score in latent units

    def correct(self, z, ctx, k, ctx_pad=None):
        return z - self.eta(k)[:, None, None] * self(z, ctx, k, ctx_pad)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- hooked generation
class _Halt(Exception):
    pass


@torch.no_grad()
def apc_generate(T, prefix_tokens, corrector=None, halt=None, freeze_tau=None, record=False, corr_mode="loop"):
    """Run the official generate() with optional corrector / halting / token freezing.

    halt: None or dict(mode='flux'|'stab', tau=float, kmin=int)
    Returns dict with final latent z (B,L,d), per-level latents (if record), executed levels.
    """
    net, vq = T.net, T.vq
    K = len(T.levels)
    orig = vq.get_next_autoregressive_input
    ctx = T.prefix_ctx(prefix_tokens)
    ctx_pad = (prefix_tokens[:, :T.P] == T.PAD) if T.PAD is not None else None
    if ctx_pad is not None:
        ctx_pad = ctx_pad & ~ctx_pad.all(1, keepdim=True)  # guard all-pad rows
    S = {"zs": [], "prev": None, "levels_run": K, "halted_at": None, "frozen": None, "frozen_frac": [],
         "halt_mask": None}

    def hook(level_idx, z_hat, z_q_level):
        z_prev = z_hat.clone()
        z_hat, x = orig(level_idx, z_hat, z_q_level)  # predictor: z^(k) = z^(k-1) + Phi_k(Up_k(E_k(r_k)))
        if S["frozen"] is not None:  # token-wise freezing: carry frozen positions forward unchanged
            m = S["frozen"][:, :, None]
            z_hat.copy_(torch.where(m, z_prev, z_hat))
        kk = torch.full((z_hat.shape[0],), level_idx, device=z_hat.device, dtype=torch.long)
        if corrector is not None and corr_mode == "loop":  # in-loop revision before the area downsample
            z_hat.copy_(corrector.correct(z_hat, ctx, kk, ctx_pad))
        x = F.interpolate(z_hat.transpose(-1, -2), size=(T.levels[level_idx + 1],), mode="area").transpose(-1, -2)
        if corrector is not None and corr_mode == "decode":  # decode-path revision: generator input untouched
            z_out = corrector.correct(z_hat, ctx, kk, ctx_pad)
            z_prev_out = S.get("last_out", z_prev)
        else:
            z_out, z_prev_out = z_hat, z_prev
        if record:
            S["zs"].append(z_out.clone())
        S["last"] = z_hat
        S["last_out"] = z_out.clone() if z_out is not z_hat else z_hat
        if level_idx >= 1 and freeze_tau is not None:
            dloc = (z_hat - z_prev).norm(dim=-1) / (z_prev.norm(dim=-1) + 1e-6)  # B L
            seg = F.interpolate(dloc[:, None, :], size=(T.levels[level_idx + 1],), mode="area")
            seg_full = F.interpolate(seg, size=(T.L,), mode="nearest")[:, 0]
            fr = seg_full < freeze_tau
            S["frozen"] = fr if S["frozen"] is None else (S["frozen"] | fr)
            S["frozen_frac"].append(S["frozen"].float().mean().item())
        if halt is not None and level_idx >= halt["kmin"]:
            if halt["mode"] == "flux":
                crit = (z_out - z_prev_out).flatten(1).norm(dim=1) / (z_prev_out.flatten(1).norm(dim=1) + 1e-6)
            else:
                a = T.tk.vqvae.decoder(z_out).argmax(-1); b = T.tk.vqvae.decoder(z_prev_out).argmax(-1)
                crit = 1.0 - (a == b).float().mean(1)
            done = crit < halt["tau"]
            S["halt_mask"] = done if S["halt_mask"] is None else S["halt_mask"]
            if bool(done.all()):
                S["halted_at"] = level_idx
                raise _Halt()
        return z_hat, x

    vq.get_next_autoregressive_input = hook
    t0 = time.perf_counter()
    try:
        g = net.generate(None, prefix_tokens, temperature=T.temp, top_k=T.top_k, top_p=T.top_p,
                         cfg_scale=T.cfg, cfg_null_eos=T.EOS)
        # final level (not processed inside the official loop): add Phi_{K-1}(E_{K-1}(r_{K-1}))
        last = g[:, -T.levels[-1]:]
        zq = vq._apply_quant_resi(vq.embeddings[K - 1](last), K - 1)
        z = S["last"] + zq
        if corrector is not None and corr_mode == "decode" and corrector.lvl.num_embeddings >= K:
            kk = torch.full((z.shape[0],), K - 1, device=z.device, dtype=torch.long)
            z = corrector.correct(z, ctx, kk, ctx_pad)
        if record:
            S["zs"].append(z.clone())
        S["levels_run"] = K
    except _Halt:
        z = S["last_out"].clone()
        S["levels_run"] = S["halted_at"] + 1
        g = None
    finally:
        vq.get_next_autoregressive_input = orig
    if T.dev == "cuda":
        torch.cuda.synchronize()
    S["time"] = time.perf_counter() - t0
    S["z"] = z
    S["idx"] = g
    return S


@torch.no_grad()
def baseline_official(T, prefix_tokens):
    """Exactly the eval_shard.run() path: generate -> idx_to_zhats -> decoder(zl[-1])."""
    g = T.net.generate(None, prefix_tokens, temperature=T.temp, top_k=T.top_k, top_p=T.top_p,
                       cfg_scale=T.cfg, cfg_null_eos=T.EOS)
    _, zl = T.vq.idx_to_zhats(g, return_zlevels=True)
    return g, zl[-1]


@torch.no_grad()
def decode_latent(T, z):
    ids = T.pick_tokens(T.tk.vqvae.decoder(z)).tolist()
    return [T.dec(r) for r in ids]


# ----------------------------------------------------------------------------- metrics
def sudoku_scores(hyps, refs):
    cells = lambda t: t.replace("|", "")
    ex = np.array([float(h == r) for h, r in zip(hyps, refs)])
    ce = np.array([np.mean([a == b for a, b in zip(cells(h), cells(r))]) for h, r in zip(hyps, refs)])
    return {"exact": ex, "cell": ce}


def item_scores(T, hyps, refs):
    if T.metric == "exact":
        return sudoku_scores(hyps, refs)
    if T.metric == "bleu":
        import sacrebleu
        return {"sbleu": np.array([sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)])}
    if T.metric == "rouge":
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
        for h, r in zip(hyps, refs):
            v = sc.score(r, h)
            for k in R:
                R[k].append(v[k].fmeasure)
        return {k: np.array(v) for k, v in R.items()}
    raise ValueError(T.metric)


def corpus_scores(T, hyps, refs):
    s = item_scores(T, hyps, refs)
    out = {k: 100 * float(v.mean()) if k != "sbleu" else float(v.mean()) for k, v in s.items()}
    if T.metric == "bleu":
        import sacrebleu
        out["bleu"] = sacrebleu.corpus_bleu(hyps, [refs]).score
    return out, s


# ----------------------------------------------------------------------------- corrector training data
@torch.no_grad()
def gt_trajectory(T, refs, all_levels=False):
    y = torch.tensor([T.target_ids(r) for r in refs], device=T.dev)
    z = T.tk.vqvae.encode_z(y)
    idx, zhats = T.vq.z_to_idx(z, return_zhats=True)
    return y, idx, torch.stack(zhats if all_levels else zhats[:-1], 1)  # B, K(-1), L, d


@torch.no_grad()
def corrupted_trajectory(T, idx, generator=None, all_levels=False):
    """NCM-style uniform code corruption (skip level 0), u ~ U(0,1) per sample."""
    B = idx.shape[0]
    u = torch.rand(B, 1, device=idx.device, generator=generator)
    noisy = idx.clone()
    l0 = T.levels[0]
    m = torch.rand(idx.shape, device=idx.device, generator=generator) < u
    m[:, :l0] = False
    acc = 0
    for li, (l, vs) in enumerate(zip(T.levels, T.vq.vqvae_vocab_sizes)):
        seg = slice(acc, acc + l)
        rnd = torch.randint(0, vs, (B, l), device=idx.device, generator=generator)
        noisy[:, seg] = torch.where(m[:, seg], rnd, noisy[:, seg]); acc += l
    zh, = T.vq.idx_to_zhats(noisy, return_zlevels=False)
    return torch.stack([t.clone() for t in (zh if all_levels else zh[:-1])], 1)


@torch.no_grad()
def rollout_trajectory(T, prefix_tokens, all_levels=False):
    S = apc_generate(T, prefix_tokens, record=True)
    return torch.stack(S["zs"] if all_levels else S["zs"][:-1], 1)


def build_train_set(T, idxs, bs, n_corrupt=2, seed=0, all_levels=False):
    """Returns tensors on CPU: prefix(N,P), y(N,ctx), z_gt(N,K-1,L,d), z_pred list of (N,K-1,L,d)."""
    gen = torch.Generator(device=T.dev); gen.manual_seed(seed)
    PF, Y, ZG, ZR, ZC = [], [], [], [], []
    for s in range(0, len(idxs), bs):
        b = idxs[s:s + bs]
        pf = torch.tensor([T.prefix(T.items[i]) for i in b], device=T.dev)
        y, idx, zg = gt_trajectory(T, [T.refs[i] for i in b], all_levels)
        zr = rollout_trajectory(T, pf, all_levels)
        zc = [corrupted_trajectory(T, idx, gen, all_levels) for _ in range(n_corrupt)]
        PF.append(pf.cpu()); Y.append(y.cpu()); ZG.append(zg.half().cpu()); ZR.append(zr.half().cpu())
        ZC.append(torch.stack(zc, 0).half().cpu())
    return {"prefix": torch.cat(PF), "y": torch.cat(Y), "zg": torch.cat(ZG), "zr": torch.cat(ZR), "zc": torch.cat(ZC, 1)}


def train_corrector(T, data, hidden=192, epochs=8, lr=3e-4, bs=32, alpha=0.1, seed=42, eta_init=0.5,
                    learn_eta=True, val_frac=0.2, p_roll=0.75, log=print):
    set_seed(seed)
    K1 = data["zg"].shape[1]
    C = LatentCorrector(T.d, K1, hidden=hidden, eta_init=eta_init, learn_eta=learn_eta).to(T.dev)
    opt = torch.optim.AdamW([p for p in C.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    N = data["prefix"].shape[0]
    perm = torch.randperm(N)
    nv = max(1, int(val_frac * N))
    vi, ti = perm[:nv], perm[nv:]
    # per-level normaliser lambda_k = 1 / E||z_gt^(k)||^2
    lam = 1.0 / data["zg"].float().pow(2).mean(dim=(0, 2, 3)).clamp_min(1e-6)
    lam = (lam / lam.sum() * K1).to(T.dev)
    nsrc = 1 + data["zc"].shape[0]
    steps = epochs * math.ceil(len(ti) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    dec = T.tk.vqvae.decoder

    def loss_on(ids, train):
        pf = data["prefix"][ids].to(T.dev)
        zg = data["zg"][ids].to(T.dev).float()
        src = (0 if torch.rand(()).item() < p_roll else torch.randint(1, nsrc, (1,)).item()) if train else 0
        zp = (data["zr"][ids] if src == 0 else data["zc"][src - 1][ids]).to(T.dev).float()
        B = zg.shape[0]
        ctx = T.prefix_ctx(pf)
        pad = (pf == T.PAD) if T.PAD is not None else None
        if pad is not None:
            pad = pad & ~pad.all(1, keepdim=True)
        ctxr = ctx[:, None].expand(B, K1, *ctx.shape[1:]).reshape(B * K1, *ctx.shape[1:])
        padr = pad[:, None].expand(B, K1, pad.shape[1]).reshape(B * K1, -1) if pad is not None else None
        kk = torch.arange(K1, device=T.dev).repeat(B)
        zt = C.correct(zp.reshape(B * K1, T.L, T.d), ctxr, kk, padr)
        zgf = zg.reshape(B * K1, T.L, T.d)
        mse = (zt - zgf).pow(2).mean(dim=(1, 2)).view(B, K1)
        # decoder-space KL on a random subset of (sample, level) pairs to bound memory for
        # large output vocabularies (Qwen3 152k / GPT-2 50k); rescaled by (B*K1)/M.
        if not hasattr(T, "_V"):
            with torch.no_grad():
                T._V = dec(zgf[:1]).shape[-1]
        V = T._V
        M = B * K1 if V * T.L * B * K1 <= 6e7 else max(1, int(6e7 // (V * T.L)))
        sel = torch.randperm(B * K1, device=T.dev)[:M]
        lp_t = F.log_softmax(dec(zt[sel]).float(), -1)
        with torch.no_grad():
            p_g = F.softmax(dec(zgf[sel]).float(), -1)
        kl_s = (p_g * (torch.log(p_g.clamp_min(1e-9)) - lp_t)).sum(-1).mean(-1)
        kl = torch.zeros(B * K1, device=T.dev).index_put((sel,), kl_s * (B * K1 / M)).view(B, K1)
        base = (zp.reshape(B * K1, T.L, T.d) - zgf).pow(2).mean(dim=(1, 2)).view(B, K1)
        L = ((mse + alpha * kl) * lam).sum(1).mean()
        return L, (mse * lam).sum(1).mean().item(), (base * lam).sum(1).mean().item()

    hist = []
    t0 = time.time()
    import copy
    C.eval()
    with torch.no_grad():
        vl0 = [loss_on(vi[s:s + bs], False) for s in range(0, len(vi), bs)]
    best_v = float(np.mean([x.item() for x, _, _ in vl0])); best_state = copy.deepcopy(C.state_dict()); best_ep = -1
    for ep in range(epochs):
        C.train()
        sh = ti[torch.randperm(len(ti))]
        tl = []
        for s in range(0, len(sh), bs):
            L, _, _ = loss_on(sh[s:s + bs], True)
            opt.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(C.parameters(), 1.0); opt.step(); sched.step()
            tl.append(L.item())
        C.eval()
        with torch.no_grad():
            vl = [loss_on(vi[s:s + bs], False) for s in range(0, len(vi), bs)]
        v = float(np.mean([a.item() for a, _, _ in vl])); vm = float(np.mean([b for _, b, _ in vl])); vb = float(np.mean([c for _, _, c in vl]))
        if v < best_v:
            best_v, best_state, best_ep = v, copy.deepcopy(C.state_dict()), ep
        hist.append({"epoch": ep, "train": float(np.mean(tl)), "val": v, "val_mse": vm, "val_mse_uncorrected": vb})
        log(f"  [C seed{seed} h{hidden}] ep{ep} train {np.mean(tl):.4f} val {v:.4f} mse {vm:.4f} (uncorr {vb:.4f}) eta {torch.sigmoid(C.eta_logit).mean().item():.3f}")
    C.load_state_dict(best_state)
    C.best_epoch, C.best_val, C.val_idx = best_ep, best_v, vi.tolist()
    log(f"  [C seed{seed}] best epoch {best_ep} (-1 = identity) val {best_v:.4f}")
    return C, hist, time.time() - t0
