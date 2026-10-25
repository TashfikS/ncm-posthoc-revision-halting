"""APC-NCM v4: on-manifold codebook projection + per-sequence decoder-margin halting with batch compaction.

The official NextConceptTransformer.generate() is run up to the start of the level loop (prefix encoding,
level-0 prediction); its local state is captured from the generate() frame at the first call of
get_next_autoregressive_input. The level loop (models/next_concept.py, eager / non-KV path) is then
re-executed with the same official sub-modules (_decode_level_logits, _sample_level, wte, lm_heads),
which is exact because every level attends only to the prefix and to itself. This makes it possible to
drop halted sequences from the batch (true compaction).
"""
import inspect, time, math
import torch, torch.nn.functional as F
import apc_lib as A


class _Capture(Exception):
    pass


@torch.no_grad()
def capture(T, pf):
    vq, net = T.vq, T.net
    orig = vq.get_next_autoregressive_input
    st = {}

    def hook(level_idx, z_hat, z_q_level):
        fl = inspect.currentframe().f_back.f_locals
        st.update(prefix_emb=fl["prefix_emb"], pbias=fl.get("_pbias"), lpe=fl["level_pos_embed"],
                  acc=fl["level_acum"], idx0=fl["idx_next"], z_hat=z_hat.clone())
        raise _Capture()

    vq.get_next_autoregressive_input = hook
    try:
        net.generate(None, pf, temperature=T.temp, top_k=T.top_k, top_p=T.top_p, cfg_scale=T.cfg, cfg_null_eos=T.EOS)
    except _Capture:
        pass
    finally:
        vq.get_next_autoregressive_input = orig
    return st


@torch.no_grad()
def project_level(T, z_prev, z_tilde, li):
    """Re-quantize the level-li residual (z_tilde - z_prev) against codebook V_li exactly as z_to_idx does."""
    vq = T.vq; L = T.L; K = len(T.levels)
    r = (z_tilde - z_prev).float()
    if li != K - 1:
        flat = vq._down_for_quant(r, li, T.levels[li]).reshape(-1, r.shape[-1])
    else:
        flat = r.reshape(-1, r.shape[-1])
    idx = torch.argmin(vq._distances(flat, vq.embeddings[li].weight), dim=1).view(r.shape[0], -1)
    q = vq._up_to_full(vq.embeddings[li](idx), li, L) if li != K - 1 else vq.embeddings[li](idx)
    return z_prev + vq._apply_quant_resi(q, li), idx


@torch.no_grad()
def margin_stats(T, z, prev_arg):
    p = F.softmax(T.tk.vqvae.decoder(z).float(), -1)
    top = p.topk(2, dim=-1).values
    mu = (top[..., 0] - top[..., 1]).min(dim=1).values
    arg = p.argmax(-1)
    stab = (arg == prev_arg).all(1) if prev_arg is not None else torch.zeros_like(mu, dtype=torch.bool)
    return mu, stab, arg


@torch.no_grad()
def generate_v4(T, pf, corrector=None, project=False, margin=None, kmin=None, record=False, use_official_l0=True):
    """Returns dict: z (B,L,d) final latents, levels (B,) executed levels, time, active_counts per level."""
    net, vq = T.net, T.vq
    K = len(T.levels); B = pf.shape[0]
    if T.dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    st = capture(T, pf)
    ctx = T.prefix_ctx(pf) if corrector is not None else None
    ctx_pad = None
    if corrector is not None and T.PAD is not None:
        ctx_pad = (pf[:, :T.P] == T.PAD); ctx_pad = ctx_pad & ~ctx_pad.all(1, keepdim=True)
    prefix_emb, pbias, lpe, acc = st["prefix_emb"], st["pbias"], st["lpe"], st["acc"]
    pb_batched = torch.is_tensor(pbias) and pbias.shape[0] == B and B > 1
    act = torch.arange(B, device=pf.device)
    z_hat = st["z_hat"]; idx_next = st["idx0"]
    out_z = torch.zeros(B, T.L, T.d, device=pf.device)
    levels = torch.full((B,), K, dtype=torch.long, device=pf.device)
    prev_arg = None; counts = []; zs = [] if record else None
    for li in range(K - 1):
        counts.append(len(act))
        z_prev = z_hat
        z_q = vq.embeddings[li](idx_next)
        z_hat, _ = vq.get_next_autoregressive_input(li, z_hat.clone(), z_q)
        if record == "pre":
            zs.append((act.clone(), z_hat.clone()))
        if corrector is not None:
            kk = torch.full((len(act),), li, device=pf.device, dtype=torch.long)
            z_hat = corrector.correct(z_hat, ctx[act], kk, ctx_pad[act] if ctx_pad is not None else None)
            if project:
                z_hat, _ = project_level(T, z_prev, z_hat, li)
        if record is True:
            zs.append((act.clone(), z_hat.clone()))
        if margin is not None and li + 1 >= kmin:
            mu, stab, arg = margin_stats(T, z_hat, prev_arg)
            done = (mu > margin) & stab
            if bool(done.any()):
                out_z[act[done]] = z_hat[done]; levels[act[done]] = li + 1
                keep = ~done
                act, z_hat, idx_next, arg = act[keep], z_hat[keep], idx_next[keep], arg[keep]
                if len(act) == 0:
                    break
            prev_arg = arg
        elif margin is not None and li + 2 >= kmin:
            prev_arg = T.tk.vqvae.decoder(z_hat).argmax(-1)
        x = F.interpolate(z_hat.transpose(-1, -2), size=(T.levels[li + 1],), mode="area").transpose(-1, -2)
        if net.use_post_upsample_conv:
            x = net._apply_post_upsample_conv_single(x, li + (1 - net._lvl_first))
        x = net.transformer.wte(x) + lpe[:, acc:acc + T.levels[li + 1], :]
        hi = li + (1 - net._lvl_first)
        pe = prefix_emb[act] if prefix_emb.shape[0] == B else prefix_emb
        pbb = pbias[act] if pb_batched else pbias
        logits = net._decode_level_logits(x, pe, li, pbb, hidx=hi)
        idx_next = net._sample_level(logits, T.temp, T.top_k, T.top_p)
        acc += T.levels[li + 1]
    if len(act) > 0:
        counts.append(len(act))
        zq = vq._apply_quant_resi(vq.embeddings[K - 1](idx_next), K - 1)
        out_z[act] = z_hat + zq
        if record:
            zs.append((act.clone(), (z_hat + zq).clone()))
    if T.dev == "cuda":
        torch.cuda.synchronize()
    return {"z": out_z, "levels": levels, "time": time.perf_counter() - t0, "counts": counts, "zs": zs}
