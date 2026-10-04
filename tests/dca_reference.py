#!/usr/bin/env python3
"""Dual Chunk Attention (DCA) numpy reference.

DCA (An et al. 2024, "Training-Free Long-Context Scaling of Large Language Models",
arXiv 2402.17463, ChunkLlama; used by Qwen2.5-1M through vLLM's DualChunkRotaryEmbedding
and dual_chunk_flash_attn backend). Notation: chunk_len c = chunk_size - local_size,
p = absolute position.

  key position               k = p mod c          (keys are chunk-periodic; roped once)
  query, same chunk          q = p mod c          (intra, causal)
  query, previous chunk      q = min(p mod c + c, chunk_size)   (succ)
  query, any earlier chunk   q = min(2c - 1, chunk_size)   (inter; qc_t[c-1], = chunk_size unless local >= c)

Two formulations, which must agree:
  * concat: per query, logits for every causal key with the query variant its chunk
    distance selects, one softmax (ChunkLlama decode; the llama.cpp build_attn_dca path);
  * lse: three partial attentions over disjoint key sets, merged with their
    log-sum-exps (vLLM's _merge_attn_outputs).

Optional temperature (vLLM, original_max_position_embeddings > 0):
  s = max(1, 0.1 * ln(n / orig) + 1), n = sequence length seen by the query (p + 1).

Usage: python3 dca_reference.py                        # self-checks on a synthetic config
       python3 dca_reference.py --check-dump <dir>     # numpy DCA vs the llama graph's captured
                                                       # layer-0 attention (DCA_TEST_DUMP=<dir> test-dca)
"""

import numpy as np


def rope_neox(x, pos, n_rot, base):
    """ggml NEOX rope. x [..., D] float64, pos broadcastable to x[..., 0]."""
    half = n_rot // 2
    inv = base ** (-2.0 * np.arange(half) / n_rot)
    ang = np.asarray(pos, dtype=np.float64)[..., None] * inv
    c, s = np.cos(ang), np.sin(ang)
    out = np.array(x, dtype=np.float64, copy=True)
    x0, x1 = x[..., :half], x[..., half:n_rot]
    out[..., :half] = x0 * c - x1 * s
    out[..., half:n_rot] = x0 * s + x1 * c
    return out


def dca_positions(p, chunk_size, local_size):
    c = chunk_size - local_size
    r = p % c
    # inter: the succ position of the chunk's last row, qc_t[c - 1] (ChunkLlama / vLLM)
    return r, np.minimum(r + c, chunk_size), np.full_like(p, min(2 * c - 1, chunk_size))


def temperature(p, orig):
    if not orig:
        return np.ones_like(p, dtype=np.float64)
    return np.maximum(1.0, 0.1 * np.log((p + 1.0) / orig) + 1.0)


def dca_concat(q, k, v, pos, chunk_size, local_size, n_rot, base, kq_scale, orig=0):
    """q [T,Hq,D], k [T,Hkv,D], v [T,Hkv,Dv], pos [T] (one sequence). -> [T,Hq,Dv]"""
    T, Hq, _ = q.shape
    Hkv = k.shape[1]
    c = chunk_size - local_size
    kp, _, _ = dca_positions(pos, chunk_size, local_size)
    kr = rope_neox(k, kp[:, None], n_rot, base)
    q_int, q_suc, q_ext = dca_positions(pos, chunk_size, local_size)
    qr = [rope_neox(q, qp[:, None], n_rot, base) for qp in (q_int, q_suc, q_ext)]
    s = temperature(pos, orig)
    out = np.zeros((T, Hq, v.shape[2]))
    g = Hq // Hkv
    for t in range(T):
        keys = np.nonzero(pos <= pos[t])[0]
        dist = pos[t] // c - pos[keys] // c
        sel = np.where(dist == 0, 0, np.where(dist == 1, 1, 2))
        for h in range(Hq):
            hk = h // g
            logit = np.array([qr[sel[i]][t, h] @ kr[j, hk] for i, j in enumerate(keys)])
            logit *= kq_scale * s[t]
            w = np.exp(logit - logit.max())
            out[t, h] = (w / w.sum()) @ v[keys, hk]
    return out


def _partial(qr, kr, v, keys, scale):
    """softmax attention of one query row over a key subset -> (out, lse)"""
    logit = (kr[keys] @ qr) * scale
    m = logit.max()
    w = np.exp(logit - m)
    return (w / w.sum()) @ v[keys], m + np.log(w.sum())


def dca_lse(q, k, v, pos, chunk_size, local_size, n_rot, base, kq_scale, orig=0):
    """vLLM-style: intra / succ / inter partial attentions merged by LSE."""
    T, Hq, _ = q.shape
    Hkv = k.shape[1]
    c = chunk_size - local_size
    kp, _, _ = dca_positions(pos, chunk_size, local_size)
    kr = rope_neox(k, kp[:, None], n_rot, base)
    qr = [rope_neox(q, qp[:, None], n_rot, base) for qp in dca_positions(pos, chunk_size, local_size)]
    s = temperature(pos, orig)
    out = np.zeros((T, Hq, v.shape[2]))
    g = Hq // Hkv
    for t in range(T):
        causal = pos <= pos[t]
        dist = pos[t] // c - pos // c
        sets = [np.nonzero(causal & (dist == 0))[0], np.nonzero(causal & (dist == 1))[0], np.nonzero(causal & (dist >= 2))[0]]
        for h in range(Hq):
            hk = h // g
            parts = [_partial(qr[i][t, h], kr[:, hk], v[:, hk], ks, kq_scale * s[t]) for i, ks in enumerate(sets) if len(ks)]
            lse = np.array([l for _, l in parts])
            w = np.exp(lse - lse.max())
            w /= w.sum()
            out[t, h] = sum(wi * o for wi, (o, _) in zip(w, parts))
    return out


def plain(q, k, v, pos, n_rot, base, kq_scale):
    T, Hq, _ = q.shape
    Hkv = k.shape[1]
    kr = rope_neox(k, pos[:, None], n_rot, base)
    qr = rope_neox(q, pos[:, None], n_rot, base)
    out = np.zeros((T, Hq, v.shape[2]))
    g = Hq // Hkv
    for t in range(T):
        keys = np.nonzero(pos <= pos[t])[0]
        for h in range(Hq):
            logit = (kr[keys, h // g] @ qr[t, h]) * kq_scale
            w = np.exp(logit - logit.max())
            out[t, h] = (w / w.sum()) @ v[keys, h // g]
    return out


def self_check():
    rng = np.random.default_rng(0)
    T, Hq, Hkv, D = 80, 4, 2, 32
    q = rng.normal(size=(T, Hq, D))
    k = rng.normal(size=(T, Hkv, D))
    v = rng.normal(size=(T, Hkv, D))
    pos = np.arange(T)
    args = dict(n_rot=D, base=10000.0, kq_scale=1 / np.sqrt(D))

    a = dca_concat(q, k, v, pos, 24, 8, orig=40, **args)
    b = dca_lse(q, k, v, pos, 24, 8, orig=40, **args)
    e1 = np.abs(a - b).max()

    # chunk_len > every position: DCA is plain rope attention
    c1 = dca_concat(q, k, v, pos, 1024, 0, **args)
    e2 = np.abs(c1 - plain(q, k, v, pos, **args)).max()

    # a small chunk changes the result (the remap is not a no-op)
    e3 = np.abs(dca_concat(q, k, v, pos, 24, 8, **args) - plain(q, k, v, pos, **args)).max()

    # relative positions never exceed chunk_size: every query/key position difference is in [0, chunk_size]
    c = 16
    p = np.arange(200)
    qi, qs, qe = dca_positions(p, 24, 8)
    kp = p % c
    worst = 0
    for t in range(200):
        for j in range(t + 1):
            d = t // c - j // c
            qp = (qi, qs, qe)[0 if d == 0 else (1 if d == 1 else 2)][t]
            worst = max(worst, qp - kp[j])
            assert qp - kp[j] >= 0
    print(f"concat vs lse   max|d| = {e1:.2e}  (must be ~0)")
    print(f"big chunk vs plain max|d| = {e2:.2e}  (must be ~0)")
    print(f"small chunk vs plain max|d| = {e3:.2e}  (must be > 0)")
    print(f"max relative position = {worst}  (must be <= chunk_size 24)")
    ok = e1 < 1e-12 and e2 < 1e-12 and e3 > 1e-3 and worst <= 24
    print("OK" if ok else "FAIL")
    return ok


def check_dump(root):
    """compare numpy DCA with the llama graph's output dumped by test-dca (DCA_TEST_DUMP)"""
    import os
    ok = True
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        meta = dict(line.split() for line in open(os.path.join(d, "meta.txt")))
        T, Hq, Hkv, D = (int(meta[k]) for k in ("T", "Hq", "Hkv", "D"))
        cs, ls, orig = int(meta["chunk_size"]), int(meta["local_size"]), int(meta["orig_ctx"])
        rd = lambda n, *shape: np.fromfile(os.path.join(d, n + ".f32"), dtype=np.float32).astype(np.float64).reshape(shape)
        q, k, v = rd("dca_q_in-0", T, Hq, D), rd("dca_k_in-0", T, Hkv, D), rd("dca_v_in-0", T, Hkv, D)
        got = rd("dca_out-0", T, Hq, D)
        pos = np.arange(T)
        args = dict(n_rot=int(meta["n_rot"]), base=float(meta["base"]), kq_scale=1 / np.sqrt(D))
        ref = dca_lse(q, k, v, pos, cs, ls, orig=orig, **args)
        pl = plain(q, k, v, pos, **args)
        e = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        ep = np.linalg.norm(got - pl) / np.linalg.norm(pl)
        good = e < 1e-5 and ep > 1e-2
        print(f"{sub}: llama vs numpy DCA (LSE merge) rel_err = {e:.2e}; vs plain rope = {ep:.2e}  {'OK' if good else 'FAIL'}")
        ok = ok and good
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--check-dump":
        raise SystemExit(0 if check_dump(sys.argv[2]) else 1)
    raise SystemExit(0 if self_check() else 1)
