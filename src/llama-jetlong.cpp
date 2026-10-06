#include "llama-jetlong.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

//
// host-side planning
//

llama_jetlong_ubatch_plan llama_jetlong_plan_ubatch(
        const llama_jetlong_cfg & cfg, int32_t n_tokens, const int32_t * pos, const int32_t * seq) {
    llama_jetlong_ubatch_plan plan;

    if (!cfg.enabled() || n_tokens <= 0) {
        return plan;
    }

    // L_s = max position of sequence s in the ubatch + 1 (the reference uses max(position_ids)+1, M:146-147;
    // with a causal KV cache the newest token of a sequence carries its max position)
    std::unordered_map<int32_t, int64_t> seq_L;
    for (int32_t i = 0; i < n_tokens; ++i) {
        auto & L = seq_L[seq[i]];
        L = std::max<int64_t>(L, (int64_t) pos[i] + 1);
    }

    int64_t L_max_ext = 0;
    for (const auto & [s, L] : seq_L) {
        if (L > cfg.w_native) {
            L_max_ext = std::max(L_max_ext, L);
        }
    }

    if (L_max_ext == 0) {
        return plan; // every sequence is in-window -> stock graph, nothing else is built
    }

    plan.active = true;
    plan.G      = llama_jetlong_group(L_max_ext, cfg.w_native);
    plan.tok_ext.resize(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) {
        plan.tok_ext[i] = seq_L[seq[i]] > cfg.w_native ? 1 : 0;
    }

    return plan;
}

void llama_jetlong_cell_state::collect(int32_t G, bool uncached,
        const std::vector<int64_t> & cand_idx,
        const std::vector<int32_t> & cand_pos,
        const std::vector<int64_t> & forced_idx,
        std::vector<int64_t> & upd_idx,
        std::vector<int32_t> & upd_delta) {
    upd_idx.clear();
    upd_delta.clear();

    std::vector<uint8_t> forced;
    if (!forced_idx.empty()) {
        forced.assign(pos.size(), 0);
        for (int64_t c : forced_idx) {
            forced[c] = 1;
        }
    }

    for (size_t i = 0; i < cand_idx.size(); ++i) {
        const int64_t c = cand_idx[i];
        const int32_t p = cand_pos[i];

        const bool need = uncached || (!forced.empty() && forced[c]) || grp[c] != G || pos[c] != p;
        if (!need) {
            continue;
        }

        upd_idx.push_back(c);
        upd_delta.push_back(llama_jetlong_delta(p, G));

        pos[c] = p;
        grp[c] = G;
    }
}

//
// graph inputs
//

void llama_jetlong_graph_inp::create(ggml_context * ctx, const llama_jetlong_rope & rope,
        int64_t n_tokens, int64_t n_upd, int64_t n_kv, int64_t n_stream, int64_t n_embd_head) {
    const int64_t n_tps = n_tokens/n_stream;

    q_dpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens*rope.n_pos());
    ggml_set_input(q_dpos);
    ggml_set_name(q_dpos, "jl_q_dpos");

    k_dpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_upd*rope.n_pos());
    ggml_set_input(k_dpos);
    ggml_set_name(k_dpos, "jl_k_dpos");

    upd_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_upd);
    ggml_set_input(upd_idxs);
    ggml_set_name(upd_idxs, "jl_upd_idxs");

    m_near = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tps, 1, n_stream);
    ggml_set_input(m_near);
    ggml_set_name(m_near, "jl_m_near");

    m_dist = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tps, 1, n_stream);
    ggml_set_input(m_dist);
    ggml_set_name(m_dist, "jl_m_dist");

    sel = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(sel);
    ggml_set_name(sel, "jl_sel");

    pass_mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd_head);
    ggml_set_input(pass_mask);
    ggml_set_name(pass_mask, "jl_pass_mask");
}

void llama_jetlong_fill_inputs(
        const llama_jetlong_graph_inp   & inp,
        const llama_jetlong_cfg         & cfg,
        const llama_jetlong_rope        & rope,
        const llama_jetlong_ubatch_plan & plan,
        const int32_t                   * tok_pos,
        const std::vector<int64_t>      & upd_idx,
        const std::vector<int32_t>      & upd_delta,
        const std::function<int32_t(int64_t, int64_t)> & cell_pos) {
    const int64_t n_tokens = inp.sel->ne[0];
    const int64_t n_upd    = inp.upd_idxs->ne[0];
    const int64_t n_kv     = inp.m_near->ne[0];
    const int64_t n_tps    = inp.m_near->ne[1];
    const int64_t n_stream = inp.m_near->ne[3];
    const int     n_pos    = rope.n_pos();

    GGML_ASSERT((int64_t) upd_idx.size() == n_upd);
    GGML_ASSERT(n_tps*n_stream == n_tokens);

    {
        int32_t * d = (int32_t *) inp.q_dpos->data;
        for (int64_t i = 0; i < n_tokens; ++i) {
            const int32_t dp = plan.tok_ext[i] ? llama_jetlong_delta(tok_pos[i], plan.G) : 0;
            for (int a = 0; a < n_pos; ++a) {
                d[a*n_tokens + i] = dp; // same delta on every M-RoPE axis (text tokens)
            }
        }
    }
    {
        int32_t * d = (int32_t *) inp.k_dpos->data;
        int32_t * x = (int32_t *) inp.upd_idxs->data;
        for (int64_t r = 0; r < n_upd; ++r) {
            x[r] = (int32_t) upd_idx[r];
            for (int a = 0; a < n_pos; ++a) {
                d[a*n_upd + r] = upd_delta[r];
            }
        }
    }
    {
        float * mn = (float *) inp.m_near->data;
        float * md = (float *) inp.m_dist->data;
        for (int64_t i = 0; i < n_tokens; ++i) {
            const int32_t p1  = tok_pos[i];
            const bool    ext = plan.tok_ext[i] != 0;
            for (int64_t j = 0; j < n_kv; ++j) {
                bool near = true;
                if (ext) {
                    const int32_t p0 = cell_pos(i, j);
                    // empty cells are masked by the KQ mask anyway; keep them "near"
                    near = p0 < 0 || (int64_t) p1 - p0 <= cfg.w0;
                }
                mn[i*n_kv + j] = near ? 1.0f : 0.0f;
                md[i*n_kv + j] = near ? 0.0f : 1.0f;
            }
        }
    }
    {
        int32_t * s = (int32_t *) inp.sel->data;
        for (int64_t i = 0; i < n_tokens; ++i) {
            s[i] = plan.tok_ext[i] ? (int32_t) (n_tokens + i) : (int32_t) i;
        }
    }
    {
        float * m = (float *) inp.pass_mask->data;
        for (int64_t d = 0; d < inp.pass_mask->ne[0]; ++d) {
            m[d] = d < rope.n_rot ? 0.0f : 1.0f;
        }
    }
}

//
// graph
//

static ggml_tensor * jl_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * dpos, const llama_jetlong_rope & rope) {
    // pure rotation by the delta position: no YaRN, no mscale (CTX2_KSHIFT.md H1), freq_scale = 1
    int sections[4] = { rope.sections[0], rope.sections[1], rope.sections[2], rope.sections[3] };
    if (rope.is_mrope()) {
        return ggml_rope_multi(ctx, x, dpos, nullptr, rope.n_rot, sections, rope.mode, 0,
                rope.freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    return ggml_rope_ext(ctx, x, dpos, nullptr, rope.n_rot, rope.mode, 0,
            rope.freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

ggml_tensor * llama_jetlong_build_attn(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        const llama_jetlong_rope      & rope,
        const llama_jetlong_graph_inp & inp,
        ggml_tensor * q,
        ggml_tensor * q_rope,
        ggml_tensor * k_rot,
        ggml_tensor * k_store,
        ggml_tensor * k,
        ggml_tensor * v,
        bool          v_trans,
        ggml_tensor * kgrp,
        int64_t       s0,
        ggml_tensor * kq_mask,
        float         kq_scale,
        ggml_tensor * out_stock) {
    const int64_t D     = k->ne[0];
    const int64_t Hkv   = k->ne[1];
    const int64_t n_kv  = k->ne[2];
    const int64_t ns    = k->ne[3];
    const int64_t H     = q->ne[1];
    const int64_t T     = q->ne[2];
    const int64_t n_rot = rope.n_rot;
    const int64_t n_upd = inp.upd_idxs->ne[0];

    GGML_ASSERT(n_rot > 0 && n_rot <= D && n_rot % 2 == 0);
    GGML_ASSERT(kgrp->type == GGML_TYPE_F32 && kgrp->ne[0] == n_rot*Hkv);
    GGML_ASSERT(q_rope->ne[0] == D && q_rope->ne[1] == H && q_rope->ne[2] == T);

    const int64_t kv_size  = kgrp->ne[1];
    const int64_t n_cells  = ggml_nelements(kgrp)/(n_rot*Hkv); // kv_size * n_stream_total

    // 1. grouped-K side cache: (re)compute the rows of this ubatch (new cells, moved cells, or all on an epoch change)
    ggml_tensor * kgrp_dep;
    {
        ggml_tensor * kc = ggml_reshape_2d(ctx, k_store, D*Hkv, ggml_nelements(k_store)/(D*Hkv));
        GGML_ASSERT(kc->ne[1] == n_cells);

        // dequantize to f32 (the correction is applied after dequant, in f32)
        ggml_tensor * rows = ggml_get_rows(ctx, kc, inp.upd_idxs); // F32 [D*Hkv, n_upd]

        ggml_tensor * kr; // [n_rot, Hkv, n_upd] in the RoPE domain
        if (k_rot) {
            // the cache holds H_R * rope(k) per R-chunk; the n_rot rotary dims live in chunk 0
            const int64_t R = k_rot->ne[0];
            GGML_ASSERT(D % R == 0 && n_rot <= R);
            ggml_tensor * c0 = ggml_view_3d(ctx, rows, R, Hkv, n_upd, D*sizeof(float), D*Hkv*sizeof(float), 0);
            c0 = ggml_cont_2d(ctx, c0, R, Hkv*n_upd);
            // H is symmetric and H*H = I: rope(k)[0:n_rot] = H[0:n_rot, :] * c0
            ggml_tensor * hv = ggml_view_2d(ctx, k_rot, R, n_rot, k_rot->nb[1], 0);
            kr = ggml_mul_mat(ctx, hv, c0);
            kr = ggml_reshape_3d(ctx, kr, n_rot, Hkv, n_upd);
        } else {
            kr = ggml_view_3d(ctx, rows, n_rot, Hkv, n_upd, D*sizeof(float), D*Hkv*sizeof(float), 0);
        }

        ggml_tensor * kg = jl_rope(ctx, kr, inp.k_dpos, rope);               // [n_rot, Hkv, n_upd]
        kg = ggml_reshape_2d(ctx, ggml_cont(ctx, kg), n_rot*Hkv, n_upd);

        ggml_tensor * dst = ggml_reshape_2d(ctx, kgrp, n_rot*Hkv, n_cells);
        kgrp_dep = ggml_set_rows(ctx, dst, kg, inp.upd_idxs);
        ggml_build_forward_expand(gf, kgrp_dep);
    }

    // [n_rot, Hkv, n_kv, ns] view of the side cache, mirroring get_k
    ggml_tensor * kg4 = ggml_view_4d(ctx, kgrp_dep, n_rot, Hkv, n_kv, ns,
            n_rot*ggml_type_size(kgrp->type),
            n_rot*Hkv*ggml_type_size(kgrp->type),
            n_rot*Hkv*kv_size*ggml_type_size(kgrp->type),
            n_rot*Hkv*kv_size*ggml_type_size(kgrp->type)*s0);

    // 2. queries: stock (near), pass-through dims (distant), grouped rotary dims (distant)
    auto split_q = [&](ggml_tensor * x) {
        // [d, H, T] -> [d, T/ns, H, ns]
        if (!ggml_is_contiguous(x)) {
            x = ggml_cont(ctx, x);
        }
        x = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2]/ns, ns, x->nb[1], x->nb[2], x->nb[3]/ns, 0);
        return ggml_permute(ctx, x, 0, 2, 1, 3);
    };

    ggml_tensor * q_near = split_q(q);

    ggml_tensor * q_pass = ggml_mul(ctx, q_rope, inp.pass_mask);   // zero the rotary dims
    if (k_rot) {
        q_pass = llama_mul_mat_hadamard(ctx, q_pass, k_rot);       // into the cache's (Hadamard) domain
    }
    q_pass = split_q(q_pass);

    ggml_tensor * q_g = ggml_view_3d(ctx, q_rope, n_rot, H, T, q_rope->nb[1], q_rope->nb[2], 0);
    q_g = jl_rope(ctx, q_g, inp.q_dpos, rope);                     // [n_rot, H, T]
    q_g = split_q(q_g);

    ggml_tensor * kp  = ggml_permute(ctx, k,   0, 2, 1, 3);        // [D,     n_kv, Hkv, ns]
    ggml_tensor * kgp = ggml_permute(ctx, kg4, 0, 2, 1, 3);        // [n_rot, n_kv, Hkv, ns]

    // 3. scores
    ggml_tensor * s_near = ggml_mul_mat(ctx, kp, q_near);
    ggml_mul_mat_set_prec(s_near, GGML_PREC_F32);

    ggml_tensor * s_pass = ggml_mul_mat(ctx, kp, q_pass);
    ggml_mul_mat_set_prec(s_pass, GGML_PREC_F32);

    ggml_tensor * s_rot = ggml_mul_mat(ctx, kgp, q_g);
    ggml_mul_mat_set_prec(s_rot, GGML_PREC_F32);

    ggml_tensor * s_dist = ggml_add(ctx, s_pass, s_rot);

    ggml_tensor * kq = ggml_add(ctx,
            ggml_mul(ctx, s_near, inp.m_near),
            ggml_mul(ctx, s_dist, inp.m_dist));                    // [n_kv, T/ns, H, ns]

    kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

    // 4. values (f32, transposed for the mat-mul)
    ggml_tensor * vt;
    {
        ggml_tensor * v32 = v->type == GGML_TYPE_F32 ? v : ggml_cast(ctx, v, GGML_TYPE_F32);
        if (!v_trans) {
            vt = ggml_cont(ctx, ggml_permute(ctx, v32, 1, 2, 0, 3)); // [n_kv, Dv, Hkv, ns]
        } else {
            vt = ggml_cont(ctx, ggml_permute(ctx, v32, 0, 2, 1, 3)); // [n_kv, Hkv, Dv, ns] -> [n_kv, Dv, Hkv, ns]
        }
    }

    ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);                 // [Dv, T/ns, H, ns]
    ggml_tensor * cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);        // [Dv, H, T/ns, ns]
    cur = ggml_cont_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

    // 5. splice: in-window sequences keep the stock rows (pure copy -> bit-identical)
    GGML_ASSERT(out_stock->ne[0] == cur->ne[0] && out_stock->ne[1] == cur->ne[1]);
    ggml_tensor * both = ggml_concat(ctx, out_stock, cur, 1);     // [Dv*H, 2T]
    ggml_tensor * res  = ggml_get_rows(ctx, both, inp.sel);       // [Dv*H, T]

    ggml_build_forward_expand(gf, res);

    return res;
}
