#pragma once

// Jet-Long (arXiv 2607.07740; reference repo NVlabs Jet-Long @a7a9071) -- CPU-first, NON-fused prototype.
//
// Semantics (modeling_qwen3_jetlong.py, see DESIGN.md for the file:line anchors):
//   - L = max position of the sequence + 1. If L <= w (native window) the model runs the stock graph
//     (reference M:149 returns plain base cos/sin). Here this is enforced per ubatch AND per sequence:
//     an in-window sequence always gets the stock attention output rows, bit for bit.
//   - Above native: G = ceil(L / w) (M:155). For a query at position p and a key at position k:
//       near    (p - k <= w0): score = <q_base(p), k_base(k)>                         (FA window (w0,0), M:469)
//       distant (p - k >  w0): score = <q_grp(p),  k_grp(k)>, where x_grp is the BASE-roped vector rotated
//                              by delta = floor(pos/G) - pos (M:396-399, rotation M:227)
//     one softmax over the union == the LSE merge of the near and distant branches (M:617, and the
//     torch oracle `_reference_jetlong_decode`, jetlm/kernels/cute_jetlong_backend.py:93).
//   - The KV cache keeps BASE-roped K (M:336). The grouped K rows (only the n_rot rotary dims) are kept in a
//     side cache, valid per G-epoch: a row is recomputed when its cell is (re)written, its position changes,
//     or G changes (epoch crossing). With `uncached`, every row is recomputed on every ubatch.

#include "ggml.h"

#include <cstdint>
#include <functional>
#include <vector>

struct llama_jetlong_cfg {
    int32_t w0       = 0;     // local (near) window; 0 = Jet-Long off (stock)
    int32_t w_native = 0;     // native window w (default: n_ctx_orig / n_ctx_train)
    bool    f16      = false; // side cache stored as f16
    bool    uncached = false; // recompute every grouped-K row every ubatch (reference / T3 control)

    bool enabled() const { return w0 > 0 && w_native > 0; }
};

struct llama_jetlong_rope {
    int32_t n_rot       = 0;
    int32_t sections[4] = {0, 0, 0, 0};
    int32_t mode        = 0;   // GGML_ROPE_TYPE_NEOX / MROPE / IMROPE (neox pairing only)
    float   freq_base   = 10000.0f;

    bool is_mrope() const { return (mode & GGML_ROPE_TYPE_MROPE) != 0; }
    int  n_pos()    const { return is_mrope() ? 4 : 1; }
};

static inline int32_t llama_jetlong_group(int64_t L, int32_t w) {
    if (w <= 0 || L <= w) {
        return 1;
    }
    return (int32_t) ((L + w - 1) / w);
}

static inline int32_t llama_jetlong_delta(int32_t p, int32_t G) {
    // floor(p/G) - p ; p >= 0
    return (G <= 1) ? 0 : (p / G) - p;
}

struct llama_jetlong_ubatch_plan {
    bool    active = false;   // at least one sequence of the ubatch is above native
    int32_t G      = 1;       // ubatch group size (max over the above-native sequences, as the reference's batch max)
    std::vector<uint8_t> tok_ext; // [n_tokens] 1 = token's sequence is above native
};

// pos[i] = sequential position of token i, seq[i] = (first) sequence id of token i
llama_jetlong_ubatch_plan llama_jetlong_plan_ubatch(
        const llama_jetlong_cfg & cfg, int32_t n_tokens, const int32_t * pos, const int32_t * seq);

// per-cell bookkeeping of the grouped-K side cache (one entry per global cell index = stream*kv_size + j)
struct llama_jetlong_cell_state {
    std::vector<int32_t> pos; // position the row was computed for
    std::vector<int32_t> grp; // G the row was computed for (0 = invalid)

    void resize(size_t n) { pos.assign(n, -1); grp.assign(n, 0); }
    void invalidate()     { std::fill(grp.begin(), grp.end(), 0); }

    // cand_idx/cand_pos: every non-empty cell visible to the ubatch (global idx, position)
    // forced_idx:        cells written by this ubatch (always recomputed)
    // out: rows to (re)compute and their correction deltas; the state is committed
    void collect(int32_t G, bool uncached,
            const std::vector<int64_t> & cand_idx,
            const std::vector<int32_t> & cand_pos,
            const std::vector<int64_t> & forced_idx,
            std::vector<int64_t> & upd_idx,
            std::vector<int32_t> & upd_delta);
};

// graph inputs (created only for an active ubatch)
struct llama_jetlong_graph_inp {
    ggml_tensor * q_dpos    = nullptr; // I32 [n_tokens*n_pos]  correction delta per query
    ggml_tensor * k_dpos    = nullptr; // I32 [n_upd*n_pos]     correction delta per recomputed K row
    ggml_tensor * upd_idxs  = nullptr; // I64 [n_upd]           global cell index of each recomputed row
    ggml_tensor * m_near    = nullptr; // F32 [n_kv, n_tps, 1, n_stream]  1 = base score
    ggml_tensor * m_dist    = nullptr; // F32 [n_kv, n_tps, 1, n_stream]  1 = grouped score
    ggml_tensor * sel       = nullptr; // I32 [n_tokens]  row select: i (stock row) or n_tokens + i (Jet-Long row)
    ggml_tensor * pass_mask = nullptr; // F32 [n_embd_head] 0 on the n_rot rotary dims, 1 on the pass-through dims

    void create(ggml_context * ctx, const llama_jetlong_rope & rope,
            int64_t n_tokens, int64_t n_upd, int64_t n_kv, int64_t n_stream, int64_t n_embd_head);
};

// host-side fill of the graph inputs
//   cell_pos(token_i, j) returns the position of cell j as seen by token i's stream, or -1 if empty
void llama_jetlong_fill_inputs(
        const llama_jetlong_graph_inp   & inp,
        const llama_jetlong_cfg         & cfg,
        const llama_jetlong_rope        & rope,
        const llama_jetlong_ubatch_plan & plan,
        const int32_t                   * tok_pos,
        const std::vector<int64_t>      & upd_idx,
        const std::vector<int32_t>      & upd_delta,
        const std::function<int32_t(int64_t, int64_t)> & cell_pos);

// Build the Jet-Long attention for one layer and splice it into the stock output.
//   q           [D, H, T]      the query exactly as the stock path uses it (after the K Hadamard, if any)
//   q_rope      [D, H, T]      the query in the RoPE domain (before the K Hadamard)
//   k_rot       [R, R] or null the K Hadamard (orthonormal, symmetric; R | D)
//   k_store     the result of the K-cache store of this ubatch (dependency + raw cache, [D*Hkv, kv_size(, n_stream)])
//   k           [D, Hkv, n_kv, ns]  the stock K view (get_k)
//   v           the stock V view (get_v): [Dv, Hkv, n_kv, ns], or transposed when v_trans
//   kgrp        F32 [n_rot*Hkv, kv_size, n_stream_total] grouped-K side cache of this layer
//   s0          first stream of the view
//   out_stock   [Dv*H, T]  the stock attention output (before any V Hadamard)
// returns [Dv*H, T]: stock rows for in-window sequences (bit-identical), Jet-Long rows otherwise
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
        ggml_tensor * out_stock);
