#pragma once

// [GPU-RESIDENT-ROUND-2a] device-built KQ mask (LLAMA_DEVICE_KQ_MASK=1, default off).
//
// The host path (llama_kv_cache::set_input_kq_mask) writes the full [n_kv, n_tps, 1, n_stream] mask on the CPU for
// every ubatch, and the scheduler then copies it to the device. With the knob on, the host writes only
//   cpos [n_kv, n_seq_max]  F32  : pos of cell j if the cell holds seq q, else LLAMA_KQ_MASK_DEV_FAR (row q per seq)
//   csel [n_tokens]         I32  : seq id of token i (seq_id[i][0], like the host path)
//   tpos [1, n_tps, 1, n_stream] F32 : pos of token i
// and the graph builds the mask on the device:
//   mask = log(step(tpos - cpos[csel] + 0.5))  ->  0 where p0 <= p1 (keep), -inf where p0 > p1 or the cell is
//   empty / not in the token's sequence (drop)
// which is bitwise the host mask for causal attention without ALiBi, SWA, 2-D positions or a tree ubatch
// (log(1) == 0 and log(0) == -inf exactly; positions are exact in F32 below 2^24).
// Host work per ubatch drops from n_kv*n_tokens mask cells to n_kv*n_seqs_in_ubatch floats, and the H2D copy shrinks
// by the same factor.

#include "ggml.h"

#include <cstdlib>
#include <cstring>

#define LLAMA_KQ_MASK_DEV_FAR 1.0e9f

static inline bool llama_kq_mask_dev_enabled() {
    // read on every graph build / reuse check (cheap) so that tests can flip it in-process
    const char * e = std::getenv("LLAMA_DEVICE_KQ_MASK");
    return e != nullptr && std::strcmp(e, "1") == 0;
}

// [GPU-RESIDENT-ROUND-2f phase 3.1] LLAMA_DEVICE_CTL_POS=1 (default off; needs LLAMA_DEVICE_CTL=1): positions input
// p0 and the device-mask token positions come from the KV device control block; accept advances are predicted on the
// host from the downloaded argmax rows instead of read back
static inline bool llama_ctl_pos_enabled() {
    const char * e = std::getenv("LLAMA_DEVICE_CTL_POS");
    return e != nullptr && std::strcmp(e, "1") == 0;
}

// [GPU-RESIDENT-ROUND-2f phase 3.3] LLAMA_DEVICE_RS_REP=1 (default off; needs LLAMA_DEVICE_CTL_POS=1 and the GDN
// replay mode LLAMA_RS_REPLAY): the KV control block gets a row 2 = n_accepted + 1, written by the verify graph,
// and the recurrent replay count n_rep (and its conv-window columns) of the next ubatch is read from it on the device
static inline bool llama_rs_rep_dev_enabled() {
    const char * e = std::getenv("LLAMA_DEVICE_RS_REP");
    return e != nullptr && std::strcmp(e, "1") == 0 && llama_ctl_pos_enabled();
}

// [GPU-RESIDENT-ROUND-2f phase 3.4a] LLAMA_DEVICE_CPOS=1 (default off; needs LLAMA_DEVICE_CTL_POS=1 and
// LLAMA_DEVICE_KQ_MASK=1): the device mask's cell positions come from the ring formula over the control block
// (p0 row + a high-water row) and csel = i / n_tps, so neither is a host input when the host check passes
static inline bool llama_cpos_dev_enabled() {
    const char * e = std::getenv("LLAMA_DEVICE_CPOS");
    return e != nullptr && std::strcmp(e, "1") == 0 && llama_ctl_pos_enabled();
}

// cpos [n_kv, n_seq_rows] F32, csel [n_tokens] I32, tpos [1, n_tps, 1, n_stream] F32 -> mask [n_kv, n_tps, 1, n_stream]
static inline ggml_tensor * llama_build_kq_mask_dev(
        ggml_context * ctx,
        ggml_tensor  * cpos,
        ggml_tensor  * csel,
        ggml_tensor  * tpos,
        ggml_type      type) {
    const int64_t n_kv     = cpos->ne[0];
    const int64_t n_tps    = tpos->ne[1];
    const int64_t n_stream = tpos->ne[3];

    ggml_tensor * cur = ggml_get_rows(ctx, cpos, csel);                // [n_kv, n_tokens]: p0 per (cell, token)
    cur = ggml_reshape_4d(ctx, cur, n_kv, n_tps, 1, n_stream);
    cur = ggml_sub(ctx, cur, tpos);                                    // p0 - p1
    cur = ggml_scale_bias(ctx, cur, -1.0f, 0.5f);                      // p1 - p0 + 0.5 (never 0)
    cur = ggml_step(ctx, cur);                                         // 1 keep, 0 drop
    cur = ggml_log(ctx, cur);                                          // 0 keep, -inf drop
    if (type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx, cur, type);
    }
    return cur;
}
