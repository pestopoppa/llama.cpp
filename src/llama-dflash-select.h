#pragma once

// Q8-SERVED-DEV-SELECT-1: greedy DFlash2 selector argmax chain as a ggml graph tail.
//
// lattice: F32 [n_embd, block_size * n_blocks], the DFlash2 selector layout. Row (b*block_size + pos), pos >= 1, holds
//   [ candidate ids (top_k, as F32) | transition scores (top_k x top_k, predecessor-major) | padding ].
// Returns F32 [block_size - 1, n_blocks]: the drafted ids per block, bit-identical to the host chain in
// common/speculative.cpp draft(): depth 1 uses predecessor 0; depth i takes argmax over scores[pred][*] with ties to the
// lowest index (ggml_argmax's DTREE-A rule == std::max_element); the drafted id is ids[argmax].

#include "ggml.h"

static inline ggml_tensor * llama_dflash2_dev_select(ggml_context * ctx, ggml_tensor * lattice,
        int64_t top_k, int64_t block_size, int64_t n_blocks) {
    GGML_ASSERT(lattice->type == GGML_TYPE_F32 && ggml_is_contiguous(lattice));
    GGML_ASSERT(block_size > 1 && lattice->ne[1] == block_size * n_blocks);
    GGML_ASSERT(lattice->ne[0] >= top_k * (top_k + 1));

    const size_t es = lattice->nb[0];
    const size_t rs = lattice->nb[1];

    ggml_tensor * chain = nullptr;
    ggml_tensor * pred  = nullptr; // I32 [1, n_blocks]
    for (int64_t pos = 1; pos < block_size; ++pos) {
        ggml_tensor * scores = ggml_view_3d(ctx, lattice, top_k, top_k, n_blocks,
                top_k * es, block_size * rs, pos * rs + top_k * es);
        ggml_tensor * sel;
        if (pred == nullptr) {
            sel = ggml_cont_2d(ctx, ggml_view_2d(ctx, scores, top_k, n_blocks, scores->nb[2], 0), top_k, n_blocks);
        } else {
            sel = ggml_get_rows(ctx, ggml_cont(ctx, scores), pred);  // [top_k, 1, n_blocks]
            sel = ggml_reshape_2d(ctx, sel, top_k, n_blocks);
        }
        ggml_tensor * am = ggml_argmax(ctx, sel);                     // I32 [n_blocks]
        pred = ggml_reshape_2d(ctx, am, 1, n_blocks);

        ggml_tensor * ids = ggml_cont(ctx, ggml_view_3d(ctx, lattice, 1, top_k, n_blocks,
                es, block_size * rs, pos * rs));                       // [1, top_k, n_blocks]
        ggml_tensor * tok = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ids, pred), 1, n_blocks);
        chain = chain ? ggml_concat(ctx, chain, tok, 0) : tok;
    }
    return ggml_cont(ctx, chain);
}
