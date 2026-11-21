#pragma once

// [GPU-RESIDENT-ROUND-2d] device DFlash2 lattice selection (LLAMA_DEVICE_LATTICE=1): the sequential draft-path
// selection over the [ids | scores] lattice, unrolled over the block positions with existing ggml ops
// (get_rows, mul, add, argmax, soft_max, concat). Gumbel-max: argmax(scores * inv_temp + noise) samples
// softmax(scores / T) exactly; inv_temp = 1, noise = 0 is the host greedy argmax (first max on ties, like
// std::max_element). Output: [1 + 2*top_k, block_size * n_blocks] rows [token | ids | probs], row 0 of each
// block zero.

#include "ggml.h"

// packed [n_embd, block_size * n_blocks] lattice, inv_temp [1, n_blocks], noise [top_k, block_size, n_blocks]
static inline ggml_tensor * llama_build_dflash_dsel(
        ggml_context * ctx0, ggml_tensor * packed, ggml_tensor * inv_temp, ggml_tensor * noise,
        int64_t top_k, int64_t block_size, int64_t n_blocks) {
        const int64_t W = 1 + 2*top_k;
    ggml_tensor * lat = ggml_reshape_3d(ctx0, packed, packed->ne[0], block_size, n_blocks);
    ggml_tensor * out = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, W, 1, n_blocks), 0.0f);
    ggml_tensor * pred = nullptr; // I32 [n_blocks]
    for (int64_t pos = 1; pos < block_size; ++pos) {
        // [ids (top_k) | scores (top_k x top_k: row = predecessor)] of this position, per block
        ggml_tensor * ids = ggml_view_2d(ctx0, lat, top_k, n_blocks, lat->nb[2], pos*lat->nb[1]);
        ggml_tensor * sc  = ggml_view_3d(ctx0, lat, top_k, top_k, n_blocks, top_k*lat->nb[0], lat->nb[2],
                                         pos*lat->nb[1] + top_k*lat->nb[0]);
        ggml_tensor * sel;
        if (pos == 1) {
            sel = ggml_cont_2d(ctx0, ggml_view_2d(ctx0, sc, top_k, n_blocks, sc->nb[2], 0), top_k, n_blocks); // predecessor 0
        } else {
            sel = ggml_get_rows(ctx0, ggml_cont(ctx0, sc), ggml_reshape_2d(ctx0, pred, 1, n_blocks));             // [top_k, 1, n_blocks]
            sel = ggml_reshape_2d(ctx0, sel, top_k, n_blocks);
        }
        ggml_tensor * scaled = ggml_mul(ctx0, sel, inv_temp);                                                 // scores / T
        ggml_tensor * g = ggml_cont_2d(ctx0, ggml_view_2d(ctx0, noise, top_k, n_blocks, noise->nb[2], pos*noise->nb[1]),
                                       top_k, n_blocks);
        pred = ggml_argmax(ctx0, ggml_add(ctx0, scaled, g));                                                       // I32 [n_blocks]

        ggml_tensor * ids_c = ggml_cont_3d(ctx0, ids, 1, top_k, n_blocks);
        ggml_tensor * tok   = ggml_get_rows(ctx0, ids_c, ggml_reshape_2d(ctx0, pred, 1, n_blocks));             // [1, 1, n_blocks]
        ggml_tensor * prob  = ggml_soft_max(ctx0, scaled);                                                         // [top_k, n_blocks]

        ggml_tensor * row = ggml_concat(ctx0, ggml_reshape_2d(ctx0, tok, 1, n_blocks), ggml_cont_2d(ctx0, ids, top_k, n_blocks), 0);
        row = ggml_concat(ctx0, row, prob, 0);
        out = ggml_concat(ctx0, out, ggml_reshape_3d(ctx0, row, W, 1, n_blocks), 1);
    }
    out = ggml_reshape_2d(ctx0, out, W, block_size * n_blocks);
    return out;
}
