#pragma once

// Q38FN-GEMV2 DRAFTVOCAB: MTP draft head restricted to a fixed token subset (header-only so tests build it against
// plain ggml). Given a gathered sub-head Ws [d, K] (rows idx of the full head W, copied verbatim, same quant type) and
// the hidden columns h [d, n]:
//   lc  = Ws . h        the SAME ggml_mul_mat kernel as the full head          [K, n]
//   out = -inf everywhere, lc[j] at row idx[j]                                 [V, n]
// Rows are copied byte-for-byte and CPU mul_mat computes each output row independently, so every kept logit equals
// the full head's logit for that token. Only the draft graph uses it; verify keeps the full lm_head, so with greedy
// verification the emitted stream is unchanged (only the acceptance rate can move).

#include "ggml.h"

#include <cmath>
#include <cstdint>

namespace mtp_vocab {

// dst [V, n] f32 <- -inf, then lc[:, c] scattered at idx. Each task owns a vocab range of every column.
static void op_scatter(ggml_tensor * dst, int ith, int nth, void * /*ud*/) {
    const ggml_tensor * lc = dst->src[0];
    const ggml_tensor * id = dst->src[1];
    const int64_t V = dst->ne[0];
    const int64_t n = dst->ne[1];
    const int64_t K = id->ne[0];
    const int64_t per = (V + nth - 1) / nth;
    const int64_t v0 = per * ith, v1 = v0 + per < V ? v0 + per : V;
    const int32_t * ix = (const int32_t *) id->data;
    for (int64_t c = 0; c < n; ++c) {
        float * o = (float *) ((char *) dst->data + c * dst->nb[1]);
        const float * l = (const float *) ((const char *) lc->data + c * lc->nb[1]);
        for (int64_t v = v0; v < v1; ++v) o[v] = -INFINITY;
        for (int64_t j = 0; j < K; ++j) {
            const int64_t v = ix[j];
            if (v >= v0 && v < v1) o[v] = l[j];
        }
    }
}

// h [d, n] -> logits [V, n]
static inline ggml_tensor * build_head(ggml_context * ctx, ggml_tensor * w_sub, ggml_tensor * idx, ggml_tensor * h, int64_t V) {
    ggml_tensor * lc = ggml_mul_mat(ctx, w_sub, h);   // [K, n]
    ggml_tensor * args[2] = { lc, idx };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, V, h->ne[1], 1, 1, args, 2, op_scatter, GGML_N_TASKS_MAX, nullptr);
}

} // namespace mtp_vocab
