#pragma once

// DRAFT-HEAD-SVD-CPU: two-stage truncated-SVD MTP draft head (CPU custom ops, header-only so tests can build it
// against plain ggml). Given the hidden column h [d] of the MTP head, the full head W [d, V] and factors
//   v [d, r] (B^T, F32)  and  a [r, V] (A = W_eff B, F16/Q8_0/F32)
// it computes
//   P   = a . (v . h)                       low-rank pre-score over the whole vocab           [V]
//   idx = top-C rows of P                   (set, unordered; C >= V selects every row)        [C] i32
//   Wc  = rows idx of W, copied verbatim    (same quant type, no requantisation)              [d, C]
//   lc  = Wc . h                            the SAME ggml_mul_mat kernel as the full head      [C]
//   out = -inf everywhere, lc at idx                                                          [V]
// Rows are gathered byte-for-byte and the CPU mul_mat kernels compute every output row independently, so the logit
// of every selected row equals the full head's logit for it (C = V reproduces the full head bit-for-bit; checked by
// tests/test-mtp-svd-head.cpp). Columns are processed one at a time (graph cost grows with n; callers bound n).

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mtp_svd {

static inline uint32_t f2key(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u); // order-preserving: larger float -> larger key
}

// top-C selection of one f32 row: 16-bit radix histogram, then the boundary bucket is resolved by nth_element.
// Single task (the row is small: V floats); NaN rows sort by bit pattern and are harmless for the exact rescoring.
static void op_select(ggml_tensor * dst, int ith, int nth, void * /*ud*/) {
    if (ith != 0) {
        return;
    }
    (void) nth;
    const ggml_tensor * p = dst->src[0];
    const int64_t V = p->ne[0];
    const int64_t C = dst->ne[0];
    const float * x = (const float *) p->data;
    int32_t * out = (int32_t *) dst->data;
    if (C >= V) {
        for (int64_t i = 0; i < V; ++i) out[i] = (int32_t) i;
        for (int64_t i = V; i < C; ++i) out[i] = (int32_t) (i % V);
        return;
    }
    std::vector<uint32_t> hist(65536, 0);
    for (int64_t i = 0; i < V; ++i) hist[f2key(x[i]) >> 16]++;
    int64_t acc = 0;
    int b = 65535;
    for (; b > 0; --b) {
        if (acc + (int64_t) hist[b] >= C) break;
        acc += hist[b];
    }
    // acc rows are strictly above bucket b; need C - acc more from bucket b
    int64_t n = 0;
    std::vector<std::pair<uint32_t, int32_t>> edge;
    edge.reserve(hist[b]);
    for (int64_t i = 0; i < V; ++i) {
        const uint32_t k = f2key(x[i]);
        const int hb = (int) (k >> 16);
        if (hb > b) {
            out[n++] = (int32_t) i;
        } else if (hb == b) {
            edge.emplace_back(k, (int32_t) i);
        }
    }
    const int64_t need = C - n;
    if (need > 0) {
        std::nth_element(edge.begin(), edge.begin() + (need - 1), edge.end(),
                         [](const auto & l, const auto & r) { return l.first > r.first || (l.first == r.first && l.second < r.second); });
        for (int64_t j = 0; j < need; ++j) out[n++] = edge[j].second;
    }
}

// verbatim row gather: dst [d, C] (type of W) <- W rows idx
static void op_gather(ggml_tensor * dst, int ith, int nth, void * /*ud*/) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * id = dst->src[1];
    const int64_t C = dst->ne[1];
    const size_t rs = ggml_row_size(w->type, w->ne[0]);
    const int32_t * ix = (const int32_t *) id->data;
    const int64_t per = (C + nth - 1) / nth;
    const int64_t i0 = per * ith, i1 = std::min<int64_t>(C, i0 + per);
    for (int64_t i = i0; i < i1; ++i) {
        std::memcpy((char *) dst->data + i * dst->nb[1], (const char *) w->data + (int64_t) ix[i] * w->nb[1], rs);
    }
}

// dst [V] f32: -inf, then lc[j] at idx[j]. Each task owns a vocab range and scans all C indices.
static void op_scatter(ggml_tensor * dst, int ith, int nth, void * /*ud*/) {
    const ggml_tensor * lc = dst->src[0];
    const ggml_tensor * id = dst->src[1];
    const int64_t V = dst->ne[0];
    const int64_t C = id->ne[0];
    const int64_t per = (V + nth - 1) / nth;
    const int64_t v0 = per * ith, v1 = std::min<int64_t>(V, v0 + per);
    float * o = (float *) dst->data;
    for (int64_t v = v0; v < v1; ++v) o[v] = -INFINITY;
    const int32_t * ix = (const int32_t *) id->data;
    const float * l = (const float *) lc->data;
    for (int64_t j = 0; j < C; ++j) {
        const int64_t v = ix[j];
        if (v >= v0 && v < v1) o[v] = l[j];
    }
}

// one column h [d, 1] -> logits [V, 1]
static ggml_tensor * head_col(ggml_context * ctx, ggml_tensor * w, ggml_tensor * h, ggml_tensor * v, ggml_tensor * a, int64_t C) {
    const int64_t V = w->ne[1];
    const int64_t d = w->ne[0];
    if (C > V) C = V;
    ggml_tensor * t = ggml_mul_mat(ctx, v, h);   // [r, 1]
    ggml_tensor * p = ggml_mul_mat(ctx, a, t);   // [V, 1]
    ggml_tensor * args_s[1] = { p };
    ggml_tensor * idx = ggml_custom_4d(ctx, GGML_TYPE_I32, C, 1, 1, 1, args_s, 1, op_select, 1, nullptr);
    ggml_tensor * args_g[2] = { w, idx };
    ggml_tensor * wc = ggml_custom_4d(ctx, w->type, d, C, 1, 1, args_g, 2, op_gather, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * lc = ggml_mul_mat(ctx, wc, h); // [C, 1]
    ggml_tensor * args_x[2] = { lc, idx };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, V, 1, 1, 1, args_x, 2, op_scatter, GGML_N_TASKS_MAX, nullptr);
}

// cur [d, n] f32 -> logits [V, n]
static ggml_tensor * build_head(ggml_context * ctx, ggml_tensor * w, ggml_tensor * cur, ggml_tensor * v, ggml_tensor * a, int64_t C) {
    const int64_t n = cur->ne[1];
    ggml_tensor * res = nullptr;
    for (int64_t j = 0; j < n; ++j) {
        ggml_tensor * h = n == 1 ? cur : ggml_view_2d(ctx, cur, cur->ne[0], 1, cur->nb[1], j * cur->nb[1]);
        if (n > 1) h = ggml_cont(ctx, h);
        ggml_tensor * o = head_col(ctx, w, h, v, a, C);
        res = res ? ggml_concat(ctx, res, o, 1) : o;
    }
    return res;
}

} // namespace mtp_svd
