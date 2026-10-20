#pragma once
// DSA-IDX.1: contract for the indexed sparse-attention op (DS41-C123.1 / D2.4 consumer).
// Adopts the FlashMLA sparse-decode contract. This header is the SPEC plus a scalar CPU reference;
// no backend kernel exists yet. Any future kernel must reproduce llama_dsa_sparse_attn_ref()
// BIT-FOR-BIT on the CPU (test-dsa-sparse-attn-oracle), the order of reduction being part of the
// contract.
//
//   indices        int32 [n_q][topk]        flat KV row = page*page_size + offset; -1 = invalid slot
//   topk_length    int32 [n_q]  (nullable)  per-query count of live slots in `indices`; slot j is live
//                                           iff j < topk_length[q]  (null => topk)
//   attn_sink      float [n_head] (nullable) per-head logit that joins the softmax denominator only
//                                           (it contributes no V row)
//   extra_indices  int32 [n_q][extra_topk] (nullable), extra_topk_length int32 [n_q] (nullable)
//                                           second index set (e.g. sliding-window rows), same rules;
//                                           reduced AFTER the main set
//   scale          applied to q.k
//
// A slot is valid iff live AND index != -1. Any other out-of-range index (< -1 or >= rows) is a
// contract violation (the reference aborts via the return value false).
// Reduction order (normative): scores in slot order (main slots, then extra slots, invalid skipped);
// m = max(scores, sink); p_j = expf(s_j - m); denom = (sum_j p_j in slot order) + expf(sink - m);
// out = (sum_j p_j * v_j in slot order) / denom, accumulated per output element in float.
// A query-head with no valid slot outputs zeros (also when only the sink is present).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_dsa_sparse_attn_args {
    int64_t n_q = 0, n_head = 0, d_qk = 0, d_v = 0;
    int64_t page_size = 1, n_pages = 0;          // KV rows = n_pages*page_size
    const float *   q = nullptr;                 // [n_q][n_head][d_qk]
    const float *   k = nullptr;                 // [rows][d_qk]
    const float *   v = nullptr;                 // [rows][d_v]
    int64_t         topk = 0;
    const int32_t * indices = nullptr;           // [n_q][topk]
    const int32_t * topk_length = nullptr;       // [n_q] or null
    const float *   attn_sink = nullptr;         // [n_head] or null
    int64_t         extra_topk = 0;
    const int32_t * extra_indices = nullptr;     // [n_q][extra_topk] or null
    const int32_t * extra_topk_length = nullptr; // [n_q] or null
    float           scale = 1.0f;
    float *         out = nullptr;               // [n_q][n_head][d_v]
};

#if defined(__GNUC__)
#define LLAMA_DSA_NOINLINE __attribute__((noinline))
#else
#define LLAMA_DSA_NOINLINE
#endif

// shared scalar dot: one definition, so every path uses identical arithmetic
static LLAMA_DSA_NOINLINE float llama_dsa_dot(const float * a, const float * b, int64_t n) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) { s += a[i] * b[i]; }
    return s;
}

// returns the valid-slot row list (in contract order) for query iq, or false on contract violation
static inline bool llama_dsa_collect_rows(const llama_dsa_sparse_attn_args & a, int64_t iq, std::vector<int32_t> & rows) {
    const int64_t n_rows = a.page_size * a.n_pages;
    rows.clear();
    auto add = [&](const int32_t * idx, int64_t topk, const int32_t * len) {
        if (!idx) { return true; }
        const int64_t live = len ? len[iq] : topk;
        for (int64_t j = 0; j < topk && j < live; ++j) {
            const int32_t r = idx[iq * topk + j];
            if (r == -1) { continue; }
            if (r < -1 || r >= n_rows) { return false; }
            rows.push_back(r);
        }
        return true;
    };
    return add(a.indices, a.topk, a.topk_length) && add(a.extra_indices, a.extra_topk, a.extra_topk_length);
}

// Reference: direct indexed gather (what a sparse kernel does).
static inline bool llama_dsa_sparse_attn_ref(const llama_dsa_sparse_attn_args & a) {
    std::vector<int32_t> rows;
    std::vector<float> s;
    for (int64_t iq = 0; iq < a.n_q; ++iq) {
        if (!llama_dsa_collect_rows(a, iq, rows)) { return false; }
        for (int64_t h = 0; h < a.n_head; ++h) {
            const float * qh = a.q + (iq * a.n_head + h) * a.d_qk;
            float * o = a.out + (iq * a.n_head + h) * a.d_v;
            s.assign(rows.size(), 0.0f);
            float m = -INFINITY;
            for (size_t j = 0; j < rows.size(); ++j) {
                s[j] = llama_dsa_dot(qh, a.k + (int64_t) rows[j] * a.d_qk, a.d_qk) * a.scale;
                m = std::fmax(m, s[j]);
            }
            const bool has_sink = a.attn_sink != nullptr;
            if (has_sink) { m = std::fmax(m, a.attn_sink[h]); }
            for (int64_t d = 0; d < a.d_v; ++d) { o[d] = 0.0f; }
            if (rows.empty()) { continue; }
            float denom = 0.0f;
            for (size_t j = 0; j < rows.size(); ++j) {
                const float p = std::exp(s[j] - m);
                denom += p;
                const float * vr = a.v + (int64_t) rows[j] * a.d_v;
                for (int64_t d = 0; d < a.d_v; ++d) { o[d] += p * vr[d]; }
            }
            if (has_sink) { denom += std::exp(a.attn_sink[h] - m); }
            for (int64_t d = 0; d < a.d_v; ++d) { o[d] /= denom; }
        }
    }
    return true;
}

// Oracle A: gather the selected rows into dense K/V, build an additive mask (-inf for invalid slots,
// padded to a fixed slot count), run plain dense masked attention over ALL slots. Same slot order.
static inline bool llama_dsa_dense_masked_oracle_gathered(const llama_dsa_sparse_attn_args & a) {
    const int64_t n_slots = a.topk + (a.extra_indices ? a.extra_topk : 0);
    std::vector<float> K(n_slots * a.d_qk), V(n_slots * a.d_v), mask(n_slots), s(n_slots);
    for (int64_t iq = 0; iq < a.n_q; ++iq) {
        int64_t slot = 0;
        auto put = [&](const int32_t * idx, int64_t topk, const int32_t * len) {
            if (!idx) { return true; }
            const int64_t live = len ? len[iq] : topk;
            for (int64_t j = 0; j < topk; ++j, ++slot) {
                const int32_t r = idx[iq * topk + j];
                const bool ok = j < live && r != -1;
                if (ok && (r < -1 || r >= a.page_size * a.n_pages)) { return false; }
                mask[slot] = ok ? 0.0f : -INFINITY;
                for (int64_t d = 0; d < a.d_qk; ++d) { K[slot * a.d_qk + d] = ok ? a.k[(int64_t) r * a.d_qk + d] : 0.0f; }
                for (int64_t d = 0; d < a.d_v;  ++d) { V[slot * a.d_v  + d] = ok ? a.v[(int64_t) r * a.d_v  + d] : 0.0f; }
            }
            return true;
        };
        if (!put(a.indices, a.topk, a.topk_length) || !put(a.extra_indices, a.extra_topk, a.extra_topk_length)) { return false; }
        for (int64_t h = 0; h < a.n_head; ++h) {
            const float * qh = a.q + (iq * a.n_head + h) * a.d_qk;
            float * o = a.out + (iq * a.n_head + h) * a.d_v;
            float m = -INFINITY;
            bool any = false;
            for (int64_t j = 0; j < n_slots; ++j) {
                s[j] = std::isinf(mask[j]) ? -INFINITY : llama_dsa_dot(qh, &K[j * a.d_qk], a.d_qk) * a.scale + mask[j];
                if (!std::isinf(mask[j])) { any = true; }
                m = std::fmax(m, s[j]);
            }
            for (int64_t d = 0; d < a.d_v; ++d) { o[d] = 0.0f; }
            if (!any) { continue; }
            if (a.attn_sink) { m = std::fmax(m, a.attn_sink[h]); }
            float denom = 0.0f;
            for (int64_t j = 0; j < n_slots; ++j) {
                const float p = std::exp(s[j] - m); // exp(-inf) == 0 exactly for masked slots
                denom += p;
                for (int64_t d = 0; d < a.d_v; ++d) { o[d] += p * V[j * a.d_v + d]; }
            }
            if (a.attn_sink) { denom += std::exp(a.attn_sink[h] - m); }
            for (int64_t d = 0; d < a.d_v; ++d) { o[d] /= denom; }
        }
    }
    return true;
}

// Oracle B: dense masked attention over the WHOLE KV (mask = selected rows). Reduction runs in
// ascending row order, so it equals the contract only when the valid rows (main then extra) are
// strictly ascending; the test generates such index sets for this oracle.
static inline bool llama_dsa_dense_masked_oracle_full_kv(const llama_dsa_sparse_attn_args & a) {
    const int64_t n_rows = a.page_size * a.n_pages;
    std::vector<float> s(n_rows);
    std::vector<int32_t> rows;
    std::vector<char> sel(n_rows);
    for (int64_t iq = 0; iq < a.n_q; ++iq) {
        if (!llama_dsa_collect_rows(a, iq, rows)) { return false; }
        std::fill(sel.begin(), sel.end(), 0);
        for (int32_t r : rows) { sel[r] = 1; }
        for (int64_t h = 0; h < a.n_head; ++h) {
            const float * qh = a.q + (iq * a.n_head + h) * a.d_qk;
            float * o = a.out + (iq * a.n_head + h) * a.d_v;
            float m = -INFINITY;
            for (int64_t r = 0; r < n_rows; ++r) {
                s[r] = sel[r] ? llama_dsa_dot(qh, a.k + r * a.d_qk, a.d_qk) * a.scale : -INFINITY;
                m = std::fmax(m, s[r]);
            }
            for (int64_t d = 0; d < a.d_v; ++d) { o[d] = 0.0f; }
            if (rows.empty()) { continue; }
            if (a.attn_sink) { m = std::fmax(m, a.attn_sink[h]); }
            float denom = 0.0f;
            for (int64_t r = 0; r < n_rows; ++r) {
                const float p = std::exp(s[r] - m);
                denom += p;
                for (int64_t d = 0; d < a.d_v; ++d) { o[d] += p * a.v[r * a.d_v + d]; }
            }
            if (a.attn_sink) { denom += std::exp(a.attn_sink[h] - m); }
            for (int64_t d = 0; d < a.d_v; ++d) { o[d] /= denom; }
        }
    }
    return true;
}
