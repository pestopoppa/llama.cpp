// test-fa-rowexact.cpp: [AKX-ALL-LB4] CPU FLASH_ATTN_EXT row-exactness, N=1 vs N=5.
//
// A verify batch of N=5 query rows must give, for every row r, the same bytes as a decode step
// (N=1) of row r alone against the same K/V and that row's mask. The CPU backend may run a
// different kernel path per N (ggml: split-KV only at N=1 for f16 KV at n_kv >= 512; iqk: the GQA
// split-KV path only at N=1), so this is a measurement of the build under test, not a property
// every build has: with GGML_FA_SPLIT_KV unset the ggml kernel is EXPECTED to differ at n_kv >= 512.
//
// Shape: the Q38FN-class CPU long-context attention (D 256, 2 KV heads, GQA 12). The mask is the
// verify-batch causal edge: row r sees cells [0, n_kv - N + r]. Inputs are deterministic.
//
// usage: test-fa-rowexact [n_kv ...]          (default 8192 65536)
//   env  FA_ROWEXACT_THREADS (default 8), FA_ROWEXACT_TYPES=f16,q8_0 (default both)
// Prints one RESULT line per (type, n_kv) and exits 0 iff every row is bit-identical.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr int64_t D      = 256;
static constexpr int64_t NH_KV  = 2;
static constexpr int64_t GQA    = 12;
static constexpr int64_t NQ     = 5;

static uint64_t g_state = 0x9E3779B97F4A7C15ull;
static float rnd(float lo, float hi) {
    g_state ^= g_state << 13; g_state ^= g_state >> 7; g_state ^= g_state << 17;
    const double u = (double) (g_state >> 11) * 0x1.0p-53;
    return (float) (lo + (hi - lo) * u);
}

static void fill_kv(ggml_tensor * t, int64_t n_kv) {
    std::vector<float> src(D * n_kv * NH_KV);
    for (auto & x : src) { x = rnd(-1.0f, 1.0f); }
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) t->data, (int64_t) src.size());
    } else {
        ggml_quantize_chunk(t->type, src.data(), t->data, 0, n_kv * NH_KV, D, nullptr);
    }
}

// mask row r: cells [0, n_kv - NQ + r] visible (0), the rest -inf
static void fill_mask_row(ggml_fp16_t * row, int64_t n_kv, int64_t r) {
    const int64_t last = n_kv - NQ + r;
    for (int64_t j = 0; j < n_kv; ++j) {
        row[j] = ggml_fp32_to_fp16(j <= last ? 0.0f : -INFINITY);
    }
}

static int run_case(ggml_type type_kv, int64_t n_kv, int n_threads) {
    const size_t kv_bytes = ggml_row_size(type_kv, D) * n_kv * NH_KV;
    const size_t mem = 2*kv_bytes + (size_t) (NQ + 1) * n_kv * sizeof(ggml_fp16_t) * 2 +
                       (size_t) 64 * D * GQA * NH_KV * NQ * sizeof(float) + (size_t) 256 * 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init(%zu) failed\n", mem); return 2; }

    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, NH_KV*GQA, 1);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, type_kv, D, n_kv, NH_KV, 1);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx, type_kv, D, n_kv, NH_KV, 1);
    ggml_tensor * m5 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, NQ, 1, 1);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) { ((float *) q->data)[i] = rnd(-1.0f, 1.0f); }
    fill_kv(k, n_kv);
    fill_kv(v, n_kv);
    for (int64_t r = 0; r < NQ; ++r) {
        fill_mask_row((ggml_fp16_t *) ((char *) m5->data + r*m5->nb[1]), n_kv, r);
    }

    const float scale = 1.0f / sqrtf((float) D);
    ggml_tensor * out5 = ggml_flash_attn_ext(ctx, q, k, v, m5, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out5, GGML_PREC_F32);
    ggml_cgraph * g5 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g5, out5);
    if (ggml_graph_compute_with_ctx(ctx, g5, n_threads) != GGML_STATUS_SUCCESS) { ggml_free(ctx); return 2; }

    int     n_diff_rows = 0;
    double  max_abs     = 0.0;
    bool    finite      = true;
    for (int64_t r = 0; r < NQ; ++r) {
        ggml_tensor * qr = ggml_view_4d(ctx, q, D, 1, NH_KV*GQA, 1, q->nb[1], q->nb[2], q->nb[3], r*q->nb[1]);
        ggml_tensor * mr = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, 1, 1, 1);
        memcpy(mr->data, (char *) m5->data + r*m5->nb[1], n_kv*sizeof(ggml_fp16_t));
        ggml_tensor * out1 = ggml_flash_attn_ext(ctx, qr, k, v, mr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out1, GGML_PREC_F32);
        ggml_cgraph * g1 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g1, out1);
        if (ggml_graph_compute_with_ctx(ctx, g1, n_threads) != GGML_STATUS_SUCCESS) { ggml_free(ctx); return 2; }

        // out5: [D, n_head, NQ, 1] -> row r is the contiguous block at r*nb[2]; out1: [D, n_head, 1, 1]
        const float * a = (const float *) ((const char *) out5->data + r*out5->nb[2]);
        const float * b = (const float *) out1->data;
        const int64_t n = D * NH_KV * GQA;
        const bool same = memcmp(a, b, n*sizeof(float)) == 0;
        double row_max = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { finite = false; }
            row_max = std::max(row_max, (double) fabsf(a[i] - b[i]));
        }
        max_abs = std::max(max_abs, row_max);
        if (!same) { n_diff_rows++; }
        printf("  type=%s n_kv=%lld row=%lld bitexact=%d max_abs_diff=%.3e\n", ggml_type_name(type_kv),
               (long long) n_kv, (long long) r, (int) same, row_max);
    }
    ggml_free(ctx);

    printf("RESULT type=%s n_kv=%lld D=%lld kv_heads=%lld gqa=%lld N=%lld rows_differing=%d max_abs_diff=%.3e finite=%d verdict=%s\n",
           ggml_type_name(type_kv), (long long) n_kv, (long long) D, (long long) NH_KV, (long long) GQA, (long long) NQ,
           n_diff_rows, max_abs, (int) finite, (n_diff_rows == 0 && finite) ? "ROW_EXACT" : "NOT_ROW_EXACT");
    return (n_diff_rows == 0 && finite) ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::vector<int64_t> kvs;
    for (int i = 1; i < argc; ++i) { kvs.push_back(atoll(argv[i])); }
    if (kvs.empty()) { kvs = { 8192, 65536 }; }

    int n_threads = 8;
    if (const char * e = getenv("FA_ROWEXACT_THREADS")) { n_threads = atoi(e); }
    std::string types = "f16,q8_0";
    if (const char * e = getenv("FA_ROWEXACT_TYPES")) { types = e; }

    printf("test-fa-rowexact: threads=%d GGML_IQK_FA=%s GGML_FA_SPLIT_KV=%s\n", n_threads,
           getenv("GGML_IQK_FA") ? getenv("GGML_IQK_FA") : "(unset)",
           getenv("GGML_FA_SPLIT_KV") ? getenv("GGML_FA_SPLIT_KV") : "(unset)");

    int rc = 0;
    for (const char * tname : { "f16", "q8_0" }) {
        if (types.find(tname) == std::string::npos) { continue; }
        const ggml_type t = std::string(tname) == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        for (int64_t kv : kvs) {
            if (kv % 32 != 0 || kv < NQ) { fprintf(stderr, "n_kv %lld must be a multiple of 32\n", (long long) kv); return 2; }
            const int r = run_case(t, kv, n_threads);
            if (r == 2) { return 2; }
            rc |= r;
        }
    }
    return rc;
}
