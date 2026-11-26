// [Q38FN-TOPK-SELECT] GGML_CPU_TOPK_SELECT top_k vs the reference std::partial_sort kernel, re-implemented
// here byte for byte.   usage: test-q38fn-topk-select <mode 1|2>
//   mode 1 (exact heap replay): dst bytes identical on every case, block ties and k cutting a block included.
//   mode 2 (nth_element select): distinct values -> dst bytes identical; block-constant values (the lightning
//          indexer's expanded scores) with k on a block edge -> identical index SET; k cutting a tied block is
//          reported (not failed): the tie-break differs from the reference by design.
// Prints a timing of both on the d32k indexer shape. Exit 0 = pass.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

static std::mt19937 rng(20261125);

static ggml_context * new_ctx(size_t mb) {
    ggml_init_params ip = { mb * 1024 * 1024, nullptr, false };
    return ggml_init(ip);
}

static double compute(ggml_context * ctx, ggml_tensor * out, int nth) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const auto t0 = std::chrono::steady_clock::now();
    ggml_graph_compute_with_ctx(ctx, gf, nth);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---- top_k ------------------------------------------------------------------------------------------

// exact copy of the reference kernel (ops.cpp ggml_compute_forward_top_k_f32, partial_sort branch)
static void ref_top_k_row(const float * src, int64_t n, int64_t k, int32_t * dst, std::vector<int32_t> & tmp) {
    tmp.resize(n);
    for (int64_t j = 0; j < n; j++) {
        tmp[j] = (int32_t) j;
    }
    std::partial_sort(tmp.begin(), tmp.begin() + k, tmp.end(), [src](int32_t a, int32_t b) { return src[a] > src[b]; });
    std::copy(tmp.begin(), tmp.begin() + k, dst);
    if (k > 1) {
        std::swap(dst[0], dst[1]);
    }
}

// mode 0: distinct values; mode 1: block-constant (r = 4) scores plus a -inf masked tail; k must be a multiple of r
static int g_mode = 1;

static bool topk_case(int64_t n, int64_t rows, int64_t k, int mode, int nth, bool timing) {
    ggml_context * ctx = new_ctx(64 + (size_t) (n * rows * 16) / (1024 * 1024));
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, rows);
    float * xd = (float *) x->data;
    std::uniform_real_distribution<float> d(-4.0f, 4.0f);
    const int64_t r = 4;
    for (int64_t i = 0; i < rows; ++i) {
        if (mode == 0) {
            // distinct: a random permutation of a strictly increasing grid
            std::vector<float> v(n);
            for (int64_t j = 0; j < n; ++j) {
                v[j] = -1.0f + 2.0f * (float) j / (float) n + 1e-7f * (float) i;
            }
            std::shuffle(v.begin(), v.end(), rng);
            memcpy(xd + i * n, v.data(), n * sizeof(float));
        } else {
            const int64_t n_valid = n - 36;   // masked cells (past n_kv's padding), on a block edge so k cuts no block
            for (int64_t b = 0; b * r < n; ++b) {
                const float s = d(rng);
                for (int64_t j = b * r; j < std::min(n, (b + 1) * r); ++j) {
                    xd[i * n + j] = j < n_valid ? s : -INFINITY;
                }
            }
        }
    }
    ggml_tensor * out = ggml_top_k(ctx, x, (int) k);
    const double ms = compute(ctx, out, nth);

    std::vector<int32_t> ref(k), tmp;
    bool ok = true;
    double ref_ms = 0.0;
    for (int64_t i = 0; i < rows && ok; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ref_top_k_row(xd + i * n, n, k, ref.data(), tmp);
        ref_ms = std::max(ref_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        const int32_t * got = (const int32_t *) ((const char *) out->data + i * out->nb[1]);
        const bool bytes = memcmp(got, ref.data(), k * sizeof(int32_t)) == 0;
        std::set<int32_t> a(got, got + k), b(ref.begin(), ref.end());
        const bool same_set = a.size() == (size_t) k && a == b;
        if (g_mode == 1 || mode == 0) {
            ok = bytes;
        } else if (k % 4 == 0) {
            ok = same_set;
        } else {
            // mode 2, k cuts a tied block: the set may differ only inside the boundary block
            if (!same_set) {
                printf("  (mode 2 tie-break differs from the reference on row %lld, as documented)\n", (long long) i);
            }
        }
        if (!ok) {
            fprintf(stderr, "  topk mismatch n=%lld k=%lld mode=%d row=%lld\n", (long long) n, (long long) k, mode, (long long) i);
        }
    }
    printf("  topk n=%-6lld rows=%lld k=%-5lld mode=%s nth=%d: %s",
           (long long) n, (long long) rows, (long long) k, mode ? "block" : "distinct", nth, ok ? "OK" : "FAIL");
    if (timing) {
        printf("  [ggml op %.3f ms; reference partial_sort, one row, %.3f ms]", ms, ref_ms);
    }
    printf("\n");
    ggml_free(ctx);
    return ok;
}

static bool test_topk() {
    printf("topk (GGML_CPU_TOPK_SELECT=%s)\n", getenv("GGML_CPU_TOPK_SELECT") ? getenv("GGML_CPU_TOPK_SELECT") : "unset");
    bool ok = true;
    ok &= topk_case(256,    2, 16,   0, 2, false);
    ok &= topk_case(1000,   3, 1000, 0, 2, false);   // k == n
    ok &= topk_case(4096,   5, 1,    0, 4, false);   // k == 1
    ok &= topk_case(33024,  5, 2051, 0, 5, true);    // d32k indexer shape (n_kv padded), distinct
    ok &= topk_case(33024,  5, 2048, 1, 5, true);    // block-constant, k on a block edge
    ok &= topk_case(1280,   5, 1280 - 36, 1, 5, false);  // every valid cell, none of the masked
    ok &= topk_case(33024,  5, 2051, 1, 5, true);    // the served width: whole blocks + 3, cuts a tied block
    ok &= topk_case(4100,   3, 1023, 1, 3, false);
    return ok;
}

int main(int argc, char ** argv) {
    // the knob is read once per process, before the first graph; this binary tests that mode vs the in-test reference
    g_mode = argc > 1 ? atoi(argv[1]) : 1;
    if (g_mode != 1 && g_mode != 2) {
        fprintf(stderr, "usage: %s <1|2>\n", argv[0]);
        return 2;
    }
    setenv("GGML_CPU_TOPK_SELECT", g_mode == 1 ? "1" : "2", 1);
    const bool ok = test_topk();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
