// [TAG_TOPK_EXACT] GGML_CPU_TOPK_EXACT=1: the row-split CPU TOP_K that must return EXACTLY the serial
// path's output (Q38FN-INDEXER-TOPK-EXACT). The reference is the serial code itself: std::partial_sort
// with the same comparator over 0..n-1, then the [0]<->[1] swap. Output is compared element by element
// (set AND order) over heavy-tie inputs: block-constant scores (the QSA indexer's expanded per-block
// scores), tiny value alphabets, all-equal rows, monotone rows (every index enters the heap), -inf tails.
// Usage: test-topk-exact [n_threads] [dump.f32 ncols nrows k]...

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static std::vector<int32_t> serial_ref(const float * row, int64_t n, int k) {
    std::vector<int32_t> t(n);
    for (int64_t j = 0; j < n; ++j) t[j] = (int32_t) j;
    std::partial_sort(t.begin(), t.begin() + k, t.end(), [row](int32_t a, int32_t b) { return row[a] > row[b]; });
    t.resize(k);
    if (k > 1) std::swap(t[0], t[1]);
    return t;
}

static int check(const std::vector<float> & data, int64_t ncols, int64_t nrows, int k, int nth, const char * tag) {
    ggml_init_params ip = { (size_t) (ncols * nrows * 8 + 64 * 1024 * 1024), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    std::copy(data.begin(), data.end(), (float *) a->data);
    ggml_tensor * out = ggml_top_k(ctx, a, k);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, nth) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }
    int bad = 0;
    for (int64_t r = 0; r < nrows; ++r) {
        const int32_t * od = (const int32_t *) ((const char *) out->data + r * out->nb[1]);
        std::vector<int32_t> got(od, od + k);
        if (got != serial_ref(data.data() + r * ncols, ncols, k)) bad++;
    }
    ggml_free(ctx);
    printf("  %-14s ncols=%lld nrows=%lld k=%d nth=%d: %s\n", tag, (long long) ncols, (long long) nrows, k, nth,
           bad ? "FAIL" : "OK");
    return bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    setenv("GGML_CPU_TOPK_EXACT", "1", 1); // read once, on the first TOP_K
    const int nth = argc > 1 ? atoi(argv[1]) : 8;
    int bad = 0, cases = 0;
    std::mt19937 rng(12345);
    for (int it = 0; it < 120; ++it) {
        const int64_t ncols = std::vector<int64_t>{4096, 8192, 20000, 32768, 55000, 65536}[it % 6];
        const int64_t nrows = 1 + (it / 6) % 5;
        const int k = std::vector<int>{2051, 2051, 1, 2, 64, 1000, 2051, 5000}[(it / 3) % 8];
        const int mode = it % 7;
        const int block = std::vector<int>{1, 4, 16, 64}[(it / 2) % 4];
        std::vector<float> d(ncols * nrows);
        std::uniform_real_distribution<float> u(-4.0f, 4.0f);
        std::uniform_int_distribution<int> small(0, 3);
        for (int64_t r = 0; r < nrows; ++r) {
            float bv = 0.0f;
            for (int64_t c = 0; c < ncols; ++c) {
                if (c % block == 0) {
                    switch (mode) {
                        case 0: bv = u(rng); break;                                   // block-constant
                        case 1: bv = (float) small(rng); break;                       // 4-value alphabet
                        case 2: bv = 1.0f; break;                                     // all equal
                        case 3: bv = (float) (c / block); break;                      // increasing: all enter
                        case 4: bv = (float) (-(c / block)); break;                   // decreasing
                        case 5: bv = std::round(u(rng) * 2.0f) + 0.001f * (c / 4096); break; // drift + ties
                        default: bv = (float) small(rng) + (float) (c / block % 3); break;
                    }
                }
                d[r * ncols + c] = (mode != 2 && c >= ncols - 37) ? -INFINITY : bv;
            }
        }
        char tag[32];
        snprintf(tag, sizeof tag, "m%d/b%d", mode, block);
        bad += check(d, ncols, nrows, std::min<int>(k, (int) ncols - 1), nth, tag);
        cases++;
    }
    for (int i = 2; i + 3 < argc; i += 4) {
        FILE * f = fopen(argv[i], "rb");
        const int64_t ncols = atoll(argv[i + 1]), nrows = atoll(argv[i + 2]);
        const int k = atoi(argv[i + 3]);
        std::vector<float> d(ncols * nrows);
        if (!f || fread(d.data(), sizeof(float), d.size(), f) != d.size()) { fprintf(stderr, "bad dump %s\n", argv[i]); return 2; }
        fclose(f);
        bad += check(d, ncols, nrows, k, nth, "dump");
        cases++;
    }
    printf("%d/%d cases exact; %s\n", cases - bad, cases, bad ? "FAILED" : "ALL OK");
    return bad ? 1 : 0;
}
