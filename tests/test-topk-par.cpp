// [TAG_TOPK_PAR] GGML_CPU_TOPK_PAR=1: the row-split CPU TOP_K (Q38FN-INDEXER-TOPK-CPU).
//
// The QSA indexer calls ggml_top_k on a few long rows (n_tokens of a verify step x n_kv cells), so the
// default one-thread-per-row TOP_K leaves most threads idle. With the knob set, each row is split over
// nth/nr threads. This test runs that path on the indexer's shapes and checks it against a std::sort
// reference: the exact index set under (value desc, index asc), for distinct values and for the
// block-constant values of the indexer's expanded per-block scores (ties on the cut).
// Usage: test-topk-par [n_threads]

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int run_case(int64_t ncols, int64_t nrows, int k, int block, int nth, uint32_t seed) {
    ggml_init_params ip = { (size_t) (ncols * nrows * 8 + 64 * 1024 * 1024), nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-4.0f, 4.0f);
    float * ad = (float *) a->data;
    for (int64_t r = 0; r < nrows; ++r) {
        float bv = 0.0f;
        for (int64_t c = 0; c < ncols; ++c) {
            if (block <= 1 || c % block == 0) {
                bv = u(rng);
            }
            // a masked tail, as the indexer's -inf cells
            ad[r * ncols + c] = (c >= ncols - 37) ? -INFINITY : bv;
        }
    }

    ggml_tensor * out = ggml_top_k(ctx, a, k);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, nth) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }

    int bad = 0;
    for (int64_t r = 0; r < nrows; ++r) {
        const float * row = ad + r * ncols;
        const int32_t * od = (const int32_t *) ((const char *) out->data + r * out->nb[1]);
        std::vector<int32_t> got(od, od + k);
        std::vector<int32_t> ref(ncols);
        for (int64_t c = 0; c < ncols; ++c) {
            ref[c] = (int32_t) c;
        }
        std::stable_sort(ref.begin(), ref.end(), [row](int32_t x, int32_t y) { return row[x] > row[y]; });
        ref.resize(k);

        std::vector<int32_t> gs = got;
        std::sort(gs.begin(), gs.end());
        if (std::adjacent_find(gs.begin(), gs.end()) != gs.end() || gs.front() < 0 || gs.back() >= ncols) {
            fprintf(stderr, "row %lld: duplicate or out-of-range index\n", (long long) r);
            bad++;
            continue;
        }
        // (value desc, index asc) is a total order, so ties have one answer: stable_sort's
        std::vector<int32_t> rs = ref;
        std::sort(rs.begin(), rs.end());
        if (rs != gs) {
            fprintf(stderr, "row %lld: index set differs from the reference\n", (long long) r);
            bad++;
        }
    }
    ggml_free(ctx);
    printf("  ncols=%lld nrows=%lld k=%d block=%d nth=%d: %s\n", (long long) ncols, (long long) nrows, k, block, nth,
           bad ? "FAIL" : "OK");
    return bad;
}

int main(int argc, char ** argv) {
    setenv("GGML_CPU_TOPK_PAR", "1", 1); // read once, on the first TOP_K
    const int nth = argc > 1 ? atoi(argv[1]) : 8;
    int bad = 0;
    // QSA indexer shapes: width = indexer_top_k + ratio - 1, n_kv 8k..64k, 1..5 rows (verify width)
    for (int64_t n : { 8192, 32768, 65536 }) {
        for (int64_t nr : { 1, 3, 5 }) {
            bad += run_case(n, nr, 2051, 1, nth, (uint32_t) (n + nr));
            bad += run_case(n, nr, 2051, 4, nth, (uint32_t) (n * 3 + nr));
        }
    }
    bad += run_case(5000, 2, 4999, 1, nth, 7);     // k close to ncols
    bad += run_case(40000, 7, 64, 1, nth, 11);     // nr >= nth on small pools falls back
    printf("%s\n", bad ? "FAILED" : "ALL OK");
    return bad ? 1 : 0;
}
