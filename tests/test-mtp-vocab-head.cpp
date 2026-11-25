// Q38FN-GEMV2 DRAFTVOCAB (src/llama-mtp-vocab-ops.h): MTP draft head restricted to a token subset.
//  For Q6_K and Q8_0 heads, n = 1 and n = 5 columns, a random K-of-V subset (plus K = V):
//   - every kept row's logit is BIT-EQUAL to the full head's logit (same ggml_mul_mat kernel, rows copied verbatim),
//     compared against the full head at the same n,
//   - every other row is -inf,
//   - when the full head's argmax is in the subset, the subset head picks the same token.
// Usage: test-mtp-vocab-head [n_threads]

#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-mtp-vocab-ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_nth = 4;

static int run(ggml_type type, int64_t d, int64_t V, int64_t K, int64_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    ggml_init_params ip = { (size_t) 512 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::vector<float> wf(d * V), hf(d * n);
    for (auto & x : wf) x = nd(rng);
    for (auto & x : hf) x = nd(rng);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, type, d, V);
    ggml_quantize_chunk(type, wf.data(), w->data, 0, V, d, nullptr);
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);
    std::memcpy(h->data, hf.data(), hf.size() * sizeof(float));

    std::vector<int32_t> all(V);
    for (int64_t i = 0; i < V; ++i) all[i] = (int32_t) i;
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<int32_t> ids(all.begin(), all.begin() + K);
    std::sort(ids.begin(), ids.end());

    ggml_tensor * ws = ggml_new_tensor_2d(ctx, type, d, K);
    const size_t rs = ggml_row_size(type, d);
    for (int64_t j = 0; j < K; ++j) {
        std::memcpy((char *) ws->data + j * ws->nb[1], (const char *) w->data + ids[j] * w->nb[1], rs);
    }
    ggml_tensor * ix = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K);
    std::memcpy(ix->data, ids.data(), K * sizeof(int32_t));

    ggml_tensor * full = ggml_mul_mat(ctx, w, h);
    ggml_tensor * sub  = mtp_vocab::build_head(ctx, ws, ix, h, V);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, sub);
    if (ggml_graph_compute_with_ctx(ctx, gf, g_nth) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }
    std::vector<char> keep(V, 0);
    for (int32_t i : ids) keep[i] = 1;
    int64_t bad = 0, argmax_checked = 0;
    for (int64_t c = 0; c < n; ++c) {
        const float * f = (const float *) ((char *) full->data + c * full->nb[1]);
        const float * s = (const float *) ((char *) sub->data + c * sub->nb[1]);
        int64_t af = 0, as = 0;
        for (int64_t v = 0; v < V; ++v) {
            if (keep[v]) {
                if (std::memcmp(&f[v], &s[v], sizeof(float)) != 0) ++bad;
            } else if (!(std::isinf(s[v]) && s[v] < 0)) {
                ++bad;
            }
            if (f[v] > f[af]) af = v;
            if (s[v] > s[as]) as = v;
        }
        if (keep[af]) {
            ++argmax_checked;
            if (as != af) ++bad;
        }
    }
    printf("  %-5s d=%lld V=%lld K=%lld n=%lld: %s (bad=%lld, argmax-in-subset cols=%lld)\n", ggml_type_name(type),
           (long long) d, (long long) V, (long long) K, (long long) n, bad ? "FAIL" : "OK", (long long) bad,
           (long long) argmax_checked);
    ggml_free(ctx);
    return bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc > 1) g_nth = std::max(1, atoi(argv[1]));
    int fails = 0;
    for (ggml_type t : { GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 }) {
        for (int64_t n : { 1, 5 }) {
            fails += run(t, 512, 4096, 700, n, 1234u + (unsigned) n);
            fails += run(t, 512, 4096, 4096, n, 99u + (unsigned) n);
        }
    }
    printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
