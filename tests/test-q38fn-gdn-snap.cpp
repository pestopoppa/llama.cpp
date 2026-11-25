// [Q38FN-GDN-SNAP] ggml_gated_delta_net_set_snap_dst (snapshots written into the cache view) vs the reference
// result + ggml_cpy into the same view: cache bytes and attention output bytes identical, for n_tokens == K,
// n_tokens < K, one and two sequences, nonzero kv_head, and the Q38FN layer shape. Exit 0 = pass.

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

// ---- gated delta net snapshot destination --------------------------------------------------------------

static void fill(ggml_tensor * t, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        p[i] = d(rng);
    }
}

static bool gdn_case(int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs, int64_t K, int64_t mem_size, int64_t kv_head) {
    ggml_context * ctx = new_ctx(512);
    const int64_t D = S_v * S_v * H;

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, n_seqs);
    ggml_tensor * g    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H, n_tokens, n_seqs);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H, n_tokens, n_seqs);
    ggml_tensor * s    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H, n_seqs);
    fill(q, -1, 1); fill(k, -0.2f, 0.2f); fill(v, -1, 1); fill(g, -0.5f, -0.01f); fill(beta, 0.1f, 0.9f); fill(s, -0.1f, 0.1f);

    // the recurrent cache: rows of D floats, mem_size rows per rollback group, K groups
    ggml_tensor * cache_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, mem_size * K);
    ggml_tensor * cache_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, mem_size * K);
    fill(cache_a, -9, 9);
    memcpy(cache_b->data, cache_a->data, ggml_nbytes(cache_a));

    const int64_t n_written = std::min(n_tokens, K);
    const size_t  row_size  = D * sizeof(float);
    auto cache_view = [&](ggml_tensor * c) {
        return ggml_view_3d(ctx, c, D, n_seqs, n_written, c->nb[1], (size_t) mem_size * row_size, (size_t) kv_head * row_size);
    };
    const int64_t attn_elems = S_v * H * n_tokens * n_seqs;

    // reference: result + ggml_cpy of the snapshot region (delta-net-base.cpp keep path)
    ggml_tensor * ra = ggml_gated_delta_net(ctx, q, k, v, g, beta, s, K);
    ggml_tensor * src = ggml_view_3d(ctx, ra, D, n_seqs, n_written,
            ggml_row_size(ra->type, D), ggml_row_size(ra->type, D * n_seqs), ggml_row_size(ra->type, attn_elems));
    ggml_tensor * cpy = ggml_cpy(ctx, src, cache_view(cache_a));

    // lever: the op writes the cache view itself
    ggml_tensor * rb = ggml_gated_delta_net(ctx, q, k, v, g, beta, s, K);
    ggml_gated_delta_net_set_snap_dst(rb, cache_view(cache_b));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cpy);
    ggml_build_forward_expand(gf, rb);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    const bool cache_ok = memcmp(cache_a->data, cache_b->data, ggml_nbytes(cache_a)) == 0;
    const bool attn_ok  = memcmp(ra->data, rb->data, attn_elems * sizeof(float)) == 0;
    // the lever must have written something (guard against a vacuous pass)
    bool changed = false;
    {
        const float * a = (const float *) ((const char *) cache_b->data + kv_head * row_size);
        const float * o = (const float *) ((const char *) ra->data) + attn_elems;
        changed = memcmp(a, o, row_size) == 0;   // slot 0, seq 0 equals the reference snapshot
    }
    const bool ok = cache_ok && attn_ok && changed;
    printf("  gdn S_v=%lld H=%lld T=%lld seqs=%lld K=%lld mem=%lld head=%lld: cache %s attn %s written %s\n",
           (long long) S_v, (long long) H, (long long) n_tokens, (long long) n_seqs, (long long) K, (long long) mem_size,
           (long long) kv_head, cache_ok ? "==" : "!=", attn_ok ? "==" : "!=", changed ? "yes" : "NO");
    ggml_free(ctx);
    return ok;
}

static bool test_gdn() {
    printf("gdn snap_dst\n");
    bool ok = true;
    ok &= gdn_case(128, 4, 5, 1, 5, 3, 1);   // MTP n-max 4 verify: n_tokens == K
    ok &= gdn_case(128, 4, 2, 1, 5, 3, 2);   // shorter verify: n_tokens < K, trailing slots untouched
    ok &= gdn_case(64,  6, 5, 2, 5, 4, 1);   // two sequences
    ok &= gdn_case(128, 48, 5, 1, 5, 2, 0);  // the Q38FN layer shape (S_v 128, 48 v heads)
    return ok;
}

int main() {
    const bool ok = test_gdn();
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
