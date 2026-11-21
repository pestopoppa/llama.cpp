// LSA-CPU-SPARSE: index-key write + decode selection + CPU flash attention.
//   1. select_topk == brute-force top-k (ties -> lower index, forced / excluded entries) on random cases
//   2. score_bf16 (AVX-512) agrees with a double reference
//   3. graph: ik write op -> decode mask ops -> FLASH_ATTN_EXT
//      a. budget covering every key: mask and FA output bitwise equal to the dense path
//      b. sparse budget: mask equals the brute-force selection, FA output bitwise equal to FA run on that mask
//   4. (optional) argv[1] / LLAMA_LSA_CPU_TEST_MAP: LSAMAP1 loads, W_Q == W_K
// exit 0 = PASS
#include "llama-lsa-cpu.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace llama_lsa_cpu;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++g_fail; } } while (0)

static std::vector<uint8_t> brute_select(const std::vector<float> & s, int top_k) {
    const int64_t n = s.size();
    std::vector<uint8_t> keep(n, 0);
    std::vector<int64_t> cand;
    for (int64_t j = 0; j < n; ++j) {
        if (std::isfinite(s[j])) cand.push_back(j);
        else if (s[j] > 0) keep[j] = 1;
    }
    std::sort(cand.begin(), cand.end(), [&](int64_t a, int64_t b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    for (int64_t i = 0; i < std::min<int64_t>(top_k, cand.size()); ++i) keep[cand[i]] = 1;
    return keep;
}

static void test_select(std::mt19937 & rng) {
    for (int rep = 0; rep < 200; ++rep) {
        const int64_t n = 1 + rng() % 20000;
        const int top_k = 1 + rng() % 5000;
        const int levels = rep % 3 == 0 ? 7 : 0; // coarse levels -> many exact ties
        std::uniform_real_distribution<float> u(-1.05f, 1.05f);
        std::vector<float> s(n);
        for (auto & x : s) {
            x = levels ? std::round(u(rng)*levels)/levels : u(rng);
            const unsigned r = rng() % 100;
            if (r < 5) x = INFINITY; else if (r < 15) x = -INFINITY;
        }
        std::vector<uint8_t> keep(n);
        const int64_t kept = select_topk(s.data(), n, top_k, keep.data());
        const auto ref = brute_select(s, top_k);
        CHECK(keep == ref, "select_topk != brute force (n %lld top_k %d rep %d)", (long long) n, top_k, rep);
        CHECK(kept == (int64_t) std::count(ref.begin(), ref.end(), 1), "kept count");
    }
}

static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }

static void test_score(std::mt19937 & rng) {
    const int64_t n = 1000;
    std::normal_distribution<float> nd;
    std::vector<uint16_t> ik(n*index_dim);
    for (auto & x : ik) x = bf16(nd(rng)*0.125f);
    float q[index_dim];
    for (auto & x : q) x = nd(rng)*0.125f;
    std::vector<float> out(n);
    score_bf16(ik.data(), 0, n, q, out.data());
    for (int64_t j = 0; j < n; ++j) {
        double r = 0;
        for (int o = 0; o < index_dim; ++o) r += (double) bf2f(ik[j*index_dim + o])*q[o];
        CHECK(std::fabs(r - out[j]) < 1e-5, "score_bf16 %lld: %f vs %f", (long long) j, out[j], r);
    }
}

// geometry of Qwen3.8-27B global layers
constexpr int QH = 24, KVH = 4, D = 256, ROPE = 64, ROWS = KVH*(D - ROPE);

static void test_graph(std::mt19937 & rng, int n_kv, int top_k, int window, int sink, bool expect_dense, int n_threads) {
    std::normal_distribution<float> nd;
    std::vector<float> wq(ROWS*index_dim), wk;
    for (auto & x : wq) x = nd(rng)*0.05f;
    wk = wq; // W_Q == W_K as in the released PCA map

    layer_state st;
    st.il = 3; st.q_heads = QH; st.kv_heads = KVH; st.head_dim = D; st.rope_dim = ROPE;
    st.wq = wq.data(); st.wk = wk.data();
    st.ik_pos.assign(n_kv, -1); st.score.assign(n_kv, 0.0f);
    st.top_k = top_k; st.window = window; st.sink = sink;

    ggml_init_params ip = { (size_t) 1024*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    st.ik = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, index_dim, n_kv);
    ggml_tensor * kf32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, KVH, n_kv);
    ggml_tensor * idx  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv);
    ggml_tensor * kc   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, KVH);
    ggml_tensor * vc   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, KVH);
    ggml_tensor * q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, QH, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1);
    ggml_tensor * qi1  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1); // "written" cell of the query token

    for (int t = 0; t < n_kv; ++t) {
        ((int64_t *) idx->data)[t] = t;
        for (int g = 0; g < KVH; ++g) for (int d = 0; d < D; ++d) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(nd(rng));
            ((float *) kf32->data)[((size_t) t*KVH + g)*D + d] = ggml_fp16_to_fp32(h);
            ((ggml_fp16_t *) kc->data)[((size_t) g*n_kv + t)*D + d] = h;
            ((ggml_fp16_t *) vc->data)[((size_t) g*n_kv + t)*D + d] = ggml_fp32_to_fp16(nd(rng));
        }
    }
    for (int i = 0; i < D*QH; ++i) ((float *) q->data)[i] = nd(rng);
    // causal single-sequence mask with a few holes (other-sequence / empty cells)
    for (int64_t r = 0; r < mask->ne[1]; ++r) for (int j = 0; j < n_kv; ++j) {
        const bool hole = r == 0 && j % 997 == 500;
        ((ggml_fp16_t *) mask->data)[r*n_kv + j] = ggml_fp32_to_fp16(r == 0 && !hole ? 0.0f : -INFINITY);
    }
    ((int64_t *) qi1->data)[0] = n_kv - 1;

    // 1) write index keys for every cell (prefill-shaped ubatch)
    ggml_tensor * written = build_ik_write(ctx, &st, kf32, idx);
    ggml_cgraph * g1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g1, written);
    ggml_graph_compute_with_ctx(ctx, g1, n_threads);
    for (int t = 0; t < n_kv; ++t) CHECK(st.ik_pos[t] == t, "ik_pos %d", t);

    // 2) decode mask + FA, and dense FA
    ggml_tensor * m2 = build_decode_mask(ctx, &st, mask, q, qi1);
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    const float scale = 1.0f/16.0f;
    ggml_tensor * fa_sparse = ggml_flash_attn_ext(ctx, qp, kc, vc, m2,   scale, 0.0f, 0.0f);
    ggml_tensor * fa_dense  = ggml_flash_attn_ext(ctx, qp, kc, vc, mask, scale, 0.0f, 0.0f);
    ggml_cgraph * g2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g2, fa_sparse);
    ggml_build_forward_expand(g2, fa_dense);
    ggml_graph_compute_with_ctx(ctx, g2, n_threads);

    // brute-force selection from the same qhat / index keys
    float qhat[index_dim];
    project_query(st, (const char *) q->data, q->nb[0], q->nb[1], qhat);
    std::vector<float> s(n_kv);
    score_bf16((const uint16_t *) st.ik->data, 0, n_kv, qhat, s.data());
    const int qpos = n_kv - 1;
    for (int j = 0; j < n_kv; ++j) {
        if (ggml_fp16_to_fp32(((ggml_fp16_t *) mask->data)[j]) == -INFINITY) s[j] = -INFINITY;
        else if (j < sink || j > qpos - window) s[j] = INFINITY;
    }
    const auto ref = brute_select(s, top_k);
    int64_t n_sel = 0, mism = 0;
    for (int j = 0; j < n_kv; ++j) {
        const bool kept = ggml_fp16_to_fp32(((ggml_fp16_t *) m2->data)[j]) != -INFINITY;
        n_sel += kept;
        mism += kept != (bool) ref[j];
    }
    CHECK(mism == 0, "mask != brute-force selection: %lld mismatches (n_kv %d top_k %d)", (long long) mism, n_kv, top_k);

    // FA on the brute-force mask
    ggml_tensor * mref = ggml_dup_tensor(ctx, mask);
    memcpy(mref->data, mask->data, ggml_nbytes(mask));
    for (int j = 0; j < n_kv; ++j) if (!ref[j]) ((ggml_fp16_t *) mref->data)[j] = ggml_fp32_to_fp16(-INFINITY);
    ggml_tensor * fa_ref = ggml_flash_attn_ext(ctx, qp, kc, vc, mref, scale, 0.0f, 0.0f);
    ggml_cgraph * g3 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g3, fa_ref);
    ggml_graph_compute_with_ctx(ctx, g3, n_threads);

    const size_t nb = ggml_nbytes(fa_sparse);
    CHECK(memcmp(fa_sparse->data, fa_ref->data, nb) == 0, "sparse FA != FA on brute-force mask");
    const bool same_dense = memcmp(fa_sparse->data, fa_dense->data, nb) == 0;
    if (expect_dense) {
        CHECK(memcmp(m2->data, mask->data, ggml_nbytes(mask)) == 0, "all-keys mask differs from dense mask");
        CHECK(same_dense, "all-keys FA != dense FA (bitwise)");
    } else {
        double md = 0;
        for (size_t i = 0; i < nb/4; ++i) md = std::max(md, (double) std::fabs(((float *) fa_sparse->data)[i] - ((float *) fa_dense->data)[i]));
        std::printf("  sparse n_kv %d kept %lld (%.1f%%), max|sparse-dense| %.3g (random data, informational)\n",
                n_kv, (long long) n_sel, 100.0*n_sel/n_kv, md);
    }
    std::printf("  graph n_kv %d top_k %d window %d sink %d threads %d: kept %lld, %s\n", n_kv, top_k, window, sink, n_threads,
            (long long) n_sel, g_fail ? "FAIL so far" : "ok");
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    std::mt19937 rng(20261120);
    test_select(rng);
    std::printf("select_topk vs brute force: %s\n", g_fail ? "FAIL" : "ok");
    test_score(rng);
    std::printf("score_bf16 vs double: %s\n", g_fail ? "FAIL" : "ok");
    for (int nt : {1, 4}) {
        test_graph(rng, 3000, 100000, 512, 4, true,  nt);   // all keys
        test_graph(rng, 9000, 1024,   128, 4, false, nt);   // sparse
    }

    const char * map = argc > 1 ? argv[1] : getenv("LLAMA_LSA_CPU_TEST_MAP");
    if (map && *map) {
        lsamap m;
        std::string err;
        const bool ok = load_lsamap(map, m, err);
        CHECK(ok, "load_lsamap(%s): %s", map, err.c_str());
        if (ok) {
            CHECK(m.rows() == ROWS && m.q_heads == QH && m.kv_heads == KVH && m.head_dim == D && m.rope_dim == ROPE, "map geometry");
            int same = 0;
            for (auto & [l, w] : m.layers) same += w.first == w.second;
            std::printf("map %s: %zu layers, W_Q == W_K on %d, identity '%s'\n", map, m.layers.size(), same, m.map_identity.c_str());
        }
    }
    std::printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
