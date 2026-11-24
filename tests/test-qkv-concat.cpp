// AK-BW-27B-ATTN-KV-CONCAT arm (b) unit test (CPU backend).
// 1. The weight allocator (ggml_backend_alloc_ctx_tensors) places wq, wk, wv created back to back in one context
//    contiguously, so a metadata-only tensor aliasing wq's data with nq + nk + nv rows IS the row concatenation
//    (the precondition llama_model_base::bind_qkv_concat checks at load).
// 2. One MUL_MAT over that alias, split into Q / K / V views, is bit-identical to three separate MUL_MATs.
// Shapes: the 27B full-attention projections scaled down in K (Q8_0, nq = 12288 -> 768, nk = nv = 1024 -> 64, K 512).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int run(int64_t K, int64_t nq, int64_t nk, int64_t nv, int64_t n_tok) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * wq = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, K, nq);
    ggml_tensor * wk = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, K, nk);
    ggml_tensor * wv = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, K, nv);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if ((char *) wk->data != (char *) wq->data + ggml_nbytes(wq) || (char *) wv->data != (char *) wk->data + ggml_nbytes(wk)) {
        fprintf(stderr, "FAIL: allocator did not place wq|wk|wv contiguously\n");
        return 1;
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto fill_q8 = [&](ggml_tensor * t) {
        std::vector<float> f(ggml_nelements(t));
        for (auto & x : f) x = u(rng);
        std::vector<char> q(ggml_nbytes(t));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, t->ne[1], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    };
    fill_q8(wq); fill_q8(wk); fill_q8(wv);

    // the alias (what bind_qkv_concat creates)
    ggml_init_params cp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * cctx = ggml_init(cp);
    ggml_tensor * cat = ggml_new_tensor_2d(cctx, GGML_TYPE_Q8_0, K, nq + nk + nv);
    cat->buffer = wq->buffer;
    cat->data   = wq->data;

    ggml_init_params gp = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * g = ggml_init(gp);
    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, K, n_tok);
    ggml_set_input(x);
    ggml_tensor * q_sep = ggml_mul_mat(g, wq, x);
    ggml_tensor * k_sep = ggml_mul_mat(g, wk, x);
    ggml_tensor * v_sep = ggml_mul_mat(g, wv, x);
    ggml_tensor * qkv   = ggml_mul_mat(g, cat, x);
    const size_t es = sizeof(float);
    ggml_tensor * q_cat = ggml_cont(g, ggml_view_2d(g, qkv, nq, n_tok, qkv->nb[1], 0));
    ggml_tensor * k_cat = ggml_cont(g, ggml_view_2d(g, qkv, nk, n_tok, qkv->nb[1], es * nq));
    ggml_tensor * v_cat = ggml_cont(g, ggml_view_2d(g, qkv, nv, n_tok, qkv->nb[1], es * (nq + nk)));
    ggml_tensor * outs[6] = { q_sep, k_sep, v_sep, q_cat, k_cat, v_cat };
    ggml_cgraph * gf = ggml_new_graph(g);
    for (auto * o : outs) { ggml_set_output(o); ggml_build_forward_expand(gf, o); }

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "FAIL: graph alloc\n");
        return 1;
    }
    std::vector<float> xin(K * n_tok);
    for (auto & v : xin) v = u(rng);
    ggml_backend_tensor_set(x, xin.data(), 0, xin.size() * sizeof(float));
    ggml_backend_graph_compute(be, gf);

    int bad = 0;
    for (int i = 0; i < 3; ++i) {
        const size_t n = ggml_nelements(outs[i]);
        std::vector<float> a(n), b(n);
        ggml_backend_tensor_get(outs[i], a.data(), 0, n * sizeof(float));
        ggml_backend_tensor_get(outs[i + 3], b.data(), 0, n * sizeof(float));
        if (memcmp(a.data(), b.data(), n * sizeof(float)) != 0) {
            double md = 0;
            for (size_t j = 0; j < n; ++j) md = std::fmax(md, std::fabs(a[j] - b[j]));
            fprintf(stderr, "FAIL: %c separate vs concat differ (max |d| %g)\n", "QKV"[i], md);
            ++bad;
        }
    }
    printf("test-qkv-concat K=%lld nq=%lld nk=%lld nv=%lld n=%lld: %s\n", (long long) K, (long long) nq, (long long) nk,
        (long long) nv, (long long) n_tok, bad ? "FAIL" : "OK (bit-identical)");

    ggml_gallocr_free(ga);
    ggml_free(g); ggml_free(cctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(be);
    return bad ? 1 : 0;
}

int main() {
    int rc = 0;
    for (int64_t n : {1, 5, 8}) {
        rc |= run(512, 768, 64, 64, n);
    }
    rc |= run(5120, 12288, 1024, 1024, 8);   // served 27B shape
    return rc;
}
