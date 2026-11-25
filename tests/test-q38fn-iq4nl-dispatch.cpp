// Q38FN-GEMV2 F1 dispatch test: run the served-shape IQ4_NL MUL_MAT_ID (ffn_down_exps: k=640 -> m=2560, top-10)
// through the SAME buffer choice and dispatch entry llama-server uses. test-backend-ops always allocates weights in the
// plain CPU buffer, so it never sees CPU_REPACK; the server's model loader puts a weight in the first extra buffer type
// whose device supports_op() accepts it (llama-model.cpp weight_buft_supported), and CPU_REPACK then computes it in its
// own extra forward, before the iqk hook. This test reproduces that choice:
//   1. allocate the expert tensor in each CPU extra buffer type in order, keep the first one supports_op() accepts
//      (fallback: plain CPU buffer), exactly like the loader;
//   2. compute MUL_MAT_ID on a CPU backend (n = 1 and 5 rows) and compare against an f32 reference built from the
//      dequantized weights (NMSE < 5e-4).
// Expectation (argv[1]): "repack" = the weight must land in CPU_REPACK (knob OFF), "plain" = it must not (knob ON, so
// the iqk hook runs and prints "[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=20 ...").
// Usage: test-q38fn-iq4nl-dispatch <repack|plain> [n_threads]   (no timing)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2 || (std::string(argv[1]) != "repack" && std::string(argv[1]) != "plain")) {
        fprintf(stderr, "usage: %s <repack|plain> [n_threads]\n", argv[0]);
        return 2;
    }
    const bool want_repack = std::string(argv[1]) == "repack";
    const int nth = argc > 2 ? atoi(argv[2]) : 8;
    const int64_t k = 640, m = 2560, n_mats = 64, n_used = 10;

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_cpu_set_n_threads(be, nth);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> wf(k * m * n_mats);
    for (auto & x : wf) x = nd(rng);
    std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_IQ4_NL, k) * m * n_mats);
    ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wf.data(), wq.data(), 0, m * n_mats, k, nullptr);
    std::vector<float> wd(k * m * n_mats);   // what the kernel actually multiplies
    ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(wq.data(), wd.data(), k * m * n_mats);

    int fails = 0;
    for (int64_t n : {1, 5}) {
        ggml_init_params wp = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * wctx = ggml_init(wp);
        ggml_tensor * as = ggml_new_tensor_3d(wctx, GGML_TYPE_IQ4_NL, k, m, n_mats);
        ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * b   = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, k, n_used, n);
        ggml_tensor * ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, n_used, n);
        ggml_tensor * out = ggml_mul_mat_id(gctx, as, b, ids);

        // the loader's choice: first extra buffer type whose supports_op accepts the op, else the plain CPU buffer
        ggml_backend_buffer_t wbuf = nullptr;
        std::string chosen = "CPU";
        for (ggml_backend_buffer_type_t * e = get_extra ? get_extra(dev) : nullptr; e && *e; ++e) {
            ggml_backend_buffer_t tb = ggml_backend_alloc_ctx_tensors_from_buft(wctx, *e);
            if (!tb) continue;
            if (ggml_backend_dev_supports_op(dev, out)) { wbuf = tb; chosen = ggml_backend_buft_name(*e); break; }
            ggml_backend_buffer_free(tb);
            as->buffer = nullptr; as->data = nullptr; as->extra = nullptr;
        }
        if (!wbuf) {
            wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_cpu_buffer_type());
        }
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(as, wq.data(), 0, wq.size());

        ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, be);
        std::vector<float> bf(k * n_used * n);
        for (auto & x : bf) x = nd(rng);
        std::vector<int32_t> idv(n_used * n);
        for (int64_t r = 0; r < n; ++r) {
            std::vector<int32_t> p(n_mats);
            for (int i = 0; i < n_mats; ++i) p[i] = i;
            std::shuffle(p.begin(), p.end(), rng);
            for (int64_t j = 0; j < n_used; ++j) idv[r * n_used + j] = p[j];
        }
        ggml_backend_tensor_set(b, bf.data(), 0, bf.size() * sizeof(float));
        ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, out);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        std::vector<float> got(m * n_used * n);
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

        double num = 0, den = 0;
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t j = 0; j < n_used; ++j) {
                const float * W = wd.data() + (size_t) idv[r * n_used + j] * m * k;
                const float * x = bf.data() + (r * n_used + j) * k;
                for (int64_t i = 0; i < m; ++i) {
                    double s = 0;
                    for (int64_t c = 0; c < k; ++c) s += (double) W[i * k + c] * x[c];
                    const double g = got[(r * n_used + j) * m + i];
                    num += (g - s) * (g - s); den += s * s;
                }
            }
        }
        const double nmse = num / den;
        const bool is_repack = chosen.find("REPACK") != std::string::npos;
        const bool ok = nmse < 5e-4 && is_repack == want_repack;
        printf("  IQ4_NL MUL_MAT_ID n=%lld: weight buffer %s (want %s), nmse %.3g: %s\n", (long long) n, chosen.c_str(),
               want_repack ? "CPU_REPACK" : "not CPU_REPACK", nmse, ok ? "OK" : "FAIL");
        fails += ok ? 0 : 1;
        ggml_backend_buffer_free(gbuf);
        ggml_backend_buffer_free(wbuf);
        ggml_free(gctx);
        ggml_free(wctx);
    }
    ggml_backend_free(be);
    printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
