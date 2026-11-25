// Q38FN-GEMV2 KLD diagnosis: error of the served-shape MUL_MAT_ID against an f64 reference, through the server's
// buffer choice (first CPU extra buffer type whose supports_op accepts the op, else plain CPU; same as the loader).
// Prints, per (type, n): weight buffer, NMSE vs the f64 dot of the dequantized weights with the f32 activations, and
// max |err| / rms(ref). Run once per knob setting (the iqk knobs are read once per process). No timing.
// Usage: test-q38fn-mmid-err [n_threads]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void run(ggml_backend_t be, ggml_backend_dev_t dev, ggml_type type, int64_t k, int64_t m, int64_t n, bool shared) {
    const int64_t n_mats = 16, n_used = 10;
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> wf(k * m * n_mats);
    for (auto & x : wf) x = nd(rng);
    std::vector<uint8_t> wq(ggml_row_size(type, k) * m * n_mats);
    ggml_quantize_chunk(type, wf.data(), wq.data(), 0, m * n_mats, k, nullptr);
    std::vector<float> wd(k * m * n_mats);
    ggml_get_type_traits(type)->to_float(wq.data(), wd.data(), k * m * n_mats);

    ggml_init_params wp = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * as = ggml_new_tensor_3d(wctx, type, k, m, n_mats);
    ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * b   = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, k, n_used, n);
    ggml_tensor * ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, n_used, n);
    ggml_tensor * out = ggml_mul_mat_id(gctx, as, b, ids);
    ggml_backend_buffer_t wbuf = nullptr;
    std::string chosen = "CPU";
    for (ggml_backend_buffer_type_t * e = get_extra ? get_extra(dev) : nullptr; e && *e; ++e) {
        ggml_backend_buffer_t tb = ggml_backend_alloc_ctx_tensors_from_buft(wctx, *e);
        if (!tb) continue;
        if (ggml_backend_dev_supports_op(dev, out)) { wbuf = tb; chosen = ggml_backend_buft_name(*e); break; }
        ggml_backend_buffer_free(tb);
        as->buffer = nullptr; as->data = nullptr; as->extra = nullptr;
    }
    if (!wbuf) wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(as, wq.data(), 0, wq.size());
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, be);
    std::vector<float> bf(k * n_used * n);
    for (auto & x : bf) x = nd(rng);
    std::vector<int32_t> idv(n_used * n), p(n_mats);
    for (int i = 0; i < n_mats; ++i) p[i] = i;
    for (int64_t r = 0; r < n; ++r) {
        if (r == 0 || !shared) std::shuffle(p.begin(), p.end(), rng);
        for (int64_t j = 0; j < n_used; ++j) idv[r * n_used + j] = p[j];
    }
    ggml_backend_tensor_set(b, bf.data(), 0, bf.size() * sizeof(float));
    ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> got(m * n_used * n);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    double num = 0, den = 0, mx = 0;
    for (int64_t r = 0; r < n; ++r) for (int64_t j = 0; j < n_used; ++j) {
        const float * W = wd.data() + (size_t) idv[r * n_used + j] * m * k;
        const float * x = bf.data() + (r * n_used + j) * k;
        for (int64_t i = 0; i < m; ++i) {
            double s = 0;
            for (int64_t c = 0; c < k; ++c) s += (double) W[i * k + c] * x[c];
            const double e = got[(r * n_used + j) * m + i] - s;
            num += e * e; den += s * s; mx = std::max(mx, std::fabs(e));
        }
    }
    const double rms = std::sqrt(den / (double) (m * n_used * n));
    printf("  %-6s k=%lld m=%lld n=%lld %s: buffer %-10s nmse %.3e  max|err|/rms %.3e\n", ggml_type_name(type),
           (long long) k, (long long) m, (long long) n, shared ? "shared" : "indep ", chosen.c_str(), num / den, mx / rms);
    ggml_backend_buffer_free(gbuf); ggml_backend_buffer_free(wbuf); ggml_free(gctx); ggml_free(wctx);
}

int main(int argc, char ** argv) {
    const int nth = argc > 1 ? atoi(argv[1]) : 4;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_cpu_set_n_threads(be, nth);
    setvbuf(stdout, nullptr, _IONBF, 0);
    for (int64_t n : {1, 2, 3, 4, 5, 8}) for (bool sh : {false, true}) {
        run(be, dev, GGML_TYPE_IQ3_S, 2560, 640, n, sh);
        run(be, dev, GGML_TYPE_IQ4_NL, 640, 2560, n, sh);
    }
    ggml_backend_free(be);
    return 0;
}
