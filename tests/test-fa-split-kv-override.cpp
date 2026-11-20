// [TAG_FA_SPLIT_KV_OVERRIDE] per-node override of the CPU split-KV flash-attention decode path
// (Q38FN-DRAFT-SPLITKV). Checks: setter/getter round trip on op_params[5], default 0, and that a
// single-row decode at n_kv >= 512 gives the same result (to f32 reduction-order tolerance) with the
// override forced on (1), forced off (2) and unset (0): on vs off differ only by the f16-V accumulation order
// (tolerance 2e-3 abs on O(0.1) outputs); unset must equal one of them exactly.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static std::vector<float> run(int mode, int nth) {
    const int64_t D = 128, H = 4, NKV = 2048;
    ggml_init_params ip = { (size_t) 256 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, H, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, H, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, H, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, NKV, 1, 1, 1);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[i] = u(rng);
    for (int64_t i = 0; i < ggml_nelements(k); ++i) ((ggml_fp16_t *) k->data)[i] = ggml_fp32_to_fp16(u(rng));
    for (int64_t i = 0; i < ggml_nelements(v); ++i) ((ggml_fp16_t *) v->data)[i] = ggml_fp32_to_fp16(u(rng));
    for (int64_t i = 0; i < ggml_nelements(m); ++i) ((ggml_fp16_t *) m->data)[i] = ggml_fp32_to_fp16(0.0f);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (ggml_flash_attn_ext_get_split_kv(out) != 0) { fprintf(stderr, "default not 0\n"); exit(1); }
    if (mode) ggml_flash_attn_ext_set_split_kv(out, mode);
    if (ggml_flash_attn_ext_get_split_kv(out) != mode) { fprintf(stderr, "round trip failed\n"); exit(1); }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, nth) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    std::vector<float> r((float *) out->data, (float *) out->data + ggml_nelements(out));
    ggml_free(ctx);
    return r;
}

int main() {
    const auto a = run(0, 8), on = run(1, 8), off = run(2, 8);
    double max_on = 0, max_off = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(on[i]) || !std::isfinite(off[i])) { fprintf(stderr, "non-finite at %zu\n", i); return 1; }
        max_on  = std::fmax(max_on,  std::fabs(on[i]  - off[i]));
        max_off = std::fmax(max_off, std::fabs(a[i]   - off[i]) < std::fabs(a[i] - on[i]) ? std::fabs(a[i] - off[i]) : std::fabs(a[i] - on[i]));
    }
    printf("max|on-off|=%.3g max|default-nearest|=%.3g\n", max_on, max_off);
    if (max_on > 2e-3 || max_off > 1e-6) { fprintf(stderr, "FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
