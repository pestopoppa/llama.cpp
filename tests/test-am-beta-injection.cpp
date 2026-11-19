// test-am-beta-injection.cpp: Attention Matching beta injection, model-free CPU test.
//
// 1. llama_kv_cell_ext carries a per-cell beta; the KQ-mask value helper is a no-op unless the default-off
//    knob LLAMA_KV_AM_BETA=1 is set, and the state wire size excludes beta (state format unchanged).
// 2. The compact-cache attention  softmax(Q C1^T / sqrt(d) + beta) C2  evaluated on the ggml CPU backend with the
//    beta carried in the soft-max mask (exactly how the KV cache feeds it) matches the numpy reference
//    (tests/am-beta-reference.py, mirrors epyc-inference-research scripts/benchmark/attention_matching.py),
//    and differs from the beta = 0 output (so the bias is not a no-op).

#include "../src/llama-kv-cells.h"
#include "test-am-beta-injection-ref.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { failures++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

// out[n x d] = softmax(Q C1^T / sqrt(d) + mask) C2, one mask row per query (all rows identical, as in the KV cache)
static std::vector<float> attend(const std::vector<float> & mask_row) {
    const int d = AM_REF_D, n = AM_REF_N, t = AM_REF_TC;
    ggml_init_params ip = { 1 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);   // ne0 = d
    ggml_tensor * c1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, t);
    ggml_tensor * c2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t, d);   // C2^T so that mul_mat(c2, w) = w C2
    ggml_tensor * mk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t, n);
    memcpy(q->data,  AM_REF_Q,  sizeof(AM_REF_Q));
    memcpy(c1->data, AM_REF_C1, sizeof(AM_REF_C1));
    for (int i = 0; i < t; i++) for (int j = 0; j < d; j++) ((float *) c2->data)[j*t + i] = AM_REF_C2[i*d + j];
    for (int r = 0; r < n; r++) for (int i = 0; i < t; i++) ((float *) mk->data)[r*t + i] = mask_row[i];

    ggml_tensor * s = ggml_mul_mat(ctx, c1, q);                                  // t x n
    ggml_tensor * w = ggml_soft_max_ext(ctx, s, mk, 1.0f/sqrtf((float) d), 0.0f); // softmax(s/sqrt(d) + mask)
    ggml_tensor * o = ggml_mul_mat(ctx, c2, w);                                  // d x n

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    std::vector<float> out((float *) o->data, (float *) o->data + d*n);
    ggml_free(ctx);
    return out;
}

static float max_abs_diff(const std::vector<float> & a, const float * b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, fabsf(a[i] - b[i]));
    return m;
}

int main() {
    // --- cells: beta is stored per cell, reset with the cell, and the mask value honours the knob
    llama_kv_cells cells;
    cells.resize(8);
    cells.pos_set(3, 3);
    llama_kv_cell_ext e = cells.ext_get(3);
    CHECK(e.beta == 0.0f, "fresh cell beta must be 0");
    e.beta = -0.75f;
    cells.ext_set(3, e);
    CHECK(cells.ext_get(3).beta == -0.75f, "ext_set must store beta");

    unsetenv("LLAMA_KV_AM_BETA");
    CHECK(!llama_kv_am_beta_enabled(), "knob must be off when unset");
    CHECK(llama_kv_am_mask_value(cells.ext_get(3), llama_kv_am_beta_enabled()) == 0.0f, "knob off: mask value must stay 0");
    setenv("LLAMA_KV_AM_BETA", "0", 1);
    CHECK(!llama_kv_am_beta_enabled(), "knob=0 must be off");
    setenv("LLAMA_KV_AM_BETA", "1", 1);
    CHECK(llama_kv_am_beta_enabled(), "knob=1 must be on");
    CHECK(llama_kv_am_mask_value(cells.ext_get(3), llama_kv_am_beta_enabled()) == -0.75f, "knob on: mask value must be beta");

    cells.rm(3);
    cells.pos_set(3, 4);
    CHECK(cells.ext_get(3).beta == 0.0f, "beta must reset when the cell is freed");

    CHECK(LLAMA_KV_CELL_EXT_WIRE_SIZE == offsetof(llama_kv_cell_ext, beta) && LLAMA_KV_CELL_EXT_WIRE_SIZE == 3*sizeof(int32_t),
          "state wire size must exclude beta (x, y, tok)");

    // --- numerical: beta through the mask reproduces the numpy compact-cache attention
    llama_kv_cells kv;
    kv.resize(AM_REF_TC);
    for (int i = 0; i < AM_REF_TC; i++) {
        kv.pos_set(i, i);
        llama_kv_cell_ext x = kv.ext_get(i);
        x.beta = AM_REF_BETA[i];
        kv.ext_set(i, x);
    }

    std::vector<float> on(AM_REF_TC), off(AM_REF_TC);
    for (int i = 0; i < AM_REF_TC; i++) {
        on[i]  = llama_kv_am_mask_value(kv.ext_get(i), true);
        off[i] = llama_kv_am_mask_value(kv.ext_get(i), false);
    }

    const float d_on  = max_abs_diff(attend(on),  AM_REF_OUT);
    const float d_off = max_abs_diff(attend(off), AM_REF_OUT_NOBETA);
    float d_sep = 0.0f;
    { const auto a = attend(on); const auto b = attend(off); for (size_t i = 0; i < a.size(); i++) d_sep = std::max(d_sep, fabsf(a[i] - b[i])); }
    printf("max|on-ref|=%.3g  max|off-ref0|=%.3g  max|on-off|=%.3g\n", d_on, d_off, d_sep);
    CHECK(d_on  < 1e-5f, "beta-on output must match the numpy reference (%.3g)", d_on);
    CHECK(d_off < 1e-5f, "beta-off output must match the numpy no-beta reference (%.3g)", d_off);
    CHECK(d_sep > 0.1f,  "beta must change the attention output (%.3g)", d_sep);

    printf("%s\n", failures ? "test-am-beta-injection: FAILED" : "test-am-beta-injection: OK");
    return failures ? 1 : 0;
}
