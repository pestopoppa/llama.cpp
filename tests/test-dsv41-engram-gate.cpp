// DS41-B3.1: the Engram gate sign must be +1 at dot == +0.0 (torch.copysign semantics),
// and bit-identical to the old ggml_sgn formulation everywhere dot != 0.
// Builds exactly the gate expression of src/models/deepseek41.cpp (keep in sync).
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static float ref_gate(float dot) { // inference/model.py: sigmoid(copysign(sqrt(clamp_min(|dot|,1e-6)), dot))
    float m = std::sqrt(std::fmax(std::fabs(dot), 1e-6f));
    return 1.0f / (1.0f + std::exp(-std::copysign(m, dot)));
}

int main() {
    const float vals[] = { 0.0f, 1e-9f, -1e-9f, 1e-6f, -1e-6f, 3e-4f, -3e-4f, 0.5f, -0.5f, 7.25f, -7.25f, 1e3f, -1e3f, 0.0f };
    const int n = sizeof(vals) / sizeof(vals[0]);

    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * dot = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    memcpy(dot->data, vals, sizeof(vals));

    auto mk = [&](bool fixed) {
        ggml_tensor * mag = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, INFINITY));
        ggml_tensor * sgn = fixed ? ggml_scale_bias(ctx, ggml_step(ctx, ggml_neg(ctx, dot)), -2.0f, 1.0f)
                                  : ggml_sgn(ctx, dot);
        return ggml_sigmoid(ctx, ggml_mul(ctx, mag, sgn));
    };
    ggml_tensor * g_new = mk(true);
    ggml_tensor * g_old = mk(false);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, g_new);
    ggml_build_forward_expand(gf, g_old);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    int pass = 0, fail = 0;
    for (int i = 0; i < n; ++i) {
        const float a = ((float *) g_new->data)[i], b = ((float *) g_old->data)[i];
        const float r = ref_gate(vals[i]);
        bool ok;
        if (vals[i] == 0.0f) {
            ok = std::fabs(a - r) <= 1e-7f && a != b; // fixed matches reference; old one was sigmoid(0)=0.5
        } else {
            ok = std::memcmp(&a, &b, sizeof(float)) == 0 && std::fabs(a - r) <= 1e-6f; // bit-identical to old
        }
        printf("dot=% .3e new=%.9g old=%.9g ref=%.9g %s\n", vals[i], a, b, r, ok ? "ok" : "FAIL");
        ok ? ++pass : ++fail;
    }
    ggml_free(ctx);
    printf("RESULT test-dsv41-engram-gate pass=%d fail=%d\n", pass, fail);
    return fail ? 1 : 0;
}
