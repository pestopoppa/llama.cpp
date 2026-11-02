#include "../src/llama-kv-rows-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <unistd.h>

int main() {
    constexpr int D = 64, PHYS = 512, ROWS = 256;
    auto * ctx = ggml_init({32u * 1024u * 1024u, nullptr, false});
    auto * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
    auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, PHYS);
    auto * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, PHYS);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ROWS, 1);
    auto * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ROWS);
    auto * qd = static_cast<float *>(q->data);
    auto * kd = static_cast<ggml_fp16_t *>(k->data);
    auto * vd = static_cast<ggml_fp16_t *>(v->data);
    auto * md = static_cast<ggml_fp16_t *>(mask->data);
    auto * rd = static_cast<int32_t *>(rows->data);
    for (int d = 0; d < D; ++d) qd[d] = float((d % 11) - 5) / 8;
    for (int i = 0; i < D * PHYS; ++i) {
        kd[i] = ggml_fp32_to_fp16(float((i % 19) - 9) / 16);
        vd[i] = ggml_fp32_to_fp16(float((i % 23) - 11) / 16);
    }
    for (int i = 0; i < ROWS; ++i) { rd[i] = -1; md[i] = ggml_fp32_to_fp16(-INFINITY); }
    rd[0] = 301; rd[1] = 7; rd[2] = PHYS + 10; // invalid row must remain masked even if mask is finite
    md[0] = md[1] = md[2] = ggml_fp32_to_fp16(0);
    auto * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0, 0);
    out->src[5] = rows; // explicit experimental op; no production graph uses it
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto * backend = ggml_backend_cuda_init(0);
    auto * cpu = ggml_backend_cpu_init();
    if (!backend || !cpu || !ggml_backend_supports_op(backend, out) || ggml_backend_supports_op(cpu, out)) return 1;
    std::fprintf(stderr, "KPF41 DIRECT PID %d backend %s\n", int(getpid()), ggml_backend_name(backend));
    std::vector<std::vector<float>> got;
    if (!llama_kv_rows_backend_compute(backend, ctx, graph, {out}, got) || got.size() != 1) return 2;
    float sa=0, sb=0;
    for (int d=0;d<D;++d) { sa += qd[d]*ggml_fp16_to_fp32(kd[301*D+d])*0.125f; sb += qd[d]*ggml_fp16_to_fp32(kd[7*D+d])*0.125f; }
    const float wa=std::exp(sa), wb=std::exp(sb);
    for (int d=0;d<D;++d) {
        const float want=(wa*ggml_fp16_to_fp32(vd[301*D+d])+wb*ggml_fp16_to_fp32(vd[7*D+d]))/(wa+wb);
        if (!std::isfinite(got[0][d]) || std::fabs(got[0][d]-want)>0.003f) {
            std::fprintf(stderr,"mismatch d=%d got=%g want=%g\n",d,got[0][d],want); return 3;
        }
    }
    // Non-padded list must not be dispatched to the direct GPU kernel.
    auto * invalid_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ROWS-1);
    out->src[5] = invalid_rows;
    if (ggml_backend_supports_op(backend, out)) return 4;
    sleep(2); // caller samples this exact process during residency
    ggml_backend_free(cpu); ggml_backend_free(backend); ggml_free(ctx);
    std::puts("KPF-41 direct gfx90a vec row parity passed");
    return 0;
}
