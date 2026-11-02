#include "../src/llama-kv-rows-reference.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

int main() {
    ggml_init_params params{8u * 1024u * 1024u, nullptr, false};
    auto * ctx = ggml_init(params);
    auto * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
    auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
    auto * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
    float qv[] = {1, 0, 0, 1, 1, 1, -1, 1};
    float kv[] = {1, 0, 0, 1, 1, 1, -1, 1};
    float vv[] = {2, 4, 8, 16, 32, 64, 128, 256};
    std::copy_n(qv, 8, static_cast<float *>(q->data));
    std::copy_n(kv, 8, static_cast<float *>(k->data));
    std::copy_n(vv, 8, static_cast<float *>(v->data));
    llama_kv_rows_plan plan{4, 2, 4, {0, 2, -1, -1, 1, 3, -1, -1}};
    const auto neginf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    const auto zero = ggml_fp32_to_fp16(0);
    std::vector<ggml_fp16_t> mask(16, neginf);
    for (int t = 0; t < 4; ++t) {
        mask[t * 4 + (t < 2 ? 0 : 1)] = zero;
        mask[t * 4 + (t < 2 ? 2 : 3)] = zero;
    }
    std::vector<ggml_tensor *> outputs;
    if (!llama_kv_rows_reference_graph(ctx, plan, q, k, v, mask, 1.0f, outputs) || outputs.size() != 2) return 1;
    auto * graph = ggml_new_graph(ctx);
    for (auto * o : outputs) ggml_build_forward_expand(graph, o);
    if (ggml_graph_compute_with_ctx(ctx, graph, 2) != GGML_STATUS_SUCCESS) return 2;
    for (int t = 0; t < 4; ++t) {
        float scores[4], total = 0;
        for (int j = 0; j < 4; ++j) {
            scores[j] = ggml_fp16_to_fp32(mask[t * 4 + j]) == 0 ?
                std::exp(qv[t * 2] * kv[j * 2] + qv[t * 2 + 1] * kv[j * 2 + 1]) : 0;
            total += scores[j];
        }
        for (int d = 0; d < 2; ++d) {
            float expected = 0;
            for (int j = 0; j < 4; ++j) expected += scores[j] * vv[j * 2 + d] / total;
            float actual = static_cast<float *>(outputs[t / 2]->data)[(t % 2) * 2 + d];
            if (std::fabs(actual - expected) > 0.001f) {
                std::fprintf(stderr, "t=%d d=%d actual=%f expected=%f\n", t, d, actual, expected);
                return 3;
            }
        }
    }
    // Omitted visible rows, repeated rows, and fully masked tokens must fail closed.
    auto malformed = plan; malformed.rows[1] = -1;
    if (llama_kv_rows_reference_graph(ctx, malformed, q, k, v, mask, 1.0f, outputs) || !outputs.empty()) return 4;
    malformed = plan; malformed.rows[1] = 0;
    if (llama_kv_rows_reference_graph(ctx, malformed, q, k, v, mask, 1.0f, outputs) || !outputs.empty()) return 5;
    auto empty = mask; empty[0] = empty[2] = neginf;
    if (llama_kv_rows_reference_graph(ctx, plan, q, k, v, empty, 1.0f, outputs) || !outputs.empty()) return 6;
    ggml_free(ctx);
    std::puts("kv_rows CPU reference graph parity passed");
    return 0;
}
