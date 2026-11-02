#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>

int main() {
    auto * ctx = ggml_init({4u * 1024u * 1024u, nullptr, false});
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 2, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 512, 1, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 512, 1, 1);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 1);
    auto * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 256);
    ggml_set_name(rows, "test_kv_rows");
    auto * out = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, rows, 0.125f, 0.0f, 0.0f);
    if (!out || out->op != GGML_OP_FLASH_ATTN_EXT || out->src[5] != rows ||
        out->src[3] != mask || out->ne[0] != 64 || out->ne[1] != 2 ||
        out->ne[2] != 1 || out->ne[3] != 1) return 1;
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (ggml_graph_get_tensor(graph, "test_kv_rows") != rows) return 2;
    auto * cpu = ggml_backend_cpu_init();
    if (!cpu || ggml_backend_supports_op(cpu, out)) return 3;

    // The serving graph gathers the full mask by physical cell into a compact
    // mask. Padding points at cell zero for the gather; the separate row input
    // still carries -1 and the row kernel suppresses that lane.
    auto * full_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 512, 1);
    auto * gather_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 256);
    auto * fm = static_cast<ggml_fp16_t *>(full_mask->data);
    auto * ids = static_cast<int32_t *>(gather_rows->data);
    for (int i = 0; i < 512; ++i) fm[i] = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < 256; ++i) ids[i] = 0;
    fm[7] = ggml_fp32_to_fp16(-2.0f);
    fm[301] = ggml_fp32_to_fp16(-3.0f);
    ids[0] = 301; ids[1] = 7;
    auto * mask_by_cell = ggml_reshape_2d(ctx, full_mask, 1, 512);
    auto * gathered = ggml_get_rows(ctx, mask_by_cell, gather_rows);
    auto * compact = ggml_reshape_2d(ctx, ggml_cast(ctx, gathered, GGML_TYPE_F16), 256, 1);
    auto * mask_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(mask_graph, compact);
    if (ggml_graph_compute_with_ctx(ctx, mask_graph, 1) != GGML_STATUS_SUCCESS) return 4;
    const auto * got = static_cast<const ggml_fp16_t *>(compact->data);
    if (ggml_fp16_to_fp32(got[0]) != -3.0f || ggml_fp16_to_fp32(got[1]) != -2.0f ||
        !std::isinf(ggml_fp16_to_fp32(got[2])) || ggml_fp16_to_fp32(got[2]) >= 0) return 5;
    ggml_backend_free(cpu);
    ggml_free(ctx);
    std::puts("row attention graph retains row input and CPU rejects it");
    return 0;
}
