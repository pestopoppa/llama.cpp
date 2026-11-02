#include "ggml.h"
#include "ggml-cpu.h"
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
    ggml_backend_free(cpu);
    ggml_free(ctx);
    std::puts("row attention graph retains row input and CPU rejects it");
    return 0;
}
