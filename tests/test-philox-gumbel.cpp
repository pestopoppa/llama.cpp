// [GPU-RESIDENT-ROUND-2f] GGML_OP_PHILOX_GUMBEL vs the host stream of common/philox.h (GPU-RESIDENT-ROUND-2d
// dsel_set_inputs): bitwise equality on every backend that supports the op, including a strided counter view
// (as read from a control-block row) and ctr < 0 (temp <= 0 block -> zeros).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../common/philox.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const int64_t top_k = 8, block = 16, n_b = 3;
    const uint64_t keys[n_b] = { ((uint64_t) (1234u ^ 0x85ebca6bU) << 32) | 0u, ((uint64_t) (77u ^ 0x85ebca6bU) << 32) | 1u,
                                 ((uint64_t) (5u ^ 0x85ebca6bU) << 32) | 2u };
    const float ctrs[n_b] = { 0.0f, 4097.0f, -1.0f };

    // reference: dsel_set_inputs layout
    std::vector<float> ref((size_t) top_k*block*n_b, 0.0f);
    for (int64_t b = 0; b < n_b; ++b) {
        if (ctrs[b] < 0) continue;
        float * nz = ref.data() + b*top_k*block;
        for (int64_t j = top_k; j < block*top_k; j += 4) {
            uint32_t r[4];
            common_philox4x32::gen(keys[b], (uint64_t) ctrs[b], (uint64_t) j, r);
            for (int q = 0; q < 4 && j + q < block*top_k; ++q) nz[j + q] = common_philox4x32::gumbel(r[q]);
        }
    }

    ggml_backend_load_all();
    int fails = 0, tested = 0;
    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) continue;
        ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * key = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_b);
        ggml_tensor * cb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, n_b);           // control block [rows, n_stream]
        ggml_tensor * ctr = ggml_view_2d(ctx, cb, 1, n_b, cb->nb[1], 3*sizeof(float)); // row 3, strided
        ggml_tensor * out = ggml_philox_gumbel(ctx, key, ctr, top_k, block, (int32_t) top_k);
        if (!ggml_backend_supports_op(be, out)) { ggml_free(ctx); ggml_backend_free(be); continue; }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<int32_t> kd(2*n_b);
        std::vector<float> cd(4*n_b, 9.0f);
        for (int64_t b = 0; b < n_b; ++b) { kd[2*b] = (int32_t) (uint32_t) keys[b]; kd[2*b+1] = (int32_t) (uint32_t) (keys[b] >> 32); cd[4*b+3] = ctrs[b]; }
        ggml_backend_tensor_set(key, kd.data(), 0, kd.size()*4);
        ggml_backend_tensor_set(cb, cd.data(), 0, cd.size()*4);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> got(ref.size());
        ggml_backend_tensor_get(out, got.data(), 0, got.size()*4);
        int bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) bad += memcmp(&ref[i], &got[i], 4) != 0;
        printf("%-12s philox_gumbel mismatches %d / %zu\n", ggml_backend_name(be), bad, ref.size());
        fails += bad != 0; tested++;
        ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    }
    printf("%s (%d backends)\n", fails == 0 && tested > 0 ? "OK" : "FAIL", tested);
    return fails == 0 && tested > 0 ? 0 : 1;
}
