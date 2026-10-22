// test-gdn-state-guard.cpp: [CH-LB1] the CPU backend must refuse GGML_OP_GATED_DELTA_NET with a
// BF16 recurrent state.
//
// ggml_gated_delta_net() accepts an F32 or a BF16 state ([TAG_GDN_STATE_BF16], selected in
// llama-model.cpp by GGML_CUDA_GDN_STATE_BF16, which is read regardless of backend). The CPU kernel
// reads and writes the state as float, so a BF16 state on CPU is reinterpreted bytes. This test
// checks the supports_op contract only (no compute): F32 state -> supported, BF16 state -> refused,
// for K = 1 and K = 4 snapshot slots, scalar and KDA (vector) gates.

#include "ggml.h"
#include "ggml-backend.h"

#include <initializer_list>
#include <cstdio>

static ggml_tensor * build_gdn(ggml_context * ctx, ggml_type state_type, int64_t K, bool kda) {
    const int64_t S = 16, H = 4, T = 2, B = 1;
    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kda ? S : 1, H, T, B);
    ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, state_type, S, S, H, B);
    return ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K);
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        fprintf(stderr, "no CPU backend device\n");
        return 1;
    }

    ggml_init_params ip = { /*.mem_size =*/ 4 * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);

    int failures = 0;
    for (int64_t K : { 1, 4 }) {
        for (bool kda : { false, true }) {
            ggml_tensor * f32  = build_gdn(ctx, GGML_TYPE_F32,  K, kda);
            ggml_tensor * bf16 = build_gdn(ctx, GGML_TYPE_BF16, K, kda);
            const bool ok_f32  = ggml_backend_dev_supports_op(cpu, f32);
            const bool ok_bf16 = ggml_backend_dev_supports_op(cpu, bf16);
            printf("K=%lld kda=%d: F32 state supported=%d (want 1), BF16 state supported=%d (want 0)\n",
                   (long long) K, (int) kda, (int) ok_f32, (int) ok_bf16);
            if (!ok_f32)  { failures++; }
            if (ok_bf16)  { failures++; }
        }
    }

    ggml_free(ctx);
    if (failures == 0) {
        printf("OK\n");
    } else {
        printf("FAIL (%d)\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
