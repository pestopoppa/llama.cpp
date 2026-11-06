#pragma once

// AK-ASM-H11: MoE router fast path on gfx90a (MI210) through the retargeted AITER topksoftmax asm kernels
// (ggml/src/ggml-hip/topk-asm-gfx90a/README.md). Compiled only with GGML_USE_HIP && GGML_HIP_TOPK_ASM_GFX90A;
// otherwise the entry point is an inline no-op and the existing topk-moe kernel runs unchanged.

#include "common.cuh"

struct ggml_cuda_topk_moe_args; // topk-moe.cuh (no include guard there)

#include <cstdint>

#if defined(GGML_USE_HIP) && defined(GGML_HIP_TOPK_ASM_GFX90A)
#define GGML_CUDA_TOPK_ASM_GFX90A_AVAILABLE

// Runs softmax -> top-k (-> renormalize) through the asm kernel and returns true, or returns false having enqueued
// nothing (the caller then runs the existing topk-moe kernel). Covers only: gfx90a, plain softmax gating (no bias,
// no sigmoid/softplus, no delayed softmax, no scale != 1), f32 contiguous logits, and an (n_expert, k) pair with an
// embedded code object: (128, 4|6|8), (256, 6|8), (384, 8), (640, 8). Q38FN's (512, 10) has no AITER kernel.
bool ggml_cuda_topk_moe_asm_gfx90a_try(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor *         logits,
                                       ggml_tensor *               weights,
                                       ggml_tensor *               ids,
                                       const ggml_tensor *         clamp,
                                       const ggml_tensor *         scale,
                                       const ggml_tensor *         bias,
                                       const ggml_cuda_topk_moe_args & args);

// Test witness (exported through the backend reg's get_proc_address): router ops the asm kernel computed.
uint64_t ggml_cuda_topk_moe_asm_gfx90a_n_calls();
#else
static inline bool ggml_cuda_topk_moe_asm_gfx90a_try(ggml_backend_cuda_context &, const ggml_tensor *, ggml_tensor *,
                                                     ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
                                                     const ggml_tensor *, const ggml_cuda_topk_moe_args &) {
    return false;
}
#endif
