#pragma once

// AK-ASM-H13b: Q38FN (qwen4exp) hyper-connection seam on the CUDA/HIP backend.
//  1. GGML_OP_MEAN_D1 on the device (C6 had no CUDA implementation: every build_hc_mix forced a graph split to the CPU,
//     ~97 GPU->CPU->GPU round trips per decoded token at -ngl 99). GGML_CUDA_MEAN_D1=0 restores the C6 placement.
//  2. Graph fusions of the hc elementwise tail (GGML_CUDA_HC_FUSE=0 disables them; the ops then run unfused):
//     - combine: ADD(res, MUL(REPEAT(b), RESHAPE(SCALE(SIGMOID(SCALE(inject)))))) -> 1 kernel (6 launches -> 1)
//     - mix tail: MEAN_D1(RESHAPE(MUL(xn, SIGMOID(u))))                         -> 1 kernel (3 launches -> 1)
//     - gate:     SILU(SCALE(x))                                                -> 1 kernel (2 launches -> 1)
//  The plan is built per graph evaluation from the node list; an intermediate is elided only when its single consumer
//  is inside the chain and it is not a graph output.

#include "common.cuh"

#include <unordered_map>
#include <unordered_set>

enum ggml_cuda_hc_kind { HC_FUSE_COMBINE = 1, HC_FUSE_MIX_TAIL = 2, HC_FUSE_SCALE_SILU = 3 };

struct ggml_cuda_hc_anchor {
    ggml_cuda_hc_kind   kind;
    const ggml_tensor * a = nullptr;  // combine: residual | mix: xn          | silu: x
    const ggml_tensor * b = nullptr;  // combine: block_out [E,1,T] | mix: u (pre-sigmoid) | silu: -
    const ggml_tensor * c = nullptr;  // combine: inject [hc,T]
    float s1 = 1.0f, b1 = 0.0f, s2 = 1.0f, b2 = 0.0f;  // the SCALE op params (scale, bias) of the chain
};

struct ggml_cuda_hc_plan {
    std::unordered_set<const ggml_tensor *>                     skip;
    std::unordered_map<const ggml_tensor *, ggml_cuda_hc_anchor> anchors;
};

bool ggml_cuda_hc_fuse_enabled();
bool ggml_cuda_mean_d1_enabled();
void ggml_cuda_hc_fuse_plan(const ggml_cgraph * cgraph, ggml_cuda_hc_plan & plan);
void ggml_cuda_hc_fuse_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_hc_anchor & an);
void ggml_cuda_op_mean_d1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
uint64_t ggml_cuda_hc_fuse_n_calls();
