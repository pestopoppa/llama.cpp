#pragma once

// AK-ASM-H13b: Q38FN (qwen4exp) hyper-connection seam on the CUDA/HIP backend.
//  1. GGML_OP_MEAN_D1 on the device (C6 had no CUDA implementation: every build_hc_mix forced a graph split to the CPU,
//     ~97 GPU->CPU->GPU round trips per decoded token at -ngl 99). GGML_CUDA_MEAN_D1=0 restores the C6 placement.
//  2. Graph fusions of the hc elementwise tail (GGML_CUDA_HC_FUSE=0 disables them; the ops then run unfused):
//     - combine: ADD(res, MUL(REPEAT(b), RESHAPE(SCALE(SIGMOID(SCALE(inject)))))) -> 1 kernel (6 launches -> 1)
//     - mix tail: MEAN_D1(RESHAPE(MUL(xn, SIGMOID(u))))                         -> 1 kernel (3 launches -> 1)
//     - gate:     SILU(SCALE(x))                                                -> 1 kernel (2 launches -> 1)
//     - norm:     MUL(RESHAPE(RMS_NORM(x [E,hc,T])), gamma [hc*E])                -> 1 kernel (2 launches -> 1, H13c)
//  The plan is built per graph evaluation from the node list; an intermediate is elided only when its single consumer
//  is inside the chain and it is not a graph output.

#include "common.cuh"

#include "hc-fuse-plan.h"

bool ggml_cuda_hc_fuse_enabled();
bool ggml_cuda_mean_d1_enabled();
bool ggml_cuda_hc_rms_fuse_enabled();  // GGML_CUDA_HC_RMS_FUSE=0 disables the H13c rms_norm*gamma fusion
void ggml_cuda_hc_fuse_plan(const ggml_cgraph * cgraph, ggml_cuda_hc_plan & plan);
void ggml_cuda_hc_fuse_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_hc_anchor & an);
void ggml_cuda_op_mean_d1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
uint64_t ggml_cuda_hc_fuse_n_calls();
