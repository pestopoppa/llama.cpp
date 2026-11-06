#pragma once

#include "common.cuh"

// AK-ASM-H14 / AK-ASM-H15 decode-path fusions (autokernel-loop.md).
//
// H14: RMS_NORM -> MUL(weight) -> ROPE [-> VIEW -> SET_ROWS(F32|F16|Q8_0)]
//      q/k head norm + rope (+ KV-cache write, + q8_0 quantize) in one launch.
//      Kill switch: GGML_CUDA_AK_H14=0.
// H15: CONCAT(conv_state, qkv^T) -> (VIEW, VIEW, CPY) x K -> SSM_CONV -> SILU
//      -> VIEW -> L2_NORM(q) -> VIEW -> L2_NORM(k)
//      gated-DeltaNet conv1d state update + conv + silu + q/k l2 norm in one launch.
//      Kill switch: GGML_CUDA_AK_H15=0.
// GGML_CUDA_AK_FUSE_LOG=1 prints one line the first time each fusion fires.
//
// Both return the number of graph nodes to skip (0 = pattern not taken, run unfused).
int ggml_cuda_ak_h14_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
int ggml_cuda_ak_h15_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
