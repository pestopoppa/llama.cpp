#pragma once
// AK-RF-27B-TAIL-FUSE-2: the remaining 27B verify-tail launch fusions (ak-27b-tail-classify-20261124 groups G2-G8).
// One knob per group, each default off (unset or != 1 = the C21 launch sequence unchanged):
//   GGML_CUDA_AK_TF2_G2=1  FFN SwiGLU (GLU) folded into the down-proj q8mf6 activation pre-pass (no GLU launch)
//   GGML_CUDA_AK_TF2_G3=1  GDN gate: ADD(alpha, dt) -> SOFTPLUS -> MUL(a) in one launch
//   GGML_CUDA_AK_TF2_G5=1  GDN gated output norm: RMS_NORM -> MUL(w) -> SILU(z) * . in one launch
//   GGML_CUDA_AK_TF2_G6=1  attention K prologue: multi-section ROPE -> VIEW -> SET_ROWS into the KV cache in one launch
// Every fused path computes the same fp32 expressions in the same order as the unfused kernels it replaces, so its
// outputs are bit-identical to the knob-off graph.
// Witness: ggml_backend_cuda_ak_tf2_n_calls(group) (proc address) and a one-shot GGML_LOG_INFO per group on its first
// fused launch: "AK-RF-27B-TAIL-FUSE-2 G<n> <what> fused path ran".
#include "common.cuh"

#include <cstdint>

bool     ggml_cuda_ak_tf2_enabled(int group);
uint64_t ggml_cuda_ak_tf2_n_calls(int group);
void     ggml_cuda_ak_tf2_note(int group, const char * what);

// G3
bool ggml_cuda_ak_tf2_g3_supported(const ggml_tensor * add, const ggml_tensor * sp, const ggml_tensor * mul);
void ggml_cuda_ak_tf2_g3(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * sp, ggml_tensor * mul);

// G5 (norm.cu)
bool ggml_cuda_ak_tf2_g5_supported(const ggml_tensor * norm, const ggml_tensor * nw, const ggml_tensor * sl,
                                   const ggml_tensor * out);
void ggml_cuda_ak_tf2_g5(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * nw,
                         const ggml_tensor * sl, ggml_tensor * out);
