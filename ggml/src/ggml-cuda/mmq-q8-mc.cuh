#pragma once
// Q8-MULTICOL-GEMV-1 (GGML_HIP_Q8MC=1, default OFF): gfx90a Q8_0 x F32 multi-column GEMV for ne11 = 2..8.
// Streams each weight row ONCE and applies it to all M columns (activations int8-quantized once, paged in LDS),
// with the next K step's weight windows in flight while the current step's dot products run (register
// double buffer). Design: /mnt/raid0/llm/tmp/q8-multicol-gemv-design-20261124.md and mmq-q8-mc.cu.
#include "common.cuh"

bool ggml_cuda_q8_mc_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_q8_mc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
