#pragma once

// AK-ASM-H9B (gfx90a, experimental): fused Q8_0 dequant-to-f16 (in LDS staging) + v_mfma_f32_32x32x8f16 GEMM for dense
// Q8_0 x f32 MUL_MAT at ne11 >= 129. Numerics = C6's own route (dequantize_block<q8_0> to f16, f32 -> f16 activations,
// f32 accumulation), in one kernel instead of two converts + rocBLAS.
// Kill switch: GGML_HIP_Q8_H9B=0. Threshold: GGML_HIP_Q8_H9B_MIN_N (default 129).
// Precedence: H9B is tried before H9 (no per-32-block rescale VALU); GGML_HIP_Q8_H9B=0 hands the shape to H9.

#include "common.cuh"

bool ggml_cuda_q8_h9b_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
bool ggml_cuda_q8_h9b_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
