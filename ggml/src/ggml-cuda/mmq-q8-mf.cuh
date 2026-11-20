#pragma once
// DTREE-0-Q8MF: gfx90a small-M (ne11 = 9..16 by default) Q8_0 x F32 MUL_MAT on v_mfma_i32_16x16x16i8, LARGE weights
// only (> 6 MiB by default). OPT-IN: GGML_HIP_Q8_MF=1 (default off -> wvsk / skinny / C6 route unchanged).
// Design note + bandwidth estimate: mmq-q8-mf.cu.
#include "common.cuh"

bool ggml_cuda_q8_mf_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_q8_mf(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
