#pragma once
// AK-ASM-H1b: wvSplitK-style skinny-M (ne11 = 5..16) Q8_0 x F32 MUL_MAT for gfx90a, LARGE weights only (> 6 MiB by
// default; the mmq-q8-skinny path keeps <= 6 MiB). Kill switch: GGML_HIP_Q8_WVSK=0 (falls back to C6 MMVQ/MMQ).
// Design note + bandwidth estimate: mmq-q8-wvsk.cu.
#include "common.cuh"

bool ggml_cuda_q8_wvsk_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_q8_wvsk(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
