#pragma once
// AK-ASM-H16 (C8-2-H16-M36): MFMA f16 skinny Q8_0 x F32 MUL_MAT for gfx90a, M = ne11 in 12..36 by default
// (production 27B DFlash2 np4 verify = 4 slots x 9 tokens = M 36; np2 = M 18). Q8_0 is dequantized to f16 in
// registers and fed to v_mfma_f32_16x16x16f16; weights stream from HBM once for all M columns.
// Kill switch: GGML_HIP_Q8_H16=0 (falls back to the C7r route). Design note + bandwidth model: mmq-q8-h16.cu.
#include "common.cuh"

bool ggml_cuda_q8_h16_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_q8_h16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
