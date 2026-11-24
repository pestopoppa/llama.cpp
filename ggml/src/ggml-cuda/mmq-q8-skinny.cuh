#pragma once
// AK-ASM-H1/H7: skinny-M (ne11 = 5..16) Q8_0 x F32 MUL_MAT fast path for CDNA (gfx90a), aimed at DFlash
// spec-verify batches. Split-K with a gfx90a-safe reduction (two-pass partials or f32 atomics; no
// MI300-only global_atomic_pk_add_bf16). Kill switch: GGML_HIP_Q8_SKINNY=0. See mmq-q8-skinny.cu.
#include "common.cuh"

bool ggml_cuda_q8_skinny_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_q8_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
