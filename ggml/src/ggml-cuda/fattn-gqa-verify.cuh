#pragma once

#include "common.cuh"

// AK-ASM-H10: GQA-packed multi-token verify attention (head_dim 256, f16 or q8_0 KV, n_q 2..16) for CDNA (MFMA).
// All q heads sharing one KV head x all query tokens are packed into 16-row MFMA tiles, so each KV row is read once per
// KV head; q8_0 K/V is dequantized in registers on its way to LDS (no full KV->f16 conversion); split-KV across
// workgroups with an LSE merge. Kill switch: GGML_CUDA_FA_GQA_VERIFY=0 (falls back to the C6 kernel selection).

bool ggml_cuda_fattn_gqa_verify_applicable(int device, const ggml_tensor * dst);

void ggml_cuda_fattn_gqa_verify(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// KPF-41 FAROWS-INPLACE (GGML_CUDA_FA_ROWS_INPLACE): K/V read in place through a device row list, see fattn.cu
void ggml_cuda_fattn_gqa_verify_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int32_t * rows, int64_t rows_s1);
