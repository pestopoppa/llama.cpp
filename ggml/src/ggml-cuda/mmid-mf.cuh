#pragma once
// AK-MMID-MF (speculative-decoding.md): gfx90a expert-grouped Q8_0 MUL_MAT_ID for few-token MoE verify (A3B: 256 experts,
// top-8, n = 1..64 tokens). OPT-IN: GGML_HIP_MMID_MF=1 (default off -> MMVQ_MMID / split16 / MMQ_MMID routing unchanged).
// Range: GGML_HIP_MMID_MF_MIN (default 1) <= n_tokens <= GGML_HIP_MMID_MF_CAP (default 16, max 64).
// GGML_HIP_MMID_MF_SLC=1: nontemporal (slc) weight loads. GGML_HIP_MMID_MF_WGT: target workgroup count (default 4 x CUs).
// Design note + byte accounting: mmid-mf.cu.
#include "common.cuh"

bool ggml_cuda_mmid_mf_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_id_mf(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             const ggml_tensor * ids, ggml_tensor * dst);
