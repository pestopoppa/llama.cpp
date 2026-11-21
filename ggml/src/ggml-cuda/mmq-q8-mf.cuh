#pragma once
// DTREE-0-Q8MF-V2: gfx90a small-M (ne11 = 9..16 by default) Q8_0 x F32 MUL_MAT, LARGE weights only (> 6 MiB by default).
// OPT-IN: GGML_HIP_Q8_MF=1 (int8 v_mfma_i32_16x16x16i8) or =2 (f16 v_mfma_f32_32x32x8f16, in-register Q8_0 dequant);
// V3 (DTREE-0-Q8MF-V3, 2 WGs/CU, 3 register stages): =3 (int8 16x16x16i8) or =4 (f16 16x16x16f16, in-register dequant);
// V5 (DTREE-0-Q8MF-V5): =6, packed (GGML_HIP_Q8_MF_INPLACE) weights only, 4 WGs/CU half-tile units, atomic split-tile tail;
// natural-layout weights run V3 MF=3. 2 register stages; GGML_HIP_Q8_MF_SLC=1 nontemporal loads.
// default off -> wvsk / skinny / C6 route unchanged. Design note + roofline accounting: mmq-q8-mf.cu.
#include "common.cuh"

bool ggml_cuda_q8_mf_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_q8_mf(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// DTREE-0-Q8MF-REPACK-INPLACE (GGML_HIP_Q8_MF_INPLACE=1 weights / =2 any buffer; needs GGML_HIP_Q8_MF=3|4|6): in-place
// load-time stage-contiguous repack of Q8_0 weights; design + consumer list in mmq-q8-mf.cu.
struct ggml_cuda_q8ip_subst {
    ggml_tensor                copy[GGML_MAX_SRC];
    ggml_tensor *              orig[GGML_MAX_SRC] = {};
    ggml_cuda_pool_alloc<char> buf[GGML_MAX_SRC];
};
bool ggml_cuda_q8ip_any();
bool ggml_cuda_q8ip_is_packed(const ggml_tensor * t);
bool ggml_cuda_q8ip_want(const ggml_tensor * t, bool weights_buffer);
void ggml_cuda_q8ip_before_write(ggml_tensor * t, size_t offset, size_t size, cudaStream_t stream);
void ggml_cuda_q8ip_after_write(ggml_tensor * t, size_t offset, size_t size, bool weights_buffer, cudaStream_t stream);
void ggml_cuda_q8ip_forget_range(const void * base, size_t size);
bool ggml_cuda_q8ip_get(const ggml_tensor * t, void * data, size_t offset, size_t size);
bool ggml_cuda_q8ip_mmq_handles(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
bool ggml_cuda_q8ip_blas_handles(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
bool ggml_cuda_q8ip_mf_handles(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_q8ip_substitute(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_q8ip_subst & s);
void ggml_cuda_q8ip_restore(ggml_tensor * dst, ggml_cuda_q8ip_subst & s);
