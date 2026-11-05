#pragma once

// AK-ASM-H4 / AK-ASM-H6 (gfx90a, experimental).
//
// H4: pipelined int8 GEMM for Q8_0 weights x q8_1 activations at large batch (ne11 > 128 by default), replacing the
//     dequantize-to-f16 + rocBLAS route. 128x128 output tile per 256-thread workgroup, v_mfma_i32_32x32x8i8, LDS
//     double buffer fed from registers, sched_group_barrier interleave of VMEM/MFMA/VALU.
//     Kill switch: GGML_HIP_Q8_PIPE=0. Batch threshold: GGML_HIP_Q8_PIPE_MIN_N (default 129).
//
// H6: fused RMS_NORM + MUL (+ activation quantize into the plain block_q8_1 layout the MMVQ and H4 paths consume),
//     one kernel. The q8_1 copy is parked in a per-context slot that the next consumer of the MUL output reuses
//     instead of quantizing again. Kill switch: GGML_HIP_RMS_Q8=0.

#include "common.cuh"

// H4: returns true if the matmul was handled.
bool ggml_cuda_q8_pipe_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Would H4 take this matmul? (host-side predicate, also used by the H6 producer to decide whether a q8_1 copy has a consumer)
bool ggml_cuda_q8_pipe_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// H6: fused rms_norm + mul + q8_1 quantize. Returns false (and does nothing) if not applicable.
bool ggml_cuda_op_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i_rms);

// q8_1 slot (H6 producer -> MMVQ / H4 consumers).
const void * ggml_cuda_q8_1_slot_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t ne10_padded);
void         ggml_cuda_q8_1_slot_observe(ggml_backend_cuda_context & ctx, const ggml_tensor * node);
void         ggml_cuda_q8_1_slot_reset(ggml_backend_cuda_context & ctx);
void         ggml_cuda_q8_1_slot_release(ggml_backend_cuda_context * ctx);
