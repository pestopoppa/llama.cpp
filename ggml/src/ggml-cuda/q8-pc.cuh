#pragma once

// AK-ASM-I8GEMM-PERCHANNEL (gfx90a, experimental, DEFAULT OFF): per-token int8 x per-output-channel int8 GEMM for dense
// Q8_0 MUL_MAT at large batch, the consumer for the AITER I8gemm_bf16_perTokenI8 technique (asm_retarget lineage
// AK-ASM-H9 -> PORT-I8GEMM-C13 -> PERCHANNEL).
//
// Numerics: GGML_HIP_Q8_PC=1 re-quantizes every Q8_0 weight uploaded to a WEIGHTS buffer (except token_embd*) IN PLACE
// at model load to per-row int8: s_r = fp16(max|w_r|/127) (0 -> 1), q = clip(rint(w/s_r), -127, 127), stored as Q8_0
// blocks that all carry s_r. That is byte-for-byte the PC8EMU layout gated by ak-asm-i8gemm-perchannel-kld-gpu-20261121
// (27B KLD 0.001, dPPL +0.08 %), so every other Q8_0 consumer (MMVQ decode, MMQ, wvsk) keeps working on it unchanged.
// The GEMM then quantizes activations per token (s_a = max|x_n|/127, f32) and accumulates int32 over the WHOLE of K on
// v_mfma_i32_32x32x8i8, one f32 epilogue s_w[m]*s_a[n] (no per-32 flush: the H9 0.52x cost is gone).
//
// Knobs: GGML_HIP_Q8_PC=1 (off by default), GGML_HIP_Q8_PC_MIN_N (default 129),
//        GGML_HIP_Q8_PC_ASSUME=1 (test only: treat any Q8_0 src0 as already per-channel; block 0's d is the row scale).

#include "common.cuh"

bool ggml_cuda_q8_pc_enabled();

// load hook: call after every host->device write of a tensor
void ggml_cuda_q8_pc_after_write(ggml_tensor * t, size_t offset, size_t size, bool weights_buffer, cudaStream_t stream);
// buffer free/clear hook
void ggml_cuda_q8_pc_forget_range(const void * base, size_t size);

bool ggml_cuda_q8_pc_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// Returns true if handled; false = caller falls back to the existing route.
bool ggml_cuda_q8_pc_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
