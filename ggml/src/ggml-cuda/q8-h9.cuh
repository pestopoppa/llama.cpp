#pragma once

// AK-ASM-H9 (gfx90a, experimental): large-batch Q8_0 x q8_1 int8 MFMA GEMM that keeps Q8_0 numerics exactly
// (int32 partial per 32-k block, then f32 FMA with d_w*d_a per block), replacing dequant-to-f16 + rocBLAS for dense
// Q8_0 MUL_MAT at ne11 >= 129 on CDNA2.
//
// Why HIP and not the AITER asm: the retargeted gemm_a8w8_m128_noSplitK uses all 512 VGPRs (accum_offset 256) and the
// full 64 KiB of LDS and applies per-token x per-channel scales once after the whole K loop. A per-32 epilogue is a
// flush every 4 MFMAs inside the hand-scheduled main loop plus 64 f32 accumulators and two scale streams per tile:
// a rewrite of the loop and the register map, with no registers or LDS left to do it in.
//
// Techniques taken from the AITER kernels: v_mfma_i32_32x32x8i8, 128x128 tile / 4 waves x 64x64, LDS double buffer fed
// one stage ahead, coalesced cooperative global loads, ping-pong int32 accumulators so the per-block f32 rescale of
// block b overlaps the MFMAs of block b+1, sched_group_barrier DS/MFMA/VALU interleave.
//
// Kill switch: GGML_HIP_Q8_H9=0. Batch threshold: GGML_HIP_Q8_H9_MIN_N (default 129).

#include "common.cuh"

// Would H9 take this matmul?
bool ggml_cuda_q8_h9_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// Returns true if the matmul was handled; false = caller falls back to the existing route.
bool ggml_cuda_q8_h9_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
