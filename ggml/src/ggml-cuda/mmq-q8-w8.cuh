#pragma once
// Q8-W8-GEMV-1 (GGML_HIP_Q8W8=1, default OFF): gfx90a Q8_0 x F32 GEMV at exactly ne11 = 8 (the served 27B DFlash2
// verify width). Full-block lanes (one 34 B Q8_0 block per lane, 2176 B per wave instruction triple), YT = 4 rows
// per wave, a 2/3-deep register pipeline that runs ACROSS row tiles, activations paged in LDS and loaded once per
// step for all 4 rows. Design: /mnt/raid0/llm/tmp/q8-w8-design-note-20261125.md and mmq-q8-w8.cu.
#include "common.cuh"

bool ggml_cuda_q8_w8_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_q8_w8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
