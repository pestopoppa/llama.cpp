//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// iqk port: MINIMAL replacement for ik_llama's iqk_quantize.h.
// The original declares ~33 ik-specific quant families (block_iq2_k, ktquants,
// 1bit, …) that v6 does not define. None of the vendored/compiled files
// (iqk_common.h, iqk_mul_mat.cpp, iqk_gemm_kquants.cpp, iqk_gemm_legacy_quants.cpp)
// actually USE those declarations — they were pulled in transitively. We keep only
// the standard q8 activation quantizers (the q8_2_x4 one is the activation format
// the Q4_K/Q8_0 kernels consume; it is defined in iqk_quantize_min.cpp).
//
#pragma once

#include <stdint.h>
#include <stddef.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#ifdef __cplusplus
#define GGML_RESTRICT
extern "C" {
#else
#define GGML_RESTRICT restrict
#endif

struct quantize_user_data;

void quantize_row_q8_0   (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1   (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1_x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_2_x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K   (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif
