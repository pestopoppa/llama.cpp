#pragma once

// TurboQuant: QJL attention score estimation with outlier correction
//
// K cache: 1-bit sign projections (256 bits) + top-8 outlier dimensions at f16
// For flash attention: computes Q·K scores directly from sign bits + outlier exact dot
//
// Block size: 128 elements (one KV head)
// Block: 58 bytes = 3.625 bits/element = 4.41x compression vs f16

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QK_TURBO 128
#define TURBO_SKETCH_DIM 256    // JL projection dimension
#define TURBO_NUM_OUTLIERS 8    // top-k outlier dimensions stored at full precision

// Block layout (58 bytes total = 3.625 bits/element):
//
//   norm:         f16 (2 bytes) — ‖key‖₂
//   signs:        256 bits (32 bytes) — sign(JL × key)
//   outlier_idx:  8 × uint8 (8 bytes) — dimension indices of top-8 outliers in JL sketch
//   outlier_val:  8 × f16 (16 bytes) — JL sketch values at outlier dimensions
//
typedef struct {
    uint16_t  norm;                           // ‖key‖₂ as f16
    uint8_t   signs[32];                      // 256 sign bits
    uint8_t   outlier_idx[TURBO_NUM_OUTLIERS]; // indices of top-8 outlier sketch dims
    uint16_t  outlier_val[TURBO_NUM_OUTLIERS]; // f16 sketch values at outlier dims
} block_turbo_q3;

#ifdef __cplusplus
static_assert(sizeof(block_turbo_q3) == 58, "block_turbo_q3 must be 58 bytes");
#else
_Static_assert(sizeof(block_turbo_q3) == 58, "block_turbo_q3 must be 58 bytes");
#endif

void quantize_row_turbo_q3_ref(const float * GGML_RESTRICT x, block_turbo_q3 * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo_q3(const block_turbo_q3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void turbo_quant_init(void);

// QJL score: project query, compute score from sign bits + outlier correction
void turbo_qjl_project_query(const float * q_float, uint8_t * q_signs, float * q_sketch);
// q_orig: original query vector (needed for outlier exact dot product)
float turbo_qjl_score_projected(const uint8_t * q_signs, const float * q_sketch,
                                 const float * q_orig,
                                 const block_turbo_q3 * k_block);
float turbo_qjl_score(const float * q_float, const block_turbo_q3 * k_block);

#ifdef __cplusplus
}
#endif
