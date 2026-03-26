#pragma once

// PolarQuant: polar coordinate KV cache quantization
//
// Implements PolarQuant (arXiv 2502.02617) for near-optimal KV cache compression.
//
// Algorithm:
//   1. Random preconditioning: x' = S × x (spreads outliers via JL lemma)
//   2. Recursive polar transform: Cartesian → (norm, angles) at log₂(d) levels
//   3. Non-uniform codebook quantization: k-means centroids on known angle distributions
//   4. Dequantization: reverse path with inverse preconditioning
//
// Block size: 128 elements (one KV head dimension)
// Compression: ~3.1 bits/element (50 bytes per 128 elements vs 256 bytes at f16)
// Quality: near-optimal distortion (within 2.7x of information-theoretic bound)

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QK_POLAR 128  // block size = KV head dimension

// Block layout (50 bytes total per 128 elements):
//   - norm:   f16 (2 bytes)
//   - lvl1:   64 angles × 4 bits = 32 bytes
//   - lvl2:   32 angles × 2 bits = 8 bytes
//   - lvl3:   16 angles × 2 bits = 4 bytes
//   - lvl4:   8 angles × 2 bits  = 2 bytes
//   - lvl567: 4+2+1 = 7 angles × 2 bits = 14 bits, packed into 2 bytes (2 bits pad)
//
// Total: 2 + 32 + 8 + 4 + 2 + 2 = 50 bytes
// Compression: 50/256 = 0.195 → 5.12x vs f16, 3.125 bits/element
typedef struct {
    uint16_t  norm;       // ‖S·x‖₂ stored as f16
    uint8_t   lvl1[32];   // 64 × 4-bit angle indices (level 1)
    uint8_t   lvl2[8];    // 32 × 2-bit angle indices (level 2)
    uint8_t   lvl3[4];    // 16 × 2-bit angle indices (level 3)
    uint8_t   lvl4[2];    //  8 × 2-bit angle indices (level 4)
    uint8_t   lvl567[2];  //  7 × 2-bit angle indices (levels 5-7, 2 bits pad)
} block_polar_q4;

#ifdef __cplusplus
static_assert(sizeof(block_polar_q4) == 50, "block_polar_q4 must be 50 bytes");
#else
_Static_assert(sizeof(block_polar_q4) == 50, "block_polar_q4 must be 50 bytes");
#endif

// Quantize: float[k] → block_polar_q4[k/QK_POLAR]
// k must be a multiple of QK_POLAR (128)
void quantize_row_polar_q4_ref(const float * GGML_RESTRICT x, block_polar_q4 * GGML_RESTRICT y, int64_t k);

// Dequantize: block_polar_q4[k/QK_POLAR] → float[k]
void dequantize_row_polar_q4(const block_polar_q4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Initialize PolarQuant preconditioning matrix and codebooks.
// Call once at startup. Thread-safe (uses internal once-flag).
void polar_quant_init(void);

#ifdef __cplusplus
}
#endif
