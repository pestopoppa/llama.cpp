#include "ggml-turbo-quant.h"
#include "ggml-polar-quant.h"  // for polar_quant_init, S_matrix

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdatomic.h>

// ============================================================================
// JL Projection Matrix (QK_TURBO × TURBO_SKETCH_DIM, ±1 entries)
// ============================================================================

static float jl_matrix[QK_TURBO * TURBO_SKETCH_DIM];  // Gaussian JL (N(0,1/S))
static atomic_int turbo_initialized = 0;
static atomic_int turbo_initializing = 0;

static uint64_t turbo_prng_state;

static uint64_t turbo_prng_next(void) {
    turbo_prng_state ^= turbo_prng_state << 13;
    turbo_prng_state ^= turbo_prng_state >> 7;
    turbo_prng_state ^= turbo_prng_state << 17;
    return turbo_prng_state;
}

void turbo_quant_init(void) {
    if (atomic_load(&turbo_initialized)) return;

    int expected = 0;
    if (!atomic_compare_exchange_strong(&turbo_initializing, &expected, 1)) {
        while (!atomic_load(&turbo_initialized)) {}
        return;
    }

    // Generate deterministic Gaussian JL matrix: N(0, 1/S)
    // Using Box-Muller on the PRNG
    turbo_prng_state = 0x515A4C5475726251ULL;
    float inv_sqrt_s = 1.0f / sqrtf((float)TURBO_SKETCH_DIM);
    for (int i = 0; i < QK_TURBO * TURBO_SKETCH_DIM; i += 2) {
        double u1 = (double)(turbo_prng_next() >> 11) / (double)(1ULL << 53);
        double u2 = (double)(turbo_prng_next() >> 11) / (double)(1ULL << 53);
        if (u1 < 1e-15) u1 = 1e-15;
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * M_PI * u2;
        jl_matrix[i]     = (float)(r * cos(theta)) * inv_sqrt_s;
        if (i + 1 < QK_TURBO * TURBO_SKETCH_DIM) {
            jl_matrix[i + 1] = (float)(r * sin(theta)) * inv_sqrt_s;
        }
    }

    atomic_store(&turbo_initialized, 1);
}

// ============================================================================
// Quantize: compute JL sketch, extract sign bits and top-8 outliers
// ============================================================================

void quantize_row_turbo_q3_ref(const float * GGML_RESTRICT x, block_turbo_q3 * GGML_RESTRICT y, int64_t k) {
    if (!atomic_load(&turbo_initialized)) turbo_quant_init();

    const int nb = (int)(k / QK_TURBO);

    for (int b = 0; b < nb; b++) {
        const float * xb = x + b * QK_TURBO;
        block_turbo_q3 * yb = y + b;

        // Compute key norm
        float norm = 0.0f;
        for (int i = 0; i < QK_TURBO; i++) norm += xb[i] * xb[i];
        norm = sqrtf(norm);
        yb->norm = ggml_fp32_to_fp16(norm);

        // JL projection: sketch[s] = Σ jl[s][j] × x[j]  (jl entries ~ N(0, 1/S))
        float sketch[TURBO_SKETCH_DIM];
        for (int s = 0; s < TURBO_SKETCH_DIM; s++) {
            float dot = 0.0f;
            const float * row = jl_matrix + s * QK_TURBO;
            for (int j = 0; j < QK_TURBO; j++) {
                dot += row[j] * xb[j];
            }
            sketch[s] = dot;
        }

        // Find top-8 outlier dimensions in the ORIGINAL key vector (largest |x[j]|)
        // These are the dimensions that contribute most to dot products.
        // Storing them at full precision allows exact computation on the highest-impact dims.
        bool used[QK_TURBO];
        memset(used, 0, sizeof(used));

        for (int o = 0; o < TURBO_NUM_OUTLIERS; o++) {
            int best = 0;
            float best_abs = -1.0f;
            for (int j = 0; j < QK_TURBO; j++) {
                if (used[j]) continue;
                float a = fabsf(xb[j]);
                if (a > best_abs) { best_abs = a; best = j; }
            }
            used[best] = true;
            yb->outlier_idx[o] = (uint8_t)best;
            yb->outlier_val[o] = ggml_fp32_to_fp16(xb[best]);
        }

        // Sign bits for ALL dimensions (including outliers — outlier correction
        // is additive, so we keep the sign bits for completeness)
        memset(yb->signs, 0, 32);
        for (int s = 0; s < TURBO_SKETCH_DIM; s++) {
            if (sketch[s] >= 0.0f) {
                yb->signs[s / 8] |= (1 << (s % 8));
            }
        }
    }
}

// ============================================================================
// Dequantize: approximate reconstruction (for V-side use)
// Uses sign bits + outlier values to reconstruct sketch, then inverse JL
// ============================================================================

void dequantize_row_turbo_q3(const block_turbo_q3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    if (!atomic_load(&turbo_initialized)) turbo_quant_init();

    const int nb = (int)(k / QK_TURBO);

    for (int b = 0; b < nb; b++) {
        const block_turbo_q3 * xb = x + b;
        float * yb = y + b * QK_TURBO;

        float norm = ggml_fp16_to_fp32(xb->norm);
        if (norm == 0.0f || !isfinite(norm)) {
            memset(yb, 0, QK_TURBO * sizeof(float));
            continue;
        }

        // Reconstruct sketch from sign bits + outlier exact values
        float sketch[TURBO_SKETCH_DIM];

        // Sign-only reconstruction: each dim ≈ ±(norm/√(d×S))
        // JL entries ~ N(0, 1/S), so E[|sketch[s]|] ≈ norm × √(d/S) × √(2/π)
        float sign_mag = norm * sqrtf((float)QK_TURBO / (float)TURBO_SKETCH_DIM)
                             * sqrtf((float)(2.0 / M_PI));
        for (int s = 0; s < TURBO_SKETCH_DIM; s++) {
            int bit = (xb->signs[s / 8] >> (s % 8)) & 1;
            sketch[s] = bit ? sign_mag : -sign_mag;
        }

        // Override outlier dims with exact values
        for (int o = 0; o < TURBO_NUM_OUTLIERS; o++) {
            int idx = xb->outlier_idx[o];
            sketch[idx] = ggml_fp16_to_fp32(xb->outlier_val[o]);
        }

        // Pseudo-inverse: y ≈ S × JL^T × sketch (Gaussian JL: JL^T × JL ≈ (1/S) × I)
        memset(yb, 0, QK_TURBO * sizeof(float));
        for (int s = 0; s < TURBO_SKETCH_DIM; s++) {
            const float * row = jl_matrix + s * QK_TURBO;
            float val = sketch[s] * (float)TURBO_SKETCH_DIM; // unscale
            for (int j = 0; j < QK_TURBO; j++) {
                yb[j] += row[j] * val;
            }
        }
    }
}

// ============================================================================
// QJL Score Functions
// ============================================================================

void turbo_qjl_project_query(const float * q_float, uint8_t * q_signs, float * q_sketch) {
    if (!atomic_load(&turbo_initialized)) turbo_quant_init();

    memset(q_signs, 0, TURBO_SKETCH_DIM / 8);

    for (int s = 0; s < TURBO_SKETCH_DIM; s++) {
        float dot = 0.0f;
        const float * row = jl_matrix + s * QK_TURBO;
        for (int j = 0; j < QK_TURBO; j++) {
            dot += row[j] * q_float[j];
        }
        q_sketch[s] = dot;  // JL entries already scaled by 1/√S
        if (dot >= 0.0f) {
            q_signs[s / 8] |= (1 << (s % 8));
        }
    }
}

// Score = sign_contribution + outlier_correction
//
// sign_contribution: approximate <q,k> from sign agreement on non-outlier dims
// outlier_correction: exact dot product on outlier dims using stored f16 values
//
// The key insight: outlier dimensions have the largest |sketch| values and thus
// contribute the most variance to the sign-bit estimator. By handling them exactly,
// we dramatically reduce the noise on the remaining sign-bit estimate.
float turbo_qjl_score_projected(const uint8_t * q_signs, const float * q_sketch,
                                 const float * q_orig,
                                 const block_turbo_q3 * k_block) {
    float k_norm = ggml_fp16_to_fp32(k_block->norm);
    if (k_norm == 0.0f) return 0.0f;

    // QJL asymmetric estimator (paper Theorem 1):
    // <q, k> ≈ ‖k‖ × √(2/π) × Σ_s q_sketch[s] × sign(k_sketch[s])
    //
    // q_sketch[s] = (1/√S) × Σ_j JL[s][j] × q[j]  (full precision)
    // sign(k_sketch[s]) = k_block->signs bit s      (1-bit quantized)
    //
    // The √(2/π) factor corrects the bias from sign quantization:
    // E[x × sign(y)] = ‖y‖ × √(2/π) × cos(angle(x,y)) when y is Gaussian.

    float asym_dot = 0.0f;
    for (int s = 0; s < TURBO_SKETCH_DIM; s++) {
        int k_sign_bit = (k_block->signs[s / 8] >> (s % 8)) & 1;
        float k_sign = k_sign_bit ? 1.0f : -1.0f;
        asym_dot += q_sketch[s] * k_sign;
    }

    // Gaussian JL asymmetric estimator:
    // jl[s][j] ~ N(0, 1/S), q_sketch[s] = Σ jl[s][j] × q[j]
    // E[q_sketch[s] × sign(k_sketch[s])] = √(2/(πS)) × <q,k> / ‖k‖
    //   (from E[X × sign(Y)] = √(2/π) × cov(X,Y)/σ_Y for jointly Gaussian X,Y)
    //   Here: cov = <q,k>/S, σ_Y = ‖k‖/√S
    //   So E = √(2/π) × (<q,k>/S) / (‖k‖/√S) = √(2/π) × <q,k>/(‖k‖×√S)
    // Summing S terms: E[asym_dot] = S × √(2/π) × <q,k>/(‖k‖×√S) = √(2S/π) × <q,k>/‖k‖
    // Therefore: <q,k> = ‖k‖ × √(π/(2S)) × asym_dot
    float sign_score = k_norm * sqrtf((float)(M_PI / (2.0 * TURBO_SKETCH_DIM))) * asym_dot;

    // Outlier correction: exact dot product on top-8 original key dimensions
    float outlier_dot = 0.0f;
    for (int o = 0; o < TURBO_NUM_OUTLIERS; o++) {
        int idx = k_block->outlier_idx[o];
        float k_val = ggml_fp16_to_fp32(k_block->outlier_val[o]);
        outlier_dot += q_orig[idx] * k_val;
    }

    // Combined: sign_score already covers all dims. Add outlier as correction.
    return sign_score + outlier_dot;
}

float turbo_qjl_score(const float * q_float, const block_turbo_q3 * k_block) {
    uint8_t q_signs[TURBO_SKETCH_DIM / 8];
    float q_sketch[TURBO_SKETCH_DIM];
    turbo_qjl_project_query(q_float, q_signs, q_sketch);
    return turbo_qjl_score_projected(q_signs, q_sketch, q_float, k_block);
}
