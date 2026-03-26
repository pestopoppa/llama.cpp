#include "ggml-polar-quant.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdatomic.h>

// ============================================================================
// Preconditioning matrix S (128×128 random orthogonal)
// ============================================================================

// Generated once via QR decomposition of deterministic Gaussian matrix.
// Stored row-major: S[i*128 + j].
float S_matrix[QK_POLAR * QK_POLAR];
float S_matrix_T[QK_POLAR * QK_POLAR]; // transpose = inverse (orthogonal)
static atomic_int polar_initialized = 0;
static atomic_int polar_initializing = 0;

// Deterministic PRNG for reproducible matrix generation (xoshiro256**)
static uint64_t prng_state[4];

static void prng_seed(uint64_t seed) {
    // SplitMix64 seeding
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        prng_state[i] = z ^ (z >> 31);
    }
}

static uint64_t prng_next(void) {
    const uint64_t result = ((prng_state[1] * 5) << 7 | (prng_state[1] * 5) >> 57) * 9;
    const uint64_t t = prng_state[1] << 17;
    prng_state[2] ^= prng_state[0];
    prng_state[3] ^= prng_state[1];
    prng_state[1] ^= prng_state[2];
    prng_state[0] ^= prng_state[3];
    prng_state[2] ^= t;
    prng_state[3] = (prng_state[3] << 45) | (prng_state[3] >> 19);
    return result;
}

// Box-Muller: two uniform → two Gaussian
static void prng_gaussian(float * out1, float * out2) {
    // Generate two uniform (0,1) from PRNG
    double u1 = (double)(prng_next() >> 11) / (double)(1ULL << 53);
    double u2 = (double)(prng_next() >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    *out1 = (float)(r * cos(theta));
    *out2 = (float)(r * sin(theta));
}

// QR decomposition via modified Gram-Schmidt.
// Input: A[n×n] row-major. Output: Q[n×n] orthogonal, row-major.
static void qr_orthogonalize(float * A, float * Q, int n) {
    // Work column-major internally for Gram-Schmidt
    float * cols = (float *)malloc(n * n * sizeof(float));

    // Transpose A (row-major) to cols (column-major)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cols[j * n + i] = A[i * n + j];

    // Modified Gram-Schmidt
    for (int j = 0; j < n; j++) {
        // Normalize column j
        float norm = 0.0f;
        for (int i = 0; i < n; i++) norm += cols[j * n + i] * cols[j * n + i];
        norm = sqrtf(norm);
        if (norm < 1e-10f) norm = 1e-10f;
        for (int i = 0; i < n; i++) cols[j * n + i] /= norm;

        // Subtract projection from remaining columns
        for (int k = j + 1; k < n; k++) {
            float dot = 0.0f;
            for (int i = 0; i < n; i++) dot += cols[j * n + i] * cols[k * n + i];
            for (int i = 0; i < n; i++) cols[k * n + i] -= dot * cols[j * n + i];
        }
    }

    // Transpose back to row-major
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Q[i * n + j] = cols[j * n + i];

    free(cols);
}

// ============================================================================
// Codebooks: pre-computed centroids for each level's angle distribution
// ============================================================================

// Level 1: 64 pairs, angles in [0, 2π), 4 bits → 16 centroids
// Levels 2-7: angles in [0, π/2], 2 bits → 4 centroids each
//
// The angle distribution at level ℓ follows:
//   f(ψ) ∝ sin^(2^(ℓ-1)-1)(2ψ) for ψ ∈ [0, π/2]  (levels ≥ 2)
//   uniform for level 1 (after preconditioning)
//
// For level 1 (uniform on [0, 2π)): centroids are just uniform intervals
// For levels 2+: we use precomputed centroids from 1-D k-means on the beta distribution

// Level 1: 16 centroids uniformly on [0, 2π)
static float codebook_lvl1[16];

// Levels 2-7: 4 centroids each on [0, π/2]
// Pre-computed via k-means on the known angle distribution.
// These are the optimal centroids for sin^(K-1)(2ψ) where K = 2^(ℓ-1).
static float codebook_lvl2[4]; // K=2:  sin^1(2ψ) → Beta(1, 1) → uniform-ish
static float codebook_lvl3[4]; // K=4:  sin^3(2ψ) → peaked
static float codebook_lvl4[4]; // K=8:  sin^7(2ψ) → sharply peaked at π/4
static float codebook_lvl5[4]; // K=16: sin^15(2ψ) → very sharp
static float codebook_lvl6[4]; // K=32: sin^31(2ψ) → near-delta at π/4
static float codebook_lvl7[4]; // K=64: sin^63(2ψ) → near-delta at π/4

static void init_codebooks(void) {
    // Level 1: uniform on [0, 2π)
    for (int i = 0; i < 16; i++) {
        codebook_lvl1[i] = (float)(2.0 * M_PI * (i + 0.5) / 16.0);
    }

    // Levels 2-7: optimal centroids for sin^(2^(l-1)-1)(2ψ) on [0, π/2].
    // As 2^(l-1) grows, the distribution concentrates at π/4.
    // Computed offline via numerical k-means on 1M samples from each distribution.

    // Codebook centroids computed via Lloyd's k-means on 500K samples from
    // the analytical angle distribution f(ψ) ∝ sin^(K-1)(2ψ), K = 2^(ℓ-1).
    // Sampling: ψ = atan2(√y, √x) where x,y ~ Gamma(K/2, 1).

    // Level 2: K=2, sin^1(2ψ), mean=0.785, std=0.342, MSE=0.0099
    codebook_lvl2[0] = 0.309332f;
    codebook_lvl2[1] = 0.633423f;
    codebook_lvl2[2] = 0.936184f;
    codebook_lvl2[3] = 1.259970f;

    // Level 3: K=4, sin^3(2ψ), mean=0.785, std=0.248, MSE=0.0062
    codebook_lvl3[0] = 0.425657f;
    codebook_lvl3[1] = 0.674394f;
    codebook_lvl3[2] = 0.896582f;
    codebook_lvl3[3] = 1.144649f;

    // Level 4: K=8, sin^7(2ψ), mean=0.786, std=0.176, MSE=0.0034
    codebook_lvl4[0] = 0.524371f;
    codebook_lvl4[1] = 0.706571f;
    codebook_lvl4[2] = 0.865686f;
    codebook_lvl4[3] = 1.046752f;

    // Level 5: K=16, sin^15(2ψ), mean=0.785, std=0.125, MSE=0.0018
    codebook_lvl5[0] = 0.598698f;
    codebook_lvl5[1] = 0.729143f;
    codebook_lvl5[2] = 0.841885f;
    codebook_lvl5[3] = 0.972014f;

    // Level 6: K=32, sin^31(2ψ), mean=0.786, std=0.089, MSE=0.0009
    codebook_lvl6[0] = 0.652555f;
    codebook_lvl6[1] = 0.745687f;
    codebook_lvl6[2] = 0.825683f;
    codebook_lvl6[3] = 0.918814f;

    // Level 7: K=64, sin^63(2ψ), mean=0.786, std=0.063, MSE=0.0005
    codebook_lvl7[0] = 0.691214f;
    codebook_lvl7[1] = 0.756897f;
    codebook_lvl7[2] = 0.813365f;
    codebook_lvl7[3] = 0.879360f;
}

// ============================================================================
// Initialization
// ============================================================================

void polar_quant_init(void) {
    if (atomic_load(&polar_initialized)) return;

    // Spin-lock: only one thread initializes
    int expected = 0;
    if (!atomic_compare_exchange_strong(&polar_initializing, &expected, 1)) {
        // Another thread is initializing — spin until done
        while (!atomic_load(&polar_initialized)) {
            // busy wait (init takes <100ms, only happens once)
        }
        return;
    }

    // Generate deterministic random orthogonal matrix S (128×128)
    float * gaussian = (float *)malloc(QK_POLAR * QK_POLAR * sizeof(float));
    prng_seed(0x506F6C6172517561ULL); // "PolarQua" as seed

    for (int i = 0; i < QK_POLAR * QK_POLAR; i += 2) {
        prng_gaussian(&gaussian[i], &gaussian[i + 1]);
    }

    qr_orthogonalize(gaussian, S_matrix, QK_POLAR);
    free(gaussian);

    // Compute transpose (= inverse for orthogonal matrix)
    for (int i = 0; i < QK_POLAR; i++)
        for (int j = 0; j < QK_POLAR; j++)
            S_matrix_T[i * QK_POLAR + j] = S_matrix[j * QK_POLAR + i];

    init_codebooks();
    atomic_store(&polar_initialized, 1);
}

// ============================================================================
// Helper: matrix-vector multiply y = M × x, both length QK_POLAR
// ============================================================================

static void matvec_128(const float * M, const float * x, float * y) {
    for (int i = 0; i < QK_POLAR; i++) {
        float sum = 0.0f;
        const float * row = M + i * QK_POLAR;
        for (int j = 0; j < QK_POLAR; j++) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

// ============================================================================
// Helper: find nearest codebook centroid index
// ============================================================================

static int nearest_centroid(float angle, const float * centroids, int n) {
    int best = 0;
    float best_dist = fabsf(angle - centroids[0]);
    for (int i = 1; i < n; i++) {
        float d = fabsf(angle - centroids[i]);
        if (d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

// For level 1 (circular): handle wraparound at 2π
static int nearest_centroid_circular(float angle, const float * centroids, int n) {
    int best = 0;
    float best_dist = 1e30f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(angle - centroids[i]);
        // Wraparound distance
        float d2 = (float)(2.0 * M_PI) - d;
        if (d2 < d) d = d2;
        if (d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

// ============================================================================
// Quantize: float[k] → block_polar_q4[k/128]
// ============================================================================

void quantize_row_polar_q4_ref(const float * GGML_RESTRICT x, block_polar_q4 * GGML_RESTRICT y, int64_t k) {
    if (!atomic_load(&polar_initialized)) polar_quant_init();

    // Debug: validate k is multiple of block size
    if (k % QK_POLAR != 0) {
        fprintf(stderr, "POLAR QUANT ERROR: k=%lld not multiple of %d\n", (long long)k, QK_POLAR);
        memset(y, 0, (k / QK_POLAR) * sizeof(block_polar_q4));
        return;
    }

    const int nb = (int)(k / QK_POLAR);

    for (int b = 0; b < nb; b++) {
        const float * xb = x + b * QK_POLAR;
        block_polar_q4 * yb = y + b;

        float precond[QK_POLAR];
        float radii[QK_POLAR];

        // Step 1: Preconditioning x' = S × x (random orthogonal, preserves norms/inner products)
        matvec_128(S_matrix, xb, precond);

        // Compute norm
        float norm = 0.0f;
        for (int i = 0; i < QK_POLAR; i++) norm += precond[i] * precond[i];
        norm = sqrtf(norm);
        yb->norm = ggml_fp32_to_fp16(norm);

        // Normalize
        if (norm > 1e-15f) {
            float inv_norm = 1.0f / norm;
            for (int i = 0; i < QK_POLAR; i++) precond[i] *= inv_norm;
        }

        // Step 2: Recursive polar transform
        // Level 1: d/2 = 64 pairs → atan2 angles in [0, 2π)
        // radii[i] = sqrt(precond[2i]^2 + precond[2i+1]^2)
        float angles_lvl1[64];
        for (int i = 0; i < 64; i++) {
            float a = precond[2 * i];
            float bb = precond[2 * i + 1];
            angles_lvl1[i] = atan2f(bb, a);
            if (angles_lvl1[i] < 0) angles_lvl1[i] += (float)(2.0 * M_PI);
            radii[i] = sqrtf(a * a + bb * bb);
        }

        // Quantize level 1: 64 × 4-bit indices
        memset(yb->lvl1, 0, 32);
        for (int i = 0; i < 64; i++) {
            int idx = nearest_centroid_circular(angles_lvl1[i], codebook_lvl1, 16);
            // Pack: two 4-bit values per byte
            yb->lvl1[i / 2] |= (uint8_t)(idx << (4 * (i % 2)));
        }

        // Levels 2-7: recursive halving
        // Each level: n_pairs pairs from radii → new radii of half size
        // angle = atan2(radii[2i+1], radii[2i]) in [0, π/2]
        int n_pairs = 32;  // level 2: 32 pairs from 64 radii
        float * cur_radii = radii;
        float next_radii[64];

        // Level 2: 32 × 2-bit
        memset(yb->lvl2, 0, 8);
        for (int i = 0; i < 32; i++) {
            float angle = atan2f(cur_radii[2 * i + 1], cur_radii[2 * i]);
            int idx = nearest_centroid(angle, codebook_lvl2, 4);
            yb->lvl2[i / 4] |= (uint8_t)(idx << (2 * (i % 4)));
            next_radii[i] = sqrtf(cur_radii[2*i]*cur_radii[2*i] + cur_radii[2*i+1]*cur_radii[2*i+1]);
        }
        memcpy(cur_radii, next_radii, 32 * sizeof(float));

        // Level 3: 16 × 2-bit
        memset(yb->lvl3, 0, 4);
        for (int i = 0; i < 16; i++) {
            float angle = atan2f(cur_radii[2 * i + 1], cur_radii[2 * i]);
            int idx = nearest_centroid(angle, codebook_lvl3, 4);
            yb->lvl3[i / 4] |= (uint8_t)(idx << (2 * (i % 4)));
            next_radii[i] = sqrtf(cur_radii[2*i]*cur_radii[2*i] + cur_radii[2*i+1]*cur_radii[2*i+1]);
        }
        memcpy(cur_radii, next_radii, 16 * sizeof(float));

        // Level 4: 8 × 2-bit
        memset(yb->lvl4, 0, 2);
        for (int i = 0; i < 8; i++) {
            float angle = atan2f(cur_radii[2 * i + 1], cur_radii[2 * i]);
            int idx = nearest_centroid(angle, codebook_lvl4, 4);
            yb->lvl4[i / 4] |= (uint8_t)(idx << (2 * (i % 4)));
            next_radii[i] = sqrtf(cur_radii[2*i]*cur_radii[2*i] + cur_radii[2*i+1]*cur_radii[2*i+1]);
        }
        memcpy(cur_radii, next_radii, 8 * sizeof(float));

        // Levels 5-7: 4+2+1 = 7 × 2-bit, packed into 2 bytes
        memset(yb->lvl567, 0, 2);

        // Level 5: 4 × 2-bit (bits 0-7 of lvl567)
        for (int i = 0; i < 4; i++) {
            float angle = atan2f(cur_radii[2 * i + 1], cur_radii[2 * i]);
            int idx = nearest_centroid(angle, codebook_lvl5, 4);
            yb->lvl567[0] |= (uint8_t)(idx << (2 * i));
            next_radii[i] = sqrtf(cur_radii[2*i]*cur_radii[2*i] + cur_radii[2*i+1]*cur_radii[2*i+1]);
        }
        memcpy(cur_radii, next_radii, 4 * sizeof(float));

        // Level 6: 2 × 2-bit (bits 0-3 of lvl567[1])
        for (int i = 0; i < 2; i++) {
            float angle = atan2f(cur_radii[2 * i + 1], cur_radii[2 * i]);
            int idx = nearest_centroid(angle, codebook_lvl6, 4);
            yb->lvl567[1] |= (uint8_t)(idx << (2 * i));
            next_radii[i] = sqrtf(cur_radii[2*i]*cur_radii[2*i] + cur_radii[2*i+1]*cur_radii[2*i+1]);
        }
        memcpy(cur_radii, next_radii, 2 * sizeof(float));

        // Level 7: 1 × 2-bit (bits 4-5 of lvl567[1])
        {
            float angle = atan2f(cur_radii[1], cur_radii[0]);
            int idx = nearest_centroid(angle, codebook_lvl7, 4);
            yb->lvl567[1] |= (uint8_t)(idx << 4);
        }
    }
}

// ============================================================================
// Dequantize: block_polar_q4[k/128] → float[k]
// ============================================================================

void dequantize_row_polar_q4(const block_polar_q4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    if (!atomic_load(&polar_initialized)) polar_quant_init();

    if (k % QK_POLAR != 0) {
        fprintf(stderr, "POLAR DEQUANT: k=%lld not multiple of %d!\n", (long long)k, QK_POLAR);
        memset(y, 0, k * sizeof(float));
        return;
    }

    const int nb = (int)(k / QK_POLAR);

    for (int b = 0; b < nb; b++) {
        const block_polar_q4 * xb = x + b;
        float * yb = y + b * QK_POLAR;

        float norm = ggml_fp16_to_fp32(xb->norm);

        // Guard: zero norm means unused/uninitialized KV slot → output zeros
        if (norm == 0.0f || !isfinite(norm)) {
            memset(yb, 0, QK_POLAR * sizeof(float));
            continue;
        }

        float radii[QK_POLAR];

        // Reconstruct from top (level 7) down to level 1

        // Level 7: 1 angle → 2 radii (start with unit radius at top)
        {
            int idx = (xb->lvl567[1] >> 4) & 0x3;
            float theta = codebook_lvl7[idx];
            radii[0] = cosf(theta);  // top-level radius = 1.0 (normalized)
            radii[1] = sinf(theta);
        }

        // Level 6: 2 angles → 4 radii
        {
            float next[4];
            for (int i = 0; i < 2; i++) {
                int idx = (xb->lvl567[1] >> (2 * i)) & 0x3;
                float theta = codebook_lvl6[idx];
                next[2 * i]     = radii[i] * cosf(theta);
                next[2 * i + 1] = radii[i] * sinf(theta);
            }
            memcpy(radii, next, 4 * sizeof(float));
        }

        // Level 5: 4 angles → 8 radii
        {
            float next[8];
            for (int i = 0; i < 4; i++) {
                int idx = (xb->lvl567[0] >> (2 * i)) & 0x3;
                float theta = codebook_lvl5[idx];
                next[2 * i]     = radii[i] * cosf(theta);
                next[2 * i + 1] = radii[i] * sinf(theta);
            }
            memcpy(radii, next, 8 * sizeof(float));
        }

        // Level 4: 8 angles → 16 radii
        {
            float next[16];
            for (int i = 0; i < 8; i++) {
                int idx = (xb->lvl4[i / 4] >> (2 * (i % 4))) & 0x3;
                float theta = codebook_lvl4[idx];
                next[2 * i]     = radii[i] * cosf(theta);
                next[2 * i + 1] = radii[i] * sinf(theta);
            }
            memcpy(radii, next, 16 * sizeof(float));
        }

        // Level 3: 16 angles → 32 radii
        {
            float next[32];
            for (int i = 0; i < 16; i++) {
                int idx = (xb->lvl3[i / 4] >> (2 * (i % 4))) & 0x3;
                float theta = codebook_lvl3[idx];
                next[2 * i]     = radii[i] * cosf(theta);
                next[2 * i + 1] = radii[i] * sinf(theta);
            }
            memcpy(radii, next, 32 * sizeof(float));
        }

        // Level 2: 32 angles → 64 radii
        {
            float next[64];
            for (int i = 0; i < 32; i++) {
                int idx = (xb->lvl2[i / 4] >> (2 * (i % 4))) & 0x3;
                float theta = codebook_lvl2[idx];
                next[2 * i]     = radii[i] * cosf(theta);
                next[2 * i + 1] = radii[i] * sinf(theta);
            }
            memcpy(radii, next, 64 * sizeof(float));
        }

        // Level 1: 64 angles → 128 Cartesian values
        {
            float cartesian[QK_POLAR];
            for (int i = 0; i < 64; i++) {
                int idx = (xb->lvl1[i / 2] >> (4 * (i % 2))) & 0xF;
                float theta = codebook_lvl1[idx];
                cartesian[2 * i]     = radii[i] * cosf(theta);
                cartesian[2 * i + 1] = radii[i] * sinf(theta);
            }

            // Inverse preconditioning: x = norm × S^T × cartesian
            matvec_128(S_matrix_T, cartesian, yb);
            for (int i = 0; i < QK_POLAR; i++) {
                yb[i] *= norm;
            }
        }
    }
}
