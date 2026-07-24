//
// iqk port: ik_llama-specific block structs (repacked _r*/_x4, q6_0) that v6's
// ggml-common.h lacks. Referenced by the kquants/legacy kernels as DEAD paths.
// Auto-extracted verbatim from ik_llama ggml/src/ggml-common.h (MIT, I. Kawrakow).
//
#pragma once
#include <stdint.h>
#ifndef QK6_0
#define QK6_0 32
#endif

#ifndef IQK_HAVE_BLOCK_IQ2_XXS_R4
#define IQK_HAVE_BLOCK_IQ2_XXS_R4
typedef struct {
    ggml_half d[4];
    uint8_t   sas[QK_K/2];
    uint8_t   qs[QK_K/2];
} block_iq2_xxs_r4;
static_assert(sizeof(block_iq2_xxs_r4) == 4*sizeof(block_iq2_xxs), "wrong iq2_xxs_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_IQ2_XS_R4
#define IQK_HAVE_BLOCK_IQ2_XS_R4
typedef struct {
    ggml_half d[4];
    uint16_t qs[QK_K/2];
    uint8_t  scales[QK_K/8];
} block_iq2_xs_r4;
static_assert(sizeof(block_iq2_xs_r4) == 4*sizeof(block_iq2_xs), "wrong iq2_xs_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_IQ2_S_R4
#define IQK_HAVE_BLOCK_IQ2_S_R4
typedef struct {
    ggml_half d[4];
    uint8_t qs[QK_K/2];
    uint8_t qh[QK_K/8];
    uint8_t signs[QK_K/2];
    uint8_t scales[QK_K/8];
} block_iq2_s_r4;
static_assert(sizeof(block_iq2_s_r4) == 4*sizeof(block_iq2_s), "wrong iq2_s_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_IQ3_XXS_R4
#define IQK_HAVE_BLOCK_IQ3_XXS_R4
typedef struct {
    ggml_half d[4];
    uint8_t   sas[QK_K/2];
    uint8_t   qs[QK_K];
} block_iq3_xxs_r4;
static_assert(sizeof(block_iq3_xxs_r4) == 4*sizeof(block_iq3_xxs), "wrong iq3_xxs_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_IQ3_S_R4
#define IQK_HAVE_BLOCK_IQ3_S_R4
typedef struct {
    ggml_half d[4];
    uint8_t qs[QK_K];
    uint8_t qh[QK_K/8];
    uint8_t signs[QK_K/2];
    uint8_t scales[4*IQ3S_N_SCALE];
} block_iq3_s_r4;
static_assert(sizeof(block_iq3_s_r4) == 4*sizeof(block_iq3_s), "wrong iq3_s_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_IQ4_NL_R4
#define IQK_HAVE_BLOCK_IQ4_NL_R4
typedef struct {
    ggml_half d[4];
    uint8_t qs[2*QK4_NL];
} block_iq4_nl_r4;
static_assert(sizeof(block_iq4_nl_r4) == 4*sizeof(ggml_half) + 2*QK4_NL, "wrong iq4_nl_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_IQ4_NL_R8
#define IQK_HAVE_BLOCK_IQ4_NL_R8
typedef struct {
    ggml_half d[8];
    uint8_t qs[4*QK4_NL];
} block_iq4_nl_r8;
static_assert(sizeof(block_iq4_nl_r8) == 8*sizeof(ggml_half) + 4*QK4_NL, "wrong iq4_nl_r8 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_IQ4_XS_R8
#define IQK_HAVE_BLOCK_IQ4_XS_R8
typedef struct {
    ggml_half d[8];
    uint8_t scales_h[QK_K/16];
    uint8_t scales_l[QK_K/ 8];
    uint8_t qs[QK_K*4];
} block_iq4_xs_r8;
static_assert(sizeof(block_iq4_xs_r8) == 8*sizeof(block_iq4_xs), "wrong iq4_xs_rs block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q2_K_R4
#define IQK_HAVE_BLOCK_Q2_K_R4
typedef struct {
    ggml_half d[8];
    uint8_t scales[QK_K/4]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K];       // quants
} block_q2_k_r4;
static_assert(sizeof(block_q2_k_r4) == 8*sizeof(ggml_half) + QK_K/4 + QK_K, "wrong q2_k_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q3_K_R4
#define IQK_HAVE_BLOCK_Q3_K_R4
typedef struct {
    ggml_half d[4];             // super-block scales
    uint8_t scales_h[QK_K/16];  // scales quantized with 6 bits (high 2 bits)
    uint8_t scales_l[QK_K/8];   // scales quantized with 6 bits (low  4 bits)
    uint8_t qh[QK_K/2];         // quants - high bit
    uint8_t qs[QK_K];           // quants - low 2 bits
} block_q3_k_r4;
static_assert(sizeof(block_q3_k_r4) == 4*sizeof(ggml_half) + QK_K/16 + QK_K/8 + QK_K/2 + QK_K, "wrong q3_k_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q4_K_R4
#define IQK_HAVE_BLOCK_Q4_K_R4
typedef struct {
    ggml_half d[8];
    uint8_t scales_h[QK_K/16];// scales and mins, quantized with 6 bits
    uint8_t scales_l[QK_K/8]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K*2];           // 4--bit quants
} block_q4_k_r4;
static_assert(sizeof(block_q4_k_r4) == 8*sizeof(ggml_half) + QK_K/16 + QK_K/8 + QK_K*2, "wrong q4_k_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q5_0_R4
#define IQK_HAVE_BLOCK_Q5_0_R4
typedef struct {
    ggml_half d[4];        // delta
    uint8_t qh[QK5_0/2];   // 5-th bit of quants
    uint8_t qs[QK5_0*2];   // nibbles / quants
} block_q5_0_r4;
static_assert(sizeof(block_q5_0_r4) == 4*sizeof(ggml_half) + QK5_0*2 + QK5_0/2, "wrong q5_0_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q5_K_R4
#define IQK_HAVE_BLOCK_Q5_K_R4
typedef struct {
    ggml_half d[8];
    uint8_t scales_h[QK_K/16];// scales and mins, quantized with 6 bits
    uint8_t scales_l[QK_K/8]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/2];           // quants, high bit
    uint8_t qs[QK_K*2];           // quants, low 4 bits
} block_q5_k_r4;
static_assert(sizeof(block_q5_k_r4) == 8*sizeof(ggml_half) + QK_K/16 + QK_K/8 + QK_K/2 + QK_K*2, "wrong q5_k_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q6_0
#define IQK_HAVE_BLOCK_Q6_0
typedef struct {
    ggml_half d;         // delta
    uint8_t qh[QK6_0/4]; // 5+6-th bit of quants
    uint8_t qs[QK6_0/2]; // nibbles / quants
} block_q6_0;
static_assert(sizeof(block_q6_0) == sizeof(ggml_half) + QK6_0/2 + QK6_0/4, "wrong q6_0 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q6_0_R4
#define IQK_HAVE_BLOCK_Q6_0_R4
typedef struct {
    ggml_half d[4];      // delta
    uint8_t qh[QK6_0];   // 5+6-th bit of quants
    uint8_t qs[QK6_0*2]; // nibbles / quants
} block_q6_0_r4;
static_assert(sizeof(block_q6_0_r4) == 4*sizeof(ggml_half) + QK6_0*2 + QK6_0, "wrong q6_0_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q6_K_R4
#define IQK_HAVE_BLOCK_Q6_K_R4
typedef struct {
    ggml_half d[4];          // super-block scale
    int8_t  scales[QK_K/4];  // scales, quantized with 8 bits
    uint8_t qh[QK_K];        // quants, upper 2 bits
    uint8_t ql[QK_K*2];      // quants, lower 4 bits
} block_q6_k_r4;
static_assert(sizeof(block_q6_k_r4) == 4*sizeof(ggml_half) + QK_K/4 + 3*QK_K, "wrong q6_k_r4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q8_0_R8
#define IQK_HAVE_BLOCK_Q8_0_R8
typedef struct {
    ggml_half d[8];
    int8_t qs[8*QK8_0];
} block_q8_0_r8;
static_assert(sizeof(block_q8_0_r8) == 8*sizeof(block_q8_0), "wrong q8_0_r8 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q8_0_X4
#define IQK_HAVE_BLOCK_Q8_0_X4
typedef struct {
    ggml_half d[4];
    int8_t qs[4*QK8_0];
} block_q8_0_x4;
static_assert(sizeof(block_q8_0_x4) == 4*sizeof(block_q8_0), "wrong q8_0_x4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q8_1_X4
#define IQK_HAVE_BLOCK_Q8_1_X4
typedef struct {
    ggml_half d[8];
    int8_t qs[4*QK8_1];
} block_q8_1_x4;
static_assert(sizeof(block_q8_1_x4) == 4*sizeof(block_q8_1), "wrong q8_1_x4 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q8_K_R16
#define IQK_HAVE_BLOCK_Q8_K_R16
typedef struct {
    ggml_half d[16];         // delta
    int8_t    qs[16*QK_K];   // quants, stored as unsigned ints
} block_q8_k_r16;
static_assert(sizeof(block_q8_k_r16) == 16*sizeof(ggml_half) + 16*QK_K, "wrong q8_k_r16 block size/padding");
#endif

#ifndef IQK_HAVE_BLOCK_Q8_K_R8
#define IQK_HAVE_BLOCK_Q8_K_R8
typedef struct {
    ggml_half d[8];         // delta
    int8_t    qs[8*QK_K];   // quants, stored as unsigned ints
} block_q8_k_r8;
static_assert(sizeof(block_q8_k_r8) == 8*sizeof(ggml_half) + 8*QK_K, "wrong q8_k_r8 block size/padding");
#endif

// iqk port: v6's block_q8_K lacks ik's precomputed `.sum` field (ik = {d, sum, qs, bsums};
// v6 = {d, qs, bsums}). Compute it from bsums for the (dead-for-us, Q8_K-activation)
// kernel paths. Σ bsums == Σ qs == ik's .sum.
static inline float iqk_block_q8_K_sum(const block_q8_K & b) {
    int s = 0;
    for (int i = 0; i < QK_K/16; ++i) s += b.bsums[i];
    return (float) s;
}

// iqk port: IQ4 nonlinear lookup table (used only by dead IQ4_K/trellis paths).
static const int8_t iq4k_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
