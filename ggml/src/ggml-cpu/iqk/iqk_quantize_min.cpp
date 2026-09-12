//
// iqk port (minimal): ONLY the runtime activation quantizer required by iqk's
// Q4_K/Q8_0 GEMM kernels — quantize_row_q8_2_x4. Extracted verbatim from
// ik_llama ggml/src/iqk/iqk_quantize.cpp (Copyright (C) 2024-2025 Iwan Kawrakow,
// MIT). The rest of iqk_quantize.cpp (offline quant machinery) is intentionally
// NOT vendored.
//
#include "iqk_config.h"

#if defined IQK_IMPLEMENT

#include "ggml-impl.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "iqk_quantize.h"

#include <cassert>
#include <cstdint>
#include <type_traits>

#if defined(__x86_64__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {
#if !defined(__aarch64__)
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#endif

template <typename Block, typename Block_x4>
void quantize_row_q8_1_x4_T(const float * x, Block * y, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    const int nb4 = 4*(nb/4);
    Block_x4 * y4 = (Block_x4 *)y;
#if defined(__aarch64__)
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[QK8_1*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[QK8_1*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[QK8_1*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[QK8_1*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }

            accv = vaddq_s32(accv, vi);
        }

        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
            } else {
                y[i].s = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
            }
        } else {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_BF16(d * vaddvq_s32(accv)).bits;
            } else {
                y[i].s = GGML_FP32_TO_BF16(d * vaddvq_s32(accv)).bits;
            }
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
#if defined(__AVX512F__) && defined(__AVX512BW__)
#ifndef NDEBUG
        const float * x_block = x;
#endif
        __m512 v0 = _mm512_loadu_ps(x);
        __m512 v1 = _mm512_loadu_ps(x + 16);
        x += 32;

        const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
        const __m512 maxAbs = _mm512_max_ps(
            _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v0), abs_mask)),
            _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v1), abs_mask)));
        const float max_scalar = _mm512_reduce_max_ps(maxAbs);
#else
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float max_scalar = _mm_cvtss_f32( max4 );
#endif

        // Quantize these floats
        float d = max_scalar / 127.f;
        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
            } else {
                y[i].d = GGML_FP32_TO_FP16(d);
            }
        } else {
            auto t = GGML_FP32_TO_BF16(d);
            d = ggml_bf16_to_fp32(t);
            if (i < nb4) {
                y4[i4].d[ir] = t.bits;
            } else {
                y[i].d = t.bits;
            }
        }
        const float id = d > 0 ? 1/d : 0.f;
#if defined(__AVX512F__) && defined(__AVX512BW__)
        const __m512 mul = _mm512_set1_ps(id);

        v0 = _mm512_mul_ps(v0, mul);
        v1 = _mm512_mul_ps(v1, mul);

        v0 = _mm512_roundscale_ps(v0, _MM_FROUND_TO_NEAREST_INT);
        v1 = _mm512_roundscale_ps(v1, _MM_FROUND_TO_NEAREST_INT);

        const __m512i i0 = _mm512_cvtps_epi32(v0);
        const __m512i i1 = _mm512_cvtps_epi32(v1);

        const int isum = _mm512_reduce_add_epi32(_mm512_add_epi32(i0, i1));
#else
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Compute the sum of the quants and set y[i].s
        int isum = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
#endif
        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * isum);
            } else {
                y[i].s = GGML_FP32_TO_FP16(d * isum);
            }
        } else {
            if (i < nb4) {
                auto i16 = (int16_t *)y4[i4].d;
                i16[ir+4] = isum;
            } else {
                auto i16 = (int16_t *)&y[i].s;
                i16[0] = isum;
            }
        }

#if defined(__AVX512F__) && defined(__AVX512BW__)
        const __m128i q0 = _mm512_cvtsepi32_epi8(i0);
        const __m128i q1 = _mm512_cvtsepi32_epi8(i1);
        const __m256i quants = _mm256_set_m128i(q1, q0);

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, quants);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, quants);
        }

#ifndef NDEBUG
        {
            __m256 r0 = _mm256_loadu_ps(x_block);
            __m256 r1 = _mm256_loadu_ps(x_block + 8);
            __m256 r2 = _mm256_loadu_ps(x_block + 16);
            __m256 r3 = _mm256_loadu_ps(x_block + 24);

            const __m256 sign_bit = _mm256_set1_ps(-0.0f);
            __m256 ref_max_abs = _mm256_andnot_ps(sign_bit, r0);
            ref_max_abs = _mm256_max_ps(ref_max_abs, _mm256_andnot_ps(sign_bit, r1));
            ref_max_abs = _mm256_max_ps(ref_max_abs, _mm256_andnot_ps(sign_bit, r2));
            ref_max_abs = _mm256_max_ps(ref_max_abs, _mm256_andnot_ps(sign_bit, r3));

            __m128 ref_max4 = _mm_max_ps(
                _mm256_extractf128_ps(ref_max_abs, 1), _mm256_castps256_ps128(ref_max_abs));
            ref_max4 = _mm_max_ps(ref_max4, _mm_movehl_ps(ref_max4, ref_max4));
            ref_max4 = _mm_max_ss(ref_max4, _mm_movehdup_ps(ref_max4));
            float ref_d = _mm_cvtss_f32(ref_max4) / 127.f;

            uint16_t ref_d_bits;
            if constexpr (std::is_same_v<Block, block_q8_1>) {
                ref_d_bits = GGML_FP32_TO_FP16(ref_d);
            } else {
                auto t = GGML_FP32_TO_BF16(ref_d);
                ref_d_bits = t.bits;
                ref_d = ggml_bf16_to_fp32(t);
            }

            const __m256 ref_mul = _mm256_set1_ps(ref_d > 0 ? 1/ref_d : 0.f);
            r0 = _mm256_round_ps(_mm256_mul_ps(r0, ref_mul), _MM_ROUND_NEAREST);
            r1 = _mm256_round_ps(_mm256_mul_ps(r1, ref_mul), _MM_ROUND_NEAREST);
            r2 = _mm256_round_ps(_mm256_mul_ps(r2, ref_mul), _MM_ROUND_NEAREST);
            r3 = _mm256_round_ps(_mm256_mul_ps(r3, ref_mul), _MM_ROUND_NEAREST);

            __m256i ri0 = _mm256_cvtps_epi32(r0);
            __m256i ri1 = _mm256_cvtps_epi32(r1);
            __m256i ri2 = _mm256_cvtps_epi32(r2);
            __m256i ri3 = _mm256_cvtps_epi32(r3);
            const int ref_isum = hsum_i32_8(
                _mm256_add_epi32(_mm256_add_epi32(ri0, ri1), _mm256_add_epi32(ri2, ri3)));

            ri0 = _mm256_packs_epi32(ri0, ri1);
            ri2 = _mm256_packs_epi32(ri2, ri3);
            ri0 = _mm256_packs_epi16(ri0, ri2);
            const __m256i ref_perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            ri0 = _mm256_permutevar8x32_epi32(ri0, ref_perm);

            alignas(32) int8_t ref_qs[QK8_1];
            _mm256_store_si256((__m256i *)ref_qs, ri0);
            const uint16_t ref_s_bits = std::is_same_v<Block, block_q8_1>
                ? GGML_FP32_TO_FP16(ref_d * ref_isum)
                : (uint16_t)(int16_t)ref_isum;
            const uint16_t actual_d = i < nb4 ? y4[i4].d[ir] : y[i].d;
            const uint16_t actual_s = i < nb4 ? y4[i4].d[ir+4] : y[i].s;
            const int8_t * actual_qs = i < nb4 ? y4[i4].qs + QK8_1*ir : y[i].qs;

            assert(actual_d == ref_d_bits);
            assert(actual_s == ref_s_bits);
            for (int j = 0; j < QK8_1; ++j) {
                assert(actual_qs[j] == ref_qs[j]);
            }
        }
#endif
#else
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );
        i2 = _mm256_packs_epi32( i2, i3 );
        i0 = _mm256_packs_epi16( i0, i2 );

        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
#endif
    }
#endif
}
} // namespace

extern "C" void quantize_row_q8_2_x4(const float * x, void * vy, int64_t k) {
    quantize_row_q8_1_x4_T<block_q8_2, block_q8_2_x4>(x, (block_q8_2 *)vy, k);
}

#endif // IQK_IMPLEMENT
