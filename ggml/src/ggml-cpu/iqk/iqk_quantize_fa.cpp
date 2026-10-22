//
// iqk port (AKX-ALL-LB4): activation quantizers needed ONLY by the vendored iqk flash-attention
// templates (fa/iqk_fa_templates.h): quantize_row_q8_0_x4 (Q rows for a q8_0 K cache) and
// quantize_row_q8_KV (Q rows for a q8_KV K cache; that K type never occurs in this tree's KV
// cache, the function exists so the template instantiations link). Function bodies extracted
// verbatim from ik_llama ggml/src/iqk/iqk_quantize.cpp @ c04881fc0 (lines 807-919 and 3971-4066;
// Copyright (C) 2024-2025 Iwan Kawrakow, MIT). Compiled only with -DGGML_IQK_FLASH_ATTENTION=ON.
// Like iqk_quantize_min.cpp, the rest of iqk_quantize.cpp is intentionally NOT vendored.
//
#include "iqk_config.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include "iqk_common.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {
// verbatim from ik_llama iqk_quantize.cpp
inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

void iqk_quantize_row_q8_KV_fa(const float * x, void * vy, int64_t k) {
    assert(k % 32 == 0);
    auto dptr = (float *)vy;
    auto q8 = (int8_t *)(dptr + 2);
#ifdef __AVX2__
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    __m256 maxAbs = _mm256_setzero_ps();
    for (int ib = 0; ib < k/8; ++ib) {
        const __m256 v = _mm256_loadu_ps(x + 8*ib);
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
    }
    const float maxScalar = hmax_f32_8(maxAbs);
    if (!maxScalar) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = maxScalar / 127.f;
    auto mul = _mm256_set1_ps(1/dptr[0]);
    auto isum = _mm256_setzero_si256();
    for (int i = 0; i < k/32; i++) {
        __m256 v0 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i +  0));
        __m256 v1 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i +  8));
        __m256 v2 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i + 16));
        __m256 v3 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i + 24));
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        isum = _mm256_add_epi32(isum, _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        i0 = _mm256_packs_epi32( i0, i1 );
        i2 = _mm256_packs_epi32( i2, i3 );
        i0 = _mm256_packs_epi16( i0, i2 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );
        _mm256_storeu_si256((__m256i *)q8, i0);
        q8 += 32;
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = hsum_i32_8(isum);
#elif defined __ARM_NEON
    int32x4_t ival[8];
    auto vmax = vdupq_n_f32(0.f);
    for (int j = 0; j < k; j += 4) {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + j)));
    }
    auto smax = vmaxvq_f32(vmax);
    if (!smax) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = smax/127;
    auto vid = vdupq_n_f32(1/dptr[0]);
    auto isum = vdupq_n_s32(0);
    for (int ib = 0; ib < k/32; ++ib) {
        auto xb = x + 32*ib;
        for (int k = 0; k < 8; ++k) {
            auto val = vld1q_f32(xb + 4*k);
            ival[k] = vcvtnq_s32_f32(vmulq_f32(val, vid));
            isum = vaddq_s32(isum, ival[k]);
        }
        for (int k = 0; k < 4; ++k) {
            auto i16 = vcombine_s16(vmovn_s32(ival[2*k+0]), vmovn_s32(ival[2*k+1]));
            vst1_s8(q8, vmovn_s16(i16));
            q8 += 8;
        }
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = vaddvq_s32(isum);
#else
    float amax = 0;
    for (int j = 0; j < k; ++j) {
        float ax = std::abs(x[j]);
        amax = std::max(amax, ax);
    }
    if (!amax) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = amax/127;
    float id = 1/dptr[0];
    int isum = 0;
    for (int i = 0; i < k; i++) {
        q8[i] = nearest_int(id*x[i]);
        isum += q8[i];
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = isum;
#endif
}
} // namespace

extern "C" {

void quantize_row_q8_0_x4(const float * x, void * vy, int64_t k) {
    const int nb = k / QK8_0;
    const int nb4 = 4*(nb/4);

    block_q8_0    * y  = (block_q8_0    *)vy;
    block_q8_0_x4 * y4 = (block_q8_0_x4 *)vy;
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

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[32*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[32*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[32*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[32*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        const float d = maxScalar / 127.f;
        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
#endif
}

void quantize_row_q8_KV(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_KV_fa(x, vy, k);
}

} // extern "C"

#endif // IQK_IMPLEMENT && GGML_IQK_FLASH_ATTENTION
