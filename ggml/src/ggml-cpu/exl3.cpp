// EXL3 (exllamav3 trellis, mul1 codebook) matrix executor for the ggml CPU backend.
//
// Design and correctness contract: EXL3-CPU-DESIGN.md (repository root).
//
// Provenance / licences:
//  * Format facts and the scalar reference follow the project's EXL3-1 oracle
//    (epyc-inference-research scripts/kernel_rnd/exl3, bc9ac47b) and EXL3-2 operators.
//  * The AVX-512 VNNI (dword) and VBMI (byte) tile-row state extraction -- make_tc_perm and the
//    per-(bits,row) index/shift tables, dword_gather/dword_codes, gather_row_bytes/shift_mask_row,
//    byte-level row pairing -- is ported from exllamav3 exllamav3_ext/cpu/moe_mul1.cpp
//    (turboderp, MIT License, Copyright (c) 2025 Turboderp; full text in exllamav3/LICENSE):
//      Permission is hereby granted, free of charge, to any person obtaining a copy of this
//      software and associated documentation files (the "Software"), to deal in the Software
//      without restriction, including without limitation the rights to use, copy, modify,
//      merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
//      permit persons to whom the Software is furnished to do so, subject to the following
//      conditions: The above copyright notice and this permission notice shall be included in
//      all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
//      WITHOUT WARRANTY OF ANY KIND.
//  * The MUL_MAT/MUL_MAT_ID-with-aux-sources integration convention follows buun-llama.cpp
//    (MIT, the ggml authors), engine 1d0f493c, ggml/src/ggml-cpu/exl3.cpp.
//
// Arithmetic (all ISA tiers bit-identical to the scalar reference in this file):
//   prep:  x' = H128(suh (.) x) * (1/sqrt128) in fp32; per 128-input block b: symmetric int8,
//          scale_b = amax_b/127, RNE, clamp +-127; f_b = k_inv*scale_b;
//          corr = c0 * sum_b scale_b * sum(x8_b)   (c0 = 1024*k_inv + fp16(0xC931), exact)
//   GEMV:  acc_b[c] = sum_{k in b} bytesum(state(k,c)*0x83DCD12D) * x8[k]   (int32, exact)
//          accf[c] = fma(float(acc_b[c]), f_b, accf[c]) in ascending b;  y' = accf + corr
//   GEMM:  W = canonical fp16 codebook (exact), y'[c] = fma chain over ascending k of x'[k]*W[k][c]
//   out:   y = (H128(y') * (1/sqrt128)) * svh     per 128-output block (butterfly order fixed)

#include "exl3.h"

#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define EXL3_X86 1
#else
#define EXL3_X86 0
#endif

namespace {

constexpr uint32_t MUL1_MULT = 0x83DCD12Du;
constexpr float    K_INV     = 1774.0f / 262144.0f;          // fp16 0x1EEE
constexpr float    C_HALF    = -10.3828125f;                 // fp16 0xC931
constexpr float    C0        = 1024.0f * K_INV + C_HALF;     // -3.453125 (exact)
constexpr float    HAD_SCALE = 0.08838834764831845f;         // float(1/sqrt(128))
constexpr int      HB        = 128;                          // Hadamard block
constexpr int      MAX_M     = 4;                            // rows per GEMV kernel call
constexpr int      KC        = 256;                          // GEMM k-chunk (inputs)

static_assert(C0 == -3.453125f, "c0 must be exact");

int g_gemv_max_rows = 8;

// GEMV split-K (decode / verify): ops with few 128-output bands (attn_k/v, ssm_out, MoE gate/up
// at N=1, shared expert) would leave most threads idle, so the k range is split into S slices
// whose raw partials are reduced in a fixed order after a barrier. S depends only on the op's
// shape and routing (never on the thread count), so results stay thread-count independent.
constexpr int64_t SPLITK_MAX_INST = 64;
int g_splitk_target = []{
    const char * e = std::getenv("GGML_EXL3_SPLITK_TARGET");
    return e ? std::max(1, std::atoi(e)) : 192;
}();

enum class Isa { scalar = 0, vnni = 1, vbmi = 2 };

// -------------------------------------------------------------------------------------------
// format tables (exllamav3 make_tc_perm & friends)
// -------------------------------------------------------------------------------------------

constexpr std::array<uint16_t, 256> make_tc_perm() {
    std::array<uint16_t, 256> p{};
    for (int t = 0; t < 32; ++t) {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4, c1 = c0 + 8;
        p[t * 8 + 0] = r0 * 16 + c0; p[t * 8 + 1] = r1 * 16 + c0;
        p[t * 8 + 2] = r2 * 16 + c0; p[t * 8 + 3] = r3 * 16 + c0;
        p[t * 8 + 4] = r0 * 16 + c1; p[t * 8 + 5] = r1 * 16 + c1;
        p[t * 8 + 6] = r2 * 16 + c1; p[t * 8 + 7] = r3 * 16 + c1;
    }
    return p;
}

constexpr std::array<uint16_t, 256> make_tc_perm_inv() {
    std::array<uint16_t, 256> inv{};
    const auto perm = make_tc_perm();
    for (int i = 0; i < 256; ++i) inv[perm[i]] = i;
    return inv;
}

constexpr auto TC_PERM_INV = make_tc_perm_inv();

template <int bits, int row, bool second_word>
constexpr std::array<int32_t, 16> make_row_indices() {
    std::array<int32_t, 16> idx{};
    const auto inv = make_tc_perm_inv();
    constexpr int words32 = bits * 256 / 32;
    for (int col = 0; col < 16; ++col) {
        const int t  = inv[row * 16 + col];
        const int b0 = t * bits + bits - 16 + 256 * bits;
        const int b1 = b0 + 16;
        idx[col] = (second_word ? (b1 - 1) / 32 : b0 / 32) % words32;
    }
    return idx;
}

template <int bits, int row, bool second_word>
constexpr uint16_t make_row_himask() {
    uint16_t mask = 0;
    const auto idx = make_row_indices<bits, row, second_word>();
    for (int col = 0; col < 16; ++col) {
        if (idx[col] >= 32) mask |= uint16_t(1) << col;
    }
    return mask;
}

template <int bits, int row>
constexpr int row_shift(int col) {
    const auto inv = make_tc_perm_inv();
    const int t  = inv[row * 16 + col];
    const int b1 = t * bits + bits + 256 * bits;
    return ((b1 - 1) / 32 + 1) * 32 - b1;
}

template <int s0, int s1>
constexpr std::array<int32_t, 16> make_lane_shifts() {
    std::array<int32_t, 16> v{};
    for (int i = 0; i < 16; ++i) v[i] = i < 8 ? s0 : s1;
    return v;
}

template <int bits, int row>
constexpr std::array<uint8_t, 64> make_row_byte_indices() {
    std::array<uint8_t, 64> idx{};
    const auto inv = make_tc_perm_inv();
    constexpr int words32 = bits * 256 / 32;
    for (int col = 0; col < 16; ++col) {
        const int t  = inv[row * 16 + col];
        const int b0 = t * bits + bits - 16 + 256 * bits;
        const int b1 = b0 + 16;
        const int w0 = (b0 / 32) % words32;
        const int w1 = ((b1 - 1) / 32) % words32;
        const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
        const int fb = shift / 8;
        for (int byte = 0; byte < 3; ++byte) {
            const int mb  = fb + byte;
            const int src = mb < 4 ? w1 * 4 + mb : w0 * 4 + (mb - 4);
            idx[col * 4 + byte] = static_cast<uint8_t>(src);
        }
        idx[col * 4 + 3] = 0;
    }
    return idx;
}

template <int bits, int row>
constexpr uint64_t make_row_byte_himask() {
    const auto idx = make_row_byte_indices<bits, row>();
    uint64_t m = 0;
    for (int i = 0; i < 64; ++i) {
        if (idx[i] >= 128) m |= uint64_t(1) << i;
    }
    return m;
}

template <int bits, int row>
constexpr bool byte_pair_ok() {
    for (int col = 0; col < 16; ++col) {
        if (row_shift<bits, row>(col) % 8 < bits) return false;
    }
    return true;
}

// -------------------------------------------------------------------------------------------
// scalar primitives (the reference)
// -------------------------------------------------------------------------------------------

inline uint16_t state_scalar(const uint16_t * tile, int bits, int t) {
    const int len   = 256 * bits;
    const int start = ((t + 1) * bits - 16 + len) % len;
    const int a     = start / 32;
    const int off   = start % 32;
    uint32_t w0, w1;
    std::memcpy(&w0, tile + 2 * a, 4);
    std::memcpy(&w1, tile + 2 * ((a + 1) % (8 * bits)), 4);
    return uint16_t(((uint64_t(w0) << 32) | w1) >> (48 - off));
}

inline int bytesum_mul1(uint32_t state) {
    const uint32_t x = state * MUL1_MULT;
    return int(x & 255) + int((x >> 8) & 255) + int((x >> 16) & 255) + int(x >> 24);
}

inline float fp16_round(float x) {
    return GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(x));
}

// canonical codebook value (exactly the one-rounding fp16 FMA of the format)
inline float mul1_canonical(int bs) {
    return fp16_round(std::fma(float(1024 + bs), K_INV, C_HALF));
}

inline void had128(float * v) {
    for (int w = 1; w < HB; w *= 2) {
        for (int base = 0; base < HB; base += 2 * w) {
            for (int i = 0; i < w; ++i) {
                const float a = v[base + i];
                const float b = v[base + w + i];
                v[base + i]     = a + b;
                v[base + w + i] = a - b;
            }
        }
    }
}

// shared output epilogue: identical code for every ISA tier
inline void epilogue(float * v, float corr, const ggml_fp16_t * svh, float * dst) {
    if (corr != 0.0f) {
        for (int i = 0; i < HB; ++i) v[i] = v[i] + corr;
    }
    had128(v);
    for (int i = 0; i < HB; ++i) {
        const float s = GGML_CPU_FP16_TO_FP32(svh[i]);
        dst[i] = (v[i] * HAD_SCALE) * s;
    }
}

// -------------------------------------------------------------------------------------------
// prepared activations
// -------------------------------------------------------------------------------------------

struct Prep {
    float   * xf;     // [k] transformed fp32
    int32_t * splat;  // [k] x8 replicated x4 (GEMV path only)
    float   * fb;     // [k/128] k_inv*scale_b
    float   * cb;     // [k/128] scale_b*sum(x8_b)  (corr = c0 * sum_b cb[b], summed by the unit)
};

// one 128-input block of one instance: independent of every other block, so prep spreads over
// (instance, block) pairs and the per-row correction is reduced later in a fixed order
void prep_block(const float * x, const ggml_fp16_t * suh, int64_t b, bool q8, const Prep & p) {
    float * v = p.xf + b * HB;
    const float * xb = x + b * HB;
    const ggml_fp16_t * sb = suh + b * HB;
    for (int i = 0; i < HB; ++i) v[i] = xb[i] * GGML_CPU_FP16_TO_FP32(sb[i]);
    had128(v);
    for (int i = 0; i < HB; ++i) v[i] = v[i] * HAD_SCALE;
    if (!q8) {
        return;
    }
    float amax = 0.0f;
    for (int i = 0; i < HB; ++i) amax = std::max(amax, std::fabs(v[i]));
    const float scale = amax / 127.0f;
    const float inv   = amax > 0.0f ? 127.0f / amax : 0.0f;
    int32_t sum = 0;
    int32_t * sp = p.splat + b * HB;
    for (int i = 0; i < HB; ++i) {
        int q = int(std::nearbyint(v[i] * inv));
        q = std::min(127, std::max(-127, q));
        sum += q;
        sp[i] = int32_t(uint32_t(uint8_t(int8_t(q))) * 0x01010101u);
    }
    p.fb[b] = K_INV * scale;
    p.cb[b] = scale * float(sum);
}

inline float corr_of(const Prep & p, int64_t kb_n) {
    float a = 0.0f;
    for (int64_t b = 0; b < kb_n; ++b) a = a + p.cb[b];
    return C0 * a;
}

inline int8_t splat_x8(int32_t s) { return int8_t(s & 0xff); }

// -------------------------------------------------------------------------------------------
// GEMV band kernels: rows m (<= MAX_M), tiles [b0, b0+B) of one 8-tile band, all k.
// `band` points at the band's first tile (band8 layout: tile (kt, b) at (kt*8 + b) * 16K).
// out[i][128] accumulates (caller zeroes).
// -------------------------------------------------------------------------------------------

struct RowsQ8 {
    const int32_t * splat[MAX_M];
    const float   * fb[MAX_M];
};

template <int bits>
void gemv_band_scalar(const uint16_t * band, int kb0, int kb1, int m, const RowsQ8 & rq, float (*out)[HB]) {
    constexpr auto perm = make_tc_perm();
    constexpr int words = 16 * bits;
    int32_t acc[MAX_M][HB];
    for (int kb = kb0; kb < kb1; ++kb) {
        std::memset(acc, 0, sizeof(acc));
        for (int kk = 0; kk < 8; ++kk) {
            const int kt = kb * 8 + kk;
            for (int b = 0; b < 8; ++b) {
                const uint16_t * tile = band + (size_t(kt) * 8 + b) * words;
                for (int t = 0; t < 256; ++t) {
                    const int bs  = bytesum_mul1(state_scalar(tile, bits, t));
                    const int pos = perm[t];
                    const int r = pos / 16, c = pos % 16;
                    for (int i = 0; i < m; ++i) {
                        acc[i][b * 16 + c] += bs * int32_t(splat_x8(rq.splat[i][kt * 16 + r]));
                    }
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            const float f = rq.fb[i][kb];
            for (int c = 0; c < HB; ++c) out[i][c] = std::fma(float(acc[i][c]), f, out[i][c]);
        }
    }
}

#if EXL3_X86

#if defined(__GNUC__)
#define EXL3_T_BW   __attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))
#define EXL3_T_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
#define EXL3_T_VBMI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,fma,f16c")))
#define EXL3_INLINE __attribute__((always_inline)) inline
#else
#define EXL3_T_BW
#define EXL3_T_VNNI
#define EXL3_T_VBMI
#define EXL3_INLINE __forceinline
#endif

constexpr __mmask16 ld_mask(int n) {
    return n >= 16 ? __mmask16(0xffff) : (n <= 0 ? __mmask16(0) : __mmask16((1u << n) - 1u));
}

template <int bits>
EXL3_T_BW EXL3_INLINE void load_tile(const uint16_t * packed, __m512i & p0, __m512i & p1, __m512i & p2, __m512i & p3) {
    constexpr int words32 = bits * 256 / 32;
    constexpr __mmask16 m0 = ld_mask(words32 - 0);
    constexpr __mmask16 m1 = ld_mask(words32 - 16);
    constexpr __mmask16 m2 = ld_mask(words32 - 32);
    constexpr __mmask16 m3 = ld_mask(words32 - 48);
    const uint32_t * pw = reinterpret_cast<const uint32_t *>(packed);
    p0 = _mm512_maskz_loadu_epi32(m0, pw);
    p1 = m1 ? _mm512_maskz_loadu_epi32(m1, pw + 16) : _mm512_setzero_si512();
    p2 = m2 ? _mm512_maskz_loadu_epi32(m2, pw + 32) : _mm512_setzero_si512();
    p3 = m3 ? _mm512_maskz_loadu_epi32(m3, pw + 48) : _mm512_setzero_si512();
}

// ---- dword extraction (VNNI tier; also K8 on VBMI) ----

template <int bits, int row>
EXL3_T_BW EXL3_INLINE void dword_gather(__m512i p0, __m512i p1, __m512i p2, __m512i p3, __m512i & a, __m512i & b) {
    alignas(64) static constexpr auto i0d = make_row_indices<bits, row, false>();
    alignas(64) static constexpr auto i1d = make_row_indices<bits, row, true>();
    const __m512i i0 = _mm512_load_si512(i0d.data());
    const __m512i i1 = _mm512_load_si512(i1d.data());
    a = _mm512_permutex2var_epi32(p0, i0, p1);
    b = _mm512_permutex2var_epi32(p0, i1, p1);
    if constexpr (bits > 4) {
        constexpr __mmask16 hm0 = make_row_himask<bits, row, false>();
        constexpr __mmask16 hm1 = make_row_himask<bits, row, true>();
        if constexpr (hm0 != 0) a = _mm512_mask_blend_epi32(hm0, a, _mm512_permutex2var_epi32(p2, i0, p3));
        if constexpr (hm1 != 0) b = _mm512_mask_blend_epi32(hm1, b, _mm512_permutex2var_epi32(p2, i1, p3));
    } else {
        (void) p2; (void) p3;
    }
}

template <int bits, int row>
EXL3_T_BW EXL3_INLINE __m512i dword_codes(__m512i a, __m512i b) {
    constexpr int s0 = row_shift<bits, row>(0);
    constexpr int s1 = row_shift<bits, row>(8);
    if constexpr (s0 == s1) {
        const __m512i c = _mm512_or_si512(_mm512_srli_epi32(b, s0), _mm512_slli_epi32(a, 32 - s0));
        return _mm512_and_si512(c, _mm512_set1_epi32(0xffff));
    } else {
        alignas(64) static constexpr auto sh  = make_lane_shifts<s0, s1>();
        alignas(64) static constexpr auto shc = make_lane_shifts<32 - s0, 32 - s1>();
        const __m512i c = _mm512_or_si512(_mm512_srlv_epi32(b, _mm512_load_si512(sh.data())),
                                          _mm512_sllv_epi32(a, _mm512_load_si512(shc.data())));
        return _mm512_and_si512(c, _mm512_set1_epi32(0xffff));
    }
}

template <int bits, int row>
EXL3_T_BW EXL3_INLINE __m512i extract_row_dword(__m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    __m512i a, b;
    dword_gather<bits, row>(p0, p1, p2, p3, a, b);
    return dword_codes<bits, row>(a, b);
}

// ---- byte extraction (VBMI tier) ----

template <int bits, int row>
EXL3_T_VBMI EXL3_INLINE __m512i gather_row_bytes(__m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    alignas(64) static constexpr auto bidx = make_row_byte_indices<bits, row>();
    const __m512i idx = _mm512_load_si512(bidx.data());
    if constexpr (bits <= 2) {
        (void) p1; (void) p2; (void) p3;
        return _mm512_permutexvar_epi8(idx, p0);
    } else if constexpr (bits <= 4) {
        (void) p2; (void) p3;
        return _mm512_permutex2var_epi8(p0, idx, p1);
    } else {
        constexpr uint64_t hm = make_row_byte_himask<bits, row>();
        if constexpr (hm == 0) {
            return _mm512_permutex2var_epi8(p0, idx, p1);
        } else if constexpr (hm == ~uint64_t(0)) {
            return _mm512_permutex2var_epi8(p2, idx, p3);
        } else {
            return _mm512_mask_blend_epi8(static_cast<__mmask64>(hm),
                _mm512_permutex2var_epi8(p0, idx, p1), _mm512_permutex2var_epi8(p2, idx, p3));
        }
    }
}

template <int bits, int row, int delta>
EXL3_T_VBMI EXL3_INLINE __m512i shift_mask_row(__m512i g) {
    constexpr int s0 = row_shift<bits, row>(0) % 8 - delta;
    constexpr int s1 = row_shift<bits, row>(8) % 8 - delta;
    static_assert(s0 >= 0 && s1 >= 0, "pairing delta exceeds sub-byte shift headroom");
    if constexpr (s0 == s1) {
        return _mm512_and_si512(_mm512_srli_epi32(g, s0), _mm512_set1_epi32(0xffff));
    } else {
        return _mm512_and_si512(_mm512_mask_blend_epi32(0xff00, _mm512_srli_epi32(g, s0), _mm512_srli_epi32(g, s1)),
                                _mm512_set1_epi32(0xffff));
    }
}

template <int bits, int row>
EXL3_T_VBMI EXL3_INLINE __m512i extract_row_byte(__m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    return shift_mask_row<bits, row, 0>(gather_row_bytes<bits, row>(p0, p1, p2, p3));
}

// ---- accumulate: rows pairs (VBMI, byte pairing) ----

template <int bits, int M, int B, int P>
EXL3_T_VBMI EXL3_INLINE void vbmi_rows(__m512i p0, __m512i p1, __m512i p2, __m512i p3, int b,
                                        const int32_t * const * sp, __m512i (&acc)[B][M]) {
    if constexpr (P < 8) {
        constexpr int R = P * 2;
        const __m512i mult = _mm512_set1_epi32(int32_t(MUL1_MULT));
        __m512i c0, c1;
        if constexpr (byte_pair_ok<bits, R>()) {
            const __m512i g = gather_row_bytes<bits, R>(p0, p1, p2, p3);
            c0 = shift_mask_row<bits, R, 0>(g);
            c1 = shift_mask_row<bits, R, bits>(g);
        } else {
            c0 = extract_row_byte<bits, R>(p0, p1, p2, p3);
            c1 = extract_row_byte<bits, R + 1>(p0, p1, p2, p3);
        }
        const __m512i prod0 = _mm512_mullo_epi32(c0, mult);
        const __m512i prod1 = _mm512_mullo_epi32(c1, mult);
        for (int i = 0; i < M; ++i) {
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod0, _mm512_set1_epi32(sp[i][R]));
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod1, _mm512_set1_epi32(sp[i][R + 1]));
        }
        vbmi_rows<bits, M, B, P + 1>(p0, p1, p2, p3, b, sp, acc);
    }
}

template <int bits, int M, int B, int R>
EXL3_T_VNNI EXL3_INLINE void vnni_rows(__m512i p0, __m512i p1, __m512i p2, __m512i p3, int b,
                                        const int32_t * const * sp, __m512i (&acc)[B][M]) {
    if constexpr (R < 16) {
        const __m512i code = extract_row_dword<bits, R>(p0, p1, p2, p3);
        const __m512i prod = _mm512_mullo_epi32(code, _mm512_set1_epi32(int32_t(MUL1_MULT)));
        for (int i = 0; i < M; ++i) {
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod, _mm512_set1_epi32(sp[i][R]));
        }
        vnni_rows<bits, M, B, R + 1>(p0, p1, p2, p3, b, sp, acc);
    }
}

// one pass over tiles [b0, b0+B) of the band for all k
template <int bits, int M, int B, bool VBMI>
EXL3_T_VBMI void gemv_pass(const uint16_t * band, int kb0, int kb1, int b0, const RowsQ8 & rq, float (*out)[HB]) {
    constexpr int words = 16 * bits;
    for (int kb = kb0; kb < kb1; ++kb) {
        __m512i acc[B][M];
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < M; ++i) acc[b][i] = _mm512_setzero_si512();
        }
        for (int kk = 0; kk < 8; ++kk) {
            const int kt = kb * 8 + kk;
            const int32_t * sp[M];
            for (int i = 0; i < M; ++i) sp[i] = rq.splat[i] + kt * 16;
            const uint16_t * row = band + size_t(kt) * 8 * words;
            _mm_prefetch(reinterpret_cast<const char *>(row + 16 * words), _MM_HINT_T0);
            for (int b = 0; b < B; ++b) {
                __m512i p0, p1, p2, p3;
                load_tile<bits>(row + (b0 + b) * words, p0, p1, p2, p3);
                if constexpr (VBMI && bits < 8) {
                    vbmi_rows<bits, M, B, 0>(p0, p1, p2, p3, b, sp, acc);
                } else {
                    vnni_rows<bits, M, B, 0>(p0, p1, p2, p3, b, sp, acc);
                }
            }
        }
        for (int i = 0; i < M; ++i) {
            const __m512 f = _mm512_set1_ps(rq.fb[i][kb]);
            for (int b = 0; b < B; ++b) {
                float * o = out[i] + (b0 + b) * 16;
                _mm512_storeu_ps(o, _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), f, _mm512_loadu_ps(o)));
            }
        }
    }
}

template <int bits, int M, bool VBMI>
void gemv_band_simd(const uint16_t * band, int kb0, int kb1, const RowsQ8 & rq, float (*out)[HB]) {
    if constexpr (M <= 2) {
        gemv_pass<bits, M, 8, VBMI>(band, kb0, kb1, 0, rq, out);
    } else {
        gemv_pass<bits, M, 4, VBMI>(band, kb0, kb1, 0, rq, out);
        gemv_pass<bits, M, 4, VBMI>(band, kb0, kb1, 4, rq, out);
    }
}

// ---- band decode to fp32 (prefill), canonical codebook ----

template <int bits, int R>
EXL3_T_VBMI EXL3_INLINE void decode_rows(__m512i p0, __m512i p1, __m512i p2, __m512i p3, bool vbmi, float * dst /* row stride HB */) {
    if constexpr (R < 16) {
        __m512i code;
        if constexpr (bits < 8) {
            code = vbmi ? extract_row_byte<bits, R>(p0, p1, p2, p3) : extract_row_dword<bits, R>(p0, p1, p2, p3);
        } else {
            code = extract_row_dword<bits, R>(p0, p1, p2, p3);
        }
        const __m512i prod = _mm512_mullo_epi32(code, _mm512_set1_epi32(int32_t(MUL1_MULT)));
        const __m512i bs   = _mm512_dpbusd_epi32(_mm512_set1_epi32(1024), prod, _mm512_set1_epi32(0x01010101));
        const __m512  v    = _mm512_fmadd_ps(_mm512_cvtepi32_ps(bs), _mm512_set1_ps(K_INV), _mm512_set1_ps(C_HALF));
        const __m512  h    = _mm512_cvtph_ps(_mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm512_storeu_ps(dst + R * HB, h);
        decode_rows<bits, R + 1>(p0, p1, p2, p3, vbmi, dst);
    }
}

// decode k-tiles [kt0, kt0+nkt) of a band into w[(kt-kt0)*16 + r][128]
template <int bits>
EXL3_T_VBMI void decode_band_simd(const uint16_t * band, int kt0, int nkt, bool vbmi, float * w) {
    constexpr int words = 16 * bits;
    for (int kt = kt0; kt < kt0 + nkt; ++kt) {
        for (int b = 0; b < 8; ++b) {
            __m512i p0, p1, p2, p3;
            load_tile<bits>(band + (size_t(kt) * 8 + b) * words, p0, p1, p2, p3);
            decode_rows<bits, 0>(p0, p1, p2, p3, vbmi, w + size_t(kt - kt0) * 16 * HB + b * 16);
        }
    }
}

// acc[m][128] += x[m][0..kc) . w[0..kc)[128]   (fma chain in ascending k per element)
template <int MR>
EXL3_T_BW void gemm_block(const float * x, int64_t xs, int kc, const float * w, float * acc) {
    for (int c = 0; c < HB; c += 64) {
        __m512 a[MR][4];
        for (int r = 0; r < MR; ++r) {
            for (int j = 0; j < 4; ++j) a[r][j] = _mm512_loadu_ps(acc + r * HB + c + j * 16);
        }
        for (int k = 0; k < kc; ++k) {
            const float * wr = w + size_t(k) * HB + c;
            const __m512 w0 = _mm512_loadu_ps(wr);
            const __m512 w1 = _mm512_loadu_ps(wr + 16);
            const __m512 w2 = _mm512_loadu_ps(wr + 32);
            const __m512 w3 = _mm512_loadu_ps(wr + 48);
            for (int r = 0; r < MR; ++r) {
                const __m512 xb = _mm512_set1_ps(x[r * xs + k]);
                a[r][0] = _mm512_fmadd_ps(xb, w0, a[r][0]);
                a[r][1] = _mm512_fmadd_ps(xb, w1, a[r][1]);
                a[r][2] = _mm512_fmadd_ps(xb, w2, a[r][2]);
                a[r][3] = _mm512_fmadd_ps(xb, w3, a[r][3]);
            }
        }
        for (int r = 0; r < MR; ++r) {
            for (int j = 0; j < 4; ++j) _mm512_storeu_ps(acc + r * HB + c + j * 16, a[r][j]);
        }
    }
}

#endif // EXL3_X86

template <int bits>
void decode_band_scalar(const uint16_t * band, int kt0, int nkt, float * w) {
    constexpr auto perm = make_tc_perm();
    constexpr int words = 16 * bits;
    for (int kt = kt0; kt < kt0 + nkt; ++kt) {
        for (int b = 0; b < 8; ++b) {
            const uint16_t * tile = band + (size_t(kt) * 8 + b) * words;
            for (int t = 0; t < 256; ++t) {
                const int pos = perm[t];
                const int r = pos / 16, c = pos % 16;
                w[(size_t(kt - kt0) * 16 + r) * HB + b * 16 + c] = mul1_canonical(bytesum_mul1(state_scalar(tile, bits, t)));
            }
        }
    }
}

void gemm_block_scalar(const float * x, int64_t xs, int m, int kc, const float * w, float * acc) {
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < HB; ++c) {
            float a = acc[r * HB + c];
            for (int k = 0; k < kc; ++k) a = std::fma(x[r * xs + k], w[size_t(k) * HB + c], a);
            acc[r * HB + c] = a;
        }
    }
}

// -------------------------------------------------------------------------------------------
// dispatch
// -------------------------------------------------------------------------------------------

Isa detect_isa() {
#if EXL3_X86 && defined(__GNUC__)
    __builtin_cpu_init();
    const bool bw = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                    __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    Isa hw = Isa::scalar;
    if (bw && __builtin_cpu_supports("avx512vnni")) {
        hw = __builtin_cpu_supports("avx512vbmi") ? Isa::vbmi : Isa::vnni;
    }
#else
    Isa hw = Isa::scalar;
#endif
    if (const char * e = std::getenv("GGML_EXL3_ISA")) {
        Isa cap = hw;
        if (!std::strcmp(e, "scalar")) cap = Isa::scalar;
        else if (!std::strcmp(e, "vnni")) cap = Isa::vnni;
        else if (!std::strcmp(e, "vbmi")) cap = Isa::vbmi;
        if (int(cap) < int(hw)) hw = cap;
    }
    return hw;
}

Isa g_hw_isa = detect_isa();
std::atomic<int> g_isa { int(g_hw_isa) };

using GemvFn = void (*)(const uint16_t *, int, int, const RowsQ8 &, float (*)[HB]);

template <int bits>
GemvFn gemv_for(Isa isa, int m) {
#if EXL3_X86
    if (isa == Isa::vbmi) {
        switch (m) {
            case 1: return gemv_band_simd<bits, 1, true>;
            case 2: return gemv_band_simd<bits, 2, true>;
            case 3: return gemv_band_simd<bits, 3, true>;
            default: return gemv_band_simd<bits, 4, true>;
        }
    }
    if (isa == Isa::vnni) {
        switch (m) {
            case 1: return gemv_band_simd<bits, 1, false>;
            case 2: return gemv_band_simd<bits, 2, false>;
            case 3: return gemv_band_simd<bits, 3, false>;
            default: return gemv_band_simd<bits, 4, false>;
        }
    }
#endif
    (void) m;
    return nullptr;
}

GemvFn gemv_dispatch(int bits, Isa isa, int m) {
    switch (bits) {
        case 1: return gemv_for<1>(isa, m);
        case 2: return gemv_for<2>(isa, m);
        case 3: return gemv_for<3>(isa, m);
        case 4: return gemv_for<4>(isa, m);
        case 5: return gemv_for<5>(isa, m);
        case 6: return gemv_for<6>(isa, m);
        case 7: return gemv_for<7>(isa, m);
        default: return gemv_for<8>(isa, m);
    }
}

void gemv_band(int bits, Isa isa, const uint16_t * band, int kb0, int kb1, int m, const RowsQ8 & rq, float (*out)[HB]) {
    if (isa != Isa::scalar) {
        gemv_dispatch(bits, isa, m)(band, kb0, kb1, rq, out);
        return;
    }
    switch (bits) {
        case 1: gemv_band_scalar<1>(band, kb0, kb1, m, rq, out); break;
        case 2: gemv_band_scalar<2>(band, kb0, kb1, m, rq, out); break;
        case 3: gemv_band_scalar<3>(band, kb0, kb1, m, rq, out); break;
        case 4: gemv_band_scalar<4>(band, kb0, kb1, m, rq, out); break;
        case 5: gemv_band_scalar<5>(band, kb0, kb1, m, rq, out); break;
        case 6: gemv_band_scalar<6>(band, kb0, kb1, m, rq, out); break;
        case 7: gemv_band_scalar<7>(band, kb0, kb1, m, rq, out); break;
        default: gemv_band_scalar<8>(band, kb0, kb1, m, rq, out); break;
    }
}

void decode_band(int bits, Isa isa, const uint16_t * band, int kt0, int nkt, float * w) {
#if EXL3_X86
    if (isa != Isa::scalar) {
        const bool vbmi = isa == Isa::vbmi;
        switch (bits) {
            case 1: decode_band_simd<1>(band, kt0, nkt, vbmi, w); return;
            case 2: decode_band_simd<2>(band, kt0, nkt, vbmi, w); return;
            case 3: decode_band_simd<3>(band, kt0, nkt, vbmi, w); return;
            case 4: decode_band_simd<4>(band, kt0, nkt, vbmi, w); return;
            case 5: decode_band_simd<5>(band, kt0, nkt, vbmi, w); return;
            case 6: decode_band_simd<6>(band, kt0, nkt, vbmi, w); return;
            case 7: decode_band_simd<7>(band, kt0, nkt, vbmi, w); return;
            default: decode_band_simd<8>(band, kt0, nkt, vbmi, w); return;
        }
    }
#endif
    switch (bits) {
        case 1: decode_band_scalar<1>(band, kt0, nkt, w); break;
        case 2: decode_band_scalar<2>(band, kt0, nkt, w); break;
        case 3: decode_band_scalar<3>(band, kt0, nkt, w); break;
        case 4: decode_band_scalar<4>(band, kt0, nkt, w); break;
        case 5: decode_band_scalar<5>(band, kt0, nkt, w); break;
        case 6: decode_band_scalar<6>(band, kt0, nkt, w); break;
        case 7: decode_band_scalar<7>(band, kt0, nkt, w); break;
        default: decode_band_scalar<8>(band, kt0, nkt, w); break;
    }
}

void gemm_rows(Isa isa, const float * x, int64_t xs, int m, int kc, const float * w, float * acc) {
#if EXL3_X86
    if (isa != Isa::scalar) {
        int r = 0;
        for (; r + 4 <= m; r += 4) gemm_block<4>(x + r * xs, xs, kc, w, acc + r * HB);
        switch (m - r) {
            case 1: gemm_block<1>(x + r * xs, xs, kc, w, acc + r * HB); break;
            case 2: gemm_block<2>(x + r * xs, xs, kc, w, acc + r * HB); break;
            case 3: gemm_block<3>(x + r * xs, xs, kc, w, acc + r * HB); break;
            default: break;
        }
        return;
    }
#endif
    gemm_block_scalar(x, xs, m, kc, w, acc);
}

// -------------------------------------------------------------------------------------------
// op geometry
// -------------------------------------------------------------------------------------------

struct Geometry {
    bool grouped;
    const ggml_tensor * w, * x, * ids, * svh, * suh;
    int64_t k, n, n_exp;
    int bits;
    int64_t n_inst;   // dense: rows; grouped: ids->ne[0]*ids->ne[1]
    int64_t n_rows_max; // largest possible per-expert instance count
};

Geometry geometry(const ggml_tensor * dst) {
    Geometry g{};
    g.grouped = dst->op == GGML_OP_MUL_MAT_ID;
    g.w   = dst->src[0];
    g.x   = dst->src[1];
    g.ids = g.grouped ? dst->src[2] : nullptr;
    g.svh = dst->src[g.grouped ? 3 : 2];
    g.suh = dst->src[g.grouped ? 4 : 3];
    g.k = g.w->ne[0];
    g.n = g.w->ne[1];
    g.n_exp = g.w->ne[2];
    g.bits = ggml_exl3_bits(g.w->type);
    if (g.grouped) {
        g.n_inst = g.ids->ne[0] * g.ids->ne[1];
        g.n_rows_max = g.ids->ne[1];
    } else {
        g.n_inst = g.x->ne[1] * g.x->ne[2] * g.x->ne[3];
        g.n_rows_max = g.n_inst;
    }
    return g;
}

bool aux_ok(const ggml_tensor * t, int64_t width, int64_t experts) {
    return t && t->type == GGML_TYPE_F16 && t->ne[0] == width && t->ne[1] == experts &&
           t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t);
}

struct Layout {
    size_t off_xf, off_splat, off_fb, off_cb, off_part, off_thread, per_thread, total;
};

// upper bound on S for this op (the plan may choose less once routing is known)
int splitk_cap(const Geometry & g) {
    if (g.n_inst > SPLITK_MAX_INST) return 1;
    const int64_t bands = g.n / HB;
    const int64_t kb_n  = g.k / HB;
    if (bands >= g_splitk_target || kb_n < 2) return 1;
    const int64_t s = (g_splitk_target + bands - 1) / bands;
    return int(std::max<int64_t>(1, std::min<int64_t>(s, kb_n / 2)));
}

Layout layout(const Geometry & g, int nth) {
    auto pad = [](size_t v) { return (v + 63) & ~size_t(63); };
    Layout l{};
    size_t o = 0;
    l.off_xf    = o; o = pad(o + size_t(g.n_inst) * g.k * sizeof(float));
    l.off_splat = o; o = pad(o + size_t(g.n_inst) * g.k * sizeof(int32_t));
    l.off_fb    = o; o = pad(o + size_t(g.n_inst) * (g.k / HB) * sizeof(float));
    l.off_cb    = o; o = pad(o + size_t(g.n_inst) * (g.k / HB) * sizeof(float));
    const int cap = splitk_cap(g);
    l.off_part  = o; o = pad(o + (cap > 1 ? size_t(g.n_inst) * g.n * cap * sizeof(float) : 0));
    l.off_thread = o;
    // per thread: GEMM weight chunk [KC][128] + accumulators [rows_max][128] (+ GEMV out [MAX_M][128])
    const size_t rows = std::max<int64_t>(g.n_rows_max, MAX_M);
    l.per_thread = pad((size_t(KC) * HB + rows * HB) * sizeof(float));
    o += l.per_thread * nth;
    l.total = o + 64;
    return l;
}

struct Inst {
    int32_t expert;
    const float * x;
    float * y;
};

struct Group {
    int32_t expert;
    int32_t first;
    int32_t count;
};

struct Unit {
    int32_t group;
    int32_t band;
    int32_t chunk;   // GEMV: rows [chunk*MAX_M, ...); GEMM: -1
    int32_t ks;      // GEMV split-K slice
};

// returns S, the GEMV split-K factor
int build_plan(const Geometry & g, const ggml_tensor * dst, std::vector<Inst> & inst,
                std::vector<Group> & groups, std::vector<Unit> & units, int gemv_max) {
    inst.clear();
    groups.clear();
    units.clear();
    const ggml_tensor * x = g.x;
    if (!g.grouped) {
        inst.reserve(g.n_inst);
        for (int64_t i3 = 0; i3 < x->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < x->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
                    const float * xp = (const float *) ((const char *) x->data + i1 * x->nb[1] + i2 * x->nb[2] + i3 * x->nb[3]);
                    float * yp = (float *) ((char *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
                    inst.push_back({ 0, xp, yp });
                }
            }
        }
        groups.push_back({ 0, 0, int32_t(inst.size()) });
    } else {
        const ggml_tensor * ids = g.ids;
        const int64_t n_used = ids->ne[0], n_tok = ids->ne[1];
        // counting sort by expert (deterministic order: expert asc, then token, slot)
        std::vector<int32_t> cnt(g.n_exp + 1, 0);
        auto expert_at = [&](int64_t t, int64_t s) {
            int32_t e;
            std::memcpy(&e, (const char *) ids->data + t * ids->nb[1] + s * ids->nb[0], sizeof(e));
            return e;
        };
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t s = 0; s < n_used; ++s) {
                const int32_t e = expert_at(t, s);
                GGML_ASSERT(e >= -1 && e < g.n_exp);
                cnt[e + 1]++;
            }
        }
        std::vector<int32_t> start(g.n_exp + 1, 0);
        for (int64_t e = 1; e <= g.n_exp; ++e) start[e] = start[e - 1] + cnt[e];
        // instances with e == -1 are placed at the end (zero output)
        const int32_t n_valid = start[g.n_exp];
        inst.resize(size_t(n_valid) + cnt[0]);
        std::vector<int32_t> fill(start.begin(), start.end());
        int32_t neg = n_valid;
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t s = 0; s < n_used; ++s) {
                const int32_t e = expert_at(t, s);
                const float * xp = (const float *) ((const char *) x->data + (s % x->ne[1]) * x->nb[1] + t * x->nb[2]);
                float * yp = (float *) ((char *) dst->data + s * dst->nb[1] + t * dst->nb[2]);
                if (e < 0) {
                    inst[neg++] = { -1, xp, yp };
                } else {
                    inst[fill[e]++] = { e, xp, yp };
                }
            }
        }
        for (int64_t e = 0; e < g.n_exp; ++e) {
            if (cnt[e + 1] > 0) groups.push_back({ int32_t(e), start[e], cnt[e + 1] });
        }
    }
    const int32_t bands = int32_t(g.n / HB);
    int64_t gemv_units = 0;
    for (const Group & gr : groups) {
        if (gr.count <= gemv_max) gemv_units += int64_t(bands) * ((gr.count + MAX_M - 1) / MAX_M);
    }
    int S = 1;
    const int cap = splitk_cap(g);
    if (cap > 1 && gemv_units > 0 && gemv_units < g_splitk_target) {
        S = int(std::min<int64_t>(cap, (g_splitk_target + gemv_units - 1) / gemv_units));
    }
    for (int32_t gi = 0; gi < int32_t(groups.size()); ++gi) {
        const Group & gr = groups[gi];
        if (gr.count <= gemv_max) {
            const int32_t chunks = (gr.count + MAX_M - 1) / MAX_M;
            for (int32_t b = 0; b < bands; ++b) {
                for (int32_t c = 0; c < chunks; ++c) {
                    for (int32_t ks = 0; ks < S; ++ks) units.push_back({ gi, b, c, ks });
                }
            }
        } else {
            for (int32_t b = 0; b < bands; ++b) units.push_back({ gi, b, -1, 0 });
        }
    }
    return S;
}

} // namespace

// -------------------------------------------------------------------------------------------
// public entry points
// -------------------------------------------------------------------------------------------

bool ggml_cpu_exl3_supports(const struct ggml_tensor * dst) {
    if (dst->op != GGML_OP_MUL_MAT && dst->op != GGML_OP_MUL_MAT_ID) return false;
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    if (!w || !x || !ggml_type_is_exl3(w->type)) return false;
    if (w->ne[0] % HB || w->ne[1] % HB || !ggml_is_contiguous(w) || w->ne[3] != 1) return false;
    if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || x->ne[0] != w->ne[0]) return false;
    if (dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float)) return false;
    const bool grouped = dst->op == GGML_OP_MUL_MAT_ID;
    const ggml_tensor * svh = dst->src[grouped ? 3 : 2];
    const ggml_tensor * suh = dst->src[grouped ? 4 : 3];
    if (grouped) {
        const ggml_tensor * ids = dst->src[2];
        if (!ids || ids->type != GGML_TYPE_I32 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            x->ne[2] != ids->ne[1] || x->ne[3] != 1) return false;
    } else if (w->ne[2] != 1) {
        return false;
    }
    // loader placement probes are built before the aux sources are attached
    if (!svh && !suh) return true;
    return aux_ok(svh, w->ne[1], w->ne[2]) && aux_ok(suh, w->ne[0], w->ne[2]);
}

size_t ggml_cpu_exl3_work_size(const struct ggml_tensor * dst, int n_threads) {
    if (!ggml_cpu_exl3_supports(dst)) return 0;
    return layout(geometry(dst), std::max(1, n_threads)).total;
}

const char * ggml_cpu_exl3_isa(void) {
    switch (Isa(g_isa.load())) {
        case Isa::vbmi: return "vbmi";
        case Isa::vnni: return "vnni";
        default:        return "scalar";
    }
}

void ggml_cpu_exl3_set_isa(const char * name) {
    Isa want = g_hw_isa;
    if (name && !std::strcmp(name, "scalar")) want = Isa::scalar;
    else if (name && !std::strcmp(name, "vnni")) want = Isa::vnni;
    else if (name && !std::strcmp(name, "vbmi")) want = Isa::vbmi;
    if (int(want) > int(g_hw_isa)) want = g_hw_isa;
    g_isa.store(int(want));
}

void ggml_cpu_exl3_set_gemv_max_rows(int rows) {
    g_gemv_max_rows = std::max(0, rows);
}

void ggml_cpu_exl3_set_splitk_target(int target) {
    g_splitk_target = std::max(1, target);
}

void ggml_cpu_exl3_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_cpu_exl3_supports(dst));
    const Geometry g = geometry(dst);
    GGML_ASSERT(g.svh && g.suh && "EXL3 matmul without svh/suh: weight not wired through the EXL3 aux map");
    const int ith = params->ith, nth = params->nth;
    const Layout L = layout(g, nth);
    GGML_ASSERT(params->wsize >= L.total);
    const Isa isa = Isa(g_isa.load());
    const int gemv_max = g_gemv_max_rows;

    char * base = (char *) params->wdata;
    base = (char *) (((uintptr_t) base + 63) & ~uintptr_t(63));
    float   * XF    = (float *)   (base + L.off_xf);
    int32_t * SPLAT = (int32_t *) (base + L.off_splat);
    float   * FB    = (float *)   (base + L.off_fb);
    float   * CB    = (float *)   (base + L.off_cb);
    float   * TH    = (float *)   (base + L.off_thread + L.per_thread * ith);

    static thread_local std::vector<Inst>  inst;
    static thread_local std::vector<Group> groups;
    static thread_local std::vector<Unit>  units;
    const int S = build_plan(g, dst, inst, groups, units, gemv_max);
    float * PART = (float *) (base + L.off_part);   // [S][n_inst][n] raw GEMV partials (S > 1)

    const int64_t k = g.k;
    const int64_t kb_n = k / HB;
    auto prep_of = [&](int64_t i) {
        return Prep { XF + i * k, SPLAT + i * k, FB + i * kb_n, CB + i * kb_n };
    };

    // ---- phase 1: prepare every (instance, 128-block) ----
    {
        const int64_t n_valid = groups.empty() ? 0 : int64_t(groups.back().first) + groups.back().count;
        int64_t item = 0;
        for (const Group & gr : groups) {
            const bool q8 = gr.count <= gemv_max;
            const ggml_fp16_t * suh = (const ggml_fp16_t *) ((const char *) g.suh->data + gr.expert * g.suh->nb[1]);
            for (int32_t j = 0; j < gr.count; ++j) {
                const int64_t i = int64_t(gr.first) + j;
                const Prep p = prep_of(i);
                for (int64_t b = 0; b < kb_n; ++b, ++item) {
                    if (item % nth == ith) prep_block(inst[i].x, suh, b, q8, p);
                }
            }
        }
        // routing entries with expert -1 (if any): zero outputs
        for (int64_t i = n_valid; i < int64_t(inst.size()); ++i) {
            if (i % nth == ith) std::memset(inst[i].y, 0, size_t(g.n) * sizeof(float));
        }
    }
    if (nth > 1) {
        if (ith == 0) ggml_threadpool_chunk_set(params->threadpool, nth);
        ggml_barrier(params->threadpool);
    }

    // ---- phase 2: units ----
    const int tiles_k = int(k / 16);
    const size_t band_words = size_t(tiles_k) * 8 * 16 * g.bits;
    const size_t exp_words  = size_t(g.n / 16) * tiles_k * 16 * g.bits;
    const uint16_t * W = (const uint16_t *) g.w->data;
    float * wbuf = TH;                  // [KC][128]
    float * abuf = TH + size_t(KC) * HB; // [rows][128]

    int64_t u = ith;
    const int64_t n_units = int64_t(units.size());
    while (u < n_units) {
        const Unit & un = units[u];
        const Group & gr = groups[un.group];
        const uint16_t * band = W + size_t(gr.expert) * exp_words + size_t(un.band) * band_words;
        const ggml_fp16_t * svh = (const ggml_fp16_t *) ((const char *) g.svh->data + gr.expert * g.svh->nb[1]) + un.band * HB;
        if (un.chunk >= 0) {
            const int32_t r0 = gr.first + un.chunk * MAX_M;
            const int m = std::min<int>(MAX_M, gr.first + gr.count - r0);
            RowsQ8 rq{};
            for (int i = 0; i < m; ++i) {
                const Prep p = prep_of(r0 + i);
                rq.splat[i] = p.splat;
                rq.fb[i] = p.fb;
            }
            float (*out)[HB] = reinterpret_cast<float (*)[HB]>(abuf);
            std::memset(abuf, 0, sizeof(float) * HB * m);
            const int kb0 = int(int64_t(un.ks) * kb_n / S);
            const int kb1 = int(int64_t(un.ks + 1) * kb_n / S);
            gemv_band(g.bits, isa, band, kb0, kb1, m, rq, out);
            if (S == 1) {
                for (int i = 0; i < m; ++i) {
                    epilogue(out[i], corr_of(prep_of(r0 + i), kb_n), svh, inst[r0 + i].y + un.band * HB);
                }
            } else {
                for (int i = 0; i < m; ++i) {
                    std::memcpy(PART + (size_t(un.ks) * g.n_inst + r0 + i) * g.n + un.band * HB, out[i], sizeof(float) * HB);
                }
            }
        } else {
            const int m = gr.count;
            std::memset(abuf, 0, sizeof(float) * HB * m);
            for (int kt0 = 0; kt0 < tiles_k; kt0 += KC / 16) {
                const int nkt = std::min(KC / 16, tiles_k - kt0);
                decode_band(g.bits, isa, band, kt0, nkt, wbuf);
                gemm_rows(isa, XF + int64_t(gr.first) * k + kt0 * 16, k, m, nkt * 16, wbuf, abuf);
            }
            for (int i = 0; i < m; ++i) {
                epilogue(abuf + i * HB, 0.0f, svh, inst[gr.first + i].y + un.band * HB);
            }
        }
        u = nth > 1 ? ggml_threadpool_chunk_add(params->threadpool, 1) : u + 1;
    }

    // ---- phase 3 (split-K only): fixed-order reduction + output epilogue per (instance, band) ----
    if (S > 1) {
        if (nth > 1) ggml_barrier(params->threadpool);
        const int64_t bands = g.n / HB;
        int64_t task = 0;
        float acc[HB];
        for (const Group & gr : groups) {
            if (gr.count > gemv_max) continue;
            const ggml_fp16_t * svh0 = (const ggml_fp16_t *) ((const char *) g.svh->data + gr.expert * g.svh->nb[1]);
            for (int32_t j = 0; j < gr.count; ++j) {
                const int64_t i = int64_t(gr.first) + j;
                for (int64_t b = 0; b < bands; ++b, ++task) {
                    if (task % nth != ith) continue;
                    std::memcpy(acc, PART + size_t(i) * g.n + b * HB, sizeof(acc));
                    for (int ks = 1; ks < S; ++ks) {
                        const float * pp = PART + (size_t(ks) * g.n_inst + i) * g.n + b * HB;
                        for (int c = 0; c < HB; ++c) acc[c] = acc[c] + pp[c];
                    }
                    epilogue(acc, corr_of(prep_of(i), kb_n), svh0 + b * HB, inst[i].y + b * HB);
                }
            }
        }
    }
}
