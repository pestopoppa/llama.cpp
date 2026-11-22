#pragma once

// [GPU-RESIDENT-ROUND-2f] Philox-4x32-10 + Gumbel for GGML_OP_PHILOX_GUMBEL, bit-identical to common/philox.h
// (the host stream of GPU-RESIDENT-ROUND-2d). Shared by the CPU and CUDA/HIP backends.

#include <stdint.h>
#include <math.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_PHILOX_HD __host__ __device__ __forceinline__
#else
#define GGML_PHILOX_HD static inline
#endif

GGML_PHILOX_HD void ggml_philox4x32_10(uint32_t key_lo, uint32_t key_hi, uint64_t ctr_hi, uint64_t ctr_lo, uint32_t out[4]) {
    uint32_t c0 = (uint32_t) ctr_lo, c1 = (uint32_t) (ctr_lo >> 32), c2 = (uint32_t) ctr_hi, c3 = (uint32_t) (ctr_hi >> 32);
    uint32_t k0 = key_lo, k1 = key_hi;
    for (int r = 0; r < 10; ++r) {
        const uint64_t p0 = (uint64_t) 0xD2511F53u * c0;
        const uint64_t p1 = (uint64_t) 0xCD9E8D57u * c2;
        const uint32_t hi0 = (uint32_t) (p0 >> 32), lo0 = (uint32_t) p0;
        const uint32_t hi1 = (uint32_t) (p1 >> 32), lo1 = (uint32_t) p1;
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
        if (r < 9) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

// common_philox4x32::gumbel: -log(-log(u)), u = ((bits >> 8) + 0.5) / 2^24
GGML_PHILOX_HD float ggml_philox_gumbel_bits(uint32_t bits) {
    const float u = ((float) (bits >> 8) + 0.5f) * (1.0f / 16777216.0f);
    return -logf(-logf(u));
}

// value of flat element j of a block with key (lo, hi) and counter ctr (ctr < 0 -> 0)
GGML_PHILOX_HD float ggml_philox_gumbel_at(uint32_t key_lo, uint32_t key_hi, float ctr, int64_t j, int32_t skip) {
    if (ctr < 0.0f || j < skip) {
        return 0.0f;
    }
    const int64_t base = skip + ((j - skip) / 4) * 4;
    uint32_t r[4];
    ggml_philox4x32_10(key_lo, key_hi, (uint64_t) ctr, (uint64_t) base, r);
    return ggml_philox_gumbel_bits(r[(j - skip) % 4]);
}
