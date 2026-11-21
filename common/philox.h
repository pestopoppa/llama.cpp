#pragma once

// [GPU-RESIDENT-ROUND-2d] Philox-4x32-10 counter RNG (Salmon et al., SC'11) and Gumbel noise for the device
// DFlash2 lattice selection. Counter-based, so a value depends only on (key, counter): the same stream can be
// produced on the host now and by a device kernel later (ROUND-2e) without carrying generator state.

#include <cmath>
#include <cstdint>

struct common_philox4x32 {
    static void round(uint32_t c[4], const uint32_t k[2]) {
        const uint64_t p0 = (uint64_t) 0xD2511F53u * c[0];
        const uint64_t p1 = (uint64_t) 0xCD9E8D57u * c[2];
        const uint32_t hi0 = (uint32_t) (p0 >> 32), lo0 = (uint32_t) p0;
        const uint32_t hi1 = (uint32_t) (p1 >> 32), lo1 = (uint32_t) p1;
        const uint32_t n0 = hi1 ^ c[1] ^ k[0];
        const uint32_t n2 = hi0 ^ c[3] ^ k[1];
        c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
    }

    // 4 x 32 random bits for (key, counter)
    static void gen(uint64_t key, uint64_t ctr_hi, uint64_t ctr_lo, uint32_t out[4]) {
        uint32_t c[4] = { (uint32_t) ctr_lo, (uint32_t) (ctr_lo >> 32), (uint32_t) ctr_hi, (uint32_t) (ctr_hi >> 32) };
        uint32_t k[2] = { (uint32_t) key, (uint32_t) (key >> 32) };
        for (int r = 0; r < 10; ++r) {
            round(c, k);
            if (r < 9) {
                k[0] += 0x9E3779B9u;
                k[1] += 0xBB67AE85u;
            }
        }
        for (int i = 0; i < 4; ++i) {
            out[i] = c[i];
        }
    }

    // uniform in (0, 1), 24-bit resolution, never 0 or 1
    static float u01(uint32_t bits) {
        return ((float) (bits >> 8) + 0.5f) * (1.0f / 16777216.0f);
    }

    // standard Gumbel noise: argmax(x_k + g_k) samples softmax(x) exactly
    static float gumbel(uint32_t bits) {
        return -std::log(-std::log(u01(bits)));
    }
};
