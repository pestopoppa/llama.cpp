#pragma once

// Minimal MD5 implementation (RFC 1321, public domain)
// Only used for corpus shard routing: MD5(gram)[:4] % n_shards

#include <cstdint>
#include <cstring>

static inline uint32_t md5_f(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
static inline uint32_t md5_g(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
static inline uint32_t md5_h(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
static inline uint32_t md5_i(uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | ~z); }
static inline uint32_t md5_rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

// Compute MD5 and return the first 4 bytes as a little-endian uint32_t.
// This matches Python's: int.from_bytes(hashlib.md5(data).digest()[:4], "little")
static inline uint32_t md5_first4(const void * data, size_t len) {
    static const uint32_t T[64] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391,
    };
    static const int s[64] = {
        7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
        5, 9,14,20,5, 9,14,20,5, 9,14,20,5, 9,14,20,
        4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
        6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21,
    };

    // Pad message: append 0x80, pad to 56 mod 64, append 64-bit length
    size_t padded_len = ((len + 8) / 64 + 1) * 64;
    uint8_t buf[256]; // sufficient for inputs up to ~190 bytes (our grams are < 128)
    uint8_t * msg;
    bool heap = padded_len > sizeof(buf);
    if (heap) {
        msg = new uint8_t[padded_len];
    } else {
        msg = buf;
    }
    memcpy(msg, data, len);
    memset(msg + len, 0, padded_len - len);
    msg[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8;
    memcpy(msg + padded_len - 8, &bit_len, 8);

    uint32_t a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;

    for (size_t offset = 0; offset < padded_len; offset += 64) {
        uint32_t M[16];
        memcpy(M, msg + offset, 64);

        uint32_t a = a0, b = b0, c = c0, d = d0;
        for (int i = 0; i < 64; i++) {
            uint32_t f_val, g;
            if (i < 16) {
                f_val = md5_f(b, c, d);
                g = i;
            } else if (i < 32) {
                f_val = md5_g(b, c, d);
                g = (5*i + 1) % 16;
            } else if (i < 48) {
                f_val = md5_h(b, c, d);
                g = (3*i + 5) % 16;
            } else {
                f_val = md5_i(b, c, d);
                g = (7*i) % 16;
            }
            uint32_t tmp = d;
            d = c;
            c = b;
            b = b + md5_rotl(a + f_val + T[i] + M[g], s[i]);
            a = tmp;
        }
        a0 += a; b0 += b; c0 += c; d0 += d;
    }

    if (heap) {
        delete[] msg;
    }

    // Return first 4 bytes of digest (a0) as little-endian uint32_t
    // MD5 digest is: a0[le] || b0[le] || c0[le] || d0[le]
    // On little-endian (x86), a0 is already the first 4 bytes
    return a0;
}
