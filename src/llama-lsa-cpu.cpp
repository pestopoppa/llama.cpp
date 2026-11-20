#include "llama-lsa-cpu.h"

#include "llama-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace llama_lsa_cpu {

// ---------------------------------------------------------------------------------------------
// config

static int env_int(const char * name, int def) {
    const char * s = getenv(name);
    return s && *s ? atoi(s) : def;
}

static std::string env_str(const char * name) {
    const char * s = getenv(name);
    return s ? std::string(s) : std::string();
}

const config & get_config() {
    static const config cfg = [] {
        config c;
        c.enabled    = env_int("LLAMA_LSA_CPU", 0) != 0;
        c.map_path   = env_str("LLAMA_LSA_CPU_MAP");
        c.map_sha256 = env_str("LLAMA_LSA_CPU_MAP_SHA256");
        c.top_k      = env_int("LLAMA_LSA_CPU_TOPK",   4096);
        c.window     = env_int("LLAMA_LSA_CPU_WINDOW",  512);
        c.sink       = env_int("LLAMA_LSA_CPU_SINK",      4);
        c.min_kv     = env_int("LLAMA_LSA_CPU_MIN_KV", 2*c.width());
        if (c.enabled && (c.top_k < 1 || c.window < 0 || c.sink < 0 || c.min_kv < 0)) {
            LLAMA_LOG_ERROR("llama_lsa_cpu: invalid budgets (top_k %d window %d sink %d min_kv %d): disabled\n",
                    c.top_k, c.window, c.sink, c.min_kv);
            c.enabled = false;
        }
        if (c.enabled && c.map_path.empty()) {
            LLAMA_LOG_ERROR("llama_lsa_cpu: LLAMA_LSA_CPU=1 without LLAMA_LSA_CPU_MAP: disabled\n");
            c.enabled = false;
        }
        return c;
    }();
    return cfg;
}

// ---------------------------------------------------------------------------------------------
// sha256 (FIPS 180-4), for the payload / file binding

static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

std::string sha256_hex(const void * data, size_t size) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

    const uint8_t * in = (const uint8_t *) data;
    std::vector<uint8_t> msg(in, in + size);
    const uint64_t bits = (uint64_t) size*8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) {
        msg.push_back(0);
    }
    for (int i = 7; i >= 0; --i) {
        msg.push_back((uint8_t) (bits >> (8*i)));
    }

    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const uint8_t * p = msg.data() + off + 4*i;
            w[i] = (uint32_t) p[0] << 24 | (uint32_t) p[1] << 16 | (uint32_t) p[2] << 8 | p[3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i-15], 7) ^ ror(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = ror(w[i-2], 17) ^ ror(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], v = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = v + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            v = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += v;
    }

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 0; i < 8; ++i) {
        out << std::setw(8) << h[i];
    }
    return out.str();
}

// ---------------------------------------------------------------------------------------------
// LSAMAP1 loader (ported from experimental/lsa-h12-indexer-20261106):
//   "LSAMAP1\0", u32 version=1, q_heads, kv_heads, head_dim, rope_dim, index_dim=64, n_layers, model_len, ident_len,
//   u8[32] source sha256, u8[32] payload sha256 (of everything after the header), model, ident,
//   n_layers x { u32 layer, bf16 Q[rows*64], bf16 K[rows*64] }   rows = kv_heads*(head_dim - rope_dim), row-major

static float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static std::string hex(const uint8_t * p, size_t n) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < n; ++i) {
        out << std::setw(2) << (int) p[i];
    }
    return out.str();
}

bool load_lsamap(const std::string & path, lsamap & out, std::string & err) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        err = "cannot open map"; return false;
    }
    const std::streamoff size = in.tellg();
    if (size < 108 || size > 512LL*1024*1024) {
        err = "map size outside bounds"; return false;
    }
    std::vector<uint8_t> bytes((size_t) size);
    in.seekg(0);
    if (!in.read((char *) bytes.data(), size)) {
        err = "cannot read map"; return false;
    }

    size_t at = 0;
    auto rd = [&](void * dst, size_t n) {
        if (n > bytes.size() - at) {
            return false;
        }
        memcpy(dst, bytes.data() + at, n);
        at += n;
        return true;
    };
    auto u32 = [&](uint32_t & v) {
        uint8_t b[4];
        if (!rd(b, 4)) {
            return false;
        }
        v = (uint32_t) b[0] | (uint32_t) b[1] << 8 | (uint32_t) b[2] << 16 | (uint32_t) b[3] << 24;
        return true;
    };

    char magic[8];
    uint32_t version, q, kv, d, rope, idx, nl, mlen, ilen;
    uint8_t source[32], payload[32];
    if (!rd(magic, 8) || memcmp(magic, "LSAMAP1\0", 8) != 0 || !u32(version) || !u32(q) || !u32(kv) || !u32(d) || !u32(rope) ||
            !u32(idx) || !u32(nl) || !u32(mlen) || !u32(ilen) || !rd(source, 32) || !rd(payload, 32)) {
        err = "malformed map header"; return false;
    }
    if (version != 1 || idx != (uint32_t) index_dim || q == 0 || kv == 0 || q % kv != 0 || rope >= d || d > 1024 || nl == 0 || nl > 1024 ||
            mlen > 4096 || ilen > 4096) {
        err = "unsupported map geometry"; return false;
    }
    const std::string payload_hash = sha256_hex(bytes.data() + at, bytes.size() - at);
    if (payload_hash != hex(payload, 32)) {
        err = "map payload hash mismatch"; return false;
    }
    const uint64_t rows = (uint64_t) kv*(d - rope);
    if ((uint64_t) mlen + ilen + (uint64_t) nl*(4 + rows*index_dim*2*2) != bytes.size() - at) {
        err = "map payload size mismatch"; return false;
    }

    lsamap m;
    m.q_heads  = (int) q;
    m.kv_heads = (int) kv;
    m.head_dim = (int) d;
    m.rope_dim = (int) rope;
    m.source_sha256  = hex(source, 32);
    m.payload_sha256 = payload_hash;
    m.model_identity.assign((const char *) bytes.data() + at, mlen); at += mlen;
    m.map_identity  .assign((const char *) bytes.data() + at, ilen); at += ilen;

    for (uint32_t i = 0; i < nl; ++i) {
        uint32_t layer;
        if (!u32(layer) || layer > 4095 || m.layers.count((int) layer)) {
            err = "duplicate or malformed layer"; return false;
        }
        auto & lp = m.layers[(int) layer];
        for (auto * buf : { &lp.first, &lp.second }) {
            buf->resize(rows*index_dim);
            for (auto & x : *buf) {
                uint8_t b[2];
                if (!rd(b, 2)) {
                    err = "truncated map buffer"; return false;
                }
                x = bf16_to_f32((uint16_t) (b[0] | b[1] << 8));
                if (!std::isfinite(x)) {
                    err = "nonfinite map value"; return false;
                }
            }
        }
    }
    if (at != bytes.size()) {
        err = "trailing bytes"; return false;
    }

    out = std::move(m);
    return true;
}

const lsamap * get_map() {
    static std::once_flag once;
    static lsamap map;
    static bool ok = false;
    std::call_once(once, [] {
        const config & c = get_config();
        if (!c.enabled) {
            return;
        }
        std::string err;
        if (!load_lsamap(c.map_path, map, err)) {
            LLAMA_LOG_ERROR("llama_lsa_cpu: map %s: %s: disabled\n", c.map_path.c_str(), err.c_str());
            return;
        }
        if (!c.map_sha256.empty() && c.map_sha256 != map.payload_sha256) {
            LLAMA_LOG_ERROR("llama_lsa_cpu: map payload sha256 %s != pinned %s: disabled\n",
                    map.payload_sha256.c_str(), c.map_sha256.c_str());
            return;
        }
        ok = true;
        LLAMA_LOG_INFO("llama_lsa_cpu: map '%s' (%s) %d layers, geometry %d/%d/%d rope %d, top_k %d window %d sink %d min_kv %d\n",
                map.map_identity.c_str(), map.model_identity.c_str(), (int) map.layers.size(),
                map.q_heads, map.kv_heads, map.head_dim, map.rope_dim, c.top_k, c.window, c.sink, c.min_kv);
    });
    return ok ? &map : nullptr;
}

} // namespace llama_lsa_cpu
