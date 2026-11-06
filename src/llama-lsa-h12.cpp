#include "llama-lsa-h12.h"

#include "llama-impl.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>

namespace llama_lsa_h12 {

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
        c.enabled      = env_int("LLAMA_LSA_H12", 0) != 0;
        c.map_path     = env_str("LLAMA_LSA_H12_MAP");
        c.map_sha256   = env_str("LLAMA_LSA_H12_MAP_SHA256");
        c.top_k        = env_int("LLAMA_LSA_H12_TOPK",   4096);
        c.window       = env_int("LLAMA_LSA_H12_WINDOW",  512);
        c.sink         = env_int("LLAMA_LSA_H12_SINK",      4);
        c.min_kv_ratio = env_int("LLAMA_LSA_H12_MIN_KV_RATIO", 8);
        if (c.enabled && (c.top_k < 1 || c.window < 0 || c.sink < 0 || c.min_kv_ratio < 1)) {
            LLAMA_LOG_ERROR("llama_lsa_h12: invalid budgets (top_k %d window %d sink %d ratio %d): disabled\n",
                    c.top_k, c.window, c.sink, c.min_kv_ratio);
            c.enabled = false;
        }
        if (c.enabled && c.map_path.empty()) {
            LLAMA_LOG_ERROR("llama_lsa_h12: LLAMA_LSA_H12=1 without LLAMA_LSA_H12_MAP: disabled\n");
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
// LSAMAP1 loader (format of experimental/lsa-native-c5-20261103 src/llama-lsa-map.cpp):
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

void expand_q(const lsamap & m, const std::vector<float> & mq, std::vector<float> & out) {
    const int D = m.head_dim, R = m.rope_dim, W = D - R, G = m.q_heads/m.kv_heads;
    const int64_t n_in = (int64_t) D*m.q_heads;
    out.assign(n_in*index_dim, 0.0f);
    for (int o = 0; o < index_dim; ++o) {
        for (int h = 0; h < m.q_heads; ++h) {
            for (int d = R; d < D; ++d) {
                out[o*n_in + h*D + d] = mq[((int64_t) (h/G)*W + d - R)*index_dim + o];
            }
        }
    }
}

void expand_k(const lsamap & m, const std::vector<float> & mk, std::vector<float> & out) {
    const int D = m.head_dim, R = m.rope_dim, W = D - R;
    const int64_t n_in = (int64_t) D*m.kv_heads;
    out.assign(n_in*index_dim, 0.0f);
    for (int o = 0; o < index_dim; ++o) {
        for (int h = 0; h < m.kv_heads; ++h) {
            for (int d = R; d < D; ++d) {
                out[o*n_in + h*D + d] = mk[((int64_t) h*W + d - R)*index_dim + o];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// graph

ggml_tensor * build_scores(ggml_context * ctx, ggml_tensor * q_raw, ggml_tensor * k_cache,
                           ggml_tensor * mq, ggml_tensor * mk, ggml_tensor * kq_mask,
                           ggml_tensor * kpos, ggml_tensor * kbias, ggml_tensor * qwin) {
    const int64_t hd       = k_cache->ne[0];
    const int64_t n_kv     = k_cache->ne[2];
    const int64_t n_stream = k_cache->ne[3];
    const int64_t n_tokens = q_raw->ne[2];
    const int64_t n_tps    = n_tokens/n_stream;

    GGML_ASSERT(n_tokens % n_stream == 0);
    GGML_ASSERT(mq->ne[0] == q_raw->ne[0]*q_raw->ne[1] && mq->ne[1] == index_dim);
    GGML_ASSERT(mk->ne[0] == hd*k_cache->ne[1]         && mk->ne[1] == index_dim);
    GGML_ASSERT(k_cache->nb[1] == ggml_row_size(k_cache->type, hd)); // one cell = one contiguous row of all kv heads

    // keys: one GEMM over full cache rows [hd*n_head_kv, n_kv, n_stream] (the RoPE dims hit zero map rows)
    ggml_tensor * kfull = ggml_view_3d(ctx, k_cache, hd*k_cache->ne[1], n_kv, n_stream, k_cache->nb[2], k_cache->nb[3], 0);
    ggml_tensor * mkb   = n_stream > 1 ? ggml_repeat_4d(ctx, mk, mk->ne[0], mk->ne[1], n_stream, 1) : mk;
    ggml_tensor * kp    = ggml_mul_mat(ctx, kfull, mkb);                       // [n_kv, 64, n_stream]
    kp = ggml_cont(ctx, ggml_permute(ctx, kp, 1, 0, 2, 3));                    // [64, n_kv, n_stream]
    kp = ggml_l2_norm(ctx, kp, 1e-12f);
    ggml_set_name(kp, "lsa_h12_khat");

    // queries: one GEMM over full q rows [hd*n_head, n_tokens] (sums the GQA group and the kv heads)
    ggml_tensor * qr = ggml_is_contiguous(q_raw) ? q_raw : ggml_cont(ctx, q_raw);
    qr = ggml_reshape_2d(ctx, qr, q_raw->ne[0]*q_raw->ne[1], n_tokens);
    ggml_tensor * qp = ggml_mul_mat(ctx, mq, qr);                              // [64, n_tokens]
    qp = ggml_l2_norm(ctx, qp, 1e-12f);
    qp = ggml_reshape_3d(ctx, qp, index_dim, n_tps, n_stream);
    ggml_set_name(qp, "lsa_h12_qhat");

    // mul_mat matches ne[2]: the queries of stream s only meet the cells of stream s
    ggml_tensor * score = ggml_mul_mat(ctx, kp, qp);                           // [n_kv, n_tps, n_stream]
    ggml_set_name(score, "lsa_h12_score");

    // local window: kpos - qpos + window > 0 -> + window_bonus
    ggml_tensor * win = ggml_add(ctx, ggml_repeat(ctx, kpos, score), qwin);
    win = ggml_scale(ctx, ggml_step(ctx, win), window_bonus);
    score = ggml_add(ctx, score, win);
    // attention sinks
    score = ggml_add(ctx, score, kbias);

    // causal / sequence mask (f16 for FA): -inf cells never win over finite ones
    ggml_tensor * m = kq_mask;
    if (m->ne[1] != n_tps) {
        m = ggml_view_4d(ctx, m, n_kv, n_tps, 1, n_stream, m->nb[1], m->nb[2], m->nb[3], 0);
    }
    m = m->type == GGML_TYPE_F32 && ggml_is_contiguous(m) ? m : ggml_cast(ctx, m, GGML_TYPE_F32);
    score = ggml_add(ctx, score, ggml_reshape_3d(ctx, m, n_kv, n_tps, n_stream));
    ggml_set_name(score, "lsa_h12_score_biased");

    return score;
}

// ---------------------------------------------------------------------------------------------
// device maps (process lifetime: a few MB per model; never freed so no backend teardown ordering issue)

struct registry_entry {
    bool                                ok = false;
    device_maps                         maps;
    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;
};

const device_maps * get_device_maps(const void * model_key, const std::string & model_name,
                                    int q_heads, int kv_heads, int head_dim, int rope_dim,
                                    const std::map<int, ggml_backend_buffer_type_t> & layer_buft) {
    const config & cfg = get_config();
    if (!cfg.enabled) {
        return nullptr;
    }

    static std::mutex mtx;
    static std::map<const void *, std::unique_ptr<registry_entry>> reg;

    std::lock_guard<std::mutex> lock(mtx);

    auto it = reg.find(model_key);
    if (it != reg.end()) {
        return it->second->ok ? &it->second->maps : nullptr;
    }
    auto entry = std::make_unique<registry_entry>();
    registry_entry & e = *entry;
    reg.emplace(model_key, std::move(entry));

    auto fail = [&](const std::string & why) -> const device_maps * {
        LLAMA_LOG_ERROR("llama_lsa_h12: %s: PCA indexer disabled (dense attention)\n", why.c_str());
        return nullptr;
    };

    if (!cfg.map_sha256.empty()) {
        std::ifstream in(cfg.map_path, std::ios::binary);
        std::vector<char> all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string got = sha256_hex(all.data(), all.size());
        if (got != cfg.map_sha256) {
            return fail("map file sha256 " + got + " != pinned " + cfg.map_sha256);
        }
    }

    lsamap m;
    std::string err;
    if (!load_lsamap(cfg.map_path, m, err)) {
        return fail("cannot load " + cfg.map_path + ": " + err);
    }
    if (m.q_heads != q_heads || m.kv_heads != kv_heads || m.head_dim != head_dim || m.rope_dim != rope_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf), "map geometry q%d/kv%d/hd%d/rope%d != model q%d/kv%d/hd%d/rope%d",
                m.q_heads, m.kv_heads, m.head_dim, m.rope_dim, q_heads, kv_heads, head_dim, rope_dim);
        return fail(buf);
    }
    if (m.layers.size() != layer_buft.size()) {
        return fail("map covers " + std::to_string(m.layers.size()) + " layers, model has " + std::to_string(layer_buft.size()) + " global attention layers");
    }
    for (const auto & lb : layer_buft) {
        if (!m.layers.count(lb.first)) {
            return fail("map has no layer " + std::to_string(lb.first));
        }
    }

    std::vector<float> eq, ek;
    for (const auto & lb : layer_buft) {
        ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * mq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) head_dim*q_heads,  index_dim);
        ggml_tensor * mk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) head_dim*kv_heads, index_dim);
        ggml_format_name(mq, "lsa_h12_mq.%d", lb.first);
        ggml_format_name(mk, "lsa_h12_mk.%d", lb.first);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, lb.second);
        if (!buf) {
            return fail("cannot allocate map buffer");
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        expand_q(m, m.layers[lb.first].first,  eq);
        expand_k(m, m.layers[lb.first].second, ek);
        ggml_backend_tensor_set(mq, eq.data(), 0, ggml_nbytes(mq));
        ggml_backend_tensor_set(mk, ek.data(), 0, ggml_nbytes(mk));
        e.ctxs.push_back(ctx);
        e.bufs.push_back(buf);
        e.maps.layers[lb.first] = { mq, mk };
    }

    e.maps.identity = m.model_identity + "/" + m.map_identity + " payload=" + m.payload_sha256 + " source=" + m.source_sha256;
    e.ok = true;

    // witness for the gate (strings-visible marker + WARN so it survives default log filters)
    LLAMA_LOG_WARN("llama_lsa_h12: LSA-H12-INDEXER PCA key scorer enabled for %s: map %s (%s), %zu layers, top_k %d window %d sink %d width %d min_kv_ratio %d\n",
            model_name.c_str(), cfg.map_path.c_str(), e.maps.identity.c_str(), e.maps.layers.size(),
            cfg.top_k, cfg.window, cfg.sink, cfg.width(), cfg.min_kv_ratio);

    return &e.maps;
}

} // namespace llama_lsa_h12
