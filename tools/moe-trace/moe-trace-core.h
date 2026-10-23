// moe-trace-core.h - model-free core of llama-moe-trace (INF-72 RT-1).
//
// Everything that decides whether a routing trace is TRUE lives here, with no dependency on
// llama.cpp or ggml, so it can be unit-tested without a model (tests/test-moe-trace-core.cpp):
//
//   * read_rows_strided   the RT-3 stride-safe reader (row by row at i * nb[1])
//   * hazard_check        the RT-3 tell-tale detector (every expert exactly the same number of
//                         times, sub-chance reuse) - fails the run loudly
//   * eor_tracker         the RT-2 EOR / IR_t rider, |E_t n E_{t-1}| / K, with chance baselines
//   * run_writer          the RT-2.1 on-disk contract (epyc.moe.routing_trace.v1): manifest +
//                         rows.bin + ids.i16, sha256 bound at close
//   * sha256              streaming, no external dependency
//   * selftest            a no-model proof that the reader and the guard work, run at startup
//
// Binary layouts (little endian, host is x86-64):
//   routing_trace.rows.bin   16-byte records:
//       u32 sample_id | u32 pos | u32 step | u16 spec_block | u8 phase (0 prefill, 1 decode)
//       | u8 spec_accepted (0 false, 1 true, 255 null)
//   routing_trace.ids.i16    int16 [rows][n_layers][max_top_k], layers with a smaller top_k pad -1
//   routing_trace.argsort.i16  (optional, SRP) int16 [rows][n_layers][n_experts_max], the FULL
//       descending-score expert order of ffn_moe_argsort, padded with -1
//
// Derived from github.com/Shriniwas410/cacheable-by-design @ 03366ef1 llama-moe-trace/moe-trace.cpp
// (MIT, Copyright (c) 2026 Shriniwas Ramesh Suram). The stride-safe row read is that tool's
// central idea; everything else here is new.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace moe_trace {

static constexpr const char * SCHEMA_ID    = "epyc.moe.routing_trace.v1";
static constexpr const char * IDS_NOTE     = "[rows, n_layers, top_k] row-major; layers with smaller top_k pad with -1";
static constexpr int          ROW_BYTES    = 16;
static constexpr int          PHASE_PREFILL = 0;
static constexpr int          PHASE_DECODE  = 1;
static constexpr uint8_t      ACC_FALSE = 0, ACC_TRUE = 1, ACC_NULL = 255;

// ---------------------------------------------------------------------------------------------
// sha256 (FIPS 180-4), streaming
// ---------------------------------------------------------------------------------------------
class sha256 {
public:
    sha256() { reset(); }

    void reset() {
        h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        len_ = 0;
        fill_ = 0;
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        len_ += n;
        while (n > 0) {
            const size_t take = std::min(n, sizeof(buf_) - fill_);
            std::memcpy(buf_ + fill_, p, take);
            fill_ += take; p += take; n -= take;
            if (fill_ == sizeof(buf_)) { block(buf_); fill_ = 0; }
        }
    }

    // digest of everything fed so far; the object stays usable (a copy is finalised)
    std::string hex() const {
        sha256 c = *this;
        const uint64_t bits = c.len_ * 8;
        const uint8_t one = 0x80, zero = 0;
        c.update(&one, 1);
        while (c.fill_ != 56) c.update(&zero, 1);
        uint8_t lenb[8];
        for (int i = 0; i < 8; i++) lenb[i] = (uint8_t) (bits >> (56 - 8 * i));
        c.update(lenb, 8);
        static const char * hx = "0123456789abcdef";
        std::string out;
        for (uint32_t w : c.h_) for (int s = 28; s >= 0; s -= 4) out.push_back(hx[(w >> s) & 0xF]);
        return out;
    }

    static std::string of(const void * data, size_t n) { sha256 s; s.update(data, n); return s.hex(); }

private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t) p[4*i] << 24) | ((uint32_t) p[4*i+1] << 16) | ((uint32_t) p[4*i+2] << 8) | (uint32_t) p[4*i+3];
        }
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }

    std::array<uint32_t, 8> h_;
    uint64_t len_;
    uint8_t  buf_[64];
    size_t   fill_;
};

// ---------------------------------------------------------------------------------------------
// RT-3: the stride-safe reader
// ---------------------------------------------------------------------------------------------
// ffn_moe_topk is ggml_argsort_top_k(): a ggml_view_4d of the [n_expert, n_tokens] argsort result with
// ne0 = k and nb1 = the PARENT's row stride (n_expert * 4 bytes). The view is NOT contiguous.
// A flat read of k * n_tokens int32 values from its base returns the first k*n_tokens/n_expert
// WHOLE argsort rows - each a permutation of all experts - so every expert shows up the same number
// of times and adjacent "tokens" never share an expert. That is the plausible-looking negative RT-3
// exists to prevent. The only correct read is one row at a time at offset i * nb[1].
struct strided_i32 {
    int64_t k      = 0;   // ne[0]: values per row
    int64_t n_rows = 0;   // ne[1]: tokens
    size_t  nb1    = 0;   // nb[1]: row stride in BYTES
};

// rd(dst, byte_offset_from_tensor_base, size): ggml_backend_tensor_get(t, dst, off, size) in the tool
using read_fn = std::function<void(void * dst, size_t offset, size_t size)>;

inline void read_rows_strided(const strided_i32 & v, const read_fn & rd, std::vector<int32_t> & out) {
    out.assign((size_t) v.k * (size_t) v.n_rows, 0);
    for (int64_t i = 0; i < v.n_rows; i++) {
        rd(out.data() + (size_t) i * (size_t) v.k, (size_t) i * v.nb1, (size_t) v.k * sizeof(int32_t));
    }
}

// narrow to int16 (RT-1 adjustment 1: int8 overflows the 256/512-expert fleet); false on any id
// outside [0, n_experts), which would mean the read or the graph is not what we think it is
inline bool narrow_ids(const std::vector<int32_t> & in, int n_experts, std::vector<int16_t> & out, std::string & err) {
    out.resize(in.size());
    if (n_experts <= 0 || n_experts > 32767) { err = "n_experts " + std::to_string(n_experts) + " outside int16"; return false; }
    for (size_t i = 0; i < in.size(); i++) {
        if (in[i] < 0 || in[i] >= n_experts) {
            err = "expert id " + std::to_string(in[i]) + " at flat index " + std::to_string(i) + " outside [0," + std::to_string(n_experts) + ")";
            return false;
        }
        out[i] = (int16_t) in[i];
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// RT-3: the hazard guard
// ---------------------------------------------------------------------------------------------
struct hazard_result {
    bool   evaluated    = false;  // false: not enough selection mass to judge (decode steps, tiny tails)
    bool   uniform      = false;  // every expert selected the same number of times (+-1)
    bool   near_uniform = false;  // coefficient of variation of the counts < 0.10 at mean >= 4
    bool   sub_chance   = false;  // adjacent-token reuse below k / n_experts
    bool   fatal        = false;
    int    count_min    = 0;
    int    count_max    = 0;
    double mean         = 0.0;
    double cv           = 0.0;
    double ir_mean      = 0.0;
    double chance       = 0.0;
    std::string why;
};

// ids: [n_tokens][k] int16 for ONE layer of ONE forward pass.
// FATAL when the per-expert selection counts are uniform (max - min <= 1), or near-uniform AND reuse is
// sub-chance. Real routing on >= 2 * n_experts selections never does this; a whole-argsort-row read always does.
// Sub-chance reuse ALONE is not fatal: it is a legitimate (negative) locality result and is reported, not blocked.
inline hazard_result hazard_check(const int16_t * ids, int n_tokens, int k, int n_experts) {
    hazard_result r;
    r.chance = n_experts > 0 ? (double) k / (double) n_experts : 0.0;
    const int64_t mass = (int64_t) n_tokens * k;
    // not judgeable: tiny expert pools, top-k a large share of the pool (counts are then uniform by construction),
    // too few tokens, or too little selection mass
    if (n_experts < 16 || (int64_t) k * 4 > n_experts || n_tokens < 8 || mass < 2LL * n_experts) return r;
    r.evaluated = true;

    std::vector<int> cnt(n_experts, 0);
    for (int64_t i = 0; i < mass; i++) {
        const int e = ids[i];
        if (e < 0 || e >= n_experts) { r.fatal = true; r.why = "expert id " + std::to_string(e) + " out of range"; return r; }
        cnt[e]++;
    }
    r.count_min = *std::min_element(cnt.begin(), cnt.end());
    r.count_max = *std::max_element(cnt.begin(), cnt.end());
    r.mean = (double) mass / (double) n_experts;
    double var = 0.0;
    for (int c : cnt) var += (c - r.mean) * (c - r.mean);
    var /= (double) n_experts;
    r.cv = r.mean > 0 ? std::sqrt(var) / r.mean : 0.0;

    std::vector<int> mark(n_experts, -1);
    int64_t overlap = 0;
    for (int t = 1; t < n_tokens; t++) {
        for (int j = 0; j < k; j++) mark[ids[(int64_t) (t - 1) * k + j]] = t;
        for (int j = 0; j < k; j++) if (mark[ids[(int64_t) t * k + j]] == t) overlap++;
    }
    r.ir_mean = (double) overlap / ((double) (n_tokens - 1) * k);

    r.uniform      = (r.count_max - r.count_min) <= 1;
    r.near_uniform = r.cv < 0.10 && r.mean >= 4.0;
    r.sub_chance   = r.ir_mean < r.chance;
    r.fatal        = r.uniform || (r.near_uniform && r.sub_chance);
    if (r.fatal) {
        char b[320];
        std::snprintf(b, sizeof b,
            "STRIDE-HAZARD SIGNATURE: per-expert counts min=%d max=%d (mean %.2f, cv %.3f)%s; adjacent-token reuse IR_t=%.4f vs chance %.4f%s - "
            "the router tensor was read flat instead of row by row at i*nb[1]",
            r.count_min, r.count_max, r.mean, r.cv, r.uniform ? " = every expert selected the same number of times" : "",
            r.ir_mean, r.chance, r.sub_chance ? " = SUB-CHANCE" : "");
        r.why = b;
    }
    return r;
}

// ---------------------------------------------------------------------------------------------
// layers, and the RT-2 EOR / IR_t rider
// ---------------------------------------------------------------------------------------------
struct layer_info {
    int il        = 0;   // graph layer index (the N of "ffn_moe_topk-N")
    int n_experts = 0;
    int top_k     = 0;
};

inline int max_top_k(const std::vector<layer_info> & ls) {
    int m = 0;
    for (const auto & l : ls) m = std::max(m, l.top_k);
    return m;
}

// IR_t = |E_t n E_{t-1}| / K over adjacent tokens of the SAME sample and layer (pairs never cross a sample
// boundary). The pair's phase is the later token's. Chance = K / n_experts (8/256 = 3.13%, 10/512 = 1.95%).
class eor_tracker {
public:
    void init(const std::vector<layer_info> & layers) {
        layers_ = layers;
        const size_t n = layers.size();
        sum_.assign(n, {0.0, 0.0}); pairs_.assign(n, {0, 0});
        prev_.assign(n, {}); have_prev_.assign(n, false);
        mark_.assign(n, {}); stamp_ = 0;
        for (size_t l = 0; l < n; l++) mark_[l].assign(layers[l].n_experts, 0);
        mk_ = max_top_k(layers);
    }

    void reset_sample() { std::fill(have_prev_.begin(), have_prev_.end(), false); }

    // ids: [n_rows][n_layers][max_k] as written to ids.i16
    void add_forward(int phase, const int16_t * ids, int n_rows) {
        const size_t nl = layers_.size();
        for (int r = 0; r < n_rows; r++) {
            for (size_t l = 0; l < nl; l++) {
                const int k = layers_[l].top_k;
                const int16_t * cur = ids + ((size_t) r * nl + l) * mk_;
                if (have_prev_[l]) {
                    ++stamp_;
                    for (int j = 0; j < k; j++) mark_[l][prev_[l][j]] = stamp_;
                    int ov = 0;
                    for (int j = 0; j < k; j++) if (mark_[l][cur[j]] == stamp_) ov++;
                    sum_[l][phase] += (double) ov / (double) k;
                    pairs_[l][phase]++;
                }
                prev_[l].assign(cur, cur + k);
                have_prev_[l] = true;
            }
        }
    }

    // the sidecar: routing_trace.eor.json
    std::string to_json() const {
        std::string o = "{\n  \"schema\": \"epyc.moe.routing_trace.eor.v1\",\n";
        o += "  \"definition\": \"IR_t = |E_t n E_{t-1}| / K over adjacent tokens of the same sample and layer; pair phase = later token; chance = K / n_experts\",\n";
        o += "  \"layers\": [\n";
        double macro[2] = {0, 0}; int mcount[2] = {0, 0};
        for (size_t l = 0; l < layers_.size(); l++) {
            const double chance = (double) layers_[l].top_k / (double) layers_[l].n_experts;
            o += "    {\"layer\": " + std::to_string(l) + ", \"il\": " + std::to_string(layers_[l].il)
               + ", \"n_experts\": " + std::to_string(layers_[l].n_experts) + ", \"top_k\": " + std::to_string(layers_[l].top_k)
               + ", \"chance\": " + num(chance);
            for (int ph = 0; ph < 2; ph++) {
                const uint64_t p = pairs_[l][ph];
                const double m = p ? sum_[l][ph] / (double) p : 0.0;
                o += std::string(", \"") + (ph ? "decode" : "prefill") + "\": {\"pairs\": " + std::to_string(p)
                   + ", \"ir_mean\": " + (p ? num(m) : "null") + ", \"ratio_to_chance\": " + (p ? num(m / chance) : "null") + "}";
                if (p) { macro[ph] += m; mcount[ph]++; }
            }
            o += (l + 1 < layers_.size()) ? "},\n" : "}\n";
        }
        o += "  ],\n  \"macro_mean_over_layers\": {";
        for (int ph = 0; ph < 2; ph++) {
            o += std::string("\"") + (ph ? "decode" : "prefill") + "\": " + (mcount[ph] ? num(macro[ph] / mcount[ph]) : "null");
            if (ph == 0) o += ", ";
        }
        o += "}\n}\n";
        return o;
    }

    // exposed for tests
    double ir_mean(size_t layer, int phase) const { return pairs_[layer][phase] ? sum_[layer][phase] / (double) pairs_[layer][phase] : -1.0; }
    uint64_t pairs(size_t layer, int phase) const { return pairs_[layer][phase]; }

private:
    static std::string num(double v) { char b[48]; std::snprintf(b, sizeof b, "%.6f", v); return b; }

    std::vector<layer_info> layers_;
    int mk_ = 0;
    std::vector<std::array<double, 2>>   sum_;
    std::vector<std::array<uint64_t, 2>> pairs_;
    std::vector<std::vector<int16_t>>    prev_;
    std::vector<bool>                    have_prev_;
    std::vector<std::vector<int>>        mark_;
    int stamp_ = 0;
};

// ---------------------------------------------------------------------------------------------
// the on-disk run (epyc.moe.routing_trace.v1)
// ---------------------------------------------------------------------------------------------
inline std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o.push_back((char) c);
        }
    }
    return o;
}

// a file that is hashed as it is written
class blob_writer {
public:
    ~blob_writer() { if (f_) std::fclose(f_); }
    bool open(const std::string & path) { path_ = path; f_ = std::fopen(path.c_str(), "wb"); bytes_ = 0; h_.reset(); return f_ != nullptr; }
    bool write(const void * p, size_t n) {
        if (n == 0) return true;
        h_.update(p, n); bytes_ += n;
        return f_ && std::fwrite(p, 1, n, f_) == n;
    }
    bool close() { if (!f_) return true; const bool ok = std::fflush(f_) == 0 && std::fclose(f_) == 0; f_ = nullptr; return ok; }
    std::string sha() const { return h_.hex(); }
    uint64_t bytes() const { return bytes_; }
    bool is_open() const { return f_ != nullptr; }
private:
    FILE * f_ = nullptr; std::string path_; sha256 h_; uint64_t bytes_ = 0;
};

struct run_meta {
    std::string model;
    std::string arch;
    std::string domain;
    int         batch_size = 0;
    std::vector<layer_info> layers;   // sorted by il; manifest "layer" is the ORDINAL in this list
    bool        srp = false;          // also write routing_trace.argsort.i16
};

struct sample_rec { uint32_t sample_id; std::string domain; uint64_t start_row; uint64_t n_rows; };

class run_writer {
public:
    // dir is created; the files are named routing_trace.*
    bool open(const std::string & dir, const run_meta & meta, std::string & err) {
        meta_ = meta; dir_ = dir; rows_n_ = 0; samples_.clear();
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) { err = "cannot create " + dir + ": " + ec.message(); return false; }
        if (!rows_.open(dir + "/routing_trace.rows.bin") || !ids_.open(dir + "/routing_trace.ids.i16")) { err = "cannot open trace files in " + dir; return false; }
        if (meta.srp && !srp_.open(dir + "/routing_trace.argsort.i16")) { err = "cannot open argsort file in " + dir; return false; }
        max_k_ = max_top_k(meta.layers);
        for (const auto & l : meta.layers) max_e_ = std::max(max_e_, l.n_experts);
        opened_ = true;
        return true;
    }

    // One forward pass of n rows. ids: [n][n_layers][max_k] (pad -1). argsort: [n][n_layers][max_e] or nullptr.
    // spec_block = n (RT-2.1 item 3: n_tokens of the forward). spec_accepted stays null: this tool never speculates.
    bool append_forward(uint32_t sample_id, const std::string & domain, uint32_t pos0, int phase, uint32_t step,
                        int n, const int16_t * ids, const int16_t * argsort, std::string & err) {
        if (!opened_) { err = "run not open"; return false; }
        if (n < 1 || n > 65535) { err = "forward of " + std::to_string(n) + " rows outside 1..65535"; return false; }
        if (samples_.empty() || samples_.back().sample_id != sample_id) {
            samples_.push_back({sample_id, domain, rows_n_, 0});
        }
        std::vector<uint8_t> rb((size_t) n * ROW_BYTES, 0);
        for (int t = 0; t < n; t++) {
            uint8_t * p = rb.data() + (size_t) t * ROW_BYTES;
            const uint32_t pos = pos0 + (uint32_t) t;
            std::memcpy(p + 0, &sample_id, 4);
            std::memcpy(p + 4, &pos, 4);
            std::memcpy(p + 8, &step, 4);
            const uint16_t blk = (uint16_t) n;
            std::memcpy(p + 12, &blk, 2);
            p[14] = (uint8_t) phase;
            p[15] = ACC_NULL;
        }
        const size_t nl = meta_.layers.size();
        if (!rows_.write(rb.data(), rb.size()) || !ids_.write(ids, (size_t) n * nl * max_k_ * sizeof(int16_t))) { err = "write failed"; return false; }
        if (meta_.srp) {
            if (!argsort) { err = "SRP enabled but no argsort tensor for this forward"; return false; }
            if (!srp_.write(argsort, (size_t) n * nl * max_e_ * sizeof(int16_t))) { err = "argsort write failed"; return false; }
        }
        samples_.back().n_rows += (uint64_t) n;
        rows_n_ += (uint64_t) n;
        return true;
    }

    uint64_t rows() const { return rows_n_; }
    size_t   n_samples() const { return samples_.size(); }
    int      max_e() const { return max_e_; }

    // finalise: flush, bind the hashes, write routing_trace.json atomically. Returns the manifest text.
    bool close(std::string & manifest_out, std::string & err) {
        if (!opened_) { err = "run not open"; return false; }
        if (rows_n_ == 0) { err = "no rows written"; return false; }
        if (!rows_.close() || !ids_.close() || (meta_.srp && !srp_.close())) { err = "flush failed"; return false; }
        std::string m = "{\n  \"schema\": \"" + std::string(SCHEMA_ID) + "\",\n";
        m += "  \"model\": \"" + json_escape(meta_.model) + "\",\n  \"arch\": \"" + json_escape(meta_.arch) + "\",\n";
        m += "  \"batch_size\": " + std::to_string(meta_.batch_size) + ",\n  \"layers\": [\n";
        for (size_t l = 0; l < meta_.layers.size(); l++) {
            m += "    {\"layer\": " + std::to_string(l) + ", \"n_experts\": " + std::to_string(meta_.layers[l].n_experts)
               + ", \"top_k\": " + std::to_string(meta_.layers[l].top_k) + ", \"id_dtype\": \"int16\"}"
               + (l + 1 < meta_.layers.size() ? ",\n" : "\n");
        }
        m += "  ],\n  \"samples\": [\n";
        for (size_t s = 0; s < samples_.size(); s++) {
            m += "    {\"sample_id\": " + std::to_string(samples_[s].sample_id) + ", \"domain\": \"" + json_escape(samples_[s].domain)
               + "\", \"start_row\": " + std::to_string(samples_[s].start_row) + ", \"n_rows\": " + std::to_string(samples_[s].n_rows) + "}"
               + (s + 1 < samples_.size() ? ",\n" : "\n");
        }
        m += "  ],\n  \"columns\": [\"sample_id\", \"pos\", \"phase\", \"step\", \"spec_block\", \"spec_accepted\"],\n";
        m += "  \"ids\": {\"dtype\": \"int16\", \"shape_note\": \"" + std::string(IDS_NOTE) + "\", \"sha256\": \"" + ids_.sha() + "\"},\n";
        m += "  \"rows_sha256\": \"" + rows_.sha() + "\"\n}\n";
        const std::string tmp = dir_ + "/routing_trace.json.tmp", fin = dir_ + "/routing_trace.json";
        FILE * f = std::fopen(tmp.c_str(), "wb");
        if (!f) { err = "cannot write " + tmp; return false; }
        const bool ok = std::fwrite(m.data(), 1, m.size(), f) == m.size() && std::fflush(f) == 0 && std::fclose(f) == 0;
        if (!ok || std::rename(tmp.c_str(), fin.c_str()) != 0) { err = "cannot finalise " + fin; return false; }
        manifest_out = m;
        opened_ = false;
        return true;
    }

    std::string srp_sha() const { return meta_.srp ? srp_.sha() : std::string(); }
    uint64_t srp_bytes() const { return meta_.srp ? srp_.bytes() : 0; }
    const std::string & dir() const { return dir_; }
    bool is_open() const { return opened_; }

private:
    run_meta meta_; std::string dir_;
    blob_writer rows_, ids_, srp_;
    std::vector<sample_rec> samples_;
    uint64_t rows_n_ = 0;
    int max_k_ = 0, max_e_ = 0;
    bool opened_ = false;
};

// ---------------------------------------------------------------------------------------------
// one forward pass, assembled from the per-layer taps
// ---------------------------------------------------------------------------------------------
enum { RC_DONE = 0, RC_ERROR = 1, RC_INCOMPLETE = 2, RC_HAZARD = 3, RC_SELFTEST = 4, RC_STOPPED = 10 };

// what the eval callback captured for one layer of one forward pass
struct layer_tap {
    int64_t k = 0, n_rows = 0, n_experts = 0;
    std::vector<int16_t> ids;      // [n_rows][k]
    std::vector<int16_t> order;    // SRP: [n_rows][n_experts], full descending order
    bool have_topk = false, have_order = false;
};

struct guard_stats {
    uint64_t checks = 0, sub_chance = 0;
    double   cv_min = 1e9, cv_max = 0.0;
};

struct assembled_forward {
    std::vector<int16_t> ids;      // [n][n_layers][max_k], -1 pad
    std::vector<int16_t> order;    // [n][n_layers][max_e], -1 pad (SRP only)
};

// Turns the taps of ONE forward of n tokens into rows, or refuses. This is where a short trace is made impossible:
//  * the layer set is fixed by the first forward and enforced on every later one (a graph that bypassed the tap, e.g.
//    qwen4exp's fused decode, or a pruned last layer, is RC_INCOMPLETE - never a silently shorter trace)
//  * every layer must have exactly n rows, the same k and n_experts as before
//  * on a prefill forward every layer is run through hazard_check (RC_HAZARD)
inline int assemble_forward(const std::map<int, layer_tap> & taps, int n, bool prefill, bool srp,
                            std::vector<layer_info> & layers, bool & layers_known,
                            assembled_forward & out, guard_stats & gs, std::string & msg) {
    if (taps.empty()) {
        msg = "no ffn_moe_topk-* tap fired in a forward of " + std::to_string(n) + " tokens: the graph path was bypassed (fused decode?) or the arch has no build_moe_ffn MoE";
        return RC_INCOMPLETE;
    }
    if (!layers_known) {
        for (const auto & kv : taps) layers.push_back({kv.first, (int) kv.second.n_experts, (int) kv.second.k});
        layers_known = true;
    }
    if (taps.size() != layers.size()) {
        msg = "forward tapped " + std::to_string(taps.size()) + " layers, expected " + std::to_string(layers.size()) + ": refusing a short trace";
        return RC_INCOMPLETE;
    }
    const size_t nl = layers.size();
    const int mk = max_top_k(layers);
    int me = 0;
    for (const auto & l : layers) me = std::max(me, l.n_experts);
    out.ids.assign((size_t) n * nl * mk, -1);
    out.order.clear();
    if (srp) out.order.assign((size_t) n * nl * me, -1);
    for (size_t l = 0; l < nl; l++) {
        auto it = taps.find(layers[l].il);
        if (it == taps.end()) { msg = "layer il=" + std::to_string(layers[l].il) + " missing from the forward: refusing a short trace"; return RC_INCOMPLETE; }
        const layer_tap & lt = it->second;
        if (!lt.have_topk || lt.n_rows != n || lt.k != layers[l].top_k || lt.n_experts != layers[l].n_experts) {
            msg = "layer il=" + std::to_string(layers[l].il) + " tapped " + std::to_string(lt.n_rows) + " rows (k=" + std::to_string(lt.k)
                + ", n_experts=" + std::to_string(lt.n_experts) + "), expected " + std::to_string(n) + " rows (k=" + std::to_string(layers[l].top_k)
                + ", n_experts=" + std::to_string(layers[l].n_experts) + "): refusing a short trace (last-layer output pruning? request logits for every position)";
            return RC_INCOMPLETE;
        }
        if (prefill) {
            const hazard_result hz = hazard_check(lt.ids.data(), n, (int) lt.k, (int) lt.n_experts);
            if (hz.evaluated) {
                gs.checks++;
                if (hz.sub_chance) gs.sub_chance++;
                gs.cv_min = std::min(gs.cv_min, hz.cv); gs.cv_max = std::max(gs.cv_max, hz.cv);
            }
            if (hz.fatal) { msg = "RT-3 HAZARD at layer il=" + std::to_string(layers[l].il) + ": " + hz.why; return RC_HAZARD; }
        }
        for (int t = 0; t < n; t++)
            std::memcpy(&out.ids[((size_t) t * nl + l) * mk], &lt.ids[(size_t) t * lt.k], (size_t) lt.k * sizeof(int16_t));
        if (srp) {
            if (!lt.have_order || lt.order.size() != (size_t) n * lt.n_experts) { msg = "SRP argsort tap missing or short for il=" + std::to_string(layers[l].il); return RC_INCOMPLETE; }
            for (int t = 0; t < n; t++)
                std::memcpy(&out.order[((size_t) t * nl + l) * me], &lt.order[(size_t) t * lt.n_experts], (size_t) lt.n_experts * sizeof(int16_t));
        }
    }
    return RC_DONE;
}

// ---------------------------------------------------------------------------------------------
// no-model selftest: the stride reader reads the right values, the guard catches the wrong read
// ---------------------------------------------------------------------------------------------
namespace detail {

struct lcg { uint64_t s; explicit lcg(uint64_t seed) : s(seed) {} uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
             double unit() { return (next() & 0xFFFFFF) / 16777216.0; } };

// Synthetic router: skewed expert popularity + temporal persistence (adjacent tokens share experts), so
// the correct read has non-uniform counts and above-chance reuse - what real routing looks like.
// Returns the full argsort matrix [n_tokens][n_experts] (descending score), row-major int32.
inline std::vector<int32_t> synth_argsort(int n_experts, int n_tokens, uint64_t seed) {
    lcg rng(seed);
    std::vector<double> pop(n_experts);
    for (int e = 0; e < n_experts; e++) pop[e] = 1.0 / (1.0 + e % 29) + 0.2 * rng.unit();
    std::vector<double> carry(n_experts, 0.0);
    std::vector<int32_t> out((size_t) n_experts * n_tokens);
    std::vector<int> order(n_experts);
    for (int t = 0; t < n_tokens; t++) {
        std::vector<double> sc(n_experts);
        for (int e = 0; e < n_experts; e++) {
            carry[e] = 0.6 * carry[e] + 0.4 * rng.unit();
            sc[e] = pop[e] + carry[e];
        }
        for (int e = 0; e < n_experts; e++) order[e] = e;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return sc[a] != sc[b] ? sc[a] > sc[b] : a < b; });
        for (int e = 0; e < n_experts; e++) out[(size_t) t * n_experts + e] = order[e];
    }
    return out;
}

// the FORBIDDEN read, kept only so the selftest can prove the guard catches it
inline std::vector<int32_t> flat_read_WRONG(const std::vector<int32_t> & mem, const strided_i32 & v) {
    return std::vector<int32_t>(mem.begin(), mem.begin() + (size_t) v.k * (size_t) v.n_rows);
}

} // namespace detail

// returns "" on success, else the first failure. Pure CPU, microseconds, run before every model load.
inline std::string selftest() {
    static const char * abc = "abc";
    if (sha256::of(abc, 3) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") return "sha256(abc) wrong";
    if (sha256::of("", 0) != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") return "sha256(empty) wrong";

    for (int n_experts : {256, 512}) {
        const int k = n_experts == 256 ? 8 : 10, n_tokens = 512;
        std::vector<int32_t> mem = detail::synth_argsort(n_experts, n_tokens, 12345 + n_experts);   // [n_tokens][n_experts]
        strided_i32 v; v.k = k; v.n_rows = n_tokens; v.nb1 = (size_t) n_experts * sizeof(int32_t);  // the top-k view of it
        const uint8_t * base = reinterpret_cast<const uint8_t *>(mem.data());
        read_fn rd = [&](void * dst, size_t off, size_t sz) { std::memcpy(dst, base + off, sz); };

        std::vector<int32_t> good;
        read_rows_strided(v, rd, good);
        for (int t = 0; t < n_tokens; t++)
            for (int j = 0; j < k; j++)
                if (good[(size_t) t * k + j] != mem[(size_t) t * n_experts + j]) return "strided read returned the wrong element";
        std::vector<int16_t> g16; std::string e;
        if (!narrow_ids(good, n_experts, g16, e)) return "narrow_ids rejected a valid read: " + e;
        const hazard_result hg = hazard_check(g16.data(), n_tokens, k, n_experts);
        if (!hg.evaluated || hg.fatal) return "guard flagged a correct strided read: " + hg.why;
        if (hg.ir_mean <= hg.chance) return "synthetic routing is not above chance; selftest fixture broken";

        const std::vector<int32_t> bad = detail::flat_read_WRONG(mem, v);
        std::vector<int16_t> b16;
        if (!narrow_ids(bad, n_experts, b16, e)) return "narrow_ids rejected the flat read: " + e;
        const hazard_result hb = hazard_check(b16.data(), n_tokens, k, n_experts);
        if (!hb.evaluated || !hb.fatal || !hb.uniform) return "guard MISSED the flat-read stride hazard (n_experts=" + std::to_string(n_experts) + ")";
    }
    int16_t big[2] = {300, 200};
    std::vector<int32_t> wide = {300, 200}; std::vector<int16_t> o; std::string e;
    if (!narrow_ids(wide, 512, o, e) || o[0] != big[0] || o[1] != big[1]) return "ids above 127 not preserved (int8 regression)";
    if (narrow_ids(wide, 256, o, e)) return "narrow_ids accepted an id >= n_experts";
    return "";
}

} // namespace moe_trace
