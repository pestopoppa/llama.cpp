#pragma once

// LSA-CPU-SPARSE: D-LSA flat key selection for CPU decode (experimental, default OFF).
//
// Released LSA map (LSAMAP1, BF16, W_Q == W_K for the PCA map) projects the RoPE-free part of the
// query / key onto 64 index dims. Per global-attention layer:
//
//   qhat   = l2norm( Wq^T . concat_kvh( sum_{g in group(kvh)} q[h = kvh*G + g, rope:head_dim] ) )
//   khat_j = l2norm( Wk^T . concat_kvh( k[j, kvh, rope:head_dim] ) )            (one key for all heads)
//
// khat is written ONCE per token into a persistent BF16 index-key cache at K-store time (the GPU
// H12 attempt re-projected the whole f16 K every ubatch, which is what sank it). At decode
// (n_tokens == 1) the scorer computes qhat . khat_j over the valid cells, picks the exact top_k
// (histogram threshold + exact tie break by lower cell index) plus the newest `window` and the oldest
// `sink` positions, and writes -inf into the KQ mask for every other cell. The CPU flash attention
// already skips -inf mask entries, so the sparse attention is the existing kernel.
//
// Env (all off by default):
//   LLAMA_LSA_CPU=1                 enable (also needs a map)
//   LLAMA_LSA_CPU_MAP=<path>        LSAMAP1 file
//   LLAMA_LSA_CPU_MAP_SHA256=<hex>  optional pin of the payload digest (mismatch -> disabled)
//   LLAMA_LSA_CPU_TOPK / _WINDOW / _SINK   budgets (4096 / 512 / 4)
//   LLAMA_LSA_CPU_MIN_KV            apply the selection only when the graph's n_kv >= this (default 2*width)

#include "ggml.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class llama_kv_cells;

namespace llama_lsa_cpu {

constexpr int index_dim = 64;

struct config {
    bool        enabled = false;
    std::string map_path;
    std::string map_sha256;
    int         top_k   = 4096;
    int         window  = 512;
    int         sink    = 4;
    int         min_kv  = 0;

    int width() const { return top_k + window + sink; }
};

// read once from the environment
const config & get_config();

struct lsamap {
    int q_heads  = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int rope_dim = 0;

    std::string model_identity;
    std::string map_identity;
    std::string source_sha256;
    std::string payload_sha256;

    // layer -> (Q map, K map), each row-major [kv_heads*(head_dim - rope_dim), 64] as in the file
    std::map<int, std::pair<std::vector<float>, std::vector<float>>> layers;

    int rows() const { return kv_heads*(head_dim - rope_dim); }
};

std::string sha256_hex(const void * data, size_t size);

bool load_lsamap(const std::string & path, lsamap & out, std::string & err);

// the process-wide map when enabled and loaded, else nullptr (logged once)
const lsamap * get_map();

// out[64] = l2norm(W^T x), W row-major [rows][64]
void project(const float * w, int rows, const float * x, float * out);

// true when `il` has a map and the layer geometry matches it
bool layer_supported(const lsamap & m, int il, int n_head, int n_head_kv, int head_dim);

// Per (KV cache, global layer) persistent state. Owned by llama_kv_cache; single stream only.
struct layer_state {
    int il = -1;
    int q_heads = 0, kv_heads = 0, head_dim = 0, rope_dim = 0;
    const float * wq = nullptr; // [rows][64]
    const float * wk = nullptr; // [rows][64]

    ggml_tensor * ik = nullptr; // BF16 [64, kv_size], host buffer; row = KV cell

    // cell positions are read at compute time (cells are applied before the graph runs);
    // nullptr -> pos = cell index (tests)
    const llama_kv_cells * cells = nullptr;

    // position the index key of a cell was written for; -1 = invalid. A cell whose position differs
    // (seq_rm + reuse, K-shift, state restore) is never scored: it is force-kept (dense-safe).
    std::vector<int32_t> ik_pos;

    // decode scratch [kv_size]
    std::vector<float> score;

    // last decode selection statistics (diagnostic)
    int64_t last_eligible = 0, last_kept = 0;

    void init(const lsamap & m, int il, uint32_t kv_size);
    void invalidate_all();
    int32_t cell_pos(int64_t cell) const;
};

// ---- decode scorer + exact top-k (exposed for tests) ----

// qhat[64] = l2norm(Wq^T concat_kvh(sum_{g in group} q[h, rope:])), q row [head_dim, q_heads] with strides nb0/nb1
void project_query(const layer_state & st, const char * q, size_t nb0, size_t nb1, float * qhat);

// out[j] = qhat . ik[j] for j in [lo, hi), ik BF16 rows of 64 (AVX-512 when available)
void score_bf16(const uint16_t * ik, int64_t lo, int64_t hi, const float * qhat, float * out);

// Exact selection over score[0..n): +inf = forced keep (window / sink / stale), -inf = excluded,
// finite = candidate. Keeps the top_k largest candidates (ties -> lower index) via a 4096-bin
// histogram threshold + exact sort of the threshold bin. keep[j] = 1 for kept. Returns #kept.
int64_t select_topk(const float * score, int64_t n, int top_k, uint8_t * keep);

// Index-key write, run at K-store time for every ubatch:
//   k_cur  F32 [head_dim, n_head_kv, n_tokens], k_idxs I64 [n_tokens] (cells, as for cpy_k)
// returns an I64 copy of k_idxs that orders the write before any consumer.
ggml_tensor * build_ik_write(ggml_context * ctx, layer_state * st, ggml_tensor * k_cur, ggml_tensor * k_idxs);

// Decode (one query token) index mask: returns kq_mask with -inf for every cell outside
// top_k + window + sink. q_cur F32 [head_dim, q_heads, 1] (logical, pre-rotation);
// ik_written = build_ik_write() result of the same graph (orders the score after the write).
ggml_tensor * build_decode_mask(ggml_context * ctx, layer_state * st, ggml_tensor * kq_mask,
                                ggml_tensor * q_cur, ggml_tensor * ik_written);

} // namespace llama_lsa_cpu
