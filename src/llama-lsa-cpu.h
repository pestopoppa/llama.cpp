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
#include <string>
#include <vector>

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

} // namespace llama_lsa_cpu
