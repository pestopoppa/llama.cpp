#pragma once

// LSA-H12-INDEXER: D-LSA PCA-map key scorer feeding the AK-ASM-H12 fused sparse flash attention.
//
// The released LSA map (LSAMAP1, BF16) projects the non-RoPE part of a normalized query / key onto
// 64 index dimensions. Per global-attention layer, per query token t and KV cell j:
//
//   qhat_t = l2norm( Mq^T . concat_kvh( sum_{g in group(kvh)} q[t, h=kvh*G+g, rope_dim:head_dim] ) )
//   khat_j = l2norm( Mk^T . concat_kvh( k[j, kvh, rope_dim:head_dim] ) )
//   score  = qhat_t . khat_j  (+ window/sink bonus) (+ KQ mask)
//   top_k  = TOP_K(score, width = top_k + window + sink)
//
// The whole scorer is ordinary ggml ops (MUL_MAT, L2_NORM, TOP_K, ...) so it stays on the GPU inside
// the captured graph. top_k + the base mask are handed to the FLASH_ATTN_EXT node as src[5]/src[6],
// exactly like qwen4exp build_attn_qsa, so ggml-cuda's qsa-sparse path runs the fused gather attention.
//
// The non-RoPE dims are invariant under (partial) RoPE, so keys are read straight from the f16 KV cache
// and queries may be taken before or after RoPE. Map expansion zero-pads the RoPE dims, so a full
// cache row (all kv heads) and a full q row (all heads) are projected by one GEMM each.
//
// Only the scorer math and the map format are ported from the LSA native/reference branches; the Ward
// bucket selection of the author's runtime is NOT reproduced (flat per-key top-k instead).
//
// Disabled by default. Env:
//   LLAMA_LSA_H12=1                 enable (also needs a map)
//   LLAMA_LSA_H12_MAP=<path>        LSAMAP1 file
//   LLAMA_LSA_H12_MAP_SHA256=<hex>  optional pin of the file digest (mismatch -> disabled)
//   LLAMA_LSA_H12_TOPK / _WINDOW / _SINK   budgets (4096 / 512 / 4)
//   LLAMA_LSA_H12_MIN_KV_RATIO      use the indexer only when n_kv >= ratio * width (8, = H12's own rule)

#include "ggml.h"
#include "ggml-backend.h"

#include <map>
#include <string>
#include <vector>

namespace llama_lsa_h12 {

constexpr int   index_dim    = 64;
constexpr float window_bonus = 4.0f; // scores are cosines in [-1, 1]: any bonus > 2 forces the cell
constexpr float sink_bonus   = 8.0f;
constexpr float empty_pos    = -1.0e9f;

struct config {
    bool        enabled      = false;
    std::string map_path;
    std::string map_sha256;
    int         top_k        = 4096;
    int         window       = 512;
    int         sink         = 4;
    int         min_kv_ratio = 8;

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

// ggml layouts (ne0 fastest), F32:
//   mq_full [head_dim*q_heads, 64]: element (h*head_dim + d, o) = Mq[(h/G)*(head_dim-rope_dim) + d - rope_dim, o], 0 for d < rope_dim
//   mk_full [head_dim*kv_heads, 64]: element (kvh*head_dim + d, o) = Mk[kvh*(head_dim-rope_dim) + d - rope_dim, o], 0 for d < rope_dim
void expand_q(const lsamap & m, const std::vector<float> & mq, std::vector<float> & out);
void expand_k(const lsamap & m, const std::vector<float> & mk, std::vector<float> & out);

// the scorer graph, shared by the model and the tests
//   q_raw   F32 [head_dim, n_head, n_tokens] (contiguous), tokens stream-major
//   k_cache [head_dim, n_head_kv, n_kv, n_stream] (get_k view, F16)
//   mq/mk   expanded maps (see above)
//   kq_mask F16/F32 [n_kv, n_tps, 1, n_stream]
//   kpos    F32 [n_kv, 1, n_stream]  cell position (empty_pos for empty cells)
//   kbias   F32 [n_kv, 1, n_stream]  sink_bonus for cells with pos < sink, else 0
//   qwin    F32 [1, n_tps, n_stream] window - query position
// returns the biased scores F32 [n_kv, n_tps, n_stream]
ggml_tensor * build_scores(ggml_context * ctx, ggml_tensor * q_raw, ggml_tensor * k_cache,
                           ggml_tensor * mq, ggml_tensor * mk, ggml_tensor * kq_mask,
                           ggml_tensor * kpos, ggml_tensor * kbias, ggml_tensor * qwin);

// device-resident expanded maps for one model; nullptr when disabled or the map failed to load / bind.
// layer_buft: global layer -> buffer type of that layer's weights.
struct device_maps {
    std::map<int, std::pair<ggml_tensor *, ggml_tensor *>> layers; // il -> (mq_full, mk_full)
    std::string identity;
};

const device_maps * get_device_maps(const void * model_key, const std::string & model_name,
                                    int q_heads, int kv_heads, int head_dim, int rope_dim,
                                    const std::map<int, ggml_backend_buffer_type_t> & layer_buft);

} // namespace llama_lsa_h12
