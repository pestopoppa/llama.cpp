#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llama_lsa {

constexpr int index_dimension = 64;
constexpr int cluster_size = 512;
constexpr int buckets_per_cluster = 64;
constexpr int bucket_capacity = 16;

struct geometry {
    int q_heads = 24;
    int kv_heads = 4;
    int head_dim = 256;
    int rope_dim = 64;
    bool valid() const;
    int map_rows() const { return kv_heads * (head_dim - rope_dim); }
};

// Contiguous channels (no element stride), arbitrary byte strides between rows
// and heads. Transposed channels and overlapping rows/heads are unsupported.
struct tensor_view {
    const void * data = nullptr;
    ggml_type type = GGML_TYPE_F32;
    int64_t rows = 0;
    int heads = 0;
    int dim = 0;
    size_t row_stride = 0;
    size_t head_stride = 0;
};

struct projection_maps {
    geometry shape;
    // Row-major [shape.map_rows(), 64], BF16 as in the released maps.
    std::vector<ggml_bf16_t> q;
    std::vector<ggml_bf16_t> k;
    std::string identity;
    bool valid() const;
};

// Inputs MUST be normalized logical Q/K before RoPE and KV quantization rotations.
// Each output row has 64 FP32 elements; key values are rounded to BF16.
bool project_queries(const tensor_view & logical_q, const projection_maps & maps,
                     float * output, std::string * error = nullptr);
bool project_keys(const tensor_view & logical_k, const projection_maps & maps,
                  float * output, std::string * error = nullptr);

struct bucket {
    float centroid[index_dimension] = {};
    int32_t members[bucket_capacity];
    bucket();
};

struct budgets {
    int top_k = 4096;
    int top_p = 1024;
    int window = 512;
    int sink = 4;
    bool valid() const;
};

// One instance per request/layer. Caller serializes access; no global cache.
// Reset on request/slot reuse or map replacement; epoch checks fail closed.
class index_cache {
public:
    void reset(uint64_t epoch, const std::string & map_identity);
    bool matches(uint64_t epoch, const std::string & map_identity) const;
    // Append only at size(); projected keys must already be BF16-rounded.
    bool append(uint64_t epoch, int64_t first, const float * keys, int64_t count,
                std::string * error = nullptr);
    // Rollback removes every cluster touching the removed suffix.
    bool truncate(uint64_t epoch, int64_t count, std::string * error = nullptr);
    bool select(uint64_t epoch, const float * index_query, int32_t position,
                const budgets & budget, std::vector<int32_t> & rows,
                std::string * error = nullptr) const;
    int64_t size() const { return keys_.size() / index_dimension; }
    const std::vector<bucket> & buckets() const { return buckets_; }
    // Author oracle uses an unstable argsort for final-round merge truncation.
    // Internal ties use lower kept pool, but ambiguous visible clusters refuse
    // selection so the graph can use dense attention until oracle parity exists.
    size_t ambiguous_tie_clusters() const;
private:
    bool initialized_ = false;
    uint64_t epoch_ = 0;
    std::string map_identity_;
    std::vector<float> keys_;
    std::vector<bucket> buckets_;
    std::vector<uint8_t> ambiguous_;
};

struct index_set {
    const int32_t * indices = nullptr; // [queries, capacity], -1 is padding
    const int32_t * lengths = nullptr; // null means capacity
    int64_t capacity = 0;
};

// DSA-IDX.1 slot order: main, then extra; duplicates remain separate slots.
// Physical Q/K must share the attention coordinate system (including rotations).
// This API never derives index projections from physical attention tensors.
struct attention_args {
    tensor_view q, k, v;
    index_set main, extra;
    const int32_t * positions = nullptr; // required causal upper bound per query
    const float * attn_sink = nullptr;   // optional denominator-only logit per head
    float scale = 1.0f;
    float softcap = 0.0f;
    float * output = nullptr;           // [q.rows, q.heads, v.dim], F32
};

struct attention_stats {
    uint64_t key_rows = 0;
    uint64_t value_rows = 0;
};

// Two-pass FP32 softmax, touching K/V only for valid selected slots. Uses the
// existing CPU dot/dequant traits for F32/F16/BF16 and quantized KV (including Q8).
// Structural errors return false before any KV access. Nonfinite selected data
// fail after selected reads; every failure leaves output untouched.
bool selected_attention_cpu(const attention_args & args, attention_stats * stats = nullptr,
                            std::string * error = nullptr);

bool cpu_has_avx512();

} // namespace llama_lsa
