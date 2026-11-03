#pragma once

#include "llama-lsa-native.h"

// This header is also usable by a CPU-only dispatcher. No HIP header, device
// initialization, or link dependency is introduced until the HIP backend is built.
namespace llama_lsa {

struct hip_context;

struct hip_limits {
    int64_t max_keys = 0;
    int64_t max_queries = 0;
    int64_t max_slots = 0; // main + extra; bounds the exact-attention workspace
    size_t max_device_bytes = 0; // optional admission cap; zero means no extra cap
};

struct hip_memory {
    size_t persistent_bytes = 0;
    size_t workspace_bytes = 0;
    size_t total_device_bytes = 0;
};

// Pure host arithmetic, also available from a HIP build before device creation.
// Persistent: 2*map_rows*64*2 + max_keys*64*4 + floor(max_keys/512)*(64*sizeof(bucket)+1).
// Workspace: max_queries*(floor(max_keys/512)*64 + q_heads*max_slots)*4 + sizeof(int)+sizeof(attention_stats).
// Use query microbatches (e.g. 32), not a full model ubatch, for max_queries.
// K append capacity is independent of this query microbatch bound.

enum class hip_result {
    success,
    invalid_argument,
    unsupported,
    resource_error,
    device_error,
    index_error,
};

hip_result hip_memory_requirements(const geometry & shape, const hip_limits & limits,
                                   hip_memory * bytes, std::string * error = nullptr);

// All tensor/index/output pointers below are device pointers on the current HIP
// device; maps and strings are host objects. stream is a hipStream_t cast to void*,
// null means the default stream. The caller owns the stream and serializes this
// request/layer context. Operations synchronize that stream before returning:
// success means device-side validation completed, not merely launch acceptance.
// On failure the dispatcher must ignore sparse output and explicitly choose its
// dense path. No implicit dense allocation, mask, selection, or KV eviction occurs.
hip_result hip_create(const projection_maps & maps, const hip_limits & limits,
                      void * stream, hip_context ** context, std::string * error = nullptr);
void hip_destroy(hip_context * context);
hip_result hip_reset(hip_context * context, uint64_t epoch, const std::string & map_identity,
                     std::string * error = nullptr);
hip_result hip_append(hip_context * context, uint64_t epoch, int64_t first,
                      const tensor_view & logical_k, std::string * error = nullptr);
hip_result hip_truncate(hip_context * context, uint64_t epoch, int64_t count,
                        std::string * error = nullptr);
hip_result hip_project_queries(hip_context * context, const tensor_view & logical_q,
                               float * output, std::string * error = nullptr);
hip_result hip_select(hip_context * context, uint64_t epoch, const float * index_queries,
                     int64_t query_count, const int32_t * positions, const budgets & budget,
                     int32_t * indices, int32_t * lengths, int64_t capacity,
                     std::string * error = nullptr);
hip_result hip_selected_attention(hip_context * context, const attention_args & args,
                                 attention_stats * host_stats = nullptr, std::string * error = nullptr);
int64_t hip_size(const hip_context * context);

// Read-only device debug views, valid until destroy; only the first size() keys
// and floor(size()/512)*64 buckets are initialized. Used by parity fixtures.
const float * hip_index_keys(const hip_context * context);
const bucket * hip_index_buckets(const hip_context * context);
int64_t hip_ambiguous_tie_clusters(const hip_context * context);

} // namespace llama_lsa
