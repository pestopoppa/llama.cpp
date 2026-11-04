#pragma once
#include "llama-lsa-map.h"
#include <memory>

namespace llama_lsa {

enum class runtime_backend { cpu, hip };
enum class dense_reason {
    none, feature_disabled, invalid_configuration, unknown_request, stale_epoch,
    unsupported_request, unsupported_layout, missing_layer, missing_history,
    noncontiguous_slots, capacity, resource, projection, ambiguous_selection,
    attention, device, invalidated
};
enum class invalidation { cache_clear, slot_reuse, sequence_copy, state_restore, shift, map_replacement };
struct runtime_options {
    bool enabled = false;
    int64_t max_keys = 0;
    int64_t max_new_tokens = 0;
    size_t max_requests = 4;
    int query_chunk = 32; // must be in [1,32]
    budgets selection;
    size_t max_host_bytes = 0;   // admission for owned maps, CPU caches and scratch
    size_t max_device_bytes = 0; // aggregate across all live request/layer contexts
    bool collect_stats = false; // trace-only; HIP counters synchronize the stream
    bool collect_phase_times = false; // host call intervals only; no new HIP synchronization
    std::string index_witness_path; // opt-in HIP diagnostic; absolute output path, one layer-3/position-5131 record
    std::string witness_model_sha256; // copied from the admitted, hashed graph manifest
    std::string witness_map_sha256;
};
struct runtime_frame {
    uint64_t slot = 0, epoch = 0;
    int layer = -1;
    std::string map_identity; // exact per-layer identity bound by loader/graph
    // Caller proves old cache cells form the sole sequence's slots/positions0..first-1.
    // New physical slots and causal positions below are checked individually.
    bool single_sequence = true, causal_text = true, contiguous_prefix = true;
    bool dca = false, speculative = false, mtp = false;
    int64_t first = 0;
    const int32_t * positions = nullptr; // HOST [logical_q.rows], position=first+i
    const int32_t * slots = nullptr;     // HOST [logical_q.rows], physicalslot=first+i
    // Logical normalized inputs before RoPE/Hk rotations; ONLY new tokens.
    tensor_view logical_q, logical_k;
    // Physical attention coordinate system, including Hk/Hv rotations. Full KV
    // storage is retained; no projection is reconstructed from quantized cache.
    tensor_view physical_q, full_k, full_v;
    float scale = 1.0f, softcap = 0.0f;
    void * stream = nullptr; // HIP backend stream; ignored by CPU
};
struct runtime_result {
    bool sparse = false; // false means caller MUST invoke ORIGINAL C5 dense path
    dense_reason reason = dense_reason::invalidated;
    runtime_backend backend = runtime_backend::cpu;
    const float * staged_output = nullptr; // host/device according to backend
    int64_t rows = 0;
    int heads = 0, dim = 0;
    uint64_t generation = 0;
    const void * owner = nullptr;
    int device = -1;
    void * stream = nullptr;
    uint64_t key_rows = 0;   // physical selected K rows read across all query heads
    uint64_t value_rows = 0; // physical selected V rows read across all query heads
    uint64_t key_append_host_us = 0;
    uint64_t query_select_host_us = 0;
    uint64_t attention_host_us = 0;
    std::string detail;
};

// No production callers. Serializes mutations. A successful output lease is
// valid only until the next runtime mutation; publish immediately in the dispatcher.
// Failures never expose partial staging and never write graph output. HIP
// dispatch copies a successful lease on its backend stream; CPU may use publish.
class runtime {
public:
    runtime(map_bundle maps, runtime_options options);
    ~runtime();
    runtime(const runtime &) = delete;
    runtime & operator=(const runtime &) = delete;
    bool valid() const;
    bool begin_request(uint64_t slot, uint64_t epoch, runtime_backend backend);
    void invalidate(uint64_t slot, invalidation cause);
    bool truncate(uint64_t slot, uint64_t epoch, int64_t count);
    bool replace_maps(map_bundle maps);
    runtime_result execute(const runtime_frame & frame);
    bool lease_valid(const runtime_result & result) const;
    bool publish_cpu(const runtime_result & result, float * destination, size_t elements);
    int64_t indexed_rows(uint64_t slot, int layer) const;
    size_t admitted_device_bytes() const;

    // Optional dependency injection for transactional failure tests. The default
    // is selected_attention_cpu; hook still receives real projected/selected rows.
    using cpu_attention_fn = bool (*)(const attention_args &, attention_stats *, std::string *, void *);
    void set_cpu_attention_for_test(cpu_attention_fn fn, void * userdata);
    struct impl;
private:
    std::unique_ptr<impl> p;
};
const char * dense_reason_name(dense_reason reason);
} // namespace llama_lsa
