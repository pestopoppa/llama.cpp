#pragma once
#include "llama-lsa-runtime.h"
#include "llama-lsa-native-hip.h"
#include <map>
#include <mutex>

namespace llama_lsa {
struct runtime_hip_scratch;
void runtime_destroy_hip(hip_context * context);
void runtime_destroy_scratch(runtime_hip_scratch * scratch);
struct runtime_hip_deleter { void operator()(hip_context * c) const { runtime_destroy_hip(c); } };
struct runtime_scratch_deleter { void operator()(runtime_hip_scratch * s) const { runtime_destroy_scratch(s); } };
struct runtime_layer {
    index_cache cpu;
    std::unique_ptr<hip_context, runtime_hip_deleter> hip;
    bool poisoned = false;
    int64_t hip_rows = 0;
    size_t host_admission = 0, device_admission = 0;
};
struct runtime_request {
    uint64_t epoch = 0;
    runtime_backend backend = runtime_backend::cpu;
    bool blocked = false;
    std::map<int, runtime_layer> layers;
    std::unique_ptr<runtime_hip_scratch, runtime_scratch_deleter> hip_scratch;
    size_t scratch_admission = 0;
};
struct runtime::impl {
    map_bundle maps;
    runtime_options options;
    bool configured = false;
    mutable std::mutex mutex;
    uint64_t generation = 0;
    std::map<uint64_t,runtime_request> requests;
    std::vector<float> output;
    runtime_result last;
    cpu_attention_fn attention = nullptr;
    void * attention_userdata = nullptr;
    size_t map_bytes = 0, host_scratch_admission = 0;
    bool configure();
    size_t host_usage() const;
    size_t device_usage() const;
    runtime_result dense(dense_reason reason, const std::string & detail = "");
    runtime_result execute_cpu(const runtime_frame & frame, runtime_request & request,
                               runtime_layer & layer, const projection_maps & maps);
    runtime_result execute_hip(const runtime_frame & frame, runtime_request & request,
                               runtime_layer & layer, const projection_maps & maps);
    bool hip_truncate_layer(runtime_layer & layer, uint64_t epoch, int64_t count);
};
bool runtime_add(size_t a, size_t b, size_t & out);
bool runtime_mul(size_t a, size_t b, size_t & out);
bool runtime_view(const tensor_view & view, int64_t rows, int heads, int dim, bool floating_only);
tensor_view runtime_slice(tensor_view view, int64_t first, int64_t count);
// Test-only narrow seam: the synthetic HIP unit passes its actual selector
// buffers through the same capture/write helper used by execute_hip.
bool hip_capture_index_witness_for_test(const runtime_options & options, const map_bundle & maps,
                                        const runtime_frame & frame, hip_context * context,
                                        const float * queries, const int32_t * indices,
                                        const int32_t * lengths, void * stream,
                                        int64_t capacity, std::string & error);
} // namespace llama_lsa
