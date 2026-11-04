#pragma once

#include "llama-lsa-runtime.h"
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_cparams;
struct llama_hparams;
struct llama_ubatch;
class llama_memory_context_i;
struct ggml_tensor;
struct ggml_cgraph;

// Qwen35 text broadcasts one causal position across four MRoPE axes.
// Reject image axes and embedding batches before sparse graph admission.
bool llama_lsa_text_positions(const llama_ubatch & ubatch);

class llama_lsa_graph_state {
public:
    llama_lsa_graph_state(const llama_model & model, const llama_cparams & cparams);
    ~llama_lsa_graph_state();
    bool admitted() const { return selector != nullptr; }
    bool using_graph() const { return active; }
    bool prepare(const llama_ubatch & ubatch, const llama_memory_context_i * mctx,
                 const llama_cparams & cparams, bool main_decoder, bool adapters_compatible);
    void invalidate();
    bool layer_enabled(int layer) const;
    void record_graph(ggml_cgraph * graph);
    static bool dispatch(void * userdata, const ggml_tensor * op, void * stream, bool device) noexcept;

private:
    bool execute(const ggml_tensor * op, void * stream, bool device);
    bool trace_failure(int layer, const char * stage, const char * reason, const std::string & detail = {});
    struct failure_info {
        std::string stage;
        std::string reason;
        std::string detail;
    };
    llama_lsa::map_bundle maps;
    std::unique_ptr<llama_lsa::runtime> selector;
    std::string model_digest;
    std::string map_digest;
    std::map<int, bool> native_success;
    std::map<int, bool> used_device;
    std::map<int, uint64_t> selected_key_rows;
    std::map<int, uint64_t> selected_value_rows;
    std::map<int, failure_info> failures;
    FILE * trace_file = nullptr;
    std::vector<int32_t> positions;
    std::vector<int32_t> slots;
    uint64_t epoch = 0;
    uint64_t cache_generation = 0;
    int32_t sequence = -1;
    int64_t first = 0;
    bool active = false;
    bool begun = false;
    bool native_failed = false;
};
