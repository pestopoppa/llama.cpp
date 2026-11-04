#include "llama-lsa-graph.h"

#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIP
#include <hip/hip_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>

namespace {
bool parse_u64(const std::string & text, uint64_t & value) {
    if (text.empty() || text[0] == '-') return false;
    char * end = nullptr;
    errno = 0;
    const unsigned long long result = std::strtoull(text.c_str(), &end, 10);
    if (errno || !end || *end || end == text.c_str()) return false;
    value = uint64_t(result);
    return true;
}

bool parse_layers(const std::string & text, std::vector<int> & layers) {
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find(',', start);
        const std::string part = text.substr(start, end == std::string::npos ? end : end-start);
        uint64_t value;
        if (!parse_u64(part, value) || value > INT_MAX) return false;
        layers.push_back(int(value));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return layers.size() == 16 && std::is_sorted(layers.begin(), layers.end()) &&
        std::adjacent_find(layers.begin(), layers.end()) == layers.end();
}

bool parse_manifest(const char * path, std::map<std::string, std::string> & fields) {
    if (!path || path[0] != '/') return false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string line;
    size_t size = 0;
    while (std::getline(in, line)) {
        size += line.size() + 1;
        if (size > 8192 || line.empty() || line.back() == '\r') return false;
        const size_t split = line.find('=');
        if (split == std::string::npos || split == 0 || split + 1 == line.size()) return false;
        if (!fields.emplace(line.substr(0, split), line.substr(split + 1)).second) return false;
    }
    static const std::array<const char *, 16> keys = {{
        "schema", "model_sha256", "map_path", "map_sha256", "source_sha256", "map_model_identity",
        "geometry", "global_layers", "max_keys", "max_new_tokens", "max_host_bytes", "max_device_bytes",
        "top_k", "top_p", "window", "sink" }};
    if (fields.size() != keys.size()) return false;
    for (const char * key : keys) if (!fields.count(key)) return false;
    return fields["schema"] == "lsa-native-manifest-v1" && fields["map_path"][0] == '/';
}

llama_lsa::tensor_view logical_view(const ggml_tensor * tensor) {
    return { tensor->data, tensor->type, tensor->ne[2], int(tensor->ne[1]), int(tensor->ne[0]),
             tensor->nb[2], tensor->nb[1] };
}

llama_lsa::tensor_view physical_view(const ggml_tensor * tensor) {
    return { tensor->data, tensor->type, tensor->ne[1], int(tensor->ne[2]), int(tensor->ne[0]),
             tensor->nb[1], tensor->nb[2] };
}

void trace_json_string(FILE * file, const std::string & value) {
    std::fputc('"', file);
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { std::fputc('\\', file); std::fputc(ch, file); }
        else if (ch < 0x20) { std::fprintf(file, "\\u%04x", unsigned(ch)); }
        else { std::fputc(ch, file); }
    }
    std::fputc('"', file);
}
}

bool llama_lsa_text_positions(const llama_ubatch & ubatch) {
    if (!ubatch.token || ubatch.embd || !ubatch.pos || ubatch.n_tokens == 0 ||
        (ubatch.n_pos != 1 && ubatch.n_pos != 4)) return false;
    for (uint32_t axis = 1; axis < ubatch.n_pos; ++axis) {
        for (uint32_t token = 0; token < ubatch.n_tokens; ++token) {
            if (ubatch.pos[size_t(axis)*ubatch.n_tokens + token] != ubatch.pos[token]) return false;
        }
    }
    return true;
}

llama_lsa_graph_state::llama_lsa_graph_state(const llama_model & model, const llama_cparams & cparams) {
    const char * trace_path = std::getenv("LLAMA_LSA_NATIVE_TRACE");
    if (trace_path && trace_path[0] == '/') {
        trace_file = std::fopen(trace_path, "a");
        if (trace_file) {
            const char * trace_mode = std::getenv("LLAMA_LSA_NATIVE_TRACE_MODE");
            trace_light = trace_mode && std::strcmp(trace_mode, "light") == 0;
            const char * phase_mode = std::getenv("LLAMA_LSA_NATIVE_PHASES");
            trace_phases = trace_light && phase_mode && std::strcmp(phase_mode, "1") == 0;
            std::fprintf(trace_file, "{\"schema\":\"lsa-native-graph-v1\",\"event\":\"admission_attempt\"}\n");
            std::fflush(trace_file);
        }
    }
    const char * path = std::getenv("LLAMA_LSA_NATIVE_MANIFEST");
    if (!path || !*path) return;
    if (model.arch != LLM_ARCH_QWEN35 || model.type != LLM_TYPE_27B ||
        model.hparams.n_layer() != 64 || model.lsa_source_path.empty() ||
        cparams.pipeline_parallel || !cparams.flash_attn || cparams.dca_chunk_size > 0) return;
    std::map<std::string, std::string> fields;
    if (!parse_manifest(path, fields) || fields["geometry"] != "24,4,256,64" ||
        model.hparams.n_head() != 24 || model.hparams.n_head_kv() != 4 ||
        model.hparams.n_embd_head_k() != 256 || model.hparams.n_rot() != 64) return;
    std::vector<int> layers, actual;
    if (!parse_layers(fields["global_layers"], layers)) return;
    for (int il = 0; il < 64; ++il) if (!model.hparams.is_recr(il)) actual.push_back(il);
    if (layers != actual) return;
    ggml_backend_dev_t attention_device = nullptr;
    for (int il : layers) {
        ggml_backend_dev_t dev = model.dev_layer(il);
        if (!dev || (attention_device && dev != attention_device)) return;
        attention_device = dev;
        const auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
#ifdef GGML_USE_HIP
        const char * name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU && name &&
            (std::strcmp(name, "ROCm") == 0 || std::strcmp(name, "HIP") == 0)) continue;
#endif
        return;
    }
    if (!llama_lsa::sha256_file(model.lsa_source_path, model_digest) ||
        !llama_lsa::sha256_file(fields["map_path"], map_digest) ||
        model_digest != fields["model_sha256"] || map_digest != fields["map_sha256"]) return;
    llama_lsa::map_bundle loaded;
    const llama_lsa::geometry shape{24,4,256,64};
    if (!llama_lsa::load_map_bundle(fields["map_path"], fields["map_model_identity"], shape,
                                   layers, fields["source_sha256"], loaded)) return;
    llama_lsa::runtime_options options;
    options.enabled = true;
    uint64_t value;
    auto get = [&](const char * key, uint64_t maximum, uint64_t & target) {
        return parse_u64(fields[key], value) && value > 0 && value <= maximum && (target = value, true);
    };
    uint64_t max_keys, max_new_tokens, max_host_bytes, max_device_bytes, top_k, top_p, window, sink;
    if (!get("max_keys", INT32_MAX-1, max_keys) || !get("max_new_tokens", INT32_MAX-1, max_new_tokens) ||
        !get("max_host_bytes", SIZE_MAX, max_host_bytes) || !get("max_device_bytes", SIZE_MAX, max_device_bytes) ||
        !get("top_k", INT_MAX, top_k) || !get("top_p", INT_MAX, top_p) ||
        !get("window", INT_MAX, window) || !get("sink", INT_MAX, sink)) return;
    options.max_keys = int64_t(max_keys);
    options.max_new_tokens = int64_t(max_new_tokens);
    options.max_requests = 1;
    options.collect_stats = trace_file != nullptr && !trace_light;
    options.collect_phase_times = trace_phases;
    options.max_host_bytes = size_t(max_host_bytes);
    options.max_device_bytes = size_t(max_device_bytes);
    options.selection.top_k = int(top_k);
    options.selection.top_p = int(top_p);
    options.selection.window = int(window);
    options.selection.sink = int(sink);
    maps = loaded;
    auto candidate = std::make_unique<llama_lsa::runtime>(std::move(loaded), options);
    if (!candidate->valid()) return;
    selector = std::move(candidate);
    if (trace_file) {
        std::fprintf(trace_file, "{\"schema\":\"lsa-native-graph-v1\",\"event\":\"admitted\",\"model_sha256\":\"%s\",\"map_sha256\":\"%s\",\"cpu_projection_profile\":\"%s\"}\n",
                     model_digest.c_str(), map_digest.c_str(), llama_lsa::cpu_projection_profile());
        std::fflush(trace_file);
    }
}

llama_lsa_graph_state::~llama_lsa_graph_state() {
    if (trace_file) {
        if (!selector) std::fprintf(trace_file, "{\"schema\":\"lsa-native-graph-v1\",\"event\":\"admission_refused\"}\n");
        std::fclose(trace_file);
    }
}

bool llama_lsa_graph_state::prepare(const llama_ubatch & ubatch, const llama_memory_context_i * mctx,
                                    const llama_cparams & cparams, bool main_decoder, bool adapters_compatible) {
    active = false;
    // A dense-only ubatch breaks the logical projection history. Do not resume
    // an old index on a later eligible token from the same sequence.
    auto decline = [&]() {
        if (begun) invalidate();
        return false;
    };
    if (!selector || !main_decoder || !adapters_compatible || !cparams.causal_attn || cparams.dca_chunk_size > 0 ||
        cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT ||
        cparams.pipeline_parallel || ubatch.n_tokens == 0 || ubatch.n_seqs_unq != 1 ||
        !llama_lsa_text_positions(ubatch)) return decline();
    const auto * hybrid = dynamic_cast<const llama_memory_hybrid_context *>(mctx);
    const auto * cache = hybrid ? hybrid->get_attn() : dynamic_cast<const llama_kv_cache_context *>(mctx);
    if (!cache || cache->type_k() != GGML_TYPE_Q8_0 || cache->type_v() != GGML_TYPE_Q8_0) return decline();
    const llama_seq_id seq = ubatch.seq_id[0][0];
    if (seq < 0 || ubatch.n_seq_id[0] != 1) return decline();
    const int64_t start = ubatch.pos[0];
    if (start < 0 || start + ubatch.n_tokens > INT32_MAX) return decline();
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != seq || ubatch.pos[i] != start + i) return decline();
    }
    std::vector<int32_t> next_slots;
    if (!cache->lsa_physical_slots(next_slots) || next_slots.size() != ubatch.n_tokens ||
        !cache->lsa_exact_prefix(seq, start + ubatch.n_tokens)) return decline();
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) if (next_slots[i] != start + i) return decline();
    const auto change = cache->lsa_last_mutation();
    if (start == 0) {
        ++epoch;
        begun = false;
    } else {
        if (!begun || sequence != seq) return decline();
        if (change.generation != cache_generation) {
            if (change.generation != cache_generation + 1 ||
                change.kind != llama_kv_cache::lsa_mutation_kind::suffix_remove ||
                change.seq_id != seq || change.from != start ||
                !selector->truncate(0, epoch, start)) {
                invalidate();
                return false;
            }
        }
    }
    positions.resize(ubatch.n_tokens);
    native_success.clear();
    used_device.clear();
    selected_key_rows.clear();
    selected_value_rows.clear();
    if (trace_file) failures.clear();
    native_failed = false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) positions[i] = int32_t(start + i);
    slots = std::move(next_slots);
    first = start;
    sequence = seq;
    cache_generation = change.generation;
    active = true;
    return true;
}

void llama_lsa_graph_state::invalidate() {
    active = false;
    begun = false;
    if (selector) selector->invalidate(0, llama_lsa::invalidation::shift);
}

bool llama_lsa_graph_state::layer_enabled(int layer) const {
    return active && maps.layers.count(layer) != 0;
}

void llama_lsa_graph_state::record_graph(ggml_cgraph * graph) {
    if (!active || !graph) return;
    if (trace_file) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * op = ggml_graph_node(graph, i);
        if (op->op != GGML_OP_LSA_NATIVE) continue;
        int32_t layer;
        std::memcpy(&layer, (const char *) op->op_params + 5*sizeof(int32_t), sizeof(layer));
        const uint64_t fallback = ggml_lsa_native_fallback_count(op);
        uint64_t dense_visible_rows = 0;
        for (int32_t pos : positions) dense_visible_rows += uint64_t(pos + 1) * 24;
        const std::string key_rows = trace_light ? "null" : std::to_string(selected_key_rows[layer]);
        const std::string value_rows = trace_light ? "null" : std::to_string(selected_value_rows[layer]);
        std::fprintf(trace_file,
            "{\"schema\":\"lsa-native-graph-v1\",\"event\":\"layer_complete\",\"sequence\":%d,\"slot\":0,\"epoch\":%llu,\"first\":%lld,\"count\":%zu,\"layer\":%d,\"backend\":\"%s\",\"native_success\":%d,\"original_flash_fallback_count\":%llu,\"selected_key_rows\":%s,\"selected_value_rows\":%s,\"dense_visible_rows\":%llu,\"model_sha256\":\"%s\",\"map_sha256\":\"%s\"",
            sequence, (unsigned long long) epoch, (long long) first, positions.size(), layer,
            used_device[layer] ? "hip" : "cpu", native_success[layer] ? 1 : 0,
            (unsigned long long) fallback,
            key_rows.c_str(), value_rows.c_str(),
            (unsigned long long) dense_visible_rows, model_digest.c_str(), map_digest.c_str());
        if (trace_light) std::fputs(",\"stats_collected\":false", trace_file);
        const auto failure = failures.find(layer);
        if (failure != failures.end()) {
            std::fputs(",\"failure_stage\":", trace_file); trace_json_string(trace_file, failure->second.stage);
            std::fputs(",\"failure_reason\":", trace_file); trace_json_string(trace_file, failure->second.reason);
            std::fputs(",\"failure_detail\":", trace_file); trace_json_string(trace_file, failure->second.detail);
        }
        std::fputs("}\n", trace_file);
    }
    std::fflush(trace_file);
    }
    if (native_failed) invalidate();
}

bool llama_lsa_graph_state::trace_failure(int layer, const char * stage, const char * reason, const std::string & detail) {
    if (trace_file) failures[layer] = {stage, reason, detail};
    return false;
}

bool llama_lsa_graph_state::dispatch(void * userdata, const ggml_tensor * op, void * stream, bool device) noexcept {
    auto * state = static_cast<llama_lsa_graph_state *>(userdata);
    try {
        if (!state) return false;
        const bool success = state->execute(op, stream, device);
        if (!success) state->native_failed = true;
        return success;
    } catch (...) {
        if (state) {
            state->native_failed = true;
            if (state->trace_file && op && op->op == GGML_OP_LSA_NATIVE) {
                try {
                    int32_t layer;
                    std::memcpy(&layer, (const char *) op->op_params + 5*sizeof(int32_t), sizeof(layer));
                    state->trace_failure(layer, "dispatch", "exception");
                } catch (...) {}
            }
        }
        return false;
    }
}

bool llama_lsa_graph_state::execute(const ggml_tensor * op, void * stream, bool device) {
    if (!op || op->op != GGML_OP_LSA_NATIVE) return false;
    int32_t layer;
    std::memcpy(&layer, (const char *) op->op_params + 5*sizeof(int32_t), sizeof(layer));
    used_device[layer] = device;
    native_success[layer] = false;
    selected_key_rows[layer] = 0;
    selected_value_rows[layer] = 0;
    if (native_failed) return trace_failure(layer, "dispatch", "cascaded_after_first_failure");
    if (!active || !selector) return trace_failure(layer, "dispatch", "inactive_or_unadmitted");
    auto it = maps.layers.find(layer);
    if (it == maps.layers.end() || !op->data || !op->src[0] || !op->src[1] || !op->src[2] ||
        !op->src[4] || !op->src[5]) return trace_failure(layer, "graph_binding", "missing_map_or_tensor");
    float scale, max_bias, softcap;
    std::memcpy(&scale, op->op_params, sizeof(scale));
    std::memcpy(&max_bias, (const char *) op->op_params + 4, sizeof(max_bias));
    std::memcpy(&softcap, (const char *) op->op_params + 8, sizeof(softcap));
    if (max_bias != 0.0f || op->src[1]->type != GGML_TYPE_Q8_0 ||
        op->src[2]->type != GGML_TYPE_Q8_0 || op->src[0]->type != GGML_TYPE_F32)
        return trace_failure(layer, "flash_contract", "unsupported_flash_parameters_or_types");
    const auto backend = device ? llama_lsa::runtime_backend::hip : llama_lsa::runtime_backend::cpu;
#ifndef GGML_USE_HIP
    if (device) return trace_failure(layer, "backend", "hip_not_built");
#endif
    if (!begun) {
        if (first != 0 || !selector->begin_request(0, epoch, backend))
            return trace_failure(layer, "request", "begin_request_failed");
        begun = true;
    }
    llama_lsa::runtime_frame frame;
    frame.slot = 0;
    frame.epoch = epoch;
    frame.layer = layer;
    frame.map_identity = it->second.identity;
    frame.first = first;
    frame.positions = positions.data();
    frame.slots = slots.data();
    frame.logical_q = logical_view(op->src[4]);
    frame.logical_k = logical_view(op->src[5]);
    frame.physical_q = physical_view(op->src[0]);
    frame.full_k = physical_view(op->src[1]);
    frame.full_v = physical_view(op->src[2]);
    frame.scale = scale;
    frame.softcap = softcap;
    frame.stream = stream;
    auto result = selector->execute(frame);
    if (trace_phases && result.sparse) {
        std::fprintf(trace_file,
            "{\"schema\":\"lsa-native-graph-v1\",\"event\":\"phase_host\",\"timing_kind\":\"host_elapsed\",\"sequence\":%d,\"slot\":0,\"epoch\":%llu,\"first\":%lld,\"count\":%zu,\"layer\":%d,\"backend\":\"%s\",\"key_append_host_us\":%llu,\"query_select_host_us\":%llu,\"attention_host_us\":%llu}\n",
            sequence, (unsigned long long) epoch, (long long) first, positions.size(), layer, device ? "hip" : "cpu",
            (unsigned long long) result.key_append_host_us,
            (unsigned long long) result.query_select_host_us,
            (unsigned long long) result.attention_host_us);
    }
    if (!result.sparse)
        return trace_failure(layer, "runtime_execute", llama_lsa::dense_reason_name(result.reason), result.detail);
    if (!selector->lease_valid(result)) return trace_failure(layer, "lease", "invalid_staging_lease");
    selected_key_rows[layer] = result.key_rows;
    selected_value_rows[layer] = result.value_rows;
    const size_t elements = size_t(result.rows) * size_t(result.heads) * size_t(result.dim);
    if (elements != ggml_nelements(op)) return trace_failure(layer, "publication", "output_shape_mismatch");
    if (!device) {
        native_success[layer] = selector->publish_cpu(result, static_cast<float *>(op->data), elements);
        if (!native_success[layer]) return trace_failure(layer, "publication", "cpu_publish_failed");
        return native_success[layer];
    }
#ifdef GGML_USE_HIP
    const hipError_t copy = hipMemcpyAsync(op->data, result.staged_output, elements*sizeof(float),
                                          hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream));
    native_success[layer] = copy == hipSuccess;
    if (!native_success[layer]) return trace_failure(layer, "publication", "hip_copy_failed", hipGetErrorString(copy));
    return native_success[layer];
#else
    return false;
#endif
}
