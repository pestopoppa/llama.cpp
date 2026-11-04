#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-batch.h"
#include "llama-lsa-graph.h"
#include "llama-lsa-native.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

static bool witness(const char * ready_name, const char * release_name) {
    const char * ready = std::getenv(ready_name);
    const char * release = std::getenv(release_name);
    if (!ready && !release) return true;
    if (!ready || !release) return false;
    { std::ofstream out(ready, std::ios::trunc); out << getpid() << '\n'; if (!out) return false; }
    for (int i = 0; i < 12000; ++i) {
        std::ifstream in(release);
        if (in.good()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

static void timeline(const char * phase) {
    const char * path = std::getenv("LSA_HIP_TIMELINE");
    if (!path) return;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::ofstream out(path, std::ios::app);
    out << phase << ' ' << ns << '\n';
}

static bool decline(void * userdata, const ggml_tensor *, void *, bool) {
    ++*static_cast<int *>(userdata);
    return false;
}

struct selected_probe {
    int calls = 0;
    int publications = 0;
    llama_lsa::attention_stats stats;
};

// C5 Q8 flash and selected CPU attention quantize Q with the Q8 dot trait.
// The scalar oracle dequantizes those exact Q8 bytes before computing logits.
// Only FP32 reduction order and exp rounding differ from the kernel path.
static constexpr float selected_q8_scalar_tolerance = 1e-5f;

static bool selected_q8(void * userdata, const ggml_tensor * op, void *, bool device) {
    auto & probe = *static_cast<selected_probe *>(userdata);
    ++probe.calls;
    if (device || !op || !op->data) return false;
    const int32_t indices[] = { 0, 1, 0, 3 };
    const int32_t lengths[] = { 2, 2 };
    const int32_t positions[] = { 1, 3 };
    const auto * q = op->src[0];
    const auto * k = op->src[1];
    const auto * v = op->src[2];
    llama_lsa::attention_args args;
    args.q = { q->data, q->type, q->ne[1], int(q->ne[2]), int(q->ne[0]), q->nb[1], q->nb[2] };
    args.k = { k->data, k->type, k->ne[1], int(k->ne[2]), int(k->ne[0]), k->nb[1], k->nb[2] };
    args.v = { v->data, v->type, v->ne[1], int(v->ne[2]), int(v->ne[0]), v->nb[1], v->nb[2] };
    args.main = { indices, lengths, 2 };
    args.positions = positions;
    args.scale = 1.0f / 32.0f;
    args.output = static_cast<float *>(op->data);
    const bool ok = llama_lsa::selected_attention_cpu(args, &probe.stats);
    if (ok) ++probe.publications;
    return ok;
}

static bool contains_op(ggml_cgraph * graph, ggml_op op) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        if (ggml_graph_node(graph, i)->op == op) return true;
    return false;
}

int main(int argc, char ** argv) {
    std::string backend_name, case_name, fixture;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--backend") == 0) backend_name = argv[i + 1];
        else if (std::strcmp(argv[i], "--case") == 0) case_name = argv[i + 1];
        else if (std::strcmp(argv[i], "--fixture") == 0) fixture = argv[i + 1];
        else return 2;
    }
    if ((backend_name != "cpu" && backend_name != "hip") || fixture.empty() ||
        (case_name != "disabled-default" && case_name != "late-fallback" &&
         case_name != "late-fallback-q8" && case_name != "selected-success-q8" &&
         case_name != "mrope-text-admission")) return 2;
    if (case_name == "mrope-text-admission" && backend_name != "cpu") return 2;
    const bool late_fallback = case_name != "disabled-default";
    const bool selected_success = case_name == "selected-success-q8";
    const bool q8 = case_name == "late-fallback-q8" || selected_success;
    if (backend_name == "hip" && selected_success) return 2;
    const int dim = backend_name == "hip" ? 256 : 32;
    const int kv_rows = backend_name == "hip" ? 256 : 4;
    const int q_rows = selected_success ? 2 : 1;
    const int q_heads = selected_success || backend_name == "hip" ? 1 : 2;

    ggml_backend_t backend = nullptr;
    if (backend_name == "cpu") {
        backend = ggml_backend_cpu_init();
    } else {
        ggml_backend_load_all();
        ggml_backend_dev_t device = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * candidate = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(candidate) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            const char * name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(candidate));
            if (name && (std::strcmp(name, "ROCm") == 0 || std::strcmp(name, "HIP") == 0)) {
                device = candidate;
                break;
            }
        }
        if (!device) return 3;
        backend = ggml_backend_dev_init(device, nullptr);
    }
    if (!backend) return 3;
    if (backend_name == "cpu") ggml_backend_cpu_set_n_threads(backend, 4);
    if (!witness("LSA_GRAPH_WITNESS_READY", "LSA_GRAPH_WITNESS_RELEASE")) return 4;
    if (case_name == "mrope-text-admission") {
        llama_ubatch batch{};
        llama_token tokens[] = { 1, 2, 3 };
        llama_pos pos[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 };
        batch.n_tokens = 3;
        batch.n_pos = 4;
        batch.token = tokens;
        batch.pos = pos;
        const bool text_ok = llama_lsa_text_positions(batch);
        pos[7] = 9;
        const bool image_rejected = !llama_lsa_text_positions(batch);
        pos[7] = 1;
        float embedding = 0.0f;
        batch.embd = &embedding;
        const bool embedding_rejected = !llama_lsa_text_positions(batch);
        batch.embd = nullptr;
        batch.token = nullptr;
        const bool token_absent_rejected = !llama_lsa_text_positions(batch);
        const bool ok = text_ok && image_rejected && embedding_rejected && token_absent_rejected;
        std::printf("case=mrope-text-admission backend=cpu text_axes_accepted=%d image_axes_rejected=%d embedding_rejected=%d token_absent_rejected=%d\n",
                    text_ok, image_rejected, embedding_rejected, token_absent_rejected);
        ggml_backend_free(backend);
        return ok ? 0 : 9;
    }

    ggml_init_params params = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 5;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, q_rows, q_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16, dim, kv_rows, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16, dim, kv_rows, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv_rows, q_rows, 1, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, selected_success ? 2 : 1);
    ggml_tensor * slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, selected_success ? 2 : 1);
    ggml_tensor * dense = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / dim, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(dense, GGML_PREC_F32);
    int callbacks = 0;
    selected_probe probe;
    ggml_tensor * selected = late_fallback
        ? ggml_lsa_native(ctx, dense, q, k, positions, slots, 3,
                          selected_success ? selected_q8 : decline,
                          selected_success ? static_cast<void *>(&probe) : static_cast<void *>(&callbacks))
        : ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / dim, 0.0f, 0.0f);
    if (case_name == "disabled-default") ggml_flash_attn_ext_set_prec(selected, GGML_PREC_F32);
    ggml_cgraph * dense_graph = ggml_new_graph(ctx);
    ggml_cgraph * selected_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(dense_graph, dense);
    ggml_build_forward_expand(selected_graph, selected);
    const bool topology = contains_op(dense_graph, GGML_OP_FLASH_ATTN_EXT) &&
        !contains_op(dense_graph, GGML_OP_LSA_NATIVE) &&
        contains_op(selected_graph, late_fallback ? GGML_OP_LSA_NATIVE : GGML_OP_FLASH_ATTN_EXT);
    if (!topology) return 6;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 7;
    std::vector<float> q_data(size_t(dim)*q_rows*q_heads);
    std::vector<ggml_fp16_t> k_data(size_t(dim)*kv_rows), v_data(k_data.size());
    std::vector<float> k_f32(k_data.size()), v_f32(k_data.size());
    std::vector<ggml_fp16_t> mask_data(size_t(kv_rows)*q_rows);
    for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = float(int(i % 17) - 8) / 11.0f;
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_f32[i] = float(int(i % 13) - 6) / 9.0f;
        v_f32[i] = float(int(i % 19) - 9) / 7.0f;
        k_data[i] = ggml_fp32_to_fp16(k_f32[i]);
        v_data[i] = ggml_fp32_to_fp16(v_f32[i]);
    }
    for (auto & x : mask_data) x = ggml_fp32_to_fp16(0.0f);
    int32_t positions_data[] = { 1, 3 };
    int32_t slots_data[] = { 1, 3 };
    int32_t zero = 0;
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size()*sizeof(float));
    if (q8) {
        std::vector<uint8_t> k_q8(ggml_nbytes(k)), v_q8(ggml_nbytes(v));
        if (ggml_quantize_chunk(GGML_TYPE_Q8_0, k_f32.data(), k_q8.data(), 0, kv_rows, dim, nullptr) != k_q8.size() ||
            ggml_quantize_chunk(GGML_TYPE_Q8_0, v_f32.data(), v_q8.data(), 0, kv_rows, dim, nullptr) != v_q8.size()) return 7;
        ggml_backend_tensor_set(k, k_q8.data(), 0, k_q8.size());
        ggml_backend_tensor_set(v, v_q8.data(), 0, v_q8.size());
    } else {
        ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size()*sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size()*sizeof(ggml_fp16_t));
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(positions, selected_success ? static_cast<void *>(positions_data) : static_cast<void *>(&zero), 0, ggml_nbytes(positions));
    ggml_backend_tensor_set(slots, selected_success ? static_cast<void *>(slots_data) : static_cast<void *>(&zero), 0, ggml_nbytes(slots));
    if (backend_name == "hip" && !witness("LSA_HIP_WITNESS_READY", "LSA_HIP_WITNESS_RELEASE")) return 4;
    if (backend_name == "hip") timeline("first_launch");
    if (ggml_backend_graph_compute(backend, dense_graph) != GGML_STATUS_SUCCESS ||
        ggml_backend_graph_compute(backend, selected_graph) != GGML_STATUS_SUCCESS) return 8;
    ggml_backend_synchronize(backend);
    if (backend_name == "hip") timeline("last_finish");
    std::vector<unsigned char> dense_bytes(ggml_nbytes(dense));
    std::vector<unsigned char> selected_bytes(ggml_nbytes(selected));
    ggml_backend_tensor_get(dense, dense_bytes.data(), 0, dense_bytes.size());
    ggml_backend_tensor_get(selected, selected_bytes.data(), 0, selected_bytes.size());
    bool equal = dense_bytes == selected_bytes;
    if (selected_success) {
        equal = probe.calls == 1 && probe.publications == 1 && probe.stats.key_rows == 4 && probe.stats.value_rows == 4;
        std::array<float, 128> decoded_k{}, decoded_v{};
        for (int row = 0; row < 4; ++row) {
            const size_t offset = size_t(row) * ggml_row_size(GGML_TYPE_Q8_0, 32);
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(static_cast<const char *>(k->data) + offset, decoded_k.data() + row*32, 32);
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(static_cast<const char *>(v->data) + offset, decoded_v.data() + row*32, 32);
        }
        const int rows[2][2] = {{0,1},{0,3}};
        for (int query = 0; query < 2; ++query) {
            std::vector<uint8_t> quantized_q(ggml_row_size(GGML_TYPE_Q8_0, 32));
            float decoded_q[32];
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(q_data.data()+query*32, quantized_q.data(), 32);
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(quantized_q.data(), decoded_q, 32);
            float scores[2];
            for (int j = 0; j < 2; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < 32; ++d) dot += decoded_q[d] * decoded_k[rows[query][j]*32+d];
                scores[j] = dot / 32.0f;
            }
            const float maximum = std::fmax(scores[0], scores[1]);
            const float w0 = std::exp(scores[0]-maximum);
            const float w1 = std::exp(scores[1]-maximum);
            for (int d = 0; d < 32; ++d) {
                const float expected = (w0*decoded_v[rows[query][0]*32+d] + w1*decoded_v[rows[query][1]*32+d])/(w0+w1);
                const auto * actual = reinterpret_cast<const float *>(selected_bytes.data());
                equal = equal && std::fabs(actual[query*32+d] - expected) <=
                                 selected_q8_scalar_tolerance*(1.0f + std::fabs(expected));
            }
        }
    }
    const uint64_t fallback_count = late_fallback ? ggml_lsa_native_fallback_count(selected) : 0;
    const bool calls = selected_success ? (probe.calls == 1 && probe.publications == 1 && fallback_count == 0) :
        (callbacks == (late_fallback ? 1 : 0) && fallback_count == (late_fallback ? 1u : 0u));
    std::printf("case=%s backend=%s op=%s original_flash_calls=%d publication_count=%d callbacks=%d equal=%d\n",
            case_name.c_str(), backend_name.c_str(), late_fallback ? "LSA_NATIVE" : "FLASH_ATTN_EXT", int(1 + fallback_count),
            probe.publications, selected_success ? probe.calls : callbacks, equal ? 1 : 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return equal && calls ? 0 : 9;
}
