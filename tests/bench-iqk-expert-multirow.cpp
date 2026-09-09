#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

uint32_t random_u32(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float value(size_t i, uint32_t & state) {
    static const float scales[] = { 0.000244140625f, 0.03125f, 1.0f, 8.0f };
    return ((int32_t) random_u32(state) / 2147483648.0f) * scales[(i / 257) % 4];
}

uint64_t hash_bytes(const void * data, size_t size) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void set_env(const char * name, const char * text) {
#if defined(_WIN32)
    _putenv_s(name, text);
#else
    setenv(name, text, 1);
#endif
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: %s q4|q5 rows[1..4] [repeats=200] [threads=48]\n", argv[0]);
        return 2;
    }

    const ggml_type type = std::strcmp(argv[1], "q4") == 0 ? GGML_TYPE_Q4_K :
                           std::strcmp(argv[1], "q5") == 0 ? GGML_TYPE_Q5_K : GGML_TYPE_COUNT;
    const int rows    = std::atoi(argv[2]);
    const int repeats = argc >= 4 ? std::atoi(argv[3]) : 200;
    const int threads = argc >= 5 ? std::atoi(argv[4]) : 48;
    if (type == GGML_TYPE_COUNT || rows < 1 || rows > 4 || repeats < 1 || threads < 1) return 2;

    set_env("GGML_IQK", "1");
    set_env("GGML_ROWEXACT_N", "16");

    const int64_t k = type == GGML_TYPE_Q4_K ? 4096 : 2048;
    const int64_t m = type == GGML_TYPE_Q4_K ? 2048 : 4096;

    ggml_init_params params = { ggml_tensor_overhead() * 12 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) return 3;

    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), type, k, m, 1);
    ggml_tensor * input   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, 1, rows);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, rows);
    ggml_tensor * output  = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_cgraph * graph   = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    ggml_backend_cpu_set_rowexact(backend.get(), true);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) return 3;

    std::vector<float> weights_f((size_t) k * m);
    uint32_t state = type == GGML_TYPE_Q4_K ? 0x31415926u : 0x27182818u;
    for (size_t i = 0; i < weights_f.size(); ++i) weights_f[i] = value(i, state);
    std::vector<uint8_t> weights_q(ggml_row_size(type, k) * (size_t) m);
    if (ggml_quantize_chunk(type, weights_f.data(), weights_q.data(), 0, m, k, nullptr) != weights_q.size()) return 3;
    ggml_backend_tensor_set(weights, weights_q.data(), 0, weights_q.size());

    std::vector<float> input_f((size_t) k * rows);
    for (size_t i = 0; i < input_f.size(); ++i) input_f[i] = value(i + 31, state);
    ggml_backend_tensor_set(input, input_f.data(), 0, input_f.size() * sizeof(float));
    std::vector<int32_t> routes(rows, 0);
    ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));

    ggml_backend_graph_plan_t plan = ggml_backend_graph_plan_create(backend.get(), graph);
    if (!plan) return 3;
    for (int i = 0; i < 10; ++i) {
        if (ggml_backend_graph_plan_compute(backend.get(), plan) != GGML_STATUS_SUCCESS) return 3;
    }

    std::vector<double> elapsed;
    elapsed.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_plan_compute(backend.get(), plan) != GGML_STATUS_SUCCESS) return 3;
        const auto end = std::chrono::steady_clock::now();
        elapsed.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::vector<float> result((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_graph_plan_free(backend.get(), plan);

    auto sorted = elapsed;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    const char * knob = std::getenv("GGML_IQK_EXPERT_MULTIROW");
    const bool enabled = knob != nullptr && std::atoi(knob) != 0;
    const char * round_text = std::getenv("GLM53_EXPERT_BENCH_ROUND");
    const int round = round_text != nullptr ? std::atoi(round_text) : 0;
    std::printf("{\"schema\":\"epyc.iqk_expert_multirow_micro.v1\",\"type\":\"%s\",\"rows\":%d,"
                "\"arm\":\"%s\",\"round\":%d,\"threads\":%d,\"warmups\":10,\"repeats\":%d,"
                "\"median_us\":%.3f,\"min_us\":%.3f,\"max_us\":%.3f,\"output_hash\":\"%016llx\","
                "\"samples_us\":[",
            type == GGML_TYPE_Q4_K ? "Q4_K" : "Q5_K", rows, enabled ? "multirow" : "serial",
            round, threads, repeats, median, sorted.front(), sorted.back(),
            (unsigned long long) hash_bytes(result.data(), result.size() * sizeof(float)));
    for (size_t i = 0; i < elapsed.size(); ++i) {
        std::printf("%s%.3f", i == 0 ? "" : ",", elapsed[i]);
    }
    std::puts("]}");
    return 0;
}
