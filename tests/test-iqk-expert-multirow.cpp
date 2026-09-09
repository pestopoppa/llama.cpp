#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

constexpr int64_t N_EXPERTS = 5;
constexpr int64_t N_USED    = 8;
constexpr int64_t N_TOKENS  = 3;

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

uint32_t random_u32(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float test_value(size_t i, uint32_t & state) {
    static const float scales[] = { 0.0f, 0.000244140625f, 0.03125f, 1.0f, 8.0f, 127.0f };
    const float unit = (int32_t) random_u32(state) / 2147483648.0f;
    float value = unit * scales[(i / 257) % (sizeof(scales) / sizeof(scales[0]))];
    if (i % 509 == 0) value = (i & 1) ? -512.0f : 512.0f;
    if (i % 521 == 0) value = (i & 1) ? -0.0f : 0.0f;
    return value;
}

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::vector<uint8_t> make_weights(ggml_type type, int64_t k, int64_t rows) {
    std::vector<float> values((size_t) k * rows);
    uint32_t state = type == GGML_TYPE_Q4_K ? 0x13579bdfu : 0x2468ace1u;
    for (size_t i = 0; i < values.size(); ++i) values[i] = test_value(i, state);

    std::vector<uint8_t> quantized(ggml_row_size(type, k) * (size_t) rows);
    const size_t written = ggml_quantize_chunk(type, values.data(), quantized.data(), 0, rows, k, nullptr);
    if (written != quantized.size()) std::abort();
    return quantized;
}

std::vector<float> compute(ggml_type type, bool multirow) {
    set_env("GGML_IQK", "1");
    set_env("GGML_ROWEXACT_N", "16");
    set_env("GGML_IQK_EXPERT_MULTIROW", multirow ? "1" : "0");
    set_env("GGML_IQK_EXPERT_MULTIROW_TRACE", multirow ? "1" : "0");

    const int64_t k = type == GGML_TYPE_Q4_K ? 4096 : 2048;
    const int64_t m = type == GGML_TYPE_Q4_K ? 2048 : 4096;

    ggml_init_params params = { ggml_tensor_overhead() * 12 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) std::abort();

    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), type, k, m, N_EXPERTS);
    ggml_tensor * input   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, N_USED, N_TOKENS);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, N_USED, N_TOKENS);
    ggml_tensor * output  = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_cgraph * graph   = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(backend.get(), 48);
    ggml_backend_cpu_set_rowexact(backend.get(), true);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) std::abort();

    auto quantized = make_weights(type, k, m * N_EXPERTS);
    ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());

    std::vector<float> activations((size_t) ggml_nelements(input));
    uint32_t state = type == GGML_TYPE_Q4_K ? 0x10203040u : 0x50607080u;
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = test_value(i + 17, state);
    ggml_backend_tensor_set(input, activations.data(), 0, activations.size() * sizeof(float));

    const int32_t routes[N_USED * N_TOKENS] = {
        0, 1, 0, 2, 3, 0, 4, 1,
        2, 0, 1, 0, 4, 3, 2, 0,
        1, 3, 0, 4, 1, 2, 3, 0,
    };
    ggml_backend_tensor_set(ids, routes, 0, sizeof(routes));

    ggml_backend_graph_plan_t plan = ggml_backend_graph_plan_create(backend.get(), graph);
    if (!plan || ggml_backend_graph_plan_compute(backend.get(), plan) != GGML_STATUS_SUCCESS) std::abort();

    std::vector<float> result((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_graph_plan_free(backend.get(), plan);
    return result;
}

#if !defined(_WIN32)
bool write_all(int fd, const void * data, size_t size) {
    const char * p = (const char *) data;
    while (size > 0) {
        const ssize_t n = write(fd, p, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        size -= (size_t) n;
    }
    return true;
}

bool read_all(int fd, void * data, size_t size) {
    char * p = (char *) data;
    while (size > 0) {
        const ssize_t n = read(fd, p, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        size -= (size_t) n;
    }
    return true;
}

std::vector<float> capture(ggml_type type, bool multirow) {
    int fds[2];
    if (pipe(fds) != 0) std::abort();
    const pid_t pid = fork();
    if (pid < 0) std::abort();
    if (pid == 0) {
        close(fds[0]);
        const auto result = compute(type, multirow);
        const size_t bytes = result.size() * sizeof(float);
        const bool ok = write_all(fds[1], &bytes, sizeof(bytes)) && write_all(fds[1], result.data(), bytes);
        close(fds[1]);
        _exit(ok ? 0 : 2);
    }

    close(fds[1]);
    size_t bytes = 0;
    if (!read_all(fds[0], &bytes, sizeof(bytes)) || bytes % sizeof(float) != 0) std::abort();
    std::vector<float> result(bytes / sizeof(float));
    if (!read_all(fds[0], result.data(), bytes)) std::abort();
    close(fds[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) std::abort();
    return result;
}
#endif

} // namespace

int main() {
#if defined(_WIN32)
    std::puts("SKIP: fork is required to isolate process-cached experiment knobs");
    return 0;
#else
    for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K }) {
        const auto serial   = capture(type, false);
        const auto multirow = capture(type, true);
        if (serial.size() != multirow.size()) {
            std::fprintf(stderr, "%s size mismatch: %zu vs %zu\n",
                    ggml_type_name(type), serial.size(), multirow.size());
            return 1;
        }
        for (size_t i = 0; i < serial.size(); ++i) {
            if (!std::isfinite(serial[i]) || !std::isfinite(multirow[i])) {
                std::fprintf(stderr, "%s produced a non-finite value at element %zu\n", ggml_type_name(type), i);
                return 2;
            }
        }
        if (std::memcmp(serial.data(), multirow.data(), serial.size() * sizeof(float)) != 0) {
            size_t first = 0;
            while (first < serial.size() &&
                    std::memcmp(&serial[first], &multirow[first], sizeof(float)) == 0) ++first;
            std::fprintf(stderr, "%s differs at element %zu: 0x%08x vs 0x%08x\n",
                    ggml_type_name(type), first,
                    first < serial.size() ? float_bits(serial[first]) : 0,
                    first < multirow.size() ? float_bits(multirow[first]) : 0);
            return 1;
        }
        std::printf("%s: %zu outputs bit-identical\n", ggml_type_name(type), serial.size());
    }
    return 0;
#endif
}
