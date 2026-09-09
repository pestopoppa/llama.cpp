#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

struct test_case {
    int64_t k;
    int64_t m;
    int64_t ny;
    int64_t heads;
    int64_t weight_heads;
    bool native_q8;
    bool noncontiguous;
};

static float weight_value(size_t i) {
    static const float magnitude[] = { 0.0009f, 0.031f, 0.47f, 7.3f };
    const float sign = (i % 2 == 0) ? 1.0f : -1.0f;
    return sign * magnitude[(i / 7) % 4] * (0.71f + 0.29f * std::sin((float) i * 0.019f));
}

static float activation_value(size_t i) {
    static const float magnitude[] = { 0.0017f, 0.083f, 0.91f, 11.0f };
    const float sign = ((i / 3) % 2 == 0) ? -1.0f : 1.0f;
    return sign * magnitude[(i / 11) % 4] * (0.63f + 0.37f * std::cos((float) i * 0.013f));
}

static std::vector<uint8_t> quantize_q8(const std::vector<float> & src, int64_t k) {
    const int64_t rows = src.size() / k;
    std::vector<uint8_t> dst(ggml_row_size(GGML_TYPE_Q8_0, k) * rows);
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), dst.data(), 0, rows, k, nullptr);
    if (written != dst.size()) {
        std::abort();
    }
    return dst;
}

static std::vector<uint8_t> make_weights_q8(const test_case & tc) {
    const int64_t rows = tc.m * tc.weight_heads;
    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, tc.k);
    const ggml_type_traits_cpu * q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::vector<uint8_t> dst(row_size * rows);
    std::vector<float> row(tc.k);
    for (int64_t ir = 0; ir < rows; ++ir) {
        for (int64_t i = 0; i < tc.k; ++i) {
            row[i] = weight_value((size_t) ir * tc.k + i);
        }
        q8->from_float(row.data(), dst.data() + ir * row_size, tc.k);
    }
    return dst;
}

static bool run_case(const test_case & tc, int repeats = 0, double * elapsed_ms = nullptr,
        std::vector<float> * captured = nullptr) {
    const int64_t input_rows = tc.ny * tc.heads;
    const int64_t base_rows = tc.noncontiguous ? 2 * input_rows : input_rows;
    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, tc.k);

    std::vector<float> input_f((size_t) tc.k * base_rows);
    for (size_t i = 0; i < input_f.size(); ++i) {
        input_f[i] = activation_value(i);
    }

    const std::vector<uint8_t> weights_q = make_weights_q8(tc);
    const std::vector<uint8_t> input_q = quantize_q8(input_f, tc.k);

    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        std::abort();
    }

    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q8_0, tc.k, tc.m, tc.weight_heads);
    const ggml_type input_type = tc.native_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
    ggml_tensor * input_base = ggml_new_tensor_3d(ctx.get(), input_type, tc.k,
                                                  tc.noncontiguous ? 2 * tc.ny : tc.ny, tc.heads);
    ggml_tensor * input = input_base;
    if (tc.noncontiguous) {
        input = ggml_view_3d(ctx.get(), input_base, tc.k, tc.ny, tc.heads,
                             2 * input_base->nb[1], input_base->nb[2], 0);
    }
    ggml_tensor * output = ggml_mul_mat(ctx.get(), weights, input);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(backend.get(), repeats > 0 ? 48 : 3);
    ggml_backend_cpu_set_rowexact(backend.get(), true);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) {
        std::abort();
    }

    ggml_backend_tensor_set(weights, weights_q.data(), 0, weights_q.size());
    if (tc.native_q8) {
        ggml_backend_tensor_set(input_base, input_q.data(), 0, input_q.size());
    } else {
        ggml_backend_tensor_set(input_base, input_f.data(), 0, input_f.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        std::abort();
    }
    if (repeats > 0) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) {
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                std::abort();
            }
        }
        const auto end = std::chrono::steady_clock::now();
        *elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count() / repeats;
    }
    std::vector<float> actual((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    if (captured != nullptr) {
        *captured = std::move(actual);
        return true;
    }

    const ggml_type_traits_cpu * q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::vector<float> expected(actual.size());
    for (int64_t h = 0; h < tc.heads; ++h) {
        const int64_t wh = h / (tc.heads / tc.weight_heads);
        for (int64_t iy = 0; iy < tc.ny; ++iy) {
            const int64_t logical_row = h * tc.ny + iy;
            const int64_t base_row = tc.noncontiguous ? 2 * logical_row : logical_row;
            const uint8_t * y = input_q.data() + base_row * row_size;
            for (int64_t im = 0; im < tc.m; ++im) {
                const uint8_t * x = weights_q.data() + (wh * tc.m + im) * row_size;
                q8->vec_dot(tc.k, &expected[(logical_row * tc.m) + im], 0, x, 0, y, 0, 1);
            }
        }
    }

    const bool equal = std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            std::fprintf(stderr, "q8 batch produced non-finite output at index %zu\n", i);
            return false;
        }
    }
    if (!equal) {
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
                std::fprintf(stderr,
                    "q8 batch mismatch k=%lld m=%lld ny=%lld heads=%lld wh=%lld native=%d noncontig=%d index=%zu expected=%a actual=%a\n",
                    (long long) tc.k, (long long) tc.m, (long long) tc.ny, (long long) tc.heads,
                    (long long) tc.weight_heads, tc.native_q8, tc.noncontiguous, i, expected[i], actual[i]);
                break;
            }
        }
    }
    return equal;
}

static void configure_env(int mode, bool trace) {
#if defined(_WIN32)
    _putenv_s("GGML_IQK", "0");
    _putenv_s("GGML_ROWEXACT_N", "4");
    _putenv_s("GGML_Q8_ROWEXACT_BATCH", std::to_string(mode).c_str());
    _putenv_s("GGML_Q8_ROWEXACT_BATCH_TRACE", trace ? "1" : "0");
#else
    setenv("GGML_IQK", "0", 1);
    setenv("GGML_ROWEXACT_N", "4", 1);
    setenv("GGML_Q8_ROWEXACT_BATCH", std::to_string(mode).c_str(), 1);
    setenv("GGML_Q8_ROWEXACT_BATCH_TRACE", trace ? "1" : "0", 1);
#endif
}

#if !defined(_WIN32)
static bool write_all(int fd, const void * data, size_t size) {
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

static bool read_all(int fd, void * data, size_t size) {
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

struct bench_result {
    double elapsed_ms;
    std::vector<float> output;
};

static bench_result capture_benchmark(const test_case & tc, int repeats, int mode) {
    int fds[2];
    if (pipe(fds) != 0) std::abort();
    const pid_t pid = fork();
    if (pid < 0) std::abort();
    if (pid == 0) {
        close(fds[0]);
        configure_env(mode, mode != 0);
        ggml_cpu_init();
        bench_result result;
        run_case(tc, repeats, &result.elapsed_ms, &result.output);
        const size_t count = result.output.size();
        const bool ok = write_all(fds[1], &result.elapsed_ms, sizeof(result.elapsed_ms)) &&
            write_all(fds[1], &count, sizeof(count)) &&
            write_all(fds[1], result.output.data(), count * sizeof(float));
        close(fds[1]);
        _exit(ok ? 0 : 2);
    }

    close(fds[1]);
    bench_result result;
    size_t count = 0;
    if (!read_all(fds[0], &result.elapsed_ms, sizeof(result.elapsed_ms)) ||
        !read_all(fds[0], &count, sizeof(count))) std::abort();
    result.output.resize(count);
    if (!read_all(fds[0], result.output.data(), count * sizeof(float))) std::abort();
    close(fds[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) std::abort();
    return result;
}

static int run_benchmarks() {
    struct benchmark_case {
        const char * name;
        test_case shape;
        int repeats;
    };
    const benchmark_case cases[] = {
        { "mla-64head",       { 512, 256, 4, 64, 64, false, false }, 200 },
        { "output-projection", { 4096, 154880, 4, 1, 1, false, false }, 10 },
    };

    for (const benchmark_case & bc : cases) {
        const bench_result off_a = capture_benchmark(bc.shape, bc.repeats, 0);
        const bench_result on_a  = capture_benchmark(bc.shape, bc.repeats, 2);
        const bench_result on_b  = capture_benchmark(bc.shape, bc.repeats, 2);
        const bench_result off_b = capture_benchmark(bc.shape, bc.repeats, 0);
        if (off_a.output.size() != on_a.output.size() ||
            std::memcmp(off_a.output.data(), on_a.output.data(), off_a.output.size() * sizeof(float)) != 0 ||
            std::memcmp(off_a.output.data(), on_b.output.data(), off_a.output.size() * sizeof(float)) != 0 ||
            std::memcmp(off_a.output.data(), off_b.output.data(), off_a.output.size() * sizeof(float)) != 0) {
            std::fprintf(stderr, "%s off/on output mismatch\n", bc.name);
            return 1;
        }
        for (float value : on_a.output) {
            if (!std::isfinite(value)) {
                std::fprintf(stderr, "%s produced non-finite output\n", bc.name);
                return 1;
            }
        }
        uint64_t checksum = 1469598103934665603ULL;
        const uint8_t * bytes = (const uint8_t *) on_a.output.data();
        for (size_t i = 0; i < on_a.output.size() * sizeof(float); ++i) {
            checksum = (checksum ^ bytes[i]) * 1099511628211ULL;
        }
        const double off_ms = (off_a.elapsed_ms + off_b.elapsed_ms) / 2;
        const double on_ms = (on_a.elapsed_ms + on_b.elapsed_ms) / 2;
        std::printf("%s repeats=%d off_ab_ms=%.3f on_ab_ms=%.3f on_ba_ms=%.3f off_ba_ms=%.3f off_mean_ms=%.3f on_mean_ms=%.3f speedup=%.4f checksum=%016llx exact=1\n",
            bc.name, bc.repeats, off_a.elapsed_ms, on_a.elapsed_ms, on_b.elapsed_ms, off_b.elapsed_ms,
            off_ms, on_ms, off_ms / on_ms,
            (unsigned long long) checksum);
    }
    return 0;
}
#endif

static int run_correctness(int mode) {
    configure_env(mode, mode != 0);
    ggml_cpu_init();

    const test_case cases[] = {
        {   512, 17, 1, 3, 3, true,  true  },
        {   256, 17, 4, 5, 5, true,  true  },
        {   512, 24, 3, 4, 2, false, true  },
        {  4096, 17, 2, 1, 1, false, false },
        { 16384, 24, 4, 1, 1, true,  false },
    };

    int failed = 0;
    for (const test_case & tc : cases) {
        failed += !run_case(tc);
    }
    std::printf("q8 rowexact batch mode=%d: %zu cases, %d failed\n",
        mode, sizeof(cases) / sizeof(cases[0]), failed);
    return failed == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--bench") {
#if defined(_WIN32)
        std::puts("SKIP: fork is required to isolate process-cached experiment knobs");
        return 0;
#else
        return run_benchmarks();
#endif
    }
    if (argc != 1) return 2;

#if defined(_WIN32)
    return run_correctness(2);
#else
    for (int mode : { 0, 1, 2 }) {
        const pid_t pid = fork();
        if (pid < 0) std::abort();
        if (pid == 0) {
            const int result = run_correctness(mode);
            std::fflush(nullptr);
            _exit(result);
        }
        int status = 0;
        if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return 1;
    }
    std::puts("q8 rowexact batch: modes 0/1/2 passed");
    return 0;
#endif
}
