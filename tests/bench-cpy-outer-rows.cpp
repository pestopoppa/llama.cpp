#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

constexpr int64_t ne0 = 1048576;
constexpr int64_t ne1 = 1;
constexpr int64_t ne2 = 4;
constexpr int64_t ne3 = 1;

uint64_t fnv1a(const std::vector<uint8_t> & data) {
    uint64_t hash = 1469598103934665603ULL;
    for (uint8_t byte : data) hash = (hash ^ byte) * 1099511628211ULL;
    return hash;
}

uint64_t next_random(uint64_t & state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

struct result {
    double elapsed_ms;
    uint64_t checksum;
    uint64_t bytes;
    int exact;
};

result run_case(int repeats) {
    const size_t rs = ggml_row_size(GGML_TYPE_F32, ne0);
    const size_t src_nb2 = rs + 4096;
    const size_t dst_nb2 = rs + 8192;
    const size_t src_bytes = rs + (ne2 - 1) * src_nb2 + 64;
    const size_t dst_bytes = rs + (ne2 - 1) * dst_nb2 + 64;

    ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(ip));
    if (!ctx) std::abort();

    ggml_tensor * src_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0,
        (int64_t) ((src_bytes + rs - 1) / rs));
    ggml_tensor * dst_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0,
        (int64_t) ((dst_bytes + rs - 1) / rs));
    ggml_tensor * src = ggml_view_4d(ctx.get(), src_base, ne0, ne1, ne2, ne3,
        rs, src_nb2, ne2 * src_nb2, 0);
    ggml_tensor * dst = ggml_view_4d(ctx.get(), dst_base, ne0, ne1, ne2, ne3,
        rs, dst_nb2, ne2 * dst_nb2, 0);
    ggml_tensor * out = ggml_cpy(ctx.get(), src, dst);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(backend.get(), 48);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) std::abort();

    std::vector<uint8_t> input(ggml_nbytes(src_base));
    std::vector<uint8_t> initial(ggml_nbytes(dst_base), 0xa5);
    std::vector<uint8_t> expected = initial;
    uint64_t random_state = 0x4d595df4d0f33173ULL;
    for (uint8_t & byte : input) byte = (uint8_t) next_random(random_state);
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        memcpy(expected.data() + (size_t) i2 * dst_nb2,
               input.data() + (size_t) i2 * src_nb2, rs);
    }

    ggml_backend_tensor_set(src_base, input.data(), 0, input.size());
    ggml_backend_tensor_set(dst_base, initial.data(), 0, initial.size());
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) std::abort();

    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) std::abort();
    }
    const auto end = std::chrono::steady_clock::now();

    std::vector<uint8_t> actual(initial.size());
    ggml_backend_tensor_get(dst_base, actual.data(), 0, actual.size());
    return {
        std::chrono::duration<double, std::milli>(end - begin).count() / repeats,
        fnv1a(actual),
        actual.size(),
        actual == expected,
    };
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

result capture(int mode, int repeats) {
    int fds[2];
    if (pipe(fds) != 0) std::abort();
    const pid_t pid = fork();
    if (pid < 0) std::abort();
    if (pid == 0) {
        close(fds[0]);
        setenv("GGML_CPY_OUTER_ROWS", mode ? "1" : "0", 1);
        setenv("GGML_CPY_OUTER_ROWS_TRACE", mode ? "1" : "0", 1);
        ggml_cpu_init();
        const result value = run_case(repeats);
        const bool ok = write_all(fds[1], &value, sizeof(value));
        close(fds[1]);
        _exit(ok ? 0 : 2);
    }
    close(fds[1]);
    result value = {};
    const bool ok = read_all(fds[0], &value, sizeof(value));
    close(fds[0]);
    int status = 0;
    if (!ok || waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) std::abort();
    return value;
}
#endif

} // namespace

int main() {
#if defined(_WIN32)
    std::puts("SKIP: fork is required to isolate the process-cached experiment knob");
    return 0;
#else
    constexpr int cycles = 5;
    constexpr int repeats = 50;
    double off_samples[2 * cycles] = {};
    double on_samples[2 * cycles] = {};
    uint64_t checksum = 0;
    int sample = 0;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        const result off_a = capture(0, repeats);
        const result on_a  = capture(1, repeats);
        const result on_b  = capture(1, repeats);
        const result off_b = capture(0, repeats);
        const result values[] = { off_a, on_a, on_b, off_b };
        for (const result & value : values) {
            if (!value.exact || value.bytes != off_a.bytes || value.checksum != off_a.checksum) {
                std::fprintf(stderr, "CPY off/on byte mismatch in cycle %d\n", cycle);
                return 1;
            }
        }
        checksum = off_a.checksum;
        off_samples[2 * cycle] = off_a.elapsed_ms;
        off_samples[2 * cycle + 1] = off_b.elapsed_ms;
        on_samples[2 * cycle] = on_a.elapsed_ms;
        on_samples[2 * cycle + 1] = on_b.elapsed_ms;
        std::printf("cycle=%d off_a_ms=%.6f on_a_ms=%.6f on_b_ms=%.6f off_b_ms=%.6f exact=1\n",
            cycle, off_a.elapsed_ms, on_a.elapsed_ms, on_b.elapsed_ms, off_b.elapsed_ms);
        sample += 2;
    }
    double off_sum = 0.0;
    double on_sum = 0.0;
    for (int i = 0; i < sample; ++i) {
        off_sum += off_samples[i];
        on_sum += on_samples[i];
    }
    const double off_mean = off_sum / sample;
    const double on_mean = on_sum / sample;
    std::printf("cpy-f32-1048576x1x4 workers=48 repeats=%d samples=%d off_mean_ms=%.6f on_mean_ms=%.6f speedup=%.6f checksum=%016llx exact=1\n",
        repeats, sample, off_mean, on_mean, off_mean / on_mean, (unsigned long long) checksum);
    return 0;
#endif
}
