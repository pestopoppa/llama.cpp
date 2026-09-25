// Parallel tensor repack determinism: 1 thread vs N threads must be byte-identical.
//
// The CPU_REPACK buffer type rewrites quantized weights into interleaved row groups when a tensor
// is set (model load). ggml_repack_row_groups (ggml-cpu/repack.cpp) spreads the row groups over an
// OpenMP team. Repacking is pure byte movement, so any thread count must produce exactly the same
// bytes; a race or an indexing error in the parallel loop shows up as a byte difference.
//
// Covered: every repack routed through ggml_repack_row_groups that this CPU selects --
//   repack_q4_0_to_q4_0_8_bl, repack_q4_K_to_q4_K_8_bl, repack_q2_K_to_q2_K_8_bl,
//   repack_iq4_nl_to_iq4_nl_8_bl, repack_mxfp4_to_mxfp4_8_bl  (x86 AVX2/AVX-512 selections)
// over 2-D and 3-D (MoE expert stack) shapes, including row-group counts that do not divide evenly
// by the thread count. A test that verifies nothing fails (no vacuous pass).
//
// GGML_IQK=1 removes the iqk-covered types from CPU_REPACK (repack.cpp), so this test forces it off.
//
// Usage: test-repack-parallel [-v]

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "repack.h"  // ggml_backend_cpu_repack_buffer_type

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

static bool g_verbose = false;

static void generate_test_data(uint8_t * data, size_t size, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < size; i++) {
        state = state * 1103515245u + 12345u;
        data[i] = static_cast<uint8_t>((state >> 16) & 0xFF);
    }
}

struct repack_case {
    const char * name;
    ggml_type    type;
    int64_t      ne0;
    int64_t      ne1;
    int64_t      ne2;
};

struct repacked {
    bool                 supported = false;
    std::vector<uint8_t> bytes;
};

// Repack `src` into a fresh CPU_REPACK tensor using n_threads OpenMP threads; return its bytes.
static repacked do_repack(const repack_case & tc, const std::vector<uint8_t> & src, int n_threads) {
    repacked out;

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx);

    ggml_tensor * t = tc.ne2 > 1 ? ggml_new_tensor_3d(ctx, tc.type, tc.ne0, tc.ne1, tc.ne2)
                                 : ggml_new_tensor_2d(ctx, tc.type, tc.ne0, tc.ne1);
    assert(t);
    assert(ggml_nbytes(t) == src.size());

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_repack_buffer_type());
    if (buf == nullptr || t->extra == nullptr) {
        // extra == nullptr: no repack selected for this type/shape on this CPU
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        return out;
    }

#if defined(_OPENMP)
    omp_set_num_threads(n_threads);
#else
    (void) n_threads;
#endif
    ggml_backend_tensor_set(t, src.data(), 0, src.size());

    out.supported = true;
    out.bytes.assign((const uint8_t *) t->data, (const uint8_t *) t->data + ggml_nbytes(t));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            g_verbose = true;
        } else {
            fprintf(stderr, "usage: %s [-v]\n", argv[0]);
            return 1;
        }
    }

    // must precede the first repack-type decision (it caches the env once)
    setenv("GGML_IQK", "0", 1);
    unsetenv("GGML_REPACK_THREADS");

    ggml_cpu_init();

#if defined(_OPENMP)
    const int max_threads = omp_get_max_threads();
    printf("OpenMP enabled, max threads: %d\n", max_threads);
#else
    const int max_threads = 1;
    printf("OpenMP not enabled: only 1-vs-1 consistency is checked\n");
#endif

    // ne1 must be a multiple of the 8-row interleave; 8*37 = 296 rows -> 37 row groups (prime, so
    // it never divides evenly by the team size); the 3-D case is an expert stack (nrows = ne1*ne2)
    std::vector<repack_case> cases;
    const struct { const char * name; ggml_type type; } types[] = {
        { "Q4_0",   GGML_TYPE_Q4_0   },
        { "Q4_K",   GGML_TYPE_Q4_K   },
        { "Q2_K",   GGML_TYPE_Q2_K   },
        { "IQ4_NL", GGML_TYPE_IQ4_NL },
        { "MXFP4",  GGML_TYPE_MXFP4  },
    };
    for (const auto & ty : types) {
        cases.push_back({ ty.name, ty.type, 4096, 128,    1 });
        cases.push_back({ ty.name, ty.type, 2048, 8 * 37, 1 });
        cases.push_back({ ty.name, ty.type, 1024, 64,     6 });
    }

    std::vector<std::pair<int, int>> thread_pairs = { { 1, 1 } };
    for (int n : { 2, 3, 4, 7, 8, 16 }) {
        if (n <= max_threads) {
            thread_pairs.push_back({ 1, n });
        }
    }
    if (max_threads > 16) {
        thread_pairs.push_back({ 1, max_threads });
    }
    if (max_threads >= 8) {
        thread_pairs.push_back({ 2, 8 });
        thread_pairs.push_back({ 4, 7 });
    }

    int n_failed   = 0;
    int n_verified = 0;
    int n_skipped  = 0;

    for (const auto & tc : cases) {
        const int64_t nrows  = tc.ne1 * tc.ne2;
        const size_t  nbytes = (size_t) nrows * ggml_row_size(tc.type, tc.ne0);
        std::vector<uint8_t> src(nbytes);
        generate_test_data(src.data(), nbytes, 0xDEADBEEFu ^ (uint32_t) tc.type ^ (uint32_t) nrows);

        printf("%-6s ne=[%lld, %lld, %lld] (%zu bytes):", tc.name, (long long) tc.ne0, (long long) tc.ne1,
               (long long) tc.ne2, nbytes);

        const repacked ref = do_repack(tc, src, 1);
        if (!ref.supported) {
            printf(" SKIPPED (no CPU_REPACK layout for this type on this CPU)\n");
            n_skipped++;
            continue;
        }
        if (ref.bytes == src) {
            printf(" FAILED (repacked bytes equal the input: repack did not run)\n");
            n_failed++;
            continue;
        }

        bool ok = true;
        for (const auto & tp : thread_pairs) {
            const repacked a = tp.first == 1 ? ref : do_repack(tc, src, tp.first);
            const repacked b = do_repack(tc, src, tp.second);
            size_t diff = SIZE_MAX;
            for (size_t i = 0; i < a.bytes.size(); i++) {
                if (a.bytes[i] != b.bytes[i]) {
                    diff = i;
                    break;
                }
            }
            if (!b.supported || a.bytes.size() != b.bytes.size() || diff != SIZE_MAX) {
                printf("\n  [%d vs %d threads] FAILED", tp.first, tp.second);
                if (diff != SIZE_MAX) {
                    printf(" first difference at byte %zu: 0x%02X vs 0x%02X", diff, a.bytes[diff], b.bytes[diff]);
                }
                ok = false;
                n_failed++;
            } else {
                n_verified++;
                if (g_verbose) {
                    printf("\n  [%d vs %d threads] ok", tp.first, tp.second);
                }
            }
        }
        printf("%s\n", ok ? " ok" : "");
    }

    printf("\n%d comparisons verified, %d failed, %d cases skipped\n", n_verified, n_failed, n_skipped);
    if (n_verified == 0) {
        printf("FAILED: nothing was verified\n");
        return 1;
    }
    return n_failed > 0 ? 1 : 0;
}
