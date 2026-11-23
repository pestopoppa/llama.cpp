// DTREE-DEVICE-TREE-ACCEPT: the device tree accept (ggml_argmax per tree row, then the accept walk over the T ids)
// must be bit-identical to the host greedy tree accept (std::max_element per full-vocab row, lowest index on ties).
//
// Synthetic tree logits with FORCED TIES: every row carries several equal maxima placed to stress the reduction
// (same thread stride, across lanes of one warp/wavefront, across warps, first/last column, and -inf / -FLT_MAX
// rows). For each row the backend argmax must equal common_dtree_row_argmax, and for random trees (parents + per-path
// drafts that follow the picks for a random prefix) the accept lengths must match.
//
// Backend: CPU by default. DTREE_TEST_BACKEND=<device name> (e.g. ROCm0) runs ggml_argmax on that device instead;
// the GPU row runs it that way inside its grant. Model-free.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"
#include "speculative-tree.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int run_case(ggml_backend_t backend, int64_t n_vocab, int64_t n_rows, std::mt19937 & rng, int64_t & n_checked) {
    std::uniform_real_distribution<float> du(-8.0f, 8.0f);
    std::vector<float> host((size_t) (n_vocab * n_rows));
    for (auto & v : host) {
        v = du(rng);
    }

    // forced ties per row: k equal maxima at stress positions
    std::uniform_int_distribution<int64_t> dcol(0, n_vocab - 1);
    for (int64_t r = 0; r < n_rows; ++r) {
        float * row = host.data() + r * n_vocab;
        std::vector<int64_t> cols;
        switch (r % 8) {
            case 0: cols = { n_vocab - 1, 0 };                         break; // first and last column
            case 1: cols = { 5, 5 + 32, 5 + 64 };                      break; // same lane, successive strides (wave32)
            case 2: cols = { 5, 5 + 64, 5 + 256, 5 + 1024 };           break; // same lane (wave64 / block strides)
            case 3: cols = { 33, 1, 17 };                              break; // different lanes of one warp
            case 4: cols = { n_vocab - 7, n_vocab / 2, n_vocab / 3 };  break; // across warps
            case 5: cols = { dcol(rng), dcol(rng), dcol(rng), dcol(rng) }; break;
            case 6: cols = { dcol(rng) };                              break; // unique max
            case 7: break;                                                    // special rows below
        }
        if (r % 8 == 7) {
            // all -FLT_MAX except two equal maxima -> exact tie at the floor value used by the kernel init
            for (int64_t c = 0; c < n_vocab; ++c) {
                row[c] = (r % 16 == 7) ? -FLT_MAX : -INFINITY;
            }
            row[n_vocab - 3] = -FLT_MAX;
            row[11] = -FLT_MAX;
            continue;
        }
        for (const int64_t c : cols) {
            row[c] = 9.5f;
        }
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_rows);
    ggml_tensor * am = ggml_argmax(ctx, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, am);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(x, host.data(), 0, host.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        ggml_backend_synchronize(backend);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }
    std::vector<int32_t> dev((size_t) n_rows);
    ggml_backend_tensor_get(am, dev.data(), 0, dev.size() * sizeof(int32_t));
    // [C19 fix-up] quiesce the device before releasing this case's device memory, then free in reverse order of creation
    // (buffer, then context). Every device object the test creates is released before the next case starts.
    ggml_backend_synchronize(backend);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    // per-row picks as the device accept sees them (llama_get_argmax_ith maps -1 to LLAMA_TOKEN_NULL -> host scan)
    int fails = 0;
    std::vector<llama_token> host_pick((size_t) n_rows), dev_pick((size_t) n_rows);
    for (int64_t r = 0; r < n_rows; ++r) {
        host_pick[r] = common_dtree_row_argmax(host.data() + r * n_vocab, (int32_t) n_vocab);
        dev_pick[r]  = dev[r] >= 0 && dev[r] < n_vocab ? (llama_token) dev[r] : host_pick[r];
        if (dev_pick[r] != host_pick[r]) {
            fprintf(stderr, "FAIL n_vocab=%lld row=%lld (case %lld): device %d host %d\n",
                    (long long) n_vocab, (long long) r, (long long) (r % 8), dev[r], host_pick[r]);
            ++fails;
        }
        ++n_checked;
    }

    // random trees over these rows: a path is a row list (anchor first) and a draft that follows the host pick for a
    // random prefix and then deviates; the walk over device picks must give the host accept length
    std::uniform_int_distribution<int> dlen(1, (int) std::min<int64_t>(n_rows - 1, 16));
    for (int t = 0; t < 64; ++t) {
        const int L = dlen(rng);
        std::vector<int32_t> rows;
        std::vector<llama_token> draft;
        for (int i = 0; i <= L; ++i) {
            rows.push_back((int32_t) (std::uniform_int_distribution<int64_t>(0, n_rows - 1)(rng)));
        }
        const int keep = std::uniform_int_distribution<int>(0, L)(rng);
        for (int i = 0; i < L; ++i) {
            const llama_token h = host_pick[rows[i]];
            draft.push_back(i < keep ? h : (llama_token) ((h + 1) % n_vocab));
        }
        std::vector<llama_token> hp, dp;
        for (const int32_t r : rows) {
            hp.push_back(host_pick[r]);
            dp.push_back(dev_pick[r]);
        }
        const int32_t nh = common_dtree_accept_walk(hp, draft);
        const int32_t nd = common_dtree_accept_walk(dp, draft);
        if (nh != nd || nh != keep) {
            fprintf(stderr, "FAIL tree %d: accept host %d device %d expected %d\n", t, nh, nd, keep);
            ++fails;
        }
    }
    return fails;
}

int main() {
    ggml_backend_load_all();

    const char * want = getenv("DTREE_TEST_BACKEND");
    ggml_backend_t backend = nullptr;
    if (want != nullptr && want[0] != '\0') {
        ggml_backend_dev_t d = ggml_backend_dev_by_name(want);
        if (d == nullptr) {
            fprintf(stderr, "DTREE_TEST_BACKEND=%s: no such device\n", want);
            return 2;
        }
        backend = ggml_backend_dev_init(d, nullptr);
    } else {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (backend == nullptr) {
        fprintf(stderr, "backend init failed\n");
        return 2;
    }
    printf("test-dtree-device-accept: backend %s\n", ggml_backend_name(backend));

    std::mt19937 rng(20261123);
    int fails = 0;
    int64_t n_checked = 0;
    // Qwen3.x vocab (248320) plus odd / small widths; tree widths 12, 16, 17 rows (anchor + T)
    const int64_t vocabs[] = { 248320, 151936, 4099, 1025, 97 };
    const int64_t rowsv[]  = { 13, 17, 32 };
    for (const int64_t nv : vocabs) {
        for (const int64_t nr : rowsv) {
            for (int rep = 0; rep < 3; ++rep) {
                fails += run_case(backend, nv, nr, rng, n_checked);
            }
        }
    }
    // [C19 fix-up] c19-dtree-gpu-diag-20261123-r2: on ROCm the process crashed at EXIT in 4/8 runs (SIGSEGV in free() inside
    // libamdhip64's __cxa_finalize while an HSA thread was still in ioctl) after a correct 930/930 PASS. Release the backend
    // explicitly before main returns and after the device is idle, so no ggml-owned HIP object (stream, graph, pool buffer)
    // is still alive when the HIP runtime finalizes; report and flush before teardown so the result survives any exit fault.
    printf("test-dtree-device-accept: %lld rows checked, %d failures -> %s\n", (long long) n_checked, fails, fails ? "FAIL" : "PASS");
    fflush(stdout);
    fflush(stderr);
    ggml_backend_synchronize(backend);
    ggml_backend_free(backend);
    backend = nullptr;
    return fails ? 1 : 0;
}
