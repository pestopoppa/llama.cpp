// AK Q38FN-CPU-MOE-GEMV-1: bit-identity witness for the two default-off CPU levers.
//   lever 1 GGML_MOE_UPGATE_SLAB    : MoE up+gate MUL_MAT_ID + SWIGLU in one iqk slab region
//   lever 2 GGML_IQK_SKINNY_BALANCE : balanced row split for skinny iqk GEMVs (Nx/nth < 32)
// Both knobs are read once per process, so the A/B is two processes:
//   GGML_IQK=1 [GGML_IQK_Q8_0=1]                          test-q38fn-moe-levers dump off.bin
//   GGML_IQK=1 [GGML_IQK_Q8_0=1] GGML_MOE_UPGATE_SLAB=1 \
//              GGML_IQK_SKINNY_BALANCE=1                  test-q38fn-moe-levers dump on.bin
//   cmp off.bin on.bin                                    -- must be byte-identical
// No args = ctest smoke (exercise every graph once).
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <map>
#include <tuple>
#include <vector>

static std::vector<float> run_graph(ggml_type type, bool moe, int64_t M, int64_t K, int64_t NT, int nth, bool invalid_id) {
    const int64_t E = 12, U = 10;
    ggml_init_params ip = { (size_t) 512 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(4242 + (int) type * 7 + (int) M);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // quantized weights are cached per (type, M, K, ne2, which) -- IQ3_S quantization dominates otherwise
    static std::map<std::tuple<int, int64_t, int64_t, int64_t, int>, std::vector<uint8_t>> cache;
    int which = 0;
    auto mk_w = [&](int64_t ne2) {
        ggml_tensor * w = ggml_new_tensor_3d(ctx, type, K, M, ne2);
        auto key = std::make_tuple((int) type, M, K, ne2, which++);
        auto it = cache.find(key);
        if (it == cache.end()) {
            std::mt19937 wr(99 + (int) type * 13 + (int) M + which);
            std::normal_distribution<float> wd(0.0f, 1.0f);
            std::vector<float> f(K * M * ne2);
            for (auto & x : f) { x = 0.05f * wd(wr); }
            std::vector<uint8_t> q(ggml_nbytes(w));
            ggml_quantize_chunk(type, f.data(), q.data(), 0, M * ne2, K, nullptr);
            it = cache.emplace(key, std::move(q)).first;
        }
        memcpy(w->data, it->second.data(), it->second.size());
        return w;
    };
    ggml_tensor * out = nullptr;
    if (moe) {
        ggml_tensor * wu = mk_w(E);
        ggml_tensor * wg = mk_w(E);
        ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, NT);
        for (int64_t i = 0; i < K * NT; ++i) { ((float *) x->data)[i] = nd(rng); }
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, NT);
        for (int64_t t = 0; t < NT; ++t) {
            for (int64_t u = 0; u < U; ++u) {
                ((int32_t *) ids->data)[t * U + u] = (int32_t) ((u * 3 + t * 5 + 7) % E);
            }
        }
        if (invalid_id) { ((int32_t *) ids->data)[4] = -1; }
        ggml_tensor * up   = ggml_mul_mat_id(ctx, wu, x, ids);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, wg, x, ids);
        out = ggml_swiglu_split(ctx, gate, up);
    } else {
        ggml_tensor * w = mk_w(1);
        w = ggml_reshape_2d(ctx, w, K, M);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NT);
        for (int64_t i = 0; i < K * NT; ++i) { ((float *) x->data)[i] = nd(rng); }
        out = ggml_mul_mat(ctx, w, x);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "CPU backend init failed\n"); exit(2); }
    ggml_backend_cpu_set_n_threads(backend, nth);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(2); }
    std::vector<float> r((float *) out->data, (float *) out->data + ggml_nelements(out));
    ggml_backend_free(backend);
    ggml_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    struct cell { ggml_type t; bool moe; int64_t M, K, NT; int nth; bool inval; };
    std::vector<cell> cells;
    for (ggml_type t : { GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS }) {
        for (int nth : { 1, 7, 24, 96 }) {
            cells.push_back({ t, true, 640, 2560, 1, nth, false });
        }
        cells.push_back({ t, true, 640, 2560, 1, 24, true });   // inactive SER route
        cells.push_back({ t, true, 640, 2560, 5, 24, false });  // MTP verify: lever 1 must decline
    }
    for (ggml_type t : { GGML_TYPE_Q8_0, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS }) {
        for (int nth : { 7, 24, 96 }) {
            for (int64_t nt : { 1, 5 }) {
                cells.push_back({ t, false,  320, 10240, nt, nth, false });  // hc_*_down
                cells.push_back({ t, false,  640,  2560, nt, nth, false });  // shexp gate/up
                cells.push_back({ t, false, 2560,   768, nt, nth, false });  // shexp-down-like
            }
        }
    }
    std::vector<float> all;
    for (const auto & c : cells) {
        auto r = run_graph(c.t, c.moe, c.M, c.K, c.NT, c.nth, c.inval);
        for (float v : r) {
            if (!(v == v)) { fprintf(stderr, "NaN in cell type=%d M=%lld nth=%d\n", (int) c.t, (long long) c.M, c.nth); return 1; }
        }
        all.insert(all.end(), r.begin(), r.end());
    }
    if (argc == 3 && std::string(argv[1]) == "dump") {
        FILE * f = fopen(argv[2], "wb");
        if (!f) { perror("open"); return 2; }
        fwrite(all.data(), sizeof(float), all.size(), f);
        fclose(f);
        printf("RESULT test-q38fn-moe-levers dump cells=%zu floats=%zu\n", cells.size(), all.size());
        return 0;
    }
    printf("RESULT test-q38fn-moe-levers smoke cells=%zu pass=1 fail=0\n", cells.size());
    return 0;
}
