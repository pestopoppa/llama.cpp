// test-moe-wsum-fold (AK-CPU-MOE-WSUM-FOLD): GGML_MOE_WSUM_FOLD=1 must be BIT-IDENTICAL to =0.
// Graph = the llama build_moe_ffn tail: X = MUL_MAT_ID(down, h, ids) [n_embd, n_used, n_tok];
// M = MUL(X, w); V_k = VIEW(M, k); out = ((V_0 + V_1) + V_2) + ... ; plus a consumer of out.
// Fold cases must engage once per compute; a case where an intermediate sum has a second consumer must
// keep the stock nodes. The knob is read once per process, so the parent re-executes itself as a child
// under each env arm and compares FNV-1a hashes of every output.
// Cases: n_used 8 / 2, n_tok 1 / 3, n_embd 2048 / 1000, threads 1, 4, 24.
// Usage: test-moe-wsum-fold          (parent; rc 0 iff all hashes equal and the fold engaged as expected)
//        test-moe-wsum-fold --child  (prints "hash <case> <hash> fold <n>")
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct tcase {
    const char * name;
    int64_t n_embd, n_used, n_tok;
    bool extra_consumer;
    bool expect_fold;
};

static const tcase CASES[] = {
    { "e2048_u8_t1",        2048, 8, 1, false, true  },
    { "e2048_u8_t3",        2048, 8, 3, false, true  },
    { "e1000_u8_t1",        1000, 8, 1, false, true  },
    { "e4096_u2_t1",        4096, 2, 1, false, true  },
    { "e2048_u8_t1_extra",  2048, 8, 1, true,  false },
};

static int child() {
    const int threads[] = { 1, 4, 24 };
    const int64_t K = 64, N_EXP = 16;
    for (const tcase & tc : CASES) {
        for (int th : threads) {
            ggml_backend_t be = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(be, th);
            ggml_init_params ip = { 128 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * c = ggml_init(ip);
            ggml_tensor * down = ggml_new_tensor_3d(c, GGML_TYPE_F32, K, tc.n_embd, N_EXP);
            ggml_tensor * h    = ggml_new_tensor_3d(c, GGML_TYPE_F32, K, tc.n_used, tc.n_tok);
            ggml_tensor * ids  = ggml_new_tensor_2d(c, GGML_TYPE_I32, tc.n_used, tc.n_tok);
            ggml_tensor * w    = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, tc.n_used, tc.n_tok);
            ggml_tensor * X    = ggml_mul_mat_id(c, down, h, ids);
            ggml_tensor * M    = ggml_mul(c, X, w);
            ggml_cgraph * gf = ggml_new_graph(c);
            ggml_build_forward_expand(gf, M);
            std::vector<ggml_tensor *> v(tc.n_used);
            for (int64_t k = 0; k < tc.n_used; k++) {
                v[k] = ggml_view_2d(c, M, tc.n_embd, tc.n_tok, M->nb[2], k * M->nb[1]);
                ggml_build_forward_expand(gf, v[k]);
            }
            ggml_tensor * out = v[0];
            ggml_tensor * mid = nullptr;
            for (int64_t k = 1; k < tc.n_used; k++) {
                out = ggml_add(c, out, v[k]);
                ggml_build_forward_expand(gf, out);
                if (k == 2) mid = out;
            }
            ggml_tensor * fin = ggml_scale(c, out, 1.0f);
            ggml_build_forward_expand(gf, fin);
            ggml_tensor * side = nullptr;
            if (tc.extra_consumer) {
                side = ggml_scale(c, mid, 2.0f);
                ggml_build_forward_expand(gf, side);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);

            uint64_t sx = 0x9E3779B97F4A7C15ull ^ (uint64_t) tc.n_embd;
            auto rnd = [&]() { sx ^= sx << 13; sx ^= sx >> 7; sx ^= sx << 17; return (float) ((int64_t) (sx % 200001) - 100000) / 77777.0f; };
            auto fill = [&](ggml_tensor * t, float scale) {
                std::vector<float> d((size_t) ggml_nelements(t));
                for (auto & x : d) x = rnd() * scale;
                ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
            };
            fill(down, 0.1f);
            for (int step = 0; step < 2; step++) {
                fill(h, 1.0f);
                std::vector<float> wv((size_t) ggml_nelements(w));
                for (auto & x : wv) x = 0.05f + (rnd() + 1.3f) * 0.2f;
                ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));
                std::vector<int32_t> iv((size_t) ggml_nelements(ids));
                for (int64_t t = 0; t < tc.n_tok; t++)
                    for (int64_t k = 0; k < tc.n_used; k++) iv[t * tc.n_used + k] = (int32_t) ((k * 5 + t * 3 + step) % N_EXP);
                ggml_backend_tensor_set(ids, iv.data(), 0, ggml_nbytes(ids));
                const long long f0 = ggml_cpu_moe_wsum_fold_count();
                if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "compute failed\n");
                    return 2;
                }
                const long long f1 = ggml_cpu_moe_wsum_fold_count();
                std::vector<uint8_t> o(ggml_nbytes(fin));
                ggml_backend_tensor_get(fin, o.data(), 0, o.size());
                uint64_t hh = fnv(o.data(), o.size());
                if (side) {
                    std::vector<uint8_t> so(ggml_nbytes(side));
                    ggml_backend_tensor_get(side, so.data(), 0, so.size());
                    hh = fnv(so.data(), so.size(), hh);
                }
                printf("hash %s/t%d#%d %016" PRIx64 " fold %lld\n", tc.name, th, step, hh, f1 - f0);
            }
            ggml_backend_buffer_free(buf);
            ggml_free(c);
            ggml_backend_free(be);
        }
    }
    return 0;
}

static bool run_arm(const char * self, const char * val, std::map<std::string, std::string> & hashes,
                    std::map<std::string, long long> & fold) {
    std::string cmd = std::string("GGML_MOE_WSUM_FOLD=") + val + " GGML_MOE_WSUM_FOLD_QUIET=1 '" + self + "' --child";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) return false;
    char line[512];
    while (fgets(line, sizeof line, p)) {
        char name[160]; char h[32]; long long n;
        if (sscanf(line, "hash %159s %31s fold %lld", name, h, &n) == 3) {
            hashes[name] = h;
            std::string base(name);
            fold[base.substr(0, base.find('#'))] += n;
        }
    }
    return pclose(p) == 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {
        return child();
    }
    std::map<std::string, std::string> h0, h1;
    std::map<std::string, long long> f0, f1;
    if (!run_arm(argv[0], "0", h0, f0) || !run_arm(argv[0], "1", h1, f1)) {
        fprintf(stderr, "FAIL: child run failed\n");
        return 1;
    }
    int bad = 0;
    if (h0.empty() || h0.size() != h1.size()) {
        fprintf(stderr, "FAIL: hash count %zu vs %zu\n", h0.size(), h1.size());
        return 1;
    }
    for (const auto & kv : h0) {
        if (h1[kv.first] != kv.second) {
            fprintf(stderr, "FAIL: %s off=%s on=%s\n", kv.first.c_str(), kv.second.c_str(), h1[kv.first].c_str());
            bad++;
        }
    }
    for (const auto & kv : f0) {
        if (kv.second != 0) { fprintf(stderr, "FAIL: %s folded with knob off\n", kv.first.c_str()); bad++; }
    }
    for (const tcase & tc : CASES) {
        for (int th : { 1, 4, 24 }) {
            const std::string name = std::string(tc.name) + "/t" + std::to_string(th);
            const long long n = f1[name];
            if (tc.expect_fold ? n != 2 : n != 0) {
                fprintf(stderr, "FAIL: %s folded %lld times with knob on (expected %d)\n", name.c_str(), n, tc.expect_fold ? 2 : 0);
                bad++;
            }
        }
    }
    printf("test-moe-wsum-fold: %zu hashes compared, %d failures; MOE_WSUM_FOLD engaged on sole-consumer tails\n", h0.size(), bad);
    return bad == 0 ? 0 : 1;
}
