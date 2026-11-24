// test-router-bf16 (AK-CPU-ROUTER-BF16): GGML_ROUTER_BF16=1 must be BIT-IDENTICAL to =0.
// Graph = MUL_MAT(W, x) with W an F32 *weight* (buffer usage WEIGHTS) and x one F32 column, the shape of
// the MoE router ffn_gate_inp at decode. Exact-BF16 W must be served from the BF16 shadow; a W with any
// non-BF16 value and a 3-column x must keep the stock F32 path. The knob is read once per process, so
// the parent re-executes itself as a child under each env arm and compares FNV-1a hashes of the output.
// Cases: K 2048 (served), 2050 and 100 (scalar leftovers: must NOT be served), N 256 / 7, threads 1, 4, 24.
// Usage: test-router-bf16          (parent; rc 0 iff all hashes equal and the shadow engaged as expected)
//        test-router-bf16 --child  (prints "hash <case> <hash> served <n>")
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
    int64_t K, N, ncols;
    bool exact;
    bool expect_served;
};

static const tcase CASES[] = {
    { "k2048_n256",        2048, 256, 1, true,  true  },
    { "k2050_n7",          2050,   7, 1, true,  false },
    { "k100_n256",          100, 256, 1, true,  false },
    { "k2048_n256_inexact",2048, 256, 1, false, false },
    { "k2048_n256_3col",   2048, 256, 3, true,  false },
};

static int child() {
    const int threads[] = { 1, 4, 24 };
    for (const tcase & tc : CASES) {
        for (int th : threads) {
            ggml_backend_t be = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(be, th);
            ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * cw = ggml_init(ip);
            ggml_context * ca = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_2d(cw, GGML_TYPE_F32, tc.K, tc.N);
            ggml_set_name(W, "ffn_gate_inp.weight");
            ggml_tensor * x = ggml_new_tensor_2d(ca, GGML_TYPE_F32, tc.K, tc.ncols);
            ggml_tensor * y = ggml_mul_mat(ca, W, x);
            ggml_cgraph * gf = ggml_new_graph(ca);
            ggml_build_forward_expand(gf, y);
            ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, be);
            ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);

            uint64_t sx = 0x9E3779B97F4A7C15ull ^ (uint64_t) tc.K * 31 ^ (uint64_t) tc.N;
            auto rnd = [&]() { sx ^= sx << 13; sx ^= sx >> 7; sx ^= sx << 17; return (float) ((int64_t) (sx % 200001) - 100000) / 77777.0f; };
            std::vector<float> w((size_t) ggml_nelements(W));
            for (auto & v : w) {
                v = rnd() * 0.05f;
                if (tc.exact) {
                    uint32_t b; memcpy(&b, &v, 4); b &= 0xFFFF0000u; memcpy(&v, &b, 4);
                }
            }
            if (!tc.exact) {
                // all exact but one value: the whole matrix must be refused
                for (auto & v : w) { uint32_t b; memcpy(&b, &v, 4); b &= 0xFFFF0000u; memcpy(&v, &b, 4); }
                uint32_t b; memcpy(&b, &w[w.size() / 2], 4); b |= 1u; memcpy(&w[w.size() / 2], &b, 4);
            }
            ggml_backend_tensor_set(W, w.data(), 0, ggml_nbytes(W));

            for (int step = 0; step < 2; step++) {
                std::vector<float> xv((size_t) ggml_nelements(x));
                for (auto & v : xv) v = rnd();
                ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
                const long long s0 = ggml_cpu_router_bf16_count();
                if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "compute failed\n");
                    return 2;
                }
                const long long s1 = ggml_cpu_router_bf16_count();
                std::vector<uint8_t> o(ggml_nbytes(y));
                ggml_backend_tensor_get(y, o.data(), 0, o.size());
                printf("hash %s/t%d#%d %016" PRIx64 " served %lld\n", tc.name, th, step, fnv(o.data(), o.size()), s1 - s0);
            }
            ggml_backend_buffer_free(ba);
            ggml_backend_buffer_free(bw);
            ggml_free(ca);
            ggml_free(cw);
            ggml_backend_free(be);
        }
    }
    return 0;
}

static bool run_arm(const char * self, const char * val, std::map<std::string, std::string> & hashes,
                    std::map<std::string, long long> & served) {
    std::string cmd = std::string("GGML_ROUTER_BF16=") + val + " GGML_ROUTER_BF16_QUIET=1 '" + self + "' --child";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) return false;
    char line[512];
    while (fgets(line, sizeof line, p)) {
        char name[160]; char h[32]; long long n;
        if (sscanf(line, "hash %159s %31s served %lld", name, h, &n) == 3) {
            hashes[name] = h;
            std::string base(name);
            served[base.substr(0, base.find('#'))] += n;
        }
    }
    return pclose(p) == 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {
        return child();
    }
    std::map<std::string, std::string> h0, h1;
    std::map<std::string, long long> s0, s1;
    if (!run_arm(argv[0], "0", h0, s0) || !run_arm(argv[0], "1", h1, s1)) {
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
    for (const auto & kv : s0) {
        if (kv.second != 0) { fprintf(stderr, "FAIL: %s served with knob off\n", kv.first.c_str()); bad++; }
    }
    for (const tcase & tc : CASES) {
        for (int th : { 1, 4, 24 }) {
            const std::string name = std::string(tc.name) + "/t" + std::to_string(th);
            const long long n = s1[name];
            if (tc.expect_served ? n != 2 : n != 0) {
                fprintf(stderr, "FAIL: %s served %lld times with knob on (expected %d)\n", name.c_str(), n, tc.expect_served ? 2 : 0);
                bad++;
            }
        }
    }
    printf("test-router-bf16: %zu hashes compared, %d failures; ROUTER_BF16 engaged on exact single-column cases\n", h0.size(), bad);
    return bad == 0 ? 0 : 1;
}
