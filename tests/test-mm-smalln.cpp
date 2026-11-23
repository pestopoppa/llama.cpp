// test-mm-smalln (C20): GGML_Q0_RM4 (tinyBLAS_Q0 4-row tiles at N=2/3, default ON) must be bit-identical to the two-row tails
// (GGML_Q0_RM4=0) and to an explicit =1. GGML_MM_SMALLN is NOT part of C20, so its arms are dropped.
// small-batch Q8_0 mul_mat path. The knobs are read once per process, so the parent re-executes itself as a child under
// each env arm and compares the FNV-1a hash of every output tensor byte for byte.
//   shapes: A3B shared expert gate/up [2048,512], down [512,2048], plus odd shapes [96,40] and [512,33] (generic fallback /
//   tail rows); N = 1..8; threads = 24, 12, 7, 1.
// Usage: test-mm-smalln            (parent: runs the arms, rc 0 iff all hashes equal and the knob path was exercised)
//        test-mm-smalln --child    (prints one "hash <shape> <N> <threads> <hex>" line per case)
#include <cinttypes>
#include <cmath>
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

static int child() {
    const int64_t shapes[][2] = { {2048, 512}, {2048, 512}, {512, 2048}, {96, 40}, {512, 33} };
    const char * names[] = { "gate", "up", "down", "odd96x40", "odd512x33" };
    const int threads[] = { 24, 12, 7, 1 };
    for (int s = 0; s < 5; s++) {
        const int64_t K = shapes[s][0], M = shapes[s][1];
        for (int N = 1; N <= 8; N++) {
            for (int th : threads) {
                ggml_backend_t be = ggml_backend_cpu_init();
                ggml_backend_cpu_set_n_threads(be, th);
                ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead() , nullptr, true };
                ggml_context * c = ggml_init(ip);
                ggml_tensor * W = ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, K, M);
                ggml_tensor * X = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, N);
                ggml_tensor * O = ggml_mul_mat(c, W, X);
                ggml_cgraph * g = ggml_new_graph(c);
                ggml_build_forward_expand(g, O);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
                std::vector<float> wf((size_t) K * M), xf((size_t) K * N);
                uint64_t st = 0x9E3779B97F4A7C15ull * (s + 1);
                for (auto & v : wf) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; v = (float) ((int64_t) (st % 2001) - 1000) / 997.0f; }
                for (size_t i = 0; i < xf.size(); i++) xf[i] = std::sin(0.37f * (float) i + (float) s) * 1.7f;
                std::vector<uint8_t> wq(ggml_nbytes(W));
                ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, M, K, nullptr);
                ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
                ggml_backend_tensor_set(X, xf.data(), 0, xf.size() * 4);
                if (ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 2; }
                std::vector<uint8_t> out(ggml_nbytes(O));
                ggml_backend_tensor_get(O, out.data(), 0, out.size());
                printf("hash %s %d %d %016" PRIx64 "\n", names[s], N, th, fnv(out.data(), out.size()));
                ggml_backend_buffer_free(buf); ggml_free(c); ggml_backend_free(be);
            }
        }
    }
    return 0;
}

static std::map<std::string, std::string> run_arm(const char * self, const char * env) {
    std::string cmd = std::string("env ") + env + " '" + self + "' --child";
    std::map<std::string, std::string> m;
    FILE * f = popen(cmd.c_str(), "r");
    if (!f) return m;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char a[64], b[64]; int n, t; char h[64];
        if (sscanf(line, "hash %63s %d %d %63s", a, &n, &t, h) == 4) m[std::string(a) + "/" + std::to_string(n) + "/" + std::to_string(t)] = h;
        (void) b;
    }
    if (pclose(f) != 0) m.clear();
    return m;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) return child();
    const char * arms[] = { "-u GGML_Q0_RM4", "GGML_Q0_RM4=0", "GGML_Q0_RM4=1" };
    const int narms = (int) (sizeof(arms) / sizeof(arms[0]));
    auto ref = run_arm(argv[0], arms[0]);
    if (ref.size() != 5 * 8 * 4) { fprintf(stderr, "reference arm produced %zu cases\n", ref.size()); return 1; }
    int bad = 0;
    for (int a = 1; a < narms; a++) {
        auto got = run_arm(argv[0], arms[a]);
        if (got.size() != ref.size()) { fprintf(stderr, "arm '%s' produced %zu cases\n", arms[a], got.size()); return 1; }
        for (auto & kv : ref) if (got[kv.first] != kv.second) { fprintf(stderr, "MISMATCH arm '%s' case %s\n", arms[a], kv.first.c_str()); bad++; }
    }
    printf("%s: %zu cases x %d knob arms vs default, %d mismatches\n", bad ? "FAIL" : "PASS", ref.size(), narms - 1, bad);
    return bad ? 1 : 0;
}
