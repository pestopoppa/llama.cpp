// test-conv-state-fwd (AK-CPU-CONV-STATE-FWD): GGML_CONV_STATE_FWD=1 must be BIT-IDENTICAL to =0.
// Graph = the llama recurrent conv shape (build_conv_state, n_rs_seq == 0): R = GET_ROWS(cache, idx) ->
// T = RESHAPE(R) [d_conv-1, ch, ns] -> C = CONCAT(T, transpose(x), 0) -> SSM_CONV(C, kernel), and
// CPY(view(C, last d_conv-1 columns), view(cache, kv_head)). Each case runs 3 consecutive steps on the
// same cache, so the write-back compounds. The knob is read once per process, so the parent
// re-executes itself as a child under each env arm and compares FNV-1a hashes of the conv output and
// the WHOLE cache after every step.
// Cases: identity / permuted index (must forward), an extra consumer of R (must NOT forward),
// ns 1 and 2, nt 1 and 3, threads 1, 4, 24.
// Usage: test-conv-state-fwd          (parent; rc 0 iff all hashes equal and forwarding engaged as expected)
//        test-conv-state-fwd --child  (prints "hash <case> <step> <out> <cache> fwd <n>")
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
    int64_t ns, nt, kv_head;
    int32_t idx[2];
    bool extra_reader;
    bool expect_fwd;
};

static const tcase CASES[] = {
    { "ident_ns1_nt1",       1, 1, 2, {2, 0}, false, true  },
    { "ident_ns2_nt1",       2, 1, 1, {1, 2}, false, true  },
    { "ident_ns2_nt3",       2, 3, 0, {0, 1}, false, true  },
    { "perm_ns2_nt1",        2, 1, 1, {2, 1}, false, true  },
    { "ident_ns1_nt1_extra", 1, 1, 2, {2, 0}, true,  false },
};

static int child() {
    const int64_t NC = 4, CH = 2048, N_RS = 4;
    const int threads[] = { 1, 4, 24 };
    for (const tcase & tc : CASES) {
        for (int th : threads) {
            ggml_backend_t be = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(be, th);
            ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * c = ggml_init(ip);
            const int64_t row = (NC - 1) * CH;
            ggml_tensor * cache = ggml_new_tensor_2d(c, GGML_TYPE_F32, row, N_RS);
            ggml_tensor * idx   = ggml_new_tensor_1d(c, GGML_TYPE_I32, tc.ns);
            ggml_tensor * x     = ggml_new_tensor_3d(c, GGML_TYPE_F32, CH, tc.nt, tc.ns);
            ggml_tensor * kern  = ggml_new_tensor_2d(c, GGML_TYPE_F32, NC, CH);
            ggml_tensor * R     = ggml_get_rows(c, cache, idx);
            ggml_tensor * T     = ggml_reshape_3d(c, R, NC - 1, CH, tc.ns);
            ggml_tensor * C     = ggml_concat(c, T, ggml_transpose(c, x), 0);
            ggml_tensor * conv  = ggml_ssm_conv(c, C, kern);
            ggml_tensor * last  = ggml_view_3d(c, C, NC - 1, CH, tc.ns, C->nb[1], C->nb[2], ggml_row_size(C->type, tc.nt));
            ggml_tensor * dstv  = ggml_view_2d(c, cache, row, tc.ns, cache->nb[1], tc.kv_head * cache->nb[1]);
            ggml_cgraph * gf = ggml_new_graph(c);
            ggml_build_forward_expand(gf, ggml_cpy(c, last, dstv));
            ggml_tensor * out = ggml_scale(c, conv, 1.0f);
            ggml_build_forward_expand(gf, out);
            ggml_tensor * side = nullptr;
            if (tc.extra_reader) {
                side = ggml_scale(c, R, 3.0f);
                ggml_build_forward_expand(gf, side);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);

            uint64_t sx = 0x9E3779B97F4A7C15ull;
            auto rnd = [&]() { sx ^= sx << 13; sx ^= sx >> 7; sx ^= sx << 17; return (float) ((int64_t) (sx % 2001) - 1000) / 1000.0f; };
            auto fill = [&](ggml_tensor * t, float scale) {
                std::vector<float> d((size_t) ggml_nelements(t));
                for (auto & v : d) v = rnd() * scale;
                ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
            };
            fill(cache, 0.5f);
            fill(kern, 0.3f);
            ggml_backend_tensor_set(idx, tc.idx, 0, tc.ns * sizeof(int32_t));

            for (int step = 0; step < 3; step++) {
                fill(x, 1.0f);
                const long long f0 = ggml_cpu_conv_state_fwd_count();
                if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "compute failed\n");
                    return 2;
                }
                const long long f1 = ggml_cpu_conv_state_fwd_count();
                std::vector<uint8_t> a(ggml_nbytes(out)), cc(ggml_nbytes(cache));
                ggml_backend_tensor_get(out, a.data(), 0, a.size());
                ggml_backend_tensor_get(cache, cc.data(), 0, cc.size());
                uint64_t ha = fnv(a.data(), a.size());
                if (side) {
                    std::vector<uint8_t> so(ggml_nbytes(side));
                    ggml_backend_tensor_get(side, so.data(), 0, so.size());
                    ha = fnv(so.data(), so.size(), ha);
                }
                printf("hash %s/t%d %d %016" PRIx64 " %016" PRIx64 " fwd %lld\n", tc.name, th, step,
                       ha, fnv(cc.data(), cc.size()), f1 - f0);
            }
            ggml_backend_buffer_free(buf);
            ggml_free(c);
            ggml_backend_free(be);
        }
    }
    return 0;
}

static bool run_arm(const char * self, const char * val, std::map<std::string, std::string> & hashes,
                    std::map<std::string, long long> & fwd) {
    std::string cmd = std::string("GGML_CONV_STATE_FWD=") + val + " GGML_CONV_STATE_FWD_QUIET=1 '" + self + "' --child";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) return false;
    char line[512];
    while (fgets(line, sizeof line, p)) {
        char name[128]; int step; char ha[32], hc[32]; long long f;
        if (sscanf(line, "hash %127s %d %31s %31s fwd %lld", name, &step, ha, hc, &f) == 5) {
            const std::string key = std::string(name) + "#" + std::to_string(step);
            hashes[key] = std::string(ha) + ":" + hc;
            fwd[name] += f;
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
        if (kv.second != 0) { fprintf(stderr, "FAIL: %s forwarded with knob off\n", kv.first.c_str()); bad++; }
    }
    for (const tcase & tc : CASES) {
        for (int th : { 1, 4, 24 }) {
            const std::string name = std::string(tc.name) + "/t" + std::to_string(th);
            const long long n = f1[name];
            if (tc.expect_fwd ? n != 3 : n != 0) {
                fprintf(stderr, "FAIL: %s forwarded %lld times with knob on (expected %d)\n", name.c_str(), n, tc.expect_fwd ? 3 : 0);
                bad++;
            }
        }
    }
    printf("test-conv-state-fwd: %zu hashes compared, %d failures; CONV_STATE_FWD engaged on sole-reader cases\n", h0.size(), bad);
    return bad == 0 ? 0 : 1;
}
