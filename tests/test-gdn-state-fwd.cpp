// test-gdn-state-fwd (AK-CPU-GDN-STATE-FWD): GGML_GDN_STATE_FWD=1 must be BIT-IDENTICAL to =0.
// Graph = the llama recurrent-layer shape: GET_ROWS(cache, idx) -> reshape -> GATED_DELTA_NET (K = 1)
// -> CPY(view(result, state), view(cache, kv_head)) plus a consumer of the attention view. Each case
// runs 3 consecutive steps on the same cache, so in-place updates compound.
// The knob is read once per process, so the parent re-executes itself as a child under each env arm
// and compares FNV-1a hashes of the attention output and the WHOLE cache after every step.
// Cases: identity index (n_seqs 1 and 2, must forward), permuted / non-identity index (must NOT forward,
// stock copy path), n_tokens 1 and 3, scalar and per-channel (KDA) gate, threads 1, 4, 24.
// Usage: test-gdn-state-fwd          (parent; rc 0 iff all hashes equal and forwarding engaged as expected)
//        test-gdn-state-fwd --child  (prints "hash <case> <step> <attn> <cache> fwd <n>")
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

struct tcase {
    const char * name;
    int64_t ns, nt, kv_head;
    int32_t idx[2];
    bool kda;
    bool expect_fwd;
};

static const tcase CASES[] = {
    { "ident_ns1_nt1",      1, 1, 2, {2, 0}, false, true  },
    { "ident_ns2_nt1",      2, 1, 1, {1, 2}, false, true  },
    { "ident_ns2_nt3",      2, 3, 0, {0, 1}, false, true  },
    { "ident_ns1_nt1_kda",  1, 1, 3, {3, 0}, true,  true  },
    { "perm_ns2_nt1",       2, 1, 1, {2, 1}, false, false },
    { "nonid_ns2_nt1",      2, 1, 1, {0, 3}, false, false },
    { "nonid_ns1_nt3_kda",  1, 3, 1, {2, 0}, true,  false },
};

static int child() {
    const int64_t S = 64, H = 4, N_RS = 4;
    const int threads[] = { 1, 4, 24 };
    for (const tcase & tc : CASES) {
        for (int th : threads) {
            ggml_backend_t be = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(be, th);
            ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * c = ggml_init(ip);
            const int64_t row = S * S * H;
            ggml_tensor * cache = ggml_new_tensor_2d(c, GGML_TYPE_F32, row, N_RS);
            ggml_tensor * idx   = ggml_new_tensor_1d(c, GGML_TYPE_I32, tc.ns);
            ggml_tensor * q     = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, H, tc.nt, tc.ns);
            ggml_tensor * k     = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, H, tc.nt, tc.ns);
            ggml_tensor * v     = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, H, tc.nt, tc.ns);
            ggml_tensor * g     = ggml_new_tensor_4d(c, GGML_TYPE_F32, tc.kda ? S : 1, H, tc.nt, tc.ns);
            ggml_tensor * beta  = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, H, tc.nt, tc.ns);
            ggml_tensor * R     = ggml_get_rows(c, cache, idx);
            ggml_tensor * T     = ggml_reshape_4d(c, R, S, S, H, tc.ns);
            ggml_tensor * res   = ggml_gated_delta_net(c, q, k, v, g, beta, T, 1);
            const int64_t attn_elems = S * H * tc.nt * tc.ns;
            ggml_tensor * attn  = ggml_view_1d(c, res, attn_elems, 0);
            ggml_tensor * st    = ggml_view_1d(c, res, row * tc.ns, attn_elems * sizeof(float));
            ggml_tensor * dstv  = ggml_view_2d(c, cache, row, tc.ns, cache->nb[1], tc.kv_head * cache->nb[1]);
            ggml_tensor * C     = ggml_cpy(c, st, dstv);
            ggml_tensor * out   = ggml_scale(c, attn, 1.0f);
            ggml_cgraph * gf = ggml_new_graph(c);
            ggml_build_forward_expand(gf, C);
            ggml_build_forward_expand(gf, out);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);

            uint64_t sx = 0x9E3779B97F4A7C15ull;
            auto rnd = [&]() { sx ^= sx << 13; sx ^= sx >> 7; sx ^= sx << 17; return (float) ((int64_t) (sx % 2001) - 1000) / 1000.0f; };
            auto fill = [&](ggml_tensor * t, float scale, float bias) {
                std::vector<float> d((size_t) ggml_nelements(t));
                for (auto & x : d) x = rnd() * scale + bias;
                ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
            };
            fill(cache, 0.1f, 0.0f);
            ggml_backend_tensor_set(idx, tc.idx, 0, tc.ns * sizeof(int32_t));

            for (int step = 0; step < 3; step++) {
                fill(q, 0.2f, 0.0f); fill(k, 0.2f, 0.0f); fill(v, 1.0f, 0.0f);
                fill(g, 0.1f, -0.3f); fill(beta, 0.3f, 0.5f);
                const long long f0 = ggml_cpu_gdn_state_fwd_count();
                if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "compute failed\n");
                    return 2;
                }
                const long long f1 = ggml_cpu_gdn_state_fwd_count();
                std::vector<uint8_t> a(ggml_nbytes(out)), cc(ggml_nbytes(cache));
                ggml_backend_tensor_get(out, a.data(), 0, a.size());
                ggml_backend_tensor_get(cache, cc.data(), 0, cc.size());
                printf("hash %s/t%d %d %016" PRIx64 " %016" PRIx64 " fwd %lld\n", tc.name, th, step,
                       fnv(a.data(), a.size()), fnv(cc.data(), cc.size()), f1 - f0);
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
    std::string cmd = std::string("GGML_GDN_STATE_FWD=") + val + " GGML_GDN_STATE_FWD_QUIET=1 '" + self + "' --child";
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
    printf("test-gdn-state-fwd: %zu hashes compared, %d failures; GDN_STATE_FWD engaged on identity cases\n", h0.size(), bad);
    return bad == 0 ? 0 : 1;
}
