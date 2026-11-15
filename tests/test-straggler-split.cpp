// C11-CPU-FATOPK-SPLIT: correctness test for GGML_STRAGGLER_SPLIT (ops.cpp ggml_compute_forward_flash_attn_ext_dsplit):
// a FLASH_ATTN_EXT with fewer q rows than threads is split over the V/output dimension. Requirements, all byte-exact:
//   (1) inside each arm, every thread count equals the 1-thread result (the 1-thread run never splits);
//   (2) the GGML_STRAGGLER_SPLIT=2 output equals GGML_STRAGGLER_SPLIT=0 (child processes, the knob is read once per
//       process), and the =2 child must report that the split path was actually taken (not a vacuous identity).
// Shapes: the Q38FN MTP d4 draft decode (D=256, 2 KV heads x GQA 12, 1 query), few-head variants that give ns = 4,
// D=128 with 2 queries, sinks, ALiBi + softcap, -INF mask runs, and an FP16-overflow row whose overflow sits in ONE
// half of DV only (the hybrid FP32 recompute must be decided per row, not per slice). Exit 0 on pass.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

struct fcase {
    int64_t D, n_head_kv, gqa, n_q, n_kv;
    bool    sinks;
    float   max_bias, softcap;
    bool    overflow_half;   // V channels [0, D/2) large and same-sign -> FP16 VKQ overflow in slice 0 only
    int     neginf_prefix;   // leading masked cells (exercises the aligned -INF run skip)
};

static uint64_t fnv(const void * p, size_t n, uint64_t h) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t run_case(const fcase & c, bool * ok) {
    ggml_init_params ip = { (size_t) 1024 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t n_head = c.n_head_kv * c.gqa;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D, c.n_q, n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.D, c.n_kv, c.n_head_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.D, c.n_kv, c.n_head_kv, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_q, 1, 1);

    std::mt19937 rng(1234 + (int) c.D + (int) c.n_kv);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int64_t i = 0; i < ggml_nelements(q); i++) ((float *) q->data)[i] = c.overflow_half ? 0.0f : u(rng);
    for (int64_t i = 0; i < ggml_nelements(k); i++) ((ggml_fp16_t *) k->data)[i] = ggml_fp32_to_fp16(u(rng));
    for (int64_t i = 0; i < ggml_nelements(v); i++) {
        const int64_t d = i % c.D;
        const float x = (c.overflow_half && d < c.D/2) ? 40.0f + u(rng) : u(rng);
        ((ggml_fp16_t *) v->data)[i] = ggml_fp32_to_fp16(x);
    }
    for (int64_t r = 0; r < m->ne[1]; r++) {
        for (int64_t i = 0; i < c.n_kv; i++) {
            float x = i < c.neginf_prefix ? -INFINITY : 0.0f;
            if (x == 0.0f && (i % 97) == 13) x = -INFINITY;       // scattered masked cells
            if (i > c.n_kv - c.n_q + r) x = -INFINITY;              // causal tail
            ((ggml_fp16_t *) m->data)[r*c.n_kv + i] = ggml_fp32_to_fp16(x);
        }
    }

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/std::sqrt((float) c.D), c.max_bias, c.softcap);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (c.sinks) {
        ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);
        for (int64_t i = 0; i < n_head; i++) ((float *) s->data)[i] = u(rng) * 2.0f;
        ggml_flash_attn_ext_add_sinks(out, s);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    const size_t nb = ggml_nbytes(out);
    memset(out->data, 0x5A, nb);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<uint8_t> ref((uint8_t *) out->data, (uint8_t *) out->data + nb);

    const int nths[] = { 2, 3, 16, 25, 48, 64, 96 };
    for (int nth : nths) {
        for (int it = 0; it < 2; it++) {
            memset(out->data, 0x5A, nb);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            if (memcmp(ref.data(), out->data, nb) != 0) {
                fprintf(stdout, "FAIL D=%lld hkv=%lld gqa=%lld nq=%lld nkv=%lld nth=%d iter=%d\n",
                        (long long) c.D, (long long) c.n_head_kv, (long long) c.gqa, (long long) c.n_q,
                        (long long) c.n_kv, nth, it);
                *ok = false;
            }
        }
    }
    const uint64_t h = fnv(ref.data(), nb, 1469598103934665603ull);
    ggml_free(ctx);
    return h;
}

static std::vector<fcase> cases() {
    return {
        { 256, 2, 12, 1, 3000,  false, 0.0f, 0.0f,  false, 0   },  // Q38FN MTP draft decode (24 rows -> ns 2 at 48)
        { 256, 2, 12, 1, 8192,  false, 0.0f, 0.0f,  false, 640 },  // longer, leading -INF runs
        { 256, 2, 12, 2, 2048,  false, 0.0f, 0.0f,  false, 0   },  // 48 rows: split only at nth >= 96
        { 256, 1, 4,  1, 4096,  false, 0.0f, 0.0f,  false, 0   },  // 4 rows -> ns 4 at nth >= 16
        { 256, 2, 12, 1, 2500,  true,  0.0f, 0.0f,  false, 0   },  // sinks
        { 128, 2, 6,  2, 1500,  false, 8.0f, 30.0f, false, 0   },  // D=128, ALiBi + softcap, 24 rows
        { 256, 2, 12, 1, 4096,  false, 0.0f, 0.0f,  true,  0   },  // FP16 overflow in slice 0 only
        { 256, 1, 2,  1, 4096,  true,  0.0f, 0.0f,  true,  0   },  // overflow + sinks, ns 4
    };
}

static int child(void) {
    bool ok = true;
    for (const fcase & c : cases()) {
        printf("H %016llx\n", (unsigned long long) run_case(c, &ok));
    }
    fflush(stdout);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {
        return child();
    }
    std::string hashes[2], notes[2];
    int rc[2] = { 0, 0 };
    for (int arm = 0; arm < 2; arm++) {
        std::string cmd = std::string(arm ? "GGML_STRAGGLER_SPLIT=2" : "GGML_STRAGGLER_SPLIT=0") +
            " GGML_FA_SPLIT_KV=0 /proc/" + std::to_string((long long) getpid()) + "/exe --child 2>&1";
        FILE * p = popen(cmd.c_str(), "r");
        if (!p) { fprintf(stderr, "popen failed\n"); return 1; }
        char buf[512];
        while (fgets(buf, sizeof buf, p)) {
            if (strncmp(buf, "H ", 2) == 0) {
                hashes[arm] += buf;
            } else {
                notes[arm] += buf;
                fputs(buf, stdout);
            }
        }
        rc[arm] = pclose(p);
    }
    const size_t n = cases().size();
    printf("test-straggler-split: %zu cases; STRAGGLER_SPLIT=0 rc=%d, STRAGGLER_SPLIT=2 rc=%d\n", n, rc[0], rc[1]);
    if (rc[0] != 0 || rc[1] != 0) { printf("FAIL: thread-count identity\n"); return 1; }
    if (notes[1].find("fa dsplit taken") == std::string::npos) {
        printf("FAIL: split path never taken with GGML_STRAGGLER_SPLIT=2 (vacuous identity)\n");
        return 1;
    }
    if (notes[0].find("fa dsplit taken") != std::string::npos) {
        printf("FAIL: split path taken with the knob off\n");
        return 1;
    }
    if (hashes[1].empty() || hashes[1] != hashes[0] ||
        (size_t) std::count(hashes[1].begin(), hashes[1].end(), '\n') != n) {
        printf("FAIL: GGML_STRAGGLER_SPLIT=2 output differs from GGML_STRAGGLER_SPLIT=0\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
