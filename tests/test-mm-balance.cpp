// C11-CPU-MM-BALANCE: correctness test for the dynamic mul_mat_id partition
// (GGML_MM_BALANCE=1, iqk_dispatch.cpp ggml_iqk_try_mul_mat_id, batch 2..8 slab path).
//
// Builds MUL_MAT_ID graphs over IQ4_XS experts (and Q8_0 / Q4_K for coverage) at the MTP-verify
// batch sizes 1..8 with heavily overlapping expert choices (so cne1 ranges 1..n_tokens), both the
// broadcast-activation (up/gate, ne11 = 1) and the per-slot-activation (down, ne11 = n_used) form.
// Requirements, all byte-exact:
//   (1) with GGML_MM_BALANCE=1 every thread count and every iteration equals the 1-thread result;
//   (2) the GGML_MM_BALANCE=1 output equals the GGML_MM_BALANCE=0 output (child process, since the
//       knob is read once per process) -- i.e. same accumulation order per output element.
// Runs with GGML_IQK=1 forced. Exit 0 on pass.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <unistd.h>

struct mcase { ggml_type type; int64_t ne00, ne01, n_as, n_used, n_tok; bool bcast; };

static uint64_t fnv(const void * p, size_t n, uint64_t h) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// returns FNV hash of the output; 0 on failure. iters x thread counts are checked against nth=1.
static uint64_t run_case(const mcase & c, int iters, bool * ok) {
    ggml_init_params ip = { (size_t) 1024 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    std::mt19937 rng(1234 + (int) c.type * 7 + (int) c.n_tok);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    ggml_tensor * as = ggml_new_tensor_3d(ctx, c.type, c.ne00, c.ne01, c.n_as);
    {
        std::vector<float> f(c.ne00 * c.ne01 * c.n_as);
        for (auto & x : f) x = nd(rng);
        ggml_quantize_chunk(c.type, f.data(), as->data, 0, c.ne01 * c.n_as, c.ne00, nullptr);
    }
    const int64_t ne11 = c.bcast ? 1 : c.n_used;
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.ne00, ne11, c.n_tok);
    for (int64_t i = 0; i < ggml_nelements(b); i++) ((float *) b->data)[i] = nd(rng);

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n_tok);
    // experts drawn from a small hot pool -> many experts shared by several tokens (cne1 > 1)
    const int64_t pool = c.n_used + 6 < c.n_as ? c.n_used + 6 : c.n_as;
    for (int64_t t = 0; t < c.n_tok; t++) {
        std::vector<int32_t> perm(pool);
        for (int64_t i = 0; i < pool; i++) perm[i] = (int32_t) ((i * 37 + 5) % c.n_as);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t k = 0; k < c.n_used; k++) ((int32_t *) ids->data)[t * c.n_used + k] = perm[k];
    }

    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    const size_t nb = ggml_nbytes(out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<uint8_t> ref((uint8_t *) out->data, (uint8_t *) out->data + nb);

    const int nths[] = { 2, 7, 13, 48 };
    for (int nth : nths) {
        for (int it = 0; it < iters; it++) {
            memset(out->data, 0xA5, nb);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            if (memcmp(ref.data(), out->data, nb) != 0) {
                fprintf(stderr, "FAIL type=%s n_tok=%lld bcast=%d nth=%d iter=%d: differs from nth=1\n",
                        ggml_type_name(c.type), (long long) c.n_tok, (int) c.bcast, nth, it);
                *ok = false;
            }
        }
    }
    const uint64_t h = fnv(ref.data(), nb, 1469598103934665603ull);
    ggml_free(ctx);
    return h;
}

static std::vector<mcase> cases() {
    std::vector<mcase> v;
    const ggml_type types[] = { GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K };
    for (ggml_type t : types) {
        for (int64_t n_tok = 1; n_tok <= 8; n_tok++) {
            v.push_back({ t, 512, 640, 64, 10, n_tok, true  });  // up/gate shape (Q38FN expert ff = 640)
            v.push_back({ t, 640, 256, 64, 10, n_tok, false });  // down shape (per-slot activations)
        }
    }
    return v;
}

static int child(void) {
    bool ok = true;
    for (const mcase & c : cases()) {
        const uint64_t h = run_case(c, 3, &ok);
        printf("%016llx\n", (unsigned long long) h);
    }
    fflush(stdout);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    setenv("GGML_IQK", "1", 0);
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {
        return child();
    }

    std::string out_on, out_off;
    int rc_on = 0, rc_off = 0;
    for (int arm = 0; arm < 2; arm++) {
        std::string cmd = std::string(arm ? "GGML_MM_BALANCE=1" : "GGML_MM_BALANCE=0") +
            " GGML_IQK=1 /proc/" + std::to_string((long long) getpid()) + "/exe --child";
        FILE * p = popen(cmd.c_str(), "r");
        if (!p) { fprintf(stderr, "popen failed\n"); return 1; }
        std::string & o = arm ? out_on : out_off;
        char buf[256];
        while (fgets(buf, sizeof buf, p)) o += buf;
        const int st = pclose(p);
        (arm ? rc_on : rc_off) = st;
    }
    const size_t n = cases().size();
    printf("test-mm-balance: %zu cases; MM_BALANCE=0 rc=%d, MM_BALANCE=1 rc=%d, lines off=%zu on=%zu\n",
           n, rc_off, rc_on,
           (size_t) std::count(out_off.begin(), out_off.end(), '\n'),
           (size_t) std::count(out_on.begin(), out_on.end(), '\n'));
    if (rc_off != 0 || rc_on != 0) { printf("FAIL: thread-count identity\n"); return 1; }
    if (out_on.empty() || out_on != out_off ||
        (size_t) std::count(out_on.begin(), out_on.end(), '\n') != n) {
        printf("FAIL: GGML_MM_BALANCE=1 output differs from GGML_MM_BALANCE=0\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
