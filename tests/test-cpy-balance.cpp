// C11-CPU-SSM-CPY: correctness test for the balanced row-copy split (default ON; forced via GGML_CPY_BALANCE=1,
// ops.cpp ggml_cpy_rows_balanced, dup_bytes / dup_flt same-shape row paths).
//
// Builds CPY graphs whose views mirror the recurrent-state snapshot write in
// src/models/delta-net-base.cpp: src = view_3d(gdn_out, D, n_seqs, n_written) with snapshot
// strides, dst = view_3d(ssm_states_all, D, n_seqs, n_written) at a kv_head offset, so neither is
// contiguous and ne01 (= n_seqs) is far below nth. Requirements, all byte-exact:
//   (1) every thread count and iteration equals the 1-thread result, INCLUDING the bytes of the
//       destination buffer outside the view (sentinel must survive: no over-write);
//   (2) the GGML_CPY_BALANCE=1 output equals GGML_CPY_BALANCE=0 (child processes, the knob is
//       read once per process). Exit 0 on pass.

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

struct ccase { ggml_type type; int64_t D, n_seqs, n_written, mem_size, kv_head; };

static uint64_t fnv(const void * p, size_t n, uint64_t h) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t run_case(const ccase & c, int iters, bool * ok) {
    ggml_init_params ip = { (size_t) 512 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t K = c.n_written + 1;
    // gdn_out-like source: [attn scores | K snapshots of D*n_seqs]
    const int64_t attn = 4096;
    const int64_t snap = c.D * c.n_seqs;
    ggml_tensor * gdn = ggml_new_tensor_1d(ctx, c.type, attn + snap * K);
    {
        std::mt19937 rng(77 + (int) c.D);
        uint8_t * p = (uint8_t *) gdn->data;
        for (size_t i = 0; i < ggml_nbytes(gdn); i++) p[i] = (uint8_t) rng();
    }
    ggml_tensor * states = ggml_new_tensor_2d(ctx, c.type, c.D, c.mem_size * K);

    ggml_tensor * src = ggml_view_3d(ctx, gdn, c.D, c.n_seqs, c.n_written,
        ggml_row_size(c.type, c.D), ggml_row_size(c.type, snap), ggml_row_size(c.type, attn));
    ggml_tensor * dst = ggml_view_3d(ctx, states, c.D, c.n_seqs, c.n_written,
        states->nb[1], (size_t) c.mem_size * states->nb[1], (size_t) c.kv_head * states->nb[1]);
    ggml_tensor * out = ggml_cpy(ctx, src, dst);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    const size_t nb = ggml_nbytes(states);
    memset(states->data, 0x5A, nb);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<uint8_t> ref((uint8_t *) states->data, (uint8_t *) states->data + nb);

    const int nths[] = { 2, 5, 7, 13, 48 };
    for (int nth : nths) {
        for (int it = 0; it < iters; it++) {
            memset(states->data, 0x5A, nb);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            if (memcmp(ref.data(), states->data, nb) != 0) {
                fprintf(stderr, "FAIL type=%s D=%lld n_seqs=%lld nw=%lld nth=%d iter=%d\n",
                        ggml_type_name(c.type), (long long) c.D, (long long) c.n_seqs,
                        (long long) c.n_written, nth, it);
                *ok = false;
            }
        }
    }
    const uint64_t h = fnv(ref.data(), nb, 1469598103934665603ull);
    ggml_free(ctx);
    return h;
}

static std::vector<ccase> cases() {
    std::vector<ccase> v;
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F16 }) {
        v.push_back({ t, 786432, 1, 5, 3, 1 });  // Q38FN snapshot write shape (MTP d4: 5 written)
        v.push_back({ t, 786432, 1, 1, 2, 0 });  // single snapshot
        v.push_back({ t, 100003, 3, 4, 4, 2 });  // odd row bytes, n_seqs > 1
        v.push_back({ t, 1000,   2, 3, 3, 1 });  // below the min-bytes threshold
    }
    return v;
}

static int child(void) {
    bool ok = true;
    for (const ccase & c : cases()) {
        printf("%016llx\n", (unsigned long long) run_case(c, 2, &ok));
    }
    fflush(stdout);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {
        return child();
    }
    std::string out_on, out_off;
    int rc_on = 0, rc_off = 0;
    for (int arm = 0; arm < 2; arm++) {
        std::string cmd = std::string(arm ? "GGML_CPY_BALANCE=1" : "GGML_CPY_BALANCE=0") +
            " /proc/" + std::to_string((long long) getpid()) + "/exe --child";
        FILE * p = popen(cmd.c_str(), "r");
        if (!p) { fprintf(stderr, "popen failed\n"); return 1; }
        std::string & o = arm ? out_on : out_off;
        char buf[256];
        while (fgets(buf, sizeof buf, p)) o += buf;
        (arm ? rc_on : rc_off) = pclose(p);
    }
    const size_t n = cases().size();
    printf("test-cpy-balance: %zu cases; CPY_BALANCE=0 rc=%d, CPY_BALANCE=1 rc=%d\n", n, rc_off, rc_on);
    if (rc_off != 0 || rc_on != 0) { printf("FAIL: thread-count identity\n"); return 1; }
    if (out_on.empty() || out_on != out_off ||
        (size_t) std::count(out_on.begin(), out_on.end(), '\n') != n) {
        printf("FAIL: GGML_CPY_BALANCE=1 output differs from GGML_CPY_BALANCE=0\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
