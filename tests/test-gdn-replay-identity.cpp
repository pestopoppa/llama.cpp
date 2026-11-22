// test-gdn-replay-identity.cpp: [RS-REPLAY] identity of GDN input-log replay (llama.cpp PR #31669,
// head 93252b852acc) against snapshot rollback, on the CPU backend and (GDN_REPLAY_BACKEND=<name>, e.g. ROCm0)
// on a device backend, so the device replay loop is checked against the device snapshot path bit for bit.
//
// Snapshot arm (what the fork serves today): run T speculative tokens with K = T snapshot slots,
// accept a of them, restore the state after a tokens (slot T-a, or s0 for a == 0), then run the
// next X tokens with K = 1.
// Replay arm: log the T tokens' F32 k/v/gate/beta, then run the X tokens from s0 with n_rep = a.
// The final state and the X tokens' attention output must be BITWISE identical (F32 log, same order).
// A keep = true arm checks that the state output is exactly the state after the replay.
// X = 1 is decode/verify; X = 2048 exercises the device replay-only launch before the chunked prefill kernel.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

struct dims {
    int64_t S, H_k, H, T, B, X;
    bool    kda;
};

// host tensor = shape + data
struct ht {
    int64_t ne[4];
    ggml_type type;
    std::vector<char> d;
    ht() : ne{0, 0, 0, 0}, type(GGML_TYPE_F32) {}
    ht(int64_t a, int64_t b, int64_t c, int64_t e, ggml_type t = GGML_TYPE_F32) : ne{a, b, c, e}, type(t) {
        d.resize((size_t) a*b*c*e*ggml_type_size(t));
    }
    float * f() { return (float *) d.data(); }
    const float * f() const { return (const float *) d.data(); }
    size_t n() const { return (size_t) ne[0]*ne[1]*ne[2]*ne[3]; }
};

static ggml_backend_t g_backend = nullptr;

using op_fn = std::function<ggml_tensor *(ggml_context *, std::vector<ggml_tensor *> &)>;

// compute op(inputs...) on g_backend; returns the full output bytes
static std::vector<char> run(const std::vector<ht *> & in, const op_fn & op) {
    ggml_init_params ip = { /*.mem_size =*/ 64*ggml_tensor_overhead() + ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> t;
    for (ht * h : in) {
        t.push_back(ggml_new_tensor_4d(ctx, h->type, h->ne[0], h->ne[1], h->ne[2], h->ne[3]));
    }
    ggml_tensor * out = op(ctx, t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    for (size_t i = 0; i < in.size(); i++) {
        ggml_backend_tensor_set(t[i], in[i]->d.data(), 0, in[i]->d.size());
    }
    ggml_backend_graph_compute(g_backend, gf);
    std::vector<char> res(ggml_nbytes(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static void fill(ht & t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    for (size_t i = 0; i < t.n(); i++) { t.f()[i] = u(rng); }
}

static void unit_rows(ht & t) {
    const int64_t S = t.ne[0];
    for (size_t r = 0; r < t.n() / S; r++) {
        float n2 = 0.0f;
        for (int64_t j = 0; j < S; j++) { n2 += t.f()[r*S + j]*t.f()[r*S + j]; }
        for (int64_t j = 0; j < S; j++) { t.f()[r*S + j] /= sqrtf(n2); }
    }
}

// tokens [t0, t0+n) of a [a, b, T, B] host tensor
static ht tokens(const ht & src, int64_t t0, int64_t n) {
    ht dst(src.ne[0], src.ne[1], n, src.ne[3]);
    const size_t row = (size_t) src.ne[0]*src.ne[1]*sizeof(float);
    for (int64_t s = 0; s < src.ne[3]; s++) {
        for (int64_t t = 0; t < n; t++) {
            memcpy(dst.d.data() + (s*n + t)*row, src.d.data() + (s*src.ne[2] + t0 + t)*row, row);
        }
    }
    return dst;
}

static op_fn gdn(int64_t K) {
    return [K](ggml_context * c, std::vector<ggml_tensor *> & t) { return ggml_gated_delta_net(c, t[0], t[1], t[2], t[3], t[4], t[5], K); };
}

static op_fn gdn_ext(bool keep) {
    return [keep](ggml_context * c, std::vector<ggml_tensor *> & t) {
        return ggml_gated_delta_net_ext(c, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], 1, keep);
    };
}

static int run_case(const dims & d) {
    std::mt19937 rng(0x5eed + (int) d.S + 7 * (int) d.T + (d.kda ? 1000 : 0) + (int) d.X);
    const int64_t S = d.S, H_k = d.H_k, H = d.H, T = d.T, B = d.B, X = d.X;
    const int64_t g0 = d.kda ? S : 1;

    ht qT(S, H_k, T, B), kT(S, H_k, T, B), vT(S, H, T, B), gT(g0, H, T, B), bT(1, H, T, B);
    ht qx(S, H_k, X, B), kx(S, H_k, X, B), vx(S, H, X, B), gx(g0, H, X, B), bx(1, H, X, B);
    ht s0(S, S, H, B);
    fill(qT, rng, -1, 1); unit_rows(qT); fill(kT, rng, -1, 1); unit_rows(kT);
    fill(qx, rng, -1, 1); unit_rows(qx); fill(kx, rng, -1, 1); unit_rows(kx);
    fill(vT, rng, -0.3f, 5.0f); fill(vx, rng, -0.3f, 5.0f);
    fill(gT, rng, -3.0f, -1e-4f); fill(gx, rng, -3.0f, -1e-4f);
    fill(bT, rng, 0.0f, 1.0f); fill(bx, rng, 0.0f, 1.0f);
    fill(s0, rng, -0.5f, 0.5f);

    // pending log: k [S,H_k,T,B], v [S,H,T,B], gb [g0,H,2T,B] (gates, then betas in element 0)
    ht pk = kT, pv = vT, pgb(g0, H, 2*T, B);
    std::fill(pgb.d.begin(), pgb.d.end(), 0);
    for (int64_t s = 0; s < B; s++) {
        for (int64_t t = 0; t < T; t++) {
            memcpy(pgb.f() + (s*2*T + t)*g0*H, gT.f() + (s*T + t)*g0*H, g0*H*sizeof(float));
            for (int64_t h = 0; h < H; h++) {
                pgb.f()[(s*2*T + T + t)*g0*H + h*g0] = bT.f()[(s*T + t)*H + h];
            }
        }
    }

    // snapshot arm, step 1: K = T slots
    const std::vector<char> snap = run({ &qT, &kT, &vT, &gT, &bT, &s0 }, gdn(T));
    const size_t state_bytes = (size_t) S*S*H*B*sizeof(float);
    const size_t attn_T      = (size_t) S*H*T*B*sizeof(float);
    const size_t attn_X      = (size_t) S*H*X*B*sizeof(float);

    int failures = 0;
    for (int64_t a = 0; a <= T; a++) {
        ht sa(S, S, H, B);
        memcpy(sa.d.data(), a == 0 ? s0.d.data() : snap.data() + attn_T + (size_t) (T - a)*state_bytes, state_bytes);
        const std::vector<char> ref = run({ &qx, &kx, &vx, &gx, &bx, &sa }, gdn(1));

        ht n_rep(B, 1, 1, 1, GGML_TYPE_I32);
        for (int64_t s = 0; s < B; s++) { ((int32_t *) n_rep.d.data())[s] = (int32_t) a; }

        const std::vector<char> rep = run({ &qx, &kx, &vx, &gx, &bx, &s0, &pk, &pv, &pgb, &n_rep }, gdn_ext(false));
        const bool attn_eq  = memcmp(ref.data(), rep.data(), attn_X) == 0;
        const bool state_eq = memcmp(ref.data() + attn_X, rep.data() + attn_X, state_bytes) == 0;

        bool keep_eq = true;
        if (T >= 2 && X == 1) {
            ht q2 = tokens(qT, 0, 2), k2 = tokens(kT, 0, 2), v2 = tokens(vT, 0, 2), g2 = tokens(gT, 0, 2), b2 = tokens(bT, 0, 2);
            const std::vector<char> repk = run({ &q2, &k2, &v2, &g2, &b2, &s0, &pk, &pv, &pgb, &n_rep }, gdn_ext(true));
            const size_t attn_2 = (size_t) S*H*2*B*sizeof(float);
            keep_eq = memcmp(repk.data() + attn_2, sa.d.data(), state_bytes) == 0;
        }

        printf("[%s] S=%lld H_k=%lld H=%lld T=%lld B=%lld X=%lld kda=%d accept=%lld: attn %s, final state %s, keep-state %s\n",
               ggml_backend_name(g_backend), (long long) S, (long long) H_k, (long long) H, (long long) T, (long long) B, (long long) X,
               (int) d.kda, (long long) a, attn_eq ? "bit-identical" : "DIFFERS", state_eq ? "bit-identical" : "DIFFERS",
               keep_eq ? "bit-identical" : "DIFFERS");
        failures += !attn_eq + !state_eq + !keep_eq;
    }
    return failures;
}

int main() {
    ggml_backend_load_all();
    const char * bname = getenv("GDN_REPLAY_BACKEND");
    g_backend = bname && bname[0] ? ggml_backend_init_by_name(bname, nullptr) : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (g_backend == nullptr) {
        fprintf(stderr, "backend %s not available\n", bname ? bname : "CPU");
        return 1;
    }
    int failures = 0;
    // 27B dims (H_k 16, H_v 48, S 128) at n_max 7 (T 8) and n_max 15 (T 16), np 1 and np 4
    failures += run_case({ 128, 16, 48,  8, 1, 1, false });
    failures += run_case({ 128, 16, 48,  8, 4, 1, false });
    failures += run_case({ 128, 16, 48, 16, 4, 1, false });
    // A3B dims (H_k 16, H_v 32) at n_max 3
    failures += run_case({ 128, 16, 32,  4, 2, 1, false });
    // small KDA (vector gate) case
    failures += run_case({ 32, 4, 4, 4, 2, 1, true });
    // prefill-sized new ubatch after a replay (device: replay-only launch, then the chunked kernel)
    failures += run_case({ 128, 16, 48,  4, 1, 2048, false });
    printf(failures == 0 ? "OK\n" : "FAIL (%d)\n", failures);
    ggml_backend_free(g_backend);
    return failures == 0 ? 0 : 1;
}
