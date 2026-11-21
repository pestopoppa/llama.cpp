// test-gdn-replay-identity.cpp: [RS-REPLAY] identity of GDN input-log replay (llama.cpp PR #31669,
// head 93252b852acc) against snapshot rollback, on the CPU backend.
//
// Snapshot arm (what the fork serves today): run T speculative tokens with K = T snapshot slots,
// accept a of them, restore the state after a tokens (slot T-a, or s0 for a == 0), then run the
// next token x with K = 1.
// Replay arm: log the T tokens' F32 k/v/gate/beta, then run token x from s0 with n_rep = a.
// The final state and token x's attention output must be BITWISE identical (F32 log, same order).
// A keep = true arm checks that the state output is exactly the state after the replay.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct dims {
    int64_t S, H_k, H, T, B;
    bool    kda;
};

static void fill(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        d[i] = u(rng);
    }
}

static void unit_rows(ggml_tensor * t) {
    float * d = (float *) t->data;
    const int64_t S = t->ne[0];
    for (int64_t r = 0; r < ggml_nelements(t) / S; r++) {
        float n2 = 0.0f;
        for (int64_t j = 0; j < S; j++) { n2 += d[r*S + j]*d[r*S + j]; }
        for (int64_t j = 0; j < S; j++) { d[r*S + j] /= sqrtf(n2); }
    }
}

// copy token range [t0, t0+n) of a [a, b, T, B] tensor into a [a, b, n, B] tensor
static void copy_tokens(ggml_tensor * dst, const ggml_tensor * src, int64_t t0, int64_t n) {
    const size_t row = src->nb[2];
    for (int64_t s = 0; s < src->ne[3]; s++) {
        for (int64_t t = 0; t < n; t++) {
            memcpy((char *) dst->data + s*dst->nb[3] + t*dst->nb[2], (const char *) src->data + s*src->nb[3] + (t0 + t)*src->nb[2], row);
        }
    }
}

static void compute(ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
}

static int run_case(const dims & d) {
    ggml_init_params ip = { /*.mem_size =*/ 2048ull * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(0x5eed + (int) d.S + 7 * (int) d.T + (d.kda ? 1000 : 0));

    const int64_t S = d.S, H_k = d.H_k, H = d.H, T = d.T, B = d.B;
    const int64_t g0 = d.kda ? S : 1;

    // the T speculative tokens and the next token x
    ggml_tensor * qT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, B);
    ggml_tensor * kT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, B);
    ggml_tensor * vT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H,   T, B);
    ggml_tensor * gT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g0, H,  T, B);
    ggml_tensor * bT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H,   T, B);
    ggml_tensor * qx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, 1, B);
    ggml_tensor * kx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, 1, B);
    ggml_tensor * vx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H,   1, B);
    ggml_tensor * gx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g0, H,  1, B);
    ggml_tensor * bx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H,   1, B);
    ggml_tensor * s0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, B);
    fill(qT, rng, -1, 1); unit_rows(qT); fill(kT, rng, -1, 1); unit_rows(kT);
    fill(qx, rng, -1, 1); unit_rows(qx); fill(kx, rng, -1, 1); unit_rows(kx);
    fill(vT, rng, -0.3f, 5.0f); fill(vx, rng, -0.3f, 5.0f);
    fill(gT, rng, -3.0f, -1e-4f); fill(gx, rng, -3.0f, -1e-4f);
    fill(bT, rng, 0.0f, 1.0f); fill(bx, rng, 0.0f, 1.0f);
    fill(s0, rng, -0.5f, 0.5f);

    // pending log: k [S,H_k,T,B], v [S,H,T,B], gb [g0,H,2T,B] (gates, then betas in element 0)
    ggml_tensor * pk  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, B);
    ggml_tensor * pv  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H,   T, B);
    ggml_tensor * pgb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g0, H, 2*T, B);
    memcpy(pk->data, kT->data, ggml_nbytes(kT));
    memcpy(pv->data, vT->data, ggml_nbytes(vT));
    memset(pgb->data, 0, ggml_nbytes(pgb));
    for (int64_t s = 0; s < B; s++) {
        for (int64_t t = 0; t < T; t++) {
            memcpy((char *) pgb->data + s*pgb->nb[3] + t*pgb->nb[2], (char *) gT->data + s*gT->nb[3] + t*gT->nb[2], gT->nb[2]);
            for (int64_t h = 0; h < H; h++) {
                ((float *) ((char *) pgb->data + s*pgb->nb[3] + (T + t)*pgb->nb[2]))[h*g0] =
                    ((float *) ((char *) bT->data + s*bT->nb[3] + t*bT->nb[2]))[h];
            }
        }
    }

    // snapshot arm, step 1: K = T slots
    ggml_tensor * snap = ggml_gated_delta_net(ctx, qT, kT, vT, gT, bT, s0, T);
    compute(ctx, snap);
    const size_t state_bytes = (size_t) S*S*H*B*sizeof(float);
    const size_t attn_T      = (size_t) S*H*T*B*sizeof(float);
    const size_t attn_1      = (size_t) S*H*1*B*sizeof(float);

    int failures = 0;
    for (int64_t a = 0; a <= T; a++) {
        // restored state after a accepted tokens
        ggml_tensor * sa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, B);
        if (a == 0) {
            memcpy(sa->data, s0->data, state_bytes);
        } else {
            memcpy(sa->data, (char *) snap->data + attn_T + (size_t) (T - a)*state_bytes, state_bytes);
        }
        ggml_tensor * ref = ggml_gated_delta_net(ctx, qx, kx, vx, gx, bx, sa, 1);
        compute(ctx, ref);

        ggml_tensor * n_rep = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B);
        for (int64_t s = 0; s < B; s++) { ((int32_t *) n_rep->data)[s] = (int32_t) a; }

        // replay arm, keep = false: final state after a replayed tokens + x
        ggml_tensor * rep = ggml_gated_delta_net_ext(ctx, qx, kx, vx, gx, bx, s0, pk, pv, pgb, n_rep, 1, false);
        compute(ctx, rep);
        const bool attn_eq  = memcmp(ref->data, rep->data, attn_1) == 0;
        const bool state_eq = memcmp((char *) ref->data + attn_1, (char *) rep->data + attn_1, state_bytes) == 0;

        // replay arm, keep = true (2 new tokens): the state output is the committed state after the replay
        bool keep_eq = true;
        if (T >= 2) {
            ggml_tensor * q2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, 2, B); copy_tokens(q2, qT, 0, 2);
            ggml_tensor * k2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, 2, B); copy_tokens(k2, kT, 0, 2);
            ggml_tensor * v2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H,   2, B); copy_tokens(v2, vT, 0, 2);
            ggml_tensor * g2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g0, H,  2, B); copy_tokens(g2, gT, 0, 2);
            ggml_tensor * b2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H,   2, B); copy_tokens(b2, bT, 0, 2);
            ggml_tensor * repk = ggml_gated_delta_net_ext(ctx, q2, k2, v2, g2, b2, s0, pk, pv, pgb, n_rep, 1, true);
            compute(ctx, repk);
            const size_t attn_2 = (size_t) S*H*2*B*sizeof(float);
            keep_eq = memcmp((char *) repk->data + attn_2, sa->data, state_bytes) == 0;
        }

        printf("S=%lld H_k=%lld H=%lld T=%lld B=%lld kda=%d accept=%lld: attn %s, final state %s, keep-state %s\n",
               (long long) S, (long long) H_k, (long long) H, (long long) T, (long long) B, (int) d.kda, (long long) a,
               attn_eq ? "bit-identical" : "DIFFERS", state_eq ? "bit-identical" : "DIFFERS", keep_eq ? "bit-identical" : "DIFFERS");
        failures += !attn_eq + !state_eq + !keep_eq;
    }

    ggml_free(ctx);
    return failures;
}

int main() {
    int failures = 0;
    // 27B dims (H_k 16, H_v 48, S 128) at n_max 7 (T 8), np 1 and np 4
    failures += run_case({ 128, 16, 48, 8, 1, false });
    failures += run_case({ 128, 16, 48, 8, 4, false });
    // A3B dims (H_k 16, H_v 32) at n_max 3
    failures += run_case({ 128, 16, 32, 4, 2, false });
    // small KDA (vector gate) case
    failures += run_case({ 32, 4, 4, 4, 2, true });
    printf(failures == 0 ? "OK\n" : "FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
