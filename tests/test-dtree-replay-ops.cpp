// test-dtree-replay-ops.cpp: [DTREE-REPLAY] op-level identity of tree verify + GDN input-log replay against tree
// verify + snapshot slots, on the CPU backend and (GDN_REPLAY_BACKEND=<name>, e.g. ROCm0) on a device backend.
//
// Snapshot arm (--draft-tree-mode batch today): ggml_gated_delta_net_tree over W columns from s0 writes slot W-1-c =
// state after column c; tree_accept(path to leaf c) commits that slot, and the next X tokens run with K = 1 from it.
// Replay arm (this change): the W columns' F32 k/v/gate/beta are logged as pending rows in COLUMN order;
// tree_accept(path) keeps the path; the next ubatch gathers the path's rows (ggml_get_rows, in path order, the graph
// the model builds) and the unchanged chain replay runs them from s0 (n_rep = path length) before the X tokens.
// The X tokens' attention output and the final state must be BITWISE identical, for every accepted path (every
// leaf choice c: side branches, partial paths and the full path; accept depths 1..W on the chain tree), with a
// different path per sequence when B > 1. keep = true must return exactly the accepted slot, and a tree walk started
// from that replayed state (the next tree verify) must equal the tree walk from the snapshot slot.

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
#include <string>
#include <vector>

struct ht {
    int64_t ne[4];
    ggml_type type;
    std::vector<char> d;
    ht() : ne{0, 0, 0, 0}, type(GGML_TYPE_F32) {}
    ht(int64_t a, int64_t b, int64_t c, int64_t e, ggml_type t = GGML_TYPE_F32) : ne{a, b, c, e}, type(t) {
        d.resize((size_t) a*b*c*e*ggml_type_size(t));
    }
    float * f() { return (float *) d.data(); }
    int32_t * i() { return (int32_t *) d.data(); }
    size_t n() const { return (size_t) ne[0]*ne[1]*ne[2]*ne[3]; }
};

static ggml_backend_t g_backend = nullptr;

using op_fn = std::function<ggml_tensor *(ggml_context *, std::vector<ggml_tensor *> &)>;

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

// the graph of the model: pending rows gathered in path order (get_rows), then the chain replay
static op_fn replay_gathered(bool keep) {
    return [keep](ggml_context * c, std::vector<ggml_tensor *> & t) {
        // t: q k v g b s0 pk pv pgb n_rep idx idx_gb
        const auto rows = [&](ggml_tensor * x, ggml_tensor * idx) {
            ggml_tensor * x3 = ggml_reshape_3d(c, x, x->ne[0]*x->ne[1], x->ne[2], x->ne[3]);
            return ggml_reshape_4d(c, ggml_get_rows(c, x3, idx), x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
        };
        return ggml_gated_delta_net_ext(c, t[0], t[1], t[2], t[3], t[4], t[5],
                rows(t[6], t[10]), rows(t[7], t[10]), rows(t[8], t[11]), t[9], 1, keep);
    };
}

static int run_case(int64_t S, int64_t H_k, int64_t H, int64_t W, int64_t B, int64_t X, const char * shape,
                    const std::vector<int32_t> & par) {
    std::mt19937 rng(0x7ee + (int) W*31 + (int) B*7 + (int) X + (int) par.size()*1000 + (int) shape[0]);
    const int64_t g0 = 1;

    ht qT(S, H_k, W, B), kT(S, H_k, W, B), vT(S, H, W, B), gT(g0, H, W, B), bT(1, H, W, B);
    ht qx(S, H_k, X, B), kx(S, H_k, X, B), vx(S, H, X, B), gx(g0, H, X, B), bx(1, H, X, B);
    ht s0(S, S, H, B);
    fill(qT, rng, -1, 1); unit_rows(qT); fill(kT, rng, -1, 1); unit_rows(kT);
    fill(qx, rng, -1, 1); unit_rows(qx); fill(kx, rng, -1, 1); unit_rows(kx);
    fill(vT, rng, -0.3f, 5.0f); fill(vx, rng, -0.3f, 5.0f);
    fill(gT, rng, -3.0f, -1e-4f); fill(gx, rng, -3.0f, -1e-4f);
    fill(bT, rng, 0.0f, 1.0f); fill(bx, rng, 0.0f, 1.0f);
    fill(s0, rng, -0.5f, 0.5f);

    ht parent(W, B, 1, 1, GGML_TYPE_I32);
    for (int64_t s = 0; s < B; s++) {
        for (int64_t t = 0; t < W; t++) { parent.i()[s*W + t] = par[t]; }
    }

    // pending rows in COLUMN order: k, v, gb (gates then betas)
    ht pk = kT, pv = vT, pgb(g0, H, 2*W, B);
    for (int64_t s = 0; s < B; s++) {
        for (int64_t t = 0; t < W; t++) {
            memcpy(pgb.f() + (s*2*W + t)*g0*H, gT.f() + (s*W + t)*g0*H, g0*H*sizeof(float));
            for (int64_t h = 0; h < H; h++) {
                pgb.f()[(s*2*W + W + t)*g0*H + h] = bT.f()[(s*W + t)*H + h];
            }
        }
    }

    const size_t state_bytes = (size_t) S*S*H*sizeof(float); // one seq
    const size_t attn_W      = (size_t) S*H*W*B*sizeof(float);
    const size_t attn_X      = (size_t) S*H*X*B*sizeof(float);
    const size_t attn_Xs     = (size_t) S*H*X*sizeof(float);   // one seq

    // snapshot arm, step 1: the tree walk
    const std::vector<char> tree = run({ &qT, &kT, &vT, &gT, &bT, &s0, &parent },
            [](ggml_context * c, std::vector<ggml_tensor *> & t) { return ggml_gated_delta_net_tree(c, t[0], t[1], t[2], t[3], t[4], t[5], t[6]); });
    // slot j of seq s: attn_W + (j*B + s)*state_bytes
    const auto slot = [&](int64_t c, int64_t s) { return tree.data() + attn_W + ((size_t) (W - 1 - c)*B + s)*state_bytes; };

    int failures = 0, n_paths = 0;
    std::vector<int> depth_seen(W + 1, 0);
    for (int64_t c = 0; c < W; c++) {
        // seq s accepts the path to leaf (c + s) % W
        ht sa(S, S, H, B);
        ht n_rep(B, 1, 1, 1, GGML_TYPE_I32), idx(W, B, 1, 1, GGML_TYPE_I32), idx_gb(2*W, B, 1, 1, GGML_TYPE_I32);
        std::string paths;
        for (int64_t s = 0; s < B; s++) {
            const int64_t leaf = (c + s) % W;
            std::vector<int32_t> p;
            for (int32_t x = (int32_t) leaf; x >= 0; x = par[x]) { p.insert(p.begin(), x); }
            depth_seen[p.size()]++;
            memcpy(sa.d.data() + s*state_bytes, slot(leaf, s), state_bytes);
            n_rep.i()[s] = (int32_t) p.size();
            for (int64_t t = 0; t < W; t++) {
                const int32_t r = t < (int64_t) p.size() ? p[t] : (int32_t) t; // as set_input_rep: the rest identity
                idx.i()[s*W + t] = r;
                idx_gb.i()[s*2*W + t] = r;
                idx_gb.i()[s*2*W + W + t] = (int32_t) W + r;
            }
            if (s == 0) {
                for (int32_t x : p) { paths += std::to_string(x) + " "; }
            }
        }

        // snapshot arm, step 2: X tokens from the accepted slot
        const std::vector<char> ref = run({ &qx, &kx, &vx, &gx, &bx, &sa },
                [](ggml_context * cc, std::vector<ggml_tensor *> & t) { return ggml_gated_delta_net(cc, t[0], t[1], t[2], t[3], t[4], t[5], 1); });
        // replay arm: gathered path rows replayed from s0, then the X tokens
        const std::vector<char> rep = run({ &qx, &kx, &vx, &gx, &bx, &s0, &pk, &pv, &pgb, &n_rep, &idx, &idx_gb }, replay_gathered(false));

        bool attn_eq = memcmp(ref.data(), rep.data(), attn_X) == 0;
        bool state_eq = memcmp(ref.data() + attn_X, rep.data() + attn_X, state_bytes*B) == 0;
        (void) attn_Xs;

        // keep = true: the committed state is exactly the accepted slot (tree verify graph: root column only)
        ht q1(S, H_k, 1, B), k1(S, H_k, 1, B), v1(S, H, 1, B), g1(g0, H, 1, B), b1(1, H, 1, B);
        for (int64_t s = 0; s < B; s++) {
            memcpy(q1.f() + s*S*H_k, qx.f() + s*S*H_k*X, S*H_k*sizeof(float));
            memcpy(k1.f() + s*S*H_k, kx.f() + s*S*H_k*X, S*H_k*sizeof(float));
            memcpy(v1.f() + s*S*H,   vx.f() + s*S*H*X,   S*H*sizeof(float));
            memcpy(g1.f() + s*g0*H,  gx.f() + s*g0*H*X,  g0*H*sizeof(float));
            memcpy(b1.f() + s*H,     bx.f() + s*H*X,     H*sizeof(float));
        }
        const std::vector<char> repk = run({ &q1, &k1, &v1, &g1, &b1, &s0, &pk, &pv, &pgb, &n_rep, &idx, &idx_gb }, replay_gathered(true));
        const size_t attn_1 = (size_t) S*H*1*B*sizeof(float);
        bool keep_eq = memcmp(repk.data() + attn_1, sa.d.data(), state_bytes*B) == 0;

        // the next tree verify from the replayed state == from the snapshot slot (outputs and every slot)
        bool tree2_eq = true;
        if (c % 3 == 0) {
            ht sr(S, S, H, B);
            memcpy(sr.d.data(), repk.data() + attn_1, state_bytes*B);
            const auto tr = [](ggml_context * cc, std::vector<ggml_tensor *> & t) { return ggml_gated_delta_net_tree(cc, t[0], t[1], t[2], t[3], t[4], t[5], t[6]); };
            const std::vector<char> t_snap = run({ &qT, &kT, &vT, &gT, &bT, &sa, &parent }, tr);
            const std::vector<char> t_rep  = run({ &qT, &kT, &vT, &gT, &bT, &sr, &parent }, tr);
            tree2_eq = t_snap == t_rep;
        }

        printf("[%s] %s S=%lld H_k=%lld H=%lld W=%lld B=%lld X=%lld path(seq0)=[ %s]: attn %s, state %s, keep %s, next-tree %s\n",
               ggml_backend_name(g_backend), shape, (long long) S, (long long) H_k, (long long) H, (long long) W, (long long) B,
               (long long) X, paths.c_str(), attn_eq ? "==" : "DIFFERS", state_eq ? "==" : "DIFFERS", keep_eq ? "==" : "DIFFERS",
               tree2_eq ? "==" : "DIFFERS");
        failures += !attn_eq + !state_eq + !keep_eq + !tree2_eq;
        n_paths += (int) B;
    }
    printf("  %s W=%lld: %d paths, accept depths covered:", shape, (long long) W, n_paths);
    for (int64_t m = 1; m <= W; m++) { if (depth_seen[m]) printf(" %lld", (long long) m); }
    printf("\n");
    return failures;
}

static std::vector<int32_t> tree_chain(int W) { std::vector<int32_t> p(W); for (int t = 0; t < W; t++) p[t] = t - 1; return p; }

static std::vector<int32_t> tree_random(int W, unsigned seed) {
    std::mt19937 r(seed);
    std::vector<int32_t> p(W, -1);
    for (int t = 1; t < W; t++) {
        // mostly extend the previous column (a deep main chain), sometimes branch off an earlier one
        p[t] = (r() % 3 != 0) ? t - 1 : (int32_t) (r() % t);
    }
    return p;
}

static std::vector<int32_t> tree_star(int W) { std::vector<int32_t> p(W, 0); p[0] = -1; return p; }

// the test-dtree-batch layout: root, a 3-deep main chain, a side branch off depth 1, a junk sibling under the root
static std::vector<int32_t> tree_side(int W) {
    std::vector<int32_t> p = { -1, 0, 1, 2, 0, 4, 5, 0 };
    while ((int) p.size() < W) { p.push_back((int32_t) p.size() - 1); }
    p.resize(W);
    return p;
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
    // 27B dims: S 128, H_k 16, H_v 48; W 12 (G12 / auto) and 16 (T16); np 1 and np 4
    for (int W : { 12, 16 }) {
        failures += run_case(128, 16, 48, W, 1, 1, "chain",  tree_chain(W));
        failures += run_case(128, 16, 48, W, 1, 1, "side",   tree_side(W));
        failures += run_case(128, 16, 48, W, 4, 1, "random", tree_random(W, 17 + W));
        failures += run_case(128, 16, 48, W, 4, 1, "star",   tree_star(W));
    }
    // a multi-token ubatch after the replayed path (a prompt continuation / chain verify)
    failures += run_case(128, 16, 48, 12, 2, 8, "random", tree_random(12, 99));
    printf(failures == 0 ? "OK\n" : "FAIL (%d)\n", failures);
    ggml_backend_free(g_backend);
    return failures == 0 ? 0 : 1;
}
