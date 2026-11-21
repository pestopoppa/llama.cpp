// [AK-DTREE-2] CPU exactness tests for the tree-verification ops.
//
//   1. a tree whose parents form a chain is bit-identical to the chain op (whole output);
//   2. for a branching tree, every root-to-leaf path is bit-identical to the chain op run on
//      that path alone (attention rows of the path's columns, and their state snapshots).
//
// Covers ggml_gated_delta_net_tree (scalar gate and KDA, GQA v_repeat) and ggml_ssm_conv_tree.
// Design reference: Infernix tree verification (Apache-2.0, Wallawalla47/Infernix@397f8cd).

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static std::mt19937 rng(1234);

static void fill(ggml_tensor * t, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        p[i] = d(rng);
    }
}

static ggml_context * new_ctx() {
    ggml_init_params ip = { 256u * 1024 * 1024, nullptr, false };
    return ggml_init(ip);
}

static void compute(ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
}

// root paths (anchor-excluded column lists) of a parent array; -1 = root
static std::vector<std::vector<int>> leaf_paths(const std::vector<int32_t> & par) {
    const int n = (int) par.size();
    std::vector<bool> has_child(n, false);
    for (int t = 0; t < n; ++t) {
        if (par[t] >= 0) {
            has_child[par[t]] = true;
        }
    }
    std::vector<std::vector<int>> out;
    for (int t = 0; t < n; ++t) {
        if (has_child[t]) {
            continue;
        }
        std::vector<int> path;
        for (int c = t; c >= 0; c = par[c]) {
            path.insert(path.begin(), c);
        }
        out.push_back(path);
    }
    return out;
}

static int n_fail = 0;

static void check(bool ok, const char * what) {
    printf("  %-72s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) {
        n_fail++;
    }
}

// ---------------------------------------------------------------- GDN

struct gdn_in {
    int64_t S, H, Hv, T;
    bool kda;
    std::vector<float> q, k, v, g, beta, s0;
};

static gdn_in gdn_make(int64_t S, int64_t H, int v_repeat, int64_t T, bool kda) {
    gdn_in in;
    in.S = S; in.H = H; in.Hv = H * v_repeat; in.T = T; in.kda = kda;
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    auto gen = [&](std::vector<float> & x, size_t n, float lo, float hi) {
        std::uniform_real_distribution<float> dd(lo, hi);
        x.resize(n);
        for (auto & e : x) {
            e = dd(rng);
        }
    };
    gen(in.q, S * H * T, -1.f, 1.f);
    gen(in.k, S * H * T, -1.f, 1.f);
    gen(in.v, S * in.Hv * T, -1.f, 1.f);
    gen(in.g, (kda ? S : 1) * in.Hv * T, -1.f, 0.f);
    gen(in.beta, in.Hv * T, 0.f, 1.f);
    gen(in.s0, S * S * in.Hv, -0.5f, 0.5f);
    (void) d;
    return in;
}

// run GDN over the columns `cols` of `in` (chain when parent == nullptr); returns raw dst
static std::vector<float> gdn_run(const gdn_in & in, const std::vector<int> & cols, const std::vector<int32_t> * parent) {
    const int64_t S = in.S, H = in.H, Hv = in.Hv, T = (int64_t) cols.size();
    const int64_t gs = in.kda ? S : 1;
    ggml_context * ctx = new_ctx();
    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, T, 1);
    ggml_tensor * g    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, gs, Hv, T, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, 1);
    ggml_tensor * st   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, 1);
    for (int64_t i = 0; i < T; ++i) {
        const int c = cols[i];
        memcpy((float *) q->data + i * S * H, in.q.data() + c * S * H, S * H * sizeof(float));
        memcpy((float *) k->data + i * S * H, in.k.data() + c * S * H, S * H * sizeof(float));
        memcpy((float *) v->data + i * S * Hv, in.v.data() + c * S * Hv, S * Hv * sizeof(float));
        memcpy((float *) g->data + i * gs * Hv, in.g.data() + c * gs * Hv, gs * Hv * sizeof(float));
        memcpy((float *) beta->data + i * Hv, in.beta.data() + c * Hv, Hv * sizeof(float));
    }
    memcpy(st->data, in.s0.data(), in.s0.size() * sizeof(float));
    ggml_tensor * out;
    if (parent) {
        ggml_tensor * par = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, T, 1);
        memcpy(par->data, parent->data(), T * sizeof(int32_t));
        out = ggml_gated_delta_net_tree(ctx, q, k, v, g, beta, st, par);
    } else {
        out = ggml_gated_delta_net(ctx, q, k, v, g, beta, st, T);
    }
    compute(ctx, out);
    std::vector<float> r((float *) out->data, (float *) out->data + ggml_nelements(out));
    // state input must be untouched (record mode)
    if (memcmp(st->data, in.s0.data(), in.s0.size() * sizeof(float)) != 0) {
        check(false, "GDN: initial state not modified");
    }
    ggml_free(ctx);
    return r;
}

static void test_gdn(int64_t S, int64_t H, int v_repeat, bool kda, const std::vector<int32_t> & par) {
    const int64_t T = (int64_t) par.size();
    gdn_in in = gdn_make(S, H, v_repeat, T, kda);
    const int64_t Hv = in.Hv;
    char buf[256];

    // 1. chain parents == chain op, bitwise
    std::vector<int32_t> chain(T);
    std::vector<int> all(T);
    for (int64_t t = 0; t < T; ++t) {
        chain[t] = (int32_t) t - 1;
        all[t] = (int) t;
    }
    {
        auto a = gdn_run(in, all, &chain);
        auto b = gdn_run(in, all, nullptr);
        snprintf(buf, sizeof(buf), "GDN chain-tree == chain  S=%lld H=%lld vr=%d kda=%d T=%lld",
                 (long long) S, (long long) H, v_repeat, kda, (long long) T);
        check(a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, buf);
    }

    // 2. every root path of the tree == chain on that path, bitwise
    auto tree = gdn_run(in, all, &par);
    const int64_t attn_elems = S * Hv * T;
    const int64_t snap = S * S * Hv;
    bool ok = true;
    for (const auto & path : leaf_paths(par)) {
        const int64_t L = (int64_t) path.size();
        auto ch = gdn_run(in, path, nullptr);
        for (int64_t i = 0; i < L; ++i) {
            const int c = path[i];
            ok &= memcmp(tree.data() + c * S * Hv, ch.data() + i * S * Hv, S * Hv * sizeof(float)) == 0;
            const float * ts = tree.data() + attn_elems + (T - 1 - c) * snap;
            const float * cs = ch.data() + S * Hv * L + (L - 1 - i) * snap;
            ok &= memcmp(ts, cs, snap * sizeof(float)) == 0;
        }
    }
    snprintf(buf, sizeof(buf), "GDN tree paths == chain  S=%lld H=%lld vr=%d kda=%d T=%lld",
             (long long) S, (long long) H, v_repeat, kda, (long long) T);
    check(ok, buf);
}

// ---------------------------------------------------------------- SSM_CONV

static std::vector<float> conv_run(const std::vector<float> & hist, const std::vector<float> & x, int64_t nc, int64_t nr,
                                   const std::vector<float> & w, const std::vector<int> & cols,
                                   const std::vector<int32_t> * parent) {
    const int64_t T = (int64_t) cols.size();
    ggml_context * ctx = new_ctx();
    ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nc - 1 + T, nr, 1);
    ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
    memcpy(c->data, w.data(), w.size() * sizeof(float));
    float * sd = (float *) sx->data;
    for (int64_t r = 0; r < nr; ++r) {
        for (int64_t h = 0; h < nc - 1; ++h) {
            sd[r * (nc - 1 + T) + h] = hist[r * (nc - 1) + h];
        }
        for (int64_t i = 0; i < T; ++i) {
            sd[r * (nc - 1 + T) + nc - 1 + i] = x[cols[i] * nr + r];
        }
    }
    ggml_tensor * out;
    if (parent) {
        ggml_tensor * par = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, T, 1);
        memcpy(par->data, parent->data(), T * sizeof(int32_t));
        out = ggml_ssm_conv_tree(ctx, sx, c, par);
    } else {
        out = ggml_ssm_conv(ctx, sx, c);
    }
    compute(ctx, out);
    std::vector<float> r((float *) out->data, (float *) out->data + ggml_nelements(out));
    ggml_free(ctx);
    return r;
}

static void test_conv(int64_t nc, int64_t nr, const std::vector<int32_t> & par) {
    const int64_t T = (int64_t) par.size();
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> hist(nr * (nc - 1)), x(T * nr), w(nc * nr);
    for (auto & e : hist) e = d(rng);
    for (auto & e : x) e = d(rng);
    for (auto & e : w) e = d(rng);
    char buf[256];

    std::vector<int32_t> chain(T);
    std::vector<int> all(T);
    for (int64_t t = 0; t < T; ++t) {
        chain[t] = (int32_t) t - 1;
        all[t] = (int) t;
    }
    {
        auto a = conv_run(hist, x, nc, nr, w, all, &chain);
        auto b = conv_run(hist, x, nc, nr, w, all, nullptr);
        { double md = 0; int mi = -1; for (size_t i = 0; i < a.size(); ++i) { double d = fabs(a[i]-b[i]); if (d > md) { md = d; mi = (int) i; } }
          if (md > 0) printf("    max |diff| %.3g at %d (%g vs %g)\n", md, mi, a[mi], b[mi]); }
        snprintf(buf, sizeof(buf), "SSM_CONV chain-tree == chain  nc=%lld nr=%lld T=%lld", (long long) nc, (long long) nr, (long long) T);
        check(memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, buf);
    }
    auto tree = conv_run(hist, x, nc, nr, w, all, &par);
    bool ok = true;
    for (const auto & path : leaf_paths(par)) {
        auto ch = conv_run(hist, x, nc, nr, w, path, nullptr);
        for (size_t i = 0; i < path.size(); ++i) {
            ok &= memcmp(tree.data() + path[i] * nr, ch.data() + i * nr, nr * sizeof(float)) == 0;
        }
    }
    snprintf(buf, sizeof(buf), "SSM_CONV tree paths == chain  nc=%lld nr=%lld T=%lld", (long long) nc, (long long) nr, (long long) T);
    check(ok, buf);

    // [DTREE-ROUND-OVERHEAD-2] fused tree conv: output bit-identical to ggml_ssm_conv_tree, and rollback slot
    // T-1-c holds column c's window minus its oldest tap ([nc-1, nr] per slot), the history aliased into slot 0
    {
        ggml_context * ctx = new_ctx();
        const int64_t nk = nc - 1;
        const int64_t slots = T + 2;   // slot stride > T, as in the cache (mem_size rows per slot)
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nk * nr, slots);
        memset(cache->data, 0, ggml_nbytes(cache));
        memcpy(cache->data, hist.data(), hist.size() * sizeof(float));   // history = slot 0 (aliased)
        ggml_tensor * cs = ggml_view_3d(ctx, cache, nk, nr, 1, nk * sizeof(float), cache->nb[1], 0);
        ggml_tensor * xs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nr, T, 1);
        memcpy(xs->data, x.data(), x.size() * sizeof(float));
        ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
        memcpy(c->data, w.data(), w.size() * sizeof(float));
        ggml_tensor * pr = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, T, 1);
        memcpy(pr->data, par.data(), T * sizeof(int32_t));
        ggml_tensor * sd = ggml_view_3d(ctx, cache, nk * nr, T, 1, cache->nb[1], cache->nb[1], 0);
        ggml_tensor * out = ggml_ssm_conv_tree_fused(ctx, cs, xs, c, pr, sd);
        compute(ctx, out);
        const bool same = memcmp(out->data, tree.data(), tree.size() * sizeof(float)) == 0;
        snprintf(buf, sizeof(buf), "SSM_CONV fused == tree  nc=%lld nr=%lld T=%lld", (long long) nc, (long long) nr, (long long) T);
        check(same, buf);
        // expected states: walk the window of each column over [hist | x]
        bool st_ok = true;
        const float * cd = (const float *) cache->data;
        for (int64_t col = 0; col < T; ++col) {
            std::vector<int> widx(nc);
            int pos = (int) (col + nk);
            widx[nc - 1] = pos;
            for (int j = (int) nc - 2; j >= 0; --j) {
                if (pos >= nk) { const int p = par[pos - nk]; pos = p >= 0 ? p + (int) nk : (int) nk - 1; } else { pos = pos - 1; }
                widx[j] = pos;
            }
            const int64_t slot = T - 1 - col;
            for (int64_t r = 0; r < nr; ++r) {
                for (int64_t j = 0; j < nk; ++j) {
                    const int q = widx[1 + j];
                    const float e = q < nk ? hist[r * nk + q] : x[(q - nk) * nr + r];
                    st_ok &= cd[slot * nk * nr + r * nk + j] == e;
                }
            }
        }
        snprintf(buf, sizeof(buf), "SSM_CONV fused states == windows  nc=%lld nr=%lld T=%lld", (long long) nc, (long long) nr, (long long) T);
        check(st_ok, buf);
        ggml_free(ctx);
    }
}

int main() {
    ggml_cpu_init();
    // trees: main chain in columns 0..K, side columns after it (Infernix layout rule), plus a
    // bushy tree with root siblings and a deep side branch.
    const std::vector<int32_t> t_small = { -1, 0, 1, 2, 0, 4, 1 };                       // K=3 + 3 side
    const std::vector<int32_t> t_bushy = { -1, 0, 1, 2, 3, 4, 5, -1, 7, 1, 9, 10, 3, 2, 13, 4 }; // 16 cols
    const std::vector<int32_t> t_roots = { -1, -1, -1, 0, 1, 2 };

    for (const auto * par : { &t_small, &t_bushy, &t_roots }) {
        test_gdn(128, 4, 1, false, *par);
        test_gdn(64, 2, 2, false, *par);
        test_gdn(32, 2, 1, true, *par);
        test_conv(4, 64, *par);
        test_conv(4, 3, *par);
    }
    printf("%s (%d failures)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
