// LSA-H12-INDEXER unit test (CPU only, no model):
//   1. LSAMAP1 loader: the released 27B PCA map (argv[1] or LLAMA_LSA_H12_TEST_MAP) loads with geometry 24/4/256/64, 16 layers,
//      payload sha256 verified; skipped (not failed) when no map is given -- a synthetic map is used for 2/3.
//   2. selection: llama_lsa_h12::build_scores + TOP_K on the ggml CPU backend select exactly the cells of a scalar double
//      reference of the LSA scorer math (sum the GQA group over the non-RoPE dims, Mq / Mk projection, L2 norm, dot,
//      window/sink bonuses, causal mask), up to cells within 1e-4 of the selection boundary (reported).
//   3. all keys: when the budget covers every visible cell, the composed (indexer-masked) FA equals the dense FA bit for bit.
#include "llama-lsa-h12.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++n_fail; } } while (0)

static constexpr int HS = 256, NH = 24, NKV = 4, ROPE = 64, IDX = 64, G = NH/NKV, W = HS - ROPE;

struct fixture {
    int64_t kv, nb, n_used, top_k, window, sink;
    std::vector<float> q;      // [HS, NH, nb]
    std::vector<float> k;      // [HS, NKV, kv]   (f16-rounded)
    std::vector<float> v;      // [HS, NKV, kv]
    std::vector<float> mask;   // [kv, nb]  0 / -inf
    std::vector<float> kpos, kbias, qwin;
    int64_t width() const { return std::min<int64_t>(kv, top_k + window + sink); }
    int64_t pos(int64_t i) const { return n_used - nb + i; }
};

static fixture make_fixture(int64_t kv, int64_t nb, int64_t n_used, int64_t top_k, int64_t window, int64_t sink, uint32_t seed) {
    fixture f{kv, nb, n_used, top_k, window, sink, {}, {}, {}, {}, {}, {}, {}};
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    f.q.resize((size_t) HS*NH*nb);
    f.k.resize((size_t) HS*NKV*kv);
    f.v.resize((size_t) HS*NKV*kv);
    for (auto & x : f.q) x = nd(rng);
    for (auto & x : f.k) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(nd(rng)));
    for (auto & x : f.v) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(nd(rng)));
    f.mask.resize((size_t) kv*nb);
    f.qwin.resize(nb);
    for (int64_t i = 0; i < nb; ++i) {
        for (int64_t j = 0; j < kv; ++j) {
            f.mask[i*kv + j] = j < n_used && j <= f.pos(i) ? 0.0f : -INFINITY;
        }
        f.qwin[i] = (float) window - (float) f.pos(i);
    }
    f.kpos.resize(kv);
    f.kbias.resize(kv);
    for (int64_t j = 0; j < kv; ++j) {
        f.kpos[j]  = j < n_used ? (float) j : llama_lsa_h12::empty_pos;
        f.kbias[j] = j < n_used && j < sink ? llama_lsa_h12::sink_bonus : 0.0f;
    }
    return f;
}

// scalar double reference of the scorer (the native LSA projection formula, without its BF16 rounding)
static std::vector<double> ref_scores(const fixture & f, const std::vector<float> & mq, const std::vector<float> & mk) {
    const int64_t kv = f.kv, nb = f.nb;
    std::vector<double> khat((size_t) kv*IDX), qhat((size_t) nb*IDX), out((size_t) kv*nb);
    auto norm = [](double * x) {
        double s = 0; for (int o = 0; o < IDX; ++o) s += x[o]*x[o];
        s = std::max(std::sqrt(s), 1e-12);
        for (int o = 0; o < IDX; ++o) x[o] /= s;
    };
    for (int64_t j = 0; j < kv; ++j) {
        double * kh = &khat[j*IDX];
        for (int o = 0; o < IDX; ++o) {
            double acc = 0;
            for (int h = 0; h < NKV; ++h) for (int d = 0; d < W; ++d) {
                acc += (double) f.k[(j*NKV + h)*HS + ROPE + d] * mk[((size_t) h*W + d)*IDX + o];
            }
            kh[o] = acc;
        }
        norm(kh);
    }
    for (int64_t t = 0; t < nb; ++t) {
        std::vector<double> in((size_t) NKV*W, 0.0);
        for (int h = 0; h < NH; ++h) for (int d = 0; d < W; ++d) {
            in[(h/G)*W + d] += f.q[(t*NH + h)*HS + ROPE + d];
        }
        double * qh = &qhat[t*IDX];
        for (int o = 0; o < IDX; ++o) {
            double acc = 0;
            for (int r = 0; r < NKV*W; ++r) acc += in[r]*mq[(size_t) r*IDX + o];
            qh[o] = acc;
        }
        norm(qh);
    }
    for (int64_t t = 0; t < nb; ++t) for (int64_t j = 0; j < kv; ++j) {
        double s = 0;
        for (int o = 0; o < IDX; ++o) s += qhat[t*IDX + o]*khat[j*IDX + o];
        if (f.kpos[j] + f.qwin[t] > 0) s += llama_lsa_h12::window_bonus;
        s += f.kbias[j];
        s += f.mask[t*kv + j];
        out[t*kv + j] = s;
    }
    return out;
}

struct graph_out {
    std::vector<int32_t> top_k;   // [width, nb]
    std::vector<float>   fa_lsa;  // [HS, NH, nb]
    std::vector<float>   fa_dense;
};

static graph_out run_graph(const fixture & f, const std::vector<float> & mq_full, const std::vector<float> & mk_full, bool with_fa) {
    ggml_init_params ip = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t kv = f.kv, nb = f.nb, kv_size = kv + 64;

    ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HS, NH, nb);
    ggml_tensor * kc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HS, NKV, kv_size, 1);
    ggml_tensor * vc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HS, NKV, kv_size, 1);
    ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nb, 1, 1);
    ggml_tensor * mq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HS*NH, IDX);
    ggml_tensor * mk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HS*NKV, IDX);
    ggml_tensor * kpos  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kv, 1, 1);
    ggml_tensor * kbias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kv, 1, 1);
    ggml_tensor * qwin  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, nb, 1);

    ggml_tensor * k = ggml_view_4d(ctx, kc, HS, NKV, kv, 1, kc->nb[1], kc->nb[2], kc->nb[3], 0);
    ggml_tensor * v = ggml_view_4d(ctx, vc, HS, NKV, kv, 1, vc->nb[1], vc->nb[2], vc->nb[3], 0);

    ggml_tensor * score = llama_lsa_h12::build_scores(ctx, q, k, mq, mk, m, kpos, kbias, qwin);
    const int64_t w = f.width();
    ggml_tensor * tk = ggml_cont(ctx, ggml_top_k(ctx, score, w));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, tk);

    ggml_tensor * fa_lsa = nullptr, * fa_dense = nullptr;
    if (with_fa) {
        ggml_tensor * tk4 = ggml_reshape_4d(ctx, tk, w, nb, 1, 1);
        ggml_tensor * all = ggml_fill(ctx, m, -INFINITY);
        all = ggml_view_4d(ctx, all, 1, all->ne[0], all->ne[1], all->ne[3], all->nb[0], all->nb[1], all->nb[2], 0);
        ggml_tensor * tk3 = ggml_view_4d(ctx, tk4, tk4->ne[0], tk4->ne[1], tk4->ne[3], 1, tk4->nb[1], tk4->nb[2], tk4->ne[3]*tk4->nb[3], 0);
        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, tk3->ne[0], tk3->ne[1], tk3->ne[2]), 0.0f);
        ggml_tensor * mm = ggml_set_rows(ctx, all, zeros, tk3);
        mm = ggml_view_4d(ctx, mm, mm->ne[1], mm->ne[2], 1, mm->ne[3], mm->nb[2], mm->nb[3], mm->nb[3], 0);
        mm = ggml_add(ctx, mm, m);
        ggml_tensor * qp = ggml_permute(ctx, ggml_reshape_4d(ctx, q, HS, NH, nb, 1), 0, 2, 1, 3);
        ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
        ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
        fa_lsa   = ggml_flash_attn_ext(ctx, qp, kp, vp, mm, 1.0f/16.0f, 0.0f, 0.0f);
        fa_dense = ggml_flash_attn_ext(ctx, qp, kp, vp, m,  1.0f/16.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa_lsa,   GGML_PREC_F32);
        ggml_flash_attn_ext_set_prec(fa_dense, GGML_PREC_F32);
        ggml_build_forward_expand(gf, fa_lsa);
        ggml_build_forward_expand(gf, fa_dense);
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 8);
    // every tensor of the context, intermediates included (small graphs)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);

    ggml_backend_tensor_set(q, f.q.data(), 0, ggml_nbytes(q));
    std::vector<ggml_fp16_t> h((size_t) HS*NKV*kv_size, ggml_fp32_to_fp16(0.0f));
    for (size_t i = 0; i < f.k.size(); ++i) h[i] = ggml_fp32_to_fp16(f.k[i]);
    ggml_backend_tensor_set(kc, h.data(), 0, ggml_nbytes(kc));
    for (size_t i = 0; i < f.v.size(); ++i) h[i] = ggml_fp32_to_fp16(f.v[i]);
    ggml_backend_tensor_set(vc, h.data(), 0, ggml_nbytes(vc));
    std::vector<ggml_fp16_t> mh(f.mask.size());
    for (size_t i = 0; i < mh.size(); ++i) mh[i] = ggml_fp32_to_fp16(f.mask[i]);
    ggml_backend_tensor_set(m, mh.data(), 0, ggml_nbytes(m));
    ggml_backend_tensor_set(mq, mq_full.data(), 0, ggml_nbytes(mq));
    ggml_backend_tensor_set(mk, mk_full.data(), 0, ggml_nbytes(mk));
    ggml_backend_tensor_set(kpos,  f.kpos.data(),  0, ggml_nbytes(kpos));
    ggml_backend_tensor_set(kbias, f.kbias.data(), 0, ggml_nbytes(kbias));
    ggml_backend_tensor_set(qwin,  f.qwin.data(),  0, ggml_nbytes(qwin));

    GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);

    graph_out r;
    r.top_k.resize(ggml_nelements(tk));
    ggml_backend_tensor_get(tk, r.top_k.data(), 0, ggml_nbytes(tk));
    if (with_fa) {
        r.fa_lsa.resize(ggml_nelements(fa_lsa));
        r.fa_dense.resize(ggml_nelements(fa_dense));
        ggml_backend_tensor_get(fa_lsa,   r.fa_lsa.data(),   0, ggml_nbytes(fa_lsa));
        ggml_backend_tensor_get(fa_dense, r.fa_dense.data(), 0, ggml_nbytes(fa_dense));
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return r;
}

static void test_selection(const char * label, const fixture & f, const llama_lsa_h12::lsamap & m, int layer) {
    std::vector<float> mq_full, mk_full;
    llama_lsa_h12::expand_q(m, m.layers.at(layer).first,  mq_full);
    llama_lsa_h12::expand_k(m, m.layers.at(layer).second, mk_full);

    const graph_out r = run_graph(f, mq_full, mk_full, false);
    const std::vector<double> ref = ref_scores(f, m.layers.at(layer).first, m.layers.at(layer).second);

    const int64_t w = f.width();
    int64_t n_mismatch = 0, n_tie = 0, n_checked = 0;
    for (int64_t t = 0; t < f.nb; ++t) {
        std::vector<std::pair<double, int64_t>> row(f.kv);
        for (int64_t j = 0; j < f.kv; ++j) row[j] = { ref[t*f.kv + j], j };
        std::sort(row.begin(), row.end(), [](auto & a, auto & b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        const double boundary = row[w - 1].first;
        const bool finite_boundary = std::isfinite(boundary);
        std::set<int64_t> got(r.top_k.begin() + t*w, r.top_k.begin() + (t + 1)*w);
        CHECK((int64_t) got.size() == w, "%s: token %lld: top_k has duplicate indices", label, (long long) t);
        for (int64_t i = 0; i < f.kv; ++i) {
            const int64_t j = row[i].second;
            const double s = row[i].first;
            if (!std::isfinite(s)) {
                continue; // -inf picks (budget > visible) are interchangeable; the attention masks them anyway
            }
            ++n_checked;
            const bool want = i < w;
            if (want != (got.count(j) > 0)) {
                if (finite_boundary && std::fabs(s - boundary) < 1e-4) {
                    ++n_tie;
                } else {
                    ++n_mismatch;
                }
            }
        }
    }
    printf("%-34s kv %6lld nb %3lld width %5lld: checked %lld cells, mismatches %lld, boundary ties (<1e-4) %lld\n",
            label, (long long) f.kv, (long long) f.nb, (long long) w, (long long) n_checked, (long long) n_mismatch, (long long) n_tie);
    CHECK(n_mismatch == 0, "%s: %lld selected cells differ from the scalar reference", label, (long long) n_mismatch);
    CHECK(n_tie <= 2*f.nb, "%s: too many boundary ties (%lld)", label, (long long) n_tie);
}

static void test_all_keys(const fixture & f, const llama_lsa_h12::lsamap & m, int layer) {
    std::vector<float> mq_full, mk_full;
    llama_lsa_h12::expand_q(m, m.layers.at(layer).first,  mq_full);
    llama_lsa_h12::expand_k(m, m.layers.at(layer).second, mk_full);
    const graph_out r = run_graph(f, mq_full, mk_full, true);
    double max_abs = 0;
    for (size_t i = 0; i < r.fa_lsa.size(); ++i) {
        max_abs = std::max(max_abs, (double) std::fabs(r.fa_lsa[i] - r.fa_dense[i]));
    }
    printf("all-keys (n_used %lld <= width %lld)  max |FA_lsa - FA_dense| = %g\n", (long long) f.n_used, (long long) f.width(), max_abs);
    CHECK(max_abs == 0.0, "all-keys composed FA differs from dense FA (max abs %g)", max_abs);
}

static llama_lsa_h12::lsamap synthetic_map(uint32_t seed) {
    llama_lsa_h12::lsamap m;
    m.q_heads = NH; m.kv_heads = NKV; m.head_dim = HS; m.rope_dim = ROPE;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    auto & l = m.layers[3];
    l.first.resize((size_t) m.rows()*IDX);
    l.second.resize((size_t) m.rows()*IDX);
    for (auto & x : l.first)  x = nd(rng);
    for (auto & x : l.second) x = nd(rng);
    return m;
}

int main(int argc, char ** argv) {
    std::string path = argc > 1 ? argv[1] : (getenv("LLAMA_LSA_H12_TEST_MAP") ? getenv("LLAMA_LSA_H12_TEST_MAP") : "");

    llama_lsa_h12::lsamap m;
    int layer = 3;
    if (!path.empty()) {
        std::string err;
        const bool ok = llama_lsa_h12::load_lsamap(path, m, err);
        CHECK(ok, "load %s: %s", path.c_str(), err.c_str());
        if (!ok) {
            return 1;
        }
        printf("map %s: model '%s' ident '%s' q%d kv%d hd%d rope%d layers %zu payload %s source %s\n", path.c_str(),
                m.model_identity.c_str(), m.map_identity.c_str(), m.q_heads, m.kv_heads, m.head_dim, m.rope_dim, m.layers.size(),
                m.payload_sha256.c_str(), m.source_sha256.c_str());
        CHECK(m.q_heads == NH && m.kv_heads == NKV && m.head_dim == HS && m.rope_dim == ROPE, "map geometry");
        CHECK(m.layers.size() == 16, "map layer count %zu", m.layers.size());
        for (int il = 3; il < 64; il += 4) {
            CHECK(m.layers.count(il) == 1, "map lacks layer %d", il);
        }
        layer = m.layers.begin()->first;
    } else {
        printf("no map given (argv[1] / LLAMA_LSA_H12_TEST_MAP): loader check skipped, synthetic map\n");
        m = synthetic_map(7);
    }

    // decode / DFlash2 verify / prefill-shaped ubatches; budget well below the visible cells
    test_selection("decode nb=1",           make_fixture(4096,  1, 4000, 256, 128, 4, 11), m, layer);
    test_selection("verify nb=8",           make_fixture(4096,  8, 4000, 256, 128, 4, 12), m, layer);
    test_selection("verify nb=16",          make_fixture(8192, 16, 8100, 512, 128, 4, 13), m, layer);
    test_selection("prefill nb=64",         make_fixture(4096, 64, 4000, 256, 128, 4, 14), m, layer);
    test_selection("budget > visible nb=4", make_fixture(4096,  4,  300, 256, 128, 4, 15), m, layer);

    test_all_keys(make_fixture(2048, 8, 300, 256, 128, 4, 21), m, layer);

    if (n_fail) {
        printf("FAILED (%d)\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
