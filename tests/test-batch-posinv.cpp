// test-batch-posinv.cpp: [GGML-DET-POSINV-TEST] batch-POSITION invariance of the per-row ops (intake-2843#03 LLM-42,
// intake-2842 VERGE). A sequence's output must not depend on WHERE it sits in a batch: at a fixed batch size M (the
// same ubatch shape), place the rows of src1 in a permuted order (reverse, rotate by one, seeded shuffle, swap first and
// last), un-permute the output and require it bitwise equal to the identity-order run. Ops: MUL_MAT (F32, Q8_0, Q4_K
// weights; the Q8_0 shape is the 27B-sized one the MI210 Q8_MF V5 / q8_skinny paths take), MUL_MAT_ID, FLASH_ATTN_EXT
// (query rows, mask rows permuted with them), RMS_NORM, GATED_DELTA_NET (sequences permuted). M in {2, 4, 8, 16}.
//
// Also reported, never asserted: batch-SIZE invariance (row r in an M-row batch vs the same row alone, M = 1). That is
// the property SPEC-ID-NOISE-1 saw break on near-ties; the column localizes it per op next to the position result.
//
// Split count: with GGML_DET_TRACE=1 the ROCm V5 path logs "q8mf6_split M= N= K= U= G=" per call; the test captures the
// log and requires one G per (M, N, K) over every permutation (and reports whether G also varies with M).
// Path witness: POSINV_EXPECT=v5det|v5|skinny|noskinny requires the matching ggml-cuda log line (exit 1 otherwise), so a
// run cannot pass on a fallback path it did not mean to test. Mutation (red): POSINV_MUTATE=1 flips the low mantissa bit
// of one output element of every permuted run; every asserted case must then FAIL. A built-in comparator self-test runs
// first in every mode.
//
// Env: POSINV_BACKEND (default CPU), POSINV_THREADS (CPU threads, default 8), POSINV_ONLY=<substring of case name>.
// Exit 0 = every asserted case position-invariant (or, with POSINV_MUTATE=1, every case caught); 1 = otherwise.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <vector>

static ggml_backend_t g_backend = nullptr;
static bool           g_mutate  = false;

// ---- log capture (path witnesses + split trace) ----
static std::mutex               g_log_mu;
static std::vector<std::string> g_log;
static void log_cb(ggml_log_level level, const char * text, void *) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_log.emplace_back(text);
    if (level >= GGML_LOG_LEVEL_WARN || getenv("POSINV_VERBOSE")) {
        fputs(text, stderr);
    }
}
static bool log_has(const char * s) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    for (auto & l : g_log) { if (l.find(s) != std::string::npos) { return true; } }
    return false;
}

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// ---- one case: a graph over n_units independent units; inputs are generated once, a permutation reorders them ----
struct pcase {
    std::string name;
    int         n_units;   // batch rows (tokens / query rows / sequences)
    size_t      out_unit;  // floats of output per unit
    // build + run with the units in order perm (slot j holds original unit perm[j]); returns dst floats, slot order
    std::function<std::vector<float>(const std::vector<int> & perm)> run;
};

// generic graph runner: build() creates the graph inside ctx and returns the output; set() uploads after allocation
static std::vector<float> run_graph(const std::function<ggml_tensor *(ggml_context *)> & build,
                                    const std::function<void()> & set, bool weights_usage = true) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * out = build(ctx);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    if (weights_usage) {
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    set();
    if (ggml_backend_graph_compute(g_backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(2);
    }
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static std::vector<float> rnd(size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) { x = d(rng); }
    return v;
}

// gather units of a [unit-major] buffer by perm
static std::vector<float> gather(const std::vector<float> & src, size_t unit, const std::vector<int> & perm) {
    std::vector<float> o(unit * perm.size());
    for (size_t j = 0; j < perm.size(); ++j) {
        memcpy(o.data() + j*unit, src.data() + (size_t) perm[j]*unit, unit*sizeof(float));
    }
    return o;
}

static std::vector<uint8_t> quantize(ggml_type t, const std::vector<float> & w, int64_t K, int64_t rows) {
    std::vector<uint8_t> q(ggml_row_size(t, K) * rows);
    ggml_quantize_chunk(t, w.data(), q.data(), 0, rows, K, nullptr);
    return q;
}

// ---- cases ----
static pcase case_mul_mat(ggml_type wt, int64_t K, int64_t N, int M) {
    auto W  = std::make_shared<std::vector<uint8_t>>();
    auto wf = rnd((size_t) K*N, 1234 + (uint32_t) K);
    if (wt == GGML_TYPE_F32) {
        W->resize(wf.size()*4); memcpy(W->data(), wf.data(), W->size());
    } else {
        *W = quantize(wt, wf, K, N);
    }
    auto X = std::make_shared<std::vector<float>>(rnd((size_t) K*M, 99 + M));
    pcase c;
    c.name = std::string("MUL_MAT ") + ggml_type_name(wt) + " K=" + std::to_string(K) + " N=" + std::to_string(N) + " M=" + std::to_string(M);
    c.n_units = M; c.out_unit = (size_t) N;
    c.run = [=](const std::vector<int> & perm) {
        ggml_tensor *w, *x;
        auto xp = gather(*X, (size_t) K, perm);
        return run_graph([&](ggml_context * ctx) {
            w = ggml_new_tensor_2d(ctx, wt, K, N);
            ggml_set_name(w, "blk.0.posinv.weight");
            x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, (int64_t) perm.size());
            return ggml_mul_mat(ctx, w, x);
        }, [&] {
            ggml_backend_tensor_set(w, W->data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(x, xp.data(), 0, ggml_nbytes(x));
        });
    };
    return c;
}

static pcase case_mul_mat_id(ggml_type wt, int64_t K, int64_t N, int E, int n_used, int M) {
    auto W = std::make_shared<std::vector<uint8_t>>(quantize(wt, rnd((size_t) K*N*E, 77), K, N*E));
    auto X = std::make_shared<std::vector<float>>(rnd((size_t) K*M, 5 + M));   // b broadcast over n_used: [K, 1, M]
    auto I = std::make_shared<std::vector<int32_t>>((size_t) n_used*M);
    std::mt19937 rng(17 + M);
    for (int t = 0; t < M; ++t) {
        std::vector<int> e(E); for (int i = 0; i < E; ++i) { e[i] = i; }
        std::shuffle(e.begin(), e.end(), rng);
        for (int u = 0; u < n_used; ++u) { (*I)[(size_t) t*n_used + u] = e[u]; }
    }
    pcase c;
    c.name = std::string("MUL_MAT_ID ") + ggml_type_name(wt) + " K=" + std::to_string(K) + " N=" + std::to_string(N) +
             " E=" + std::to_string(E) + " used=" + std::to_string(n_used) + " M=" + std::to_string(M);
    c.n_units = M; c.out_unit = (size_t) N*n_used;
    c.run = [=](const std::vector<int> & perm) {
        ggml_tensor *w, *x, *ids;
        auto xp = gather(*X, (size_t) K, perm);
        std::vector<int32_t> ip(I->size());
        for (size_t j = 0; j < perm.size(); ++j) {
            memcpy(ip.data() + j*n_used, I->data() + (size_t) perm[j]*n_used, n_used*sizeof(int32_t));
        }
        return run_graph([&](ggml_context * ctx) {
            w   = ggml_new_tensor_3d(ctx, wt, K, N, E);
            x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, (int64_t) perm.size());
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, (int64_t) perm.size());
            return ggml_mul_mat_id(ctx, w, x, ids);
        }, [&] {
            ggml_backend_tensor_set(w, W->data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(x, xp.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, ip.data(), 0, ggml_nbytes(ids));
        }, false);
    };
    return c;
}

static pcase case_rms_norm(int64_t K, int M) {
    auto X = std::make_shared<std::vector<float>>(rnd((size_t) K*M, 31 + M, -4.0f, 4.0f));
    pcase c;
    c.name = "RMS_NORM K=" + std::to_string(K) + " M=" + std::to_string(M);
    c.n_units = M; c.out_unit = (size_t) K;
    c.run = [=](const std::vector<int> & perm) {
        ggml_tensor * x;
        auto xp = gather(*X, (size_t) K, perm);
        return run_graph([&](ggml_context * ctx) {
            x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, (int64_t) perm.size());
            return ggml_rms_norm(ctx, x, 1e-6f);
        }, [&] { ggml_backend_tensor_set(x, xp.data(), 0, ggml_nbytes(x)); }, false);
    };
    return c;
}

// q [D, M, H], k/v F16 [D, KV, H_kv], mask F16 [KV, M] (random -inf holes, row j of the mask belongs to query row j);
// output [D, H, M]: unit = one query row (D*H floats)
static pcase case_fa(int64_t D, int64_t H, int64_t H_kv, int64_t KV, int M) {
    auto Q = std::make_shared<std::vector<float>>(rnd((size_t) D*H*M, 41 + M));   // unit-major staging: [M][H][D]
    auto Kf = rnd((size_t) D*KV*H_kv, 43), Vf = rnd((size_t) D*KV*H_kv, 47);
    auto Kh = std::make_shared<std::vector<ggml_fp16_t>>(Kf.size()), Vh = std::make_shared<std::vector<ggml_fp16_t>>(Vf.size());
    ggml_fp32_to_fp16_row(Kf.data(), Kh->data(), (int64_t) Kf.size());
    ggml_fp32_to_fp16_row(Vf.data(), Vh->data(), (int64_t) Vf.size());
    auto Mk = std::make_shared<std::vector<float>>((size_t) KV*M);   // [M][KV]
    std::mt19937 rng(53 + M);
    for (int j = 0; j < M; ++j) {
        const int64_t valid = KV - (int64_t) (rng() % (KV/4));   // causal-like tail per row
        for (int64_t t = 0; t < KV; ++t) { (*Mk)[(size_t) j*KV + t] = t < valid ? 0.0f : -INFINITY; }
    }
    pcase c;
    c.name = "FLASH_ATTN_EXT D=" + std::to_string(D) + " H=" + std::to_string(H) + "/" + std::to_string(H_kv) +
             " KV=" + std::to_string(KV) + " M=" + std::to_string(M);
    c.n_units = M; c.out_unit = (size_t) D*H;
    c.run = [=](const std::vector<int> & perm) {
        const int64_t m = (int64_t) perm.size();
        auto qp = gather(*Q, (size_t) D*H, perm);   // [m][H][D]
        std::vector<float> qd((size_t) D*m*H);       // ggml layout [D, m, H]
        for (int64_t j = 0; j < m; ++j) for (int64_t h = 0; h < H; ++h) {
            memcpy(qd.data() + ((size_t) h*m + j)*D, qp.data() + ((size_t) j*H + h)*D, D*sizeof(float));
        }
        auto mp = gather(*Mk, (size_t) KV, perm);
        std::vector<ggml_fp16_t> mh(mp.size());
        ggml_fp32_to_fp16_row(mp.data(), mh.data(), (int64_t) mp.size());
        ggml_tensor *q, *k, *v, *msk;
        return run_graph([&](ggml_context * ctx) {
            q   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, m, H);
            k   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, KV, H_kv);
            v   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, KV, H_kv);
            msk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, KV, m);
            ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, msk, 1.0f/sqrtf((float) D), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
            return o;
        }, [&] {
            ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q));
            ggml_backend_tensor_set(k, Kh->data(), 0, ggml_nbytes(k));
            ggml_backend_tensor_set(v, Vh->data(), 0, ggml_nbytes(v));
            ggml_backend_tensor_set(msk, mh.data(), 0, ggml_nbytes(msk));
        }, false);
    };
    return c;
}

// GDN: unit = one sequence (T tokens); the attention block of a sequence is S*H*T contiguous floats and its final state
// S*S*H floats after all attention rows; the runner re-packs both into one unit-major vector.
static pcase case_gdn(int64_t S, int64_t H_k, int64_t H, int64_t T, int B) {
    const size_t uq = (size_t) S*H_k*T, uv = (size_t) S*H*T, ug = (size_t) H*T, us = (size_t) S*S*H;
    auto q = std::make_shared<std::vector<float>>(rnd(uq*B, 61)), k = std::make_shared<std::vector<float>>(rnd(uq*B, 67));
    auto v = std::make_shared<std::vector<float>>(rnd(uv*B, 71));
    auto g = std::make_shared<std::vector<float>>(rnd(ug*B, 73, -0.05f, -1e-4f)), b = std::make_shared<std::vector<float>>(rnd(ug*B, 79, 0.0f, 1.0f));
    auto s = std::make_shared<std::vector<float>>(rnd(us*B, 83, -0.1f, 0.1f));
    // L2-normalise q/k rows as the model does (keeps the state bounded)
    for (auto * vec : { q.get(), k.get() }) {
        for (size_t r = 0; r < vec->size()/S; ++r) {
            double n = 0; for (int64_t i = 0; i < S; ++i) { n += (double) (*vec)[r*S+i]*(*vec)[r*S+i]; }
            const float sc = (float) (1.0/std::sqrt(n + 1e-12));
            for (int64_t i = 0; i < S; ++i) { (*vec)[r*S+i] *= sc; }
        }
    }
    pcase c;
    c.name = "GATED_DELTA_NET S=" + std::to_string(S) + " H=" + std::to_string(H_k) + "/" + std::to_string(H) +
             " T=" + std::to_string(T) + " seqs=" + std::to_string(B);
    c.n_units = B; c.out_unit = uv + us;
    c.run = [=](const std::vector<int> & perm) {
        const int64_t nb = (int64_t) perm.size();
        auto qp = gather(*q, uq, perm), kp = gather(*k, uq, perm), vp = gather(*v, uv, perm);
        auto gp = gather(*g, ug, perm), bp = gather(*b, ug, perm), sp = gather(*s, us, perm);
        ggml_tensor *tq, *tk, *tv, *tg, *tb, *ts;
        auto raw = run_graph([&](ggml_context * ctx) {
            tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, nb);
            tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, nb);
            tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, nb);
            tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, nb);
            tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, nb);
            ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, nb);
            return ggml_gated_delta_net(ctx, tq, tk, tv, tg, tb, ts, 1);
        }, [&] {
            ggml_backend_tensor_set(tq, qp.data(), 0, ggml_nbytes(tq));
            ggml_backend_tensor_set(tk, kp.data(), 0, ggml_nbytes(tk));
            ggml_backend_tensor_set(tv, vp.data(), 0, ggml_nbytes(tv));
            ggml_backend_tensor_set(tg, gp.data(), 0, ggml_nbytes(tg));
            ggml_backend_tensor_set(tb, bp.data(), 0, ggml_nbytes(tb));
            ggml_backend_tensor_set(ts, sp.data(), 0, ggml_nbytes(ts));
        }, false);
        std::vector<float> o((uv + us)*nb);
        for (int64_t j = 0; j < nb; ++j) {
            memcpy(o.data() + j*(uv+us),      raw.data() + j*uv,               uv*sizeof(float));
            memcpy(o.data() + j*(uv+us) + uv, raw.data() + nb*uv + j*us,       us*sizeof(float));
        }
        return o;
    };
    return c;
}

// ---- comparison ----
struct cmp_res { int bad_units = 0; size_t bad_elems = 0; float max_abs = 0.0f; };

// out_perm is in slot order for perm; compare its unit j against ref unit perm[j], bitwise
static cmp_res compare(const std::vector<float> & ref, const std::vector<float> & out_perm, size_t unit, const std::vector<int> & perm) {
    cmp_res r;
    for (size_t j = 0; j < perm.size(); ++j) {
        const float * a = ref.data() + (size_t) perm[j]*unit;
        const float * b = out_perm.data() + j*unit;
        size_t bad = 0;
        for (size_t i = 0; i < unit; ++i) {
            if (f2u(a[i]) != f2u(b[i])) {
                ++bad;
                const float d = std::fabs(a[i] - b[i]);
                if (!(d <= r.max_abs)) { r.max_abs = std::isnan(d) ? INFINITY : d; }
            }
        }
        if (bad) { r.bad_units++; r.bad_elems += bad; }
    }
    return r;
}

static std::vector<std::pair<std::string, std::vector<int>>> perms_for(int n) {
    std::vector<std::pair<std::string, std::vector<int>>> ps;
    std::vector<int> id(n); for (int i = 0; i < n; ++i) { id[i] = i; }
    auto rev = id; std::reverse(rev.begin(), rev.end());                ps.push_back({"reverse", rev});
    auto rot = id; std::rotate(rot.begin(), rot.begin() + 1, rot.end()); ps.push_back({"rotate1", rot});
    auto sw = id; std::swap(sw.front(), sw.back());                     ps.push_back({"swap0k", sw});
    auto sh = id; std::mt19937 rng(7 + n); std::shuffle(sh.begin(), sh.end(), rng); ps.push_back({"shuffle", sh});
    return ps;
}

// comparator self-test: identical buffers pass, a one-ulp flip in any unit is caught
static bool comparator_selftest() {
    std::vector<float> a = rnd(64, 3);
    std::vector<int> p = {3, 1, 0, 2};
    auto b = gather(a, 16, p);
    if (compare(a, b, 16, p).bad_units != 0) { return false; }
    uint32_t u = f2u(b[37]) ^ 1u; memcpy(&b[37], &u, 4);
    const cmp_res r = compare(a, b, 16, p);
    return r.bad_units == 1 && r.bad_elems == 1;
}

int main() {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    const char * bname = getenv("POSINV_BACKEND");
    const std::string bn = bname && *bname ? bname : "CPU";
    g_backend = ggml_backend_init_by_name(bn.c_str(), nullptr);
    if (!g_backend) {
        printf("SKIP: backend %s not available\n", bn.c_str());
        return 0;
    }
    const bool is_cpu = bn == "CPU";
    if (is_cpu) {
        const char * th = getenv("POSINV_THREADS");
        ggml_backend_cpu_set_n_threads(g_backend, th && *th ? atoi(th) : 8);
    }
    g_mutate = getenv("POSINV_MUTATE") && atoi(getenv("POSINV_MUTATE")) != 0;
    const char * only = getenv("POSINV_ONLY");
    const char * det  = getenv("GGML_DETERMINISTIC");
    printf("test-batch-posinv: backend=%s GGML_DETERMINISTIC=%s mutate=%d\n", bn.c_str(), det ? det : "(unset)", g_mutate ? 1 : 0);

    if (!comparator_selftest()) {
        printf("FAIL comparator self-test\n");
        return 1;
    }
    printf("comparator self-test ok\n");

    std::vector<pcase> cases;
    const int Ms[] = {2, 4, 8, 16};
    for (int M : Ms) {
        cases.push_back(case_mul_mat(GGML_TYPE_Q8_0, 4096, 2048, M));   // 8.9 MB > Q8_MF min_bytes: V5 / skinny eligible
        cases.push_back(case_mul_mat(GGML_TYPE_Q4_K, 4096, 1024, M));
        cases.push_back(case_mul_mat(GGML_TYPE_F32, 1024, 512, M));
        cases.push_back(case_mul_mat_id(GGML_TYPE_Q8_0, 1024, 512, 16, 4, M));
        cases.push_back(case_fa(128, 8, 2, 512, M));
        cases.push_back(case_rms_norm(5120, M));
        cases.push_back(case_gdn(128, 4, 8, 4, M));
    }

    int n_run = 0, n_fail = 0, n_caught = 0;
    std::map<std::string, std::pair<int, int>> per_op;   // op -> (cases, position-variant cases)
    printf("%-58s %-8s %-9s %-10s %-12s %s\n", "case", "perm", "pos", "bad_units", "max_abs", "size(M vs 1)");
    for (auto & c : cases) {
        if (only && *only && c.name.find(only) == std::string::npos) { continue; }
        std::vector<int> id(c.n_units); for (int i = 0; i < c.n_units; ++i) { id[i] = i; }
        const std::vector<float> ref = c.run(id);
        const std::vector<float> ref2 = c.run(id);
        const cmp_res rr = compare(ref, ref2, c.out_unit, id);
        // batch-SIZE column (report only): unit 0 alone (M = 1) vs unit 0 at row 0 of the full batch
        std::string size_col = "-";
        {
            const std::vector<float> solo = c.run({0});
            const cmp_res sr = compare(ref, solo, c.out_unit, {0});
            char b[64]; snprintf(b, sizeof b, sr.bad_units ? "DIFF %.2e" : "same", sr.max_abs); size_col = b;
        }
        bool case_bad = rr.bad_units != 0;
        if (rr.bad_units) {
            printf("%-58s %-8s %-9s %-10d %-12.3e %s\n", c.name.c_str(), "repeat", "RUN-NONDET", rr.bad_units, rr.max_abs, size_col.c_str());
        }
        for (auto & [pn, p] : perms_for(c.n_units)) {
            std::vector<float> out = c.run(p);
            if (g_mutate) { const size_t i = out.size()/2; uint32_t u = f2u(out[i]) ^ 1u; memcpy(&out[i], &u, 4); }
            const cmp_res r = compare(ref, out, c.out_unit, p);
            if (r.bad_units) { case_bad = true; }
            printf("%-58s %-8s %-9s %-10d %-12.3e %s\n", c.name.c_str(), pn.c_str(), r.bad_units ? "VARIANT" : "invariant",
                   r.bad_units, r.max_abs, size_col.c_str());
        }
        const std::string op = c.name.substr(0, c.name.find(' ', c.name.find(' ') + 1));
        per_op[op].first++;
        if (case_bad) { per_op[op].second++; }
        n_run++;
        if (case_bad) { n_fail++; n_caught++; }
    }

    // split-count assertion: one G per (M, N, K) over every permutation
    bool split_ok = true; int n_split = 0;
    {
        std::map<std::string, std::set<int>> g_by_mnk;
        std::map<std::string, std::set<int>> g_by_nk;
        std::lock_guard<std::mutex> lk(g_log_mu);
        for (auto & l : g_log) {
            int M, N, K, U, G, d;
            const size_t pos = l.find("q8mf6_split ");
            if (pos != std::string::npos && sscanf(l.c_str() + pos, "q8mf6_split M=%d N=%d K=%d U=%d G=%d det=%d", &M, &N, &K, &U, &G, &d) == 6) {
                n_split++;
                g_by_mnk[std::to_string(M) + "," + std::to_string(N) + "," + std::to_string(K)].insert(G);
                g_by_nk[std::to_string(N) + "," + std::to_string(K)].insert(G);
            }
        }
        for (auto & [k, gs] : g_by_mnk) {
            if (gs.size() != 1) { split_ok = false; printf("FAIL split count varies at (M,N,K)=(%s): %zu values\n", k.c_str(), gs.size()); }
        }
        for (auto & [k, gs] : g_by_nk) {
            printf("split (N,K)=(%s): G %s across M (%zu value%s)\n", k.c_str(), gs.size() == 1 ? "constant" : "varies", gs.size(), gs.size() == 1 ? "" : "s");
        }
        printf("split trace lines: %d%s\n", n_split, n_split ? "" : " (GGML_DET_TRACE unset or V5 path not taken)");
    }

    // path witnesses
    bool path_ok = true;
    const char * expect = getenv("POSINV_EXPECT");
    if (expect && *expect) {
        const std::string e = expect;
        const bool v5   = log_has("V5 4-WG/CU half-tile packed path ran");
        const bool v5d  = log_has("V5 fixed-order split reduction engaged");
        const bool sk   = log_has("q8_skinny: gfx90a skinny-M Q8_0 split-K path ran");
        printf("path witness: V5=%d V5det=%d skinny=%d (expect %s)\n", v5, v5d, sk, e.c_str());
        if (e == "v5det")    { path_ok = v5 && v5d && n_split > 0; }
        if (e == "v5")       { path_ok = v5 && !v5d; }
        if (e == "skinny")   { path_ok = sk; }
        if (e == "noskinny") { path_ok = !sk; }
        if (!path_ok) { printf("FAIL path witness for %s\n", e.c_str()); }
    }

    printf("\nper-op summary (position-variant cases / cases):\n");
    for (auto & [op, cnt] : per_op) { printf("  %-24s %d/%d\n", op.c_str(), cnt.second, cnt.first); }

    ggml_backend_free(g_backend);
    if (g_mutate) {
        const bool ok = n_run > 0 && n_caught == n_run && path_ok;
        printf("%s: mutation caught in %d/%d cases\n", ok ? "PASS (red)" : "FAIL (mutant survived)", n_caught, n_run);
        return ok ? 0 : 1;
    }
    const bool ok = n_run > 0 && n_fail == 0 && split_ok && path_ok;
    printf("%s: %d/%d cases position-invariant\n", ok ? "PASS" : "FAIL", n_run - n_fail, n_run);
    return ok ? 0 : 1;
}
