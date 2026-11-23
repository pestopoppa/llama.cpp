// test-gdn-fused.cpp: [GDN-FUSED-VARIANT] the fused float2-reduction GATED_DELTA_NET kernel (GGML_HIP_GDN_FUSED=1)
// against the existing kernel (knob unset) on the same device backend, in one process: the knob is read per launch.
// Verify shapes of the 27B (H_k 16, H_v 48, S_v 128), n_seqs 1 and 4, T = 1/4/8/12/16 rows, K = 1 and K = T,
// F32 and BF16 state, mild gate (log-gate in [-0.05, -1e-4], so the state survives every row).
// F32: the reduction order differs (8-lane pair reduction vs warp reduction), so attention and every state slot
// must agree to max |a-b| <= 1e-4 * (1 + |b|) and NMSE <= 1e-9. BF16 state/attention: <= 1 bf16 ulp relative
// (2^-7) plus 1e-3 absolute, NMSE <= 1e-5. The fused route must actually run (launch witness grows).
// GDN_FUSED_BACKEND=<name> (default ROCm0); exits 0 with "SKIP" when that backend is absent (CPU-only build).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static ggml_backend_t g_backend = nullptr;
typedef uint64_t (*n_calls_fn)();
static n_calls_fn g_fused_calls = nullptr;

static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }

struct inputs {
    std::vector<float> q, k, v, g, b;
    std::vector<char>  s0;
};

// run the op once; returns the full dst as floats (attention rows then K state slots)
static std::vector<float> run(int64_t S, int64_t H_k, int64_t H, int64_t T, int64_t B, int64_t K, ggml_type st, const inputs & in) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, st, S, S, H, B);
    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, b, s, K);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    ggml_backend_tensor_set(q, in.q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, in.k.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, in.v.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(g, in.g.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(b, in.b.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(s, in.s0.data(), 0, ggml_nbytes(s));
    ggml_backend_graph_compute(g_backend, gf);
    std::vector<float> res(ggml_nelements(out));
    if (out->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    } else {
        std::vector<uint16_t> h(ggml_nelements(out));
        ggml_backend_tensor_get(out, h.data(), 0, ggml_nbytes(out));
        for (size_t i = 0; i < h.size(); i++) { res[i] = bf16_to_f32(h[i]); }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// cols: GGML_HIP_GDN_FUSED_COLS for the knob-on run (1 = r3 default; 2/4 = [GDN-COLWIDTH], K=T only)
static int run_case(int64_t T, int64_t B, int64_t K, ggml_type st, int cols = 1, int64_t S = 128) {
    const int64_t H_k = 16, H = 48;
    std::mt19937 rng(0x6d4e + 31 * (int) T + 7 * (int) B + (int) K + (st == GGML_TYPE_BF16 ? 500 : 0) + (int) S);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    inputs in;
    auto unit = [&](std::vector<float> & x) {
        x.resize(S * H_k * T * B);
        for (auto & e : x) { e = u(rng); }
        for (size_t r = 0; r < x.size() / S; r++) {
            float n2 = 0; for (int64_t j = 0; j < S; j++) { n2 += x[r*S + j]*x[r*S + j]; }
            for (int64_t j = 0; j < S; j++) { x[r*S + j] /= sqrtf(n2); }
        }
    };
    unit(in.q); unit(in.k);
    in.v.resize(S * H * T * B); for (auto & e : in.v) { e = -0.3f + 2.65f * (u(rng) + 1.0f); }
    in.g.resize(H * T * B);     for (auto & e : in.g) { e = -0.05f + (0.05f - 1e-4f) * 0.5f * (u(rng) + 1.0f); }
    in.b.resize(H * T * B);     for (auto & e : in.b) { e = 0.5f * (u(rng) + 1.0f); }
    const size_t ns = (size_t) S * S * H * B;
    std::vector<float> s0f(ns); for (auto & e : s0f) { e = 0.5f * u(rng); }
    if (st == GGML_TYPE_F32) {
        in.s0.resize(ns * 4); memcpy(in.s0.data(), s0f.data(), ns * 4);
    } else {
        in.s0.resize(ns * 2);
        uint16_t * h = (uint16_t *) in.s0.data();
        for (size_t i = 0; i < ns; i++) { h[i] = ggml_fp32_to_bf16(s0f[i]).bits; }
    }

    setenv("GGML_HIP_GDN_FUSED", "0", 1);   // [C19] fused is default ON: pin the reference to the unfused route
    const std::vector<float> ref = run(S, H_k, H, T, B, K, st, in);
    const uint64_t c0 = g_fused_calls ? g_fused_calls() : 0;
    setenv("GGML_HIP_GDN_FUSED", "1", 1);
    if (K <= 1) {
        setenv("GGML_HIP_GDN_FUSED_K1", "1", 1);   // [C19] the K=1 rows route is opt-in (default off); test it explicitly
    }
    char cbuf[8]; snprintf(cbuf, sizeof(cbuf), "%d", cols);
    if (cols > 0) {
        setenv("GGML_HIP_GDN_FUSED_COLS", cbuf, 1);
    } else {
        unsetenv("GGML_HIP_GDN_FUSED_COLS");   // cols 0 = automatic size/dtype dispatch (the default)
    }
    const std::vector<float> cand = run(S, H_k, H, T, B, K, st, in);
    unsetenv("GGML_HIP_GDN_FUSED");
    unsetenv("GGML_HIP_GDN_FUSED_COLS");
    unsetenv("GGML_HIP_GDN_FUSED_K1");
    const uint64_t c1 = g_fused_calls ? g_fused_calls() : 0;

    const bool bf = st == GGML_TYPE_BF16;
    const double rel = bf ? 1.0 / 128.0 : 1e-4, abs_tol = bf ? 1e-3 : 0.0, nmse_max = bf ? 1e-5 : 1e-9;
    double num = 0, den = 0, maxe = 0; size_t bad = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = (double) cand[i] - ref[i];
        num += d * d; den += (double) ref[i] * ref[i];
        const double e = fabs(d);
        maxe = e > maxe ? e : maxe;
        if (!(e <= abs_tol + rel * (1.0 + fabs(ref[i])))) { bad++; }
    }
    const double nmse = den > 0 ? num / den : num;
    const bool routed = g_fused_calls == nullptr || c1 > c0;
    const bool ok = bad == 0 && nmse <= nmse_max && routed && ref.size() == cand.size();
    printf("[%s] S=%lld H_k=16 H=48 T=%lld n_seqs=%lld K=%lld state=%s cols=%d: max|d| %.3g nmse %.3g out-of-tol %zu fused-route %s -> %s\n",
           ggml_backend_name(g_backend), (long long) S, (long long) T, (long long) B, (long long) K, bf ? "bf16" : "f32", cols, maxe, nmse, bad,
           routed ? "ran" : "DID NOT RUN", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// [DTREE-GDN-BATCH] tree-walk cases (GGML_HIP_GDN_FUSED_TREE=1, F32 state, K = T). Two checks per tree:
//  (a) knob on vs knob off (the recurrent tree kernel) on the same backend: same F32 bound as above;
//  (b) path identity, n_seqs 1: for every column c whose root path has >= 2 columns, the tree op's attention row c and
//      its snapshot slot (T-1-c) must be BIT-IDENTICAL to the same fused tree kernel run as a chain (parent t-1) on that path's rows
//      (last attention row, slot 0). The tree walk must do exactly the chain arithmetic along the path.
static std::vector<int32_t> tree_parents(int kind, int64_t T, int64_t seq) {
    std::vector<int32_t> p(T);
    for (int64_t t = 0; t < T; ++t) {
        if (t == 0) {
            p[t] = -1;
        } else if (kind == 3) {   // greedy layout: main chain, one alternative per chain node (DFlash2 batch tree)
            const int64_t K = T - T / 3;
            p[t] = t < K ? (int32_t) (t - 1) : (int32_t) (((t - K) * 2) % K);
        } else if (kind == 1) {   // Infernix layout: main chain 0..K-1, side columns hang off it
            const int64_t K = (T + 1) / 2;
            p[t] = t < K ? (int32_t) (t - 1) : (int32_t) ((t * 7 + seq) % K);
        } else {                  // bushy: several root children, parents anywhere before
            p[t] = (t % 3 == 0) ? -1 : (int32_t) ((t * 5 + seq) % t);
        }
    }
    return p;
}

static std::vector<float> run_tree(int64_t S, int64_t H_k, int64_t H, int64_t T, int64_t B, const inputs & in,
                                   const std::vector<int32_t> & par) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, B);
    ggml_tensor * pt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, T, B);
    ggml_tensor * out = ggml_gated_delta_net_tree(ctx, q, k, v, g, b, s, pt);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    ggml_backend_tensor_set(q, in.q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, in.k.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, in.v.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(g, in.g.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(b, in.b.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(s, in.s0.data(), 0, ggml_nbytes(s));
    ggml_backend_tensor_set(pt, par.data(), 0, ggml_nbytes(pt));
    ggml_backend_graph_compute(g_backend, gf);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static int run_tree_case(int64_t T, int64_t B, int kind, int64_t S = 128) {
    const int64_t H_k = 16, H = 48;
    std::mt19937 rng(0x7e3 + 31 * (int) T + 7 * (int) B + 101 * kind + (int) S);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    inputs in;
    auto unit = [&](std::vector<float> & x) {
        x.resize(S * H_k * T * B);
        for (auto & e : x) { e = u(rng); }
        for (size_t r = 0; r < x.size() / S; r++) {
            float n2 = 0; for (int64_t j = 0; j < S; j++) { n2 += x[r*S + j]*x[r*S + j]; }
            for (int64_t j = 0; j < S; j++) { x[r*S + j] /= sqrtf(n2); }
        }
    };
    unit(in.q); unit(in.k);
    in.v.resize(S * H * T * B); for (auto & e : in.v) { e = -0.3f + 2.65f * (u(rng) + 1.0f); }
    in.g.resize(H * T * B);     for (auto & e : in.g) { e = -0.05f + (0.05f - 1e-4f) * 0.5f * (u(rng) + 1.0f); }
    in.b.resize(H * T * B);     for (auto & e : in.b) { e = 0.5f * (u(rng) + 1.0f); }
    const size_t ns = (size_t) S * S * H * B;
    in.s0.resize(ns * 4);
    { float * f = (float *) in.s0.data(); for (size_t i = 0; i < ns; i++) { f[i] = 0.5f * u(rng); } }
    std::vector<int32_t> par;
    for (int64_t sq = 0; sq < B; sq++) { auto p = tree_parents(kind, T, sq); par.insert(par.end(), p.begin(), p.end()); }

    setenv("GGML_HIP_GDN_FUSED", "1", 1);
    unsetenv("GGML_HIP_GDN_FUSED_TREE");
    const std::vector<float> ref = run_tree(S, H_k, H, T, B, in, par);
    const uint64_t c0 = g_fused_calls ? g_fused_calls() : 0;
    setenv("GGML_HIP_GDN_FUSED_TREE", "1", 1);
    const std::vector<float> cand = run_tree(S, H_k, H, T, B, in, par);
    const uint64_t c1 = g_fused_calls ? g_fused_calls() : 0;

    double num = 0, den = 0, maxe = 0; size_t bad = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = (double) cand[i] - ref[i];
        num += d * d; den += (double) ref[i] * ref[i];
        const double e = fabs(d);
        maxe = e > maxe ? e : maxe;
        if (!(e <= 1e-4 * (1.0 + fabs(ref[i])))) { bad++; }
    }
    const double nmse = den > 0 ? num / den : num;
    const bool routed = g_fused_calls == nullptr || c1 > c0;

    // (b) path identity (n_seqs 1 only)
    size_t n_paths = 0, path_bad = 0;
    if (B == 1) {
        const size_t attn_n = (size_t) S * H * T;
        for (int64_t c = 0; c < T; c++) {
            std::vector<int64_t> path;
            for (int64_t x = c; x >= 0; x = par[x]) { path.insert(path.begin(), x); }
            const int64_t L = (int64_t) path.size();
            if (L < 2) { continue; }   // a 1-row chain is a K=1 op, which the fused kernel does not take
            inputs pin;
            pin.s0 = in.s0;
            for (int64_t x : path) {
                pin.q.insert(pin.q.end(), in.q.begin() + x * S * H_k, in.q.begin() + (x + 1) * S * H_k);
                pin.k.insert(pin.k.end(), in.k.begin() + x * S * H_k, in.k.begin() + (x + 1) * S * H_k);
                pin.v.insert(pin.v.end(), in.v.begin() + x * S * H, in.v.begin() + (x + 1) * S * H);
                pin.g.insert(pin.g.end(), in.g.begin() + x * H, in.g.begin() + (x + 1) * H);
                pin.b.insert(pin.b.end(), in.b.begin() + x * H, in.b.begin() + (x + 1) * H);
            }
            // the same tree kernel instantiation over the path's rows as a chain (parent t-1): the plain fused K=T kernel is a
            // different template instance whose FMA contraction may differ (smoke: ~1e-7 diffs at S_v 128), so it is held
            // only to check (a)'s bound
            std::vector<int32_t> cpar(L);
            for (int64_t t = 0; t < L; t++) { cpar[t] = (int32_t) t - 1; }
            const std::vector<float> ch = run_tree(S, H_k, H, L, 1, pin, cpar);
            const size_t ch_attn = (size_t) S * H * L;
            size_t diff = 0;
            for (int64_t i = 0; i < S * H; i++) {   // attention row c vs chain row L-1
                if (memcmp(&cand[c * S * H + i], &ch[(L - 1) * S * H + i], 4) != 0) { diff++; }
            }
            const size_t tslot = attn_n + (size_t) (T - 1 - c) * ns;
            for (size_t i = 0; i < ns; i++) {        // snapshot of column c vs chain slot 0
                if (memcmp(&cand[tslot + i], &ch[ch_attn + i], 4) != 0) { diff++; }
            }
            n_paths++;
            path_bad += diff != 0;
            if (diff != 0) { printf("  path to column %lld (len %lld): %zu elements differ\n", (long long) c, (long long) L, diff); }
        }
    }
    unsetenv("GGML_HIP_GDN_FUSED_TREE");
    unsetenv("GGML_HIP_GDN_FUSED");

    const bool ok = bad == 0 && nmse <= 1e-9 && routed && ref.size() == cand.size() && path_bad == 0;
    printf("[%s] TREE kind=%d S=%lld T=%lld n_seqs=%lld: vs recurrent max|d| %.3g nmse %.3g out-of-tol %zu; path-identity %zu/%zu bit-exact; fused-route %s -> %s\n",
           ggml_backend_name(g_backend), kind, (long long) S, (long long) T, (long long) B, maxe, nmse, bad,
           n_paths - path_bad, n_paths, routed ? "ran" : "DID NOT RUN", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    // The knob is read on the host at dispatch. CUDA/HIP graph replay of an identical graph skips host dispatch, so a
    // knob flip between two runs could replay the previous run's kernels (seen: knob-on bit-identical to knob-off and the
    // launch witness not advancing). Graphs off makes every run dispatch.
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    ggml_backend_load_all();
    const char * bname = getenv("GDN_FUSED_BACKEND");
    g_backend = ggml_backend_init_by_name(bname && bname[0] ? bname : "ROCm0", nullptr);
    if (g_backend == nullptr) {
        printf("SKIP: backend %s not available\n", bname && bname[0] ? bname : "ROCm0");
        return 0;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(g_backend));
    g_fused_calls = (n_calls_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_gdn_fused_n_calls");
    int failures = 0, n = 0;
    for (ggml_type st : { GGML_TYPE_F32, GGML_TYPE_BF16 }) {
        for (int64_t B : { 1, 4 }) {
            for (int64_t T : { 1, 4, 8, 12, 16 }) {
                failures += run_case(T, B, 1, st); n++;
                if (T > 1) {
                    for (int cols : { 0, 1, 2, 4 }) { failures += run_case(T, B, T, st, cols); n++; }
                }
            }
        }
    }
    // [GDN-COLWIDTH] S_v 64 K=T (width 4 clamps to 2 there)
    for (ggml_type st : { GGML_TYPE_F32, GGML_TYPE_BF16 }) {
        for (int64_t T : { 8, 16 }) {
            for (int cols : { 0, 1, 2, 4 }) { failures += run_case(T, 1, T, st, cols, 64); n++; }
        }
    }
    // [DTREE-GDN-BATCH] tree walk on the fused kernel
    for (int kind : { 1, 2, 3 }) {
        for (int64_t B : { 1, 4 }) {
            for (int64_t T : { 4, 8, 12, 16 }) {
                failures += run_tree_case(T, B, kind); n++;
            }
        }
        failures += run_tree_case(16, 1, kind, 64); n++;
    }
    // [GDN-COLWIDTH-DISPATCH] the automatic width table at the measured shapes (H = 48 v-heads)
    typedef int (*auto_cols_fn)(int, int64_t, size_t);
    auto_cols_fn ac = (auto_cols_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_gdn_fused_auto_cols");
    if (ac != nullptr) {
        struct { int sv; int64_t hs; size_t esz; int want; } tab[] = {
            {128, 48, 4, 2}, {128, 96, 4, 4}, {128, 192, 4, 4}, {64, 48, 4, 1}, {64, 96, 4, 1}, {64, 192, 4, 2},
            {128, 48, 2, 1}, {128, 96, 2, 1}, {128, 192, 2, 4}, {64, 192, 2, 1},
        };
        for (auto & e : tab) {
            const int got = ac(e.sv, e.hs, e.esz);
            const bool ok = got == e.want;
            printf("auto-cols S_v=%d H*n_seqs=%lld state=%s -> %d (want %d) %s\n", e.sv, (long long) e.hs, e.esz == 4 ? "f32" : "bf16",
                   got, e.want, ok ? "OK" : "FAIL");
            failures += !ok; n++;
        }
    } else {
        printf("auto-cols hook missing FAIL\n");
        failures++; n++;
    }
    printf("%d/%d fused-vs-existing cases OK\n", n - failures, n);
    printf(failures == 0 ? "OK\n" : "FAIL (%d)\n", failures);
    ggml_backend_free(g_backend);
    return failures == 0 ? 0 : 1;
}
