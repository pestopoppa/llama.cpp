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

    unsetenv("GGML_HIP_GDN_FUSED");
    const std::vector<float> ref = run(S, H_k, H, T, B, K, st, in);
    const uint64_t c0 = g_fused_calls ? g_fused_calls() : 0;
    setenv("GGML_HIP_GDN_FUSED", "1", 1);
    char cbuf[8]; snprintf(cbuf, sizeof(cbuf), "%d", cols);
    setenv("GGML_HIP_GDN_FUSED_COLS", cbuf, 1);
    const std::vector<float> cand = run(S, H_k, H, T, B, K, st, in);
    unsetenv("GGML_HIP_GDN_FUSED");
    unsetenv("GGML_HIP_GDN_FUSED_COLS");
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
                    for (int cols : { 1, 2, 4 }) { failures += run_case(T, B, T, st, cols); n++; }
                }
            }
        }
    }
    // [GDN-COLWIDTH] S_v 64 K=T (width 4 clamps to 2 there)
    for (ggml_type st : { GGML_TYPE_F32, GGML_TYPE_BF16 }) {
        for (int64_t T : { 8, 16 }) {
            for (int cols : { 1, 2, 4 }) { failures += run_case(T, 1, T, st, cols, 64); n++; }
        }
    }
    printf("%d/%d fused-vs-existing cases OK\n", n - failures, n);
    printf(failures == 0 ? "OK\n" : "FAIL (%d)\n", failures);
    ggml_backend_free(g_backend);
    return failures == 0 ? 0 : 1;
}
