// test-gdn-gate-fold.cpp: AK-BW-GDN-GATE-FOLD (GGML_HIP_GDN_GATE_FOLD=1) against the unfused path on the same device
// backend, in one process (the knob is read per graph evaluation; CUDA graphs are off so every run dispatches).
//
// Graph = the qwen35 linear-attention gate subgraph feeding GATED_DELTA_NET, at 27B shapes (n_embd 5120, H_v 48,
// H_k 16, S_v 128), Q8_0 ssm_beta / ssm_alpha, allocated by ggml-gallocr (so intermediates share memory as in serving):
//   beta = sigmoid(reshape(mul_mat(Wb, x)));  g = reshape(mul(softplus(add(reshape(mul_mat(Wa, x)), dt)), A))
//   out  = gated_delta_net(q, k, v, g, beta, s0, K)
// Checks per case:
//   1. fold ON: beta and g match a host double-precision reference (dequantized Q8_0 weights, f32 activations) to
//      |d| <= 2e-5 * (1 + |ref|)  (the fold reads f32 activations; only the f32 summation order differs);
//   2. fold ON GDN output (attention + snapshot slots) matches the same GDN op fed the host-reference beta/g directly
//      (fold OFF, no projections) to |d| <= 1e-4 * (1 + |ref|), NMSE <= 1e-8;
//   3. the fold actually ran (launch witness advanced) and did NOT run with the knob off;
//   4. fold OFF (stock path: int8-quantized activations) is reported against the same reference (NMSE <= 1e-3 asserted,
//      a sanity bound only: the stock path is the less exact one).
// GDN_GATE_FOLD_BACKEND=<name> (default ROCm0); exits 0 with "SKIP" when that backend is absent (CPU-only build).

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
static n_calls_fn g_fold_calls = nullptr;
static n_calls_fn g_fold_unsafe = nullptr;
static int g_n_ok = 0, g_n_guard = 0, g_n_fail = 0;

static const int64_t NE = 5120, H = 48, H_K = 16, S = 128;

struct inputs {
    std::vector<float> x, q, k, v, dt, a, s0;
    std::vector<uint8_t> wb, wa;   // Q8_0 bytes
    std::vector<float> beta_ref, g_ref;
};

struct outputs {
    std::vector<float> beta, g, out;
};

// full = true: the gate subgraph + GDN; false: GDN fed beta_ref / g_ref directly
static outputs run(bool full, int64_t T, int64_t B, int64_t K, const inputs & in) {
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t ntok = T * B;
    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_K, T, B);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_K, T, B);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * s0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, B);
    for (ggml_tensor * t : { q, k, v, s0 }) ggml_set_input(t);
    ggml_tensor * x = nullptr, * wb = nullptr, * wa = nullptr, * dt = nullptr, * a = nullptr;
    ggml_tensor * beta_in = nullptr, * g_in = nullptr;
    ggml_tensor * beta, * gate_mul, * gate;
    if (full) {
        x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE, ntok);
        wb = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, NE, H);
        wa = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, NE, H);
        dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        for (ggml_tensor * t : { x, wb, wa, dt, a }) ggml_set_input(t);
        // keep x live to the anchor: in this tiny graph gallocr otherwise reuses its memory at ntok=1 (r0 smoke), and the
        // planner's C7R guard then (correctly) runs the chain unfused, which this test would read as a route failure
        ggml_set_output(x);
        // exactly the qwen35.cpp build_layer_attn_linear order
        beta = ggml_mul_mat(ctx, wb, x);
        beta = ggml_reshape_4d(ctx, beta, 1, H, T, B);
        beta = ggml_sigmoid(ctx, beta);
        ggml_tensor * alpha = ggml_mul_mat(ctx, wa, x);
        alpha = ggml_reshape_3d(ctx, alpha, H, T, B);
        ggml_tensor * alpha_biased   = ggml_add(ctx, alpha, dt);
        ggml_tensor * alpha_softplus = ggml_softplus(ctx, alpha_biased);
        gate_mul = ggml_mul(ctx, alpha_softplus, a);
        gate     = ggml_reshape_4d(ctx, gate_mul, 1, H, T, B);
    } else {
        beta_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
        g_in    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
        ggml_set_input(beta_in); ggml_set_input(g_in);
        beta = beta_in; gate_mul = g_in; gate = g_in;
    }
    ggml_set_output(beta);
    ggml_set_output(gate_mul);
    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, gate, beta, s0, K);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, beta);
    ggml_build_forward_expand(gf, gate_mul);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "gallocr failed\n");
        exit(2);
    }
    ggml_backend_tensor_set(q, in.q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, in.k.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, in.v.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(s0, in.s0.data(), 0, ggml_nbytes(s0));
    if (full) {
        ggml_backend_tensor_set(x, in.x.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(wb, in.wb.data(), 0, ggml_nbytes(wb));
        ggml_backend_tensor_set(wa, in.wa.data(), 0, ggml_nbytes(wa));
        ggml_backend_tensor_set(dt, in.dt.data(), 0, ggml_nbytes(dt));
        ggml_backend_tensor_set(a, in.a.data(), 0, ggml_nbytes(a));
    } else {
        ggml_backend_tensor_set(beta_in, in.beta_ref.data(), 0, ggml_nbytes(beta_in));
        ggml_backend_tensor_set(g_in, in.g_ref.data(), 0, ggml_nbytes(g_in));
    }
    ggml_backend_graph_compute(g_backend, gf);
    outputs o;
    o.beta.resize(ggml_nelements(beta)); o.g.resize(ggml_nelements(gate_mul)); o.out.resize(ggml_nelements(out));
    ggml_backend_tensor_get(beta, o.beta.data(), 0, ggml_nbytes(beta));
    ggml_backend_tensor_get(gate_mul, o.g.data(), 0, ggml_nbytes(gate_mul));
    ggml_backend_tensor_get(out, o.out.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return o;
}

struct cmp { double maxe = 0, nmse = 0; size_t bad = 0; };

static cmp compare(const std::vector<float> & c, const std::vector<float> & r, double rel) {
    cmp res; double num = 0, den = 0;
    for (size_t i = 0; i < r.size(); i++) {
        const double d = (double) c[i] - r[i];
        num += d * d; den += (double) r[i] * r[i];
        const double e = fabs(d);
        res.maxe = e > res.maxe ? e : res.maxe;
        if (!(e <= rel * (1.0 + fabs(r[i])))) res.bad++;
    }
    res.nmse = den > 0 ? num / den : num;
    return res;
}

static int run_case(int64_t T, int64_t B, int64_t K) {
    const int64_t ntok = T * B;
    std::mt19937 rng(0x9a7e + 31 * (int) T + 7 * (int) B + (int) K);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    inputs in;
    in.x.resize(NE * ntok); for (auto & e : in.x) e = u(rng);
    auto unit = [&](std::vector<float> & z) {
        z.resize(S * H_K * T * B);
        for (auto & e : z) e = u(rng);
        for (size_t r = 0; r < z.size() / S; r++) {
            float n2 = 0; for (int64_t j = 0; j < S; j++) n2 += z[r*S + j] * z[r*S + j];
            for (int64_t j = 0; j < S; j++) z[r*S + j] /= sqrtf(n2);
        }
    };
    unit(in.q); unit(in.k);
    in.v.resize(S * H * T * B); for (auto & e : in.v) e = u(rng);
    in.s0.resize(S * S * H * B); for (auto & e : in.s0) e = 0.5f * u(rng);
    in.dt.resize(H); for (auto & e : in.dt) e = u(rng);
    in.a.resize(H);  for (auto & e : in.a) e = -(0.01f + 0.045f * (u(rng) + 1.0f));   // -exp(A_log), mild decay
    std::vector<float> wbf(NE * H), waf(NE * H);
    for (auto & e : wbf) e = 0.02f * u(rng);
    for (auto & e : waf) e = 0.02f * u(rng);
    const size_t rs = ggml_row_size(GGML_TYPE_Q8_0, NE);
    in.wb.resize(rs * H); in.wa.resize(rs * H);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wbf.data(), in.wb.data(), 0, H, NE, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, waf.data(), in.wa.data(), 0, H, NE, nullptr);
    // host reference from the dequantized weights
    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<float> wbd(NE * H), wad(NE * H);
    tt->to_float(in.wb.data(), wbd.data(), NE * H);
    tt->to_float(in.wa.data(), wad.data(), NE * H);
    in.beta_ref.resize(H * ntok); in.g_ref.resize(H * ntok);
    for (int64_t t = 0; t < ntok; t++) {
        for (int64_t h = 0; h < H; h++) {
            double sb = 0, sa = 0;
            for (int64_t i = 0; i < NE; i++) {
                sb += (double) wbd[h * NE + i] * in.x[t * NE + i];
                sa += (double) wad[h * NE + i] * in.x[t * NE + i];
            }
            in.beta_ref[t * H + h] = (float) (1.0 / (1.0 + exp(-sb)));
            const double z = sa + in.dt[h];
            const double sp = z > 20.0 ? z : log1p(exp(z));
            in.g_ref[t * H + h] = (float) (sp * in.a[h]);
        }
    }

    const uint64_t c0 = g_fold_calls ? g_fold_calls() : 0;
    const uint64_t u0 = g_fold_unsafe ? g_fold_unsafe() : 0;
    unsetenv("GGML_HIP_GDN_GATE_FOLD");
    const outputs off = run(true, T, B, K, in);
    const uint64_t c1 = g_fold_calls ? g_fold_calls() : 0;
    const outputs ref = run(false, T, B, K, in);
    setenv("GGML_HIP_GDN_GATE_FOLD", "1", 1);
    const outputs on = run(true, T, B, K, in);
    unsetenv("GGML_HIP_GDN_GATE_FOLD");
    const uint64_t c2 = g_fold_calls ? g_fold_calls() : 0;
    const uint64_t u2 = g_fold_unsafe ? g_fold_unsafe() : 0;

    const cmp cb  = compare(on.beta, in.beta_ref, 2e-5);
    const cmp cg  = compare(on.g, in.g_ref, 2e-5);
    const cmp co  = compare(on.out, ref.out, 1e-4);
    const cmp ob  = compare(off.beta, in.beta_ref, 1e-2);
    const cmp og  = compare(off.g, in.g_ref, 1e-2);
    const cmp oo  = compare(off.out, ref.out, 1e-2);
    const bool routed     = g_fold_calls != nullptr && c2 == c1 + 1;
    const bool off_clean  = g_fold_calls != nullptr && c1 == c0;
    // GUARD_REFUSED: the planner's C7R guard refused the fold (unsafe witness advanced, no fold launch) and the ON run equals
    // the stock path (same unfused kernels): not a failure of the fold, but the fold is unproven for that case
    const cmp sb = compare(on.beta, off.beta, 1e-5), sg = compare(on.g, off.g, 1e-5), so = compare(on.out, off.out, 1e-5);
    const bool guard_refused = g_fold_unsafe != nullptr && u2 > u0 && c2 == c1 && off_clean &&
                               sb.bad == 0 && sg.bad == 0 && so.bad == 0 && ob.nmse <= 1e-3 && og.nmse <= 1e-3;
    const bool ok = cb.bad == 0 && cg.bad == 0 && co.bad == 0 && co.nmse <= 1e-8 && routed && off_clean &&
                    ob.nmse <= 1e-3 && og.nmse <= 1e-3;
    printf("[%s] T=%lld n_seqs=%lld K=%lld ntok=%lld | ON beta max|d| %.2e g %.2e out max|d| %.2e nmse %.2e out-of-tol %zu/%zu/%zu"
           " | OFF(stock) beta nmse %.2e g nmse %.2e out nmse %.2e | fold %s, off-run %s -> %s\n",
           ggml_backend_name(g_backend), (long long) T, (long long) B, (long long) K, (long long) ntok,
           cb.maxe, cg.maxe, co.maxe, co.nmse, cb.bad, cg.bad, co.bad, ob.nmse, og.nmse, oo.nmse,
           routed ? "ran" : "DID NOT RUN", off_clean ? "unfused" : "FOLDED (knob leak)",
           ok ? "OK" : guard_refused ? "GUARD_REFUSED (C7R guard refused the fold; ON == stock path)" : "FAIL");
    if (ok) g_n_ok++; else if (guard_refused) g_n_guard++; else g_n_fail++;
    return ok || guard_refused ? 0 : 1;
}

int main() {
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    ggml_backend_load_all();
    const char * bname = getenv("GDN_GATE_FOLD_BACKEND");
    g_backend = ggml_backend_init_by_name(bname && bname[0] ? bname : "ROCm0", nullptr);
    if (g_backend == nullptr) {
        printf("SKIP: backend %s not available\n", bname && bname[0] ? bname : "ROCm0");
        return 0;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(g_backend));
    g_fold_calls = (n_calls_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_gdn_gate_fold_n_calls");
    g_fold_unsafe = (n_calls_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_gdn_gate_fold_n_unsafe");
    if (g_fold_calls == nullptr) {
        printf("gate-fold witness hook missing FAIL\n");
        return 1;
    }
    int failures = 0, n = 0;
    struct { int64_t T, B; } shapes[] = { {1, 1}, {4, 1}, {8, 1}, {12, 1}, {16, 1}, {8, 2}, {4, 4} };
    for (auto & s : shapes) {
        failures += run_case(s.T, s.B, 1); n++;
        if (s.T > 1) { failures += run_case(s.T, s.B, s.T); n++; }
    }
    printf("%d folded OK / %d guard-refused / %d FAIL (of %d)\n", g_n_ok, g_n_guard, g_n_fail, n);
    printf(failures == 0 ? "OK\n" : "FAIL (%d)\n", failures);
    ggml_backend_free(g_backend);
    return failures == 0 ? 0 : 1;
}
