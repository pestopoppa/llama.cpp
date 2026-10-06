// CTX-KSHIFT probes (H1 mscale re-applied per K-shift, H2 pos_div shift sign, H3 Hadamard nrot vs partial rotary).
// Builds the ggml ops exactly as llama_kv_cache::build_rope_shift / build_graph_shift do, on the CPU backend.
// Sites at ffc1bac82: src/llama-kv-cache.cpp:1977 (build_rope_shift), :2056 (build_graph_shift),
// :1453 (build_input_k_rot), src/llama-kv-cells.h:440 (pos_add), :469 (pos_div).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama-kv-cells.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <random>
#include <vector>

static const int HD = 256, NROT = 64, NH = 2, NC = 8;
static const int N_CTX_ORIG = 262144;
static const float BASE = 10000000.0f;

struct rp { float fs, ext, attn; };
static const rp YARN = {0.5f, 1.0f, 1.0f};
static const rp PLAIN = {1.0f, 0.0f, 1.0f};

static ggml_backend_t be;


// run a graph. The builder expands its own nodes (in order) into gf and returns the output tensor.
using builder = std::function<ggml_tensor *(ggml_context *, ggml_cgraph *)>;
using inputs_t = std::vector<std::pair<const char *, std::vector<char>>>;
static std::vector<float> run(const builder & b, const inputs_t & inputs) {
    ggml_init_params ip = {ggml_tensor_overhead()*512 + ggml_graph_overhead_custom(512, false), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
    ggml_tensor * out = b(ctx, gf);
    ggml_tensor * outf = ggml_cast(ctx, out, GGML_TYPE_F32);
    ggml_build_forward_expand(gf, outf);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (auto & in : inputs) {
        ggml_tensor * t = ggml_get_tensor(ctx, in.first);
        ggml_backend_tensor_set(t, in.second.data(), 0, in.second.size());
    }
    ggml_backend_graph_compute(be, gf);
    std::vector<float> r(ggml_nelements(outf));
    ggml_backend_tensor_get(outf, r.data(), 0, r.size()*4);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}
template <class T> static std::vector<char> bytes(const std::vector<T> & v) {
    std::vector<char> r(v.size()*sizeof(T)); memcpy(r.data(), v.data(), r.size()); return r;
}
static ggml_tensor * inp(ggml_context * c, const char * name, ggml_type t, int64_t a, int64_t b = 1, int64_t d = 1) {
    ggml_tensor * x = ggml_new_tensor_3d(c, t, a, b, d); ggml_set_name(x, name); ggml_set_input(x); return x;
}
static std::vector<char> to_type(const std::vector<float> & f, ggml_type t) {
    if (t == GGML_TYPE_F32) return bytes(f);
    std::vector<char> r(ggml_row_size(t, f.size()));
    const auto * tt = ggml_get_type_traits(t);
    tt->from_float_ref(f.data(), r.data(), f.size());
    return r;
}
static ggml_tensor * rope(ggml_context * c, ggml_tensor * x, ggml_tensor * pos, const rp & p) {
    return ggml_rope_ext(c, x, pos, nullptr, NROT, GGML_ROPE_TYPE_NEOX, N_CTX_ORIG, BASE, p.fs, p.ext, p.attn, 32.0f, 1.0f);
}
// verbatim copy of llama_mul_mat_hadamard (src/llama-impl.h:57)
static ggml_tensor * had(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const auto n = rot->ne[0];
    ggml_tensor * res = !ggml_is_contiguous(cur) ? ggml_cont_2d(ctx, cur, n, ggml_nelements(cur)/n) : ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat(ctx, rot, res);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}
// verbatim ggml_gen_hadamard (src/llama-kv-cache.cpp:24)
static std::vector<float> gen_had(int n) {
    std::vector<float> data(n*n);
    data[0] = 1.0/sqrtf(n);
    for (int s = 1; s < n; s *= 2) for (int i = 0; i < s; i++) for (int j = 0; j < s; j++) {
        const float val = data[i*n + j];
        data[(i + s)*n + j] = val; data[i*n + j + s] = val; data[(i + s)*n + j + s] = -val;
    }
    return data;
}
// llama_kv_cache::build_input_k_rot nrot derivation (src/llama-kv-cache.cpp:1454-1461)
static int derive_nrot(int n_embd_head_k_all) {
    int nrot = 64;
    do { nrot *= 2; } while (n_embd_head_k_all % nrot == 0);
    return nrot / 2;
}
static double maxabs(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0; for (size_t i = 0; i < a.size(); i++) m = std::max(m, (double) fabsf(a[i]-b[i])); return m;
}
static double rms(const std::vector<float> & a, const std::vector<float> & b) {
    double s = 0; for (size_t i = 0; i < a.size(); i++) { double d = a[i]-b[i]; s += d*d; } return sqrt(s/a.size());
}
static double nrm(const std::vector<float> & k, bool rot_only) {
    double s = 0; for (size_t i = 0; i < k.size(); i++) { if (rot_only && (i % HD) >= NROT) continue; s += (double) k[i]*k[i]; } return sqrt(s);
}
static std::vector<float> rand_k() {
    std::mt19937 g(1234); std::normal_distribution<float> nd(0, 1);
    std::vector<float> k(HD*NH*NC); for (auto & x : k) x = nd(g); return k;
}
static std::vector<int32_t> poss(int base) { std::vector<int32_t> p(NC); for (int i = 0; i < NC; i++) p[i] = base + 3*i; return p; }
static std::vector<int32_t> add(std::vector<int32_t> p, int d) { for (auto & x : p) x += d; return p; }

// stored K = (optional hadamard over full head) (rope(K0,pos)); returns FLOAT (unquantized) values
static std::vector<float> fresh_float(const std::vector<float> & k0, const std::vector<int32_t> & pos, const rp & p, const std::vector<float> * rot) {
    return run([&](ggml_context * c, ggml_cgraph *) {
        ggml_tensor * k = inp(c, "k", GGML_TYPE_F32, HD, NH, NC); ggml_tensor * ps = inp(c, "p", GGML_TYPE_I32, NC);
        ggml_tensor * r = rope(c, k, ps, p);
        if (rot) { ggml_tensor * R = inp(c, "R", GGML_TYPE_F32, HD, HD); r = had(c, r, R); }
        return r;
    }, rot ? inputs_t{{"k", bytes(k0)}, {"p", bytes(pos)}, {"R", bytes(*rot)}} : inputs_t{{"k", bytes(k0)}, {"p", bytes(pos)}});
}
static std::vector<float> roundtrip(const std::vector<float> & f, ggml_type t) {
    std::vector<char> q = to_type(f, t); std::vector<float> r(f.size());
    const auto * tt = ggml_get_type_traits(t);
    if (t == GGML_TYPE_F32) return f;
    tt->to_float(q.data(), r.data(), f.size());
    return r;
}
// the K-shift graph. mode 0 = non-quantized path (kv-cache.cpp:2018, rope_ext_inplace on n_rot view);
// mode 1 = quantized path as shipped (cast -> hadamard(nrot derived) -> rope -> hadamard -> cpy, on the n_rot view, :2005-2020);
// mode 2 = candidate fix: same but on the FULL head view.
static std::vector<float> kshift(const std::vector<float> & kcache_f, ggml_type store, const std::vector<int32_t> & shift,
        const rp & p, int mode, int nrot_h) {
    std::vector<float> R = gen_had(nrot_h);
    inputs_t in = {{"kc", to_type(kcache_f, store)}, {"s", bytes(shift)}, {"R", bytes(R)}};
    return run([&](ggml_context * c, ggml_cgraph * gf) {
        ggml_tensor * kc = inp(c, "kc", store, HD, NH, NC); ggml_tensor * sh = inp(c, "s", GGML_TYPE_I32, NC);
        ggml_tensor * Rm = inp(c, "R", GGML_TYPE_F32, nrot_h, nrot_h);
        const int64_t w = mode == 2 ? HD : NROT;
        ggml_tensor * v = ggml_view_3d(c, kc, w, NH, NC, ggml_row_size(store, HD), ggml_row_size(store, HD*NH), 0);
        ggml_tensor * node;
        if (mode == 0) {
            node = ggml_rope_ext_inplace(c, v, sh, nullptr, NROT, GGML_ROPE_TYPE_NEOX, N_CTX_ORIG, BASE, p.fs, p.ext, p.attn, 32.0f, 1.0f);
        } else {
            ggml_tensor * t = ggml_cast(c, v, GGML_TYPE_F32);
            t = had(c, t, Rm);
            t = ggml_rope_ext(c, t, sh, nullptr, NROT, GGML_ROPE_TYPE_NEOX, N_CTX_ORIG, BASE, p.fs, p.ext, p.attn, 32.0f, 1.0f);
            t = had(c, t, Rm);
            if (mode == 1) {
                // The shipped ggml_cpy(t, v) writes a NON-CONTIGUOUS q8_0 destination (n_rot view of a wider head); the CPU
                // backend aborts there (ggml-cpu/ops.cpp:323 "not implemented", requires ggml_is_contiguous(dst)). Return the
                // pre-cpy tensor so the host can emulate the store and the numerics of the shipped graph can still be measured.
                return t;
            }
            node = ggml_cpy(c, t, v);
        }
        ggml_build_forward_expand(gf, node);
        return ggml_view_3d(c, kc, HD, NH, NC, kc->nb[1], kc->nb[2], 0);
    }, in);
}

static int fails = 0;
static void verdict(const char * name, bool confirmed, const char * detail) {
    printf("VERDICT %s %s  (%s)\n", name, confirmed ? "CONFIRMED" : "REFUTED", detail);
}

int main() {
    ggml_backend_load_all();
    be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 8);
    const auto K0 = rand_k();
    const rp cfgs[2] = {PLAIN, YARN}; const char * cn[2] = {"noYaRN", "YaRN fs0.5 ext1"};
    char buf[512];

    // ---------------- H1 ----------------
    // f32 and f16 cache; shift by d repeatedly; compare norm of rotated dims against fresh rope at the final position.
    printf("== H1 (mscale re-applied per shift); f16 K, d=37, p0=100\n");
    bool h1_conf = false; double h1_yarn_ratio1 = 0, h1_plain_ratio1 = 0;
    for (int ci = 0; ci < 2; ci++) for (ggml_type st : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        std::vector<float> cache = roundtrip(fresh_float(K0, poss(100), cfgs[ci], nullptr), st);
        for (int n = 1; n <= 4; n++) {
            cache = roundtrip(kshift(cache, st, std::vector<int32_t>(NC, 37), cfgs[ci], 0, 0 + 256 /*unused*/), st);
            auto fresh = fresh_float(K0, add(poss(100), 37*n), cfgs[ci], nullptr);
            double r_rot = nrm(cache, true)/nrm(fresh, true), r_all = nrm(cache, false)/nrm(fresh, false);
            printf("  %-16s %s n=%d  ratio(rotdims)=%.5f  ratio(fullhead)=%.5f  expected 1.0693^n=%.5f  maxabs=%.3e\n", cn[ci],
                   st == GGML_TYPE_F16 ? "f16" : "f32", n, r_rot, r_all, powf(1.0f + 0.1f*logf(2.0f), n), maxabs(cache, fresh));
            if (st == GGML_TYPE_F16 && n == 1) { if (ci == 1) h1_yarn_ratio1 = r_rot; else h1_plain_ratio1 = r_rot; }
            if (ci == 1 && st == GGML_TYPE_F16 && fabs(r_rot - 1.0) > 1e-3) h1_conf = true;
        }
    }
    snprintf(buf, sizeof buf, "rotdim norm ratio after 1 shift: noYaRN %.5f, YaRN %.5f; threshold 1.000+/-1e-3", h1_plain_ratio1, h1_yarn_ratio1);
    verdict("H1", h1_conf, buf);
    {   // fix check: shift with attn_factor = 1/(1+0.1 ln(1/fs)) as in build_rope_shift after the fix (src/llama-kv-cache.cpp)
        rp fx = YARN; fx.attn = 1.0f / (1.0f + 0.1f*logf(1.0f / YARN.fs));
        std::vector<float> cache = fresh_float(K0, poss(100), YARN, nullptr);
        for (int n = 1; n <= 4; n++) {
            cache = kshift(cache, GGML_TYPE_F32, std::vector<int32_t>(NC, 37), fx, 0, 256);
            auto fresh = fresh_float(K0, add(poss(100), 37*n), YARN, nullptr);
            printf("  [fix formula] YaRN f32 n=%d ratio(rotdims)=%.5f maxabs=%.3e\n", n, nrm(cache, true)/nrm(fresh, true), maxabs(cache, fresh));
        }
    }

    // ---------------- H2 ----------------
    // 3 cells at positions {10,11,13}... (use NC cells), seq_div G=2 via llama_kv_cells; K roped at p_old, shifted by cells.get_shift, vs fresh at p_old/2.
    printf("== H2 (pos_div shift sign); noYaRN, f32 K, G=2 (uses the live llama_kv_cells::pos_div: CONFIRMED on ffc1bac82, REFUTED once the sign fix commit is present)\n");
    double h2_err = 0, h2_err_signflip = 0, h2_err_add = 0;
    {
        llama_kv_cells cells; cells.resize(NC);
        std::vector<int32_t> pold(NC), pnew(NC);
        llama_seq_id sid = 0;
        for (int i = 0; i < NC; i++) { pold[i] = 10 + 3*i + (i == 0 ? 0 : 0); cells.pos_set(i, pold[i]); cells.seq_add(i, sid); }
        for (int i = 0; i < NC; i++) cells.pos_div(i, 2);
        std::vector<int32_t> sh(NC), shneg(NC), exp_delta(NC);
        for (int i = 0; i < NC; i++) { pnew[i] = pold[i]/2; sh[i] = cells.get_shift(i); shneg[i] = -sh[i]; exp_delta[i] = pnew[i] - pold[i]; }
        printf("  cell: p_old p_new get_shift expected(floor(p/G)-p)\n");
        for (int i = 0; i < NC; i++) printf("    %2d  %3d %3d  %4d  %4d\n", i, pold[i], pnew[i], sh[i], exp_delta[i]);
        auto cache = fresh_float(K0, pold, PLAIN, nullptr);
        auto fresh = fresh_float(K0, pnew, PLAIN, nullptr);
        h2_err         = maxabs(kshift(cache, GGML_TYPE_F32, sh,        PLAIN, 0, 256), fresh);
        h2_err_signflip= maxabs(kshift(cache, GGML_TYPE_F32, shneg,     PLAIN, 0, 256), fresh);
        h2_err_add     = maxabs(kshift(cache, GGML_TYPE_F32, exp_delta, PLAIN, 0, 256), fresh);
        // control: pos_add by d=-5 path
        llama_kv_cells c2; c2.resize(NC);
        for (int i = 0; i < NC; i++) { c2.pos_set(i, pold[i]); c2.seq_add(i, sid); c2.pos_add(i, -5); }
        std::vector<int32_t> sh2(NC); for (int i = 0; i < NC; i++) sh2[i] = c2.get_shift(i);
        double ctl = maxabs(kshift(cache, GGML_TYPE_F32, sh2, PLAIN, 0, 256), fresh_float(K0, add(pold, -5), PLAIN, nullptr));
        printf("  max-abs: get_shift as shipped %.3e | negated get_shift %.3e | expected floor(p/G)-p %.3e | pos_add(-5) control %.3e\n",
               h2_err, h2_err_signflip, h2_err_add, ctl);
    }
    snprintf(buf, sizeof buf, "max-abs err as shipped %.3e (threshold 1e-6); with expected delta %.3e", h2_err, h2_err_add);
    verdict("H2", h2_err > 1e-6, buf);

    // ---------------- H3 ----------------
    printf("== H3 (Hadamard nrot vs partial rotary); head=%d n_rot=%d, derived nrot=%d, noYaRN, d=37, p0=100\n", HD, NROT, derive_nrot(HD));
    const int nrot_h = derive_nrot(HD);
    auto R = gen_had(nrot_h);
    double worst_ratio = 0, f16_err = 0;
    for (int ci = 0; ci < 1; ci++) {
        auto truth = fresh_float(K0, add(poss(100), 37), cfgs[ci], &R);   // float Hadamard-domain K at p+d
        auto rt = roundtrip(truth, GGML_TYPE_Q8_0);                         // plain q8_0 round trip of fresh data
        double e_rt_rms = rms(rt, truth), e_rt_max = maxabs(rt, truth);
        auto cached = roundtrip(fresh_float(K0, poss(100), cfgs[ci], &R), GGML_TYPE_Q8_0);
        for (int mode = 1; mode <= 2; mode++) {
            auto raw = kshift(cached, GGML_TYPE_Q8_0, std::vector<int32_t>(NC, 37), cfgs[ci], mode, nrot_h);
            std::vector<float> sh;
            if (mode == 1) {   // host emulation of the strided store: replace first NROT dims of each head row, requantize those blocks
                sh = cached;
                for (int r = 0; r < NH*NC; r++) {
                    std::vector<float> seg(raw.begin() + r*NROT, raw.begin() + (r+1)*NROT);
                    seg = roundtrip(seg, GGML_TYPE_Q8_0);
                    std::copy(seg.begin(), seg.end(), sh.begin() + r*HD);
                }
            } else sh = roundtrip(raw, GGML_TYPE_Q8_0);
            double e_rms = rms(sh, truth), e_max = maxabs(sh, truth);
            printf("  q8_0 %-26s: rms err vs truth %.3e (roundtrip %.3e, ratio %.2f) | max-abs %.3e (roundtrip %.3e, ratio %.2f)\n",
                   mode == 1 ? "shipped (n_rot view)" : "candidate (full-head view)", e_rms, e_rt_rms, e_rms/e_rt_rms, e_max, e_rt_max, e_max/e_rt_max);
            if (mode == 1) worst_ratio = std::max(e_rms/e_rt_rms, e_max/e_rt_max);
        }
        // f16 control (no hadamard in f16 path): fresh f16 vs shifted f16, Hadamard-free domain
        auto t16 = fresh_float(K0, add(poss(100), 37), cfgs[ci], nullptr);
        auto c16 = roundtrip(fresh_float(K0, poss(100), cfgs[ci], nullptr), GGML_TYPE_F16);
        auto s16 = roundtrip(kshift(c16, GGML_TYPE_F16, std::vector<int32_t>(NC, 37), cfgs[ci], 0, 256), GGML_TYPE_F16);
        f16_err = maxabs(s16, t16);
        printf("  f16 control (non-quantized path): max-abs vs fresh %.3e (f16 roundtrip %.3e)\n", f16_err, maxabs(roundtrip(t16, GGML_TYPE_F16), t16));
    }
    snprintf(buf, sizeof buf, "q8_0 shifted error / plain roundtrip error = %.2f (threshold 2.0); nrot derived = %d vs n_rot = %d", worst_ratio, nrot_h, NROT);
    verdict("H3", worst_ratio > 2.0, buf);
    (void) fails;
    ggml_backend_free(be);
    return 0;
}
