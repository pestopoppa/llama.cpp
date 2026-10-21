// DS41-W5.1a / DS41-T3.2 -- the block quantize-dequantize (QDQ) op, without weights.
//
// Normative: DeepSeek-V4.1-Flash-1017 inference/kernel.py act_quant_kernel / fp4_quant_kernel
// (inplace=True). src/llama-qdq.h restates the formulas.
//
// What this pins:
//   1. the e4m3fn and e2m1 element rounders equal an INDEPENDENT oracle -- nearest code from an
//      enumerated code table, ties to the even code -- on every grid point, every midpoint and
//      the f32 neighbours of both, plus a random sweep;
//   2. the block op (all three modes) equals a second, independently written block oracle on
//      random, outlier, tiny, huge, all-zero and exact-power-of-two-amax blocks;
//   3. hand-computed golden values (worked out in the comments);
//   4. the ggml graph op equals the scalar reference BIT FOR BIT at 1 and at 7 threads, on a
//      contiguous and on a permuted (non-contiguous) input;
//   5. LLAMA_QDQ_NONE inserts no node (the "off" graph is the unmodified graph);
//   6. the LLAMA_DSV41_QDQ site parser, including the abort-worthy unknown token.

#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-qdq.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_pass = 0;
static int n_fail = 0;

static void check(bool ok, const char * what) {
    if (ok) {
        n_pass++;
    } else {
        n_fail++;
        std::printf("FAIL: %s\n", what);
    }
}

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float    from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// ---------------------------------------------------------------------------------------------
// 1. independent element oracle: enumerate the non-negative finite codes, pick the nearest,
//    ties to the even code. Sign is symmetric.
// ---------------------------------------------------------------------------------------------

struct code_table {
    std::vector<double> v; // value of code i, ascending (codes are monotone in value for sign 0)
};

// e4m3fn: 4 exponent bits (bias 7), 3 mantissa bits, no inf, 0x7F = NaN -> codes 0..0x7E
static code_table make_e4m3() {
    code_table t;
    for (int c = 0; c <= 0x7E; ++c) {
        const int e = (c >> 3) & 0xF;
        const int m = c & 0x7;
        t.v.push_back(e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m/8.0, e - 7));
    }
    return t;
}

// e2m1: 2 exponent bits (bias 1), 1 mantissa bit -> codes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6
static code_table make_e2m1() {
    code_table t;
    for (int c = 0; c <= 7; ++c) {
        const int e = (c >> 1) & 0x3;
        const int m = c & 0x1;
        t.v.push_back(e == 0 ? 0.5*m : std::ldexp(1.0 + m/2.0, e - 1));
    }
    return t;
}

static float oracle_round(const code_table & t, float x) {
    const double a = std::fabs((double) x);
    int    best   = 0;
    double best_d = std::fabs(a - t.v[0]);
    for (int c = 1; c < (int) t.v.size(); ++c) {
        const double d = std::fabs(a - t.v[c]);
        if (d < best_d || (d == best_d && (c % 2) == 0)) {
            best = c;
            best_d = d;
        }
    }
    const float r = (float) t.v[best];
    return std::signbit(x) ? -r : r;
}

static bool same_bits(float a, float b) {
    return bits(a) == bits(b);
}

static void test_element_rounders() {
    const code_table e4m3 = make_e4m3();
    const code_table e2m1 = make_e2m1();

    struct fmt { const code_table * t; float (*f)(float); const char * name; };
    const fmt fmts[] = {
        { &e4m3, llama_qdq_round_e4m3, "e4m3" },
        { &e2m1, llama_qdq_round_e2m1, "e2m1" },
    };

    for (const auto & F : fmts) {
        const auto & v = F.t->v;
        std::vector<float> probes;
        for (size_t i = 0; i < v.size(); ++i) {
            const float g = (float) v[i];
            probes.push_back(g);
            probes.push_back(std::nextafter(g, 0.0f));
            probes.push_back(std::nextafter(g, 1e9f));
            if (i + 1 < v.size()) {
                const float mid = (float) (0.5*(v[i] + v[i + 1])); // exact: few-bit values
                probes.push_back(mid);
                probes.push_back(std::nextafter(mid, 0.0f));
                probes.push_back(std::nextafter(mid, 1e9f));
            }
        }
        // f32 subnormals and the smallest normals collapse onto 0
        probes.push_back(from_bits(1));
        probes.push_back(from_bits(0x00800000));
        probes.push_back(0.0f);

        const float vmax = (float) v.back();
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::uniform_int_distribution<int>    ex(-20, 0);
        for (int i = 0; i < 200000; ++i) {
            probes.push_back(std::ldexp(u(rng), ex(rng)) * vmax);
        }

        int bad = 0;
        for (float p : probes) {
            if (std::fabs(p) > vmax) {
                continue;
            }
            for (float s : { p, -p }) {
                const float got = F.f(s);
                const float exp = oracle_round(*F.t, s);
                if (!same_bits(got, exp) && !(got == 0.0f && exp == 0.0f)) {
                    if (bad < 5) {
                        std::printf("  %s(%a) = %a, oracle %a\n", F.name, s, got, exp);
                    }
                    bad++;
                }
            }
        }
        char msg[96];
        std::snprintf(msg, sizeof(msg), "%s rounder == enumerated-code oracle (%zu probes)", F.name, probes.size());
        check(bad == 0, msg);
    }

    // signed zero survives (the reference casts -tiny to fp8 -0)
    check(std::signbit(llama_qdq_round_e4m3(-1e-30f)), "e4m3(-tiny) is -0");
    check(std::signbit(llama_qdq_round_e2m1(-0.1f)),   "e2m1(-0.1) is -0");
}

// ---------------------------------------------------------------------------------------------
// 2. independent block oracle (double precision where the reference's f32 steps are exact,
//    frexp for ceil(log2), the code-table rounders)
// ---------------------------------------------------------------------------------------------

// ceil(log2(x)) for a NORMAL positive f32, from frexp (x = m * 2^e, m in [0.5, 1))
static int ceil_log2_normal(float x) {
    int e;
    const float m = std::frexp(x, &e);
    return m == 0.5f ? e - 1 : e;
}

static void oracle_block(llama_qdq_mode mode, const float * x, float * y, int B) {
    static const code_table e4m3 = make_e4m3();
    static const code_table e2m1 = make_e2m1();

    float amax = 0.0f;
    for (int i = 0; i < B; ++i) {
        amax = std::fmax(amax, std::fabs(x[i]));
    }

    if (mode == LLAMA_QDQ_FP8_E4M3_UE8M0_B32) {
        amax = std::fmax(amax, 1e-4f);
        const float r = amax * (float) (1.0/448.0);          // f32 multiply, as the kernel
        const float s = std::ldexp(1.0f, ceil_log2_normal(r)); // r is normal for amax >= 1e-4
        for (int i = 0; i < B; ++i) {
            float t = x[i] / s;
            t = std::fmin(std::fmax(t, -448.0f), 448.0f);
            y[i] = oracle_round(e4m3, t) * s;
        }
    } else if (mode == LLAMA_QDQ_FP4_E2M1_UE8M0_B32) {
        amax = std::fmax(amax, std::ldexp(6.0f, -126));
        const float r = amax * (float) (1.0/6.0);
        // fast_log2_ceil of an f32 subnormal is -126; only the floor itself can get there
        const float s = r < std::ldexp(1.0f, -126) ? std::ldexp(1.0f, -126) : std::ldexp(1.0f, ceil_log2_normal(r));
        for (int i = 0; i < B; ++i) {
            float t = x[i] / s;
            t = std::fmin(std::fmax(t, -6.0f), 6.0f);
            y[i] = oracle_round(e2m1, t) * s;
        }
    } else {
        amax = std::fmax(amax, std::ldexp(6.0f, -9));
        const float s = oracle_round(e4m3, std::fmin(amax / 6.0f, 448.0f));
        for (int i = 0; i < B; ++i) {
            float t = x[i] / s;
            t = std::fmin(std::fmax(t, -6.0f), 6.0f);
            y[i] = oracle_round(e2m1, t) * s;
        }
    }
}

static const llama_qdq_mode MODES[] = {
    LLAMA_QDQ_FP8_E4M3_UE8M0_B32,
    LLAMA_QDQ_FP4_E2M1_UE8M0_B32,
    LLAMA_QDQ_FP4_E2M1_E4M3_B16,
};

static std::vector<float> make_data(int kind, int n, std::mt19937 & rng) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::student_t_distribution<float> st(3.0f);
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) {
        switch (kind) {
            case 0: x[i] = g(rng); break;                                   // gauss
            case 1: x[i] = st(rng); break;                                  // heavy tail
            case 2: x[i] = g(rng) * ((i % 97) == 0 ? 40.0f : 1.0f); break;  // outlier channels
            case 3: x[i] = g(rng) * 1e-7f; break;                           // below the amax floors
            case 4: x[i] = g(rng) * 3000.0f; break;                         // past e4m3 saturation
            case 5: x[i] = 0.0f; break;                                     // all zero
            case 6: x[i] = (i % 16 == 3) ? 448.0f : g(rng); break;          // power-of-two-ish amax
            case 7: x[i] = std::ldexp(g(rng), (i % 40) - 20); break;        // mixed binades
        }
    }
    return x;
}

static void test_block_oracle() {
    std::mt19937 rng(42);
    for (llama_qdq_mode mode : MODES) {
        const int B = (int) llama_qdq_block_size(mode);
        int bad = 0;
        int total = 0;
        for (int kind = 0; kind < 8; ++kind) {
            const int n = 4096;
            std::vector<float> x = make_data(kind, n, rng);
            std::vector<float> y(n), z(n);
            llama_qdq_ref(mode, x.data(), y.data(), n);
            for (int b = 0; b < n; b += B) {
                oracle_block(mode, x.data() + b, z.data() + b, B);
            }
            for (int i = 0; i < n; ++i) {
                total++;
                if (!same_bits(y[i], z[i]) && !(y[i] == 0.0f && z[i] == 0.0f)) {
                    if (bad < 5) {
                        std::printf("  %s kind %d [%d]: x=%a got %a oracle %a\n",
                                llama_qdq_mode_name(mode), kind, i, x[i], y[i], z[i]);
                    }
                    bad++;
                }
            }
        }
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s block == independent oracle (%d values)", llama_qdq_mode_name(mode), total);
        check(bad == 0, msg);

        // in-place (x == y) is allowed and gives the same bits
        std::vector<float> x = make_data(0, 256, rng);
        std::vector<float> y(256);
        llama_qdq_ref(mode, x.data(), y.data(), 256);
        llama_qdq_ref(mode, x.data(), x.data(), 256);
        check(std::memcmp(x.data(), y.data(), 256*sizeof(float)) == 0, "in-place == out-of-place");

        // idempotent: a dequantized block is already on the grid
        std::vector<float> w(256);
        llama_qdq_ref(mode, y.data(), w.data(), 256);
        std::snprintf(msg, sizeof(msg), "%s idempotent", llama_qdq_mode_name(mode));
        check(std::memcmp(w.data(), y.data(), 256*sizeof(float)) == 0, msg);
    }
}

// ---------------------------------------------------------------------------------------------
// 3. hand-computed goldens
// ---------------------------------------------------------------------------------------------

static void test_goldens() {
    // FP8, block amax 1.0: 1/448 = 2^-8.81 -> s = 2^-8.
    //   0.3  -> 76.8  (binade [64,128), quantum 8)  -> 80  -> 0.3125
    //   0.01 -> 2.56  (binade [2,4),   quantum .25) -> 2.5 -> 0.009765625
    //   1.0  -> 256 exactly                                -> 1.0
    {
        float x[32] = {0};
        x[0] = 1.0f; x[1] = 0.3f; x[2] = 0.01f; x[3] = -0.3f;
        float y[32];
        llama_qdq_ref(LLAMA_QDQ_FP8_E4M3_UE8M0_B32, x, y, 32);
        check(y[0] == 1.0f,         "fp8 golden 1.0");
        check(y[1] == 0.3125f,      "fp8 golden 0.3 -> 0.3125");
        check(y[2] == 0.009765625f, "fp8 golden 0.01 -> 0.009765625");
        check(y[3] == -0.3125f,     "fp8 golden -0.3 -> -0.3125");
        check(y[4] == 0.0f,         "fp8 golden 0 -> 0");
    }
    // FP8, all-zero block: floor 1e-4 -> s = 2^-22, output all zero
    {
        float x[32] = {0};
        float y[32];
        llama_qdq_ref(LLAMA_QDQ_FP8_E4M3_UE8M0_B32, x, y, 32);
        bool ok = true;
        for (float v : y) ok = ok && v == 0.0f;
        check(ok, "fp8 golden all-zero block");
    }
    // FP4 ue8m0, block amax 1.0: 1/6 = 2^-2.58 -> s = 2^-2.
    //   0.3   -> 1.2 -> 1.0  -> 0.25
    //   0.6   -> 2.4 -> 2    -> 0.5
    //   0.875 -> 3.5 -> tie, even code is 4 -> 1.0
    //   0.625 -> 2.5 -> tie, even code is 2 -> 0.5
    {
        float x[32] = {0};
        x[0] = 1.0f; x[1] = 0.3f; x[2] = 0.6f; x[3] = 0.875f; x[4] = 0.625f;
        float y[32];
        llama_qdq_ref(LLAMA_QDQ_FP4_E2M1_UE8M0_B32, x, y, 32);
        check(y[0] == 1.0f,  "fp4/ue8m0 golden 1.0");
        check(y[1] == 0.25f, "fp4/ue8m0 golden 0.3 -> 0.25");
        check(y[2] == 0.5f,  "fp4/ue8m0 golden 0.6 -> 0.5");
        check(y[3] == 1.0f,  "fp4/ue8m0 golden 0.875 -> 1.0 (tie to even)");
        check(y[4] == 0.5f,  "fp4/ue8m0 golden 0.625 -> 0.5 (tie to even)");
    }
    // FP4 e4m3/16, block amax 1.0: 1/6 = 0.1667 -> e4m3 (binade [.125,.25), quantum 2^-6):
    //   10.67 -> 11 -> s = 0.171875.
    //   1.0 -> 5.818 -> e2m1 6 -> 1.03125 (the amax itself comes back LARGER)
    //   0.5 -> 2.909 -> 3      -> 0.515625
    {
        float x[16] = {0};
        x[0] = 1.0f; x[1] = 0.5f;
        float y[16];
        llama_qdq_ref(LLAMA_QDQ_FP4_E2M1_E4M3_B16, x, y, 16);
        check(y[0] == 1.03125f,  "fp4/e4m3 golden 1.0 -> 1.03125");
        check(y[1] == 0.515625f, "fp4/e4m3 golden 0.5 -> 0.515625");
    }
    // blocks are independent: a huge value in block 1 does not touch block 0
    {
        float x[64] = {0};
        x[1] = 0.3f; x[40] = 1000.0f; x[0] = 1.0f;
        float y[64];
        llama_qdq_ref(LLAMA_QDQ_FP8_E4M3_UE8M0_B32, x, y, 64);
        check(y[1] == 0.3125f, "fp8 blocks are independent");
    }
}

// ---------------------------------------------------------------------------------------------
// 4./5. the graph op
// ---------------------------------------------------------------------------------------------

static std::vector<float> run_graph(llama_qdq_mode mode, const std::vector<float> & x,
        int64_t ne0, int64_t ne1, int64_t ne2, bool permuted, int n_threads, int * n_nodes_out) {
    ggml_init_params params = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    std::memcpy(a->data, x.data(), x.size()*sizeof(float));

    ggml_tensor * in = a;
    if (permuted) {
        // swap dims 1 and 2: same ne0 rows, non-contiguous view
        in = ggml_permute(ctx, a, 0, 2, 1, 3);
    }

    ggml_tensor * out = llama_build_qdq(ctx, in, mode);
    if (mode == LLAMA_QDQ_NONE) {
        check(out == in, "NONE returns its input (no node)");
    }
    out = ggml_cont(ctx, out); // a sink node so the graph is never empty

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (n_nodes_out) {
        *n_nodes_out = ggml_graph_n_nodes(gf);
    }

    std::vector<float> r(ggml_nelements(out));
    std::memcpy(r.data(), out->data, r.size()*sizeof(float));
    ggml_free(ctx);
    return r;
}

static void test_graph_op() {
    std::mt19937 rng(7);
    const int64_t ne0 = 512, ne1 = 6, ne2 = 3; // 512 = DS41 head dim; 6 x 3 rows
    const int64_t n = ne0*ne1*ne2;

    for (llama_qdq_mode mode : MODES) {
        std::vector<float> x = make_data(2, (int) n, rng);

        // scalar reference over the logical (possibly permuted) layout
        std::vector<float> ref(n);
        llama_qdq_ref(mode, x.data(), ref.data(), n);

        // permuted input: logical row (i1, i2) of the view is physical row (i2, i1) of a
        std::vector<float> xp(n);
        for (int64_t i2 = 0; i2 < ne1; ++i2) {        // view ne2 = a ne1
            for (int64_t i1 = 0; i1 < ne2; ++i1) {    // view ne1 = a ne2
                std::memcpy(&xp[(i2*ne2 + i1)*ne0], &x[(i1*ne1 + i2)*ne0], ne0*sizeof(float));
            }
        }
        std::vector<float> refp(n);
        llama_qdq_ref(mode, xp.data(), refp.data(), n);

        for (int nt : { 1, 7 }) {
            int nn = 0;
            const std::vector<float> g = run_graph(mode, x, ne0, ne1, ne2, false, nt, &nn);
            char msg[128];
            std::snprintf(msg, sizeof(msg), "%s graph op == scalar ref, contiguous, %d threads", llama_qdq_mode_name(mode), nt);
            check(std::memcmp(g.data(), ref.data(), n*sizeof(float)) == 0, msg);

            const std::vector<float> gp = run_graph(mode, x, ne0, ne1, ne2, true, nt, nullptr);
            std::snprintf(msg, sizeof(msg), "%s graph op == scalar ref, permuted input, %d threads", llama_qdq_mode_name(mode), nt);
            check(std::memcmp(gp.data(), refp.data(), n*sizeof(float)) == 0, msg);
        }
    }

    // off: the input passes through untouched and only the sink node is in the graph
    {
        std::vector<float> x = make_data(0, (int) n, rng);
        int nn = -1;
        const std::vector<float> g = run_graph(LLAMA_QDQ_NONE, x, ne0, ne1, ne2, false, 4, &nn);
        check(std::memcmp(g.data(), x.data(), n*sizeof(float)) == 0, "NONE output is bit-identical to the input");
        check(nn == 1, "NONE adds no node (graph = the sink only)");
    }
}

// ---------------------------------------------------------------------------------------------
// 6. site parser
// ---------------------------------------------------------------------------------------------

static void test_parser() {
    llama_dsv41_qdq_sites s;
    std::string bad;

    check(llama_dsv41_qdq_parse(nullptr, s) && !s.any(), "parse(nullptr) = off");
    check(llama_dsv41_qdq_parse("", s) && !s.any(), "parse('') = off");
    check(llama_dsv41_qdq_parse("none", s) && !s.any(), "parse('none') = off");
    check(llama_dsv41_qdq_parse("0", s) && !s.any(), "parse('0') = off");

    check(llama_dsv41_qdq_parse("wo_a", s) && s.wo_a && !s.kv_win && !s.kv_comp && !s.idx_k && !s.idx_q,
            "parse('wo_a') = wo_a only");
    check(llama_dsv41_qdq_parse(" KV ", s) && !s.wo_a && s.kv_win && s.kv_comp && s.idx_k && s.idx_q,
            "parse(' KV ') = the T3.2 set");
    check(llama_dsv41_qdq_parse("all", s) && s.wo_a && s.kv_win && s.kv_comp && s.idx_k && s.idx_q,
            "parse('all') = everything");
    check(llama_dsv41_qdq_parse("idx_q,kv_comp", s) && !s.wo_a && !s.kv_win && s.kv_comp && !s.idx_k && s.idx_q,
            "parse('idx_q,kv_comp')");

    check(!llama_dsv41_qdq_parse("wo_a,kvwin", s, &bad) && bad == "kvwin" && !s.any(),
            "parse rejects an unknown token and returns all-off");
    check(!llama_dsv41_qdq_parse("1", s, &bad) && bad == "1", "parse rejects '1' (ambiguous)");
}

int main() {
    test_element_rounders();
    test_block_oracle();
    test_goldens();
    test_graph_op();
    test_parser();

    std::printf("test-dsv41-qdq: pass=%d fail=%d\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
