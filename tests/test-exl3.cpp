// EXL3 CPU executor tests (EXL3-CPU-DESIGN.md section 7).
//
//  P2  every ISA tier == the scalar tier, bit-exact (GEMV int8 path and GEMM fp32 path)
//  P3  operator output vs an independent canonical reference (materialized_weight_fp32_fma_v1,
//      written here from the EXL3-1 formulas, not from exl3.cpp): rel-L2 envelopes
//      GEMV (int8 activations) <= 0.02, GEMM (fp32) <= 2e-3
//  P1  real fixtures (EXL3-1 real-mul1-k3/k4): operator output vs the fixture's canonical outputs
//
// usage: test-exl3 [fixtures_dir] ; fixtures default to the research repo checkout.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
const char * ggml_cpu_exl3_isa(void);
void         ggml_cpu_exl3_set_isa(const char * name);
void         ggml_cpu_exl3_set_gemv_max_rows(int rows);
void         ggml_cpu_exl3_set_splitk_target(int target);
}

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

// machine-readable proposition results (--json): consumed by tools/exl3/emit_verifier.py
struct Prop { std::string fixture, path, proposition; double value, envelope; bool pass; };
static std::vector<Prop> g_props;
static int g_isa_cmp = 0, g_isa_bad = 0, g_thr_cmp = 0, g_thr_bad = 0;
static double g_worst[2] = { 0, 0 };

// ---------------------------------------------------------------- independent reference

static float h2f(uint16_t h) { return ggml_fp16_to_fp32(h); }
static uint16_t f2h(float f) { return ggml_fp32_to_fp16(f); }
static float hround(float f) { return h2f(f2h(f)); }

static int state_index(int r, int c) {   // EXL3-1 oracle formula
    return ((c % 8) * 4 + (r % 8) / 2) * 8 + r % 2 + (r / 8) * 2 + (c / 8) * 4;
}

static uint16_t window(const uint16_t * tile, int K, int i) {   // EXL3-1: MSB-first u32 LE stream
    const int L = 256 * K;
    const int start = ((i + 257) * K - 16) % L;
    uint32_t v = 0;
    for (int b = 0; b < 16; ++b) {
        const int bit = (start + b) % L;
        const int word = bit / 32;
        uint32_t w;
        std::memcpy(&w, tile + 2 * word, 4);
        v = (v << 1) | ((w >> (31 - bit % 32)) & 1);
    }
    return uint16_t(v);
}

static float mul1(uint16_t s) {
    const uint32_t x = uint32_t(s) * 0x83DCD12Du;
    const int bs = int(x & 255) + int((x >> 8) & 255) + int((x >> 16) & 255) + int(x >> 24);
    const double exact = (1024.0 + bs) * (1774.0 / 262144.0) + (-10.3828125);
    return hround(float(exact));   // exact value fits fp32; one rounding to fp16
}

struct Mat {
    int K, k, n;
    std::vector<uint16_t> native;   // [k/16][n/16][16K]
    std::vector<uint16_t> suh, svh; // fp16
};

// W[k][n] canonical: fp16 RNE after each stage, H products ascending fp32 FMA
static std::vector<float> materialize(const Mat & m) {
    const int k = m.k, n = m.n, nn = n / 16;
    std::vector<float> q(size_t(k) * n);
    for (int kt = 0; kt < k / 16; ++kt) {
        for (int nt = 0; nt < nn; ++nt) {
            const uint16_t * tile = m.native.data() + (size_t(kt) * nn + nt) * 16 * m.K;
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    q[size_t(kt * 16 + r) * n + nt * 16 + c] = mul1(window(tile, m.K, state_index(r, c)));
                }
            }
        }
    }
    const float s = float(1.0 / std::sqrt(128.0));
    auto H = [&](int i, int j) { return (__builtin_popcount(i & j) & 1) ? -s : s; };
    std::vector<float> L(q.size()), W(q.size());
    for (int base = 0; base < k; base += 128) {
        for (int c = 0; c < n; ++c) {
            for (int i = 0; i < 128; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < 128; ++j) acc = std::fma(H(i, j), q[size_t(base + j) * n + c], acc);
                L[size_t(base + i) * n + c] = hround(hround(acc) * h2f(m.suh[base + i]));
            }
        }
    }
    for (int r = 0; r < k; ++r) {
        for (int base = 0; base < n; base += 128) {
            for (int i = 0; i < 128; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < 128; ++j) acc = std::fma(H(i, j), L[size_t(r) * n + base + j], acc);
                W[size_t(r) * n + base + i] = hround(hround(acc) * h2f(m.svh[base + i]));
            }
        }
    }
    return W;
}

static std::vector<uint16_t> to_band8(const Mat & m) {
    const int nk = m.k / 16, nn = m.n / 16, words = 16 * m.K;
    std::vector<uint16_t> out(m.native.size());
    for (int kt = 0; kt < nk; ++kt) {
        for (int nt = 0; nt < nn; ++nt) {
            const size_t src = (size_t(kt) * nn + nt) * words;
            const size_t dst = ((size_t(nt / 8) * nk * 8) + size_t(kt) * 8 + nt % 8) * words;
            std::memcpy(out.data() + dst, m.native.data() + src, words * 2);
        }
    }
    return out;
}

static Mat random_mat(std::mt19937 & rng, int K, int k, int n) {
    Mat m{K, k, n, {}, {}, {}};
    m.native.resize(size_t(k / 16) * (n / 16) * 16 * K);
    for (auto & w : m.native) w = uint16_t(rng());
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    for (int i = 0; i < k; ++i) m.suh.push_back(f2h((rng() & 1 ? -1.f : 1.f) * 0.012f * ud(rng)));
    for (int i = 0; i < n; ++i) m.svh.push_back(f2h((rng() & 1 ? -1.f : 1.f) * ud(rng)));
    return m;
}

static double rel_l2(const float * a, const float * b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { num += double(a[i] - b[i]) * (a[i] - b[i]); den += double(b[i]) * b[i]; }
    return std::sqrt(num / std::max(den, 1e-30));
}

// ---------------------------------------------------------------- ggml runner

struct Run { std::vector<float> y; };

// dense: x [k, rows] -> y [n, rows];  experts: x [k, nx, T], ids [nu, T] -> y [n, nu, T]
static std::vector<float> run_ggml(const std::vector<Mat> & ms, const std::vector<float> & x, int rows,
                                   const std::vector<int32_t> * ids, int nu, int nx, int n_threads) {
    const Mat & m0 = ms[0];
    const int E = int(ms.size());
    const bool grouped = ids != nullptr;
    size_t mem = 0;
    for (const auto & m : ms) mem += m.native.size() * 2 + (m.k + m.n) * 2;
    mem += x.size() * 4 * 2 + size_t(m0.n) * (grouped ? nu * rows : rows) * 4 + (ids ? ids->size() * 4 : 0);
    mem += 64 * 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const ggml_type type = ggml_type(GGML_TYPE_EXL3_1 + m0.K - 1);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, type, m0.k, m0.n, E);
    ggml_tensor * suh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, m0.k, E);
    ggml_tensor * svh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, m0.n, E);
    const size_t per = ggml_nbytes(w) / E;
    for (int e = 0; e < E; ++e) {
        const auto b8 = to_band8(ms[e]);
        if (b8.size() * 2 != per) { std::printf("size mismatch %zu vs %zu\n", b8.size() * 2, per); std::abort(); }
        std::memcpy((char *) w->data + e * per, b8.data(), per);
        std::memcpy((char *) suh->data + e * suh->nb[1], ms[e].suh.data(), m0.k * 2);
        std::memcpy((char *) svh->data + e * svh->nb[1], ms[e].svh.data(), m0.n * 2);
    }
    ggml_tensor * out;
    if (!grouped) {
        ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m0.k, rows);
        std::memcpy(xt->data, x.data(), x.size() * 4);
        ggml_tensor * w2 = ggml_view_2d(ctx, w, m0.k, m0.n, w->nb[1], 0);
        ggml_tensor * suh2 = ggml_view_1d(ctx, suh, m0.k, 0);
        ggml_tensor * svh2 = ggml_view_1d(ctx, svh, m0.n, 0);
        out = ggml_mul_mat(ctx, w2, xt);
        out->src[2] = svh2;
        out->src[3] = suh2;
    } else {
        ggml_tensor * xt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, m0.k, nx, rows);
        std::memcpy(xt->data, x.data(), x.size() * 4);
        ggml_tensor * it = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nu, rows);
        std::memcpy(it->data, ids->data(), ids->size() * 4);
        out = ggml_mul_mat_id(ctx, w, xt, it);
        out->src[3] = svh;
        out->src[4] = suh;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    std::vector<float> y(ggml_nelements(out));
    std::memcpy(y.data(), out->data, y.size() * 4);
    ggml_free(ctx);
    return y;
}

static std::vector<float> ref_dense(const std::vector<float> & W, int k, int n, const float * x) {
    std::vector<float> y(n);
    for (int c = 0; c < n; ++c) {
        double a = 0;
        for (int r = 0; r < k; ++r) a += double(x[r]) * W[size_t(r) * n + c];
        y[c] = float(a);
    }
    return y;
}

static const char * ISAS[] = { "scalar", "vnni", "vbmi" };

// ---------------------------------------------------------------- tests

static void test_dense(std::mt19937 & rng, int K, int k, int n, int rows, int threads) {
    Mat m = random_mat(rng, K, k, n);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(size_t(k) * rows);
    for (auto & v : x) v = nd(rng);
    const auto W = materialize(m);
    for (int path = 0; path < 2; ++path) {   // 0: GEMV int8, 1: GEMM fp32
        ggml_cpu_exl3_set_gemv_max_rows(path == 0 ? 1 << 20 : 0);
        std::vector<float> y_scalar;
        for (const char * isa : ISAS) {
            ggml_cpu_exl3_set_isa(isa);
            if (std::strcmp(ggml_cpu_exl3_isa(), isa) != 0) continue;   // tier not on this CPU
            auto y = run_ggml({ m }, x, rows, nullptr, 0, 0, threads);
            if (y_scalar.empty()) {
                y_scalar = y;
                double worst = 0;
                for (int r = 0; r < rows; ++r) {
                    const auto yr = ref_dense(W, k, n, x.data() + size_t(r) * k);
                    worst = std::max(worst, rel_l2(y.data() + size_t(r) * n, yr.data(), n));
                }
                const double env = path == 0 ? 0.02 : 2e-3;
                g_worst[path] = std::max(g_worst[path], worst);
                CHECK(worst <= env, "dense K%d %dx%d rows%d %s: rel-L2 vs canonical %.3e > %.0e",
                      K, k, n, rows, path ? "gemm" : "gemv", worst, env);
                std::printf("  dense K%d %4dx%-4d rows %2d %s: canonical rel-L2 %.2e\n", K, k, n, rows, path ? "gemm" : "gemv", worst);
            } else {
                const bool same = std::memcmp(y.data(), y_scalar.data(), y.size() * 4) == 0;
                g_isa_cmp++; g_isa_bad += !same;
                CHECK(same, "dense K%d %dx%d rows%d %s: ISA %s differs from scalar", K, k, n, rows, path ? "gemm" : "gemv", isa);
                // thread-count independence (split-K S and unit plans never depend on nth)
                const auto y1 = run_ggml({ m }, x, rows, nullptr, 0, 0, 1);
                g_thr_cmp++; g_thr_bad += std::memcmp(y.data(), y1.data(), y.size() * 4) != 0;
                CHECK(std::memcmp(y.data(), y1.data(), y.size() * 4) == 0,
                      "dense K%d %dx%d rows%d %s ISA %s: %d threads differs from 1 thread", K, k, n, rows, path ? "gemm" : "gemv", isa, threads);
            }
        }
    }
    ggml_cpu_exl3_set_isa(nullptr);
    ggml_cpu_exl3_set_gemv_max_rows(8);
}

static void test_experts(std::mt19937 & rng, int K, int k, int n, int E, int nu, int T, int nx, int threads) {
    std::vector<Mat> ms;
    for (int e = 0; e < E; ++e) ms.push_back(random_mat(rng, K, k, n));
    std::vector<std::vector<float>> Ws;
    for (auto & m : ms) Ws.push_back(materialize(m));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(size_t(k) * nx * T);
    for (auto & v : x) v = nd(rng);
    std::vector<int32_t> ids(size_t(nu) * T);
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> perm(E);
        for (int e = 0; e < E; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int s = 0; s < nu; ++s) ids[size_t(t) * nu + s] = perm[s];
    }
    for (int path = 0; path < 2; ++path) {
        ggml_cpu_exl3_set_gemv_max_rows(path == 0 ? 1 << 20 : 0);
        std::vector<float> y_scalar;
        for (const char * isa : ISAS) {
            ggml_cpu_exl3_set_isa(isa);
            if (std::strcmp(ggml_cpu_exl3_isa(), isa) != 0) continue;
            auto y = run_ggml(ms, x, T, &ids, nu, nx, threads);
            if (y_scalar.empty()) {
                y_scalar = y;
                double worst = 0;
                for (int t = 0; t < T; ++t) {
                    for (int s = 0; s < nu; ++s) {
                        const int e = ids[size_t(t) * nu + s];
                        const float * xr = x.data() + (size_t(t) * nx + (s % nx)) * k;
                        const auto yr = ref_dense(Ws[e], k, n, xr);
                        worst = std::max(worst, rel_l2(y.data() + (size_t(t) * nu + s) * n, yr.data(), n));
                    }
                }
                const double env = path == 0 ? 0.02 : 2e-3;
                g_worst[path] = std::max(g_worst[path], worst);
                CHECK(worst <= env, "experts K%d %dx%d E%d nu%d T%d nx%d %s: rel-L2 %.3e > %.0e",
                      K, k, n, E, nu, T, nx, path ? "gemm" : "gemv", worst, env);
                std::printf("  experts K%d %4dx%-4d E%d nu%d T%2d nx%d %s: canonical rel-L2 %.2e\n",
                            K, k, n, E, nu, T, nx, path ? "gemm" : "gemv", worst);
            } else {
                const bool same = std::memcmp(y.data(), y_scalar.data(), y.size() * 4) == 0;
                g_isa_cmp++; g_isa_bad += !same;
                CHECK(same, "experts K%d T%d %s: ISA %s differs from scalar", K, T, path ? "gemm" : "gemv", isa);
            }
        }
    }
    ggml_cpu_exl3_set_isa(nullptr);
    ggml_cpu_exl3_set_gemv_max_rows(8);
}

// minimal parsing of the EXL3-1 fixture json: "activations" and reference "operator_outputs"
static std::vector<std::vector<double>> json_matrix(const std::string & s, const std::string & key) {
    std::vector<std::vector<double>> out;
    size_t p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return out;
    p = s.find('[', p);
    int depth = 0;
    std::vector<double> row;
    std::string num;
    for (size_t i = p; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '[') { depth++; continue; }
        if (c == ']' || c == ',') {
            if (!num.empty()) { row.push_back(std::stod(num)); num.clear(); }
            if (c == ']') {
                if (depth == 2) { out.push_back(row); row.clear(); }
                if (--depth == 0) break;
            }
            continue;
        }
        if (!std::isspace((unsigned char) c)) num += c;
    }
    return out;
}

static std::vector<char> slurp(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void test_fixture(const std::string & dir, int K) {
    const auto tr = slurp(dir + "/trellis.bin");
    const auto su = slurp(dir + "/suh.bin");
    const auto sv = slurp(dir + "/svh.bin");
    const auto js = slurp(dir + "/fixture.json");
    if (tr.empty() || su.empty() || sv.empty() || js.empty()) {
        CHECK(false, "real fixture %s missing (fixtures never skip silently)", dir.c_str());
        return;
    }
    Mat m{K, 128, 128, {}, {}, {}};
    m.native.resize(tr.size() / 2);
    std::memcpy(m.native.data(), tr.data(), tr.size());
    m.suh.resize(128); std::memcpy(m.suh.data(), su.data(), 256);
    m.svh.resize(128); std::memcpy(m.svh.data(), sv.data(), 256);
    const std::string s(js.begin(), js.end());
    const auto acts = json_matrix(s, "activations");
    const auto outs = json_matrix(s, "operator_outputs");
    CHECK(acts.size() == 3 && outs.size() == 3, "fixture parse %s", dir.c_str());
    if (acts.size() != outs.size() || acts.empty()) return;
    const int rows = int(acts.size());
    std::vector<float> x;
    for (auto & r : acts) for (double v : r) x.push_back(float(v));
    // independent reference reproduces the fixture outputs (proves the test's own oracle)
    const auto W = materialize(m);
    double self = 0;
    for (int r = 0; r < rows; ++r) {
        const auto yr = ref_dense(W, 128, 128, x.data() + r * 128);
        std::vector<float> fo(outs[r].begin(), outs[r].end());
        self = std::max(self, rel_l2(yr.data(), fo.data(), 128));
    }
    CHECK(self < 1e-6, "fixture %s: test oracle vs fixture outputs %.3e", dir.c_str(), self);
    for (int path = 0; path < 2; ++path) {
        ggml_cpu_exl3_set_gemv_max_rows(path == 0 ? 1 << 20 : 0);
        auto y = run_ggml({ m }, x, rows, nullptr, 0, 0, 4);
        double worst = 0;
        for (int r = 0; r < rows; ++r) {
            std::vector<float> fo(outs[r].begin(), outs[r].end());
            worst = std::max(worst, rel_l2(y.data() + r * 128, fo.data(), 128));
        }
        const double env = path == 0 ? 0.02 : 2e-3;
        CHECK(worst <= env, "fixture %s %s: rel-L2 %.3e > %.0e", dir.c_str(), path ? "gemm" : "gemv", worst, env);
        g_props.push_back({ dir.substr(dir.find_last_of('/') + 1), path ? "exl3_cpu_gemm_fp32" : "exl3_cpu_gemv_int8",
                            std::string("llama.cpp CPU EXL3 executor (") + (path ? "fp32 GEMM prefill" : "int8 GEMV decode") +
                            ") output on the real fixture activations is within relative-L2 " + (path ? "2e-3" : "0.02") +
                            " of the fixture's canonical operator outputs (materialized_weight_fp32_fma_v1)", worst, env, worst <= env });
        std::printf("  real fixture %s %s: rel-L2 vs canonical outputs %.2e (oracle self-check %.1e)\n",
                    dir.substr(dir.find_last_of('/') + 1).c_str(), path ? "gemm" : "gemv", worst, self);
    }
    ggml_cpu_exl3_set_gemv_max_rows(8);
}

// cache-resident single-matrix timing (diagnostic only, not a measurement-constitution result)
static void bench(int K, int k, int n, int rows, int threads, int iters) {
    std::mt19937 rng(7);
    Mat m = random_mat(rng, K, k, n);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(size_t(k) * rows);
    for (auto & v : x) v = nd(rng);
    size_t mem = m.native.size() * 2 + size_t(k + n) * 2 + x.size() * 4 + size_t(n) * rows * 4 + 64 * 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, ggml_type(GGML_TYPE_EXL3_1 + K - 1), k, n);
    ggml_tensor * suh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, k);
    ggml_tensor * svh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    const auto b8 = to_band8(m);
    std::memcpy(w->data, b8.data(), ggml_nbytes(w));
    std::memcpy(suh->data, m.suh.data(), k * 2);
    std::memcpy(svh->data, m.svh.data(), n * 2);
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
    std::memcpy(xt->data, x.data(), x.size() * 4);
    ggml_tensor * out = ggml_mul_mat(ctx, w, xt);
    out->src[2] = svh;
    out->src[3] = suh;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_cplan plan = ggml_graph_plan(gf, threads, nullptr);
    std::vector<uint8_t> work(plan.work_size + 64);
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);   // warm
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < iters; ++i) ggml_graph_compute(gf, &plan);
    const double us = double(ggml_time_us() - t0) / iters;
    const double bytes = double(ggml_nbytes(w));
    std::printf("bench K%d %5dx%-5d rows %d threads %d isa %s: %.1f us/op, %.2f GB/s weights, %.2f Gw/s\n",
                K, k, n, rows, threads, ggml_cpu_exl3_isa(), us, bytes / us / 1e3, double(k) * n / us / 1e3);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        ggml_time_init();
        const int threads = argc > 2 ? std::atoi(argv[2]) : 1;
        for (const char * isa : { "vbmi", "vnni", "scalar" }) {
            ggml_cpu_exl3_set_isa(isa);
            if (std::strcmp(ggml_cpu_exl3_isa(), isa) != 0) continue;
            const int it = std::strcmp(isa, "scalar") == 0 ? 3 : 300;
            bench(3, 2560, 640, 1, threads, it);
            bench(3, 640, 2560, 1, threads, it);
            bench(4, 2560, 640, 1, threads, it);
            bench(5, 2560, 1280, 1, threads, it);
            bench(3, 2560, 640, 4, threads, it);
            bench(3, 2560, 640, 8, threads, it);
        }
        ggml_cpu_exl3_set_isa(nullptr);
        ggml_cpu_exl3_set_gemv_max_rows(1 << 20);
        std::printf("GEMV int8 path:\n");
        for (int r : { 2, 4, 8, 12, 16 }) bench(3, 2560, 640, r, threads, 100);
        ggml_cpu_exl3_set_gemv_max_rows(0);
        std::printf("GEMM fp32 path:\n");
        for (int r : { 2, 4, 8, 12, 16, 32 }) bench(3, 2560, 640, r, threads, 100);
        bench(5, 2560, 1280, 128, threads, 5);
        return 0;
    }
    std::string fix = "/workspace/repos/epyc-inference-research/scripts/kernel_rnd/exl3/fixtures";
    std::string json_out;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--json" && i + 1 < argc) json_out = argv[++i];
        else fix = argv[i];
    }
    std::printf("EXL3 CPU executor tests; best ISA on this host: %s\n", ggml_cpu_exl3_isa());
    std::mt19937 rng(1234);
    test_fixture(fix + "/real-mul1-k3", 3);
    test_fixture(fix + "/real-mul1-k4", 4);
    for (int K : { 1, 2, 3, 4, 5, 6, 7, 8 }) {
        test_dense(rng, K, 256, 256, 1, 3);
    }
    for (int rows : { 2, 3, 4, 5, 8, 13 }) {
        test_dense(rng, 3, 384, 256, rows, 4);
    }
    test_dense(rng, 5, 640, 384, 1, 7);
    test_dense(rng, 5, 640, 384, 6, 2);
    test_experts(rng, 3, 256, 128, 6, 3, 1, 1, 5);    // decode: shared input per token
    test_experts(rng, 3, 128, 256, 6, 3, 1, 3, 3);    // decode: per-slot input (down_proj shape)
    test_experts(rng, 3, 256, 128, 6, 3, 7, 1, 4);    // small batch
    test_experts(rng, 4, 128, 256, 8, 4, 24, 4, 6);   // prefill-ish, mixed group sizes
    // split-K heavy shapes (kb_n = 20: S up to 10)
    test_dense(rng, 3, 2560, 256, 1, 6);
    test_dense(rng, 5, 2560, 128, 2, 5);
    test_experts(rng, 3, 2560, 128, 4, 2, 1, 1, 6);
    test_experts(rng, 3, 2560, 128, 5, 3, 2, 3, 4);
    // the same with split-K disabled (S = 1 path)
    ggml_cpu_exl3_set_splitk_target(1);
    std::printf("split-K disabled:\n");
    test_dense(rng, 3, 384, 256, 1, 4);
    test_dense(rng, 5, 640, 384, 3, 3);
    test_experts(rng, 3, 256, 128, 6, 3, 1, 1, 5);
    ggml_cpu_exl3_set_splitk_target(192);
    g_props.push_back({ "synthetic-k1-k8", "exl3_cpu_isa_tiers",
                        std::string("every enabled ISA tier (") + ggml_cpu_exl3_isa() + " and below) produces bit-identical outputs to the scalar tier on all synthetic dense and expert cases (GEMV and GEMM paths)",
                        double(g_isa_bad), 0.0, g_isa_bad == 0 && g_isa_cmp > 0 });
    g_props.push_back({ "synthetic-k1-k8", "exl3_cpu_thread_count",
                        "outputs are bit-identical between the multi-thread and the single-thread run on all synthetic dense cases",
                        double(g_thr_bad), 0.0, g_thr_bad == 0 && g_thr_cmp > 0 });
    g_props.push_back({ "synthetic-k1-k8", "exl3_cpu_gemv_int8",
                        "int8 GEMV output is within relative-L2 0.02 of an independent canonical reference on all synthetic cases",
                        g_worst[0], 0.02, g_worst[0] <= 0.02 });
    g_props.push_back({ "synthetic-k1-k8", "exl3_cpu_gemm_fp32",
                        "fp32 GEMM output is within relative-L2 2e-3 of an independent canonical reference on all synthetic cases",
                        g_worst[1], 2e-3, g_worst[1] <= 2e-3 });
    if (!json_out.empty()) {
        FILE * f = std::fopen(json_out.c_str(), "w");
        std::fprintf(f, "{\"isa\": \"%s\", \"failures\": %d, \"isa_comparisons\": %d, \"thread_comparisons\": %d, \"props\": [", ggml_cpu_exl3_isa(), g_fail, g_isa_cmp, g_thr_cmp);
        for (size_t i = 0; i < g_props.size(); ++i) {
            const Prop & p = g_props[i];
            std::fprintf(f, "%s{\"fixture\": \"%s\", \"path\": \"%s\", \"proposition\": \"%s\", \"value\": %.9g, \"envelope\": %.9g, \"pass\": %s}",
                         i ? ", " : "", p.fixture.c_str(), p.path.c_str(), p.proposition.c_str(), p.value, p.envelope, p.pass ? "true" : "false");
        }
        std::fprintf(f, "]}\n");
        std::fclose(f);
    }
    if (g_fail) {
        std::printf("%d FAILURES\n", g_fail);
        return 1;
    }
    std::printf("ALL PASS\n");
    return 0;
}
