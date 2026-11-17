// test-akx-numerics.cpp: [AKX-ALL-31.NATIVE] native producer of AK_NUMERICS_V1 receipts.
//
// Usage: test-akx-numerics <inputs.txt> [backend-name (default CPU)]
// The inputs file is written by loop/numerics_native.py (the host-side generator owns the planted
// rows and the integer accumulator vectors; this binary owns only the kernel evaluation):
//   MAX <rows> <cols>\n  rows*cols tokens (nan/inf/-inf/number)
//   DOT <rows> <n>\n     rows*n tokens (A) then rows*n tokens (B)
// The real backend kernels are driven through ggml graphs:
//   row max          GGML_OP_POOL_2D (MAX), kernel (cols,1), stride (cols,1) over a [cols,rows] F32 tensor
//   row dot (fp32)   GGML_OP_MUL_MAT of one [n,1] A row and one [n,1] B row (F32 x F32)
// Three receipts are printed, one per variant, each `AK_NUMERICS_V1 {json}` on its own line:
//   incumbent          the unmodified kernels (must pass both gates)
//   control_clamp_max  inputs are ggml_clamp'd to +-3e38 first, so +-Inf become finite (must fail non_finite)
//   control_f16_dot    the dot result is rounded through F16 (must fail the fp32 accumulator gate)
// Exit 0 when all three receipts were produced; judging is the Python side's job.
//
// [AKX-ALL-31.EDGE] Usage: test-akx-numerics --edge <cases.txt> [backend-name (default CPU)]
// Each line `EDGE <MUL_MAT|MUL_MAT_ID> <ggml type name> <k> <m> <n_mats> <n_used> <w1,w2,..> <family> <seed>`
// (written by loop/numerics_native.py edge-cases). Inputs are generated HERE with the counter-based
// splitmix64 scheme `akx-edge-splitmix64-v1`, bit-identical to loop/numerics_edge.py (which checks each
// receipt's `probe`). Families: near_f16_max, f16_subnormal, cancel_rows, tile_const. Weights are
// quantized with ggml_quantize_chunk; the reference is float64 over the DEQUANTIZED weights (type
// traits to_float), so what is measured is the kernel's activation handling and accumulation. Per case,
// three variants run through the real backend kernel at every width: incumbent, twin_row (one 32-block
// of one seeded weight row negated) and twin_tile (that block on a seeded 16-row tile) -- the twins are
// compared with the CLEAN reference and must go red. One `AK_NUMERICS_EDGE_V1 {json}` line per
// (case, width, variant) with nmse, max_abs (+ argmax row/col), rms_ref, nonfinite; then
// `AK_NUMERICS_EDGE_DONE {json}`. Optional env AKX_THREADS (CPU backend threads and host threads).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

static bool read_vals(FILE * f, size_t n, std::vector<float> & out) {
    char tok[64];
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        if (fscanf(f, "%63s", tok) != 1) return false;
        out[i] = (float) strtod(tok, nullptr);   // strtod parses nan / inf / -inf
    }
    return true;
}

static void emit_num(std::string & s, float v) {
    char b[64];
    if (std::isnan(v))      s += "\"nan\"";
    else if (std::isinf(v)) s += v > 0 ? "\"inf\"" : "\"-inf\"";
    else { snprintf(b, sizeof b, "%.9g", v); s += b; }
}

struct run_t { std::vector<float> mx, dt; };

static bool eval(ggml_backend_t be, int64_t R, int64_t C, const std::vector<float> & mx_in,
                 int64_t DR, int64_t N, const std::vector<float> & A, const std::vector<float> & B,
                 bool clamp_ctl, bool f16_ctl, run_t & out) {
    ggml_init_params ip = { /*mem_size*/ 1u << 24, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * X  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, R);
    ggml_tensor * Am = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, DR);
    ggml_tensor * Bm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, DR);
    ggml_tensor * in = X;
    if (clamp_ctl) in = ggml_clamp(ctx, ggml_dup(ctx, X), -3e38f, 3e38f);
    ggml_tensor * P = ggml_pool_2d(ctx, in, GGML_OP_POOL_MAX, (int) C, 1, (int) C, 1, 0, 0);
    ggml_set_output(P);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, P);
    std::vector<ggml_tensor *> dots;
    for (int64_t i = 0; i < DR; i++) {
        ggml_tensor * a = ggml_view_2d(ctx, Am, N, 1, Am->nb[1], i * Am->nb[1]);
        ggml_tensor * b = ggml_view_2d(ctx, Bm, N, 1, Bm->nb[1], i * Bm->nb[1]);
        ggml_tensor * d = ggml_mul_mat(ctx, a, b);
        if (f16_ctl) d = ggml_cast(ctx, ggml_cast(ctx, d, GGML_TYPE_F16), GGML_TYPE_F32);
        ggml_set_output(d);
        ggml_build_forward_expand(g, d);
        dots.push_back(d);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(X, mx_in.data(), 0, ggml_nbytes(X));
    ggml_backend_tensor_set(Am, A.data(), 0, ggml_nbytes(Am));
    ggml_backend_tensor_set(Bm, B.data(), 0, ggml_nbytes(Bm));
    bool ok = ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.mx.resize(R);
        ggml_backend_tensor_get(P, out.mx.data(), 0, sizeof(float) * R);
        out.dt.resize(DR);
        for (int64_t i = 0; i < DR; i++) ggml_backend_tensor_get(dots[i], &out.dt[i], 0, sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static void receipt(const char * variant, const char * dev, const run_t & r, const run_t & ref_for_other, bool use_mx, bool use_dt) {
    // a control only perturbs ONE op; the other field carries the incumbent value so only the
    // intended gate can see the planted defect.
    const std::vector<float> & mx = use_mx ? r.mx : ref_for_other.mx;
    const std::vector<float> & dt = use_dt ? r.dt : ref_for_other.dt;
    std::string s = "AK_NUMERICS_V1 {\"schema\": \"epyc.autokernel.numerics_gates.v1\", \"variant\": \"";
    s += variant; s += "\", \"backend\": \""; s += dev; s += "\", \"max_rows\": [";
    for (size_t i = 0; i < mx.size(); i++) { if (i) s += ", "; emit_num(s, mx[i]); }
    s += "], \"dot_rows\": [";
    for (size_t i = 0; i < dt.size(); i++) { if (i) s += ", "; emit_num(s, dt[i]); }
    s += "]}";
    puts(s.c_str());
    fflush(stdout);
}

// Residency hold (GPU legs): AKX_HOLD_S=<seconds> keeps a 256 MiB device buffer alive that long after the
// receipts, so an external sampler (KFD pid + VRAM) observes the process DURING, not after. Off by default.
static void akx_hold(ggml_backend_t be) {
    if (const char * h = getenv("AKX_HOLD_S")) {
        ggml_backend_buffer_t ballast = ggml_backend_alloc_buffer(be, (size_t) 256 << 20);
        fprintf(stderr, "AKX_HOLD ballast=%s for %s s\n", ballast ? "ok" : "FAILED", h);
        if (ballast) {
            struct timespec ts = { (time_t) atoi(h), 0 };
            nanosleep(&ts, nullptr);
            ggml_backend_buffer_free(ballast);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// [AKX-ALL-31.EDGE] edge-value families, per-element max |err| beside NMSE, seeded-defect twins.
// Generator, ids, twin selection and metrics mirror loop/numerics_edge.py; keep the two in sync.

static inline uint64_t akx_sm64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline uint64_t akx_base(uint64_t seed, uint64_t stream) { return akx_sm64((seed << 8) | stream); }

// u in [-1, 1), 2^-23 granularity (exact in float)
static inline float akx_u(uint64_t base, uint64_t i) {
    return (float) ((double) (akx_sm64(base + i) >> 40) * (1.0 / 8388608.0) - 1.0);
}

static void akx_parallel(int64_t n, int nth, const std::function<void(int64_t, int64_t)> & fn) {
    if (n <= 0) return;
    const int64_t t = std::max<int64_t>(1, std::min<int64_t>(nth, n));
    if (t == 1) { fn(0, n); return; }
    std::vector<std::thread> ts;
    ts.reserve((size_t) t);
    for (int64_t i = 0; i < t; i++) ts.emplace_back(fn, n * i / t, n * (i + 1) / t);
    for (auto & th : ts) th.join();
}

struct akx_edge_case {
    std::string op, type, family;
    int64_t k = 0, m = 0, n_mats = 1, n_used = 1;
    std::vector<int64_t> widths;
    uint64_t seed = 0;
};

static bool akx_read_cases(const char * path, std::vector<akx_edge_case> & out) {
    FILE * f = fopen(path, "r");
    if (!f) return false;
    char tag[16], op[32], ty[32], ws[128], fam[32];
    long long k, m, e, u;
    unsigned long long seed;
    int rc;
    while ((rc = fscanf(f, "%15s %31s %31s %lld %lld %lld %lld %127s %31s %llu",
                        tag, op, ty, &k, &m, &e, &u, ws, fam, &seed)) == 10) {
        if (strcmp(tag, "EDGE") != 0) { fclose(f); return false; }
        akx_edge_case c;
        c.op = op; c.type = ty; c.family = fam;
        c.k = k; c.m = m; c.n_mats = e; c.n_used = u; c.seed = seed;
        const char * p = ws;
        while (*p) {
            char * end = nullptr;
            const long long w = strtoll(p, &end, 10);
            if (end == p || w <= 0 || (*end != ',' && *end != '\0')) { fclose(f); return false; }
            c.widths.push_back(w);
            p = *end == ',' ? end + 1 : end;
        }
        out.push_back(c);
    }
    fclose(f);
    return rc == EOF && !out.empty();
}

static bool akx_type_by_name(const std::string & name, ggml_type & out) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * nm = ggml_type_name((ggml_type) t);
        if (nm && name == nm) { out = (ggml_type) t; return true; }
    }
    return false;
}

static int akx_family(const std::string & f) {
    if (f == "near_f16_max")  return 0;
    if (f == "f16_subnormal") return 1;
    if (f == "cancel_rows")   return 2;
    if (f == "tile_const")    return 3;
    return -1;
}

struct akx_twin { int64_t e = -1, row0 = 0, rows = 0, block = 0; };

// seeded row (tile=false) or 16-row tile (tile=true) of the expert token 0 / slot 0 routes to; the
// 32-block with the largest |sum_rows sum_j W*x0| for token 0, slot 0 (lowest index on ties)
static akx_twin akx_twin_target(const akx_edge_case & c, int64_t m, int64_t k, const std::vector<float> & W,
                                const std::vector<float> & X, const std::vector<int32_t> & ids, bool tile) {
    akx_twin tw;
    const uint64_t bt = akx_base(c.seed, 3);
    tw.e = ids[0];
    if (!tile) {
        tw.row0 = (int64_t) (akx_sm64(bt) % (uint64_t) m);
        tw.rows = 1;
    } else {
        const uint64_t tiles = (uint64_t) ((m + 15) / 16);
        tw.row0 = (int64_t) (akx_sm64(bt + 1) % tiles) * 16;
        tw.rows = std::min<int64_t>(16, m - tw.row0);
    }
    double best = -1.0;
    for (int64_t b = 0; b < k / 32; b++) {
        double s = 0.0;
        for (int64_t r = 0; r < tw.rows; r++) {
            const float * wr = W.data() + (tw.e * m + tw.row0 + r) * k + b * 32;
            for (int64_t j = 0; j < 32; j++) s += (double) wr[j] * (double) X[b * 32 + j];
        }
        if (std::fabs(s) > best) { best = std::fabs(s); tw.block = b; }
    }
    return tw;
}

struct akx_metric { double nmse = 0, max_abs = 0, rms_ref = 0, max_ref = 0; int64_t arg = 0, nonfinite = 0; };

static akx_metric akx_measure(const float * o, const double * ref, int64_t n) {
    akx_metric r;
    double se = 0.0, sr = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const double d = ref[i];
        sr += d * d;
        r.max_ref = std::max(r.max_ref, std::fabs(d));
        if (!std::isfinite(o[i])) { r.nonfinite++; continue; }
        const double e = std::fabs((double) o[i] - d);
        se += e * e;
        if (e > r.max_abs) { r.max_abs = e; r.arg = i; }
    }
    r.nmse = sr > 0 ? se / sr : (se == 0 ? 0.0 : (double) INFINITY);
    r.rms_ref = std::sqrt(sr / (double) std::max<int64_t>(1, n));
    return r;
}

static void emit_dbl(std::string & s, double v) {
    char b[64];
    if (std::isnan(v))      s += "\"nan\"";
    else if (std::isinf(v)) s += v > 0 ? "\"inf\"" : "\"-inf\"";
    else { snprintf(b, sizeof b, "%.17g", v); s += b; }
}

static void akx_edge_receipt(const char * dev, const akx_edge_case & c, int64_t n, int64_t nmax, const char * variant,
                             bool supported, const akx_metric & r, int64_t m, const float probe[4], const akx_twin & tw) {
    char b[256];
    std::string s = "AK_NUMERICS_EDGE_V1 {\"schema\": \"epyc.autokernel.numerics_edge.v1\", "
                    "\"gen\": \"akx-edge-splitmix64-v1\", \"backend\": \"";
    s += dev; s += "\", \"op\": \""; s += c.op; s += "\", \"type\": \""; s += c.type;
    snprintf(b, sizeof b, "\", \"k\": %" PRId64 ", \"m\": %" PRId64 ", \"n_mats\": %" PRId64 ", \"n_used\": %" PRId64
             ", \"n\": %" PRId64 ", \"nmax\": %" PRId64 ", \"family\": \"", c.k, c.m, c.n_mats, c.n_used, n, nmax);
    s += b; s += c.family;
    snprintf(b, sizeof b, "\", \"seed\": %" PRIu64 ", \"variant\": \"%s\", \"supported\": %s, \"nmse\": ",
             c.seed, variant, supported ? "true" : "false");
    s += b; emit_dbl(s, r.nmse);
    s += ", \"max_abs\": "; emit_dbl(s, r.max_abs);
    snprintf(b, sizeof b, ", \"arg_row\": %" PRId64 ", \"arg_col\": %" PRId64 ", \"rms_ref\": ", r.arg % m, r.arg / m);
    s += b; emit_dbl(s, r.rms_ref);
    s += ", \"max_ref\": "; emit_dbl(s, r.max_ref);
    snprintf(b, sizeof b, ", \"nonfinite\": %" PRId64 ", \"probe\": [", r.nonfinite);
    s += b;
    for (int i = 0; i < 4; i++) { if (i) s += ", "; emit_num(s, probe[i]); }
    s += "], \"twin\": ";
    if (tw.e < 0) {
        s += "null";
    } else {
        snprintf(b, sizeof b, "{\"expert\": %" PRId64 ", \"row0\": %" PRId64 ", \"rows\": %" PRId64 ", \"block\": %" PRId64 "}",
                 tw.e, tw.row0, tw.rows, tw.block);
        s += b;
    }
    s += "}";
    puts(s.c_str());
}

static bool akx_edge_one(ggml_backend_t be, const char * dev, const akx_edge_case & c, int nth, long & n_out) {
    ggml_type ty;
    if (!akx_type_by_name(c.type, ty)) { fprintf(stderr, "edge: unknown ggml type %s\n", c.type.c_str()); return false; }
    const bool mmid = c.op == "MUL_MAT_ID";
    const int fam = akx_family(c.family);
    const int64_t k = c.k, m = c.m, E = mmid ? c.n_mats : 1, U = mmid ? c.n_used : 1, slots = U;
    if ((!mmid && c.op != "MUL_MAT") || fam < 0 || k <= 0 || m <= 0 || k % 32 != 0 || k % ggml_blck_size(ty) != 0 ||
        (fam == 2 && k % 64 != 0) || E < 1 || U < 1 || U > E || c.widths.empty() ||
        (!mmid && (c.n_mats != 1 || c.n_used != 1))) {
        fprintf(stderr, "edge: invalid case %s %s k=%" PRId64 " m=%" PRId64 " %s\n", c.op.c_str(), c.type.c_str(), k, m, c.family.c_str());
        return false;
    }
    const int64_t nmax = *std::max_element(c.widths.begin(), c.widths.end());
    const int64_t nW = E * m * k, nX = nmax * slots * k, kb = k / 32, tr = (m + 15) / 16, h = k / 2;
    const uint64_t bw = akx_base(c.seed, 0), bx = akx_base(c.seed, 1), bd = akx_base(c.seed, 2);

    // --- inputs (numerics_edge.w_at / x_at)
    std::vector<float> W((size_t) nW), X((size_t) nX);
    akx_parallel(E * m, nth, [&](int64_t a, int64_t b) {
        for (int64_t er = a; er < b; er++) {
            const int64_t e = er / m, r = er % m;
            float * row = W.data() + er * k;
            for (int64_t col = 0; col < k; col++) {
                float w;
                if (fam == 3) {
                    w = akx_u(bw, (uint64_t) ((e * tr + r / 16) * kb + col / 32));
                } else if (fam == 2) {
                    w = akx_u(bw, (uint64_t) (er * k + (col < h ? col : col - h)));
                    if (col >= h) w = -w;
                } else {
                    w = akx_u(bw, (uint64_t) (er * k + col));
                    if (fam == 1) w *= 1.0f / 4096.0f;
                }
                row[col] = w;
            }
        }
    });
    for (int64_t ts = 0; ts < nmax * slots; ts++) {
        for (int64_t col = 0; col < k; col++) {
            float x;
            if (fam == 3) {
                x = akx_u(bx, (uint64_t) (ts * kb + col / 32));
            } else if (fam == 2) {
                const int64_t c0 = col < h ? col : col - h;
                x = akx_u(bx, (uint64_t) (ts * k + c0));
                if (col >= h) {
                    const float d = akx_u(bd, (uint64_t) (ts * k + c0)) * (1.0f / 4096.0f);
                    x = x + d;
                }
            } else {
                x = akx_u(bx, (uint64_t) (ts * k + col));
                if (fam == 0) x *= 61440.0f;
                if (fam == 1) x *= 1.0f / 1024.0f;
            }
            X[(size_t) (ts * k + col)] = x;
        }
    }
    const float probe[4] = { W[0], W[(size_t) nW - 1], X[0], X[(size_t) nX - 1] };
    // served_shape_cases.route_ids: row t, slot s -> n_mats - 1 - (29 t + s * step) % n_mats
    std::vector<int32_t> ids((size_t) (nmax * U));
    const int64_t step = (E / U) | 1;
    for (int64_t t = 0; t < nmax; t++)
        for (int64_t s = 0; s < U; s++)
            ids[(size_t) (t * U + s)] = mmid ? (int32_t) (E - 1 - (t * 29 + s * step) % E) : 0;

    // --- quantize (all rows), dequantize for the reference
    const size_t rs = ggml_row_size(ty, k);
    std::vector<uint8_t> Q(rs * (size_t) (E * m));
    std::vector<float> imat((size_t) k, 1.0f);
    const float * im = ggml_quantize_requires_imatrix(ty) ? imat.data() : nullptr;
    ggml_quantize_init(ty);
    akx_parallel(E * m, nth, [&](int64_t a, int64_t b) {
        ggml_quantize_chunk(ty, W.data(), Q.data(), a * k, b - a, k, im);
    });
    std::vector<float> Wd((size_t) nW);
    const ggml_type_traits * tt = ggml_get_type_traits(ty);
    if (tt && tt->to_float) {
        akx_parallel(E * m, nth, [&](int64_t a, int64_t b) {
            for (int64_t er = a; er < b; er++) tt->to_float(Q.data() + (size_t) er * rs, Wd.data() + er * k, k);
        });
    } else if (ty == GGML_TYPE_F32) {
        memcpy(Wd.data(), Q.data(), Q.size());
    } else {
        fprintf(stderr, "edge: type %s has no to_float\n", c.type.c_str());
        return false;
    }
    // float64 reference, layout [nmax][slots][m] == the ggml output layout ([m,n] / [m,n_used,n])
    std::vector<double> ref((size_t) (nmax * slots * m));
    akx_parallel(nmax * slots * m, nth, [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; i++) {
            const int64_t ts = i / m, r = i % m;
            const float * wr = Wd.data() + ((int64_t) ids[(size_t) ts] * m + r) * k;
            const float * xr = X.data() + ts * k;
            double s = 0.0;
            for (int64_t col = 0; col < k; col++) s += (double) wr[col] * (double) xr[col];
            ref[(size_t) i] = s;
        }
    });

    // --- graph: one weight tensor, every width as a prefix view of the activations (and ids)
    ggml_init_params ip = { /*mem_size*/ 1u << 24, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * A = mmid ? ggml_new_tensor_3d(ctx, ty, k, m, E) : ggml_new_tensor_2d(ctx, ty, k, m);
    ggml_tensor * B = mmid ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, U, nmax) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, nmax);
    ggml_tensor * I = mmid ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, nmax) : nullptr;
    std::vector<ggml_tensor *> outs;
    for (int64_t n : c.widths) {
        ggml_tensor * o;
        if (mmid) {
            ggml_tensor * bv = ggml_view_3d(ctx, B, k, U, n, B->nb[1], B->nb[2], 0);
            ggml_tensor * iv = ggml_view_2d(ctx, I, U, n, I->nb[1], 0);
            o = ggml_mul_mat_id(ctx, A, bv, iv);
        } else {
            ggml_tensor * bv = ggml_view_2d(ctx, B, k, n, B->nb[1], 0);
            o = ggml_mul_mat(ctx, A, bv);
        }
        ggml_set_output(o);
        outs.push_back(o);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "edge: buffer allocation failed\n"); ggml_free(ctx); return false; }
    if (ggml_nbytes(A) != Q.size()) {
        fprintf(stderr, "edge: weight bytes %zu != quantized %zu\n", ggml_nbytes(A), Q.size());
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    ggml_cgraph * g = ggml_new_graph(ctx);
    std::vector<bool> sup(outs.size(), false);
    bool any = false;
    for (size_t i = 0; i < outs.size(); i++) {
        sup[i] = ggml_backend_supports_op(be, outs[i]);
        if (sup[i]) { ggml_build_forward_expand(g, outs[i]); any = true; }
    }
    ggml_backend_tensor_set(B, X.data(), 0, ggml_nbytes(B));
    if (I) ggml_backend_tensor_set(I, ids.data(), 0, ggml_nbytes(I));

    static const char * vnames[3] = { "incumbent", "twin_row", "twin_tile" };
    bool ok = true;
    for (int v = 0; v < 3 && ok; v++) {
        akx_twin tw;
        std::vector<uint8_t> Qt;
        const std::vector<uint8_t> * qv = &Q;
        if (v > 0) {
            tw = akx_twin_target(c, m, k, W, X, ids, v == 2);
            Qt = Q;
            auto flip = [&]() {
                for (int64_t r = 0; r < tw.rows; r++) {
                    float * blk = W.data() + (tw.e * m + tw.row0 + r) * k + tw.block * 32;
                    for (int64_t j = 0; j < 32; j++) blk[j] = -blk[j];
                }
            };
            flip();
            for (int64_t r = 0; r < tw.rows; r++)
                ggml_quantize_chunk(ty, W.data(), Qt.data(), (tw.e * m + tw.row0 + r) * k, 1, k, im);
            flip();   // negation is exact: W is restored bit for bit
            qv = &Qt;
        }
        ggml_backend_tensor_set(A, qv->data(), 0, ggml_nbytes(A));
        if (any && ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "edge: graph compute failed (%s %s %s seed %" PRIu64 ")\n", c.op.c_str(), c.type.c_str(),
                    c.family.c_str(), c.seed);
            ok = false;
            break;
        }
        for (size_t i = 0; i < outs.size(); i++) {
            const int64_t n = c.widths[i], N = n * slots * m;
            akx_metric r;
            if (sup[i]) {
                std::vector<float> o((size_t) N);
                ggml_backend_tensor_get(outs[i], o.data(), 0, sizeof(float) * (size_t) N);
                r = akx_measure(o.data(), ref.data(), N);
            }
            akx_edge_receipt(dev, c, n, nmax, vnames[v], sup[i], r, m, probe, tw);
            n_out++;
        }
    }
    fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static int run_edge(const char * cases_path, const char * want) {
    std::vector<akx_edge_case> cases;
    if (!akx_read_cases(cases_path, cases)) { fprintf(stderr, "malformed edge cases file %s\n", cases_path); return 2; }
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(want);
    if (!dev) { fprintf(stderr, "backend %s not found\n", want); return 3; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { fprintf(stderr, "backend %s init failed\n", want); return 3; }
    int nth = (int) std::max(1u, std::thread::hardware_concurrency());
    if (const char * t = getenv("AKX_THREADS")) nth = std::max(1, atoi(t));
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    ggml_backend_set_n_threads_t set_threads = reg ?
        (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads") : nullptr;
    if (set_threads) set_threads(be, nth);
    const char * name = ggml_backend_dev_name(dev);
    long n_out = 0;
    for (const auto & c : cases) {
        if (!akx_edge_one(be, name, c, nth, n_out)) { ggml_backend_free(be); return 4; }
    }
    printf("AK_NUMERICS_EDGE_DONE {\"cases\": %zu, \"receipts\": %ld, \"backend\": \"%s\", \"threads\": %d}\n",
           cases.size(), n_out, name, nth);
    fflush(stdout);
    akx_hold(be);
    ggml_backend_free(be);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "--edge") == 0) {
        if (argc < 3) { fprintf(stderr, "usage: %s --edge cases.txt [backend]\n", argv[0]); return 2; }
        return run_edge(argv[2], argc > 3 ? argv[3] : "CPU");
    }
    if (argc < 2) { fprintf(stderr, "usage: %s inputs.txt [backend]\n", argv[0]); return 2; }
    const char * want = argc > 2 ? argv[2] : "CPU";
    FILE * f = fopen(argv[1], "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    long R = 0, C = 0, DR = 0, N = 0;
    std::vector<float> mx_in, A, B;
    if (fscanf(f, " MAX %ld %ld", &R, &C) != 2 || !read_vals(f, (size_t) (R * C), mx_in) ||
        fscanf(f, " DOT %ld %ld", &DR, &N) != 2 || !read_vals(f, (size_t) (DR * N), A) ||
        !read_vals(f, (size_t) (DR * N), B)) {
        fprintf(stderr, "malformed inputs file\n");
        return 2;
    }
    fclose(f);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(want);
    if (!dev) { fprintf(stderr, "backend %s not found\n", want); return 3; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { fprintf(stderr, "backend %s init failed\n", want); return 3; }
    const char * name = ggml_backend_dev_name(dev);
    run_t inc, c1, c2;
    if (!eval(be, R, C, mx_in, DR, N, A, B, false, false, inc) ||
        !eval(be, R, C, mx_in, DR, N, A, B, true,  false, c1) ||
        !eval(be, R, C, mx_in, DR, N, A, B, false, true,  c2)) {
        fprintf(stderr, "graph compute failed on %s\n", name);
        return 4;
    }
    receipt("incumbent",         name, inc, inc, true,  true);
    receipt("control_clamp_max", name, c1,  inc, true,  false);
    receipt("control_f16_dot",   name, c2,  inc, false, true);
    akx_hold(be);
    ggml_backend_free(be);
    return 0;
}
