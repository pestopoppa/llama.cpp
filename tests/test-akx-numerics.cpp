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
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

int main(int argc, char ** argv) {
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
    // Residency hold (GPU legs): AKX_HOLD_S=<seconds> keeps a 256 MiB device buffer alive that long after the
    // receipts, so an external sampler (KFD pid + VRAM) observes the process DURING, not after. Off by default.
    if (const char * h = getenv("AKX_HOLD_S")) {
        ggml_backend_buffer_t ballast = ggml_backend_alloc_buffer(be, (size_t) 256 << 20);
        fprintf(stderr, "AKX_HOLD ballast=%s for %s s\n", ballast ? "ok" : "FAILED", h);
        if (ballast) {
            struct timespec ts = { (time_t) atoi(h), 0 };
            nanosleep(&ts, nullptr);
            ggml_backend_buffer_free(ballast);
        }
    }
    ggml_backend_free(be);
    return 0;
}
