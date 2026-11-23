// AK-Q38FN-CPU-MMID-GATEUP-FUSE: correctness test for the GGML_MMID_GATEUP_FUSE=1 single-token
// MUL_MAT_ID(up) + MUL_MAT_ID(gate) + GLU(swiglu split) fusion (iqk, non-Q8_0 types, flat slab).
// The knob is read once per process, so the A/B is two processes:
//   GGML_IQK=1                           test-mmid-gateup-fuse write <ref.bin>   -- unfused outputs
//   GGML_IQK=1 GGML_MMID_GATEUP_FUSE=1   test-mmid-gateup-fuse cmp   <ref.bin>   -- fused, NMSE vs ref
//   test-mmid-gateup-fuse                                                       -- ctest: every cell vs reference
// Every cell ALSO checks its output against a double-precision reference (to_float dequant . x, silu(g)*u),
// rel L2 < 0.02 (the Q8_K activation quantization bound), in both arms. A fused run must print the
// "[iqk] ACTIVE: GGML_MMID_GATEUP_FUSE" marker (checked by the caller / ctest regex).
// Cells: IQ3_S at the Q38FN shape (K=2560, N=640, 16 experts, 10 used) at 1/7/24 threads (odd thread
// counts put slab boundaries mid-expert), IQ4_XS small at 3 threads, and an IQ3_S repeated-expert cell
// (the fused path must decline and the unfused path must still be right).
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const double NMSE_MAX = 1e-5;  // the fused epilogue's silu is iqk's, not ggml_vec_silu bit-for-bit

struct Cell { ggml_type type; int64_t K, N, E; std::vector<int32_t> ids; int nt; const char * name; };

static std::vector<float> run_cell(const Cell & c, double & ref_rel) {
    const int64_t K = c.K, N = c.N, E = c.E, U = (int64_t) c.ids.size();
    const size_t wbytes = ggml_row_size(c.type, K) * N * E;
    ggml_init_params ip = { 2 * wbytes + 64u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(4321 + (int) c.type + (int) K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::vector<float>> wf(2);
    auto mk_w = [&](int w) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, c.type, K, N, E);
        wf[w].resize((size_t) K * N * E);
        for (auto & x : wf[w]) { x = 0.05f * nd(rng); }
        ggml_quantize_chunk(c.type, wf[w].data(), t->data, 0, N * E, K, nullptr);
        return t;
    };
    ggml_tensor * wu = mk_w(0);
    ggml_tensor * wg = mk_w(1);
    ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, 1);
    std::vector<float> xv(K);
    for (int64_t i = 0; i < K; ++i) { xv[i] = nd(rng); ((float *) x->data)[i] = xv[i]; }
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, 1);
    memcpy(ids->data, c.ids.data(), U * sizeof(int32_t));

    ggml_tensor * up   = ggml_mul_mat_id(ctx, wu, x, ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, wg, x, ids);
    ggml_tensor * out  = ggml_swiglu_split(ctx, gate, up);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, c.nt) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(2); }
    std::vector<float> r((float *) out->data, (float *) out->data + ggml_nelements(out));

    // double reference from the DEQUANTIZED weights (what the kernel actually multiplies)
    const ggml_type_traits * tt = ggml_get_type_traits(c.type);
    const size_t rs = ggml_row_size(c.type, K);
    std::vector<float> row(K);
    double e2 = 0, r2 = 0;
    for (int64_t k = 0; k < U; ++k) {
        const int64_t e = c.ids[k];
        for (int64_t n = 0; n < N; ++n) {
            double d[2];
            for (int w = 0; w < 2; ++w) {
                const char * src = (const char *) (w == 0 ? wu : wg)->data + (size_t) e * rs * N + (size_t) n * rs;
                tt->to_float(src, row.data(), K);
                double acc = 0; for (int64_t i = 0; i < K; ++i) { acc += (double) row[i] * xv[i]; }
                d[w] = acc;
            }
            const double ref = d[1] / (1.0 + std::exp(-d[1])) * d[0];
            const double got = r[(size_t) k * N + n];
            e2 += (got - ref) * (got - ref); r2 += ref * ref;
        }
    }
    ref_rel = std::sqrt(e2 / (r2 + 1e-30));
    ggml_free(ctx);
    return r;
}

static std::vector<Cell> cells() {
    std::vector<int32_t> ids10 = { 13, 2, 7, 0, 11, 5, 15, 9, 3, 8 };
    return {
        { GGML_TYPE_IQ3_S, 2560, 640, 16, ids10, 1,  "IQ3_S q38fn nt=1"  },
        { GGML_TYPE_IQ3_S, 2560, 640, 16, ids10, 7,  "IQ3_S q38fn nt=7"  },
        { GGML_TYPE_IQ3_S, 2560, 640, 16, ids10, 24, "IQ3_S q38fn nt=24" },
        { GGML_TYPE_IQ4_XS, 512, 256,  8, { 5, 1, 6, 2 }, 3, "IQ4_XS small nt=3" },
        { GGML_TYPE_IQ3_S, 512, 128,  8, { 3, 1, 3, 6 }, 4, "IQ3_S dup-expert nt=4 (declines)" },
    };
}

int main(int argc, char ** argv) {
    const char * fz = getenv("GGML_MMID_GATEUP_FUSE");
    fprintf(stderr, "GGML_MMID_GATEUP_FUSE=%s GGML_IQK=%s\n", fz ? fz : "(unset)", getenv("GGML_IQK") ? getenv("GGML_IQK") : "(unset)");
    const auto cs = cells();
    std::vector<std::vector<float>> outs;
    int pass = 0, fail = 0;
    for (const auto & c : cs) {
        double rel = 0;
        outs.push_back(run_cell(c, rel));
        const bool ok = std::isfinite(rel) && rel < 0.02;
        printf("%-34s ref_rel=%.3e %s\n", c.name, rel, ok ? "ok" : "FAIL");
        ok ? ++pass : ++fail;
    }
    if (argc == 3 && std::string(argv[1]) == "write") {
        FILE * f = fopen(argv[2], "wb");
        if (!f) { perror("open"); return 2; }
        for (const auto & o : outs) { fwrite(o.data(), sizeof(float), o.size(), f); }
        fclose(f);
    } else if (argc == 3 && std::string(argv[1]) == "cmp") {
        FILE * f = fopen(argv[2], "rb");
        if (!f) { perror("open"); return 2; }
        for (size_t c = 0; c < outs.size(); ++c) {
            std::vector<float> ref(outs[c].size());
            if (fread(ref.data(), sizeof(float), ref.size(), f) != ref.size()) { fprintf(stderr, "short ref file\n"); return 2; }
            double num = 0, den = 0, maxd = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double d = (double) outs[c][i] - ref[i];
                num += d * d; den += (double) ref[i] * ref[i]; maxd = std::fmax(maxd, std::fabs(d));
            }
            const double nmse = den > 0 ? num / den : num;
            const bool ok = std::isfinite(nmse) && nmse <= NMSE_MAX;
            printf("%-34s vs-ref nmse=%.3e maxabs=%.3e %s\n", cs[c].name, nmse, maxd, ok ? "ok" : "FAIL");
            ok ? ++pass : ++fail;
        }
        fclose(f);
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [write <file> | cmp <file>]\n", argv[0]);
        return 2;
    }
    printf("RESULT test-mmid-gateup-fuse pass=%d fail=%d\n", pass, fail);
    return fail ? 1 : 0;
}
