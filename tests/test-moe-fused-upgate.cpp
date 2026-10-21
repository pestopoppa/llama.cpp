// CH-19: fused MoE up+gate+SwiGLU (cplan.fused_moe_upgate, needs GGML_IQK=1).
// The switch is argv: `--fused on|off` (default off) sets cplan.fused_moe_upgate, exactly what
// llama_context does via ggml_backend_cpu_set_fused_moe_upgate. GGML_CPU_FUSED_UPGATE=0 is only a kill switch.
//   test-moe-fused-upgate [--fused off] write <ref.bin>   -- run the graph, write outputs (unfused reference)
//   test-moe-fused-upgate --fused on    cmp   <ref.bin>   -- run the graph, NMSE vs ref file
//   test-moe-fused-upgate cmp   <ref.bin> <cell 0..3> -- run ONE cell only (one process per cell, so the
//                                                       once-per-process ACTIVE marker proves THAT cell fused)
// The graph is MUL_MAT_ID(up) -> MUL_MAT_ID(gate) -> GLU(swiglu split) on a single token, i.e. the
// node pattern llm_graph_context::build_moe_ffn emits. The queue job greps stderr for the
// "[iqk] ACTIVE: fused MoE up+gate" line to prove the fused path actually ran.
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

static const double NMSE_MAX = 1e-5; // predeclared: iqk's activation is not ggml_silu bit-for-bit
static bool g_fused = false;          // --fused on|off

static std::vector<float> run_case(ggml_type type, int n_threads) {
    const int64_t K = 512, N = 256, E = 8, U = 4;
    ggml_init_params ip = { 256u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(1234 + (int) type);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    auto mk_w = [&]() {
        ggml_tensor * w = ggml_new_tensor_3d(ctx, type, K, N, E);
        std::vector<float> f(K * N * E);
        for (auto & x : f) { x = 0.1f * nd(rng); }
        ggml_quantize_chunk(type, f.data(), w->data, 0, N * E, K, nullptr);
        return w;
    };
    ggml_tensor * wu = mk_w();
    ggml_tensor * wg = mk_w();
    ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, 1);
    for (int64_t i = 0; i < K; ++i) { ((float *) x->data)[i] = nd(rng); }
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, 1);
    const int32_t sel[U] = { 5, 1, 6, 2 };
    memcpy(ids->data, sel, sizeof(sel));

    ggml_tensor * up   = ggml_mul_mat_id(ctx, wu, x, ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, wg, x, ids);
    ggml_tensor * out  = ggml_swiglu_split(ctx, gate, up);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    // CH-19.1: record the node order the fusion predicate sees (expected: gate, up, glu -- the DFS
    // visits glu->src[0] = gate first). Goes to stderr next to the ACTIVE marker.
    fprintf(stderr, "graph(type=%d nt=%d):", (int) type, n_threads);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        fprintf(stderr, " %s%s", ggml_op_desc(n), n == up ? "(up)" : n == gate ? "(gate)" : "");
    }
    fprintf(stderr, "\n");
    ggml_cplan cplan = ggml_graph_plan(gf, n_threads, nullptr);
    std::vector<uint8_t> work(cplan.work_size + 1);
    cplan.work_data        = work.data();
    cplan.fused_moe_upgate = g_fused;
    if (ggml_graph_compute(gf, &cplan) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(2); }

    std::vector<float> r((float *) out->data, (float *) out->data + ggml_nelements(out));
    ggml_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--fused") {
        const std::string v = argv[2];
        if (v == "on" || v == "1") { g_fused = true; } else if (v == "off" || v == "0") { g_fused = false; }
        else { fprintf(stderr, "--fused on|off\n"); return 2; }
        argv[2] = argv[0];
        argv += 2; argc -= 2;
    }
    fprintf(stderr, "fused_moe_upgate=%d\n", g_fused ? 1 : 0);
    if (argc == 1) { // ctest smoke: just exercise the graph
        for (ggml_type t : { GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 }) { (void) run_case(t, 2); }
        printf("RESULT test-moe-fused-upgate smoke pass=1 fail=0\n");
        return 0;
    }
    if (argc != 3 && argc != 4) { fprintf(stderr, "usage: %s [write <file> | cmp <file> [cell]]\n", argv[0]); return 2; }
    const std::string mode = argv[1];
    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 };
    const int       nts[]   = { 1, 8 };
    const char * names[4] = { "Q4_K nt=1", "Q4_K nt=8", "Q8_0 nt=1", "Q8_0 nt=8" };
    int pass = 0, fail = 0;

    const int only = argc == 4 ? atoi(argv[3]) : -1;
    if (argc == 4 && (mode != "cmp" || only < 0 || only > 3)) { fprintf(stderr, "cell selector: cmp only, 0..3\n"); return 2; }

    std::vector<float> all;
    size_t per = 0;
    for (int c = 0; c < 4; ++c) {
        if (only >= 0 && c != only) { continue; }
        auto r = run_case(types[c / 2], nts[c % 2]);
        per = r.size();
        all.insert(all.end(), r.begin(), r.end());
    }
    if (mode == "write") {
        FILE * f = fopen(argv[2], "wb");
        if (!f) { perror("open"); return 2; }
        fwrite(all.data(), sizeof(float), all.size(), f);
        fclose(f);
        printf("wrote %zu floats\n", all.size());
        printf("RESULT test-moe-fused-upgate pass=1 fail=0\n");
        return 0;
    }
    FILE * f = fopen(argv[2], "rb");
    if (!f) { perror("open"); return 2; }
    std::vector<float> ref(4 * per);
    if (fread(ref.data(), sizeof(float), ref.size(), f) != ref.size()) { fprintf(stderr, "short ref file\n"); return 2; }
    fclose(f);
    for (int c = 0; c < 4; ++c) {
        if (only >= 0 && c != only) { continue; }
        const size_t off = only >= 0 ? 0 : c * per; // offset into `all`
        double num = 0, den = 0, maxd = 0;
        for (size_t k = 0; k < per; ++k) {
            const size_t i = c * per + k;           // offset into `ref`
            const double d = (double) all[off + k] - ref[i];
            num += d * d; den += (double) ref[i] * ref[i];
            maxd = std::fmax(maxd, std::fabs(d));
        }
        const double nmse = den > 0 ? num / den : num;
        const bool ok = std::isfinite(nmse) && nmse <= NMSE_MAX;
        printf("%-10s nmse=%.3e maxabs=%.3e %s\n", names[c], nmse, maxd, ok ? "ok" : "FAIL");
        ok ? ++pass : ++fail;
    }
    printf("RESULT test-moe-fused-upgate pass=%d fail=%d\n", pass, fail);
    return fail ? 1 : 0;
}
