// CH-19: Q8_0-only opt-in fused MoE up+gate+SwiGLU; C5 Q4_K/Q5_K fusion is unchanged.
// The switch is argv: `--fused on|off` (default off) sets cplan.fused_moe_upgate_q8 only for Q8_0, exactly what
// llama_context does via ggml_backend_cpu_set_fused_moe_upgate. Q4_K uses the existing C5 fusion path in both arms.
//   test-moe-fused-upgate [--fused off] write <ref.bin>   -- write same-image control outputs (Q4_K C5 fusion remains active)
//   test-moe-fused-upgate --fused on    cmp   <ref.bin>   -- run the graph, NMSE vs ref file
//   test-moe-fused-upgate cmp   <ref.bin> <cell 0..3> -- run ONE cell only (one process per cell, so the
//                                                       once-per-process ACTIVE marker proves THAT cell fused)
// The graph is MUL_MAT_ID(up) -> MUL_MAT_ID(gate) -> GLU(swiglu split) on a single token, i.e. the
// node pattern llm_graph_context::build_moe_ffn emits. The qualifying job must require GGML_IQK=1 and GGML_IQK_Q8_0=1 for Q8_0 on, and grep stderr for the
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

static std::vector<float> run_case(ggml_type type, int n_threads, int64_t K = 512, bool cached_toggle = false) {
    const int64_t N = 256, E = 8, U = 4;
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
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "CPU backend init failed\n"); exit(2); }
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    auto output = [&]() { return std::vector<float>((float *) out->data, (float *) out->data + ggml_nelements(out)); };
    std::vector<float> r;
    if (cached_toggle) {
        const ggml_cplan raw = ggml_graph_plan(gf, n_threads, nullptr);
        const size_t required = (size_t) raw.n_threads * (size_t) (K / 32) * 36;
        fprintf(stderr, "workspace-gap: raw=%zu required=%zu nth=%d\n", raw.work_size, required, raw.n_threads);
        if (raw.work_size >= required) { fprintf(stderr, "workspace fixture is not discriminating\n"); exit(2); }
        ggml_backend_cpu_set_fused_moe_upgate(backend, false);
        ggml_backend_graph_plan_t plan = ggml_backend_graph_plan_create(backend, gf);
        if (!plan || ggml_backend_graph_plan_compute(backend, plan) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cached OFF compute failed\n"); exit(2);
        }
        const auto off = output();
        ggml_backend_cpu_set_fused_moe_upgate(backend, true);
        if (ggml_backend_graph_plan_compute(backend, plan) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cached ON compute failed\n"); exit(2);
        }
        const auto on = output();
        double num = 0, den = 0;
        for (size_t i = 0; i < on.size(); ++i) {
            const double d = (double) on[i] - off[i];
            num += d*d; den += (double) off[i]*off[i];
        }
        const double nmse = den > 0 ? num/den : num;
        if (!std::isfinite(nmse) || nmse > NMSE_MAX) {
            fprintf(stderr, "cached ON mismatch: nmse=%.3e\n", nmse); exit(1);
        }
        ggml_backend_cpu_set_fused_moe_upgate(backend, false);
        if (ggml_backend_graph_plan_compute(backend, plan) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cached OFF re-compute failed\n"); exit(2);
        }
        r = output();
        if (r.size() != off.size() || memcmp(r.data(), off.data(), r.size()*sizeof(float)) != 0) {
            fprintf(stderr, "cached OFF output changed after ON toggle\n"); exit(1);
        }
        fprintf(stderr, "cached OFF->ON->OFF nmse=%.3e off_equal=1\n", nmse);
        ggml_backend_graph_plan_free(backend, plan);
    } else {
        ggml_backend_cpu_set_fused_moe_upgate(backend, g_fused);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "CPU backend compute failed\n"); exit(2);
        }
        r = output();
    }
    ggml_backend_free(backend);
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
    fprintf(stderr, "fused_moe_upgate_q8=%d (Q4_K legacy unchanged)\n", g_fused ? 1 : 0);
    if (argc == 2 && std::string(argv[1]) == "--workspace-gap") {
        (void) run_case(GGML_TYPE_Q8_0, 8, 8192, true);
        printf("RESULT test-moe-fused-upgate workspace-gap pass=1 fail=0\n");
        return 0;
    }
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
