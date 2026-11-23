// AK-Q38FN-CPU-GDNPROJ-FUSE: fused single-row sibling MUL_MAT pass vs the stock per-node path.
//
// GGML_Q38FN_GDNPROJ_FUSE=1 (default off) lets the CPU backend compute a run of ADJACENT
// MUL_MAT nodes that share one single-row F32 src1 as one pass (one activation quantization,
// one barrier, rows byte-balanced over the team). The claim is bit-exactness: every output
// element is the same vec_dot over the same quantized activation as the unfused node.
//
// Same process, same knob: graph A places the siblings back to back (fuses), graph B
// interleaves an unrelated op between them (cannot fuse -> stock path). Outputs must be
// bit-identical, A must engage the fused pass (witness counter) and B must not.
// Cases: the Qwen3.8-Flash-Next GDN in-projection set (K=2560; Q8_0 10240 + Q8_0 6144 + F32 48
// + F32 48) and an odd-shaped mix, at 1, 3, 7 and 24 threads.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct sib { ggml_type type; int64_t rows; };

static ggml_backend_cpu_mm_sibling_fuse_n_calls_t g_calls = nullptr;

static std::vector<std::vector<float>> run(const std::vector<sib> & sibs, int64_t K, int nth, bool adjacent, uint64_t * fused) {
    size_t mem = 64u * 1024 * 1024;
    for (const auto & s : sibs) { mem += (size_t) s.rows * K * 4 + (size_t) s.rows * 16; }
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(4321 + (unsigned) K);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    for (int64_t i = 0; i < K; ++i) { ((float *) x->data)[i] = nd(rng); }
    ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    for (int64_t i = 0; i < 64; ++i) { ((float *) y->data)[i] = nd(rng); }

    std::vector<ggml_tensor *> outs;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_scale(ctx, x, 1.0f)); // x stays an input, not a node
    for (size_t i = 0; i < sibs.size(); ++i) {
        const auto & s = sibs[i];
        ggml_tensor * w = ggml_new_tensor_2d(ctx, s.type, K, s.rows);
        std::vector<float> f((size_t) K * s.rows);
        for (auto & v : f) { v = 0.05f * nd(rng); }
        if (s.type == GGML_TYPE_F32) {
            memcpy(w->data, f.data(), f.size() * sizeof(float));
        } else {
            ggml_quantize_chunk(s.type, f.data(), w->data, 0, s.rows, K, nullptr);
        }
        ggml_tensor * o = ggml_mul_mat(ctx, w, x);
        outs.push_back(o);
        if (!adjacent && i > 0) {
            ggml_build_forward_expand(gf, ggml_scale(ctx, y, 0.5f + (float) i));
        }
        ggml_build_forward_expand(gf, o);
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { fprintf(stderr, "CPU backend init failed\n"); exit(2); }
    ggml_backend_cpu_set_n_threads(be, nth);
    const uint64_t c0 = g_calls();
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(2); }
    *fused = g_calls() - c0;
    ggml_backend_free(be);

    std::vector<std::vector<float>> r;
    for (auto * o : outs) { r.emplace_back((float *) o->data, (float *) o->data + ggml_nelements(o)); }
    ggml_free(ctx);
    return r;
}

int main() {
    setenv("GGML_Q38FN_GDNPROJ_FUSE", "1", 1);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    g_calls = (ggml_backend_cpu_mm_sibling_fuse_n_calls_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_mm_sibling_fuse_n_calls");
    if (!g_calls) { fprintf(stderr, "FAIL: witness ggml_backend_cpu_mm_sibling_fuse_n_calls missing\n"); return 1; }

    struct tcase { const char * name; int64_t K; std::vector<sib> sibs; };
    const std::vector<tcase> cases = {
        { "q38fn-gdn-inproj", 2560, { { GGML_TYPE_Q8_0, 10240 }, { GGML_TYPE_Q8_0, 6144 }, { GGML_TYPE_F32, 48 }, { GGML_TYPE_F32, 48 } } },
        { "odd-mix",           256, { { GGML_TYPE_Q8_0, 37 }, { GGML_TYPE_F32, 5 }, { GGML_TYPE_Q8_0, 1000 } } },
        { "f32-pair",          512, { { GGML_TYPE_F32, 3 }, { GGML_TYPE_F32, 200 } } },
    };
    int fails = 0;
    for (const auto & c : cases) {
        for (int nth : { 1, 3, 7, 24 }) {
            uint64_t fa = 0, fb = 0;
            const auto a = run(c.sibs, c.K, nth, true,  &fa);
            const auto b = run(c.sibs, c.K, nth, false, &fb);
            size_t diff = 0, n = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                n += a[i].size();
                for (size_t j = 0; j < a[i].size(); ++j) {
                    if (memcmp(&a[i][j], &b[i][j], sizeof(float)) != 0) { diff++; }
                }
            }
            const bool ok = diff == 0 && fa == 1 && fb == 0;
            printf("%-18s nth=%-2d elems=%zu bitdiff=%zu fused_groups(A)=%llu fused_groups(B)=%llu %s\n",
                   c.name, nth, n, diff, (unsigned long long) fa, (unsigned long long) fb, ok ? "OK" : "FAIL");
            fails += ok ? 0 : 1;
        }
    }
    printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
