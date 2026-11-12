// C11-CPU-BARRIER-COST: correctness test for dependency-aware barrier elision
// (GGML_BARRIER_DEP=1, ggml-cpu.c "pure epochs").
//
// Builds a graph of several independent lanes of pure ops whose nodes are INTERLEAVED in graph
// order (so adjacent nodes are frequently independent and the elision fires), mixed with the
// hazards it must not elide across: read-after-write chains, in-place ops (write-after-read on
// the same bytes), two lanes writing disjoint and overlapping views of one shared buffer, a
// cross-lane reduction, and a MUL_MAT (impure: always fenced).  Shapes span the solo path
// (tiny rows) and the multi-threaded path (5 x 2560, the MTP-d4 trunk row count).
//
// The multi-threaded result must be BIT-IDENTICAL to the single-threaded one (at nth=1 there are
// no barriers to elide), on every one of many iterations, for several thread counts.
// Exit 0 on pass.  The test forces GGML_BARRIER_DEP=1 unless the caller set it.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

static const int N_LANES = 6;
static const int N_STEPS = 24;

struct graph_t {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> outputs;
};

static graph_t build(int64_t ne0, int64_t ne1) {
    graph_t g;
    ggml_init_params ip = { /*.mem_size =*/ (size_t) 512 * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    g.ctx = ggml_init(ip);
    g.gf  = ggml_new_graph_custom(g.ctx, 16384, false);

    ggml_tensor * lane[N_LANES];
    for (int l = 0; l < N_LANES; l++) {
        lane[l] = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, ne0, ne1);
        g.inputs.push_back(lane[l]);
    }
    ggml_tensor * w = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, ne0);
    g.inputs.push_back(w);
    // shared buffer that lanes write through views: lanes 0/1 disjoint halves, lanes 2/3 the same half
    ggml_tensor * shared = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 2 * ne0);
    g.inputs.push_back(shared);
    ggml_tensor * mm_w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, ne0, ne0 / 8);
    g.inputs.push_back(mm_w);

    for (int s = 0; s < N_STEPS; s++) {
        for (int l = 0; l < N_LANES; l++) {
            ggml_tensor * x = lane[l];
            switch ((s + l) % 8) {
                case 0: x = ggml_add(g.ctx, x, w); break;
                case 1: x = ggml_mul(g.ctx, x, w); break;
                case 2: x = ggml_scale(g.ctx, x, 0.75f + 0.01f * l); break;
                case 3: x = ggml_rms_norm(g.ctx, x, 1e-6f); break;           // may fuse with the MUL below
                case 4: x = ggml_mul_inplace(g.ctx, x, w); break;            // in place: WAR/WAW on x's bytes
                case 5: x = ggml_silu(g.ctx, x); break;
                case 6: x = ggml_add_inplace(g.ctx, x, lane[(l + 1) % N_LANES]); break; // cross-lane read
                case 7: x = ggml_cont(g.ctx, ggml_transpose(g.ctx, ggml_cont(g.ctx, ggml_transpose(g.ctx, x)))); break;
            }
            lane[l] = x;
            ggml_build_forward_expand(g.gf, x);
        }
        if (s % 6 == 5) {
            // views of the shared buffer: 0 and 1 disjoint, 2 and 3 overlapping (WAW across lanes)
            const size_t half = (size_t) ne0 * sizeof(float);
            for (int l = 0; l < 4; l++) {
                const size_t off = (l == 1) ? half : 0;
                ggml_tensor * row0 = ggml_view_1d(g.ctx, lane[l], ne0, 0);
                ggml_tensor * dst  = ggml_view_1d(g.ctx, shared, ne0, l < 2 ? off : half);
                ggml_tensor * c    = ggml_cpy(g.ctx, row0, dst);
                ggml_build_forward_expand(g.gf, c);
                // read the shared buffer right after writing it (RAW through a different view)
                ggml_tensor * rd = ggml_view_2d(g.ctx, shared, ne0, 1, half, l < 2 ? off : half);
                lane[l] = ggml_add(g.ctx, lane[l], rd);
                ggml_build_forward_expand(g.gf, lane[l]);
            }
            // an impure node in the middle: mat-vec over lane 4
            ggml_tensor * mm = ggml_mul_mat(g.ctx, mm_w, lane[4]);
            ggml_build_forward_expand(g.gf, mm);
            g.outputs.push_back(mm);
        }
    }
    ggml_tensor * acc = lane[0];
    for (int l = 1; l < N_LANES; l++) {
        acc = ggml_add(g.ctx, acc, lane[l]);
    }
    ggml_build_forward_expand(g.gf, acc);
    for (int l = 0; l < N_LANES; l++) {
        g.outputs.push_back(lane[l]);
    }
    g.outputs.push_back(acc);
    g.outputs.push_back(shared);
    return g;
}

static void fill_inputs(graph_t & g, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (ggml_tensor * t : g.inputs) {
        float * p = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            p[i] = d(rng);
        }
    }
}

static std::vector<uint8_t> snapshot(const graph_t & g) {
    std::vector<uint8_t> out;
    for (ggml_tensor * t : g.outputs) {
        const uint8_t * p = (const uint8_t *) t->data;
        out.insert(out.end(), p, p + ggml_nbytes(t));
    }
    return out;
}

static bool run_case(int64_t ne0, int64_t ne1, int n_threads, int iters) {
    graph_t g = build(ne0, ne1);
    std::vector<std::vector<uint8_t>> ref(iters);
    for (int it = 0; it < iters; it++) {
        fill_inputs(g, 1234u + it);
        ggml_graph_compute_with_ctx(g.ctx, g.gf, 1);
        ref[it] = snapshot(g);
    }
    bool ok = true;
    for (int it = 0; it < iters && ok; it++) {
        fill_inputs(g, 1234u + it);
        ggml_graph_compute_with_ctx(g.ctx, g.gf, n_threads);
        if (snapshot(g) != ref[it]) {
            fprintf(stderr, "FAIL ne=%lldx%lld nth=%d iter=%d: multi-thread result differs from 1-thread\n",
                    (long long) ne0, (long long) ne1, n_threads, it);
            ok = false;
        }
    }
    printf("%s ne=%lldx%lld nth=%d iters=%d nodes=%d\n", ok ? "ok  " : "FAIL",
           (long long) ne0, (long long) ne1, n_threads, iters, ggml_graph_n_nodes(g.gf));
    ggml_free(g.ctx);
    return ok;
}

int main(void) {
    if (getenv("GGML_BARRIER_DEP") == nullptr) {
        setenv("GGML_BARRIER_DEP", "1", 1);
    }
    printf("GGML_BARRIER_DEP=%s\n", getenv("GGML_BARRIER_DEP"));

    bool ok = true;
    const int64_t shapes[][2] = { {64, 1}, {256, 5}, {2560, 1}, {2560, 5}, {4096, 8} };
    const int threads[] = { 2, 8, 24 };
    for (const auto & sh : shapes) {
        for (int nt : threads) {
            ok = run_case(sh[0], sh[1], nt, 20) && ok;
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
