// AK-BW-GDN-GATE-FOLD (autokernel-loop.md): beta/alpha projections + GDN gate math in one launch per layer.
//
// Grid: one workgroup per output row r in [0, 2H): rows [0, H) are ssm_beta, [H, 2H) ssm_alpha (the two Q8_0 weights
// are read in place, no load-time concat). 256 threads = 8 groups of 32 lanes; group j walks Q8_0 blocks j, j+8, ...,
// lane i owns element i of the block, and every lane keeps one f32 accumulator per token (ntok <= 16, unrolled). The
// activations are read as f32 (no int8 activation quantization, so this is closer to the f32 product than the stock
// q8sk/MMVQ path). Reduction: 32-lane shuffle tree, then the 8 group partials summed in a fixed order by one thread
// per token: deterministic, no atomics, no zero-init. Epilogue (unfused op semantics of ggml-cuda unary.cu):
//   beta[t, h] = 1 / (1 + exp(-b))
//   g[t, h]    = softplus(a + dt[h]) * ssm_a[h],  softplus(z) = z > 20 ? z : log(1 + exp(z))
// written to the SIGMOID and MUL nodes' own buffers, [H, ntok] with h fastest (the GDN [1, H, T, S] layout).

#include "gdn-gate-fold.cuh"

#include <atomic>
#include <cstdlib>

static std::atomic<uint64_t> g_gdn_gate_fold_n_calls{0};
static std::atomic<uint64_t> g_gdn_gate_fold_n_unsafe{0};

uint64_t ggml_cuda_gdn_gate_fold_n_calls() {
    return g_gdn_gate_fold_n_calls.load();
}

bool ggml_cuda_gdn_gate_fold_enabled() {
    // read per graph evaluation so a test can flip it between two computes in one process (with CUDA graphs off)
    const char * e = getenv("GGML_HIP_GDN_GATE_FOLD");
    return e != nullptr && e[0] == '1';
}

#define GDN_GF_THREADS 256
#define GDN_GF_GROUPS  (GDN_GF_THREADS / 32)

template <int MAXT>
static __global__ void __launch_bounds__(GDN_GF_THREADS)
gdn_gate_fold_kernel(const char * __restrict__ wb, const char * __restrict__ wa, const int64_t nb1b, const int64_t nb1a,
                     const int nblk, const float * __restrict__ x, const int64_t sx1, const int ntok, const int H,
                     const float * __restrict__ dt, const float * __restrict__ aw,
                     float * __restrict__ beta, float * __restrict__ g) {
    const int  r    = blockIdx.x;
    const bool is_a = r >= H;
    const int  h    = is_a ? r - H : r;
    const block_q8_0 * w = (const block_q8_0 *) (is_a ? wa + (int64_t) h * nb1a : wb + (int64_t) h * nb1b);

    const int lane = threadIdx.x % 32;
    const int grp  = threadIdx.x / 32;

    float acc[MAXT];
#pragma unroll
    for (int m = 0; m < MAXT; ++m) {
        acc[m] = 0.0f;
    }
    for (int b = grp; b < nblk; b += GDN_GF_GROUPS) {
        const float   wv = __half2float(w[b].d) * (float) w[b].qs[lane];
        const float * xp = x + (int64_t) b * 32 + lane;
#pragma unroll
        for (int m = 0; m < MAXT; ++m) {
            if (m < ntok) {
                acc[m] += wv * xp[(int64_t) m * sx1];
            }
        }
    }

    __shared__ float red[MAXT][GDN_GF_GROUPS];
#pragma unroll
    for (int m = 0; m < MAXT; ++m) {
        if (m < ntok) {
            const float v = warp_reduce_sum<32>(acc[m]);
            if (lane == 0) {
                red[m][grp] = v;
            }
        }
    }
    __syncthreads();

    const int m = threadIdx.x;
    if (m < ntok) {
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < GDN_GF_GROUPS; ++j) {
            s += red[m][j];
        }
        if (!is_a) {
            beta[(int64_t) m * H + h] = 1.0f / (1.0f + expf(-s));
        } else {
            const float z  = s + dt[h];
            const float sp = (z > 20.0f) ? z : logf(1.0f + expf(z));
            g[(int64_t) m * H + h] = sp * aw[h];
        }
    }
}

void ggml_cuda_gdn_gate_fold_plan(const ggml_cgraph * cgraph, ggml_cuda_gdn_gate_plan & plan) {
    plan.skip.clear();
    plan.anchors.clear();
    plan.chains.clear();
    plan.n_matched = 0;
    plan.n_unsafe  = 0;
    if (!ggml_cuda_gdn_gate_fold_enabled()) {
        return;
    }
    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return;
    }
    ggml_gdn_gate_plan_build(cgraph, plan);
    if (plan.n_unsafe > 0 && g_gdn_gate_fold_n_unsafe.fetch_add(plan.n_unsafe) == 0) {
        GGML_LOG_WARN("GDN_GATE_FOLD: %d of %d matched gate chains run unfused (an input's memory was reused before the "
                      "anchor)\n", plan.n_unsafe, plan.n_matched);
    }
}

void ggml_cuda_gdn_gate_fold_run(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_gate_anchor & an) {
    cudaStream_t stream = ctx.stream();
    const int H    = (int) an.H;
    const int ntok = (int) an.ntok;
    const int nblk = (int) (an.x->ne[0] / 32);
    GGML_ASSERT(ntok >= 1 && ntok <= GDN_GATE_FOLD_MAXT);
    gdn_gate_fold_kernel<GDN_GATE_FOLD_MAXT><<<2 * H, GDN_GF_THREADS, 0, stream>>>(
        (const char *) an.wb->data, (const char *) an.wa->data, (int64_t) an.wb->nb[1], (int64_t) an.wa->nb[1], nblk,
        (const float *) an.x->data, (int64_t) (an.x->nb[1] / sizeof(float)), ntok, H,
        (const float *) an.dt->data, (const float *) an.a->data, (float *) an.beta->data, (float *) an.g->data);
    CUDA_CHECK(cudaGetLastError());
    // route witness (row packets grep it; also the build's strings content marker)
    if (g_gdn_gate_fold_n_calls.fetch_add(1) == 0) {
        GGML_LOG_WARN("GDN_GATE_FOLD: ggml_cuda_gdn_gate_fold beta/alpha projection + gate folded into one launch "
                      "(call 1, H %d, ntok %d, K %d, GGML_HIP_GDN_GATE_FOLD=1)\n", H, ntok, (int) an.x->ne[0]);
    }
}
