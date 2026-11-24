#include "router-fuse.cuh"

#include <atomic>
#include <climits>
#include <cmath>
#include <cstdlib>

// AK-RF-A3B-ROUTER-FUSE (see router-fuse.cuh). The top-k phase is the topk-moe kernel's softmax gating path
// (topk-moe.cu: softmax, NaN -> -FLT_MAX, k rounds of (value desc, expert asc) argmax with a per-thread taken mask,
// optional clamp-renorm, scale) on logits read from the scratch instead of from the MUL_MAT output tensor.

#define ROUTER_FUSE_WARPS 2  // experts per block (one 32-lane warp each); 256 experts -> 128 blocks on 104 CUs

static std::atomic<uint64_t> g_router_fuse_n_calls{0};

uint64_t ggml_cuda_router_fuse_n_calls() {
    return g_router_fuse_n_calls.load(std::memory_order_relaxed);
}

bool ggml_cuda_router_fuse_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("GGML_HIP_ROUTER_FUSE");
        return s != nullptr && atoi(s) != 0;
    }();
    return enabled;
}

bool ggml_cuda_router_fuse_rows_ok(const int64_t n_rows) {
    static const int min_n = [] {
        const char * s = getenv("GGML_HIP_ROUTER_FUSE_MIN_N");
        const int    v = s != nullptr ? atoi(s) : 1;
        return v < 1 ? 1 : v;
    }();
    return n_rows >= min_n && n_rows <= ROUTER_FUSE_MAX_ROWS;
}

template <int n_experts, int NR>
__launch_bounds__(ROUTER_FUSE_WARPS * WARP_SIZE, 1) __global__
    void router_fuse_kernel(const float * __restrict__ W,        // [K, n_experts], row stride w_stride
                            const float * __restrict__ x,        // [K, n_rows], row stride x_stride
                            float *                    scratch,  // [n_experts, n_rows]
                            unsigned int *             counter,
                            float *                    weights,  // [n_expert_used, n_rows] contiguous
                            int32_t *                  ids,      // [n_expert_used, n_rows], row stride ids_stride
                            const int                  K,
                            const int64_t              w_stride,
                            const int64_t              x_stride,
                            const int64_t              ids_stride,
                            const int                  n_rows,
                            const int                  n_expert_used,
                            const bool                 with_norm,
                            const float                clamp_val,
                            const float                scale_val) {
    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int expert = blockIdx.x * ROUTER_FUSE_WARPS + warp;

    // phase 1: router GEMV, one warp per expert, all rows at once (W is read once; x stays L1/L2 resident)
    if (expert < n_experts) {
        float acc[NR];
#pragma unroll
        for (int t = 0; t < NR; ++t) {
            acc[t] = 0.0f;
        }
        const float * w = W + (int64_t) expert * w_stride;
        for (int k = lane * 4; k < K; k += WARP_SIZE * 4) {
            const float4 wv = *(const float4 *) (w + k);
#pragma unroll
            for (int t = 0; t < NR; ++t) {
                if (t < n_rows) {
                    const float4 xv = *(const float4 *) (x + (int64_t) t * x_stride + k);
                    acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NR; ++t) {
            if (t < n_rows) {
                const float s = warp_reduce_sum(acc[t]);
                if (lane == 0) {
                    scratch[t * n_experts + expert] = s;
                }
            }
        }
    }

    // last-block hand-off: publish this block's logits, count it, and let the final block run the top-k
    __shared__ bool s_last;
    __threadfence();
    __syncthreads();
    if (lane == 0 && warp == 0) {
        const unsigned int prev = atomicAdd(counter, 1u);
        s_last                  = prev == gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    // phase 2: softmax -> top-k -> (renorm) -> scale, one warp per row
    constexpr int experts_per_thread = n_experts / WARP_SIZE;
    static_assert(n_experts % WARP_SIZE == 0, "n_experts must be a multiple of WARP_SIZE");
    static_assert(experts_per_thread <= 32, "taken mask is 32 bits");

    for (int row = warp; row < n_rows; row += ROUTER_FUSE_WARPS) {
        // volatile: the other blocks' logits were written by other CUs; bypass this CU's L1
        const volatile float * lg = scratch + row * n_experts;

        float wt[experts_per_thread];
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            wt[i] = lg[lane + i * WARP_SIZE];
        }

        // softmax (topk-moe softmax_warp_inplace<.., false>)
        float max_val = -INFINITY;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            max_val = max(max_val, wt[i]);
        }
        max_val   = warp_reduce_max(max_val);
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            const float val = expf(wt[i] - max_val);
            wt[i]           = val;
            sum += val;
        }
        sum                 = warp_reduce_sum(sum);
        const float inv_sum = 1.0f / sum;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            wt[i] *= inv_sum;
        }

#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            if (__isnanf(wt[i])) {
                wt[i] = -FLT_MAX;
            }
        }

        float output_weights[experts_per_thread];
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] = 0.f;
        }

        float    wt_sum = 0.f;
        uint32_t taken  = 0;
        for (int k = 0; k < n_expert_used; k++) {
            float sel_val    = -INFINITY;
            int   sel_expert = INT_MAX;
#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                const int e = lane + i * WARP_SIZE;
                if (!((taken >> i) & 1u) && (sel_expert == INT_MAX || wt[i] > sel_val)) {
                    sel_val    = wt[i];
                    sel_expert = e;
                }
            }
#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, sel_val, mask, WARP_SIZE);
                const int   e   = __shfl_xor_sync(0xFFFFFFFF, sel_expert, mask, WARP_SIZE);
                if (val > sel_val || (val == sel_val && e < sel_expert)) {
                    sel_val    = val;
                    sel_expert = e;
                }
            }
            if ((sel_expert & (WARP_SIZE - 1)) == lane) {
                taken |= 1u << (sel_expert / WARP_SIZE);
                ids[row * ids_stride + k] = sel_expert;
                if (with_norm) {
                    wt_sum += sel_val;
                }
            }
            if ((k & (WARP_SIZE - 1)) == lane) {
                output_weights[k / WARP_SIZE] = sel_val;
            }
        }

        if (with_norm) {
            wt_sum              = warp_reduce_sum(wt_sum);
            wt_sum              = max(wt_sum, clamp_val);
            const float inv_ws  = 1.0f / wt_sum;
#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                output_weights[i] *= inv_ws;
            }
        }

#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            const int idx = i * WARP_SIZE + lane;
            if (idx < n_expert_used) {
                weights[row * n_expert_used + idx] = output_weights[i] * scale_val;
            }
        }
    }

    // self-reset for the next launch (stream ordered after this kernel)
    if (lane == 0 && warp == 0) {
        atomicExch(counter, 0u);
    }
}

template <int n_experts>
static void router_fuse_launch_rows(cudaStream_t stream, dim3 grid, dim3 block, const float * W, const float * x,
                                    float * scratch, unsigned int * counter, float * weights, int32_t * ids, int K,
                                    int64_t w_stride, int64_t x_stride, int64_t ids_stride, int n_rows,
                                    int n_expert_used, bool with_norm, float clamp_val, float scale_val) {
#define ROUTER_FUSE_LAUNCH(NR)                                                                                    \
    router_fuse_kernel<n_experts, NR><<<grid, block, 0, stream>>>(W, x, scratch, counter, weights, ids, K, w_stride, \
                                                                  x_stride, ids_stride, n_rows, n_expert_used,      \
                                                                  with_norm, clamp_val, scale_val)
    if (n_rows <= 1) {
        ROUTER_FUSE_LAUNCH(1);
    } else if (n_rows <= 2) {
        ROUTER_FUSE_LAUNCH(2);
    } else if (n_rows <= 4) {
        ROUTER_FUSE_LAUNCH(4);
    } else {
        ROUTER_FUSE_LAUNCH(8);
    }
#undef ROUTER_FUSE_LAUNCH
}

bool ggml_cuda_op_router_fuse(ggml_backend_cuda_context & ctx,
                              const ggml_tensor *         mul_mat,
                              ggml_tensor *               weights,
                              ggml_tensor *               ids,
                              const ggml_tensor *         clamp,
                              const ggml_tensor *         scale) {
    const ggml_tensor * W = mul_mat->src[0];
    const ggml_tensor * x = mul_mat->src[1];

    if (W->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || mul_mat->type != GGML_TYPE_F32 ||
        weights->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (W->ne[2] != 1 || W->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    if (W->nb[0] != sizeof(float) || x->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t) ||
        !ggml_is_contiguous(weights)) {
        return false;
    }

    const int64_t K             = W->ne[0];
    const int64_t n_experts     = W->ne[1];
    const int64_t n_rows        = x->ne[1];
    const int64_t n_expert_used = weights->ne[1];

    if (x->ne[0] != K || K % 4 != 0 || K > INT_MAX || !ggml_cuda_router_fuse_rows_ok(n_rows)) {
        return false;
    }
    if (n_experts != 64 && n_experts != 128 && n_experts != 256 && n_experts != 512) {
        return false;
    }
    if (n_expert_used < 1 || n_expert_used > n_experts || ggml_nelements(weights) != n_expert_used * n_rows ||
        ids->ne[0] != n_expert_used || ids->ne[1] != n_rows) {
        return false;
    }
    // float4 loads
    if (((uintptr_t) W->data % 16) != 0 || ((uintptr_t) x->data % 16) != 0 || (W->nb[1] % 16) != 0 ||
        (x->nb[1] % 16) != 0) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    if (ctx.router_fuse_counters == nullptr) {
        // first use: allocate the per-stream completion counters outside stream capture (graph warmup runs the
        // first evaluations eagerly; a captured first call declines and the unfused nodes are captured instead)
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
        if (cap != cudaStreamCaptureStatusNone) {
            return false;
        }
        void * p = nullptr;
        if (cudaMalloc(&p, GGML_CUDA_MAX_STREAMS * sizeof(unsigned int)) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        CUDA_CHECK(cudaMemsetAsync(p, 0, GGML_CUDA_MAX_STREAMS * sizeof(unsigned int), stream));
        ctx.router_fuse_counters = (unsigned int *) p;
    }
    unsigned int * counter = ctx.router_fuse_counters + ctx.curr_stream_no;

    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_rows * n_experts);

    const float clamp_val  = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const float scale_val  = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;
    const bool  with_norm  = clamp != nullptr;
    const dim3  block(WARP_SIZE, ROUTER_FUSE_WARPS, 1);
    const dim3  grid((unsigned) ((n_experts + ROUTER_FUSE_WARPS - 1) / ROUTER_FUSE_WARPS), 1, 1);

    const float * W_d        = (const float *) W->data;
    const float * x_d        = (const float *) x->data;
    const int64_t w_stride   = W->nb[1] / sizeof(float);
    const int64_t x_stride   = x->nb[1] / sizeof(float);
    const int64_t ids_stride = ids->nb[1] / sizeof(int32_t);
    float *       weights_d  = (float *) weights->data;
    int32_t *     ids_d      = (int32_t *) ids->data;

    switch (n_experts) {
        case 64:
            router_fuse_launch_rows<64>(stream, grid, block, W_d, x_d, scratch.get(), counter, weights_d, ids_d, (int) K,
                                        w_stride, x_stride, ids_stride, (int) n_rows, (int) n_expert_used, with_norm,
                                        clamp_val, scale_val);
            break;
        case 128:
            router_fuse_launch_rows<128>(stream, grid, block, W_d, x_d, scratch.get(), counter, weights_d, ids_d,
                                         (int) K, w_stride, x_stride, ids_stride, (int) n_rows, (int) n_expert_used,
                                         with_norm, clamp_val, scale_val);
            break;
        case 256:
            router_fuse_launch_rows<256>(stream, grid, block, W_d, x_d, scratch.get(), counter, weights_d, ids_d,
                                         (int) K, w_stride, x_stride, ids_stride, (int) n_rows, (int) n_expert_used,
                                         with_norm, clamp_val, scale_val);
            break;
        case 512:
            router_fuse_launch_rows<512>(stream, grid, block, W_d, x_d, scratch.get(), counter, weights_d, ids_d,
                                         (int) K, w_stride, x_stride, ids_stride, (int) n_rows, (int) n_expert_used,
                                         with_norm, clamp_val, scale_val);
            break;
        default:
            GGML_ABORT("router fuse: unsupported n_experts");
    }
    CUDA_CHECK(cudaGetLastError());

    const uint64_t n = g_router_fuse_n_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1) {
        GGML_LOG_INFO("ggml_cuda_router_fuse: GGML_HIP_ROUTER_FUSE fused router GEMV + top-k ran (n_embd=%lld "
                      "n_expert=%lld k=%lld rows=%lld norm=%d)\n",
                      (long long) K, (long long) n_experts, (long long) n_expert_used, (long long) n_rows,
                      with_norm ? 1 : 0);
    }
    return true;
}
