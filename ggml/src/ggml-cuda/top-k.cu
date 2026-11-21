#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE


// ---------------------------------------------------------------------------------------------
// AK-ASM-H12: radix-select top-k for rows wider than the bitonic limit (no CUB on HIP)

#include <cstdlib>

bool ggml_cuda_top_k_radix_enabled() {
    static const bool v = [] {
        const char * s = getenv("GGML_CUDA_TOPK_RADIX");
        return s == nullptr || atoi(s) != 0;
    }();
    return v;
}

static __device__ __forceinline__ uint32_t top_k_radix_key(const float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u); // order-preserving: larger float -> larger key
}

// one workgroup per row: 4 passes of an 8-bit histogram find the k-th largest key T, then a collect
// pass writes every index with key > T plus the first (k - #greater) indices with key == T (unordered)
template <int NT>
static __global__ void __launch_bounds__(NT) k_top_k_radix(const float * src, int * dst, const int ncols, const int k) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ uint32_t s_mask;
    __shared__ int      s_remaining;
    __shared__ int      s_gt;
    __shared__ int      s_eq;

    const int64_t row = blockIdx.x;
    const float * x = src + row*ncols;
    int         * y = dst + row*k;
    const int     t = threadIdx.x;

    if (t == 0) {
        s_prefix    = 0;
        s_mask      = 0;
        s_remaining = k;
        s_gt        = 0;
        s_eq        = 0;
    }

    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += NT) {
            hist[i] = 0;
        }
        __syncthreads();
        const uint32_t prefix = s_prefix;
        const uint32_t mask   = s_mask;
        for (int i = t; i < ncols; i += NT) {
            const uint32_t key = top_k_radix_key(x[i]);
            if ((key & mask) == prefix) {
                atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
            }
        }
        __syncthreads();
        if (t == 0) {
            int remaining = s_remaining;
            int b = 255;
            for (; b > 0; --b) {
                const int c = (int) hist[b];
                if (c >= remaining) {
                    break;
                }
                remaining -= c;
            }
            s_remaining = remaining;
            s_prefix    = prefix | ((uint32_t) b << shift);
            s_mask      = mask | (0xFFu << shift);
        }
        __syncthreads();
    }

    const uint32_t T        = s_prefix;
    const int      n_eq     = s_remaining;
    const int      n_gt_all = k - n_eq;

    for (int i = t; i < ncols; i += NT) {
        const uint32_t key = top_k_radix_key(x[i]);
        if (key > T) {
            y[atomicAdd(&s_gt, 1)] = i;
        } else if (key == T) {
            const int e = atomicAdd(&s_eq, 1);
            if (e < n_eq) {
                y[n_gt_all + e] = i;
            }
        }
    }
}

static void top_k_radix_cuda(const float * src, int * dst, const int ncols, const int64_t nrows, const int k, cudaStream_t stream) {
    constexpr int NT = 256;
    k_top_k_radix<NT><<<nrows, NT, 0, stream>>>(src, dst, ncols, k);
    static int n_calls = 0;
    if (n_calls++ == 0) {
        GGML_LOG_INFO("ggml_cuda_top_k_radix: radix select top-k ran (call 1)\n");
    }
}

// ---------------------------------------------------------------------------------------------
// AK-GAP-1-IDSORT: multi-block small-k top-k (GGML_HIP_TOPK_FAST, default ON since C17; =0 disables). The radix select above
// runs ONE workgroup per row, so a single vocab-wide row (A3B-MTP drafter backend top_k(10) over n_vocab)
// occupies one CU for 5 full passes (~1.3 ms on MI210). Here each element is packed into a 64-bit key
// (order-preserving float key << 32 | ~index): a strict total order in which ties go to the LOWEST index.
// Pass 1: many blocks per row, each thread keeps a sorted register list of its K best, then k rounds of a
// block-wide max emit the block's k best. Pass 2: one block per row merges the partial lists the same way.
// Output: the k indices sorted by value descending, ties by ascending index (deterministic).

bool ggml_cuda_top_k_fast_enabled() {
    static const bool v = [] {
        const char * s = getenv("GGML_HIP_TOPK_FAST");
        return s == nullptr || atoi(s) != 0;   // C17: default ON (=0 disables)
    }();
    return v;
}

static constexpr int TOPK_FAST_KMAX = 16;
static constexpr int TOPK_FAST_NT   = 256;
static constexpr int TOPK_FAST_EPT  = 8; // elements per thread per pass-1 block

static __device__ __forceinline__ uint64_t top_k_fast_pack(const float f, const int i) {
    return ((uint64_t) top_k_radix_key(f) << 32) | (uint64_t) (0xFFFFFFFFu - (uint32_t) i);
}

template <int K>
static __device__ __forceinline__ void top_k_fast_insert(uint64_t (&l)[K], uint64_t v) {
    if (v <= l[K - 1]) {
        return;
    }
    l[K - 1] = v;
#pragma unroll
    for (int j = K - 1; j > 0; --j) {
        if (l[j] > l[j - 1]) {
            const uint64_t tmp = l[j]; l[j] = l[j - 1]; l[j - 1] = tmp;
        }
    }
}

static __device__ __forceinline__ uint64_t top_k_fast_warp_max(uint64_t v) {
    constexpr int ws = ggml_cuda_get_physical_warp_size();
#pragma unroll
    for (int off = ws/2; off > 0; off >>= 1) {
        const uint32_t lo = __shfl_xor((uint32_t) (v & 0xFFFFFFFFu), off, ws);
        const uint32_t hi = __shfl_xor((uint32_t) (v >> 32), off, ws);
        const uint64_t o  = ((uint64_t) hi << 32) | lo;
        v = o > v ? o : v;
    }
    return v;
}

// k rounds of block max over each thread's list head; out(r, value) is called by thread 0
template <int NT, int K, typename F>
static __device__ __forceinline__ void top_k_fast_emit(uint64_t (&l)[K], const int k, F out) {
    constexpr int ws = ggml_cuda_get_physical_warp_size();
    constexpr int nw = NT / ws;
    __shared__ uint64_t s_w[nw];
    const int t = threadIdx.x;
    for (int r = 0; r < k; ++r) {
        uint64_t m = top_k_fast_warp_max(l[0]);
        if (t % ws == 0) {
            s_w[t / ws] = m;
        }
        __syncthreads();
        m = s_w[0];
#pragma unroll
        for (int w = 1; w < nw; ++w) {
            m = s_w[w] > m ? s_w[w] : m;
        }
        if (t == 0) {
            out(r, m);
        }
        if (l[0] == m) { // unique for real elements (distinct indices); 0 sentinels pop harmlessly
#pragma unroll
            for (int j = 0; j < K - 1; ++j) {
                l[j] = l[j + 1];
            }
            l[K - 1] = 0;
        }
        __syncthreads();
    }
}

template <int NT, int K>
static __global__ void __launch_bounds__(NT) k_top_k_fast_p1(const float * src, uint64_t * part, const int ncols, const int k, const int chunk) {
    const int64_t row = blockIdx.y;
    const int     b   = blockIdx.x;
    const float * x   = src + row*ncols;
    uint64_t    * p   = part + (row*gridDim.x + b)*k;
    const int start = b*chunk;
    const int end   = min(ncols, start + chunk);

    uint64_t l[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        l[j] = 0;
    }
    for (int i = start + threadIdx.x; i < end; i += NT) {
        top_k_fast_insert<K>(l, top_k_fast_pack(x[i], i));
    }
    top_k_fast_emit<NT, K>(l, k, [&](int r, uint64_t m) { p[r] = m; });
}

template <int NT, int K>
static __global__ void __launch_bounds__(NT) k_top_k_fast_p2(const uint64_t * part, int * dst, const int nparts, const int k) {
    const int64_t row = blockIdx.x;
    const uint64_t * p = part + row*nparts;
    int            * y = dst + row*k;

    uint64_t l[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        l[j] = 0;
    }
    for (int i = threadIdx.x; i < nparts; i += NT) {
        top_k_fast_insert<K>(l, p[i]);
    }
    top_k_fast_emit<NT, K>(l, k, [&](int r, uint64_t m) { y[r] = (int) (0xFFFFFFFFu - (uint32_t) (m & 0xFFFFFFFFu)); });
}

static void top_k_fast_cuda(ggml_cuda_pool & pool, const float * src, int * dst, const int ncols, const int64_t nrows, const int k, cudaStream_t stream) {
    constexpr int NT = TOPK_FAST_NT;
    constexpr int K  = TOPK_FAST_KMAX;
    const int chunk = NT*TOPK_FAST_EPT;
    const int nb    = (ncols + chunk - 1) / chunk;
    ggml_cuda_pool_alloc<uint64_t> part(pool, (size_t) nrows*nb*k);
    k_top_k_fast_p1<NT, K><<<dim3(nb, nrows, 1), NT, 0, stream>>>(src, part.get(), ncols, k, chunk);
    k_top_k_fast_p2<NT, K><<<nrows, NT, 0, stream>>>(part.get(), dst, nb*k, k);
    static int n_calls = 0;
    if (n_calls++ == 0) {
        GGML_LOG_INFO("ggml_cuda_top_k_fast: GGML_HIP_TOPK_FAST multi-block top-k ran (call 1, ncols=%d k=%d nrows=%lld nb=%d)\n",
                      ncols, k, (long long) nrows, nb);
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    if (ncols > 1024 && k <= TOPK_FAST_KMAX && nrows <= 65535 && ggml_cuda_top_k_fast_enabled()) {
        top_k_fast_cuda(pool, src0_d, dst_d, (int) ncols, nrows, (int) k, stream);
        return;
    }
    if (ncols > 1024 && ggml_cuda_top_k_radix_enabled()) {
        top_k_radix_cuda(src0_d, dst_d, (int) ncols, nrows, (int) k, stream);
        return;
    }
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
