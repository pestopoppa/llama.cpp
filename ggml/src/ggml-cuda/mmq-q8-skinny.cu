// AK-ASM-H1 + AK-ASM-H7 (autokernel-loop.md): skinny-M Q8_0 MUL_MAT for DFlash verify on gfx90a.
//
// Design
//  * Weights are read in place (no repack copy: a repacked duplicate of a 28 GiB Q8_0 model does not fit next to
//    the original). Two Q8_0 blocks = 68 B = 17 dwords, and every eligible row starts 4-byte aligned
//    (ne00 % 64 == 0), so a lane loads one block PAIR as 17 aligned dword loads. Two pairs are issued before any
//    math, i.e. ~34 global loads in flight per lane (AITER-style deep prefetch, report item #1: ~18).
//  * Activations are quantized once per call to int8 + f32 scale per 32 (same precision model as the CPU
//    reference, vec_dot_type Q8_0) and staged per K-chunk in LDS; dot products use v_dot4_i32_i8.
//  * Grid = (row groups of 32, K-splits). 8 lanes cooperate on one row inside a chunk, 8 rows per wave.
//  * Split-K reduction (H7): "twopass" (default, deterministic): partials [ksplit][M][N] in a pool buffer, then a
//    reduce kernel; "atomic": dst zeroed then f32 atomicAdd (global_atomic_add_f32 exists on gfx90a, coarse-grained
//    device memory only). ksplit == 1 writes dst directly.
//  Env: GGML_HIP_Q8_SKINNY=0 disables (falls back to MMVQ/MMQ exactly as C6), GGML_HIP_Q8_SKINNY_MIN (default 5),
//       GGML_HIP_Q8_SKINNY_MAX (default 16, cap 16), GGML_HIP_Q8_SKINNY_KC (blocks per K-chunk, 16..64, default auto),
//       GGML_HIP_Q8_SKINNY_REDUCE=twopass|atomic.
#include "mmq-q8-skinny.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>

#define Q8SK_MAXM   16
#define Q8SK_MAXKC  64
#define Q8SK_ROWS_PER_WG 32
#define Q8SK_THREADS 256

static int q8sk_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

struct q8sk_cfg {
    bool enabled; int mmin; int mmax; int kc; bool atomic;
};

static const q8sk_cfg & q8sk_get_cfg() {
    static const q8sk_cfg cfg = [] {
        q8sk_cfg c;
        c.enabled = q8sk_env_int("GGML_HIP_Q8_SKINNY", 1) != 0;
        c.mmin    = std::max(2, q8sk_env_int("GGML_HIP_Q8_SKINNY_MIN", 5));
        c.mmax    = std::min(Q8SK_MAXM, q8sk_env_int("GGML_HIP_Q8_SKINNY_MAX", Q8SK_MAXM));
        c.kc      = q8sk_env_int("GGML_HIP_Q8_SKINNY_KC", 0);
        if (c.kc != 0) { c.kc = std::max(16, std::min(Q8SK_MAXKC, c.kc)) & ~1; }
        const char * r = getenv("GGML_HIP_Q8_SKINNY_REDUCE");
        c.atomic  = r && strcmp(r, "atomic") == 0;
        return c;
    }();
    return cfg;
}

bool ggml_cuda_q8_skinny_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    const q8sk_cfg & cfg = q8sk_get_cfg();
    if (!cfg.enabled || !GGML_CUDA_CC_IS_CDNA(cc)) {
        return false;
    }
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= cfg.mmin && src1->ne[1] <= cfg.mmax &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src0->ne[0] % 64 == 0 && src0->nb[1] % 4 == 0 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        src0->ne[1] <= INT32_MAX / Q8SK_MAXM;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

// y: [M][K] f32 (row stride s1 floats) -> yq: [M][nblk*32] int8, yd: [M][nblk] f32
static __global__ void q8sk_quantize(const float * __restrict__ y, const int64_t s1, const int nblk,
                                     int8_t * __restrict__ yq, float * __restrict__ yd) {
    const int c   = blockIdx.y;
    const int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int i   = threadIdx.x % 32;
    if (blk >= nblk) {
        return;
    }
    const float x = y[c * s1 + (int64_t) blk * 32 + i];
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    }
    const float d  = amax / 127.0f;
    const int   q  = amax == 0.0f ? 0 : (int) roundf(x / d);
    yq[((int64_t) c * nblk + blk) * 32 + i] = (int8_t) q;
    if (i == 0) {
        yd[(int64_t) c * nblk + blk] = d;
    }
}

// One Q8_0 block pair (17 dwords) against M LDS-staged activation columns.
// block 0: d = low half of dword 0, qs dword k = bytes [2+4k, 6+4k); block 1: d = high half of dword 8, qs = dwords 9..16
template <int M>
static __device__ __forceinline__ void q8sk_pair(const uint32_t (&v)[17], const int pb,
        const int (&ys)[M][Q8SK_MAXKC * 8], const float (&yds)[M][Q8SK_MAXKC], float (&acc)[M]) {
    const float d0 = __half2float(__ushort_as_half((unsigned short) (v[0] & 0xFFFFu)));
    const float d1 = __half2float(__ushort_as_half((unsigned short) (v[8] >> 16)));
    int q0[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        q0[k] = (int) __builtin_amdgcn_alignbyte(v[k + 1], v[k], 2);
    }
#pragma unroll
    for (int c = 0; c < M; ++c) {
        const int4 * y0 = (const int4 *) &ys[c][pb * 8];
        const int4 a0 = y0[0], a1 = y0[1], b0 = y0[2], b1 = y0[3];
        int s0 = 0, s1 = 0;
        s0 = ggml_cuda_dp4a(q0[0], a0.x, s0); s0 = ggml_cuda_dp4a(q0[1], a0.y, s0);
        s0 = ggml_cuda_dp4a(q0[2], a0.z, s0); s0 = ggml_cuda_dp4a(q0[3], a0.w, s0);
        s0 = ggml_cuda_dp4a(q0[4], a1.x, s0); s0 = ggml_cuda_dp4a(q0[5], a1.y, s0);
        s0 = ggml_cuda_dp4a(q0[6], a1.z, s0); s0 = ggml_cuda_dp4a(q0[7], a1.w, s0);
        s1 = ggml_cuda_dp4a((int) v[ 9], b0.x, s1); s1 = ggml_cuda_dp4a((int) v[10], b0.y, s1);
        s1 = ggml_cuda_dp4a((int) v[11], b0.z, s1); s1 = ggml_cuda_dp4a((int) v[12], b0.w, s1);
        s1 = ggml_cuda_dp4a((int) v[13], b1.x, s1); s1 = ggml_cuda_dp4a((int) v[14], b1.y, s1);
        s1 = ggml_cuda_dp4a((int) v[15], b1.z, s1); s1 = ggml_cuda_dp4a((int) v[16], b1.w, s1);
        acc[c] += d0 * yds[c][pb] * (float) s0 + d1 * yds[c][pb + 1] * (float) s1;
    }
}

template <int M>
static __global__ void __launch_bounds__(Q8SK_THREADS)
q8sk_mul_mat(const char * __restrict__ W, const int64_t nb01, const int nblk, const int N,
             const int8_t * __restrict__ yq, const float * __restrict__ yd, const int KC,
             float * __restrict__ dst, const int64_t dst_s1, float * __restrict__ partial, const int mode) {
    __shared__ __align__(16) int ys [M][Q8SK_MAXKC * 8];
    __shared__ float yds[M][Q8SK_MAXKC];

    const int tid  = threadIdx.x;
    const int lane = tid % 64;
    const int sub  = lane / 8;
    const int l8   = lane % 8;
    const int row  = blockIdx.x * Q8SK_ROWS_PER_WG + (tid / 64) * 8 + sub;

    const int kb0 = blockIdx.y * KC;
    const int kb1 = min(nblk, kb0 + KC);
    const int nb  = kb1 - kb0;

    const int * yq32 = (const int *) yq;
    for (int i = tid; i < M * nb * 8; i += Q8SK_THREADS) {
        const int c = i / (nb * 8);
        const int j = i - c * nb * 8;
        ys[c][j] = yq32[((int64_t) c * nblk + kb0) * 8 + j];
    }
    for (int i = tid; i < M * nb; i += Q8SK_THREADS) {
        const int c = i / nb;
        const int j = i - c * nb;
        yds[c][j] = yd[(int64_t) c * nblk + kb0 + j];
    }
    __syncthreads();

    float acc[M];
#pragma unroll
    for (int c = 0; c < M; ++c) {
        acc[c] = 0.0f;
    }

    if (row < N) {
        const uint32_t * wrow = (const uint32_t *) (W + (int64_t) row * nb01);
        const int p0 = kb0 / 2;
        const int p1 = kb1 / 2;
        for (int p = p0 + l8; p < p1; p += 16) {
            const bool has2 = p + 8 < p1;
            uint32_t v[2][17];
#pragma unroll
            for (int k = 0; k < 17; ++k) {
                v[0][k] = __builtin_nontemporal_load(wrow + (int64_t) p * 17 + k);
            }
            if (has2) {
#pragma unroll
                for (int k = 0; k < 17; ++k) {
                    v[1][k] = __builtin_nontemporal_load(wrow + (int64_t) (p + 8) * 17 + k);
                }
            }
            q8sk_pair<M>(v[0], p * 2 - kb0, ys, yds, acc);
            if (has2) {
                q8sk_pair<M>(v[1], (p + 8) * 2 - kb0, ys, yds, acc);
            }
        }
    }

#pragma unroll
    for (int c = 0; c < M; ++c) {
        acc[c] += __shfl_xor(acc[c], 1, 8);
        acc[c] += __shfl_xor(acc[c], 2, 8);
        acc[c] += __shfl_xor(acc[c], 4, 8);
    }
    if (row >= N) {
        return;
    }
#pragma unroll
    for (int c = 0; c < M; ++c) {
        if (c % 8 != l8) {
            continue;
        }
        if (mode == 0) {
            dst[(int64_t) c * dst_s1 + row] = acc[c];
        } else if (mode == 1) {
            partial[((int64_t) blockIdx.y * M + c) * N + row] = acc[c];
        } else {
            unsafeAtomicAdd(&dst[(int64_t) c * dst_s1 + row], acc[c]);
        }
    }
}

static __global__ void q8sk_reduce(const float * __restrict__ partial, const int ksplit, const int M, const int N,
                                   float * __restrict__ dst, const int64_t dst_s1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int c   = blockIdx.y;
    if (row >= N) {
        return;
    }
    float s = 0.0f;
    for (int k = 0; k < ksplit; ++k) {
        s += partial[((int64_t) k * M + c) * N + row];
    }
    dst[(int64_t) c * dst_s1 + row] = s;
}

template <int M>
static void q8sk_launch(const char * W, int64_t nb01, int nblk, int N, const int8_t * yq, const float * yd, int KC,
                        int ksplit, float * dst, int64_t dst_s1, float * partial, int mode, cudaStream_t stream) {
    const dim3 grid((N + Q8SK_ROWS_PER_WG - 1) / Q8SK_ROWS_PER_WG, ksplit, 1);
    q8sk_mul_mat<M><<<grid, Q8SK_THREADS, 0, stream>>>(W, nb01, nblk, N, yq, yd, KC, dst, dst_s1, partial, mode);
}

void ggml_cuda_mul_mat_q8_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8sk_cfg & cfg = q8sk_get_cfg();
    {
        // Path witness (gate packets grep it; also the build's strings content marker).
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8_skinny: gfx90a skinny-M Q8_0 split-K path ran (call 1, reduce=%s)\n", cfg.atomic ? "atomic" : "twopass");
        }
    }
    cudaStream_t stream = ctx.stream();

    const int     N      = (int) src0->ne[1];
    const int     M      = (int) src1->ne[1];
    const int     nblk   = (int) (src0->ne[0] / 32);
    const int64_t s1     = src1->nb[1] / sizeof(float);
    const int64_t dst_s1 = dst->nb[1]  / sizeof(float);

    // K-chunk: enough workgroups to cover 104 CUs ~4x over, never above the LDS budget.
    int KC = cfg.kc;
    if (KC == 0) {
        const int row_groups = (N + Q8SK_ROWS_PER_WG - 1) / Q8SK_ROWS_PER_WG;
        KC = Q8SK_MAXKC;
        while (KC > 16 && (int64_t) row_groups * ((nblk + KC - 1) / KC) < 416) {
            KC /= 2;
        }
    }
    const int ksplit = (nblk + KC - 1) / KC;

    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), (size_t) M * nblk * 32);
    ggml_cuda_pool_alloc<float>  yd(ctx.pool(), (size_t) M * nblk);
    {
        const dim3 grid((nblk + 7) / 8, M, 1);
        q8sk_quantize<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, yq.get(), yd.get());
    }

    int mode = ksplit == 1 ? 0 : (cfg.atomic ? 2 : 1);
    float * dstd = (float *) dst->data;
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (mode == 1) {
        part.alloc((size_t) ksplit * M * N);
    } else if (mode == 2) {
        for (int c = 0; c < M; ++c) {
            CUDA_CHECK(cudaMemsetAsync(dstd + c * dst_s1, 0, (size_t) N * sizeof(float), stream));
        }
    }

    const char * W = (const char *) src0->data;
    float * pp = mode == 1 ? part.get() : nullptr;
    switch (M) {
        case  2: q8sk_launch< 2>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case  3: q8sk_launch< 3>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case  4: q8sk_launch< 4>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case  5: q8sk_launch< 5>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case  6: q8sk_launch< 6>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case  7: q8sk_launch< 7>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case  8: q8sk_launch< 8>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case  9: q8sk_launch< 9>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case 10: q8sk_launch<10>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case 11: q8sk_launch<11>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case 12: q8sk_launch<12>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case 13: q8sk_launch<13>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case 14: q8sk_launch<14>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case 15: q8sk_launch<15>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        case 16: q8sk_launch<16>(W, src0->nb[1], nblk, N, yq.get(), yd.get(), KC, ksplit, dstd, dst_s1, pp, mode, stream); break;
        default: GGML_ABORT("q8 skinny: unsupported M=%d", M);
    }
    CUDA_CHECK(cudaGetLastError());

    if (mode == 1) {
        const dim3 grid((N + 255) / 256, M, 1);
        q8sk_reduce<<<grid, 256, 0, stream>>>(part.get(), ksplit, M, N, dstd, dst_s1);
        CUDA_CHECK(cudaGetLastError());
    }
}
