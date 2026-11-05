// AK-ASM-H1 + AK-ASM-H7 (autokernel-loop.md): skinny-M Q8_0 MUL_MAT for DFlash verify on gfx90a.
//
// Design (iteration 2, after gate r3: dot4 v1 scaled ~linearly with M and lost 2.6x at M=16)
//  * MFMA: v_mfma_i32_16x16x16i8, weight rows = srcA (16-row tile per wave), activation columns = srcB (M padded to 16
//    with zeros), so the math cost per weight byte is independent of M. Per Q8_0 block: 2 MFMAs from a zero accumulator,
//    then f32 += i32 * d_w[row] * d_y[col] (output lane layout D[4*(lane/16)+v][lane%16], as ggml mma.cuh J_MAJOR).
//  * Weights in place (no repacked copy). A wave streams 16 rows x 8 blocks (16 x 272 B) per step with fully coalesced
//    dword loads (adjacent lanes -> adjacent dwords of one row, 17 loads in flight per lane), double-buffered in
//    registers: the next step's loads are issued before the current step's math. The step is staged in a wave-private
//    LDS slab with a 69-dword row stride (conflict-free across the 16 rows) and A fragments / d_w are read as u16.
//  * Activations are quantized once per call to int8 + f32 scale per 32 into a 16-column zero-padded buffer and read as
//    srcB fragments straight from global/L2 (tiny, shared by every wave).
//  * Split-K reduction (H7): "twopass" (default, deterministic): partials [ksplit][M][N] in a pool buffer, then a
//    reduce kernel; "atomic": dst zeroed then f32 atomicAdd (global_atomic_add_f32 exists on gfx90a, coarse-grained
//    device memory only). ksplit == 1 writes dst directly.
//  Env: GGML_HIP_Q8_SKINNY=0 disables (falls back to MMVQ/MMQ exactly as C6), GGML_HIP_Q8_SKINNY_MIN (default 5),
//       GGML_HIP_Q8_SKINNY_MAX (default 16, cap 16), GGML_HIP_Q8_SKINNY_KC (blocks per K-split, multiple of 8, default auto),
//       GGML_HIP_Q8_SKINNY_REDUCE=twopass|atomic.
#include "mmq-q8-skinny.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>

#define Q8SK_MAXM        16
#define Q8SK_STEP_BLK    8                       // blocks per wave step (16 rows x 272 B)
#define Q8SK_ROW_DW      (Q8SK_STEP_BLK * 34 / 4) // 68 dwords per row per step
#define Q8SK_LDS_STRIDE  69                      // padded row stride in dwords
#define Q8SK_WAVES       4
#define Q8SK_ROWS_PER_WG (16 * Q8SK_WAVES)
#define Q8SK_THREADS     (64 * Q8SK_WAVES)

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
        if (c.kc != 0) { c.kc = std::max(Q8SK_STEP_BLK, c.kc / Q8SK_STEP_BLK * Q8SK_STEP_BLK); }
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
        ggml_is_contiguous(src0) && src0->ne[0] % (32 * Q8SK_STEP_BLK) == 0 && src0->nb[1] % 4 == 0 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        src0->ne[1] <= INT32_MAX / Q8SK_MAXM;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

// y: [M][K] f32 (row stride s1 floats) -> yq: [16][nblk*32] int8, yd: [16][nblk] f32; columns >= M are zero.
static __global__ void q8sk_quantize(const float * __restrict__ y, const int64_t s1, const int nblk, const int M,
                                     int8_t * __restrict__ yq, float * __restrict__ yd) {
    const int c   = blockIdx.y;
    const int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int i   = threadIdx.x % 32;
    if (blk >= nblk) {
        return;
    }
    const float x = c < M ? y[c * s1 + (int64_t) blk * 32 + i] : 0.0f;
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

typedef int q8sk_int4 __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int q8sk_lds_u32_at_u16(const unsigned short * p) {
    return (int) ((uint32_t) p[0] | ((uint32_t) p[1] << 16));
}

static __global__ void __launch_bounds__(Q8SK_THREADS)
q8sk_mul_mat(const char * __restrict__ W, const int64_t nb01, const int nblk, const int N, const int M,
             const int8_t * __restrict__ yq, const float * __restrict__ yd, const int KC,
             float * __restrict__ dst, const int64_t dst_s1, float * __restrict__ partial, const int mode) {
#if defined(CDNA)
    __shared__ uint32_t slab[Q8SK_WAVES][16 * Q8SK_LDS_STRIDE];

    const int wave = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int r    = lane % 16;   // srcA row / srcB column
    const int q    = lane / 16;   // k dword (0..3) / output row group
    const int row0 = blockIdx.x * Q8SK_ROWS_PER_WG + wave * 16;
    if (row0 >= N) {
        return;
    }
    uint32_t * sl = slab[wave];
    const unsigned short * sl16 = (const unsigned short *) sl;

    const int kb0 = blockIdx.y * KC;
    const int kb1 = min(nblk, kb0 + KC);

    // per-lane coalesced load plan: dword t*64 + lane of the 16 x 68 step -> (row, col)
    const char * wbase = W + (int64_t) row0 * nb01;
    uint32_t goff[17];   // byte offset from wbase (< 16 rows x row bytes, fits 32 bits)
    int ldst[17];
#pragma unroll
    for (int t = 0; t < 17; ++t) {
        const int idx = t * 64 + lane;
        const int rr  = idx / Q8SK_ROW_DW;
        const int cc  = idx - rr * Q8SK_ROW_DW;
        const int row = min(row0 + rr, N - 1);
        goff[t] = (uint32_t) ((int64_t) (row - row0) * nb01) + 4u * cc;
        ldst[t] = rr * Q8SK_LDS_STRIDE + cc;
    }

    const int * yq32 = (const int *) yq;
    const int64_t ycol = (int64_t) r * nblk;   // this lane's activation column (zero-padded to 16)

    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const q8sk_int4 zero = {0, 0, 0, 0};

    // Software pipeline: everything step s+1 needs from global (17 weight dwords + 8x(2 activation dwords + 1 scale))
    // is issued before step s's MFMA math; the wait at the top of the next iteration is the only vmcnt drain.
    uint32_t g[17];
    int   nb0[Q8SK_STEP_BLK], nb1[Q8SK_STEP_BLK];
    float ndy[Q8SK_STEP_BLK];
#pragma unroll
    for (int t = 0; t < 17; ++t) {
        g[t] = __builtin_nontemporal_load((const uint32_t *) (wbase + goff[t] + (int64_t) kb0 * 34));
    }
#pragma unroll
    for (int b = 0; b < Q8SK_STEP_BLK; ++b) {
        nb0[b] = yq32[(ycol + kb0 + b) * 8 + q];
        nb1[b] = yq32[(ycol + kb0 + b) * 8 + 4 + q];
        ndy[b] = yd[ycol + kb0 + b];
    }

    for (int kb = kb0; kb < kb1; kb += Q8SK_STEP_BLK) {
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int t = 0; t < 17; ++t) {
            sl[ldst[t]] = g[t];
        }
        int   b0[Q8SK_STEP_BLK], b1[Q8SK_STEP_BLK];
        float dy[Q8SK_STEP_BLK];
#pragma unroll
        for (int b = 0; b < Q8SK_STEP_BLK; ++b) {
            b0[b] = nb0[b]; b1[b] = nb1[b]; dy[b] = ndy[b];
        }
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        // unconditional prefetch (the last step re-reads its own L2-hot step): branch-free loop body
        const int kn = kb + Q8SK_STEP_BLK < kb1 ? kb + Q8SK_STEP_BLK : kb;
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int t = 0; t < 17; ++t) {
            g[t] = __builtin_nontemporal_load((const uint32_t *) (wbase + goff[t] + (int64_t) kn * 34));
        }
#pragma unroll
        for (int b = 0; b < Q8SK_STEP_BLK; ++b) {
            nb0[b] = yq32[(ycol + kn + b) * 8 + q];
            nb1[b] = yq32[(ycol + kn + b) * 8 + 4 + q];
            ndy[b] = yd[ycol + kn + b];
        }
        __builtin_amdgcn_sched_barrier(0);

#pragma unroll
        for (int b = 0; b < Q8SK_STEP_BLK; ++b) {
            // A fragment: row r, qs dwords q and 4+q of block b (byte offset b*34 + 2 + 4*kd in the staged row)
            const unsigned short * ar = sl16 + r * (2 * Q8SK_LDS_STRIDE) + b * 17 + 1;
            const int a0 = q8sk_lds_u32_at_u16(ar + 2 * q);
            const int a1 = q8sk_lds_u32_at_u16(ar + 2 * (4 + q));
            q8sk_int4 acc = __builtin_amdgcn_mfma_i32_16x16x16i8(a0, b0[b], zero, 0, 0, 0);
            acc           = __builtin_amdgcn_mfma_i32_16x16x16i8(a1, b1[b], acc,  0, 0, 0);
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const float dw = __half2float(__ushort_as_half(sl16[(4 * q + v) * (2 * Q8SK_LDS_STRIDE) + b * 17]));
                f[v] += (float) acc[v] * (dw * dy[b]);
            }
        }
    }

    if (r >= M) {
        return;
    }
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const int row = row0 + 4 * q + v;
        if (row >= N) {
            continue;
        }
        if (mode == 0) {
            dst[(int64_t) r * dst_s1 + row] = f[v];
        } else if (mode == 1) {
            partial[((int64_t) blockIdx.y * M + r) * N + row] = f[v];
        } else {
            unsafeAtomicAdd(&dst[(int64_t) r * dst_s1 + row], f[v]);
        }
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(nblk); GGML_UNUSED(N); GGML_UNUSED(M); GGML_UNUSED(yq);
    GGML_UNUSED(yd); GGML_UNUSED(KC); GGML_UNUSED(dst); GGML_UNUSED(dst_s1); GGML_UNUSED(partial); GGML_UNUSED(mode);
    NO_DEVICE_CODE;
#endif // defined(CDNA)
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

    // K-split: enough waves to cover 104 CUs ~8 waves deep, in whole 8-block steps.
    const int row_groups = (N + Q8SK_ROWS_PER_WG - 1) / Q8SK_ROWS_PER_WG;
    int KC = cfg.kc;
    if (KC == 0) {
        int ksplit = 1;
        while ((int64_t) row_groups * ksplit < 2 * 104 && nblk / (ksplit * 2) >= 4 * Q8SK_STEP_BLK) {
            ksplit *= 2;
        }
        KC = (nblk / Q8SK_STEP_BLK + ksplit - 1) / ksplit * Q8SK_STEP_BLK;
    }
    const int ksplit = (nblk + KC - 1) / KC;

    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), (size_t) Q8SK_MAXM * nblk * 32);
    ggml_cuda_pool_alloc<float>  yd(ctx.pool(), (size_t) Q8SK_MAXM * nblk);
    {
        const dim3 grid((nblk + 7) / 8, Q8SK_MAXM, 1);
        q8sk_quantize<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, M, yq.get(), yd.get());
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

    GGML_ASSERT(M >= 1 && M <= Q8SK_MAXM);
    const dim3 grid(row_groups, ksplit, 1);
    q8sk_mul_mat<<<grid, Q8SK_THREADS, 0, stream>>>((const char *) src0->data, src0->nb[1], nblk, N, M, yq.get(), yd.get(), KC,
                                                   dstd, dst_s1, mode == 1 ? part.get() : nullptr, mode);
    CUDA_CHECK(cudaGetLastError());

    if (mode == 1) {
        const dim3 grid((N + 255) / 256, M, 1);
        q8sk_reduce<<<grid, 256, 0, stream>>>(part.get(), ksplit, M, N, dstd, dst_s1);
        CUDA_CHECK(cudaGetLastError());
    }
}
