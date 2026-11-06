// AK-ASM-H1 + AK-ASM-H7 (autokernel-loop.md): skinny-M Q8_0 MUL_MAT for DFlash verify on gfx90a.
//
// Design (iteration 3)
//  History: it1 (37ad5c4e9, dot4) was correct but scaled with M (gate r3 geomean 0.70 vs C6); it2 (b353187c0, MFMA,
//  VGPR-staged loads, fixed split-K) went flat in M but reached only ~75% of C6's bandwidth (gate r4 0.75): 212 VGPRs
//  (2 waves/SIMD), and a fixed split-K grid of 4-wave WGs left a large last-round tail (e.g. 1280 waves over 832 slots).
//  * MFMA: v_mfma_i32_16x16x16i8, weight rows = srcA (16-row tile), activation columns = srcB (M zero-padded to 16): math
//    per weight byte is independent of M. Per Q8_0 block: 2 MFMAs from a zero accumulator, then f32 += i32*d_w*d_y
//    (output lane layout D[4*(lane/16)+v][lane%16], ggml mma.cuh J_MAJOR; validated by gate r4 correctness).
//  * Weight staging by LDS-DMA (MUBUF buffer_load_dword ... lds, the AITER technique; same intrinsic C6 already ships
//    in mmvq.cu's Q8 prefetch kernel): no VGPRs hold weight data. A work item is 16 rows x 8 Q8_0 blocks (16 x 272 B);
//    each of its 17 wave-wide DMA instructions writes 64 contiguous LDS dwords from 64 adjacent global dwords of one
//    row (coalesced). Double-buffered per wave: item i+1's DMA is in flight while item i is computed. The buffer
//    resource is bounded to the tile's valid rows, so rows >= N read as 0 (no clamping).
//  * Stream-K persistent grid: exactly (CUs x resident waves) one-wave workgroups; the linear sequence of work items
//    (tile-major, K-minor) is split into equal contiguous ranges, one per wave, so every wave streams the same number of
//    bytes and there is no last-round tail. A wave accumulates in registers while it stays inside one tile and flushes
//    with f32 atomicAdd (H7: global_atomic_add_f32, gfx90a, coarse-grained memory; no MI300-only
//    global_atomic_pk_add_bf16) when the tile changes or its range ends. dst is zeroed first. Summation order across
//    waves is not deterministic (about 2-3 flushes per output element).
//  * Activations are quantized once per call to int8 + f32 scale per 32 into a 16-column zero-padded buffer and read as
//    srcB fragments from global/L2, issued before the next item's DMA so the in-order vmcnt wait leaves that DMA in flight.
//  Env: GGML_HIP_Q8_SKINNY=0 disables (falls back to MMVQ/MMQ exactly as C6), GGML_HIP_Q8_SKINNY_MIN (default 5),
//       GGML_HIP_Q8_SKINNY_MAX (default 16, cap 16), GGML_HIP_Q8_SKINNY_WAVES (persistent wave count, default
//       CUs x occupancy), GGML_HIP_Q8_SKINNY_MAX_BYTES (shape gate, default 6 MiB of Q8_0 weight; 0 = no limit).
#include "mmq-q8-skinny.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>

#define Q8SK_MAXM     16
#define Q8SK_STEP_BLK 8                         // Q8_0 blocks per work item and row
#define Q8SK_ROW_DW   (Q8SK_STEP_BLK * 34 / 4)  // 68 dwords per row per item
#define Q8SK_ITEM_DW  (16 * Q8SK_ROW_DW)        // 1088 dwords = 17 wave-wide DMA instructions
#define Q8SK_NDMA     (Q8SK_ITEM_DW / 64)       // 17

static int q8sk_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

struct q8sk_cfg {
    bool enabled; int mmin; int mmax; int waves; int64_t max_bytes;
};

// Shape gate (derived post hoc from gate r5, 2026-11-05, MI210, 39 cells x M 5/8/16): the fast path beat C6 on every
// weight <= 5.6 MB (1024x5120 1.20x, 512x2048 2.0x, 2048x512 1.3x, 32x2048 4.5x) and lost or tied from 8.9 MB up
// (2048x4096 1.00x, 4096x2048 0.95x, 8192x2048 0.80x, the large 27B shapes 0.75-0.81x, where C6 MMVQ/MMQ sits near the
// plain-load ceiling). Default: take the fast path only when the Q8_0 weight is <= 6 MiB. Override with
// GGML_HIP_Q8_SKINNY_MAX_BYTES (bytes; 0 = no limit).
#define Q8SK_DEFAULT_MAX_BYTES (6ll << 20)

static const q8sk_cfg & q8sk_get_cfg() {
    static const q8sk_cfg cfg = [] {
        q8sk_cfg c;
        c.enabled = q8sk_env_int("GGML_HIP_Q8_SKINNY", 1) != 0;
        c.mmin    = std::max(2, q8sk_env_int("GGML_HIP_Q8_SKINNY_MIN", 5));
        c.mmax    = std::min(Q8SK_MAXM, q8sk_env_int("GGML_HIP_Q8_SKINNY_MAX", Q8SK_MAXM));
        c.waves   = std::max(0, q8sk_env_int("GGML_HIP_Q8_SKINNY_WAVES", 0));
        const char * mb = getenv("GGML_HIP_Q8_SKINNY_MAX_BYTES");
        c.max_bytes = (mb && *mb) ? std::max(0ll, atoll(mb)) : Q8SK_DEFAULT_MAX_BYTES;
        return c;
    }();
    return cfg;
}

bool ggml_cuda_q8_skinny_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    const q8sk_cfg & cfg = q8sk_get_cfg();
    if (!cfg.enabled || !GGML_CUDA_CC_IS_CDNA2(cc)) {
        return false;
    }
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= cfg.mmin && src1->ne[1] <= cfg.mmax &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src0->ne[0] % (32 * Q8SK_STEP_BLK) == 0 && src0->nb[1] % 4 == 0 &&
        (int64_t) 16 * src0->nb[1] < INT32_MAX &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        src0->ne[1] <= INT32_MAX / Q8SK_MAXM &&
        (cfg.max_bytes == 0 || (int64_t) ggml_nbytes(src0) <= cfg.max_bytes);
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

static __global__ void q8sk_zero(float * __restrict__ dst, const int N, const int64_t dst_s1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        dst[(int64_t) blockIdx.y * dst_s1 + row] = 0.0f;
    }
}

#if defined(GGML_USE_HIP)
typedef int q8sk_int4 __attribute__((ext_vector_type(4)));

// gfx9 raw buffer resource (V#): [base:64][num_records:32][cfg:32]; cfg as mmvq.cu q8pf_make_rsrc (CK gfx9 3rd dword).
static __device__ __forceinline__ q8sk_int4 q8sk_make_rsrc(const void * base, uint32_t num_bytes) {
    union { q8sk_int4 v; struct { const void * p; uint32_t range; uint32_t cfg; } s; } u;
    u.s.p     = base;
    u.s.range = num_bytes;
    u.s.cfg   = 0x00020000u;
    return u.v;
}

// global -> LDS DMA, one dword per lane; the LDS destination is M0 (wave-uniform) + 4*lane.
__device__ void q8sk_raw_buffer_load_lds(
    q8sk_int4 rsrc, __attribute__((address_space(3))) uint32_t * lds_ptr,
    int size, int voffset, int soffset, int offset, int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

static __device__ __forceinline__ int q8sk_lds_u32_at_u16(const unsigned short * p) {
    return (int) ((uint32_t) p[0] | ((uint32_t) p[1] << 16));
}
#endif // defined(GGML_USE_HIP)

static __global__ void __launch_bounds__(64)
q8sk_mul_mat(const char * __restrict__ W, const int64_t nb01, const int nblk, const int N, const int M,
             const int8_t * __restrict__ yq, const float * __restrict__ yd, const int n_items, const int nwaves,
             float * __restrict__ dst, const int64_t dst_s1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    __shared__ uint32_t slab[2][Q8SK_ITEM_DW];

    const int lane = threadIdx.x;
    const int r    = lane % 16;   // srcA row / srcB column
    const int q    = lane / 16;   // k dword (0..3) / output row group
    const int S    = nblk / Q8SK_STEP_BLK;   // items per tile

    // 32-bit item indices (n_items < 2^31 checked on the host): no 64-bit division in the loop
    const int i0 = (int) ((int64_t) blockIdx.x * n_items / nwaves);
    const int i1 = (int) ((int64_t) (blockIdx.x + 1) * n_items / nwaves);
    if (i0 >= i1) {
        return;
    }

    // per-lane DMA plan: instruction t moves LDS dwords [64t, 64t+64) <- (row, col) of the 16 x 68 item
    int voff[Q8SK_NDMA];   // byte offset inside the tile, excluding the item's K offset
#pragma unroll
    for (int t = 0; t < Q8SK_NDMA; ++t) {
        const int idx = t * 64 + lane;
        const int rr  = idx / Q8SK_ROW_DW;
        const int cc  = idx - rr * Q8SK_ROW_DW;
        voff[t] = rr * (int) nb01 + 4 * cc;
    }

    const int * yq32 = (const int *) yq;
    const int64_t ycol = (int64_t) r * nblk;

    auto dma_item = [&](const int it, const int buf) {
        const int tile = it / S;
        const int kb   = (it - tile * S) * Q8SK_STEP_BLK;
        const int row0 = tile * 16;
        const uint32_t nrows = (uint32_t) min(16, N - row0);
        const q8sk_int4 rsrc = q8sk_make_rsrc(W + (int64_t) row0 * nb01, nrows * (uint32_t) nb01);
        __attribute__((address_space(3))) uint32_t * lds = (__attribute__((address_space(3))) uint32_t *) &slab[buf][0];
#pragma unroll
        for (int t = 0; t < Q8SK_NDMA; ++t) {
            q8sk_raw_buffer_load_lds(rsrc, lds + t * 64, 4, voff[t], kb * 34, 0, 0);
        }
    };

    int   b0[Q8SK_STEP_BLK], b1[Q8SK_STEP_BLK];
    float dy[Q8SK_STEP_BLK];
    auto load_act = [&](const int it) {
        const int tile = it / S;
        const int kb   = (it - tile * S) * Q8SK_STEP_BLK;
#pragma unroll
        for (int b = 0; b < Q8SK_STEP_BLK; ++b) {
            b0[b] = yq32[(ycol + kb + b) * 8 + q];
            b1[b] = yq32[(ycol + kb + b) * 8 + 4 + q];
            dy[b] = yd[ycol + kb + b];
        }
    };

    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const q8sk_int4 zero = {0, 0, 0, 0};

    dma_item(i0, 0);
    for (int it = i0; it < i1; ++it) {
        const int buf = (it - i0) & 1;
        load_act(it);
        __builtin_amdgcn_sched_barrier(0);
        if (it + 1 < i1) {
            dma_item(it + 1, buf ^ 1);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(17)" ::: "memory");   // item it's DMA + activations landed; it+1's DMA in flight
        } else {
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
        __builtin_amdgcn_sched_barrier(0);

        const unsigned short * sl16 = (const unsigned short *) &slab[buf][0];
#pragma unroll
        for (int b = 0; b < Q8SK_STEP_BLK; ++b) {
            // A fragment: row r, qs dwords q and 4+q of block b (byte b*34 + 2 + 4*kd of the row's 272 B)
            const unsigned short * ar = sl16 + r * (2 * Q8SK_ROW_DW) + b * 17 + 1;
            const int a0 = q8sk_lds_u32_at_u16(ar + 2 * q);
            const int a1 = q8sk_lds_u32_at_u16(ar + 2 * (4 + q));
            q8sk_int4 acc = __builtin_amdgcn_mfma_i32_16x16x16i8(a0, b0[b], zero, 0, 0, 0);
            acc           = __builtin_amdgcn_mfma_i32_16x16x16i8(a1, b1[b], acc,  0, 0, 0);
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const float dw = __half2float(__ushort_as_half(sl16[(4 * q + v) * (2 * Q8SK_ROW_DW) + b * 17]));
                f[v] += (float) acc[v] * (dw * dy[b]);
            }
        }
        // all LDS reads of this buffer retire before the DMA one item ahead (issued next iteration) overwrites it
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);

        const int tile = it / S;
        if (it + 1 == i1 || (it + 1) / S != tile) {
            if (r < M) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int row = tile * 16 + 4 * q + v;
                    if (row < N) {
                        unsafeAtomicAdd(&dst[(int64_t) r * dst_s1 + row], f[v]);
                    }
                }
            }
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                f[v] = 0.0f;
            }
        }
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(nblk); GGML_UNUSED(N); GGML_UNUSED(M); GGML_UNUSED(yq);
    GGML_UNUSED(yd); GGML_UNUSED(n_items); GGML_UNUSED(nwaves); GGML_UNUSED(dst); GGML_UNUSED(dst_s1);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

void ggml_cuda_mul_mat_q8_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8sk_cfg & cfg = q8sk_get_cfg();
    {
        // Path witness (gate packets grep it; also the build's strings content marker).
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8_skinny: gfx90a skinny-M Q8_0 split-K path ran (call 1, reduce=atomic, stream-k lds-dma)\n");
        }
    }
    cudaStream_t stream = ctx.stream();

    const int     N      = (int) src0->ne[1];
    const int     M      = (int) src1->ne[1];
    const int     nblk   = (int) (src0->ne[0] / 32);
    const int64_t s1     = src1->nb[1] / sizeof(float);
    const int64_t dst_s1 = dst->nb[1]  / sizeof(float);
    GGML_ASSERT(M >= 1 && M <= Q8SK_MAXM);

    // persistent wave count = CUs x resident one-wave workgroups (queried once per device)
    static int waves_per_cu[GGML_CUDA_MAX_DEVICES] = {0};
    if (waves_per_cu[ctx.device] == 0) {
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, q8sk_mul_mat, 64, 0));
        waves_per_cu[ctx.device] = std::max(1, nb);
    }
    const int64_t n_tiles = (N + 15) / 16;
    const int64_t n_items = n_tiles * (nblk / Q8SK_STEP_BLK);
    int nwaves = cfg.waves > 0 ? cfg.waves : ggml_cuda_info().devices[ctx.device].nsm * waves_per_cu[ctx.device];
    GGML_ASSERT(n_items < INT32_MAX);
    nwaves = (int) std::min<int64_t>(nwaves, n_items);

    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), (size_t) Q8SK_MAXM * nblk * 32);
    ggml_cuda_pool_alloc<float>  yd(ctx.pool(), (size_t) Q8SK_MAXM * nblk);
    {
        const dim3 grid((nblk + 7) / 8, Q8SK_MAXM, 1);
        q8sk_quantize<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, M, yq.get(), yd.get());
    }
    float * dstd = (float *) dst->data;
    {
        const dim3 grid((N + 255) / 256, M, 1);
        q8sk_zero<<<grid, 256, 0, stream>>>(dstd, N, dst_s1);
    }
    q8sk_mul_mat<<<nwaves, 64, 0, stream>>>((const char *) src0->data, src0->nb[1], nblk, N, M, yq.get(), yd.get(),
                                            (int) n_items, nwaves, dstd, dst_s1);
    CUDA_CHECK(cudaGetLastError());
}
