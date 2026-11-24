// Q8-MULTICOL-GEMV-1 (handoffs/active/rocm-kernel-authoring.md): gfx90a Q8_0 multi-column GEMV, ne11 = 2..8.
// Opt-in GGML_HIP_Q8MC=1 (default OFF). Routed ahead of MMVQ (ne11 2..4) and q8wv (ne11 5..8).
//
// Why (C21 census, 27B Q8_0 verify, MI210): width 1 MMVQ streams at 68-76% of HBM peak, but width 2..4 MMVQ falls to
// 39-50% and width 5..8 q8wv to 49-57%, for the same weight bytes.
//  * MMVQ ncols 2..4 (GCN table): nwarps 2, rows/block 2 (4 at ncols 4), vdr 2: each lane moves 8 B of a row per k
//    iteration via get_int_b2 (four 16-bit loads, Q8_0 qs are only 2-byte aligned) -> ~2 KB per wave in flight per row,
//    and every row-block re-reads all ncols activation rows from L2 (+1.06..2.1x the weight bytes in L2 traffic).
//  * q8wv ncols 5..8: 16 B nontemporal windows (good), but UN=1 and NO cross-step prefetch: a wave issues one K step
//    (4 rows x 1088 B), waits for it, then runs M x 4 x 4 dp4a before the next step's loads leave. With 13-15 waves/CU
//    (one 64 KB-LDS WG per CU) the per-wave load->use bubble is not hidden (VALU 0.38-0.43, 49-57% of peak).
// This kernel keeps q8wv's memory path (16 B slc windows, 1088 B contiguous per wave instruction, activations staged
// in LDS once per page, DPP 64-lane reduce, no atomics, deterministic) and adds:
//  * a two-deep register pipeline: step k+1's weight windows are issued before step k's math, so every wave always
//    has 4 rows x 1088 B outstanding while it computes (2x the in-flight bytes of q8wv at the same occupancy);
//  * exact-M instantiations 2..8 (no zero column), one code path for widths 2..8.
// Env: GGML_HIP_Q8MC=1 enables; GGML_HIP_Q8MC_MIN / _MAX (ne11 range, default 2..8); GGML_HIP_Q8MC_MIN_BYTES (default
//      1 MiB, weight strictly larger); GGML_HIP_Q8MC_WAVES (max waves per WG, default 16).
#include "mmq-q8-mc.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>

#define Q8MC_MAXM    8
#define Q8MC_YT      4
#define Q8MC_LDS_B   65536
#define Q8MC_MAXWAVE 16
#define Q8MC_STEP    32

static int q8mc_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

struct q8mc_cfg {
    bool enabled; int mmin; int mmax; int waves; int64_t min_bytes;
};

static const q8mc_cfg & q8mc_get_cfg() {
    static const q8mc_cfg cfg = [] {
        q8mc_cfg c;
        c.enabled = q8mc_env_int("GGML_HIP_Q8MC", 0) == 1;
        c.mmin    = std::max(2, q8mc_env_int("GGML_HIP_Q8MC_MIN", 2));
        c.mmax    = std::min(Q8MC_MAXM, q8mc_env_int("GGML_HIP_Q8MC_MAX", Q8MC_MAXM));
        c.waves   = std::min(Q8MC_MAXWAVE, std::max(1, q8mc_env_int("GGML_HIP_Q8MC_WAVES", Q8MC_MAXWAVE)));
        const char * mb = getenv("GGML_HIP_Q8MC_MIN_BYTES");
        c.min_bytes = (mb && *mb) ? std::max(0ll, atoll(mb)) : (1ll << 20);
        if (c.enabled) {
            GGML_LOG_INFO("ggml_cuda_q8_mc: GGML_HIP_Q8MC=1 multi-column Q8_0 GEMV armed (M %d..%d, > %lld B, %d waves)\n",
                          c.mmin, c.mmax, (long long) c.min_bytes, c.waves);
        }
        return c;
    }();
    return cfg;
}

bool ggml_cuda_q8_mc_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    const q8mc_cfg & cfg = q8mc_get_cfg();
    if (!cfg.enabled || !GGML_CUDA_CC_IS_CDNA2(cc)) {
        return false;
    }
    const int64_t nbytes = (int64_t) ggml_nbytes(src0);
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= cfg.mmin && src1->ne[1] <= cfg.mmax &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src0->nb[1] % 4 == 0 && src0->ne[0] % 64 == 0 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        nbytes > cfg.min_bytes && nbytes < (int64_t) INT32_MAX - 4096;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

// y: [M][K] f32 (row stride s1 floats) -> yq: [M][nblk*32] int8, yd: [M][nblk] f32 (one 32-wide group per block)
static __global__ void q8mc_quantize(const float * __restrict__ y, const int64_t s1, const int nblk,
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
    const float d = amax / 127.0f;
    const int   q = amax == 0.0f ? 0 : (int) roundf(x / d);
    yq[((int64_t) c * nblk + blk) * 32 + i] = (int8_t) q;
    if (i == 0) {
        yd[(int64_t) c * nblk + blk] = d;
    }
}

#if defined(GGML_USE_HIP)
typedef int q8mc_int4 __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ q8mc_int4 q8mc_make_rsrc(const void * base, uint32_t num_bytes) {
    union { q8mc_int4 v; struct { const void * p; uint32_t range; uint32_t cfg; } s; } u;
    u.s.p     = base;
    u.s.range = num_bytes;
    u.s.cfg   = 0x00020000u;   // gfx9 raw buffer dword3 (as q8wv / q8pf)
    return u.v;
}

__device__ q8mc_int4 q8mc_buf_x4(q8mc_int4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ int       q8mc_buf_x1(q8mc_int4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ short     q8mc_buf_s16(q8mc_int4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i16");

// LDS-only WG barrier: must NOT drain vmcnt (the next step's weight windows are in flight across it)
#define Q8MC_LDS_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
#define Q8MC_DPP_ADD(x, ctrl) asm("s_nop 0\n\tv_add_f32 %0, %2, %3 " ctrl " bound_ctrl:0" : "=v"(x) : "0"(x), "v"(x), "v"(x))

struct q8mc_win {
    q8mc_int4 a[Q8MC_YT];
    int       b[Q8MC_YT];
    short     s[Q8MC_YT];
};

// lane l covers half h = l&1 (16 qs bytes) of block k1 + l/2: one wave instruction per row = 32 blocks = 1088 B.
static __device__ __forceinline__ void q8mc_issue(q8mc_win & w, const q8mc_int4 rsrc, const int k1, const int hb,
                                                  const int h, const int row0, const bool active, const int N,
                                                  const int nb01, const int nblk) {
    const int bb = k1 + hb;
    const int tb = bb * 34 + 2 + 16 * h;
#pragma unroll
    for (int y = 0; y < Q8MC_YT; ++y) {
        const int  row = row0 + y;
        const bool ok  = active && row < N && bb < nblk;
        const int  rb  = row * nb01;
        const int  vo  = ok ? rb + (tb & ~3) : 0x7ffffff0;   // outside the V# range -> reads 0
        const int  vs  = ok ? rb + bb * 34   : 0x7ffffff0;
        w.a[y] = q8mc_buf_x4(rsrc, vo,      0, 2);
        w.b[y] = q8mc_buf_x1(rsrc, vo + 16, 0, 2);
        w.s[y] = q8mc_buf_s16(rsrc, vs,     0, 0);
    }
}
#endif // defined(GGML_USE_HIP)

template <int M>
static __global__ void __launch_bounds__(64 * Q8MC_MAXWAVE)
q8mc_mul_mat(const char * __restrict__ W, const int nb01, const int nblk, const int N, const uint32_t wbytes,
             const int8_t * __restrict__ yq, const float * __restrict__ yd, const int P, const int n_tiles, const int n_rnd,
             float * __restrict__ dst, const int64_t dst_s1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    __shared__ q8mc_int4 lds[Q8MC_LDS_B / 16];
    float * lds_d = (float *) &lds[M * P * 2];

    const int lane = threadIdx.x;
    const int wv   = blockDim.y;
    const int tid  = threadIdx.y * 64 + lane;
    const int nthr = 64 * wv;
    const int h    = lane & 1;
    const int hb   = lane >> 1;
    const q8mc_int4 rsrc = q8mc_make_rsrc(W, wbytes);
    const q8mc_int4 * yq4 = (const q8mc_int4 *) yq;

    int page = INT_MIN / 2;

    for (int rnd = 0; rnd < n_rnd; ++rnd) {
        const int  t      = (rnd * gridDim.x + blockIdx.x) * wv + threadIdx.y;
        const bool active = t < n_tiles;
        const int  row0   = t * Q8MC_YT;

        float acc[Q8MC_YT][M];
#pragma unroll
        for (int y = 0; y < Q8MC_YT; ++y) {
#pragma unroll
            for (int n = 0; n < M; ++n) {
                acc[y][n] = 0.0f;
            }
        }

        q8mc_win cur;
        q8mc_issue(cur, rsrc, 0, hb, h, row0, active, N, nb01, nblk);

        for (int k1 = 0; k1 < nblk; k1 += Q8MC_STEP) {
            // 1) next step's windows leave before this step's wait (past-the-end steps read 0 and are discarded)
            q8mc_win nxt;
            q8mc_issue(nxt, rsrc, k1 + Q8MC_STEP, hb, h, row0, active && k1 + Q8MC_STEP < nblk, N, nb01, nblk);

            // 2) activation page restage, uniform across the WG
            if (k1 < page || k1 >= page + P) {
                Q8MC_LDS_BARRIER();
                for (int i = tid; i < M * P * 2; i += nthr) {
                    const int n = i / (2 * P);
                    const int r = i - n * 2 * P;
                    const int b = k1 + r / 2;
                    q8mc_int4 v = {0, 0, 0, 0};
                    if (b < nblk) {
                        v = yq4[((int64_t) n * nblk + b) * 2 + (r & 1)];
                    }
                    lds[i] = v;
                }
                for (int i = tid; i < M * P; i += nthr) {
                    const int n = i / P;
                    const int b = k1 + (i - n * P);
                    lds_d[i] = b < nblk ? yd[(int64_t) n * nblk + b] : 0.0f;
                }
                Q8MC_LDS_BARRIER();
                page = k1;
            }

            // 3) math on the current step
            const int sh = (((k1 + hb) * 34 + 2 + 16 * h) & 3);
            const int bl = k1 - page + hb;
            int   qw[Q8MC_YT][4];
            float dw[Q8MC_YT];
#pragma unroll
            for (int y = 0; y < Q8MC_YT; ++y) {
                const int d5[5] = {cur.a[y][0], cur.a[y][1], cur.a[y][2], cur.a[y][3], cur.b[y]};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qw[y][j] = (int) __builtin_amdgcn_alignbyte((uint32_t) d5[j + 1], (uint32_t) d5[j], (uint32_t) sh);
                }
                dw[y] = __half2float(__ushort_as_half((unsigned short) cur.s[y]));
            }
#pragma unroll
            for (int n = 0; n < M; ++n) {
                const q8mc_int4 a  = lds[(n * P + bl) * 2 + h];
                const float     da = lds_d[n * P + bl];
#pragma unroll
                for (int y = 0; y < Q8MC_YT; ++y) {
                    int s = ggml_cuda_dp4a(qw[y][0], a[0], 0);
                    s     = ggml_cuda_dp4a(qw[y][1], a[1], s);
                    s     = ggml_cuda_dp4a(qw[y][2], a[2], s);
                    s     = ggml_cuda_dp4a(qw[y][3], a[3], s);
                    acc[y][n] += (float) s * (dw[y] * da);
                }
            }
            cur = nxt;
        }

#pragma unroll
        for (int y = 0; y < Q8MC_YT; ++y) {
#pragma unroll
            for (int n = 0; n < M; ++n) {
                Q8MC_DPP_ADD(acc[y][n], "row_shr:8");
                Q8MC_DPP_ADD(acc[y][n], "row_shr:4");
                Q8MC_DPP_ADD(acc[y][n], "row_shr:2");
                Q8MC_DPP_ADD(acc[y][n], "wave_shr:1");
                Q8MC_DPP_ADD(acc[y][n], "row_bcast:15");
                Q8MC_DPP_ADD(acc[y][n], "row_bcast:31");
            }
        }
        if (active && lane == 63) {
#pragma unroll
            for (int y = 0; y < Q8MC_YT; ++y) {
                if (row0 + y < N) {
#pragma unroll
                    for (int n = 0; n < M; ++n) {
                        dst[(int64_t) n * dst_s1 + row0 + y] = acc[y][n];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(nblk); GGML_UNUSED(N); GGML_UNUSED(wbytes); GGML_UNUSED(yq);
    GGML_UNUSED(yd); GGML_UNUSED(P); GGML_UNUSED(n_tiles); GGML_UNUSED(n_rnd); GGML_UNUSED(dst); GGML_UNUSED(dst_s1);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

template <int M>
static void q8mc_launch(const char * W, int nb01, int nblk, int N, uint32_t wbytes, const int8_t * yq, const float * yd,
                        int ncu, int max_waves, float * dst, int64_t dst_s1, cudaStream_t stream) {
    int P = (Q8MC_LDS_B / (36 * M)) / Q8MC_STEP * Q8MC_STEP;
    P = std::min(P, (nblk + Q8MC_STEP - 1) / Q8MC_STEP * Q8MC_STEP);
    GGML_ASSERT(P >= Q8MC_STEP && M * P * 36 <= Q8MC_LDS_B);

    const int n_tiles = (N + Q8MC_YT - 1) / Q8MC_YT;
    const int n_rnd   = (n_tiles + ncu * max_waves - 1) / (ncu * max_waves);
    const int wv      = (n_tiles + ncu * n_rnd - 1) / (ncu * n_rnd);
    const dim3 block(64, wv, 1);
    q8mc_mul_mat<M><<<ncu, block, 0, stream>>>(W, nb01, nblk, N, wbytes, yq, yd, P, n_tiles, n_rnd, dst, dst_s1);
}

void ggml_cuda_mul_mat_q8_mc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8mc_cfg & cfg = q8mc_get_cfg();
    {
        // Path witness (gate rows grep it; also the build's strings content marker).
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8_mc: gfx90a Q8MC multi-column Q8_0 GEMV ran (call 1, 2-deep weight pipeline)\n");
        }
    }
    cudaStream_t stream = ctx.stream();

    const int     N      = (int) src0->ne[1];
    const int     M      = (int) src1->ne[1];
    const int     nblk   = (int) (src0->ne[0] / 32);
    const int64_t s1     = src1->nb[1] / sizeof(float);
    const int64_t dst_s1 = dst->nb[1]  / sizeof(float);
    GGML_ASSERT(M >= 2 && M <= Q8MC_MAXM);

    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), (size_t) M * nblk * 32);
    ggml_cuda_pool_alloc<float>  yd(ctx.pool(), (size_t) M * nblk);
    {
        const dim3 grid((nblk + 7) / 8, M, 1);
        q8mc_quantize<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, yq.get(), yd.get());
    }
    const char *   W      = (const char *) src0->data;
    const int      nb01   = (int) src0->nb[1];
    const uint32_t wbytes = (uint32_t) ggml_nbytes(src0);
    const int      ncu    = ggml_cuda_info().devices[ctx.device].nsm;
    float *        d      = (float *) dst->data;
    switch (M) {
#define Q8MC_CASE(m) case m: q8mc_launch<m>(W, nb01, nblk, N, wbytes, yq.get(), yd.get(), ncu, cfg.waves, d, dst_s1, stream); break;
        Q8MC_CASE(2) Q8MC_CASE(3) Q8MC_CASE(4) Q8MC_CASE(5) Q8MC_CASE(6) Q8MC_CASE(7) Q8MC_CASE(8)
#undef Q8MC_CASE
        default: GGML_ABORT("q8_mc: unsupported M=%d", M);
    }
    CUDA_CHECK(cudaGetLastError());
}
