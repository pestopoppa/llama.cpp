// AK-ASM-H1b (autokernel-loop.md AK-ASM-H1): wvSplitK-style skinny-M Q8_0 MUL_MAT for DFlash verify on gfx90a,
// for LARGE weights (> 6 MiB by default). Small weights stay on mmq-q8-skinny.cu (it wins 1.2-4.5x there); large ones
// lost there (0.75-0.83x of C6, gate r5 14372feeb), C6 sitting at ~570 GB/s = ~35% of MI210 HBM (1.64 TB/s).
//
// Design note (2026-11-06) - port of AITER wvSplitK_hf_big_ (csrc/kernels/custom_kernels.cu) to Q8_0 weights
//  What AITER does and why it streams fast:
//   * grid = #CUs, ONE persistent 1024-thread WG per CU (64 KB static LDS forces 1 WG/CU), 16 waves.
//   * a wave owns YTILE whole weight rows (columns of their B) over the FULL K; the 64 lanes split K ("split-K inside
//     the wave"): every lane issues one 16 B nontemporal load per row per UNRL step, so one wave instruction moves
//     1 KB contiguous of one row (perfect coalescing, no LDS for weights, no MFMA), and YTILE*UNRL such loads are
//     issued back to back before any math -> YTILE*UNRL KB in flight per wave, 16 waves per CU.
//   * activations (N <= 4 columns there) are paged into LDS in K chunks that fit 64 KB ("PCML"), all waves of the WG
//     walk the same K chunk sequence (row loop rounded up so every wave runs the same trip count -> barriers are safe).
//   * the activation chunk in LDS is reused by the YTILE rows of every wave of the CU; the per-row result is reduced
//     across the 64 lanes with 6 DPP adds (row_shr 8/4/2, wave_shr 1, row_bcast 15/31) and lane 63 stores it: no
//     atomics, no split-K across waves, deterministic.
//   * mindiv(): the number of active waves per WG is reduced so the row tiles divide evenly over CUs (no last-round tail).
//  Q8_0 variant (this file):
//   * lane l of a wave handles half h = l%2 (16 qs bytes) of Q8_0 block bb = k1 + 32u + l/2: one wave instruction per
//     row covers 32 blocks = 1088 contiguous bytes. Q8_0 qs are only 2-byte aligned, so each lane loads an aligned
//     20 B window (buffer_load_dwordx4 + buffer_load_dword, slc = nontemporal like AITER's loadnt) and funnel-shifts it
//     (v_alignbyte by 0 or 2 bytes) - plus a 16-bit load of the block scale (same cache line, TA hit). Raw buffer loads
//     with the tensor as the V# range: past-the-end windows and invalid lanes (offset forced out of range) read 0.
//   * activations are quantized once per call to int8 + f32 scale per 32 (q8_1-like, same precision model as C6
//     MMVQ's vec_dot_type), paged into LDS as [M][P blocks][32 B] + [M][P] f32, P = largest multiple of 32*UNRL with
//     M*P*36 <= 64 KB (M=16: 96 blocks, M=8: 224, M=5: 320). The resident page is kept across row tiles, so when K
//     fits one page (K <= 7168 at M=8, <= 10240 at M=5) the activations are staged ONCE per kernel.
//   * math: v_dot4_i32_i8 x4 per (row, column, half-block), f32 acc += i32 * d_w * d_a. Accumulators YTILE*M f32.
//     YTILE = 4 rows/wave; UNRL = 2 for M <= 5, else 1 (16 waves/CU cap the kernel at 128 VGPR; UNRL 2 at M 6..8
//     and odd M 9..15 spilled - odd M > 8 runs the M+1 instantiation with a zero column). ISA (ROCm 6.2, gfx90a):
//     79-126 VGPR, 0 VGPR/SGPR spills, 0 scratch, LDS 65536, v_dot4c_i32_i8 + v_alignbyte_b32 + buffer_load_dwordx4 slc.
//   * weight loads for step k1 are issued BEFORE the (rare) page-restage barrier, so HBM stays busy across it.
//  Bandwidth estimate (MI210: 104 CU, 1.7 GHz, HBM 1.64 TB/s peak, ~1.35 TB/s practical stream):
//   * bytes in flight: 16 waves x 4 rows x UNRL x 1088 B = 70 KB/CU (M>5) .. 139 KB/CU (M<=5); Little's law at
//     1.35 TB/s x ~2 us loaded latency needs ~26 KB/CU -> 2.7-5x headroom (it3 had ~60 KB/CU but LDS-DMA dword ops).
//   * LDS reads: M*(16+4) B per lane per 4 rows*17 B of weight = 4.7 B/B at M=16 -> cap ~4.8 TB/s of weight
//     (128 B/clk/CU); 1.5 B/B at M=5. VALU (~7 lane-ops per column per 16 B): cap ~6 TB/s at M=16.
//   * L2 activation staging when K > P (M=16): 16*nblk*36 B per WG per row-tile round vs 14 waves*4 rows*nb01 of
//     weight -> +30% L2 (not HBM) reads, 2 extra barriers per round. mindiv balance >= 97% on every 27B shape.
//   * => expected 0.95-1.2 TB/s at M<=8 (1.65-2.1x C6) and 0.8-1.05 TB/s at M=16 (1.4-1.8x C6); pessimistic
//     (window loads splitting TA segments, barrier serialization): ~0.7 TB/s = 1.2x. Gate target 1.10x = ~630 GB/s.
//  Env: GGML_HIP_Q8_WVSK=0 disables (C6 route), GGML_HIP_Q8_WVSK_MIN_BYTES (default 6 MiB: fast path only when the
//       Q8_0 weight is strictly larger), GGML_HIP_Q8_WVSK_MIN / _MAX (M range, default 5..16), GGML_HIP_Q8_WVSK_WAVES
//       (max waves per WG, default 16; 8 at M >= 12), GGML_HIP_Q8_WVSK_M12_MIN_ROWS (M >= 12 needs >= this many rows,
//       default 4096; 0 = off).
#include "mmq-q8-wvsk.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>

#define Q8WV_MAXM    16
#define Q8WV_YT      4
#define Q8WV_LDS_B   65536
#define Q8WV_MAXWAVE 16

static int q8wv_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

struct q8wv_cfg {
    bool enabled; int mmin; int mmax; int waves; int64_t min_bytes; int big_m_min_rows;
};

static const q8wv_cfg & q8wv_get_cfg() {
    static const q8wv_cfg cfg = [] {
        q8wv_cfg c;
        c.enabled = q8wv_env_int("GGML_HIP_Q8_WVSK", 1) != 0;
        c.mmin    = std::max(2, q8wv_env_int("GGML_HIP_Q8_WVSK_MIN", 5));
        c.mmax    = std::min(Q8WV_MAXM, q8wv_env_int("GGML_HIP_Q8_WVSK_MAX", Q8WV_MAXM));
        c.waves   = std::min(Q8WV_MAXWAVE, std::max(1, q8wv_env_int("GGML_HIP_Q8_WVSK_WAVES", Q8WV_MAXWAVE)));
        // Post hoc from gate r1 (ea922d61f): at M >= 12 the only A3B loss was 2048x4096 M16 (0.966x; 2048 rows = too few row
        // tiles for 104 CUs), while every >= 4096-row shape won. M >= 12 therefore needs >= 4096 rows; r2 confirms on fresh reps.
        c.big_m_min_rows = std::max(0, q8wv_env_int("GGML_HIP_Q8_WVSK_M12_MIN_ROWS", 4096));
        const char * mb = getenv("GGML_HIP_Q8_WVSK_MIN_BYTES");
        c.min_bytes = (mb && *mb) ? std::max(0ll, atoll(mb)) : (6ll << 20);
        return c;
    }();
    return cfg;
}

bool ggml_cuda_q8_wvsk_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    const q8wv_cfg & cfg = q8wv_get_cfg();
    if (!cfg.enabled || !GGML_CUDA_CC_IS_CDNA2(cc)) {
        return false;
    }
    const int64_t nbytes = (int64_t) ggml_nbytes(src0);
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= cfg.mmin && src1->ne[1] <= cfg.mmax &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src0->nb[1] % 4 == 0 && src0->ne[0] % 64 == 0 &&   // even nblk: no V#-straddling window at the tensor end
       
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        nbytes > cfg.min_bytes && (src1->ne[1] < 12 || src0->ne[1] >= cfg.big_m_min_rows) && nbytes < (int64_t) INT32_MAX - 4096;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

// y: [M][K] f32 (row stride s1 floats) -> yq: [MK][nblk*32] int8, yd: [MK][nblk] f32 (columns >= M zero)
static __global__ void q8wv_quantize(const float * __restrict__ y, const int64_t s1, const int nblk, const int M,
                                     int8_t * __restrict__ yq, float * __restrict__ yd) {
    const int c   = blockIdx.y;
    const int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int i   = threadIdx.x % 32;
    if (blk >= nblk) {
        return;
    }
    const float x = c < M ? y[c * s1 + (int64_t) blk * 32 + i] : 0.0f;   // padded columns are 0
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
typedef int q8wv_int4 __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ q8wv_int4 q8wv_make_rsrc(const void * base, uint32_t num_bytes) {
    union { q8wv_int4 v; struct { const void * p; uint32_t range; uint32_t cfg; } s; } u;
    u.s.p     = base;
    u.s.range = num_bytes;
    u.s.cfg   = 0x00020000u;   // gfx9 raw buffer dword3, as mmvq.cu q8pf_make_rsrc / CK
    return u.v;
}

// aux: bit0 glc, bit1 slc (slc = nontemporal streaming, AITER loadnt)
__device__ q8wv_int4 q8wv_buf_x4(q8wv_int4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ int       q8wv_buf_x1(q8wv_int4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ short     q8wv_buf_s16(q8wv_int4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i16");

// LDS-only WG barrier: __syncthreads() would also drain vmcnt, i.e. wait for this step's weight loads issued just above
#define Q8WV_LDS_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
#define Q8WV_DPP_ADD(x, ctrl) asm("s_nop 0\n\tv_add_f32 %0, %2, %3 " ctrl " bound_ctrl:0" : "=v"(x) : "0"(x), "v"(x), "v"(x))
#endif // defined(GGML_USE_HIP)

// grid = #CUs, block = (64, wv); every wave runs the same number of row-tile iterations (n_rnd) so the WG-wide
// barriers of the activation page restage are uniform.
template <int M, int UN, int YT, int MAXW>
static __global__ void __launch_bounds__(64 * MAXW)
q8wv_mul_mat(const char * __restrict__ W, const int nb01, const int nblk, const int N, const uint32_t wbytes,
             const int8_t * __restrict__ yq, const float * __restrict__ yd, const int P, const int n_tiles, const int n_rnd,
             const int m_real, float * __restrict__ dst, const int64_t dst_s1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    __shared__ q8wv_int4 lds[Q8WV_LDS_B / 16];
    float * lds_d = (float *) &lds[M * P * 2];

    const int lane = threadIdx.x;
    const int wv   = blockDim.y;
    const int tid  = threadIdx.y * 64 + lane;
    const int nthr = 64 * wv;
    const int h    = lane & 1;
    const int hb   = lane >> 1;
    const q8wv_int4 rsrc = q8wv_make_rsrc(W, wbytes);
    const q8wv_int4 * yq4 = (const q8wv_int4 *) yq;

    int page = INT_MIN / 2;   // first block of the resident activation page

    for (int rnd = 0; rnd < n_rnd; ++rnd) {
        const int  t      = (rnd * gridDim.x + blockIdx.x) * wv + threadIdx.y;
        const bool active = t < n_tiles;
        const int  row0   = t * YT;

        float acc[YT][M];
#pragma unroll
        for (int y = 0; y < YT; ++y) {
#pragma unroll
            for (int n = 0; n < M; ++n) {
                acc[y][n] = 0.0f;
            }
        }

        for (int k1 = 0; k1 < nblk; k1 += 32 * UN) {
            // 1) weight windows for this step, all issued before any wait (AITER: bigB loads first)
            q8wv_int4 wa[YT][UN];
            int       wb[YT][UN];
            short     ws[YT][UN];
            int       sh[UN];
#pragma unroll
            for (int u = 0; u < UN; ++u) {
                const int bb = k1 + 32 * u + hb;
                const int tb = bb * 34 + 2 + 16 * h;   // first qs byte of this lane's half block (row-relative)
                sh[u] = tb & 3;
#pragma unroll
                for (int y = 0; y < YT; ++y) {
                    const int  row = row0 + y;
                    const bool ok  = active && row < N && bb < nblk;
                    const int  rb  = row * nb01;
                    const int  vo  = ok ? rb + (tb & ~3) : 0x7ffffff0;   // out of V# range -> reads 0
                    const int  vs  = ok ? rb + bb * 34   : 0x7ffffff0;
                    wa[y][u] = q8wv_buf_x4(rsrc, vo,      0, 2);
                    wb[y][u] = q8wv_buf_x1(rsrc, vo + 16, 0, 2);
                    ws[y][u] = q8wv_buf_s16(rsrc, vs,     0, 0);
                }
            }

            // 2) activation page restage (uniform across the WG: every wave has the same k1 sequence)
            if (k1 < page || k1 >= page + P) {
                Q8WV_LDS_BARRIER();
                for (int i = tid; i < M * P * 2; i += nthr) {
                    const int n = i / (2 * P);
                    const int r = i - n * 2 * P;
                    const int b = k1 + r / 2;
                    q8wv_int4 v = {0, 0, 0, 0};
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
                Q8WV_LDS_BARRIER();
                page = k1;
            }

            // 3) math: 4 x v_dot4_i32_i8 per (row, column, half block)
#pragma unroll
            for (int u = 0; u < UN; ++u) {
                const int bl = k1 - page + 32 * u + hb;
                int qw[YT][4];
                float dw[YT];
#pragma unroll
                for (int y = 0; y < YT; ++y) {
                    const int d5[5] = {wa[y][u][0], wa[y][u][1], wa[y][u][2], wa[y][u][3], wb[y][u]};
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        qw[y][j] = (int) __builtin_amdgcn_alignbyte((uint32_t) d5[j + 1], (uint32_t) d5[j], (uint32_t) sh[u]);
                    }
                    dw[y] = __half2float(__ushort_as_half((unsigned short) ws[y][u]));
                }
#pragma unroll
                for (int n = 0; n < M; ++n) {
                    const q8wv_int4 a  = lds[(n * P + bl) * 2 + h];
                    const float     da = lds_d[n * P + bl];
#pragma unroll
                    for (int y = 0; y < YT; ++y) {
                        int s = ggml_cuda_dp4a(qw[y][0], a[0], 0);
                        s     = ggml_cuda_dp4a(qw[y][1], a[1], s);
                        s     = ggml_cuda_dp4a(qw[y][2], a[2], s);
                        s     = ggml_cuda_dp4a(qw[y][3], a[3], s);
                        acc[y][n] += (float) s * (dw[y] * da);
                    }
                }
            }
        }

        // 4) 64-lane DPP reduction (AITER), lane 63 holds the row total
#pragma unroll
        for (int y = 0; y < YT; ++y) {
#pragma unroll
            for (int n = 0; n < M; ++n) {
                Q8WV_DPP_ADD(acc[y][n], "row_shr:8");
                Q8WV_DPP_ADD(acc[y][n], "row_shr:4");
                Q8WV_DPP_ADD(acc[y][n], "row_shr:2");
                Q8WV_DPP_ADD(acc[y][n], "wave_shr:1");
                Q8WV_DPP_ADD(acc[y][n], "row_bcast:15");
                Q8WV_DPP_ADD(acc[y][n], "row_bcast:31");
            }
        }
        if (active && lane == 63) {
#pragma unroll
            for (int y = 0; y < YT; ++y) {
                if (row0 + y < N) {
#pragma unroll
                    for (int n = 0; n < M - 1; ++n) {
                        dst[(int64_t) n * dst_s1 + row0 + y] = acc[y][n];
                    }
                    if (m_real == M) {   // else the last column is the zero pad
                        dst[(int64_t) (M - 1) * dst_s1 + row0 + y] = acc[y][M - 1];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(nblk); GGML_UNUSED(N); GGML_UNUSED(wbytes); GGML_UNUSED(yq);
    GGML_UNUSED(yd); GGML_UNUSED(P); GGML_UNUSED(n_tiles); GGML_UNUSED(n_rnd); GGML_UNUSED(m_real); GGML_UNUSED(dst); GGML_UNUSED(dst_s1);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

template <int M>
static void q8wv_launch(const char * W, int nb01, int nblk, int N, uint32_t wbytes, const int8_t * yq, const float * yd,
                        int ncu, int max_waves, int m_real, float * dst, int64_t dst_s1, cudaStream_t stream) {
    constexpr int UN   = M <= 5 ? 2 : 1;
    // M >= 12: 8 rows per wave at 8 waves/CU (256-VGPR budget) - halves the per-weight-byte LDS activation reads that capped
    // r1 M=16 at ~550-600 GB/s on every large shape regardless of page restaging (gate r1 per-cell table).
    constexpr int YT   = M >= 12 ? 8 : Q8WV_YT;
    constexpr int MAXW = M >= 12 ? 8 : Q8WV_MAXWAVE;
    max_waves = std::min(max_waves, MAXW);   // UN=2 at M 6..8 spills under the 128-VGPR cap of 16 waves/CU
    constexpr int STEP = 32 * UN;
    int P = (Q8WV_LDS_B / (36 * M)) / STEP * STEP;
    P = std::min(P, (nblk + STEP - 1) / STEP * STEP);
    GGML_ASSERT(P >= STEP && M * P * 36 <= Q8WV_LDS_B);

    // mindiv (AITER): fewest rounds at max_waves, then the fewest waves per WG that still need only that many rounds
    const int n_tiles = (N + YT - 1) / YT;
    const int n_rnd   = (n_tiles + ncu * max_waves - 1) / (ncu * max_waves);
    const int wv      = (n_tiles + ncu * n_rnd - 1) / (ncu * n_rnd);
    const dim3 block(64, wv, 1);
    q8wv_mul_mat<M, UN, YT, MAXW><<<ncu, block, 0, stream>>>(W, nb01, nblk, N, wbytes, yq, yd, P, n_tiles, n_rnd, m_real, dst, dst_s1);
}

void ggml_cuda_mul_mat_q8_wvsk(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8wv_cfg & cfg = q8wv_get_cfg();
    {
        // Path witness (gate packets grep it; also the build's strings content marker).
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8_wvsk: gfx90a wvSplitK Q8_0 path ran (call 1, dpp reduce, lds act pages)\n");
        }
    }
    cudaStream_t stream = ctx.stream();

    const int     N      = (int) src0->ne[1];
    const int     M      = (int) src1->ne[1];
    const int     nblk   = (int) (src0->ne[0] / 32);
    const int64_t s1     = src1->nb[1] / sizeof(float);
    const int64_t dst_s1 = dst->nb[1]  / sizeof(float);
    GGML_ASSERT(M >= 1 && M <= Q8WV_MAXM);

    // odd M > 8 runs the M+1 instantiation with a zero column (odd 9..15 spill under the 128-VGPR cap; even do not)
    const int MK = (M > 8 && (M & 1)) ? M + 1 : M;
    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), (size_t) MK * nblk * 32);
    ggml_cuda_pool_alloc<float>  yd(ctx.pool(), (size_t) MK * nblk);
    {
        const dim3 grid((nblk + 7) / 8, MK, 1);
        q8wv_quantize<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, M, yq.get(), yd.get());
    }
    const char *   W      = (const char *) src0->data;
    const int      nb01   = (int) src0->nb[1];
    const uint32_t wbytes = (uint32_t) ggml_nbytes(src0);
    const int      ncu    = ggml_cuda_info().devices[ctx.device].nsm;
    float *        d      = (float *) dst->data;
    switch (MK) {
#define Q8WV_CASE(m) case m: q8wv_launch<m>(W, nb01, nblk, N, wbytes, yq.get(), yd.get(), ncu, cfg.waves, M, d, dst_s1, stream); break;
        Q8WV_CASE(2)  Q8WV_CASE(3)  Q8WV_CASE(4)  Q8WV_CASE(5)  Q8WV_CASE(6)  Q8WV_CASE(7)  Q8WV_CASE(8)
        Q8WV_CASE(10) Q8WV_CASE(12) Q8WV_CASE(14) Q8WV_CASE(16)
#undef Q8WV_CASE
        default: GGML_ABORT("q8_wvsk: unsupported M=%d", MK);
    }
    CUDA_CHECK(cudaGetLastError());
}
