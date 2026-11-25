// Q8-W8-GEMV-1 (handoffs/active/rocm-kernel-authoring.md): gfx90a Q8_0 GEMV at ne11 = 8 exactly. Opt-in GGML_HIP_Q8W8=1
// (default OFF), routed ahead of Q8_MF / q8wv. Design note: /mnt/raid0/llm/tmp/q8-w8-design-note-20261125.md.
//
// Why: the served 27B DFlash2 verify runs 97% of its calls at w8, where stock q8wv streams at ~55% of the 1638 GB/s peak.
// C22 Q8MC showed that at w8 the limit is registers + VALU per byte, not the memory path alone:
//  * r1 (YT=4, two-deep, half-block lanes) sat at 128 VGPRs (the 16-waves/CU cap) -> 0.95-1.01x;
//  * r2 (YT=2, three-deep) doubled the per-row activation LDS reads and reductions per byte -> 0.86-0.93x.
// What this kernel changes:
//  1. FULL-BLOCK LANES. Lane l owns Q8_0 block k1 + l (34 B) of each of its 4 rows: one 36 B aligned window per row
//     (buffer_load_dwordx4 x2 + dword, slc) = 2176 contiguous bytes per wave instruction triple (q8wv/q8mc: 1088 B per
//     triple). The scale costs no load: an even block's f16 d is the low half of its window's first dword, an odd
//     block's d is the high half of the even neighbour's last dword (DPP row_shr:1). Per (row, column, block) the math
//     is 8 dp4a + 1 cvt + 1 mul + 1 fma (half-block lanes: 2 x (4 + 3) = 14 VALU) -> ~21% less VALU per byte, and half
//     the activation scale reads and the per-byte reduction count.
//  2. REGISTER BUDGET FROM OCCUPANCY, NOT SPILLS. __launch_bounds__(512): 8 waves per WG, 2 waves per SIMD, so the
//     unified gfx90a VGPR/AGPR file gives each wave up to 256 registers (the compiler places the 32 accumulators and the
//     window ring in ArchVGPR/AGPR as it sees fit). Bytes in flight per CU: 8 waves x (DP-1) steps x 4 rows x 2176 B =
//     139 KB at DP 3 (q8wv: 16 waves x 4 x 1088 B = 70 KB, and only during each wave's load phase).
//  3. PIPELINE ACROSS TILES. The K x tile space of a wave is flattened (i = round * S + step), so the prefetch of the
//     next tile's first steps overlaps the current tile's last step and its DPP reduce: at K = 5120 a tile is only
//     2.5 steps, a per-tile pipeline would drain every 3 steps.
//  4. ACTIVATIONS ONCE PER STEP. Quantised once per call (int8 + f32 scale per 32, the MMVQ q8_1 precision model),
//     paged in LDS as [8][P][32 B] + [8][P] f32 (one page for K <= 6144 at P = 192), each lane reads its column's
//     32 B + scale once per step and applies it to all 4 rows.
// Deterministic (no atomics; lane 63 stores after the DPP reduce). Not bit-exact vs q8wv (different summation order).
// Env: GGML_HIP_Q8W8=4 design 5 (q8wp: design 4 math on an MFMA-order packed weight copy; same MYT / KS knobs),
//      GGML_HIP_Q8W8=3 design 4 (q8wm int8 MFMA; GGML_HIP_Q8W8_MYT 1|2|4 default 2, GGML_HIP_Q8W8_KS 0=auto),
//      GGML_HIP_Q8W8=1 enables design 1, =2 design 3 (q8wo, below; GGML_HIP_Q8W8_YT 2|4, default 2); GGML_HIP_Q8W8_DEPTH 2|3 (default 3); GGML_HIP_Q8W8_WAVES (max waves per WG, <= 8,
//      default 8); GGML_HIP_Q8W8_MIN_BYTES (default 8 MiB, weight strictly larger; 27B attn_k/v stays stock).
#include "mmq-q8-w8.cuh"
#include "mmq-q8-mf.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>
#include <map>
#include <mutex>
#include <cstring>

#define Q8W8_M       8
#define Q8W8_YT      4
#define Q8W8_STEP    64      // blocks per step = lanes per wave
#define Q8W8_LDS_B   65536
#define Q8W8_MAXWAVE 8
#define Q8WO_MAXWAVE 16     // design 3: 1024-thread WG, 4 waves/SIMD at <= 128 VGPR

static int q8w8_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

struct q8w8_cfg {
    bool enabled; int mode; int depth; int waves; int yt; int myt; int ks; int64_t min_bytes; int inplace;
};

static const q8w8_cfg & q8w8_get_cfg() {
    static const q8w8_cfg cfg = [] {
        q8w8_cfg c;
        c.mode    = q8w8_env_int("GGML_HIP_Q8W8", 0);
        c.enabled = c.mode == 1 || c.mode == 2 || c.mode == 3 || c.mode == 4;
        c.myt     = q8w8_env_int("GGML_HIP_Q8W8_MYT", 2);   // design 4: 16-row tiles per wave, 1|2|4
        c.myt     = c.myt == 4 ? 4 : (c.myt == 1 ? 1 : 2);
        c.ks      = q8w8_env_int("GGML_HIP_Q8W8_KS", 0);    // design 4: waves per WG splitting K (0 = auto)
        c.depth   = q8w8_env_int("GGML_HIP_Q8W8_DEPTH", 3) == 2 ? 2 : 3;
        c.yt      = q8w8_env_int("GGML_HIP_Q8W8_YT", 2) == 4 ? 4 : 2;   // design 3 only
        c.waves   = std::min(c.mode == 2 ? Q8WO_MAXWAVE : Q8W8_MAXWAVE,
                             std::max(1, q8w8_env_int("GGML_HIP_Q8W8_WAVES", c.mode == 2 ? Q8WO_MAXWAVE : Q8W8_MAXWAVE)));
        const char * mb = getenv("GGML_HIP_Q8W8_MIN_BYTES");
        c.min_bytes = (mb && *mb) ? std::max(0ll, atoll(mb)) : (8ll << 20);
        // Q8-W8-INPLACE-E2E-1: GGML_HIP_Q8W8_INPLACE=1 (weights buffers) / =2 (any buffer, test-backend-ops), design 5 only
        c.inplace = c.mode == 4 ? std::max(0, std::min(2, q8w8_env_int("GGML_HIP_Q8W8_INPLACE", 0))) : 0;
        if (c.inplace > 0) {
            GGML_LOG_INFO("ggml_cuda_q8_w8: GGML_HIP_Q8W8_INPLACE=%d design 5 q8wp layout repacked IN PLACE at load "
                          "(no cached copy; q8wp reads it at n 1..8, prefill cuBLAS converts it, others unpack on read)\n", c.inplace);
        }
        if (c.enabled && c.mode == 4) {
            GGML_LOG_INFO("ggml_cuda_q8_w8: GGML_HIP_Q8W8=4 w8 Q8_0 GEMV armed (design 5 q8wp: MFMA-order packed weight copy, "
                          "1 KB contiguous dwordx4 per wave load, int8 MFMA 16x16x16, YT %d x 16 rows, KS %d, > %lld B)\n",
                          c.myt, c.ks, (long long) c.min_bytes);
        } else if (c.enabled && c.mode == 3) {
            GGML_LOG_INFO("ggml_cuda_q8_w8: GGML_HIP_Q8W8=3 w8 Q8_0 GEMV armed (design 4 q8wm: int8 MFMA 16x16x16, raw-pair "
                          "register operands, YT %d x 16 rows, KS %d, > %lld B)\n", c.myt, c.ks, (long long) c.min_bytes);
        } else if (c.enabled && c.mode == 2) {
            GGML_LOG_INFO("ggml_cuda_q8_w8: GGML_HIP_Q8W8=2 w8 Q8_0 GEMV armed (design 3 occupancy-first q8wo: half-block lanes, "
                          "YT %d, 2-deep flattened prefetch, %d waves, > %lld B)\n", c.yt, c.waves, (long long) c.min_bytes);
        } else if (c.enabled) {
            GGML_LOG_INFO("ggml_cuda_q8_w8: GGML_HIP_Q8W8=1 w8 Q8_0 GEMV armed (ne11 == 8, full-block lanes, YT 4, depth %d, "
                          "%d waves, > %lld B)\n", c.depth, c.waves, (long long) c.min_bytes);
        }
        return c;
    }();
    return cfg;
}

bool ggml_cuda_q8_w8_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    const q8w8_cfg & cfg = q8w8_get_cfg();
    if (!cfg.enabled || !GGML_CUDA_CC_IS_CDNA2(cc) || cfg.inplace > 0) {
        return false;   // Q8W8_INPLACE: no cached copies; packed weights route through ggml_cuda_q8_w8_packed_handles
    }
    const int64_t nbytes = (int64_t) ggml_nbytes(src0);
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] == Q8W8_M &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src0->nb[1] % 4 == 0 && src0->ne[0] % 64 == 0 &&   // even nblk: block 0 of every row is 4-aligned
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        nbytes > cfg.min_bytes && nbytes < (int64_t) INT32_MAX - 4096 &&
        (cfg.mode != 4 || (src0->ne[0] % 256 == 0 && src0->ne[1] % 16 == 0));   // design 5: 4-pair scale groups, whole tiles
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

// y: [8][K] f32 (row stride s1 floats) -> yq: [8][nblk*32] int8, yd: [8][nblk] f32
static __global__ void q8w8_quantize(const float * __restrict__ y, const int64_t s1, const int nblk,
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
typedef int q8w8_int4 __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ q8w8_int4 q8w8_make_rsrc(const void * base, uint32_t num_bytes) {
    union { q8w8_int4 v; struct { const void * p; uint32_t range; uint32_t cfg; } s; } u;
    u.s.p     = base;
    u.s.range = num_bytes;
    u.s.cfg   = 0x00020000u;   // gfx9 raw buffer dword3 (as q8wv / q8mc / q8pf)
    return u.v;
}

__device__ q8w8_int4 q8w8_buf_x4(q8w8_int4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ int       q8w8_buf_x1(q8w8_int4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ short     q8w8_buf_s16(q8w8_int4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i16");

// LDS-only WG barrier: must NOT drain vmcnt (the next steps' weight windows are in flight across it)
#define Q8W8_LDS_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
#define Q8W8_DPP_ADD(x, ctrl) asm("s_nop 0\n\tv_add_f32 %0, %2, %3 " ctrl " bound_ctrl:0" : "=v"(x) : "0"(x), "v"(x), "v"(x))

// one step of one wave: 4 rows x 36 B aligned window per lane (lane = block k1 + lane)
struct q8w8_win {
    q8w8_int4 a[Q8W8_YT];
    q8w8_int4 b[Q8W8_YT];
    int       c[Q8W8_YT];
};

static __device__ __forceinline__ void q8w8_issue(q8w8_win & w, const q8w8_int4 rsrc, const int i, const int I, const int S,
                                                  const int tile_base, const int tile_stride, const int n_tiles,
                                                  const int lane, const int N, const int nb01, const int nblk) {
    const int  r    = i / S;
    const int  s    = i - r * S;
    const int  t    = tile_base + r * tile_stride;
    const int  bb   = s * Q8W8_STEP + lane;
    const bool okb  = i < I && t < n_tiles && bb < nblk;
    const int  woff = (bb * 34 + 2) & ~3;   // even block: its d + qs; odd block: its qs (d is in the even neighbour's window)
#pragma unroll
    for (int y = 0; y < Q8W8_YT; ++y) {
        const int  row = t * Q8W8_YT + y;
        const bool ok  = okb && row < N;
        const int  vo  = ok ? row * nb01 + woff : 0x7fffff00;   // outside the V# range -> reads 0
        w.a[y] = q8w8_buf_x4(rsrc, vo,      0, 2);
        w.b[y] = q8w8_buf_x4(rsrc, vo + 16, 0, 2);
        w.c[y] = q8w8_buf_x1(rsrc, vo + 32, 0, 2);
    }
}
#endif // defined(GGML_USE_HIP)

// grid = #CUs, block = (64, wv <= 8). Every wave runs the same I = n_rnd * S iterations (uniform page barriers).
template <int DP>
static __global__ void __launch_bounds__(64 * Q8W8_MAXWAVE)
q8w8_mul_mat(const char * __restrict__ W, const int nb01, const int nblk, const int N, const uint32_t wbytes,
             const int8_t * __restrict__ yq, const float * __restrict__ yd, const int P, const int n_tiles, const int n_rnd,
             float * __restrict__ dst, const int64_t dst_s1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    static_assert(DP == 2 || DP == 3, "q8w8 pipeline depth");
    constexpr int M = Q8W8_M;
    __shared__ q8w8_int4 lds[Q8W8_LDS_B / 16];
    float * lds_d = (float *) &lds[M * P * 2];

    const int lane = threadIdx.x;
    const int wv   = blockDim.y;
    const int tid  = threadIdx.y * 64 + lane;
    const int nthr = 64 * wv;
    const int S    = (nblk + Q8W8_STEP - 1) / Q8W8_STEP;
    const int I    = n_rnd * S;
    const int tile_base   = blockIdx.x * wv + threadIdx.y;
    const int tile_stride = gridDim.x * wv;
    const bool odd = lane & 1;
    const q8w8_int4 rsrc = q8w8_make_rsrc(W, wbytes);
    const q8w8_int4 * yq4 = (const q8w8_int4 *) yq;

    int page = INT_MIN / 2;

    float acc[Q8W8_YT][M];
#pragma unroll
    for (int y = 0; y < Q8W8_YT; ++y) {
#pragma unroll
        for (int n = 0; n < M; ++n) {
            acc[y][n] = 0.0f;
        }
    }

    q8w8_win cur, mid;
    q8w8_issue(cur, rsrc, 0, I, S, tile_base, tile_stride, n_tiles, lane, N, nb01, nblk);
    if constexpr (DP == 3) {
        q8w8_issue(mid, rsrc, 1, I, S, tile_base, tile_stride, n_tiles, lane, N, nb01, nblk);
    }

    for (int i = 0; i < I; ++i) {
        // 1) the window DP-1 iterations ahead leaves before this iteration's wait (may belong to the next tile)
        q8w8_win nxt;
        q8w8_issue(nxt, rsrc, i + DP - 1, I, S, tile_base, tile_stride, n_tiles, lane, N, nb01, nblk);

        const int r  = i / S;
        const int s  = i - r * S;
        const int k1 = s * Q8W8_STEP;

        // 2) activation page restage, uniform across the WG
        if (k1 < page || k1 >= page + P) {
            Q8W8_LDS_BARRIER();
            for (int j = tid; j < M * P * 2; j += nthr) {
                const int n = j / (2 * P);
                const int q = j - n * 2 * P;
                const int b = k1 + q / 2;
                q8w8_int4 v = {0, 0, 0, 0};
                if (b < nblk) {
                    v = yq4[((int64_t) n * nblk + b) * 2 + (q & 1)];
                }
                lds[j] = v;
            }
            for (int j = tid; j < M * P; j += nthr) {
                const int n = j / P;
                const int b = k1 + (j - n * P);
                lds_d[j] = b < nblk ? yd[(int64_t) n * nblk + b] : 0.0f;
            }
            Q8W8_LDS_BARRIER();
            page = k1;
        }

        // 3) unpack: 8 qs ints per row (v_alignbyte by 2 for even blocks, 0 for odd), d from the window / DPP neighbour
        const uint32_t sh = odd ? 0u : 2u;
        int   qw[Q8W8_YT][8];
        float dw[Q8W8_YT];
#pragma unroll
        for (int y = 0; y < Q8W8_YT; ++y) {
            const int d9[9] = {cur.a[y][0], cur.a[y][1], cur.a[y][2], cur.a[y][3],
                               cur.b[y][0], cur.b[y][1], cur.b[y][2], cur.b[y][3], cur.c[y]};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                qw[y][j] = (int) __builtin_amdgcn_alignbyte((uint32_t) d9[j + 1], (uint32_t) d9[j], sh);
            }
            // row_shr:1 -> lane l reads lane l-1 (odd lanes only use it; l-1 is the same row of 16, the same step)
            const int nb8 = __builtin_amdgcn_update_dpp(0, cur.c[y], 0x111, 0xf, 0xf, false);
            const uint32_t dbits = odd ? ((uint32_t) nb8 >> 16) : ((uint32_t) d9[0] & 0xffffu);
            dw[y] = __half2float(__ushort_as_half((unsigned short) dbits));
        }

        // 4) math: per column, one 32 B activation load shared by the 4 rows
        const int bl = k1 - page + lane;
#pragma unroll
        for (int n = 0; n < M; ++n) {
            const q8w8_int4 a0 = lds[(n * P + bl) * 2 + 0];
            const q8w8_int4 a1 = lds[(n * P + bl) * 2 + 1];
            const float     da = lds_d[n * P + bl];
#pragma unroll
            for (int y = 0; y < Q8W8_YT; ++y) {
                int v = ggml_cuda_dp4a(qw[y][0], a0[0], 0);
                v     = ggml_cuda_dp4a(qw[y][1], a0[1], v);
                v     = ggml_cuda_dp4a(qw[y][2], a0[2], v);
                v     = ggml_cuda_dp4a(qw[y][3], a0[3], v);
                v     = ggml_cuda_dp4a(qw[y][4], a1[0], v);
                v     = ggml_cuda_dp4a(qw[y][5], a1[1], v);
                v     = ggml_cuda_dp4a(qw[y][6], a1[2], v);
                v     = ggml_cuda_dp4a(qw[y][7], a1[3], v);
                acc[y][n] += (float) v * (dw[y] * da);
            }
        }

        if constexpr (DP == 3) {
            cur = mid;
            mid = nxt;
        } else {
            cur = nxt;
        }

        // 5) tile end: 64-lane DPP reduce, lane 63 stores, accumulators reset (the prefetch keeps running)
        if (s == S - 1) {
            const int t = tile_base + r * tile_stride;
#pragma unroll
            for (int y = 0; y < Q8W8_YT; ++y) {
#pragma unroll
                for (int n = 0; n < M; ++n) {
                    Q8W8_DPP_ADD(acc[y][n], "row_shr:8");
                    Q8W8_DPP_ADD(acc[y][n], "row_shr:4");
                    Q8W8_DPP_ADD(acc[y][n], "row_shr:2");
                    Q8W8_DPP_ADD(acc[y][n], "wave_shr:1");
                    Q8W8_DPP_ADD(acc[y][n], "row_bcast:15");
                    Q8W8_DPP_ADD(acc[y][n], "row_bcast:31");
                }
            }
            if (t < n_tiles && lane == 63) {
#pragma unroll
                for (int y = 0; y < Q8W8_YT; ++y) {
                    const int row = t * Q8W8_YT + y;
                    if (row < N) {
#pragma unroll
                        for (int n = 0; n < M; ++n) {
                            dst[(int64_t) n * dst_s1 + row] = acc[y][n];
                        }
                    }
                }
            }
#pragma unroll
            for (int y = 0; y < Q8W8_YT; ++y) {
#pragma unroll
                for (int n = 0; n < M; ++n) {
                    acc[y][n] = 0.0f;
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


// ---------------------------------------------------------------------------------------------------------------------
// Design 3 (GGML_HIP_Q8W8=2, "q8wo", occupancy-first). Limiter of stock q8wv at w8 (census r2: VALU 0.38-0.43, 13-15
// waves/CU, 49-57% of peak; attn_q = 2 rounds x 5 steps, 7.5 us per step, ~6.8 MB in flight): q8wv at M 6..8 runs UNRL 1
// with NO prefetch (UN 2 spills at 128 VGPR), so every wave does load -> wait -> ~360 VALU of math -> load. All 16 waves
// of a CU issue together and compute together (one WG, uniform k1 sequence), so HBM idles while the SIMDs run the
// ~3.4 us math phase of a step and the SIMDs idle while HBM drains it: VALU 0.41 and BW 55% are the two halves of one
// serialized loop. Design 1 (2 waves/SIMD x 2..3 deep) only traded waves for depth -> same overlap -> 0.94-0.98x.
// This kernel keeps everything q8wv does (1024-thread WG, 4 waves/SIMD, one 64 KB LDS activation page staged once per
// kernel when K fits, half-block lanes, 1088 B per wave instruction, DPP reduce, deterministic) and adds a 2-deep
// register ring over the FLATTENED (row tile, K step) sequence of the wave: the loads of step i+1 are in flight while
// step i computes, also across tile boundaries. To fit the ring under 128 VGPR the tile is YT = 2 rows (16 f32 acc +
// 2 x 2 x 6 window regs) by default; YT = 4 (GGML_HIP_Q8W8_YT=4) is the register-tight variant (32 acc + 48 ring).
template <int YT>
struct q8wo_win {
    q8w8_int4 a[YT];
    int       b[YT];
    short     s[YT];
};

template <int YT>
static __device__ __forceinline__ void q8wo_issue(q8wo_win<YT> & w, const q8w8_int4 rsrc, const int t, const bool active,
                                                  const int k1, const int hb, const int h, const int N, const int nb01,
                                                  const int nblk) {
    const int bb = k1 + hb;
    const int tb = bb * 34 + 2 + 16 * h;
#pragma unroll
    for (int y = 0; y < YT; ++y) {
        const int  row = t * YT + y;
        const bool ok  = active && row < N && bb < nblk;
        const int  rb  = row * nb01;
        const int  vo  = ok ? rb + (tb & ~3) : 0x7ffffff0;
        const int  vs  = ok ? rb + bb * 34   : 0x7ffffff0;
        w.a[y] = q8w8_buf_x4(rsrc, vo,      0, 2);
        w.b[y] = q8w8_buf_x1(rsrc, vo + 16, 0, 2);
        w.s[y] = q8w8_buf_s16(rsrc, vs,     0, 0);
    }
}

template <int YT>
static __global__ void __launch_bounds__(64 * Q8WO_MAXWAVE)
q8wo_mul_mat(const char * __restrict__ W, const int nb01, const int nblk, const int N, const uint32_t wbytes,
             const int8_t * __restrict__ yq, const float * __restrict__ yd, const int P, const int n_tiles, const int n_rnd,
             float * __restrict__ dst, const int64_t dst_s1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    constexpr int M = Q8W8_M;
    __shared__ q8w8_int4 lds[Q8W8_LDS_B / 16];
    float * lds_d = (float *) &lds[M * P * 2];

    const int lane = threadIdx.x;
    const int wv   = blockDim.y;
    const int tid  = threadIdx.y * 64 + lane;
    const int nthr = 64 * wv;
    const int h    = lane & 1;
    const int hb   = lane >> 1;
    const q8w8_int4 rsrc = q8w8_make_rsrc(W, wbytes);
    const q8w8_int4 * yq4 = (const q8w8_int4 *) yq;

    const int S = (nblk + 31) / 32;
    const int I = n_rnd * S;
    auto tile_of = [&](int r) { return (r * (int) gridDim.x + (int) blockIdx.x) * wv + (int) threadIdx.y; };

    float acc[YT][M];
#pragma unroll
    for (int y = 0; y < YT; ++y) {
#pragma unroll
        for (int n = 0; n < M; ++n) {
            acc[y][n] = 0.0f;
        }
    }

    q8wo_win<YT> cur, nxt;
    {
        const int t = tile_of(0);
        q8wo_issue<YT>(cur, rsrc, t, t < n_tiles, 0, hb, h, N, nb01, nblk);
    }
    int page = INT_MIN / 2;
    int r = 0, s = 0;
    for (int i = 0; i < I; ++i) {
        // 1) prefetch step i+1 (possibly the next tile's first step) before anything waits
        int r1 = r, s1 = s + 1;
        if (s1 == S) { s1 = 0; ++r1; }
        if (i + 1 < I) {
            const int t1 = tile_of(r1);
            q8wo_issue<YT>(nxt, rsrc, t1, t1 < n_tiles, 32 * s1, hb, h, N, nb01, nblk);
        }
        const int k1 = 32 * s;
        // 2) activation page restage (uniform over the WG; LDS-only barrier keeps both ring slots in flight)
        if (k1 < page || k1 >= page + P) {
            Q8W8_LDS_BARRIER();
            for (int j = tid; j < M * P * 2; j += nthr) {
                const int n  = j / (2 * P);
                const int rr = j - n * 2 * P;
                const int b  = k1 + rr / 2;
                q8w8_int4 v = {0, 0, 0, 0};
                if (b < nblk) {
                    v = yq4[((int64_t) n * nblk + b) * 2 + (rr & 1)];
                }
                lds[j] = v;
            }
            for (int j = tid; j < M * P; j += nthr) {
                const int n = j / P;
                const int b = k1 + (j - n * P);
                lds_d[j] = b < nblk ? yd[(int64_t) n * nblk + b] : 0.0f;
            }
            Q8W8_LDS_BARRIER();
            page = k1;
        }
        // 3) math on the current slot
        {
            const int bb = k1 + hb;
            const uint32_t al = (uint32_t) ((bb * 34 + 2 + 16 * h) & 3);
            const int bl = k1 - page + hb;
            int   qw[YT][4];
            float dw[YT];
#pragma unroll
            for (int y = 0; y < YT; ++y) {
                const int d5[5] = {cur.a[y][0], cur.a[y][1], cur.a[y][2], cur.a[y][3], cur.b[y]};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qw[y][j] = (int) __builtin_amdgcn_alignbyte((uint32_t) d5[j + 1], (uint32_t) d5[j], al);
                }
                dw[y] = __half2float(__ushort_as_half((unsigned short) cur.s[y]));
            }
#pragma unroll
            for (int n = 0; n < M; ++n) {
                const q8w8_int4 a  = lds[(n * P + bl) * 2 + h];
                const float     da = lds_d[n * P + bl];
#pragma unroll
                for (int y = 0; y < YT; ++y) {
                    int q = ggml_cuda_dp4a(qw[y][0], a[0], 0);
                    q     = ggml_cuda_dp4a(qw[y][1], a[1], q);
                    q     = ggml_cuda_dp4a(qw[y][2], a[2], q);
                    q     = ggml_cuda_dp4a(qw[y][3], a[3], q);
                    acc[y][n] += (float) q * (dw[y] * da);
                }
            }
        }
        // 4) tile done: DPP reduce, lane 63 stores, reset
        if (s == S - 1) {
#pragma unroll
            for (int y = 0; y < YT; ++y) {
#pragma unroll
                for (int n = 0; n < M; ++n) {
                    Q8W8_DPP_ADD(acc[y][n], "row_shr:8");
                    Q8W8_DPP_ADD(acc[y][n], "row_shr:4");
                    Q8W8_DPP_ADD(acc[y][n], "row_shr:2");
                    Q8W8_DPP_ADD(acc[y][n], "wave_shr:1");
                    Q8W8_DPP_ADD(acc[y][n], "row_bcast:15");
                    Q8W8_DPP_ADD(acc[y][n], "row_bcast:31");
                }
            }
            const int t = tile_of(r);
            if (t < n_tiles && lane == 63) {
#pragma unroll
                for (int y = 0; y < YT; ++y) {
                    const int row = t * YT + y;
                    if (row < N) {
#pragma unroll
                        for (int n = 0; n < M; ++n) {
                            dst[(int64_t) n * dst_s1 + row] = acc[y][n];
                        }
                    }
                }
            }
#pragma unroll
            for (int y = 0; y < YT; ++y) {
#pragma unroll
                for (int n = 0; n < M; ++n) {
                    acc[y][n] = 0.0f;
                }
            }
        }
        cur = nxt;
        r = r1; s = s1;
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(nblk); GGML_UNUSED(N); GGML_UNUSED(wbytes); GGML_UNUSED(yq);
    GGML_UNUSED(yd); GGML_UNUSED(P); GGML_UNUSED(n_tiles); GGML_UNUSED(n_rnd); GGML_UNUSED(dst); GGML_UNUSED(dst_s1);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

// ---------------------------------------------------------------------------------------------------------------------
// Design 4 (GGML_HIP_Q8W8=3, "q8wm"): purpose-built M = 8 int8 MFMA GEMV, weights straight from registers (no LDS, no
// repack). Counter row ak-27b-q8wv-w8-counters r2: stock q8wv at w8 issues 0.101 wave instructions per weight byte
// (w1: 0.041); the 8-column dp4a + per-half-block scale epilogue is 7 VALU per (row, column, 16 B). Here:
//  * v_mfma_i32_16x16x16i8, srcB = 16 weight rows (lane l: row l%16, k group g = l/16), srcA = 16 activation "columns"
//    c' = 8 * parity + column. The unit of K is a block PAIR (2 Q8_0 blocks = 68 B = 17 dwords of a row, dword aligned
//    because nblk is even). Each lane loads 5 raw dwords of its row, starting at pair dword s_g = {0, 5, 10, 12}[g]
//    (buffer_load_dwordx4 + dword): one wave instruction pair covers 16 rows x 68 contiguous B.
//  * The scale bytes and the block boundary are handled in the ACTIVATION, not the weights: activation column (p, col)
//    is laid out in raw pair byte coordinates, = q8 activation of block 2*pair+p at that block's qs bytes and 0 at the
//    two f16 scale positions, at the other block's bytes and at the 3 duplicated dwords of group 3. So 5 MFMAs per pair
//    give the exact per-(row, column, block) i32 dot products with no v_alignbyte, no masking and no padding columns.
//  * Epilogue per pair: lane holds D[c' = 4g + v][row l%16] (one parity g/2, 4 columns): 4 x (cvt, mul, fma) with the
//    lane's own block scale (one ushort load of the row's d, same lines) and 4 activation scales (one dwordx4).
//  * Activation operands (5 dwords per lane per pair, + 4 scales) are shared by YT 16-row tiles per wave.
//  * KS waves per WG split K of the same tile (deterministic LDS reduce at the end); grid = ceil(N / 16 YT) WGs.
//  Static estimate per (pair, tile): 2 + 1 VMEM, 5 MFMA, 1 + 12 VALU, + 3 / YT shared loads -> ~0.02 inst / weight B.
// ---------------------------------------------------------------------------------------------------------------------
static const char q8wm_build_marker[] __attribute__((used)) =
    "ggml-cuda: Q8W8 design 4 q8wm M8 int8 MFMA GEMV (GGML_HIP_Q8W8=3)";

// y: [8][K] f32 -> actA [npairs][16 c'][4 g][5 dwords] int8 (raw pair byte coordinates), actD [npairs][16 c'] f32.
// One WG (256 threads) per pair.
static __global__ void __launch_bounds__(256) q8wm_prep(const float * __restrict__ y, const int64_t s1,
                                                        int * __restrict__ actA, float * __restrict__ actD) {
    __shared__ int8_t sq[16][32];   // [c' = 8 p + col][i]
    const int pr  = blockIdx.x;
    const int tid = threadIdx.x;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int t   = tid + 256 * r;
        const int e   = t >> 5;       // c'
        const int i   = t & 31;
        const int p   = e >> 3;
        const int col = e & 7;
        const float x = y[col * s1 + (int64_t) pr * 64 + 32 * p + i];
        float amax = fabsf(x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, o, 32));
        }
        const float d = amax / 127.0f;
        sq[e][i] = (int8_t) (amax == 0.0f ? 0 : (int) roundf(x / d));
        if (i == 0) {
            actD[(int64_t) pr * 16 + e] = d;
        }
    }
    __syncthreads();
    for (int q = tid; q < 320; q += 256) {
        const int e  = q / 20;
        const int g  = (q / 5) & 3;
        const int ii = q % 5;
        const int p  = e >> 3;
        const int sg = g == 3 ? 12 : 5 * g;
        uint32_t v = 0;
        if (g != 3 || ii >= 3) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int b  = 4 * (sg + ii) + j;       // byte in the 68 B pair
                const int k  = p == 0 ? b - 2 : b - 36; // qs index of block 2 pr + p
                if (k >= 0 && k < 32) {
                    v |= (uint32_t) (uint8_t) sq[e][k] << (8 * j);
                }
            }
        }
        actA[(int64_t) pr * 320 + q] = (int) v;
    }
}

#if defined(GGML_USE_HIP)
typedef int q8wm_v4i __attribute__((ext_vector_type(4)));
#endif

template <int YT>
static __global__ void __launch_bounds__(1024)
q8wm_mul_mat(const char * __restrict__ W, const int nb01, const int npairs, const int N, const uint32_t wbytes,
             const int * __restrict__ actA, const uint32_t abytes, const float * __restrict__ actD, const uint32_t dbytes,
             float * __restrict__ dst, const int64_t dst_s1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    extern __shared__ float q8wm_red[];   // [KS][YT][4][64]
    constexpr uint32_t OOB = 0x7ffffff0u;
    const int lane = threadIdx.x;
    const int kw   = threadIdx.y;
    const int KS   = blockDim.y;
    const int g    = lane >> 4;
    const int r16  = lane & 15;
    const int sg4  = 4 * (g == 3 ? 12 : 5 * g);   // byte offset of this lane's window in a pair
    const int pout = g >> 1;                      // parity of this lane's D rows (c' = 4 g + v)
    const q8w8_int4 wrsrc = q8w8_make_rsrc(W, wbytes);
    const q8w8_int4 arsrc = q8w8_make_rsrc(actA, abytes);
    const q8w8_int4 drsrc = q8w8_make_rsrc(actD, dbytes);

    const int row0 = blockIdx.x * 16 * YT;
    int vrow[YT];
#pragma unroll
    for (int y = 0; y < YT; ++y) {
        const int row = row0 + 16 * y + r16;
        vrow[y] = row < N ? row * nb01 : (int) OOB;
    }
    const int p0 = (int) (((int64_t) kw * npairs) / KS);
    const int p1 = (int) (((int64_t) (kw + 1) * npairs) / KS);

    float acc[YT][4];
#pragma unroll
    for (int y = 0; y < YT; ++y) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            acc[y][v] = 0.0f;
        }
    }
    const int aoff = (r16 * 4 + g) * 20;   // this lane's activation operand (c' = r16, group g) within a pair
    for (int pr = p0; pr < p1; ++pr) {
        const int wo = pr * 68;
        q8w8_int4 wa[YT];
        int       wb[YT];
        short     ws[YT];
#pragma unroll
        for (int y = 0; y < YT; ++y) {
            const int vo = vrow[y] == (int) OOB ? (int) OOB : vrow[y] + wo;
            wa[y] = q8w8_buf_x4(wrsrc, vo == (int) OOB ? vo : vo + sg4,      0, 2);
            wb[y] = q8w8_buf_x1(wrsrc, vo == (int) OOB ? vo : vo + sg4 + 16, 0, 2);
            ws[y] = q8w8_buf_s16(wrsrc, vo == (int) OOB ? vo : vo + 34 * pout, 0, 0);
        }
        const q8w8_int4 aa = q8w8_buf_x4(arsrc, pr * 1280 + aoff,      0, 0);
        const int       ab = q8w8_buf_x1(arsrc, pr * 1280 + aoff + 16, 0, 0);
        const q8w8_int4 dd = q8w8_buf_x4(drsrc, pr * 64 + 16 * g,      0, 0);
        const float da[4] = {__int_as_float(dd[0]), __int_as_float(dd[1]), __int_as_float(dd[2]), __int_as_float(dd[3])};
#pragma unroll
        for (int y = 0; y < YT; ++y) {
            const q8wm_v4i z = {0, 0, 0, 0};
            q8wm_v4i s = __builtin_amdgcn_mfma_i32_16x16x16i8(aa[0], wa[y][0], z, 0, 0, 0);
            s          = __builtin_amdgcn_mfma_i32_16x16x16i8(aa[1], wa[y][1], s, 0, 0, 0);
            s          = __builtin_amdgcn_mfma_i32_16x16x16i8(aa[2], wa[y][2], s, 0, 0, 0);
            s          = __builtin_amdgcn_mfma_i32_16x16x16i8(aa[3], wa[y][3], s, 0, 0, 0);
            s          = __builtin_amdgcn_mfma_i32_16x16x16i8(ab,    wb[y],    s, 0, 0, 0);
            const float dw = __half2float(__ushort_as_half((unsigned short) ws[y]));
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                acc[y][v] += (float) s[v] * (dw * da[v]);
            }
        }
    }
    // parity 0 (g 0,1) + parity 1 (g 2,3): lanes l and l ^ 32 hold the same row and columns
#pragma unroll
    for (int y = 0; y < YT; ++y) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            acc[y][v] += __shfl_xor(acc[y][v], 32, 64);
        }
    }
    if (KS > 1) {
#pragma unroll
        for (int y = 0; y < YT; ++y) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                q8wm_red[((kw * YT + y) * 4 + v) * 64 + lane] = acc[y][v];
            }
        }
        __syncthreads();
        if (kw != 0) {
            return;
        }
        for (int k = 1; k < KS; ++k) {
#pragma unroll
            for (int y = 0; y < YT; ++y) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    acc[y][v] += q8wm_red[((k * YT + y) * 4 + v) * 64 + lane];
                }
            }
        }
    }
    if (g < 2) {
#pragma unroll
        for (int y = 0; y < YT; ++y) {
            const int row = row0 + 16 * y + r16;
            if (row < N) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    dst[(int64_t) (4 * g + v) * dst_s1 + row] = acc[y][v];
                }
            }
        }
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(npairs); GGML_UNUSED(N); GGML_UNUSED(wbytes); GGML_UNUSED(actA);
    GGML_UNUSED(abytes); GGML_UNUSED(actD); GGML_UNUSED(dbytes); GGML_UNUSED(dst); GGML_UNUSED(dst_s1);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

// ---------------------------------------------------------------------------------------------------------------------
// Design 5 (GGML_HIP_Q8W8=4, "q8wp"): design 4's int8 MFMA math on a weight layout PRE-PACKED IN MFMA OPERAND ORDER, so
// every weight load is one fully used, 1 KB contiguous wave dwordx4 (design 4 lost 0.48x to row-scattered 16 B loads:
// ~3x TA/L1 line requests per byte). The served V5I in-place layout ([64-row tile][256-k stage][64 rows][272 B]) is
// row-major inside a stage, NOT operand order (a lane-contiguous load of it lands row c/17 in lane c), so V5 stages it
// through LDS with barriers; this design repacks into its own layout instead.
//  Packed layout per 16-row tile (16 * nb01 bytes in both layouts: a permutation inside the tile, so it can later go in
//  place through the q8ip mechanism with no VRAM overhead; this experiment keeps a cached copy, see below):
//   qs:     [pair pr][lane l][4 dwords]   l = 16 g + r: dword j = row r, block 2 pr + (j >> 1), qs bytes 16 (j & 1) + 4 g ..
//           -> MFMA j of the pair takes dword j of every lane as srcB (16 rows x 16 k, k group g), 1024 B per pair.
//   scales: [4-pair group][lane l][2 f16] l = 16 g + r: f16 i = d of row r, block 2 (4 pg + 2 (g & 1) + i) + (g >> 1)
//           -> one 256 B contiguous wave dword load per 4 pairs; the other half of the pairs comes from lane l ^ 16.
//  Activations as design 4 (srcA c' = 8 parity + column, zero at the other parity's MFMAs) in the same lane order:
//  actA [pr][lane][4] int8x4 (1 KB per pair, L2 resident, shared by YT tiles), actD [pr][16 c'] f32.
//  Per (pair, tile): 1 dwordx4 weight load, 4 MFMA, 4 cvt + 4 mul + 4 fma + 2 scale select/cvt; per pair shared by YT:
//  1 dwordx4 actA + 1 dwordx4 actD; per 4 pairs: 1 dword scale load + 1 lane swap. No LDS, no barrier in the stream;
//  KS waves per WG split K in whole 4-pair groups (deterministic LDS reduce once at the end, as design 4).
//  Repack: one gather pass per weight on first use, cached by data pointer (a full extra copy - op microbench only;
//  dropped on any write / free of the range via ggml_cuda_q8w8_invalidate).
// ---------------------------------------------------------------------------------------------------------------------
static const char q8wp_build_marker[] __attribute__((used)) =
    "ggml-cuda: Q8W8 design 5 q8wp M8 int8 MFMA GEMV on MFMA-order packed weights (GGML_HIP_Q8W8=4)";

// natural Q8_0 [N][nb01] -> packed, one 4 B output word per thread iteration
static __global__ void q8wp_repack(const char * __restrict__ W, const int nb01, const int npairs, const int64_t nwords,
                                   uint32_t * __restrict__ P) {
    const int64_t tw = (int64_t) nb01 * 4;   // words per 16-row tile
    for (int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; q < nwords; q += (int64_t) gridDim.x * blockDim.x) {
        const int64_t T = q / tw;
        const int     c = (int) (q % tw);
        uint32_t v = 0;
        if (c < npairs * 256) {
            const int pr = c >> 8, lane = (c >> 2) & 63, j = c & 3;
            const int r  = lane & 15, g = lane >> 4;
            const uint8_t * s = (const uint8_t *) W + (T * 16 + r) * nb01 + (2 * pr + (j >> 1)) * 34 + 2 + 16 * (j & 1) + 4 * g;
            v = (uint32_t) s[0] | ((uint32_t) s[1] << 8) | ((uint32_t) s[2] << 16) | ((uint32_t) s[3] << 24);
        } else {
            const int sc = c - npairs * 256;
            const int pg = sc >> 6, lane = sc & 63;
            const int r  = lane & 15, g = lane >> 4;
            const char * row = W + (T * 16 + r) * nb01;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int pair = 4 * pg + 2 * (g & 1) + i;
                v |= (uint32_t) *(const uint16_t *) (row + (2 * pair + (g >> 1)) * 34) << (16 * i);
            }
        }
        P[q] = v;
    }
}

// y: [8][K] f32 -> actA [npairs][64 lanes][4] int8x4, actD [npairs][16 c'] f32. One WG (256 threads) per pair.
static __global__ void __launch_bounds__(256) q8wp_prep(const float * __restrict__ y, const int64_t s1, const int M,
                                                        int * __restrict__ actA, float * __restrict__ actD) {
    __shared__ int8_t sq[16][32];   // [c' = 8 p + col][i]
    const int pr  = blockIdx.x;
    const int tid = threadIdx.x;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int t   = tid + 256 * r;
        const int e   = t >> 5;
        const int i   = t & 31;
        const int p   = e >> 3;
        const int col = e & 7;
        const float x = col < M ? y[col * s1 + (int64_t) pr * 64 + 32 * p + i] : 0.0f;   // M < 8: zero columns
        float amax = fabsf(x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, o, 32));
        }
        const float d = amax / 127.0f;
        sq[e][i] = (int8_t) (amax == 0.0f ? 0 : (int) roundf(x / d));
        if (i == 0) {
            actD[(int64_t) pr * 16 + e] = d;
        }
    }
    __syncthreads();
    const int lane = tid >> 2, j = tid & 3;
    const int a = lane & 15, g = lane >> 4;
    uint32_t v = 0;
    if ((a >> 3) == (j >> 1)) {
        const int k0 = 16 * (j & 1) + 4 * g;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            v |= (uint32_t) (uint8_t) sq[a][k0 + b] << (8 * b);
        }
    }
    actA[(int64_t) pr * 256 + tid] = (int) v;
}

template <int YT>
static __global__ void __launch_bounds__(1024)
q8wp_mul_mat(const char * __restrict__ P, const int npairs, const int N, const uint32_t pbytes,
             const int * __restrict__ actA, const uint32_t abytes, const float * __restrict__ actD, const uint32_t dbytes,
             float * __restrict__ dst, const int64_t dst_s1, const int M) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    extern __shared__ float q8wp_red[];   // [KS][YT][4][64]
    constexpr int OOB = 0x7ffffff0;
    const int lane = threadIdx.x;
    const int kw   = threadIdx.y;
    const int KS   = blockDim.y;
    const int g    = lane >> 4;
    const int r16  = lane & 15;
    const bool mine0 = (g & 1) == 0;   // this lane's scale word holds pairs 0,1 of a group (else 2,3)
    const q8w8_int4 wrsrc = q8w8_make_rsrc(P, pbytes);
    const q8w8_int4 arsrc = q8w8_make_rsrc(actA, abytes);
    const q8w8_int4 drsrc = q8w8_make_rsrc(actD, dbytes);
    const int tileb = npairs * 1088;

    int tb[YT];
#pragma unroll
    for (int y = 0; y < YT; ++y) {
        const int tile = blockIdx.x * YT + y;
        tb[y] = tile * 16 < N ? tile * tileb : OOB;
    }
    const int ng = npairs / 4;
    const int g0 = (int) (((int64_t) kw * ng) / KS);
    const int g1 = (int) (((int64_t) (kw + 1) * ng) / KS);

    float acc[YT][4];
#pragma unroll
    for (int y = 0; y < YT; ++y) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            acc[y][v] = 0.0f;
        }
    }
    const int lo16 = lane * 16;
    const int sofs = npairs * 1024 + lane * 4;
    for (int pg = g0; pg < g1; ++pg) {
        int sw[YT], so[YT];
#pragma unroll
        for (int y = 0; y < YT; ++y) {
            sw[y] = q8w8_buf_x1(wrsrc, tb[y] == OOB ? OOB : tb[y] + sofs + pg * 256, 0, 2);
        }
#pragma unroll
        for (int y = 0; y < YT; ++y) {
            so[y] = __shfl_xor(sw[y], 16, 64);
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int pr = 4 * pg + t;
            q8w8_int4 wa[YT];
#pragma unroll
            for (int y = 0; y < YT; ++y) {
                wa[y] = q8w8_buf_x4(wrsrc, tb[y] == OOB ? OOB : tb[y] + pr * 1024 + lo16, 0, 2);
            }
            const q8w8_int4 aa = q8w8_buf_x4(arsrc, pr * 1024 + lo16, 0, 0);
            const q8w8_int4 dd = q8w8_buf_x4(drsrc, pr * 64 + 16 * g, 0, 0);
            const float da[4] = {__int_as_float(dd[0]), __int_as_float(dd[1]), __int_as_float(dd[2]), __int_as_float(dd[3])};
#pragma unroll
            for (int y = 0; y < YT; ++y) {
                const q8wm_v4i z = {0, 0, 0, 0};
                q8wm_v4i s = __builtin_amdgcn_mfma_i32_16x16x16i8(aa[0], wa[y][0], z, 0, 0, 0);
                s          = __builtin_amdgcn_mfma_i32_16x16x16i8(aa[1], wa[y][1], s, 0, 0, 0);
                s          = __builtin_amdgcn_mfma_i32_16x16x16i8(aa[2], wa[y][2], s, 0, 0, 0);
                s          = __builtin_amdgcn_mfma_i32_16x16x16i8(aa[3], wa[y][3], s, 0, 0, 0);
                const int  src = ((t >> 1) == 0) == mine0 ? sw[y] : so[y];
                const float dw = __half2float(__ushort_as_half((unsigned short) ((t & 1) ? ((uint32_t) src >> 16) : ((uint32_t) src & 0xffffu))));
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    acc[y][v] += (float) s[v] * (dw * da[v]);
                }
            }
        }
    }
#pragma unroll
    for (int y = 0; y < YT; ++y) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            acc[y][v] += __shfl_xor(acc[y][v], 32, 64);
        }
    }
    if (KS > 1) {
#pragma unroll
        for (int y = 0; y < YT; ++y) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                q8wp_red[((kw * YT + y) * 4 + v) * 64 + lane] = acc[y][v];
            }
        }
        __syncthreads();
        if (kw != 0) {
            return;
        }
        for (int k = 1; k < KS; ++k) {
#pragma unroll
            for (int y = 0; y < YT; ++y) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    acc[y][v] += q8wp_red[((k * YT + y) * 4 + v) * 64 + lane];
                }
            }
        }
    }
    if (g < 2) {
#pragma unroll
        for (int y = 0; y < YT; ++y) {
            const int row = (blockIdx.x * YT + y) * 16 + r16;
            if (row < N) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    if (4 * g + v < M) {
                        dst[(int64_t) (4 * g + v) * dst_s1 + row] = acc[y][v];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED(P); GGML_UNUSED(npairs); GGML_UNUSED(N); GGML_UNUSED(pbytes); GGML_UNUSED(actA);
    GGML_UNUSED(abytes); GGML_UNUSED(actD); GGML_UNUSED(dbytes); GGML_UNUSED(dst); GGML_UNUSED(dst_s1); GGML_UNUSED(M);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

// KS over whole 4-pair groups: best wave-slot fill over ncu x 16 resident waves, at least 2 groups per wave.
template <int YT>
static void q8wp_launch(const char * P, int nblk, int N, uint32_t pbytes, const int * actA, const float * actD,
                        int ncu, int ks_env, float * dst, int64_t dst_s1, int M, cudaStream_t stream) {
    const int npairs = nblk / 2;
    const int ng     = npairs / 4;
    const int tiles  = (N / 16 + YT - 1) / YT;
    const int slots  = ncu * 16;
    int ks = ks_env;
    if (ks <= 0) {
        double best = -1.0;
        ks = 1;
        for (int k = 1; k <= 16 && ng / k >= 2; ++k) {
            const int64_t w   = (int64_t) tiles * k;
            const double  eff = (double) w / (double) (((w + slots - 1) / slots) * slots);
            if (eff > best + 0.01) {
                best = eff;
                ks   = k;
            }
        }
    }
    ks = std::max(1, std::min(std::min(16, ng), ks));
    const dim3   block(64, ks, 1);
    const size_t shm = ks > 1 ? (size_t) ks * YT * 4 * 64 * sizeof(float) : 0;
    q8wp_mul_mat<YT><<<tiles, block, shm, stream>>>(P, npairs, N, pbytes, actA, (uint32_t) (npairs * 1024), actD,
                                                    (uint32_t) (npairs * 64), dst, dst_s1, M);
}

static std::mutex                                        q8wp_mu;
static std::map<const char *, std::pair<size_t, char *>> q8wp_cache;   // natural data -> (nbytes, packed copy)
static std::atomic<int>                                  q8wp_n{0};

void ggml_cuda_q8w8_invalidate(const void * base, size_t size) {
    if (q8wp_n.load() == 0 || base == nullptr) {
        return;
    }
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    std::lock_guard<std::mutex> lk(q8wp_mu);
    for (auto it = q8wp_cache.begin(); it != q8wp_cache.end(); ) {
        if (it->first < hi && it->first + it->second.first > lo) {
            CUDA_CHECK(cudaFree(it->second.second));   // hipFree synchronises the device first
            it = q8wp_cache.erase(it);
            q8wp_n.fetch_sub(1);
        } else {
            ++it;
        }
    }
}

static const char * q8wp_get_packed(const ggml_tensor * src0, cudaStream_t stream) {
    const char * W      = (const char *) src0->data;
    const size_t nbytes = ggml_nbytes(src0);
    std::lock_guard<std::mutex> lk(q8wp_mu);
    auto it = q8wp_cache.find(W);
    if (it != q8wp_cache.end() && it->second.first == nbytes) {
        return it->second.second;
    }
    char * P = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &P, nbytes));
    const int     nb01   = (int) src0->nb[1];
    const int     npairs = (int) (src0->ne[0] / 64);
    const int64_t nwords = (int64_t) nbytes / 4;
    const int     grid   = (int) std::min<int64_t>((nwords + 255) / 256, 65535);
    q8wp_repack<<<grid, 256, 0, stream>>>(W, nb01, npairs, nwords, (uint32_t *) P);
    CUDA_CHECK(cudaGetLastError());
    if (it != q8wp_cache.end()) {
        CUDA_CHECK(cudaFree(it->second.second));
        q8wp_cache.erase(it);
        q8wp_n.fetch_sub(1);
    }
    q8wp_cache.emplace(W, std::make_pair(nbytes, P));
    q8wp_n.fetch_add(1);
    static std::atomic<int64_t> n_packed{0};
    if (n_packed.fetch_add(1) == 0) {
        GGML_LOG_INFO("ggml_cuda_q8_w8: design 5 q8wp packed copy made (first weight: %lld B, N %lld, K %lld)\n",
                      (long long) nbytes, (long long) src0->ne[1], (long long) src0->ne[0]);
    }
    return P;
}

// ---------------------------------------------------------------------------------------------------------------------
// Q8-W8-INPLACE-E2E-1 (GGML_HIP_Q8W8=4 + GGML_HIP_Q8W8_INPLACE=1|2): the q8wp layout REPLACES the natural bytes of a weight
// once at upload, through the q8ip in-place mechanism (mmq-q8-mf.cu keeps the registry; a tensor is in exactly one format:
// natural, V5I stage, or q8wp). The q8wp layout is a permutation inside each 16-row tile (16 * nb01 B in both layouts),
// so the repack is done in place through a <= 16 MiB scratch, whole tiles at a time; no extra VRAM persists.
// Consumers of a q8wp-packed weight (safety argument: every one either reads the layout or gets the natural bytes):
//   MUL_MAT n 1..8            -> q8wp reads it directly (n < 8: activation columns >= n are zero, dst rows >= n not stored)
//   MUL_MAT n > 16 (prefill)  -> cuBLAS route: q8wp -> natural f16 convert in one pass (ggml_cuda_q8wp_convert_f16)
//   anything else (n 9..16, MMQ, MMVQ fusions, GET_ROWS, CPY, ...) -> transient natural copy (ggml_cuda_q8ip_substitute
//                                with ggml_cuda_q8wp_unpack), fusions over the weight are disabled (is_packed is true)
//   get_tensor / buffer cpy / partial set_tensor -> natural layout (q8ip get / unpack-before-write)
// ---------------------------------------------------------------------------------------------------------------------
static const char q8wpip_build_marker[] __attribute__((used)) =
    "ggml-cuda: Q8W8 design 5 q8wp IN-PLACE load-time repack (GGML_HIP_Q8W8_INPLACE)";

// packed word q -> natural: qs words are 4 contiguous natural bytes; scale words hold two f16 d (2 natural offsets)
static __device__ __forceinline__ void q8wp_word_src(const int64_t q, const int64_t tw, const int npairs, const int64_t nb01,
                                                     int64_t & o0, int64_t & o1, bool & is_scale) {
    const int64_t T = q / tw;
    const int     c = (int) (q % tw);
    if (c < npairs * 256) {
        const int pr = c >> 8, lane = (c >> 2) & 63, j = c & 3;
        const int r  = lane & 15, g = lane >> 4;
        o0 = (T * 16 + r) * nb01 + (2 * pr + (j >> 1)) * 34 + 2 + 16 * (j & 1) + 4 * g;
        o1 = 0;
        is_scale = false;
    } else {
        const int sc = c - npairs * 256;
        const int pg = sc >> 6, lane = sc & 63;
        const int r  = lane & 15, g = lane >> 4;
        const int64_t row = (T * 16 + r) * nb01;
        o0 = row + (2 * (4 * pg + 2 * (g & 1) + 0) + (g >> 1)) * 34;
        o1 = row + (2 * (4 * pg + 2 * (g & 1) + 1) + (g >> 1)) * 34;
        is_scale = true;
    }
}

static __global__ void q8wp_unpack(const uint32_t * __restrict__ P, const int nb01, const int npairs, const int64_t nwords,
                                   char * __restrict__ W) {
    const int64_t tw = (int64_t) nb01 * 4;
    for (int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; q < nwords; q += (int64_t) gridDim.x * blockDim.x) {
        int64_t o0, o1;
        bool    sc;
        q8wp_word_src(q, tw, npairs, nb01, o0, o1, sc);
        const uint32_t v = P[q];
        if (sc) {   // f16 d at the start of a 34 B block: 2 B aligned
            *(uint16_t *) (W + o0) = (uint16_t) (v & 0xffffu);
            *(uint16_t *) (W + o1) = (uint16_t) (v >> 16);
        } else {    // qs bytes at block + 2 + 4 k: 2 B aligned
            *(uint16_t *) (W + o0)     = (uint16_t) (v & 0xffffu);
            *(uint16_t *) (W + o0 + 2) = (uint16_t) (v >> 16);
        }
    }
}

// q8wp-packed Q8_0 [N][K] -> natural f16 [N][K]; one thread per 8 outputs (row, block b, quarter sub)
static __global__ void q8wp_to_f16(const char * __restrict__ P, half * __restrict__ y, const int64_t nthr, const int nblk) {
    const int64_t t = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nthr) {
        return;
    }
    const int     npairs = nblk / 2;
    const int64_t n   = t / 4;
    const int     sub = (int) (t % 4);
    const int64_t row = n / nblk;
    const int     b   = (int) (n - row * nblk);
    const int64_t T   = row >> 4;
    const int     r   = (int) (row & 15);
    const uint32_t * tile = (const uint32_t *) (P + T * (int64_t) npairs * 1088);
    const int pr = b >> 1, j = 2 * (b & 1) + (sub >> 1), g0 = 2 * (sub & 1);
    const uint32_t w0 = tile[pr * 256 + (16 * g0 + r) * 4 + j];
    const uint32_t w1 = tile[pr * 256 + (16 * (g0 + 1) + r) * 4 + j];
    const int pair = b >> 1, h = b & 1, tq = pair & 3;
    const int g  = 2 * h + (tq >> 1);
    const uint32_t sw = tile[npairs * 256 + (pair >> 2) * 64 + 16 * g + r];
    const half d = __ushort_as_half((unsigned short) ((tq & 1) ? (sw >> 16) : (sw & 0xffffu)));
    const uint32_t ws[2] = {w0, w1};
    half2 o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t w = ws[k >> 1];
        const int sh = 16 * (k & 1);
        o[k] = __hmul2(make_half2((int8_t) ((w >> sh) & 0xff), (int8_t) ((w >> (sh + 8)) & 0xff)), __half2half2(d));
    }
    *((uint4 *) (y + 8 * t)) = *((const uint4 *) o);
}

int ggml_cuda_q8wp_ip_mode() {
#if defined(GGML_USE_HIP)
    return q8w8_get_cfg().inplace;
#else
    return 0;
#endif
}

bool ggml_cuda_q8wp_ip_shape_ok(const ggml_tensor * t) {
    const q8w8_cfg & cfg = q8w8_get_cfg();
    const int64_t nbytes = (int64_t) ggml_nbytes(t);
    // GET_ROWS-only weights (token_embd*, DFlash2 selector_predecessor / selector_successor) stay natural
    return t->type == GGML_TYPE_Q8_0 && t->view_src == nullptr && t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t) &&
        t->ne[0] % 256 == 0 && t->ne[1] % 16 == 0 && ((uintptr_t) t->data) % 16 == 0 &&
        nbytes > cfg.min_bytes && nbytes < (int64_t) INT32_MAX - 4096 && strncmp(t->name, "token_embd", 10) != 0 &&
        strstr(t->name, "selector_successor") == nullptr && strstr(t->name, "selector_predecessor") == nullptr;
}

// out of place (src != dst), whole 16-row tiles; to_packed: natural src -> packed dst, else packed src -> natural dst
void ggml_cuda_q8wp_permute(const char * src, char * dst, int64_t N, int64_t K, bool to_packed, cudaStream_t stream) {
    const int     nb01   = (int) (K / 32 * 34);
    const int     npairs = (int) (K / 64);
    const int64_t nwords = N * nb01 / 4;
    const int     grid   = (int) std::min<int64_t>((nwords + 255) / 256, 65535);
    if (to_packed) {
        q8wp_repack<<<grid, 256, 0, stream>>>(src, nb01, npairs, nwords, (uint32_t *) dst);
    } else {
        q8wp_unpack<<<grid, 256, 0, stream>>>((const uint32_t *) src, nb01, npairs, nwords, dst);
    }
    CUDA_CHECK(cudaGetLastError());
}

// in place through a transient scratch (<= 16 MiB, whole tiles), synchronous on `stream`
void ggml_cuda_q8wp_permute_inplace(char * data, int64_t N, int64_t K, bool to_packed, cudaStream_t stream) {
    const int64_t tileb = 16 * (K / 32 * 34);
    const int64_t ntile = N / 16;
    const int64_t bt    = std::max<int64_t>(1, std::min<int64_t>(ntile, (16ll << 20) / tileb));
    char * scratch = nullptr;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMalloc((void **) &scratch, (size_t) (bt * tileb)));
    for (int64_t t0 = 0; t0 < ntile; t0 += bt) {
        const int64_t nt = std::min(bt, ntile - t0);
        char * p = data + t0 * tileb;
        CUDA_CHECK(cudaMemcpyAsync(scratch, p, (size_t) (nt * tileb), cudaMemcpyDeviceToDevice, stream));
        ggml_cuda_q8wp_permute(scratch, p, nt * 16, K, to_packed, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(scratch));
}

void ggml_cuda_q8wp_convert_f16(const void * P, half * y, int64_t N, int64_t K, cudaStream_t stream) {
    GGML_ASSERT(K % 256 == 0 && N % 16 == 0 && (uintptr_t) y % 16 == 0);
    const int64_t nthr = N * K / 8;
    q8wp_to_f16<<<(nthr + 255) / 256, 256, 0, stream>>>((const char *) P, y, nthr, (int) (K / 32));
    CUDA_CHECK(cudaGetLastError());
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        GGML_LOG_INFO("ggml_cuda_q8_w8: q8wp in-place packed q8_0->f16 prefill convert ran\n");
    }
}

// n 1..8 on an in-place q8wp weight (the base tensor itself, not a view)
bool ggml_cuda_q8_w8_packed_handles(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    return q8w8_get_cfg().inplace > 0 && GGML_CUDA_CC_IS_CDNA2(cc) && src0->view_src == nullptr &&
        ggml_cuda_q8ip_is_q8wp(src0) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= 1 && src1->ne[1] <= Q8W8_M && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

// KS (waves per WG splitting K): best wave-slot fill over ncu x 16 resident waves, at least 8 pairs per wave.
template <int YT>
static void q8wm_launch(const char * W, int nb01, int nblk, int N, uint32_t wbytes, const int * actA, const float * actD,
                        int ncu, int ks_env, float * dst, int64_t dst_s1, cudaStream_t stream) {
    const int npairs = nblk / 2;
    const int tiles  = (N + 16 * YT - 1) / (16 * YT);
    const int slots  = ncu * 16;
    int ks = ks_env;
    if (ks <= 0) {
        double best = -1.0;
        ks = 1;
        for (int k = 1; k <= 16 && npairs / k >= 8; ++k) {
            const int64_t w   = (int64_t) tiles * k;
            const double  eff = (double) w / (double) (((w + slots - 1) / slots) * slots);
            if (eff > best + 0.01) {
                best = eff;
                ks   = k;
            }
        }
    }
    ks = std::max(1, std::min(16, ks));
    const dim3   block(64, ks, 1);
    const size_t shm = ks > 1 ? (size_t) ks * YT * 4 * 64 * sizeof(float) : 0;
    q8wm_mul_mat<YT><<<tiles, block, shm, stream>>>(W, nb01, npairs, N, wbytes, actA, (uint32_t) (npairs * 1280), actD,
                                                    (uint32_t) (npairs * 64), dst, dst_s1);
}

template <int YT>
static void q8wo_launch(const char * W, int nb01, int nblk, int N, uint32_t wbytes, const int8_t * yq, const float * yd,
                        int ncu, int max_waves, float * dst, int64_t dst_s1, cudaStream_t stream) {
    constexpr int M = Q8W8_M;
    int P = (Q8W8_LDS_B / (36 * M)) / 32 * 32;   // 224 blocks at M 8 (as q8wv)
    P = std::min(P, (nblk + 31) / 32 * 32);
    GGML_ASSERT(P >= 32 && M * P * 36 <= Q8W8_LDS_B);
    const int n_tiles = (N + YT - 1) / YT;
    const int n_rnd   = (n_tiles + ncu * max_waves - 1) / (ncu * max_waves);
    const int wv      = (n_tiles + ncu * n_rnd - 1) / (ncu * n_rnd);
    const dim3 block(64, wv, 1);
    q8wo_mul_mat<YT><<<ncu, block, 0, stream>>>(W, nb01, nblk, N, wbytes, yq, yd, P, n_tiles, n_rnd, dst, dst_s1);
}

template <int DP>
static void q8w8_launch(const char * W, int nb01, int nblk, int N, uint32_t wbytes, const int8_t * yq, const float * yd,
                        int ncu, int max_waves, float * dst, int64_t dst_s1, cudaStream_t stream) {
    constexpr int M = Q8W8_M;
    int P = (Q8W8_LDS_B / (36 * M)) / Q8W8_STEP * Q8W8_STEP;   // 192 blocks
    P = std::min(P, (nblk + Q8W8_STEP - 1) / Q8W8_STEP * Q8W8_STEP);
    GGML_ASSERT(P >= Q8W8_STEP && M * P * 36 <= Q8W8_LDS_B);

    // mindiv: fewest rounds at max_waves, then the fewest waves per WG that still need only that many rounds
    const int n_tiles = (N + Q8W8_YT - 1) / Q8W8_YT;
    const int n_rnd   = (n_tiles + ncu * max_waves - 1) / (ncu * max_waves);
    const int wv      = (n_tiles + ncu * n_rnd - 1) / (ncu * n_rnd);
    const dim3 block(64, wv, 1);
    q8w8_mul_mat<DP><<<ncu, block, 0, stream>>>(W, nb01, nblk, N, wbytes, yq, yd, P, n_tiles, n_rnd, dst, dst_s1);
}

void ggml_cuda_mul_mat_q8_w8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8w8_cfg & cfg = q8w8_get_cfg();
    {
        // Path witness (rows grep it; also the build's strings content marker).
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            if (cfg.mode == 4) {
                GGML_LOG_INFO("ggml_cuda_q8_w8: gfx90a Q8W8 design 5 q8wp w8 Q8_0 int8 MFMA packed GEMV ran (call 1, YT %d)\n", cfg.myt);
            } else if (cfg.mode == 3) {
                GGML_LOG_INFO("ggml_cuda_q8_w8: gfx90a Q8W8 design 4 q8wm w8 Q8_0 int8 MFMA GEMV ran (call 1, YT %d)\n", cfg.myt);
            } else if (cfg.mode == 2) {
                GGML_LOG_INFO("ggml_cuda_q8_w8: gfx90a Q8W8 design 3 q8wo w8 Q8_0 GEMV ran (call 1, YT %d, 2-deep)\n", cfg.yt);
            } else {
                GGML_LOG_INFO("ggml_cuda_q8_w8: gfx90a Q8W8 w8 Q8_0 GEMV ran (call 1, full-block lanes, depth %d)\n", cfg.depth);
            }
        }
    }
    cudaStream_t stream = ctx.stream();

    const int     N      = (int) src0->ne[1];
    const int     nblk   = (int) (src0->ne[0] / 32);
    const int64_t s1     = src1->nb[1] / sizeof(float);
    const int64_t dst_s1 = dst->nb[1]  / sizeof(float);
    const bool    ip     = cfg.mode == 4 && cfg.inplace > 0 && ggml_cuda_q8ip_is_q8wp(src0);   // Q8-W8-INPLACE-E2E-1
    const int     M      = (int) src1->ne[1];
    GGML_ASSERT(ip ? (M >= 1 && M <= Q8W8_M && src0->view_src == nullptr) : M == Q8W8_M);

    if (cfg.mode == 4) {
        if (ip) {
            static std::atomic<int64_t> n_ip{0};
            const int64_t c = n_ip.fetch_add(1) + 1;
            if (c == 1 || (M != Q8W8_M && c < 4096 && (c & (c - 1)) == 0)) {
                GGML_LOG_INFO("ggml_cuda_q8_w8: design 5 q8wp ran on IN-PLACE packed weight (call %lld, n=%d, %s)\n",
                              (long long) c, M, src0->name);
            }
        }
        const int npairs = nblk / 2;
        ggml_cuda_pool_alloc<int>   actA(ctx.pool(), (size_t) npairs * 256);
        ggml_cuda_pool_alloc<float> actD(ctx.pool(), (size_t) npairs * 16);
        q8wp_prep<<<npairs, 256, 0, stream>>>((const float *) src1->data, s1, M, actA.get(), actD.get());
        const char *   P      = ip ? (const char *) src0->data : q8wp_get_packed(src0, stream);
        const uint32_t pbytes = (uint32_t) ggml_nbytes(src0);
        const int      ncu    = ggml_cuda_info().devices[ctx.device].nsm;
        float *        d      = (float *) dst->data;
        // YT 4 is not instantiated: it spills (128 VGPR + 48 B scratch per lane, static gate). MYT=4 runs YT 2.
        if (cfg.myt == 1) {
            q8wp_launch<1>(P, nblk, N, pbytes, actA.get(), actD.get(), ncu, cfg.ks, d, dst_s1, M, stream);
        } else {
            q8wp_launch<2>(P, nblk, N, pbytes, actA.get(), actD.get(), ncu, cfg.ks, d, dst_s1, M, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (cfg.mode == 3) {
        const int npairs = nblk / 2;
        ggml_cuda_pool_alloc<int>   actA(ctx.pool(), (size_t) npairs * 320);
        ggml_cuda_pool_alloc<float> actD(ctx.pool(), (size_t) npairs * 16);
        q8wm_prep<<<npairs, 256, 0, stream>>>((const float *) src1->data, s1, actA.get(), actD.get());
        const char *   W      = (const char *) src0->data;
        const int      nb01   = (int) src0->nb[1];
        const uint32_t wbytes = (uint32_t) ggml_nbytes(src0);
        const int      ncu    = ggml_cuda_info().devices[ctx.device].nsm;
        float *        d      = (float *) dst->data;
        if (cfg.myt == 4) {
            q8wm_launch<4>(W, nb01, nblk, N, wbytes, actA.get(), actD.get(), ncu, cfg.ks, d, dst_s1, stream);
        } else if (cfg.myt == 1) {
            q8wm_launch<1>(W, nb01, nblk, N, wbytes, actA.get(), actD.get(), ncu, cfg.ks, d, dst_s1, stream);
        } else {
            q8wm_launch<2>(W, nb01, nblk, N, wbytes, actA.get(), actD.get(), ncu, cfg.ks, d, dst_s1, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), (size_t) Q8W8_M * nblk * 32);
    ggml_cuda_pool_alloc<float>  yd(ctx.pool(), (size_t) Q8W8_M * nblk);
    {
        const dim3 grid((nblk + 7) / 8, Q8W8_M, 1);
        q8w8_quantize<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, yq.get(), yd.get());
    }
    const char *   W      = (const char *) src0->data;
    const int      nb01   = (int) src0->nb[1];
    const uint32_t wbytes = (uint32_t) ggml_nbytes(src0);
    const int      ncu    = ggml_cuda_info().devices[ctx.device].nsm;
    float *        d      = (float *) dst->data;
    if (cfg.mode == 2) {
        if (cfg.yt == 4) {
            q8wo_launch<4>(W, nb01, nblk, N, wbytes, yq.get(), yd.get(), ncu, cfg.waves, d, dst_s1, stream);
        } else {
            q8wo_launch<2>(W, nb01, nblk, N, wbytes, yq.get(), yd.get(), ncu, cfg.waves, d, dst_s1, stream);
        }
    } else if (cfg.depth == 2) {
        q8w8_launch<2>(W, nb01, nblk, N, wbytes, yq.get(), yd.get(), ncu, cfg.waves, d, dst_s1, stream);
    } else {
        q8w8_launch<3>(W, nb01, nblk, N, wbytes, yq.get(), yd.get(), ncu, cfg.waves, d, dst_s1, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
