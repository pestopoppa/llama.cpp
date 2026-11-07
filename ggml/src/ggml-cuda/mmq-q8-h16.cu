// AK-ASM-H16 (kernel-champion-and-promotion.md C8-2-H16-M36): MFMA f16 skinny Q8_0 MUL_MAT for gfx90a at the real
// DFlash2 verify widths. Production 27B DFlash2 np4 verify = 4 slots x 9 tokens = M 36 (np2: 18, np1: 9); at C7r every
// M >= 12 on a large Q8_0 weight takes MMQ (+ stream-k fixups), measured at ~31% of MI210 HBM (memo 2026-11-07 Q4).
// C7r coverage below this path: wvSplitK (mmq-q8-wvsk.cu) M 5..11 on weights > 6 MiB, skinny split-K (mmq-q8-skinny.cu)
// M 5..16 on weights <= 6 MiB. This path: M 12..36 on weights > 6 MiB by default (dispatch order: wvsk, skinny, h16).
//
// Design note (2026-11-07)
//  Bytes-bound target: stream every weight byte from HBM once for all M columns (1.6 TB/s peak, ~1.35 TB/s practical).
//  * Math: v_mfma_f32_16x16x16f16, srcA = 16 weight rows x 16 k (lane l: row l%16, k group l/16), srcB = 16 k x 16
//    activation columns (lane l: column l%16, same k group), D[4*(l/16)+v][l%16] (ggml mma.cuh AMD layout, as validated
//    by the H1 skinny kernel). NT = ceil(M/16) column tiles (M 36 -> 3, 16 of 48 columns padding). int8 MFMA is the
//    f16 rate on gfx90a (program.md pitfalls), so f16 costs nothing in math and keeps f32 accumulation across blocks.
//  * K permutation: the reduction is order-free, so A and B only have to agree. Per 64-k step, k group g of a lane takes
//    the 16 contiguous qs bytes of half (g%2) of Q8_0 block (2*step + g/2); MFMA substep s uses bytes 4s..4s+3 of them.
//    B lane (column c, group g) then needs activations k = 64*step + 16g + 4s + 0..3: one contiguous 32 B f16 chunk.
//  * Weight loads (wvsk recipe): raw buffer loads with the tensor as the V# range (rows >= N and past-the-end windows read
//    0), an aligned 20 B window per lane (dwordx4 + dword, slc nontemporal) funnel-shifted by 0/2 bytes, plus the 16-bit
//    block scale. One wave instruction moves 16 rows x 68 contiguous bytes. No LDS, no barriers.
//  * Dequant in registers, 7 VALU per 4 weights: x ^ 0x80808080, 2 x v_perm_b32 into f16 (1024 + u) pairs, v_pk_add_f16
//    -1152 (exact small integers), v_pk_mul_f16 by the block scale (one rounding: q * d in f16).
//  * Activations: converted once per call f32 -> f16 [M][K] (same precision model as the cuBLAS f16 route ggml already
//    uses for dequant + GEMM), read from L2 as srcB fragments; the RT row tiles of a wave reuse each fragment, and the 4
//    waves of a workgroup walk the same k range on adjacent rows (L1 reuse).
//  * Pipelining: per 128-k iteration a wave issues this iteration's B loads, then (sched_barrier) the NEXT iteration's
//    weight windows, then waits only for the former (in-order vmcnt) and computes: one iteration of weights is always in
//    flight behind the math. 4.3 KB (RT 2) / 8.7 KB (RT 4) per wave per iteration.
//  * Parallelism: a wave owns 16*RT rows over a k slice; split-K (grid.y) only to reach ~2 waves per SIMD, capped so
//    the f32 partials (KS x M x N x 4 B, written + re-read by a deterministic reduce kernel) stay <= 15% of the weight
//    bytes. KS = 1 writes dst directly (no reduce launch). Deterministic: no atomics.
//  Bandwidth / roofline model (MI210: 104 CU, 416 SIMD, 1.7 GHz, 256 f16 MFMA flop/clk/SIMD = 181 TFLOP/s):
//   * MFMA flop per weight = 32 * NT -> at 1 TB/s: M <= 16 16% of MFMA peak, M 17..32 32%, M 33..36 48%; the math
//     overlaps the next iteration's weight stream. VALU dequant ~1.75 op/weight -> ~25% of VALU at 1 TB/s.
//   * B (activation) L2 reads per weight byte: M*2/(16*RT*1.0625) = 2.1 (RT 2) / 1.06 (RT 4) at M 36, before L1 reuse.
//   * expected 0.75-1.0 TB/s (47-62% of 1.6 TB/s peak) on the large 27B shapes, vs ~0.5 TB/s for C7r MMQ at M 36:
//     1.3-1.8x op-level; pessimistic (L2-bound B reads or exposed B latency at ~1 wave/SIMD) ~0.6 TB/s = 1.15x.
//  Env: GGML_HIP_Q8_H16=0 disables (C7r route); GGML_HIP_Q8_H16_MIN / _MAX (M range, default 12..36, cap 48);
//       GGML_HIP_Q8_H16_MIN_BYTES (default 6 MiB: only strictly larger weights); GGML_HIP_Q8_H16_RT (2 | 4 row tiles per
//       wave, default 2); GGML_HIP_Q8_H16_KSPLIT (0 = auto); GGML_HIP_Q8_H16_TW (auto split-K target waves, default
//       8 x CUs); GGML_HIP_Q8_H16_PART_PCT (partials cap, % of weight bytes, default 15).
#include "mmq-q8-h16.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>

#define Q8H16_MAXM   48
#define Q8H16_WAVES  4      // waves per workgroup (adjacent row groups, same k slice)
#define Q8H16_IT_K   128    // k per pipeline iteration = 2 steps of 64 = 4 Q8_0 blocks

static int q8h16_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

struct q8h16_cfg {
    bool enabled; int mmin; int mmax; int rt; int ksplit; int tw; int part_pct; int64_t min_bytes;
};

static const q8h16_cfg & q8h16_get_cfg() {
    static const q8h16_cfg cfg = [] {
        q8h16_cfg c;
        c.enabled  = q8h16_env_int("GGML_HIP_Q8_H16", 1) != 0;
        c.mmin     = std::max(2, q8h16_env_int("GGML_HIP_Q8_H16_MIN", 12));
        c.mmax     = std::min(Q8H16_MAXM, q8h16_env_int("GGML_HIP_Q8_H16_MAX", 36));
        c.rt       = q8h16_env_int("GGML_HIP_Q8_H16_RT", 2) == 4 ? 4 : 2;
        c.ksplit   = std::max(0, std::min(16, q8h16_env_int("GGML_HIP_Q8_H16_KSPLIT", 0)));
        c.tw       = std::max(0, q8h16_env_int("GGML_HIP_Q8_H16_TW", 0));
        c.part_pct = std::max(0, q8h16_env_int("GGML_HIP_Q8_H16_PART_PCT", 15));
        const char * mb = getenv("GGML_HIP_Q8_H16_MIN_BYTES");
        c.min_bytes = (mb && *mb) ? std::max(0ll, atoll(mb)) : (6ll << 20);
        return c;
    }();
    return cfg;
}

bool ggml_cuda_q8_h16_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    const q8h16_cfg & cfg = q8h16_get_cfg();
    if (!cfg.enabled || !GGML_CUDA_CC_IS_CDNA2(cc)) {
        return false;
    }
    const int64_t nbytes = (int64_t) ggml_nbytes(src0);
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= cfg.mmin && src1->ne[1] <= cfg.mmax &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src0->ne[0] % (2 * Q8H16_IT_K) == 0 && src0->nb[1] % 4 == 0 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        nbytes > cfg.min_bytes && nbytes < (int64_t) INT32_MAX - 4096;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

// y: [M][K] f32 (column stride s1 floats) -> a: [M][K] f16. 4 elements per thread (K % 256 == 0).
static __global__ void q8h16_act_f16(const float * __restrict__ y, const int64_t s1, const int K, half * __restrict__ a) {
    const int c = blockIdx.y;
    const int k = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= K) {
        return;
    }
    const float * yc = y + c * s1 + k;
    half2 * ac = (half2 *) (a + (int64_t) c * K + k);
    ac[0] = make_half2(__float2half(yc[0]), __float2half(yc[1]));
    ac[1] = make_half2(__float2half(yc[2]), __float2half(yc[3]));
}

// dst[c][row] = sum over ks splits of part[split][c][row] (fixed order: deterministic)
static __global__ void q8h16_reduce(const float * __restrict__ part, const int ks, const int64_t split_stride, const int N,
                                    float * __restrict__ dst, const int64_t dst_s1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int c   = blockIdx.y;
    if (row >= N) {
        return;
    }
    const float * p = part + (int64_t) c * N + row;
    float s = 0.0f;
    for (int y = 0; y < ks; ++y) {
        s += p[y * split_stride];
    }
    dst[(int64_t) c * dst_s1 + row] = s;
}

#if defined(GGML_USE_HIP)
typedef int      q8h16_i32x4 __attribute__((ext_vector_type(4)));
typedef float    q8h16_f32x4 __attribute__((ext_vector_type(4)));
typedef _Float16 q8h16_f16x4 __attribute__((ext_vector_type(4)));
typedef _Float16 q8h16_f16x2 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ q8h16_i32x4 q8h16_make_rsrc(const void * base, uint32_t num_bytes) {
    union { q8h16_i32x4 v; struct { const void * p; uint32_t range; uint32_t cfg; } s; } u;
    u.s.p     = base;
    u.s.range = num_bytes;
    u.s.cfg   = 0x00020000u;   // gfx9 raw buffer dword3, as mmq-q8-wvsk.cu / mmvq.cu q8pf_make_rsrc / CK
    return u.v;
}

// aux: bit0 glc, bit1 slc (slc = nontemporal streaming)
__device__ q8h16_i32x4 q8h16_buf_x4(q8h16_i32x4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ int         q8h16_buf_x1(q8h16_i32x4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ short       q8h16_buf_s16(q8h16_i32x4 rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i16");

struct q8h16_wfrag {
    q8h16_i32x4 a;   // aligned 16 B window
    int         b;   // next dword (funnel-shift tail)
    uint32_t    s;   // f16 block scale (zero-extended: a packed short pair would need a v_bfi, i.e. a vmcnt wait)
};

// weight windows of iteration `it` (blocks 4*it .. 4*it+3) for the RT row tiles of this lane
template <int RT>
static __device__ __forceinline__ void q8h16_load_w(q8h16_wfrag (&w)[RT][2], const q8h16_i32x4 rsrc, const int (&rb)[RT],
                                                    const int it, const int g, const bool valid) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int bb = 4 * it + 2 * u + (g >> 1);
        const int tb = bb * 34 + 2 + 16 * (g & 1);   // first qs byte of this lane's half block (row-relative)
#pragma unroll
        for (int j = 0; j < RT; ++j) {
            const bool ok = valid && rb[j] >= 0;   // !valid: every window out of range -> no memory traffic
            const int  vo = ok ? rb[j] + (tb & ~3) : 0x7ffffff0;   // out of V# range -> reads 0
            const int  vs = ok ? rb[j] + bb * 34   : 0x7ffffff0;
            w[j][u].a = q8h16_buf_x4(rsrc, vo,      0, 2);
            w[j][u].b = q8h16_buf_x1(rsrc, vo + 16, 0, 2);
            w[j][u].s = (uint32_t) (unsigned short) q8h16_buf_s16(rsrc, vs, 0, 0);
        }
    }
}

// 16 Q8_0 weights of one half block -> 4 MFMA srcA fragments (4 f16 each), scaled by the block scale
static __device__ __forceinline__ void q8h16_dequant(const q8h16_wfrag & w, const uint32_t sh, q8h16_f16x4 (&a)[4]) {
    const uint32_t d5[5] = {(uint32_t) w.a[0], (uint32_t) w.a[1], (uint32_t) w.a[2], (uint32_t) w.a[3], (uint32_t) w.b};
    const _Float16 dh = __builtin_bit_cast(_Float16, (unsigned short) (w.s & 0xffffu));
    const q8h16_f16x2 d2   = {dh, dh};
    const q8h16_f16x2 bias = {(_Float16) 1152.0f, (_Float16) 1152.0f};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint32_t q  = __builtin_amdgcn_alignbyte(d5[s + 1], d5[s], sh);
        const uint32_t x  = q ^ 0x80808080u;                         // int8 -> u = x + 128
        const uint32_t lo = __builtin_amdgcn_perm(0x64646464u, x, 0x04010400u);   // f16 pair (1024 + u0, 1024 + u1)
        const uint32_t hi = __builtin_amdgcn_perm(0x64646464u, x, 0x04030402u);   // f16 pair (1024 + u2, 1024 + u3)
        const q8h16_f16x2 l = (__builtin_bit_cast(q8h16_f16x2, lo) - bias) * d2;
        const q8h16_f16x2 h = (__builtin_bit_cast(q8h16_f16x2, hi) - bias) * d2;
        a[s] = (q8h16_f16x4) {l[0], l[1], h[0], h[1]};
    }
}
#endif // defined(GGML_USE_HIP)

#if defined(GGML_USE_HIP)
// One 128-k iteration: compute on `cur` (issued one iteration earlier) while `nxt` (iteration it+1) is in flight.
template <int NT, int RT>
static __device__ __forceinline__ void q8h16_step(q8h16_f32x4 (&acc)[RT][NT], const q8h16_wfrag (&cur)[RT][2],
                                                  q8h16_wfrag (&nxt)[RT][2], const uint4 * const (&bp)[NT],
                                                  const q8h16_i32x4 rsrc, const int (&rb)[RT], const int it, const int g,
                                                  const bool nxt_valid) {
    // 1) this iteration's activation fragments (L2/L1 hits), issued BEFORE the next weight windows so the in-order
    //    vmcnt wait below drains only them (and this iteration's weights) while the next weights stay in flight
    uint4 bq[NT][2][2];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            bq[t][u][0] = bp[t][16 * it + 8 * u + 0];
            bq[t][u][1] = bp[t][16 * it + 8 * u + 1];
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    // 2) next iteration's weight windows. Unconditional: past the last iteration the windows are out of range and read 0
    //    without touching memory; a conditional load would make the vmcnt count path-dependent (compiler -> vmcnt(0)).
    q8h16_load_w<RT>(nxt, rsrc, rb, it + 1, g, nxt_valid);
    __builtin_amdgcn_sched_barrier(0);
    // 3) dequant + MFMA
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const uint32_t sh = (uint32_t) ((((4 * it + 2 * u + (g >> 1)) * 34) + 2 + 16 * (g & 1)) & 3);
#pragma unroll
        for (int j = 0; j < RT; ++j) {
            q8h16_f16x4 a[4];
            q8h16_dequant(cur[j][u], sh, a);
#pragma unroll
            for (int s = 0; s < 4; ++s) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const uint4 & q = bq[t][u][s >> 1];
                    const uint2 b2 = (s & 1) ? make_uint2(q.z, q.w) : make_uint2(q.x, q.y);
                    const q8h16_f16x4 b = __builtin_bit_cast(q8h16_f16x4, b2);
                    acc[j][t] = __builtin_amdgcn_mfma_f32_16x16x16f16(a[s], b, acc[j][t], 0, 0, 0);
                }
            }
        }
    }
}
#endif // defined(GGML_USE_HIP)

// grid = (ceil(groups / WAVES), ks), block = (64, WAVES). Wave = one group of 16*RT rows x the k slice of blockIdx.y.
// out: ks == 1 -> dst (column stride out_s1); ks > 1 -> partials [ks][M][N] (out_s1 = N, split_stride = M*N).
template <int NT, int RT>
static __global__ void __launch_bounds__(64 * Q8H16_WAVES)
q8h16_mul_mat(const char * __restrict__ W, const int nb01, const int N, const uint32_t wbytes,
              const half * __restrict__ act, const int K, const int M, const int nit, const int ks,
              float * __restrict__ out, const int64_t out_s1, const int64_t split_stride) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    const int lane = threadIdx.x;
    const int r    = lane & 15;
    const int g    = lane >> 4;
    const int grp  = blockIdx.x * blockDim.y + threadIdx.y;
    const int row0 = grp * 16 * RT;
    if (row0 >= N) {
        return;   // whole wave; the kernel has no barriers
    }
    const int split = blockIdx.y;
    const int np    = nit / 2;   // iteration pairs (K % 256 == 0); every split gets >= 1 pair (ks <= np)
    const int it0   = 2 * (int) ((int64_t) split * np / ks);
    const int it1   = 2 * (int) ((int64_t) (split + 1) * np / ks);

    const q8h16_i32x4 rsrc = q8h16_make_rsrc(W, wbytes);

    int rb[RT];
#pragma unroll
    for (int j = 0; j < RT; ++j) {
        const int row = row0 + 16 * j + r;
        rb[j] = row < N ? row * nb01 : -1;
    }
    // srcB: column c = 16t + r (clamped to a valid column: D columns >= M are never stored, and MFMA columns are
    // independent, so their content does not matter), 32 B chunk at k = 128*it + 64*u + 16*g
    const uint4 * bp[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int c = min(16 * t + r, M - 1);
        bp[t] = (const uint4 *) (act + (int64_t) c * K + 16 * g);
    }

    q8h16_f32x4 acc[RT][NT];
#pragma unroll
    for (int j = 0; j < RT; ++j) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[j][t] = (q8h16_f32x4) {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    q8h16_wfrag cur[RT][2];
    q8h16_load_w<RT>(cur, rsrc, rb, it0, g, true);

    // ping-pong buffers, two iterations per trip (even iteration count per split): no nxt -> cur register copy, which
    // would force a vmcnt(0) on the prefetch at the end of every iteration (seen in the ISA of the first draft)
    q8h16_wfrag alt[RT][2];
    for (int it = it0; it < it1; it += 2) {
        q8h16_step<NT, RT>(acc, cur, alt, bp, rsrc, rb, it,     g, true);
        q8h16_step<NT, RT>(acc, alt, cur, bp, rsrc, rb, it + 1, g, it + 2 < it1);
    }

    // D[4g + v][r]: weight row row0 + 16j + 4g + v, column 16t + r
    float * o = out + (int64_t) split * split_stride;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int c = 16 * t + r;
        if (c < M) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int row = row0 + 16 * j + 4 * g + v;
                    if (row < N) {
                        o[(int64_t) c * out_s1 + row] = acc[j][t][v];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(N); GGML_UNUSED(wbytes); GGML_UNUSED(act); GGML_UNUSED(K);
    GGML_UNUSED(M); GGML_UNUSED(nit); GGML_UNUSED(ks); GGML_UNUSED(out); GGML_UNUSED(out_s1); GGML_UNUSED(split_stride);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

template <int NT, int RT>
static void q8h16_launch(const char * W, int nb01, int N, uint32_t wbytes, const half * act, int K, int M, int nit, int ks,
                         float * out, int64_t out_s1, int64_t split_stride, cudaStream_t stream) {
    const int groups = (N + 16 * RT - 1) / (16 * RT);
    const dim3 grid((groups + Q8H16_WAVES - 1) / Q8H16_WAVES, ks, 1);
    const dim3 block(64, Q8H16_WAVES, 1);
    q8h16_mul_mat<NT, RT><<<grid, block, 0, stream>>>(W, nb01, N, wbytes, act, K, M, nit, ks, out, out_s1, split_stride);
}

void ggml_cuda_mul_mat_q8_h16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8h16_cfg & cfg = q8h16_get_cfg();
    cudaStream_t stream = ctx.stream();

    const int     N      = (int) src0->ne[1];
    const int     K      = (int) src0->ne[0];
    const int     M      = (int) src1->ne[1];
    const int     nb01   = (int) src0->nb[1];
    const int     nit    = K / Q8H16_IT_K;
    const int64_t s1     = src1->nb[1] / sizeof(float);
    const int64_t dst_s1 = dst->nb[1]  / sizeof(float);
    const int     ncu    = ggml_cuda_info().devices[ctx.device].nsm;
    GGML_ASSERT(M >= 1 && M <= Q8H16_MAXM && K % (2 * Q8H16_IT_K) == 0);

    // split-K: enough waves for ~2 per SIMD, at least 2 iterations (256 k) per split, partials <= part_pct of weight bytes
    const int rt     = cfg.rt;
    const int groups = (N + 16 * rt - 1) / (16 * rt);
    int ks = cfg.ksplit;
    if (ks == 0) {
        const int tw = cfg.tw > 0 ? cfg.tw : 8 * ncu;
        ks = (tw + groups - 1) / groups;
        const int part_cap = (int) (((int64_t) cfg.part_pct * nb01) / (100ll * 8 * M));   // KS*M*N*8 B <= pct * N*nb01
        ks = std::min(ks, std::max(1, part_cap));
    }
    ks = std::max(1, std::min({ks, 16, std::max(1, nit / 2)}));   // <= iteration pairs: every split has >= 256 k

    {
        // Route witness at WARN (llama-server hides ggml INFO; program.md pitfalls), once per distinct M; also the
        // build's strings content marker.
        static std::atomic<uint64_t> seen{0};
        const uint64_t bit = 1ull << (M & 63);
        if ((seen.fetch_or(bit) & bit) == 0) {
            GGML_LOG_WARN("ggml_cuda_q8_h16: gfx90a MFMA f16 skinny Q8_0 path ran (M=%d N=%d K=%d rt=%d ksplit=%d)\n",
                          M, N, K, rt, ks);
        }
    }

    ggml_cuda_pool_alloc<half> act(ctx.pool(), (size_t) M * K);
    {
        const dim3 grid((K / 4 + 255) / 256, M, 1);
        q8h16_act_f16<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, K, act.get());
    }

    const char *   W      = (const char *) src0->data;
    const uint32_t wbytes = (uint32_t) ggml_nbytes(src0);
    float *        d      = (float *) dst->data;
    const int      nt     = (M + 15) / 16;

    ggml_cuda_pool_alloc<float> part(ctx.pool());
    float * out          = d;
    int64_t out_s1       = dst_s1;
    int64_t split_stride = 0;
    if (ks > 1) {
        out          = part.alloc((size_t) ks * M * N);
        out_s1       = N;
        split_stride = (int64_t) M * N;
    }

#define Q8H16_CASE(NT_, RT_) \
    if (nt == NT_ && rt == RT_) { q8h16_launch<NT_, RT_>(W, nb01, N, wbytes, act.get(), K, M, nit, ks, out, out_s1, split_stride, stream); } else
    Q8H16_CASE(1, 2) Q8H16_CASE(2, 2) Q8H16_CASE(3, 2)
    Q8H16_CASE(1, 4) Q8H16_CASE(2, 4) Q8H16_CASE(3, 4)
    { GGML_ABORT("q8_h16: unsupported nt=%d rt=%d", nt, rt); }
#undef Q8H16_CASE
    CUDA_CHECK(cudaGetLastError());

    if (ks > 1) {
        const dim3 grid((N + 255) / 256, M, 1);
        q8h16_reduce<<<grid, 256, 0, stream>>>(out, ks, split_stride, N, d, dst_s1);
        CUDA_CHECK(cudaGetLastError());
    }
}
