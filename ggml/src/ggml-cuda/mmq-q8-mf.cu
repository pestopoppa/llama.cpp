// DTREE-0-Q8MF (speculative-decoding.md): gfx90a small-M Q8_0 MUL_MAT on v_mfma_i32_16x16x16i8 for DFlash / tree verify
// (M = 9..16 verify rows, including the LM head at M = W), LARGE weights only (> 6 MiB by default). OPT-IN via
// GGML_HIP_Q8_MF=1; routed in ggml-cuda.cu BEFORE the wvsk branch, so with the knob off nothing changes.
//
// Why: mmq-q8-wvsk.cu streams each weight byte once at M <= 8 (1.6-2.0x C6), but its per-(row, column) v_dot4 math and
// LDS activation pages scale with M, so at M 12..16 it loses its lead (gate r1/r2; 27B verify W12 51.1 ms with
// GGML_HIP_Q8_WVSK_MAX=16 vs 61 ms default). The MFMA does 16 rows x 16 columns x 16 k per instruction, so the math cost
// is the same for any M <= 16 and the kernel stays a pure weight stream.
//
// Design (2026-11-20):
//  * grid = #CUs, ONE persistent workgroup per CU (48 KB static LDS), block = (64, WV), WV <= 16 waves. A workgroup
//    walks 16-row weight tiles t = gi, gi + G, ...; for each tile EVERY wave streams its own K-slice of the same 16 rows
//    (K split across the workgroup's waves), so each weight byte is loaded exactly once, by exactly one lane.
//  * activations live in VGPRs: wave w owns blocks [kc0 + w*KB, +KB) of its K chunk and loads that slice of the <= 16
//    q8_1-quantized activation columns ONCE per kernel (2 VGPR per block per lane: KB = 10..12 -> 20-24 VGPR at
//    K = 5120). No LDS restage of activations, no per-tile activation traffic at all.
//  * MFMA roles: srcA = activations (16 columns x 16 k), srcB = weights (16 k x 16 rows), D[col][row]. With this
//    orientation lane l owns weight row j = l%16 in BOTH the B operand and the D tile (D lane l reg r = D[4*(l/16)+r][l%16]),
//    so each lane needs only the fp16 scale of the row it loaded itself; the activation scales of its 4 columns come
//    from one ds_read_b128 of an LDS table [block][16] f32 (<= 16 KB, filled once per kernel).
//  * k permutation inside a Q8_0 block: lane group g = l/16 holds qs bytes 8g..8g+7 of the block for both operands;
//    MFMA #1 consumes bytes 8g..8g+3, MFMA #2 bytes 8g+4..8g+7, accumulating into one i32 tile (same block -> same
//    scales). The dot product is permutation invariant as long as A and B use the same permutation (they do).
//  * epilogue per block: acc_f32[r] += (float) i32[r] * (d_w * d_a[4g + r]) - i32 -> f32 once per block (Q8_0 scales
//    change every 32 k, so the i32 tile cannot run longer than one block).
//  * weight loads: Q8_0 qs are only 2-byte aligned, so each lane loads an aligned 12 B window (buffer_load_dwordx2 +
//    buffer_load_dword, slc = nontemporal as AITER loadnt / wvsk) and funnel-shifts it with v_alignbyte (0 or 2 bytes,
//    wave-uniform: it depends on the block parity only), plus a 16-bit load of the block scale (same cache line). Raw
//    buffer loads with the tensor as the V# range; the lane offset is row*nb01 + 8g (VGPR), the block offset is
//    wave-uniform (soffset), rows >= N are clamped to N-1 (dropped at the store) and blocks past the chunk to its last block
//    (zero activations and scales there).
//  * software pipeline: a tile's K-slice is processed in CPT (even) chunks of U blocks with two register buffers; the
//    loads of chunk c+1 are issued before chunk c's math, and the last chunk of a tile issues the NEXT tile's first
//    chunk, so the loads of tile t+1 are in flight across tile t's reduction barrier.
//  * reduction: once per 16x16 tile, each wave writes its 4 f32 per lane to LDS (double-buffered slot -> one LDS-only
//    barrier per tile, which does not drain vmcnt), and 256 threads sum the WV partials in wave order (deterministic).
//  * odd M / M < 16: the MFMA is 16 columns wide regardless; quantize writes zero columns >= M and only columns < M are
//    stored, so there is no per-M instantiation and no odd-M special case (the box's "pad to a multiple of 4" is
//    subsumed: the pad is to 16, for free).
//  * K chunks: a wave holds at most KB = 16 blocks (32 VGPR). K > 16*16 blocks (K > 8192, e.g. ffn_down K = 17408 -> C = 3),
//    or a poorly balanced tile count, splits K into C chunks; WGs are statically assigned to chunks (G = #CU / C each),
//    partials go to a [C][M][N] f32 scratch and a tiny kernel sums them in chunk order (deterministic). C is picked to
//    minimise rounds * blocks-per-chunk (the per-CU stream time, +10% per extra chunk) with >= 64 blocks per chunk.
//  Bandwidth estimate (MI210: 104 CU, 1.7 GHz, HBM 1.64 TB/s peak, ~1.35 TB/s practical):
//   * bytes in flight per CU: WV (14-16) waves x 64 lanes x U (4-6) blocks x 8.5 B = 30-50 KB, double-buffered; Little's
//     law at 1.35 TB/s x ~2 us needs ~26 KB/CU.
//   * math per block per wave (16 rows x 34 B = 544 B): 2 MFMA (2 x 32 clk) + ~14 VALU (56 clk) + 1 ds_read_b128 ->
//     ~120 clk/SIMD -> ~3 TB/s compute cap per GPU, > 2x HBM.
//   * per-tile overhead: 1 KB LDS write per wave + one barrier + 256-thread reduce; at K = 5120 a tile is 87 KB of weight.
//   * target: 27B verify W12 ~44 ms (vs 51.1 ms wvsk16 / 61 ms default) = ~1.2 TB/s on the GEMM bytes.
//  Env: GGML_HIP_Q8_MF=1 enables (default 0), GGML_HIP_Q8_MF_MIN / _MAX (M range, default 9..16),
//       GGML_HIP_Q8_MF_MIN_BYTES (default 6 MiB: only weights strictly larger), GGML_HIP_Q8_MF_MAXC (max K chunks, default 4).
#include "mmq-q8-mf.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>
#include <type_traits>
#include <utility>

#define Q8MF_MAXM    16
#define Q8MF_MAXWAVE 16
#define Q8MF_KBMAX   16   // KB 20/24 spilled (ISA: 10/17 VGPR spills, scratch 20/40 B) under the 128-VGPR cap
#define Q8MF_MAXB    (Q8MF_MAXWAVE * Q8MF_KBMAX)   // blocks per K chunk
#define Q8MF_MINB    64                            // fewest blocks per chunk when splitting K for balance

static int q8mf_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

struct q8mf_cfg {
    bool enabled; int mmin; int mmax; int maxc; int64_t min_bytes;
};

static const q8mf_cfg & q8mf_get_cfg() {
    static const q8mf_cfg cfg = [] {
        q8mf_cfg c;
        c.enabled = q8mf_env_int("GGML_HIP_Q8_MF", 0) != 0;
        c.mmin    = std::max(1, q8mf_env_int("GGML_HIP_Q8_MF_MIN", 9));
        c.mmax    = std::min(Q8MF_MAXM, q8mf_env_int("GGML_HIP_Q8_MF_MAX", 16));
        c.maxc    = std::max(1, q8mf_env_int("GGML_HIP_Q8_MF_MAXC", 4));
        const char * mb = getenv("GGML_HIP_Q8_MF_MIN_BYTES");
        c.min_bytes = (mb && *mb) ? std::max(0ll, atoll(mb)) : (6ll << 20);
        return c;
    }();
    return cfg;
}

bool ggml_cuda_q8_mf_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    const q8mf_cfg & cfg = q8mf_get_cfg();
    if (!cfg.enabled || !GGML_CUDA_CC_IS_CDNA2(cc)) {
        return false;
    }
    const int64_t nbytes = (int64_t) ggml_nbytes(src0);
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= cfg.mmin && src1->ne[1] <= cfg.mmax &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src0->nb[1] % 4 == 0 &&
        src0->ne[0] / 32 <= 8 * Q8MF_MAXB &&   // K <= 65536: K chunks never outnumber the CUs
        src0->ne[0] % 64 == 0 &&   // even nblk: the last block of a row is odd -> its 12 B window never straddles the V# end
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        nbytes > cfg.min_bytes && nbytes < (int64_t) INT32_MAX - 4096;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

// y: [M][K] f32 (row stride s1 floats) -> yq: [nblk][4 g][16 col][8 B] int8 (qs bytes 8g..8g+7 of the block), yd: [nblk][16]
// f32. Columns >= M are zero. One 32-thread group per (block, column).
static __global__ void q8mf_quantize(const float * __restrict__ y, const int64_t s1, const int nblk, const int M,
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
    const float d = amax / 127.0f;
    const int   q = amax == 0.0f ? 0 : (int) roundf(x / d);
    yq[(((int64_t) blk * 4 + i / 8) * 16 + c) * 8 + (i % 8)] = (int8_t) q;
    if (i == 0) {
        yd[(int64_t) blk * 16 + c] = d;
    }
}

// sum the C chunk partials [C][M][N] into dst (chunk order fixed -> deterministic)
static __global__ void q8mf_reduce_chunks(const float * __restrict__ part, const int C, const int M, const int N,
                                          float * __restrict__ dst, const int64_t dst_s1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y;
    if (row >= N) {
        return;
    }
    float s = 0.0f;
    for (int c = 0; c < C; ++c) {
        s += part[((int64_t) c * M + col) * N + row];
    }
    dst[(int64_t) col * dst_s1 + row] = s;
}

#if defined(GGML_USE_HIP)
typedef int q8mf_v4i __attribute__((ext_vector_type(4)));
typedef int q8mf_v2i __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ q8mf_v4i q8mf_make_rsrc(const void * base, uint32_t num_bytes) {
    union { q8mf_v4i v; struct { const void * p; uint32_t range; uint32_t cfg; } s; } u;
    u.s.p     = base;
    u.s.range = num_bytes;
    u.s.cfg   = 0x00020000u;   // gfx9 raw buffer dword3, as mmq-q8-wvsk.cu / mmvq.cu / CK
    return u.v;
}

// aux: bit0 glc, bit1 slc (slc = nontemporal streaming)
__device__ q8mf_v2i q8mf_buf_x2(q8mf_v4i rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");
__device__ int      q8mf_buf_x1(q8mf_v4i rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ short    q8mf_buf_s16(q8mf_v4i rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i16");

// LDS-only WG barrier: __syncthreads() would also drain vmcnt, i.e. wait for the next tile's weight loads issued above
#define Q8MF_LDS_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

// issue the weight loads of blocks kw0 + I0 .. kw0 + I0 + U - 1 of one tile into (wa, wb, ws). vrow = row * nb01 (the only
// per-lane offset; rows >= N are clamped to N-1 by the caller and their results discarded at the store), g8 = 8 * g. The
// block offset is wave-uniform and goes in soffset (SGPR), so no per-block VGPR stays live across the tile loop. Blocks
// past the chunk are clamped to kc1-1 (a real block with a finite scale): their activations and activation scales are 0,
// so they contribute exactly 0.
template <int U, int I0>
static __device__ __forceinline__ void q8mf_issue(q8mf_v2i (&wa)[U], int (&wb)[U], short (&ws)[U], const q8mf_v4i rsrc,
                                                  const int vrow, const int g8, const int kw0, const int kc1) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int b  = min(kw0 + I0 + u, kc1 - 1);   // uniform
        const int so = (b * 34 + 2) & ~3;            // aligned start of the block's qs window (8g is 0 mod 4)
        wa[u] = q8mf_buf_x2(rsrc, vrow + g8, so,     2);
        wb[u] = q8mf_buf_x1(rsrc, vrow + g8, so + 8, 2);
        ws[u] = q8mf_buf_s16(rsrc, vrow,     b * 34, 0);
    }
}

template <int U, int I0, int KB>
static __device__ __forceinline__ void q8mf_compute(const q8mf_v2i (&wa)[U], const int (&wb)[U], const short (&ws)[U],
                                                    const q8mf_v2i (&act)[KB], const float4 * __restrict__ lds_da4,
                                                    const int kw0, const int kc0, const int g, float (&acc)[4]) {
    // block b = kw0 + i: qs offset b*34 + 2 + 8g = 2 (mod 4) for even b, 0 for odd b (row base and 8g are 0 mod 4)
    const uint32_t sh_even = (kw0 & 1) ? 0u : 2u;
    const uint32_t sh_odd  = 2u - sh_even;
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int      i  = I0 + u;
        const uint32_t sh = (i & 1) ? sh_odd : sh_even;
        const int w0 = (int) __builtin_amdgcn_alignbyte((uint32_t) wa[u][1], (uint32_t) wa[u][0], sh);
        const int w1 = (int) __builtin_amdgcn_alignbyte((uint32_t) wb[u],    (uint32_t) wa[u][1], sh);
        const q8mf_v4i z = {0, 0, 0, 0};
        q8mf_v4i s = __builtin_amdgcn_mfma_i32_16x16x16i8(act[i][0], w0, z, 0, 0, 0);
        s          = __builtin_amdgcn_mfma_i32_16x16x16i8(act[i][1], w1, s, 0, 0, 0);
        const float4 da = lds_da4[(kw0 - kc0 + i) * 4 + g];
        const float  dw = __half2float(__ushort_as_half((unsigned short) ws[u]));
        acc[0] += (float) s[0] * (dw * da.x);
        acc[1] += (float) s[1] * (dw * da.y);
        acc[2] += (float) s[2] * (dw * da.z);
        acc[3] += (float) s[3] * (dw * da.w);
    }
}

// compile-time unrolled loop: f(std::integral_constant<int, 0>), ..., f(std::integral_constant<int, n-1>)
template <typename F, int... Cs>
static __device__ __forceinline__ void q8mf_static_for(F && f, std::integer_sequence<int, Cs...>) {
    (f(std::integral_constant<int, Cs>{}), ...);
}
#endif // defined(GGML_USE_HIP)

// grid = G * C (WG -> chunk = blockIdx.x / G), block = (64, WV). Every wave of a WG walks the same tile sequence, so the
// per-tile LDS barrier is uniform.
template <int KB, int U>
static __global__ void __launch_bounds__(64 * Q8MF_MAXWAVE)
q8mf_mul_mat(const char * __restrict__ W, const int nb01, const int N, const uint32_t wbytes,
             const int8_t * __restrict__ yq, const float * __restrict__ yd, const int nblk, const int Bc, const int G,
             const int n_tiles, const int m_real, float * __restrict__ out, const int64_t out_s1, const int64_t out_cs) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    constexpr int CPT = KB / U;
    static_assert(CPT * U == KB && CPT % 2 == 0, "q8mf: KB must split into an even number of U-block chunks");
    __shared__ float4 lds_da4[Q8MF_MAXB * 4];
    __shared__ float4 lds_part[2][Q8MF_MAXWAVE][64];

    const int lane  = threadIdx.x;
    const int w     = __builtin_amdgcn_readfirstlane(threadIdx.y);   // wave-uniform -> SGPR math for every block offset
    const int wv    = blockDim.y;
    const int tid   = w * 64 + lane;
    const int nthr  = 64 * wv;
    const int chunk = blockIdx.x / G;
    const int gi    = blockIdx.x - chunk * G;
    const int kc0   = chunk * Bc;
    const int kc1   = min(kc0 + Bc, nblk);
    const int kw0   = kc0 + w * KB;     // this wave's first block
    const int col   = lane & 15;        // srcA row (activation column) / srcB + D column (weight row)
    const int g     = lane >> 4;        // k group

    // activation scales of this chunk -> LDS [block - kc0][16], zero past the chunk (blocks a wave holds beyond kc1)
    {
        float * lds_da = (float *) lds_da4;
        const int nreal = (kc1 - kc0) * 16;
        for (int i = tid; i < wv * KB * 16; i += nthr) {
            lds_da[i] = i < nreal ? yd[(int64_t) kc0 * 16 + i] : 0.0f;
        }
    }
    // this wave's K-slice of the activations -> VGPRs, once per kernel
    q8mf_v2i act[KB];
    {
        const q8mf_v2i * yq2 = (const q8mf_v2i *) yq;
#pragma unroll
        for (int i = 0; i < KB; ++i) {
            const int b = kw0 + i;
            act[i] = b < kc1 ? yq2[((int64_t) b * 4 + g) * 16 + col] : q8mf_v2i{0, 0};
        }
    }
    __syncthreads();

    const q8mf_v4i rsrc = q8mf_make_rsrc(W, wbytes);
    q8mf_v2i wa0[U], wa1[U];
    int      wb0[U], wb1[U];
    short    ws0[U], ws1[U];

    const int g8 = 8 * g;
    // row base of tile t for this lane (rows past N clamped: valid loads, results dropped at the store)
    auto vrow_of = [&](const int t) { return min(t * 16 + col, N - 1) * nb01; };
    q8mf_issue<U, 0>(wa0, wb0, ws0, rsrc, vrow_of(gi), g8, kw0, kc1);
    int slot = 0;
    for (int t = gi; t < n_tiles; t += G) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int vrow  = vrow_of(t);
        const int vnext = vrow_of(t + G);
        // chunk pairs (c, c+1) with compile-time indices: c in buffer 0, c+1 in buffer 1; each buffer is refilled right
        // after its math, the last refill being the next tile's chunk 0 (in flight across the reduction barrier)
        q8mf_static_for([&](auto cc) {
            constexpr int c = 2 * decltype(cc)::value;
            q8mf_issue<U, (c + 1) * U>(wa1, wb1, ws1, rsrc, vrow, g8, kw0, kc1);
            q8mf_compute<U, c * U, KB>(wa0, wb0, ws0, act, lds_da4, kw0, kc0, g, acc);
            if constexpr (c + 2 < CPT) {
                q8mf_issue<U, (c + 2) * U>(wa0, wb0, ws0, rsrc, vrow, g8, kw0, kc1);
            } else {
                q8mf_issue<U, 0>(wa0, wb0, ws0, rsrc, vnext, g8, kw0, kc1);
            }
            q8mf_compute<U, (c + 1) * U, KB>(wa1, wb1, ws1, act, lds_da4, kw0, kc0, g, acc);
        }, std::make_integer_sequence<int, CPT / 2>{});

        // reduce the WV K-slice partials of this 16x16 tile once (double-buffered slot -> one LDS barrier per tile)
        lds_part[slot][w][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        Q8MF_LDS_BARRIER();
        for (int e = tid; e < 256; e += nthr) {
            const int oc = e >> 4;                    // output column (activation column)
            const int jr = e & 15;                    // row within the tile
            const int sl = 16 * (oc >> 2) + jr;       // D lane holding (oc, jr): lane l reg r = D[4*(l/16)+r][l%16]
            const int r  = oc & 3;
            float s = 0.0f;
            for (int ww = 0; ww < wv; ++ww) {
                s += ((const float *) &lds_part[slot][ww][sl])[r];
            }
            const int row = t * 16 + jr;
            if (oc < m_real && row < N) {
                out[(int64_t) chunk * out_cs + (int64_t) oc * out_s1 + row] = s;
            }
        }
        slot ^= 1;
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(N); GGML_UNUSED(wbytes); GGML_UNUSED(yq); GGML_UNUSED(yd);
    GGML_UNUSED(nblk); GGML_UNUSED(Bc); GGML_UNUSED(G); GGML_UNUSED(n_tiles); GGML_UNUSED(m_real); GGML_UNUSED(out);
    GGML_UNUSED(out_s1); GGML_UNUSED(out_cs);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

struct q8mf_plan {
    int C; int Bc; int G; int KB; int wv;
};

// K chunking + WG split: fewest chunks that fit KB <= Q8MF_KBMAX, then up to cfg.maxc chunks if that lowers the per-CU stream
// time (rounds * blocks per chunk), never below Q8MF_MINB blocks per chunk.
static q8mf_plan q8mf_make_plan(const int nblk, const int n_tiles, const int ncu, const int maxc) {
    const int cmin = (nblk + Q8MF_MAXB - 1) / Q8MF_MAXB;
    q8mf_plan best = {0, 0, 0, 0, 0};
    int64_t best_cost = INT64_MAX;
    for (int c = cmin; c <= std::max(cmin, maxc); ++c) {
        const int bc = (nblk + c - 1) / c;
        if (c > cmin && bc < Q8MF_MINB) {
            break;
        }
        const int cc = (nblk + bc - 1) / bc;   // chunks actually needed at this chunk size
        const int g  = std::min(ncu / cc, n_tiles);
        if (g < 1) {
            break;
        }
        // +10% per extra chunk: an extra chunk costs the partial round trip + reduce launch and fewer waves (bytes in
        // flight) per WG, so it must buy a clearly better round balance
        const int64_t cost = (int64_t) ((n_tiles + g - 1) / g) * bc * (10 + (c - cmin)) / 10;
        if (cost < best_cost) {
            best_cost = cost;
            best.C = cc; best.Bc = bc; best.G = g;
        }
    }
    GGML_ASSERT(best.C >= 1 && "q8_mf: no K chunking fits");
    static const int kbs[] = {4, 8, 12, 16};
    for (int kb : kbs) {
        if (16 * kb >= best.Bc) {
            best.KB = kb;
            break;
        }
    }
    GGML_ASSERT(best.KB > 0);
    best.wv = (best.Bc + best.KB - 1) / best.KB;
    return best;
}

void ggml_cuda_mul_mat_q8_mf(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8mf_cfg & cfg = q8mf_get_cfg();
    {
        // Path witness (gate packets grep it; also a content marker of the build).
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8_mf: gfx90a MFMA i8 small-M Q8_0 path ran (call 1, GGML_HIP_Q8_MF, k split across waves, act in vgpr)\n");
        }
    }
    cudaStream_t stream = ctx.stream();

    const int     N       = (int) src0->ne[1];
    const int     M       = (int) src1->ne[1];
    const int     nblk    = (int) (src0->ne[0] / 32);
    const int64_t s1      = src1->nb[1] / sizeof(float);
    const int64_t dst_s1  = dst->nb[1]  / sizeof(float);
    const int     n_tiles = (N + 15) / 16;
    const int     ncu     = ggml_cuda_info().devices[ctx.device].nsm;
    GGML_ASSERT(M >= 1 && M <= Q8MF_MAXM);

    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), (size_t) nblk * 16 * 32);
    ggml_cuda_pool_alloc<float>  yd(ctx.pool(), (size_t) nblk * 16);
    {
        const dim3 grid((nblk + 7) / 8, 16, 1);
        q8mf_quantize<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, M, yq.get(), yd.get());
    }

    const q8mf_plan p = q8mf_make_plan(nblk, n_tiles, ncu, cfg.maxc);
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    float * out    = (float *) dst->data;
    int64_t out_s1 = dst_s1;
    int64_t out_cs = 0;
    if (p.C > 1) {
        part.alloc((size_t) p.C * M * N);
        out    = part.get();
        out_s1 = N;
        out_cs = (int64_t) M * N;
    }

    const char *   W      = (const char *) src0->data;
    const int      nb01   = (int) src0->nb[1];
    const uint32_t wbytes = (uint32_t) ggml_nbytes(src0);
    const dim3     grid(p.G * p.C, 1, 1);
    const dim3     block(64, p.wv, 1);
    switch (p.KB) {
#define Q8MF_CASE(kb, u) case kb: q8mf_mul_mat<kb, u><<<grid, block, 0, stream>>>(W, nb01, N, wbytes, yq.get(), yd.get(), \
                                      nblk, p.Bc, p.G, n_tiles, M, out, out_s1, out_cs); break;
        Q8MF_CASE(4, 2) Q8MF_CASE(8, 4) Q8MF_CASE(12, 6) Q8MF_CASE(16, 4)
#undef Q8MF_CASE
        default: GGML_ABORT("q8_mf: unsupported KB=%d", p.KB);
    }
    if (p.C > 1) {
        const dim3 rgrid((N + 255) / 256, M, 1);
        q8mf_reduce_chunks<<<rgrid, 256, 0, stream>>>(part.get(), p.C, M, N, (float *) dst->data, dst_s1);
    }
    CUDA_CHECK(cudaGetLastError());
}
