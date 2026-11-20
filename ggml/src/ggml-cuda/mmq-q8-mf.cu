// DTREE-0-Q8MF-V2 (speculative-decoding.md): roofline redesign of the gfx90a small-M Q8_0 MUL_MAT for DFlash / tree
// verify (M = 9..16 verify rows, including the LM head at M = W), LARGE weights only (> 6 MiB by default). OPT-IN:
//   GGML_HIP_Q8_MF=1  int8: q8_1-style activations (int8 + per-block scale, quantised once per call) x int8 weights on
//                     v_mfma_i32_16x16x16i8, per-block i32 -> f32 scale epilogue
//   GGML_HIP_Q8_MF=2  f16:  f16 activations (converted once per call) x Q8_0 weights dequantised IN REGISTERS to f16
//                     (scale folded in) on v_mfma_f32_32x32x8f16, f32 accumulation over all of K, no epilogue
// Routed in ggml-cuda.cu BEFORE the wvsk branch, CDNA2 only, so with the knob off nothing changes.
//
// Roofline (27B Q8_0, MI210: 104 CU, 1.7 GHz, ~1.6 TB/s HBM, 181 TOPS int8 / 181 TFLOPS f16 MFMA): at M <= 16 the verify
// GEMM is a weight stream: ~28 GB/pass = ~17.5 ms at any M, i.e. ~15.4 GB/s = ~9 B/clk per CU. Target t16 ~ t1.
//
// Why v1 (2c9381805) falls short of the HBM rate (offline accounting of its ISA, isa-stats.json of v1):
//  * per Q8_0 block and lane it issued 3 loads (buffer_load_dwordx2 + buffer_load_dword + buffer_load_ushort) for 8.5
//    useful bytes, with lane = weight row: one wave instruction touches 16 rows x 32-34 B, i.e. 16-32 distinct 64 B
//    lines, each line partially, and the SAME line is requested again by the next instruction (dword after dwordx2,
//    ushort scale) and by the next block (a 64 B line holds ~1.9 blocks). ~48-80 line requests per 544 B of weights
//    = 7-11 B per TA/TCP tag lookup, against a need of ~9 B/clk/CU; with slc (stream) loads the L1 does not keep the
//    line, so the re-requests go to L2: ~3-5x L2 request amplification -> L2-request bound well below HBM rate.
//  * 16 waves x 16 rows = 256 concurrent 34 B-stride row streams per CU (poor DRAM page locality).
//  * a full-WG LDS barrier per 16-row tile synchronises 16 waves every ~87 KB of weight.
//  * K > 8192 (ffn_down) needed C chunks + a second (reduce) launch; at N = 5120 G = 104/C WGs per chunk with an uneven
//    tile count -> idle CUs in the tail.
//
// V2 design (both variants share the memory path; only the LDS -> MFMA fragment code differs):
//  * STAGE = 64 weight rows x 8 Q8_0 blocks (256 k) = 64 x 272 B = 17408 B. 272 = 17 x 16, and the row stride is a
//    multiple of 16 when K % 256 == 0, so every row segment of a stage is 17 ALIGNED dwordx4 chunks: the 256 threads load
//    the stage as 1088 fully used 16 B chunks (4.25 buffer_load_dwordx4 per thread, slc), each 64 B line requested
//    exactly once by 4 adjacent lanes. ~1 KB per wave instruction vs ~180 B in v1.
//  * the stage is written raw to LDS (ds_write_b128) and re-read in MFMA fragment layout (Q8_0 qs are only 2-byte aligned:
//    aligned dword reads + v_alignbyte, parity of the block is compile-time) -> LDS traffic ~3x the weight bytes, well
//    under the 128 B/clk/CU LDS rate (the HBM share is ~9 B/clk/CU).
//  * activations are quantised (MF=1: int8 + f32 scale per (block, column), q8_1 style) / converted (MF=2: f16) ONCE per
//    call into a stage-major buffer (L2 resident, <= 16 x K x 2 B) and staged into LDS together with the weights: one
//    activation stage (4.5 / 8 KB) feeds all 64 rows of the stage -> activation L2 traffic 26 % / 47 % of the weight
//    bytes, all 16 MFMA columns fed from one K-tile weight load.
//  * double-buffered LDS stage + TWO more stages in flight in VGPRs (prefetch distance 2 stages, ~35 KB weight bytes in
//    flight per CU, Little: 1.6 TB/s x ~2 us / 104 CU ~ 31 KB), ONE LDS-only barrier per stage (does not drain vmcnt).
//  * stream-K: the flattened (64-row tile, stage) space of T = ceil(N/64) * K/256 stages is cut into G = #CU contiguous
//    ranges of L = ceil(T/G) stages, one persistent WG (4 waves) per CU, so every CU streams the same byte count at any
//    N (N = 5120: 80 tiles x 20 stages / 104 = 15.4 stages each; LM head: 746 each). Tiles fully inside a range are
//    stored directly; the <= 2 boundary tiles per WG write a 16 x 64 f32 partial and bump a per-tile counter; the LAST
//    arriving WG sums the partials in WG order (deterministic, independent of arrival) and stores. No second launch,
//    counters are zeroed by the activation pre-pass (same stream, before the GEMM).
//  * MF=1 fragments: wave w owns rows 16w..16w+15 of the tile and all 8 blocks of the stage. MFMA srcA = activations
//    (16 columns x 16 k), srcB = weights (16 k x 16 rows), D[col][row]: lane l holds weight row l%16 in B and D, and
//    qs bytes 8g..8g+7 (g = l/16) of each block; 2 MFMA per block into a fresh i32 tile, then acc += i32 * (d_w * d_a).
//  * MF=2 fragments: wave w owns rows 32*(w&1).. +31 and blocks 4*(w>>1).. +3 of every stage (the two K halves are summed
//    through LDS once per tile flush). MFMA srcA = activations (32 "rows" = columns, 16..31 zero), srcB = weights; lane l
//    holds weight row l%32 and qs bytes 16h..16h+15 (h = l/32) of each block, dequantised with the magic-number trick
//    (x ^ 0x80 | 0x6400 as f16 = 1152 + x, v_perm_b32) then (h - 1152) * d_w in packed f16; 4 MFMA per block.
//    At M <= 16 half of each 32x32x8 MFMA is padding: its per-byte MFMA cost is 2x MF=1's (still ~1.9x the HBM share).
//  * odd M / M < 16: zero activation columns, only columns < M are stored. Rows >= N clamp to N-1 at load, dropped at store.
//  Env: GGML_HIP_Q8_MF=1|2 (V2) or 3|4 (V3, below) (default 0 = off), GGML_HIP_Q8_MF_MIN / _MAX (M range, default 9..16),
//       GGML_HIP_Q8_MF_MIN_BYTES (default 6 MiB: only weights strictly larger), GGML_HIP_Q8_MF_WG (WG count, default #CU
//       for V2, 2 x #CU for V3), GGML_HIP_Q8_MF_SLC (V3 only: 1 = slc/nontemporal weight loads, default 0).
//
// DTREE-0-Q8MF-V3 (speculative-decoding.md): V3 memory path. Gate V2 (ak-dtree0-q8mf2-gate-gpu-20261120) was flat in M but
// its op rate topped out at 0.75-0.95 TB/s (head 248320x5120 n1: 949 GB/s = 9.1 GB/s/CU, 1.9 us per 17 KB stage).
//  Why (Little's law, offline): V2 keeps 1 WG of 4 waves per CU (LDS-bound) and 2 register stages; the commit of stage u+1
//  waits until it lands, so the bytes in flight per CU oscillate between ~1 stage (17 KB, just before a commit returns)
//  and 2 stages (35 KB, just after the next issue): ~26 KB average. 9.1 GB/s/CU x L = 26 KB -> loaded latency L ~ 2.8 us,
//  i.e. V2 is latency-bound. 1.25 TB/s needs 12 GB/s/CU x ~3 us (latency rises with load) ~ 36 KB in flight AT ALL TIMES
//  per CU; wvsk (1.15-1.24 TB/s at small M) keeps ~70 KB/CU in flight (16 waves x 4 rows x 1088 B).
//  Second V2 cost: every boundary (stream-K partial) tile ran __threadfence + __syncthreads mid-stream, which drains vmcnt
//  (gfx9 has one in-order vmcnt for loads and stores): a full pipeline drain (~1 latency) per WG in the middle of ~16-19
//  stages (~5-8 %). Third: partition L = ceil(T/G) idles CUs (N=5120, K=6144: 19 x 102 WGs, 2 CUs idle).
//  V3 (GGML_HIP_Q8_MF=3: int8 v_mfma_i32_16x16x16i8, same fragments as MF=1; =4: f16 v_mfma_f32_16x16x16f16 with the
//  in-register magic-number dequant, f32 accumulation over all K - 16x16x16 instead of MF=2's 32x32x8 because at M <= 16
//  half of a 32-wide MFMA is padding: half the MFMA cycles per byte, no cross-wave K-half reduction, no LDS scratch):
//  * TWO 4-wave WGs per CU (__launch_bounds__(256, 2): <= 256 VGPR+AGPR per wave; LDS per WG = ONE stage buffer, 23 / 26
//    KB, so 2 x <= 64 KB): the two WGs run their barriers independently, one computes / syncs while the other streams.
//  * THREE register stages per WG (prefetch distance 3): at the commit of stage u the WG still has u+1, u+2 fully in
//    flight -> >= 2 stages x 17.4 KB x 2 WGs = 70 KB in flight per CU at the minimum, up to 104 KB (2-3x Little's 36 KB,
//    the wvsk operating point). Per wave 21 x dwordx4 outstanding (vmcnt cap 63). Single LDS buffer: 2 LDS-only barriers
//    per stage (after the commit; after the compute), neither waits on vmcnt.
//  * stream-K with a balanced split: WG g streams stages [floor(gT/G), floor((g+1)T/G)) (+-1 stage between WGs, owner of
//    stage u = floor(((u+1)G - 1)/T)); G = 2 x #CU (all 208 WG slots, each CU gets 2).
//  * boundary tiles: the first partial tile's accumulator is HELD in 4 VGPRs and written together with the last one AFTER
//    the stream (the only fence/syncthreads of the kernel run with nothing useful in flight) -> no mid-stream drain.
//  * rows >= N and stages past the WG's range: full offset in voffset (soffset 0), so the raw-buffer range check returns 0
//    with no memory traffic (no row clamp multiply, no clamped re-reads of the last stage at the tail).
//  * activation chunks whose columns are all >= M are pointed out of range too: activation L2 traffic scales with M
//    (MF=3 26 % / MF=4 47 % of the weight bytes at M = 16, 1/16 of that at M = 1), which is what routes n 1..8 here.
//  * weight loads default to NOT slc: a 272 B row segment of a stage shares its first / last 128 B L2 line with the
//    neighbouring k-stage, read one stage (~3 us) later; with slc that line may be evicted and refetched (up to ~23-47 %
//    HBM overfetch, offline line accounting); GGML_HIP_Q8_MF_SLC=1 restores slc for an A/B.
//  MFMA load per CU per 2 stages (one per WG): 2 x 4 waves x 16 MFMA x 32 clk on 4 SIMDs = 1024 clk per 34.8 KB -> cap
//  ~58 GB/s/CU (5x the HBM share); LDS ~3 B per weight byte (cap ~40 GB/s/CU at 128 B/clk).
#include "mmq-q8-mf.cuh"

#include <atomic>
#include <climits>
#include <cstdlib>

#define Q8MF_MAXM    16
#define Q8MF_RT      64                          // weight rows per tile
#define Q8MF_KS      8                           // Q8_0 blocks per stage (256 k)
#define Q8MF_ROWB    (Q8MF_KS * 34)              // 272 B: one row of a stage = 17 aligned dwordx4
#define Q8MF_WSTAGE  (Q8MF_RT * Q8MF_ROWB)       // 17408 B of weights per stage
#define Q8MF_NCHUNK  (Q8MF_WSTAGE / 16)          // 1088 dwordx4 chunks
#define Q8MF_NT      256                         // threads per WG (4 waves)
#define Q8MF_NLD     5                           // ceil(1088 / 256) weight loads per thread per stage
#define Q8MF_A1      (Q8MF_KS * 512 + Q8MF_KS * 64)   // MF=1 activation stage: [8][4 g][16 col][8 B] int8 + [8][16] f32
#define Q8MF_A2      (Q8MF_KS * 1024)                 // MF=2 activation stage: [8][2 h][16 col][16] f16

static int q8mf_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

struct q8mf_cfg {
    int variant; int mmin; int mmax; int wg; int slc; int64_t min_bytes;
};

static const q8mf_cfg & q8mf_get_cfg() {
    static const q8mf_cfg cfg = [] {
        q8mf_cfg c;
        c.variant = q8mf_env_int("GGML_HIP_Q8_MF", 0);
        if (c.variant < 1 || c.variant > 4) {
            c.variant = 0;
        }
        c.slc     = q8mf_env_int("GGML_HIP_Q8_MF_SLC", 0) != 0 ? 1 : 0;
        c.mmin    = std::max(1, q8mf_env_int("GGML_HIP_Q8_MF_MIN", 9));
        c.mmax    = std::min(Q8MF_MAXM, q8mf_env_int("GGML_HIP_Q8_MF_MAX", 16));
        c.wg      = std::max(0, q8mf_env_int("GGML_HIP_Q8_MF_WG", 0));
        const char * mb = getenv("GGML_HIP_Q8_MF_MIN_BYTES");
        c.min_bytes = (mb && *mb) ? std::max(0ll, atoll(mb)) : (6ll << 20);
        return c;
    }();
    return cfg;
}

bool ggml_cuda_q8_mf_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    const q8mf_cfg & cfg = q8mf_get_cfg();
    if (cfg.variant == 0 || !GGML_CUDA_CC_IS_CDNA2(cc)) {
        return false;
    }
    const int64_t nbytes = (int64_t) ggml_nbytes(src0);
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= cfg.mmin && src1->ne[1] <= cfg.mmax &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) &&
        src0->ne[0] % 256 == 0 &&                         // whole stages; row stride 34 * K/32 is then a multiple of 16
        ((uintptr_t) src0->data) % 16 == 0 &&             // dwordx4 weight chunks
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        nbytes > cfg.min_bytes && nbytes < (int64_t) INT32_MAX - 4096;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

// zero the per-tile stream-K counters (grid-stride over the whole pre-pass grid)
static __device__ __forceinline__ void q8mf_zero_counters(int * __restrict__ cnt, const int ncnt) {
    const int nthr = gridDim.x * gridDim.y * blockDim.x;
    for (int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; i < ncnt; i += nthr) {
        cnt[i] = 0;
    }
}

// MF=1 pre-pass. y: [M][K] f32 (row stride s1 floats) -> stage-major act: stage s at s * Q8MF_A1:
//   [8 blk][4 g][16 col][8 B] int8 (qs bytes 8g..8g+7 of the block) then [8 blk][16 col] f32 scale. Columns >= M are zero.
// One 32-thread group per (block, column).
static __global__ void q8mf_pre_i8(const float * __restrict__ y, const int64_t s1, const int nblk, const int M,
                                   char * __restrict__ act, int * __restrict__ cnt, const int ncnt) {
    q8mf_zero_counters(cnt, ncnt);
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
    char * st = act + (int64_t) (blk / Q8MF_KS) * Q8MF_A1;
    const int b = blk % Q8MF_KS;
    st[b * 512 + ((i / 8) * 16 + c) * 8 + (i % 8)] = (char) q;
    if (i == 0) {
        ((float *) (st + Q8MF_KS * 512))[b * 16 + c] = d;
    }
}

// MF=2 pre-pass. y -> act [nblk][2 h][16 col][16 k] f16 (k = 32 blk + 16 h + e). Columns >= M are zero.
static __global__ void q8mf_pre_f16(const float * __restrict__ y, const int64_t s1, const int nblk, const int M,
                                    char * __restrict__ act, int * __restrict__ cnt, const int ncnt) {
    q8mf_zero_counters(cnt, ncnt);
    const int c   = blockIdx.y;
    const int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int i   = threadIdx.x % 32;
    if (blk >= nblk) {
        return;
    }
    const float x = c < M ? y[c * s1 + (int64_t) blk * 32 + i] : 0.0f;
    ((__half *) act)[(((int64_t) blk * 2 + i / 16) * 16 + c) * 16 + (i % 16)] = __float2half(x);
}

// MF=4 (V3) pre-pass. y -> act [nblk][4 g][16 col][8 k] f16 (k = 32 blk + 8 g + e): lane (col, g) of the 16x16x16 MFMA A
// fragment reads its 8 values of a block as one 16 B ds_read. Columns >= M are zero.
static __global__ void q8mf_pre_f16x16(const float * __restrict__ y, const int64_t s1, const int nblk, const int M,
                                       char * __restrict__ act, int * __restrict__ cnt, const int ncnt) {
    q8mf_zero_counters(cnt, ncnt);
    const int c   = blockIdx.y;
    const int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int i   = threadIdx.x % 32;
    if (blk >= nblk) {
        return;
    }
    const float x = c < M ? y[c * s1 + (int64_t) blk * 32 + i] : 0.0f;
    ((__half *) act)[(((int64_t) blk * 4 + i / 8) * 16 + c) * 8 + (i % 8)] = __float2half(x);
}

#if defined(GGML_USE_HIP)
typedef int      q8mf_v4i  __attribute__((ext_vector_type(4)));
typedef float    q8mf_v16f __attribute__((ext_vector_type(16)));
typedef _Float16 q8mf_v4h  __attribute__((ext_vector_type(4)));
typedef _Float16 q8mf_v2h  __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ q8mf_v4i q8mf_make_rsrc(const void * base, uint32_t num_bytes) {
    union { q8mf_v4i v; struct { const void * p; uint32_t range; uint32_t cfg; } s; } u;
    u.s.p     = base;
    u.s.range = num_bytes;
    u.s.cfg   = 0x00020000u;   // gfx9 raw buffer dword3, as mmq-q8-wvsk.cu / mmvq.cu / CK
    return u.v;
}

// aux: bit0 glc, bit1 slc (slc = nontemporal streaming). Out-of-range voffset (>= num_records) returns 0, no traffic.
__device__ q8mf_v4i q8mf_buf_x4(q8mf_v4i rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// LDS-only WG barrier: __syncthreads() would also drain vmcnt, i.e. wait for the prefetched stages' loads
#define Q8MF_LDS_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

// per-thread stage registers: 5 weight chunks + 2 activation chunks
struct q8mf_regs {
    q8mf_v4i w[Q8MF_NLD];
    q8mf_v4i a[2];
};

// weight chunk q (0..1087) of a stage = row q/17, byte 16*(q%17) of that row's 272 B segment
struct q8mf_geom {
    int r[Q8MF_NLD];     // row within the tile
    int c16[Q8MF_NLD];   // byte offset within the row segment
};

// issue the global loads of stage u (clamped by the caller to a real stage): weight rows clamp to N-1, the 5th chunk of
// waves 1..3 (q >= 1088) and activation chunks past the stage are pointed out of range (load 0, no memory traffic).
template <int ASTAGE>
static __device__ __forceinline__ void q8mf_issue(q8mf_regs & R, const q8mf_geom & G, const int u, const int S,
                                                  const q8mf_v4i wrsrc, const q8mf_v4i arsrc, const int nb01, const int N,
                                                  const uint32_t wbytes, const uint32_t abytes, const int tid) {
    const int tile = u / S;
    const int ks   = u - tile * S;
    const int soff = tile * Q8MF_RT * nb01 + ks * Q8MF_ROWB;   // uniform (SGPR)
    const int rlim = N - 1 - tile * Q8MF_RT;
#pragma unroll
    for (int i = 0; i < Q8MF_NLD; ++i) {
        // soffset must stay uniform (a lane-dependent soffset makes the compiler emit a readfirstlane waterfall loop);
        // the invalid 5th chunk goes out of range through voffset alone (voffset >= num_records -> 0, no traffic)
        const int vo = (i < Q8MF_NLD - 1 || tid < Q8MF_NCHUNK - (Q8MF_NLD - 1) * Q8MF_NT) ?
            min(G.r[i], rlim) * nb01 + G.c16[i] : (int) wbytes;
        R.w[i] = q8mf_buf_x4(wrsrc, vo, soff, 2);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int q = tid + i * Q8MF_NT;
        R.a[i] = q8mf_buf_x4(arsrc, q * 16 < ASTAGE ? ks * ASTAGE + q * 16 : (int) abytes, 0, 0);
    }
}

// branch-free commit: chunks past the stage (5th weight chunk of waves 1..3, MF=1 activation chunk 2 of threads >= 32) are
// written to a per-lane dummy slot instead of being skipped: a divergent ds_write becomes an exec branch, which splits the
// block and makes the waitcnt pass drain the whole prefetch (vmcnt(0)) at the next stage commit.
template <int ASTAGE>
static __device__ __forceinline__ void q8mf_commit(const q8mf_regs & R, char * __restrict__ lds, char * __restrict__ dummy,
                                                   const int tid) {
#pragma unroll
    for (int i = 0; i < Q8MF_NLD; ++i) {
        const int q = tid + i * Q8MF_NT;
        char * dst = (i < Q8MF_NLD - 1 || q < Q8MF_NCHUNK) ? lds + q * 16 : dummy + (tid & 63) * 16;
        *(q8mf_v4i *) dst = R.w[i];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int q = tid + i * Q8MF_NT;
        char * dst = q * 16 < ASTAGE ? lds + Q8MF_WSTAGE + q * 16 : dummy + (tid & 63) * 16;
        *(q8mf_v4i *) dst = R.a[i];
    }
}

// MF=1: one stage, wave w = rows 16w..16w+15, all 8 blocks. lane l: weight row j = l%16, k group g = l/16.
static __device__ __forceinline__ void q8mf_compute_i8(const char * __restrict__ lds, const int w, const int lane,
                                                       float (&acc)[4]) {
    const int j = lane & 15;
    const int g = lane >> 4;
    const char * lw = lds + (16 * w + j) * Q8MF_ROWB;
    const char * la = lds + Q8MF_WSTAGE;
#pragma unroll
    for (int i = 0; i < Q8MF_KS; ++i) {
        int w0, w1;
        if (i % 2 == 0) {   // qs of an even block start at 2 mod 4: aligned 12 B window + funnel shift by 2 bytes
            const int * p = (const int *) (lw + 34 * i + 8 * g);
            const int d0 = p[0], d1 = p[1], d2 = p[2];
            w0 = (int) __builtin_amdgcn_alignbyte((uint32_t) d1, (uint32_t) d0, 2);
            w1 = (int) __builtin_amdgcn_alignbyte((uint32_t) d2, (uint32_t) d1, 2);
        } else {            // odd block: qs dword aligned
            const int * p = (const int *) (lw + 34 * i + 2 + 8 * g);
            w0 = p[0];
            w1 = p[1];
        }
        const float dw = __half2float(*(const __half *) (lw + 34 * i));
        const int2   a  = *(const int2 *) (la + i * 512 + lane * 8);
        const float4 da = *(const float4 *) (la + Q8MF_KS * 512 + i * 64 + g * 16);
        const q8mf_v4i z = {0, 0, 0, 0};
        q8mf_v4i s = __builtin_amdgcn_mfma_i32_16x16x16i8(a.x, w0, z, 0, 0, 0);
        s          = __builtin_amdgcn_mfma_i32_16x16x16i8(a.y, w1, s, 0, 0, 0);
        acc[0] += (float) s[0] * (dw * da.x);
        acc[1] += (float) s[1] * (dw * da.y);
        acc[2] += (float) s[2] * (dw * da.z);
        acc[3] += (float) s[3] * (dw * da.w);
    }
}

// 4 int8 (one dword, k order = byte order) -> 4 f16 * d: (x ^ 0x80) | 0x6400 is the f16 1024 + (x + 128) exactly, so
// (h - 1152) is x exactly and the scale is applied after (a fused h*d - 1152*d would lose ~half an int step).
static __device__ __forceinline__ q8mf_v4h q8mf_dequant4(const int w, const q8mf_v2h d2) {
    const uint32_t u  = (uint32_t) w ^ 0x80808080u;
    const uint32_t lo = __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u);   // bytes {u0, 0x64, u1, 0x64}
    const uint32_t hi = __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u);   // bytes {u2, 0x64, u3, 0x64}
    const q8mf_v2h off = {(_Float16) 1152.0f, (_Float16) 1152.0f};
    const q8mf_v2h l = (__builtin_bit_cast(q8mf_v2h, lo) - off) * d2;
    const q8mf_v2h h = (__builtin_bit_cast(q8mf_v2h, hi) - off) * d2;
    return q8mf_v4h{l.x, l.y, h.x, h.y};
}

// MF=2: one stage, wave w = rows 32*(w&1).. +31, blocks 4*(w>>1).. +3. lane l: weight row j = l%32, k half h = l/32.
static __device__ __forceinline__ void q8mf_compute_f16(const char * __restrict__ lds, const int w, const int lane,
                                                        q8mf_v16f & acc) {
    const int j  = lane & 31;
    const int h  = lane >> 5;
    const int kg = w >> 1;
    const char * lw = lds + (32 * (w & 1) + j) * Q8MF_ROWB + kg * (4 * 34);
    const char * la = lds + Q8MF_WSTAGE + kg * (4 * 1024) + (h * 16 + (j & 15)) * 32;
    const bool  act_lane = j < 16;   // srcA rows 16..31 (activation columns 16..31) are zero padding
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) {   // block 4*kg + ii: parity == parity of ii (4*kg is even)
        int wq[4];
        if (ii % 2 == 0) {
            const int * p = (const int *) (lw + 34 * ii + 16 * h);
            const int d0 = p[0], d1 = p[1], d2 = p[2], d3 = p[3], d4 = p[4];
            wq[0] = (int) __builtin_amdgcn_alignbyte((uint32_t) d1, (uint32_t) d0, 2);
            wq[1] = (int) __builtin_amdgcn_alignbyte((uint32_t) d2, (uint32_t) d1, 2);
            wq[2] = (int) __builtin_amdgcn_alignbyte((uint32_t) d3, (uint32_t) d2, 2);
            wq[3] = (int) __builtin_amdgcn_alignbyte((uint32_t) d4, (uint32_t) d3, 2);
        } else {
            const int * p = (const int *) (lw + 34 * ii + 2 + 16 * h);
            wq[0] = p[0]; wq[1] = p[1]; wq[2] = p[2]; wq[3] = p[3];
        }
        const _Float16 dw = *(const _Float16 *) (lw + 34 * ii);
        const q8mf_v2h d2 = {dw, dw};
        const q8mf_v4i z4 = {0, 0, 0, 0};
        const q8mf_v4i a01 = act_lane ? *(const q8mf_v4i *) (la + ii * 1024)      : z4;   // k 16h + 0..7
        const q8mf_v4i a23 = act_lane ? *(const q8mf_v4i *) (la + ii * 1024 + 16) : z4;   // k 16h + 8..15
        const q8mf_v4h A0 = __builtin_bit_cast(q8mf_v4h, int2{a01[0], a01[1]});
        const q8mf_v4h A1 = __builtin_bit_cast(q8mf_v4h, int2{a01[2], a01[3]});
        const q8mf_v4h A2 = __builtin_bit_cast(q8mf_v4h, int2{a23[0], a23[1]});
        const q8mf_v4h A3 = __builtin_bit_cast(q8mf_v4h, int2{a23[2], a23[3]});
        acc = __builtin_amdgcn_mfma_f32_32x32x8f16(A0, q8mf_dequant4(wq[0], d2), acc, 0, 0, 0);
        acc = __builtin_amdgcn_mfma_f32_32x32x8f16(A1, q8mf_dequant4(wq[1], d2), acc, 0, 0, 0);
        acc = __builtin_amdgcn_mfma_f32_32x32x8f16(A2, q8mf_dequant4(wq[2], d2), acc, 0, 0, 0);
        acc = __builtin_amdgcn_mfma_f32_32x32x8f16(A3, q8mf_dequant4(wq[3], d2), acc, 0, 0, 0);
    }
}

typedef float q8mf_v4f __attribute__((ext_vector_type(4)));

// MF=4 (V3): one stage, wave w = rows 16w..16w+15, all 8 blocks; lane l: weight row j = l%16, k group g = l/16 holding qs
// bytes 8g..8g+7 of each block (MF=1's weight fragment), dequantised to f16 (scale folded in), 2 x 16x16x16 f16 MFMA per
// block: srcA = activations (16 columns x 16 k), srcB = weights, D[col][row] as MF=1, f32 accumulation over all of K.
static __device__ __forceinline__ void q8mf_compute_f16x16(const char * __restrict__ lds, const int w, const int lane,
                                                           q8mf_v4f & acc) {
    const int j = lane & 15;
    const int g = lane >> 4;
    const char * lw = lds + (16 * w + j) * Q8MF_ROWB;
    const char * la = lds + Q8MF_WSTAGE;
#pragma unroll
    for (int i = 0; i < Q8MF_KS; ++i) {
        int w0, w1;
        if (i % 2 == 0) {
            const int * p = (const int *) (lw + 34 * i + 8 * g);
            const int d0 = p[0], d1 = p[1], d2 = p[2];
            w0 = (int) __builtin_amdgcn_alignbyte((uint32_t) d1, (uint32_t) d0, 2);
            w1 = (int) __builtin_amdgcn_alignbyte((uint32_t) d2, (uint32_t) d1, 2);
        } else {
            const int * p = (const int *) (lw + 34 * i + 2 + 8 * g);
            w0 = p[0];
            w1 = p[1];
        }
        const _Float16 dw = *(const _Float16 *) (lw + 34 * i);
        const q8mf_v2h d2 = {dw, dw};
        const q8mf_v4i a  = *(const q8mf_v4i *) (la + i * 1024 + lane * 16);   // [blk][g][col][8 f16], lane = 16 g + col
        const q8mf_v4h A0 = __builtin_bit_cast(q8mf_v4h, int2{a[0], a[1]});    // k 8g + 0..3
        const q8mf_v4h A1 = __builtin_bit_cast(q8mf_v4h, int2{a[2], a[3]});    // k 8g + 4..7
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(A0, q8mf_dequant4(w0, d2), acc, 0, 0, 0);
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(A1, q8mf_dequant4(w1, d2), acc, 0, 0, 0);
    }
}

// V3 stage registers / issue: full byte offset in voffset (soffset 0), so out-of-range lanes (rows >= N, the 5th weight
// chunk of waves 1..3, activation chunks of columns >= M, stages past the WG's range) read 0 with no memory traffic.
template <int AUX>
static __device__ __forceinline__ void q8mf3_issue(q8mf_regs & R, const uint32_t (&vb)[Q8MF_NLD], const uint32_t (&ab)[2],
                                                   const uint32_t sw, const uint32_t sa,
                                                   const q8mf_v4i wrsrc, const q8mf_v4i arsrc) {
#pragma unroll
    for (int i = 0; i < Q8MF_NLD; ++i) {
        R.w[i] = q8mf_buf_x4(wrsrc, (int) (sw + vb[i]), 0, AUX);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        R.a[i] = q8mf_buf_x4(arsrc, (int) (sa + ab[i]), 0, 0);
    }
}
#endif // defined(GGML_USE_HIP)

// grid = G persistent WGs, block = 256 (4 waves). WG g streams stages [g*L, min(g*L + L, T)) of the flattened
// (tile, stage) space (stage u -> tile u / S, k stage u % S).
template <int V>
static __global__ void __launch_bounds__(Q8MF_NT)
q8mf2_mul_mat(const char * __restrict__ W, const int nb01, const int N, const uint32_t wbytes,
              const char * __restrict__ act, const uint32_t abytes, const int S, const int T, const int L, const int M,
              float * __restrict__ out, const int64_t out_s1, float * __restrict__ part, int * __restrict__ cnt) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    constexpr int ASTAGE = V == 1 ? Q8MF_A1 : Q8MF_A2;
    constexpr int BUF    = Q8MF_WSTAGE + ASTAGE;
    constexpr int SCR    = V == 1 ? 0 : 2 * 8 * 64 * 4;   // MF=2 K-half reduction scratch [rg][8 reg][64 lane] f32
    __shared__ __attribute__((aligned(16))) char lds[2 * BUF + SCR + 64 * 16];   // + per-lane dummy commit slot
    __shared__ int s_last;

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int w    = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int g    = blockIdx.x;
    const int u0   = g * L;
    const int u1   = min(u0 + L, T);
    if (u0 >= u1) {
        return;
    }

    q8mf_geom G;
#pragma unroll
    for (int i = 0; i < Q8MF_NLD; ++i) {
        const int q = min(tid + i * Q8MF_NT, Q8MF_NCHUNK - 1);
        G.r[i]   = q / 17;
        G.c16[i] = (q % 17) * 16;
    }
    const q8mf_v4i wrsrc = q8mf_make_rsrc(W, wbytes);
    const q8mf_v4i arsrc = q8mf_make_rsrc(act, abytes);

    float     acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q8mf_v16f acc2    = {};

    // flush the accumulator of the tile whose last stage in this WG's range is u
    auto flush = [&](const int u) {
        const int  t       = u / S;
        const int  ts      = t * S;
        const bool partial = ts < u0 || ts + S > u1;
        // owner lanes and their (column, row-in-tile) elements
        constexpr int NV = V == 1 ? 4 : 8;
        float v[NV];
        int   cols[NV], rl[NV];
        bool  owner;
        if constexpr (V == 1) {
            owner = true;
#pragma unroll
            for (int r = 0; r < 4; ++r) {   // D lane l reg r = D[col 4*(l/16) + r][row l%16]
                v[r] = acc1[r]; cols[r] = 4 * (lane >> 4) + r; rl[r] = 16 * w + (lane & 15);
            }
        } else {
            float * scr = (float *) (lds + 2 * BUF);
            const int rg = w & 1;
            if (w >= 2) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    scr[(rg * 8 + r) * 64 + lane] = acc2[r];
                }
            }
            Q8MF_LDS_BARRIER();
            owner = w < 2;
#pragma unroll
            for (int r = 0; r < 8; ++r) {   // D lane l reg r = D[col 8*(r/4) + 4*(l/32) + r%4][row l%32]; regs 8..15 = cols >= 16
                v[r]    = owner ? acc2[r] + scr[(rg * 8 + r) * 64 + lane] : 0.0f;
                cols[r] = 8 * (r >> 2) + 4 * (lane >> 5) + (r & 3);
                rl[r]   = 32 * rg + (lane & 31);
            }
        }
        if (!partial) {
            if (owner) {
#pragma unroll
                for (int e = 0; e < NV; ++e) {
                    const int row = t * Q8MF_RT + rl[e];
                    if (cols[e] < M && row < N) {
                        out[(int64_t) cols[e] * out_s1 + row] = v[e];
                    }
                }
            }
        } else {
            const int slot = ts < u0 ? 0 : 1;
            float * pp = part + ((int64_t) g * 2 + slot) * (Q8MF_MAXM * Q8MF_RT);
            if (owner) {
#pragma unroll
                for (int e = 0; e < NV; ++e) {
                    pp[cols[e] * Q8MF_RT + rl[e]] = v[e];
                }
            }
            __threadfence();
            __syncthreads();
            const int g0 = ts / L;
            const int g1 = (ts + S - 1) / L;
            if (tid == 0) {
                s_last = atomicAdd(cnt + t, 1) == g1 - g0;
            }
            __syncthreads();
            if (s_last) {
                __threadfence();
                for (int e = tid; e < Q8MF_MAXM * Q8MF_RT; e += Q8MF_NT) {
                    const int col = e / Q8MF_RT;
                    const int r   = e % Q8MF_RT;
                    const int row = t * Q8MF_RT + r;
                    if (col < M && row < N) {
                        float s = 0.0f;
                        for (int gg = g0; gg <= g1; ++gg) {   // WG order: deterministic
                            const int sl = ts < gg * L ? 0 : 1;
                            s += __builtin_nontemporal_load(part + ((int64_t) gg * 2 + sl) * (Q8MF_MAXM * Q8MF_RT) + e);
                        }
                        out[(int64_t) col * out_s1 + row] = s;
                    }
                }
            }
        }
        if constexpr (V == 1) {
            acc1[0] = acc1[1] = acc1[2] = acc1[3] = 0.0f;
        } else {
            acc2 = q8mf_v16f{};
        }
    };
    auto compute = [&](const char * buf) {
        if constexpr (V == 1) {
            q8mf_compute_i8(buf, w, lane, acc1);
        } else {
            q8mf_compute_f16(buf, w, lane, acc2);
        }
    };
    auto is_end = [&](const int u) { return u == u1 - 1 || (u + 1) % S == 0; };

    char * buf0  = lds;
    char * buf1  = lds + BUF;
    char * dummy = lds + 2 * BUF + SCR;
    q8mf_regs RA, RB;
    q8mf_issue<ASTAGE>(RA, G, u0, S, wrsrc, arsrc, nb01, N, wbytes, abytes, tid);
    q8mf_commit<ASTAGE>(RA, buf0, dummy, tid);
    // compiler barriers: RA's loads must be issued before RB's (vmcnt retires in issue order; the waitcnt pass then lets
    // RB stay in flight while RA is committed)
    asm volatile("" ::: "memory");
    q8mf_issue<ASTAGE>(RA, G, min(u0 + 1, u1 - 1), S, wrsrc, arsrc, nb01, N, wbytes, abytes, tid);
    asm volatile("" ::: "memory");
    q8mf_issue<ASTAGE>(RB, G, min(u0 + 2, u1 - 1), S, wrsrc, arsrc, nb01, N, wbytes, abytes, tid);
    Q8MF_LDS_BARRIER();
    // invariant at the top of an iteration: buf0 = stage u, RA = stage u+1, RB = stage u+2 (in flight, clamped).
    // ONE loop exit, at the bottom: a mid-body break makes the waitcnt pass lose the in-flight order across the back
    // edge and wait vmcnt(6..0) at the first commit (drains the 2-stage prefetch every other stage; ISA-verified). An odd
    // stage count runs one extra (clamped, harmless) commit/issue/barrier instead.
    int u = u0;
    do {
        compute(buf0);
        if (is_end(u)) {
            flush(u);
        }
        q8mf_commit<ASTAGE>(RA, buf1, dummy, tid);
        q8mf_issue<ASTAGE>(RA, G, min(u + 3, u1 - 1), S, wrsrc, arsrc, nb01, N, wbytes, abytes, tid);
        Q8MF_LDS_BARRIER();
        if (u + 1 < u1) {
            compute(buf1);
            if (is_end(u + 1)) {
                flush(u + 1);
            }
        }
        q8mf_commit<ASTAGE>(RB, buf0, dummy, tid);
        q8mf_issue<ASTAGE>(RB, G, min(u + 4, u1 - 1), S, wrsrc, arsrc, nb01, N, wbytes, abytes, tid);
        Q8MF_LDS_BARRIER();
        u += 2;
    } while (u < u1);
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(N); GGML_UNUSED(wbytes); GGML_UNUSED(act); GGML_UNUSED(abytes);
    GGML_UNUSED(S); GGML_UNUSED(T); GGML_UNUSED(L); GGML_UNUSED(M); GGML_UNUSED(out); GGML_UNUSED(out_s1);
    GGML_UNUSED(part); GGML_UNUSED(cnt);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

// V3: grid = G persistent WGs (2 per CU), block = 256 (4 waves). WG g streams stages [u0, u1) = [floor(gT/G),
// floor((g+1)T/G)) of the flattened (tile, stage) space; owner(u) = floor(((u+1)G - 1)/T). Three register stages in
// flight, one LDS stage buffer. V = 3 (int8) or 4 (f16); AUX = weight-load cache bits (0, or 2 = slc).
#define Q8MF3_P 3
static __device__ __forceinline__ int q8mf3_u0(const int g, const int T, const int G) {
    return (int) (((int64_t) g * T) / G);
}
static __device__ __forceinline__ int q8mf3_owner(const int u, const int T, const int G) {
    return (int) ((((int64_t) u + 1) * G - 1) / T);
}

template <int V, int AUX>
static __global__ void __launch_bounds__(Q8MF_NT, 2)
q8mf3_mul_mat(const char * __restrict__ W, const int nb01, const uint32_t wbytes, const char * __restrict__ act,
              const uint32_t abytes, const int S, const int T, const int G, const int M, const int N,
              float * __restrict__ out, const int64_t out_s1, float * __restrict__ part, int * __restrict__ cnt) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    constexpr int ASTAGE = V == 3 ? Q8MF_A1 : Q8MF_A2;   // MF=3: MF=1's int8 + scale stage; MF=4: [8][4 g][16 col][8] f16
    __shared__ __attribute__((aligned(16))) char lds[Q8MF_WSTAGE + ASTAGE + 64 * 16];   // + per-lane dummy commit slot
    __shared__ int s_last[2];

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int w    = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int g    = blockIdx.x;
    const int u0   = q8mf3_u0(g, T, G);
    const int u1   = q8mf3_u0(g + 1, T, G);   // > u0 since G <= T

    constexpr uint32_t OOB = 0x80000000u;      // > any wbytes / abytes (< INT32_MAX - 4096); sum with a base stays < 2^32
    uint32_t vb[Q8MF_NLD];
#pragma unroll
    for (int i = 0; i < Q8MF_NLD; ++i) {
        const int q = tid + i * Q8MF_NT;
        vb[i] = q < Q8MF_NCHUNK ? (uint32_t) ((q / 17) * nb01 + (q % 17) * 16) : OOB;
    }
    uint32_t ab[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int q = tid + i * Q8MF_NT;
        bool ok;
        if constexpr (V == 3) {   // int8 part [8][4 g][16 col][8 B]: chunk = 2 columns; scales [8][16 col] f32: chunk = 4 columns
            ok = q < 256 ? 2 * (q % 8) < M : (q < 288 && 4 * ((q - 256) % 4) < M);
        } else {                  // [8][4 g][16 col][16 B]: chunk = 1 column
            ok = q < 512 && (q % 16) < M;
        }
        ab[i] = ok ? (uint32_t) (q * 16) : OOB;
    }
    const q8mf_v4i wrsrc = q8mf_make_rsrc(W, wbytes);
    const q8mf_v4i arsrc = q8mf_make_rsrc(act, abytes);
    const uint32_t tstride = (uint32_t) Q8MF_RT * (uint32_t) nb01;

    // issue pointer: stage ui = (it, iks); stages >= u1 go fully out of range
    int ui = u0, it = u0 / S, iks = u0 - (u0 / S) * S;
    auto issue_next = [&](q8mf_regs & R) {
        const bool live = ui < u1;
        const uint32_t sw = live ? (uint32_t) it * tstride + (uint32_t) (iks * Q8MF_ROWB) : wbytes;
        const uint32_t sa = live ? (uint32_t) (iks * ASTAGE) : abytes;
        q8mf3_issue<AUX>(R, vb, ab, sw, sa, wrsrc, arsrc);
        ++ui;
        if (++iks == S) {
            iks = 0;
            ++it;
        }
    };

    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // MF=3
    q8mf_v4f acc4 = {0.0f, 0.0f, 0.0f, 0.0f};    // MF=4
    float pacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};    // held first partial tile (slot 0)
    bool  have_p  = false;                       // uniform
    const int rl  = 16 * w + (lane & 15);        // D lane l reg r = D[col 4*(l/16) + r][row-in-tile l%16]
    const int c0  = 4 * (lane >> 4);

    // compute pointer: stage uc = (ct, cks)
    int ct = u0 / S, cks = u0 - (u0 / S) * S;
    auto end_of_stage = [&](const int uc) {   // tile flush when uc is the tile's last stage in this WG's range
        if (cks == S - 1 || uc == u1 - 1) {
            float v[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                v[r] = V == 3 ? acc1[r] : acc4[r];
            }
            const int  ts      = ct * S;
            const bool partial = ts < u0 || ts + S > u1;
            if (!partial) {
                const int row = ct * Q8MF_RT + rl;
                if (row < N) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        if (c0 + r < M) {
                            out[(int64_t) (c0 + r) * out_s1 + row] = v[r];
                        }
                    }
                }
            } else if (uc != u1 - 1) {   // first partial tile (starts before u0, ends inside the range): hold to the end
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    pacc[r] = v[r];
                }
                have_p = true;
            }
            if (uc != u1 - 1) {   // the last tile's accumulator stays live for the epilogue
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    acc1[r] = 0.0f;
                    acc4[r] = 0.0f;
                }
            }
        }
        if (++cks == S) {
            cks = 0;
            ++ct;
        }
    };

    char * dummy = lds + Q8MF_WSTAGE + ASTAGE;
    q8mf_regs R[Q8MF3_P];
#pragma unroll
    for (int j = 0; j < Q8MF3_P; ++j) {
        issue_next(R[j]);
        asm volatile("" ::: "memory");   // keep R[0], R[1], R[2] in issue order (vmcnt retires in order)
    }
    // invariant at the top of step j of an iteration: R[j] holds stage u + j, R[j+1..] the next ones (in flight).
    // ONE loop exit at the bottom (a mid-body exit makes the waitcnt pass drain the prefetch, see V2); a stage count that
    // is not a multiple of 3 runs up to 2 extra (out-of-range, zero-traffic) commit / issue / barrier steps.
    int u = u0;
    do {
#pragma unroll
        for (int j = 0; j < Q8MF3_P; ++j) {
            q8mf_commit<ASTAGE>(R[j], lds, dummy, tid);
            issue_next(R[j]);
            Q8MF_LDS_BARRIER();
            if (u + j < u1) {
                if constexpr (V == 3) {
                    q8mf_compute_i8(lds, w, lane, acc1);
                } else {
                    q8mf_compute_f16x16(lds, w, lane, acc4);
                }
                end_of_stage(u + j);
            }
            Q8MF_LDS_BARRIER();   // every wave is done reading the buffer before the next commit overwrites it
        }
        u += Q8MF3_P;
    } while (u < u1);

    // epilogue: <= 2 partial tiles. A = held first partial (slot 0, tile u0 / S); B = last tile if partial.
    const int  tA  = u0 / S;
    const int  tB  = (u1 - 1) / S;
    const bool pB  = tB * S < u0 || tB * S + S > u1;
    const int  sB  = tB * S < u0 ? 0 : 1;
    if (!have_p && !pB) {
        return;
    }
    float * pg = part + (int64_t) g * 2 * (Q8MF_MAXM * Q8MF_RT);
    if (have_p) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            pg[(c0 + r) * Q8MF_RT + rl] = pacc[r];
        }
    }
    if (pB) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            pg[sB * (Q8MF_MAXM * Q8MF_RT) + (c0 + r) * Q8MF_RT + rl] = V == 3 ? acc1[r] : acc4[r];
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last[0] = 0;
        s_last[1] = 0;
        if (have_p) {
            const int ts = tA * S;
            s_last[0] = atomicAdd(cnt + tA, 1) == q8mf3_owner(ts + S - 1, T, G) - q8mf3_owner(ts, T, G);
        }
        if (pB) {
            const int ts = tB * S;
            s_last[1] = atomicAdd(cnt + tB, 1) == q8mf3_owner(ts + S - 1, T, G) - q8mf3_owner(ts, T, G);
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        if (!s_last[k]) {
            continue;
        }
        __threadfence();
        const int t  = k == 0 ? tA : tB;
        const int ts = t * S;
        const int g0 = q8mf3_owner(ts, T, G);
        const int g1 = q8mf3_owner(ts + S - 1, T, G);
        for (int e = tid; e < Q8MF_MAXM * Q8MF_RT; e += Q8MF_NT) {
            const int col = e / Q8MF_RT;
            const int r   = e % Q8MF_RT;
            const int row = t * Q8MF_RT + r;
            if (col < M && row < N) {
                float s = 0.0f;
                for (int gg = g0; gg <= g1; ++gg) {   // WG order: deterministic
                    const int sl = ts < q8mf3_u0(gg, T, G) ? 0 : 1;
                    s += __builtin_nontemporal_load(part + ((int64_t) gg * 2 + sl) * (Q8MF_MAXM * Q8MF_RT) + e);
                }
                out[(int64_t) col * out_s1 + row] = s;
            }
        }
    }
#else
    GGML_UNUSED(W); GGML_UNUSED(nb01); GGML_UNUSED(wbytes); GGML_UNUSED(act); GGML_UNUSED(abytes); GGML_UNUSED(S);
    GGML_UNUSED(T); GGML_UNUSED(G); GGML_UNUSED(M); GGML_UNUSED(N); GGML_UNUSED(out); GGML_UNUSED(out_s1);
    GGML_UNUSED(part); GGML_UNUSED(cnt);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

void ggml_cuda_mul_mat_q8_mf(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8mf_cfg & cfg = q8mf_get_cfg();
    {
        // Path witness (gate packets grep it; also a content marker of the build).
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            if (cfg.variant <= 2) {
                GGML_LOG_INFO("ggml_cuda_q8_mf: V2 stream-K small-M Q8_0 path ran (call 1, GGML_HIP_Q8_MF=%d, %s)\n",
                    cfg.variant, cfg.variant == 1 ? "int8 mfma_i32_16x16x16i8" : "f16 mfma_f32_32x32x8f16 in-register dequant");
            } else {
                GGML_LOG_INFO("ggml_cuda_q8_mf: V3 2-WG/CU 3-stage stream-K small-M Q8_0 path ran (call 1, GGML_HIP_Q8_MF=%d, %s, "
                    "slc=%d)\n", cfg.variant, cfg.variant == 3 ? "int8 mfma_i32_16x16x16i8" : "f16 mfma_f32_16x16x16f16 in-register dequant",
                    cfg.slc);
            }
        }
    }
    GGML_ASSERT(cfg.variant >= 1 && cfg.variant <= 4);
    cudaStream_t stream = ctx.stream();

    const int     N       = (int) src0->ne[1];
    const int     M       = (int) src1->ne[1];
    const int     nblk    = (int) (src0->ne[0] / 32);
    const int64_t s1      = src1->nb[1] / sizeof(float);
    const int64_t dst_s1  = dst->nb[1]  / sizeof(float);
    const int     n_tiles = (N + Q8MF_RT - 1) / Q8MF_RT;
    const int     S       = nblk / Q8MF_KS;
    const int     T       = n_tiles * S;
    const int     ncu     = ggml_cuda_info().devices[ctx.device].nsm;
    GGML_ASSERT(M >= 1 && M <= Q8MF_MAXM && nblk % Q8MF_KS == 0);

    const bool v3 = cfg.variant >= 3;
    int G, L = 0;
    if (v3) {   // balanced split, 2 WGs per CU (every WG gets >= 1 stage)
        G = std::max(1, std::min(cfg.wg > 0 ? cfg.wg : 2 * ncu, T));
    } else {
        const int G0 = std::max(1, std::min(cfg.wg > 0 ? cfg.wg : ncu, T));
        L = (T + G0 - 1) / G0;
        G = (T + L - 1) / L;
    }

    const int  astage = cfg.variant == 1 || cfg.variant == 3 ? Q8MF_A1 : Q8MF_A2;
    const size_t abytes = (size_t) S * astage;
    ggml_cuda_pool_alloc<char>  act(ctx.pool(), abytes);
    ggml_cuda_pool_alloc<int>   cnt(ctx.pool(), (size_t) n_tiles);
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) G * 2 * Q8MF_MAXM * Q8MF_RT);
    {
        const dim3 grid((nblk + 7) / 8, Q8MF_MAXM, 1);
        if (cfg.variant == 1 || cfg.variant == 3) {
            q8mf_pre_i8<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, M, act.get(), cnt.get(), n_tiles);
        } else if (cfg.variant == 2) {
            q8mf_pre_f16<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, M, act.get(), cnt.get(), n_tiles);
        } else {
            q8mf_pre_f16x16<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, M, act.get(), cnt.get(), n_tiles);
        }
    }

    const char *   W      = (const char *) src0->data;
    const int      nb01   = (int) src0->nb[1];
    const uint32_t wbytes = (uint32_t) ggml_nbytes(src0);   // = N * nb01 (contiguous): rows >= N are out of range
    float *        d      = (float *) dst->data;
    switch (cfg.variant * 2 + cfg.slc) {
        case 2: case 3:
            q8mf2_mul_mat<1><<<G, Q8MF_NT, 0, stream>>>(W, nb01, N, wbytes, act.get(), (uint32_t) abytes, S, T, L, M, d, dst_s1, part.get(), cnt.get());
            break;
        case 4: case 5:
            q8mf2_mul_mat<2><<<G, Q8MF_NT, 0, stream>>>(W, nb01, N, wbytes, act.get(), (uint32_t) abytes, S, T, L, M, d, dst_s1, part.get(), cnt.get());
            break;
#define Q8MF3_CASE(c, v, aux) case c: \
            q8mf3_mul_mat<v, aux><<<G, Q8MF_NT, 0, stream>>>(W, nb01, wbytes, act.get(), (uint32_t) abytes, S, T, G, M, N, d, dst_s1, part.get(), cnt.get()); \
            break;
        Q8MF3_CASE(6, 3, 0) Q8MF3_CASE(7, 3, 2) Q8MF3_CASE(8, 4, 0) Q8MF3_CASE(9, 4, 2)
#undef Q8MF3_CASE
        default: GGML_ABORT("q8_mf: bad variant %d", cfg.variant);
    }
    CUDA_CHECK(cudaGetLastError());
}
