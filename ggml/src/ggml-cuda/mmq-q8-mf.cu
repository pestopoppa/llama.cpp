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

#include "quantize.cuh"
#include "mmq.cuh"
#include "convert.cuh"
#include <atomic>
#include <climits>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <cstring>
#include <vector>

#define Q8MF_MAXM    16
#define Q8MF5_KC     16                          // V4: Q8_0 blocks per LDS activation chunk (512 k)
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
    int variant; int mmin; int mmax; int wg; int slc; int repack; int64_t min_bytes;
};

static const q8mf_cfg & q8mf_get_cfg() {
    static const q8mf_cfg cfg = [] {
        q8mf_cfg c;
        c.variant = q8mf_env_int("GGML_HIP_Q8_MF", 0);
        if (c.variant < 1 || c.variant > 6) {
            c.variant = 0;
        }
        c.slc     = q8mf_env_int("GGML_HIP_Q8_MF_SLC", 0) != 0 ? 1 : 0;
        c.mmin    = std::max(1, q8mf_env_int("GGML_HIP_Q8_MF_MIN", c.variant == 5 ? 1 : 9));
        c.mmax    = std::min(Q8MF_MAXM, q8mf_env_int("GGML_HIP_Q8_MF_MAX", 16));
        c.wg      = std::max(0, q8mf_env_int("GGML_HIP_Q8_MF_WG", 0));
        // DTREE-0-Q8MF-HBM-CEILING: 1 = stage-contiguous repacked weight copy cached per (data, N, K) (perf), 2 = repack on
        // every call (correctness: test buffers reuse addresses). V3 only.
        c.repack  = std::max(0, std::min(2, q8mf_env_int("GGML_HIP_Q8_MF_REPACK", 0)));
        const char * mb = getenv("GGML_HIP_Q8_MF_MIN_BYTES");
        c.min_bytes = (mb && *mb) ? std::max(0ll, atoll(mb)) : (c.variant == 5 ? 0ll : (6ll << 20));
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
    if (cfg.variant == 5) {   // V4 mmvq-multicol: any M in [mmin..mmax] <= 16, K % 512 == 0
        return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            src1->ne[1] >= cfg.mmin && src1->ne[1] <= cfg.mmax &&
            src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
            ggml_is_contiguous(src0) && src0->ne[0] % (32 * Q8MF5_KC) == 0 &&
            src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
            dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
            nbytes > cfg.min_bytes && nbytes < (int64_t) INT32_MAX - 4096;
    }
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
                                   char * __restrict__ act, int * __restrict__ cnt, const int ncnt,
                                   float * __restrict__ z = nullptr, const int64_t zs1 = 0, const int zN = 0) {
    q8mf_zero_counters(cnt, ncnt);
    if (z != nullptr) {   // V5: zero dst [M][zN] (row stride zs1) for the atomic partial-tile adds
        const int nthr = gridDim.x * gridDim.y * blockDim.x;
        for (int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; i < M * zN; i += nthr) {
            z[(int64_t) (i / zN) * zs1 + i % zN] = 0.0f;
        }
    }
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
q8mf3_mul_mat(const char * __restrict__ W, const int nb01, const int kstride, const uint32_t tstride, const uint32_t wbytes,
              const char * __restrict__ act,
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

    // issue pointer: stage ui = (it, iks); stages >= u1 go fully out of range
    int ui = u0, it = u0 / S, iks = u0 - (u0 / S) * S;
    auto issue_next = [&](q8mf_regs & R) {
        const bool live = ui < u1;
        const uint32_t sw = live ? (uint32_t) it * tstride + (uint32_t) (iks * kstride) : wbytes;
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


// ---------------------------------------------------------------------------------------------------------------------
// DTREE-0-Q8MF-V4 (GGML_HIP_Q8_MF=5) "V4 mmvq-multicol": MMVQ that serves up to 16 columns.
//   Keeps MMVQ's Q8_0 weight access (4 lanes per 34-B block, 2 ints/lane via get_int_b2, one wave = 16 contiguous
//   blocks of a row) and its occupancy (small register file: RPW x NC <= 32 f32 accumulators). Activations are
//   quantised ONCE per call to q8_1 (quantize_row_q8_1_cuda, the MMVQ path) and the NC columns' q8_1 tile for the
//   current 512-k chunk is staged in LDS (NC*16*36 B <= 9.2 KB) and shared by the 4 waves of the WG. int8 math via
//   ggml_cuda_dp4a (v_dot4_i32_i8 on gfx90a). WG = 4 waves, each wave owns RPW rows; grid = ceil(N / (4*RPW)).
//   NC buckets: 1, 2, 4, 8, 16 (M padded up to the bucket; padded columns are zero and never written).
// ---------------------------------------------------------------------------------------------------------------------
static const char q8mf5_build_marker[] __attribute__((used)) = "ggml-cuda: V4 mmvq-multicol Q8_0 small-M kernel (GGML_HIP_Q8_MF=5)";

template <int NC, int RPW>
static __global__ void __launch_bounds__(256, 2)
q8mf5_mul_mat(const char * __restrict__ W, const int nb01, const int N, const block_q8_1 * __restrict__ Y,
              const int s11, const int nchunk, const int M, float * __restrict__ dst, const int64_t dst_s1) {
    __shared__ __align__(16) int   s_q[NC * Q8MF5_KC * 8];
    __shared__               float s_d[NC * Q8MF5_KC];
    const int tid  = threadIdx.x;
    const int wave = tid / 64;
    const int lane = tid % 64;
    const int b    = lane / 4;       // block within the chunk
    const int i    = lane % 4;       // int pair within the block
    const int row0 = (blockIdx.x * 4 + wave) * RPW;

    const block_q8_0 * wr[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        wr[r] = (const block_q8_0 *) (W + (int64_t) min(row0 + r, N - 1) * nb01);
    }
    float acc[RPW][NC];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    for (int ch = 0; ch < nchunk; ++ch) {
        const int kb = ch * Q8MF5_KC + b;
        // issue the weight loads first so they overlap the LDS fill + barrier
        int   w0[RPW], w1[RPW];
        float dw[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const block_q8_0 * bw = wr[r] + kb;
            w0[r] = get_int_b2(bw->qs, 2 * i + 0);
            w1[r] = get_int_b2(bw->qs, 2 * i + 1);
            dw[r] = __half2float(bw->d);
        }
        __syncthreads();
        for (int e = tid; e < NC * Q8MF5_KC * 8; e += 256) {
            const int c = e / (Q8MF5_KC * 8);
            const int k = (e / 8) % Q8MF5_KC;
            const int j = e % 8;
            s_q[e] = c < M ? ((const int *) Y[(int64_t) c * s11 + ch * Q8MF5_KC + k].qs)[j] : 0;
        }
        for (int e = tid; e < NC * Q8MF5_KC; e += 256) {
            const int c = e / Q8MF5_KC;
            s_d[e] = c < M ? __low2float(Y[(int64_t) c * s11 + ch * Q8MF5_KC + e % Q8MF5_KC].ds) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int2  a  = *(const int2 *) &s_q[(c * Q8MF5_KC + b) * 8 + 2 * i];
            const float da = s_d[c * Q8MF5_KC + b];
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const int sumi = ggml_cuda_dp4a(w1[r], a.y, ggml_cuda_dp4a(w0[r], a.x, 0));
                acc[r][c] += dw[r] * da * (float) sumi;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float v = warp_reduce_sum<64>(acc[r][c]);
            if (lane == ((r * NC + c) & 63) && c < M && row0 + r < N) {
                dst[(int64_t) c * dst_s1 + row0 + r] = v;
            }
        }
    }
}

static void ggml_cuda_mul_mat_q8_mf5(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    {
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8_mf: V4 mmvq-multicol small-M Q8_0 path ran (call 1, GGML_HIP_Q8_MF=5, %s)\n",
                q8mf5_build_marker);
        }
    }
    cudaStream_t stream = ctx.stream();
    const int64_t ne10 = src1->ne[0];
    const int     M    = (int) src1->ne[1];
    const int     N    = (int) src0->ne[1];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> q(ctx.pool(), (size_t) M * ne10_padded * sizeof(block_q8_1) / QK8_1);
    const int64_t s11 = src1->nb[1] / sizeof(float);
    quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q.get(), src0->type, ne10, s11, s11 * M, s11 * M,
        ne10_padded, M, 1, 1, stream);
    const block_q8_1 * Y = (const block_q8_1 *) q.get();
    const int ys     = (int) (ne10_padded / QK8_1);
    const int nchunk = (int) (ne10 / (32 * Q8MF5_KC));
    const int nb01   = (int) src0->nb[1];
    const int64_t dst_s1 = dst->nb[1] / sizeof(float);
    const char * W = (const char *) src0->data;
    float * d = (float *) dst->data;
#define Q8MF5_LAUNCH(NC, RPW) \
    q8mf5_mul_mat<NC, RPW><<<(N + 4 * RPW - 1) / (4 * RPW), 256, 0, stream>>>(W, nb01, N, Y, ys, nchunk, M, d, dst_s1)
    if (M <= 1) {        Q8MF5_LAUNCH(1, 4);
    } else if (M <= 2) { Q8MF5_LAUNCH(2, 4);
    } else if (M <= 4) { Q8MF5_LAUNCH(4, 4);
    } else if (M <= 8) { Q8MF5_LAUNCH(8, 4);
    } else {             Q8MF5_LAUNCH(16, 2);
    }
#undef Q8MF5_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// DTREE-0-Q8MF-HBM-CEILING: stage-contiguous repack. Natural layout: a stage = 64 rows x 272 B segments at row stride nb01
// (64 scattered 272 B pieces, each straddling 3-4 128 B lines shared with the neighbouring k-stages). Packed layout:
// [tile][k-stage][64 rows][272 B] -> each stage is ONE contiguous, 128 B aligned 17408 B block and a WG's stream-K range
// is one contiguous byte range. Block bytes are unchanged (pure permutation; rows >= N are zero), so the compute path is
// identical. Grid-stride over 16 B chunks.
static __global__ void q8mf_repack_stage(const char * __restrict__ W, const int64_t nb01, const int N, const int S,
                                         const int64_t nchunk, char * __restrict__ P) {
    for (int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; q < nchunk; q += (int64_t) gridDim.x * blockDim.x) {
        const int64_t st  = q / Q8MF_NCHUNK;         // packed stage index = tile * S + ks
        const int     c   = (int) (q % Q8MF_NCHUNK);
        const int64_t t   = st / S;
        const int     ks  = (int) (st % S);
        const int     row = (int) (t * Q8MF_RT) + c / 17;
        int4 v = make_int4(0, 0, 0, 0);
        if (row < N) {
            v = *(const int4 *) (W + row * nb01 + (int64_t) ks * Q8MF_ROWB + (c % 17) * 16);
        }
        *(int4 *) (P + q * 16) = v;
    }
}

struct q8mf_pack_key {
    const void * data; int64_t n; int64_t k; int dev;
    bool operator<(const q8mf_pack_key & o) const {
        return std::tie(data, n, k, dev) < std::tie(o.data, o.n, o.k, o.dev);
    }
};

// returns the packed copy (cached for repack == 1, a pool buffer refreshed every call for repack == 2)
static const char * q8mf_get_packed(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, int repack,
                                    ggml_cuda_pool_alloc<char> & tmp, size_t & pbytes) {
    static std::map<q8mf_pack_key, char *> cache;
    static std::mutex mu;
    const int     N       = (int) src0->ne[1];
    const int     S       = (int) (src0->ne[0] / 256);
    const int64_t n_tiles = (N + Q8MF_RT - 1) / Q8MF_RT;
    pbytes = (size_t) n_tiles * S * Q8MF_WSTAGE;
    const int64_t nchunk = (int64_t) pbytes / 16;
    cudaStream_t stream = ctx.stream();
    auto launch = [&](char * P) {
        const int grid = (int) std::min<int64_t>((nchunk + 255) / 256, 65535);
        q8mf_repack_stage<<<grid, 256, 0, stream>>>((const char *) src0->data, (int64_t) src0->nb[1], N, S, nchunk, P);
        CUDA_CHECK(cudaGetLastError());
    };
    if (repack == 2) {
        char * P = tmp.alloc(ctx.pool(), pbytes);
        launch(P);
        return P;
    }
    std::lock_guard<std::mutex> lk(mu);
    const q8mf_pack_key key{src0->data, src0->ne[1], src0->ne[0], ctx.device};
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    char * P = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &P, pbytes));   // experiment: never freed (one copy per weight; op-perf / A/B only)
    launch(P);
    cache.emplace(key, P);
    return P;
}

// ---------------------------------------------------------------------------------------------------------------------
// DTREE-0-Q8MF-REPACK-INPLACE (GGML_HIP_Q8_MF_INPLACE=1|2, needs GGML_HIP_Q8_MF=3|4): the HBM-CEILING stage-contiguous
// layout applied IN PLACE once, when a weight's upload completes. No cached copy, no persistent extra VRAM.
//   Packed layout [tile][k-stage][64 rows][272 B] is a permutation INSIDE each 64-row tile: a tile is 64*nb01 contiguous
//   bytes in both layouts (nb01 = S*272), so with N % 64 == 0 the packed tensor has exactly ggml_nbytes bytes.
//   Upload: after the write that completes the tensor (offset + size == nbytes; the llama loader writes chunks in order)
//   the tiles are permuted on device, batch by batch, through a transient <= 16 MiB cudaMalloc scratch that is freed
//   before returning (not the pool, which would keep it).
//   Consumers: MUL_MAT src0 at n <= 16 -> V3 reads the packed layout directly (n1 included, overriding MF_MIN);
//   every other consumer (MMQ prefill n > 16, MMVQ / fusions, get_rows, cpy, ...) gets a transient natural copy in a
//   pool buffer (ggml_cuda_q8ip_substitute) - correct for every op, cost one extra weight pass on that op only;
//   fusions involving a packed weight are disabled; get_tensor unpacks on the host (CPU fallback / state save read the
//   natural layout); buffer cpy_tensor refuses (generic get/set path); a write into a packed tensor first unpacks it.
//   =1 packs only tensors in GGML_BACKEND_BUFFER_USAGE_WEIGHTS buffers (model weights), =2 any buffer (test-backend-ops).
//   token_embd* is never packed (its consumer is get_rows).
// ---------------------------------------------------------------------------------------------------------------------
static const char q8ip_build_marker[] __attribute__((used)) =
    "ggml-cuda: Q8_0 in-place stage-contiguous load-time repack (GGML_HIP_Q8_MF_INPLACE)";

static int q8ip_mode() {
    static const int m = [] {
        const int v = q8mf_env_int("GGML_HIP_Q8_MF_INPLACE", 0);
        const int variant = q8mf_get_cfg().variant;
        if (v <= 0 || (variant != 3 && variant != 4 && variant != 6)) {
            return 0;
        }
        GGML_LOG_INFO("ggml_cuda_q8ip: in-place Q8_0 stage-contiguous repack enabled (GGML_HIP_Q8_MF_INPLACE=%d, MF=%d)\n",
            v, variant);
        return std::min(v, 2);
    }();
    return m;
}

static std::mutex                               q8ip_mu;
static std::map<const void *, size_t>           q8ip_packed;   // data -> nbytes
static std::map<const void *, size_t>           q8ip_progress; // data -> bytes written in order since offset 0
static std::atomic<int>                         q8ip_count{0};

bool ggml_cuda_q8ip_any() {
    return q8ip_count.load(std::memory_order_relaxed) > 0;
}

bool ggml_cuda_q8ip_is_packed(const ggml_tensor * t) {
    if (!ggml_cuda_q8ip_any() || t == nullptr || t->type != GGML_TYPE_Q8_0) {
        return false;
    }
    std::lock_guard<std::mutex> lk(q8ip_mu);
    return q8ip_packed.count(t->data) != 0;
}

// a view of a packed tensor (e.g. a weight slice) - returns the packed base or nullptr
static const ggml_tensor * q8ip_packed_base(const ggml_tensor * t) {
    if (!ggml_cuda_q8ip_any() || t == nullptr) {
        return nullptr;
    }
    const ggml_tensor * b = t->view_src ? t->view_src : t;
    return ggml_cuda_q8ip_is_packed(b) ? b : nullptr;
}

// AK-BW-27B-DRAFT-Q8IP-PERMUTE: GET_ROWS-only weights (DFlash2 selector_predecessor/successor, rank x n_vocab) cannot read the
// packed layout, so packing them makes every draft graph unpack the whole tensor into a transient copy
// (ggml_cuda_q8ip_substitute; 4.4 x 96 us per draft segment on C18). GGML_HIP_Q8IP_NOPACK_GETROWS=1 leaves them
// natural. Default off (C18 behaviour). Bit-identical: GET_ROWS reads the same natural bytes either way.
static bool q8ip_nopack_getrows() {
    static const bool on = [] {
        const char * v = getenv("GGML_HIP_Q8IP_NOPACK_GETROWS");
        return v != nullptr && atoi(v) != 0;
    }();
    return on;
}

static bool q8ip_shape_ok(const ggml_tensor * t) {
    const q8mf_cfg & cfg = q8mf_get_cfg();
    const int64_t nbytes = (int64_t) ggml_nbytes(t);
    // both DFlash2 GET_ROWS selectors (llama-arch.cpp: SELECTOR_PREV/NEXT -> GGML_OP_GET_ROWS); selector_hidden is a
    // MUL_MAT weight and stays packed
    if (q8ip_nopack_getrows() &&
        (strstr(t->name, "selector_successor") != nullptr || strstr(t->name, "selector_predecessor") != nullptr)) {
        static std::atomic<int> n_skip{0};
        if (n_skip.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8ip: NOPACK_GETROWS leaves %s natural (GET_ROWS consumer)\n", t->name);
        }
        return false;
    }
    return t->type == GGML_TYPE_Q8_0 && t->view_src == nullptr && t->ne[2] == 1 && t->ne[3] == 1 &&
        ggml_is_contiguous(t) && t->ne[0] % 256 == 0 && t->ne[1] % Q8MF_RT == 0 && ((uintptr_t) t->data) % 16 == 0 &&
        nbytes > cfg.min_bytes && nbytes < (int64_t) INT32_MAX - 4096 && strncmp(t->name, "token_embd", 10) != 0;
}

bool ggml_cuda_q8ip_want(const ggml_tensor * t, bool weights_buffer) {
    const int mode = q8ip_mode();
    if (mode == 0 || (mode == 1 && !weights_buffer)) {
        return false;
    }
    return q8ip_shape_ok(t);
}

// natural <-> packed, chunk q of the packed range [tile0 ..): packed q*16 <-> natural row*nb01 + ks*272 + (c%17)*16
template <bool TO_PACKED>
static __global__ void q8ip_permute(const char * __restrict__ src, char * __restrict__ dst, const int64_t nb01, const int S,
                                    const int64_t nchunk) {
    for (int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; q < nchunk; q += (int64_t) gridDim.x * blockDim.x) {
        const int64_t st  = q / Q8MF_NCHUNK;
        const int     c   = (int) (q % Q8MF_NCHUNK);
        const int64_t t   = st / S;
        const int     ks  = (int) (st % S);
        const int64_t row = t * Q8MF_RT + c / 17;
        const int64_t nat = row * nb01 + (int64_t) ks * Q8MF_ROWB + (c % 17) * 16;
        if (TO_PACKED) {
            *(int4 *) (dst + q * 16) = *(const int4 *) (src + nat);
        } else {
            *(int4 *) (dst + nat) = *(const int4 *) (src + q * 16);
        }
    }
}

// in place, through a transient scratch (<= 16 MiB, whole tiles), synchronous on `stream`
static void q8ip_permute_inplace(char * data, const int64_t N, const int64_t K, bool to_packed, cudaStream_t stream) {
    const int64_t nb01  = K / 32 * 34;
    const int     S     = (int) (K / 256);
    const int64_t tileb = Q8MF_RT * nb01;
    const int64_t ntile = N / Q8MF_RT;
    const int64_t bt    = std::max<int64_t>(1, std::min<int64_t>(ntile, (16ll << 20) / tileb));
    char * scratch = nullptr;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMalloc((void **) &scratch, (size_t) (bt * tileb)));
    for (int64_t t0 = 0; t0 < ntile; t0 += bt) {
        const int64_t nt  = std::min(bt, ntile - t0);
        char *        dst = data + t0 * tileb;
        CUDA_CHECK(cudaMemcpyAsync(scratch, dst, (size_t) (nt * tileb), cudaMemcpyDeviceToDevice, stream));
        const int64_t nchunk = nt * tileb / 16;
        const int     grid   = (int) std::min<int64_t>((nchunk + 255) / 256, 65535);
        if (to_packed) {
            q8ip_permute<true><<<grid, 256, 0, stream>>>(scratch, dst, nb01, S, nchunk);
        } else {
            q8ip_permute<false><<<grid, 256, 0, stream>>>(scratch, dst, nb01, S, nchunk);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(scratch));
}

// ---------------------------------------------------------------------------------------------------------------------
// AK-ASM-I8GEMM-PERCHANNEL-GEMV (GGML_HIP_I8PC=1; needs GGML_HIP_Q8_MF=6 and GGML_HIP_Q8_MF_INPLACE=1|2; default OFF).
// "Per-channel" = one scale per OUTPUT ROW (output channel) instead of Q8_0's one f16 scale per 32-weight block. At the
// load-time in-place repack an eligible Q8_0 weight is REQUANTIZED (lossy, measured on 27B by AK-ASM-I8GEMM-PERCHANNEL-KLD:
// KLD 0.001015, dPPL +0.082 %) into a dense int8 layout:
//   s_row = f16_round_up(max_b(|d_b| * max|q_b|) / 127)     (f16-representable, so the unpack back to Q8_0 is exact)
//   q'    = clamp(rint(q * d_b / s_row), -127, 127)
//   bytes [tile = 64 rows][k-stage = 256 k][64 rows][256 B] int8, then N f32 row scales at byte N * K.
// N * (K + 4) <= N * K * 34 / 32 for K >= 64, so it fits in the Q8_0 allocation; the tail bytes are unused.
// The half-tile unit (32 rows x 1 k-stage) is 8192 contiguous B instead of 8704 (1/17 fewer weight bytes, no per-block
// scale load or multiply): q8pc_mul_mat (below q8mf6) reads it at n 1..16 and applies s_row once at the flush.
// Every other consumer (prefill MMQ/cuBLAS, get_rows, cpy, fusions, host get) gets a Q8_0 natural copy with d = s_row in
// every block (ggml_cuda_q8ip_substitute / ggml_cuda_q8ip_get), i.e. the model is consistently the requantized one.
// Shape gate: the 27B verify K dims {5120, 6144, 17408} (GGML_HIP_I8PC_ANYSHAPE=1 lifts it for tests), on top of the
// q8ip gate (N % 64 == 0, K % 256 == 0, contiguous 2-D, < 2 GiB, not token_embd).
// ---------------------------------------------------------------------------------------------------------------------
static const char q8pc_build_marker[] __attribute__((used)) =
    "ggml-cuda: I8PC per-output-channel int8 requant layout + q8pc_mul_mat (GGML_HIP_I8PC)";

#define Q8PC_HB    (32 * 256)          // 8192 B: half-tile unit (32 rows x 256 int8)
#define Q8PC_STAGE (Q8MF_RT * 256)     // 16384 B: one k-stage of a 64-row tile
#define Q8PC_LROW  272                 // LDS row pitch (256 + 16 B pad: rows shift 4 banks, as Q8_0's 272 B rows)

static bool q8pc_on() {
    static const bool on = [] {
        if (q8mf_env_int("GGML_HIP_I8PC", 0) == 0) {
            return false;
        }
        if (q8mf_get_cfg().variant != 6) {
            GGML_LOG_WARN("ggml_cuda_q8pc: GGML_HIP_I8PC=1 ignored: needs GGML_HIP_Q8_MF=6 (and GGML_HIP_Q8_MF_INPLACE)\n");
            return false;
        }
        GGML_LOG_INFO("ggml_cuda_q8pc: I8PC per-channel int8 requant enabled (%s)\n", q8pc_build_marker);
        return true;
    }();
    return on;
}

static std::map<const void *, int> q8pc_set;   // data -> 1, guarded by q8ip_mu; subset of q8ip_packed

static bool q8pc_shape_ok(const ggml_tensor * t) {
    static const bool any = q8mf_env_int("GGML_HIP_I8PC_ANYSHAPE", 0) != 0;
    const int64_t K = t->ne[0];
    return any || K == 5120 || K == 6144 || K == 17408;
}

// base tensor (not a view) packed as I8PC?
static bool q8pc_is_base(const ggml_tensor * b) {
    if (b == nullptr || !ggml_cuda_q8ip_any()) {
        return false;
    }
    std::lock_guard<std::mutex> lk(q8ip_mu);
    return q8pc_set.count(b->data) != 0;
}

// same lookup-by-data semantics as ggml_cuda_q8ip_is_packed (a slice view has other data and gets a natural copy)
static bool ggml_cuda_q8pc_is_packed(const ggml_tensor * t) {
    return t != nullptr && q8pc_is_base(t);
}

static __host__ __device__ __forceinline__ int64_t q8pc_off(const int64_t r, const int64_t k, const int S) {
    return (((r / Q8MF_RT) * S + k / 256) * Q8MF_RT + r % Q8MF_RT) * 256 + k % 256;
}

// natural Q8_0 rows (src, row pitch nb01) -> I8PC int8 (dst, local rows from 0) + f32 row scales (sc). One WG per row.
static __global__ void q8pc_pack(const char * __restrict__ src, char * __restrict__ dst, float * __restrict__ sc,
                                 const int64_t nb01, const int nb, const int S) {
    __shared__ float red[256];
    const int64_t r   = blockIdx.x;
    const int     tid = threadIdx.x;
    const char *  row = src + r * nb01;
    float v = 0.0f;
    for (int b = tid; b < nb; b += blockDim.x) {
        const char * blk = row + (int64_t) b * 34;
        const float  d   = fabsf(__half2float(*(const __half *) blk));
        int m = 0;
        for (int e = 0; e < 32; ++e) {
            m = max(m, abs((int) (int8_t) blk[2 + e]));
        }
        v = fmaxf(v, d * (float) m);
    }
    red[tid] = v;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (tid < o) {
            red[tid] = fmaxf(red[tid], red[tid + o]);
        }
        __syncthreads();
    }
    const float s0 = red[0] / 127.0f;
    __half h = __float2half(s0);
    if (__half2float(h) < s0) {   // round up: |q'| <= 127 without clamping in exact arithmetic
        h = __ushort_as_half((unsigned short) (__half_as_ushort(h) + 1));
    }
    const float s   = __half2float(h);
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int b = tid; b < nb; b += blockDim.x) {
        const char * blk = row + (int64_t) b * 34;
        const float  d   = __half2float(*(const __half *) blk);
        char *       o   = dst + q8pc_off(r, (int64_t) b * 32, S);
        for (int e = 0; e < 32; ++e) {
            const float x = (float) (int8_t) blk[2 + e] * d * inv;
            o[e] = (char) (int) fminf(127.0f, fmaxf(-127.0f, rintf(x)));
        }
    }
    if (tid == 0) {
        sc[r] = s;
    }
}

// I8PC (src, local rows from 0, scales sc) -> natural Q8_0 rows (dst, pitch nb01), d = s_row in every block (exact)
static __global__ void q8pc_unpack(const char * __restrict__ src, const float * __restrict__ sc, char * __restrict__ dst,
                                   const int64_t nb01, const int nb, const int S, const int64_t nrows) {
    const int64_t total = nrows * nb;
    for (int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; q < total; q += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = q / nb;
        const int     b = (int) (q % nb);
        char *        o = dst + r * nb01 + (int64_t) b * 34;
        const char *  i = src + q8pc_off(r, (int64_t) b * 32, S);
        *(__half *) o = __float2half(sc[r]);
        for (int e = 0; e < 32; ++e) {
            o[2 + e] = i[e];
        }
    }
}

// in place natural -> I8PC: ascending tile batches (the I8PC output of batch t ends at (t0+nt)*64*K <= the natural start
// of batch t+1), scales collected in a transient buffer and written at N*K after the last batch (that range still holds
// natural bytes until then). Synchronous on `stream`, transient cudaMalloc freed before return.
static void q8pc_pack_inplace(char * data, const int64_t N, const int64_t K, cudaStream_t stream) {
    const int64_t nb01  = K / 32 * 34;
    const int     S     = (int) (K / 256);
    const int64_t tileb = Q8MF_RT * nb01;
    const int64_t ntile = N / Q8MF_RT;
    const int64_t bt    = std::max<int64_t>(1, std::min<int64_t>(ntile, (16ll << 20) / tileb));
    char *  scratch = nullptr;
    float * sc      = nullptr;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMalloc((void **) &scratch, (size_t) (bt * tileb)));
    CUDA_CHECK(cudaMalloc((void **) &sc, (size_t) N * sizeof(float)));
    for (int64_t t0 = 0; t0 < ntile; t0 += bt) {
        const int64_t nt = std::min(bt, ntile - t0);
        CUDA_CHECK(cudaMemcpyAsync(scratch, data + t0 * tileb, (size_t) (nt * tileb), cudaMemcpyDeviceToDevice, stream));
        q8pc_pack<<<(unsigned) (nt * Q8MF_RT), 256, 0, stream>>>(scratch, data + t0 * Q8MF_RT * K, sc + t0 * Q8MF_RT, nb01,
                                                                (int) (K / 32), S);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpyAsync(data + N * K, sc, (size_t) N * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(scratch));
    CUDA_CHECK(cudaFree(sc));
}

// in place I8PC -> natural Q8_0 (d = s_row): scales saved first, then DESCENDING tile batches (the natural output of batch
// t starts at t0*64*nb01 >= the I8PC end of every lower batch).
static void q8pc_unpack_inplace(char * data, const int64_t N, const int64_t K, cudaStream_t stream) {
    const int64_t nb01  = K / 32 * 34;
    const int     S     = (int) (K / 256);
    const int64_t tileb = Q8MF_RT * nb01;
    const int64_t pcb   = Q8MF_RT * K;
    const int64_t ntile = N / Q8MF_RT;
    const int64_t bt    = std::max<int64_t>(1, std::min<int64_t>(ntile, (16ll << 20) / tileb));
    char *  scratch = nullptr;
    float * sc      = nullptr;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMalloc((void **) &scratch, (size_t) (bt * pcb)));
    CUDA_CHECK(cudaMalloc((void **) &sc, (size_t) N * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(sc, data + N * K, (size_t) N * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    const int64_t nbatch = (ntile + bt - 1) / bt;
    for (int64_t bi = nbatch - 1; bi >= 0; --bi) {
        const int64_t t0 = bi * bt;
        const int64_t nt = std::min(bt, ntile - t0);
        CUDA_CHECK(cudaMemcpyAsync(scratch, data + t0 * pcb, (size_t) (nt * pcb), cudaMemcpyDeviceToDevice, stream));
        const int64_t total = nt * Q8MF_RT * (K / 32);
        const int     grid  = (int) std::min<int64_t>((total + 255) / 256, 65535);
        q8pc_unpack<<<grid, 256, 0, stream>>>(scratch, sc + t0 * Q8MF_RT, data + t0 * tileb, nb01, (int) (K / 32), S,
                                              nt * Q8MF_RT);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(scratch));
    CUDA_CHECK(cudaFree(sc));
}

void ggml_cuda_q8ip_before_write(ggml_tensor * t, size_t offset, size_t size, cudaStream_t stream) {
    if (!ggml_cuda_q8ip_any()) {
        return;
    }
    const ggml_tensor * b = q8ip_packed_base(t);
    if (b == nullptr) {
        return;
    }
    const size_t boff = (size_t) ((const char *) t->data - (const char *) b->data) + offset;
    const bool whole = boff == 0 && size >= ggml_nbytes(b);
    const bool pc = q8pc_is_base(b);
    if (!whole) {   // keep the bytes outside the write: back to natural first
        if (pc) {   // I8PC: the kept bytes come back as the requantized Q8_0 (d = s_row), not the original blocks
            q8pc_unpack_inplace((char *) b->data, b->ne[1], b->ne[0], stream);
        } else {
            q8ip_permute_inplace((char *) b->data, b->ne[1], b->ne[0], false, stream);
        }
    }
    std::lock_guard<std::mutex> lk(q8ip_mu);
    q8pc_set.erase(b->data);
    q8ip_packed.erase(b->data);
    q8ip_count.fetch_sub(1);
}

void ggml_cuda_q8ip_after_write(ggml_tensor * t, size_t offset, size_t size, bool weights_buffer, cudaStream_t stream) {
    if (!ggml_cuda_q8ip_want(t, weights_buffer)) {
        return;
    }
    const size_t nbytes = ggml_nbytes(t);
    {
        std::lock_guard<std::mutex> lk(q8ip_mu);
        size_t & p = q8ip_progress[t->data];
        if (offset == 0) {
            p = 0;
        }
        if (offset != p) {   // out-of-order chunking: never pack this tensor (stays natural, always correct)
            q8ip_progress.erase(t->data);
            return;
        }
        p += size;
        if (p < nbytes) {
            return;
        }
        q8ip_progress.erase(t->data);
    }
    const bool pc = q8pc_on() && q8pc_shape_ok(t);
    if (pc) {   // AK-ASM-I8GEMM-PERCHANNEL-GEMV: requantize to the per-row int8 layout instead of the Q8_0 permutation
        q8pc_pack_inplace((char *) t->data, t->ne[1], t->ne[0], stream);
    } else {
        q8ip_permute_inplace((char *) t->data, t->ne[1], t->ne[0], true, stream);
    }
    std::lock_guard<std::mutex> lk(q8ip_mu);
    if (pc) {
        q8pc_set[t->data] = 1;
        static std::atomic<int64_t> n_pc{0};
        const int64_t n = n_pc.fetch_add(1) + 1;
        if (n == 1 || (n & (n - 1)) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8pc: requantized %lld Q8_0 tensors to I8PC per-channel int8 (last %s %lldx%lld)\n",
                (long long) n, t->name, (long long) t->ne[0], (long long) t->ne[1]);
        }
    }
    if (q8ip_packed.emplace(t->data, nbytes).second) {
        q8ip_count.fetch_add(1);
    }
    static std::atomic<int64_t> n_packed{0};
    static std::atomic<int64_t> b_packed{0};
    const int64_t n = n_packed.fetch_add(1) + 1;
    const int64_t bb = b_packed.fetch_add((int64_t) nbytes) + (int64_t) nbytes;
    if (n == 1 || (n & (n - 1)) == 0) {
        GGML_LOG_INFO("ggml_cuda_q8ip: packed %lld Q8_0 tensors in place (%.1f MiB; last %s %lldx%lld)\n",
            (long long) n, bb / 1048576.0, t->name, (long long) t->ne[0], (long long) t->ne[1]);
    }
}

void ggml_cuda_q8ip_forget_range(const void * base, size_t size) {
    if (!ggml_cuda_q8ip_any()) {
        std::lock_guard<std::mutex> lk(q8ip_mu);
        q8ip_progress.clear();
        return;
    }
    std::lock_guard<std::mutex> lk(q8ip_mu);
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    for (auto it = q8pc_set.lower_bound(lo); it != q8pc_set.end() && (const char *) it->first < hi; ) {
        it = q8pc_set.erase(it);
    }
    for (auto it = q8ip_packed.lower_bound(lo); it != q8ip_packed.end() && (const char *) it->first < hi; ) {
        it = q8ip_packed.erase(it);
        q8ip_count.fetch_sub(1);
    }
    for (auto it = q8ip_progress.lower_bound(lo); it != q8ip_progress.end() && (const char *) it->first < hi; ) {
        it = q8ip_progress.erase(it);
    }
}

// host-side read of a (view of a) packed tensor in the natural layout
bool ggml_cuda_q8ip_get(const ggml_tensor * t, void * data, size_t offset, size_t size) {
    const ggml_tensor * b = q8ip_packed_base(t);
    if (b == nullptr) {
        return false;
    }
    const size_t  nbytes = ggml_nbytes(b);
    const int64_t nb01   = b->nb[1];
    const int     S      = (int) (b->ne[0] / 256);
    std::vector<char> pk(nbytes), nat(nbytes);
    CUDA_CHECK(cudaMemcpy(pk.data(), b->data, nbytes, cudaMemcpyDeviceToHost));
    const int64_t nchunk = q8pc_is_base(b) ? 0 : (int64_t) nbytes / 16;
    if (nchunk == 0) {   // I8PC: natural Q8_0 with d = s_row in every block
        const int64_t N = b->ne[1], K = b->ne[0];
        const float * sc = (const float *) (pk.data() + N * K);
        for (int64_t r = 0; r < N; ++r) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(sc[r]);
            for (int64_t bb = 0; bb < K / 32; ++bb) {
                char * o = nat.data() + r * nb01 + bb * 34;
                memcpy(o, &d, 2);
                memcpy(o + 2, pk.data() + q8pc_off(r, bb * 32, S), 32);
            }
        }
    }
    for (int64_t q = 0; q < nchunk; ++q) {
        const int64_t st = q / Q8MF_NCHUNK;
        const int     c  = (int) (q % Q8MF_NCHUNK);
        const int64_t row = (st / S) * Q8MF_RT + c / 17;
        memcpy(nat.data() + row * nb01 + (st % S) * Q8MF_ROWB + (c % 17) * 16, pk.data() + q * 16, 16);
    }
    const size_t boff = (size_t) ((const char *) t->data - (const char *) b->data) + offset;
    memcpy(data, nat.data() + boff, size);
    return true;
}

bool ggml_cuda_q8ip_mf_handles(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return ggml_cuda_q8ip_is_packed(src0) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= 1 && src1->ne[1] <= Q8MF_MAXM && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0;
}

// any other consumer: point dst->src[i] at a transient natural copy (pool) for the duration of the op
void ggml_cuda_q8ip_substitute(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_q8ip_subst & s) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        ggml_tensor * src = dst->src[i];
        const ggml_tensor * b = q8ip_packed_base(src);
        if (b == nullptr) {
            continue;
        }
        if (dst->op == GGML_OP_MUL_MAT && i == 0 && src == b && ggml_cuda_q8ip_mf_handles(src, dst->src[1], dst)) {
            continue;
        }
        if (dst->op == GGML_OP_MUL_MAT && i == 0 && src == b &&
            ggml_cuda_q8ip_mmq_handles(src, dst->src[1], dst, ggml_cuda_info().devices[ctx.device].cc)) {
            continue;   // DTREE-0-Q8MF-PACKED-CONSUMERS: MMQ reads the packed layout
        }
        if (dst->op == GGML_OP_MUL_MAT && i == 0 && src == b &&
            ggml_cuda_q8ip_blas_handles(src, dst->src[1], dst, ggml_cuda_info().devices[ctx.device].cc)) {
            continue;   // DTREE-0-Q8MF-PACKED-CONSUMERS: packed -> f16 convert + cuBLAS
        }
        static std::atomic<int64_t> n_sub{0};
        if (n_sub.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8ip: op %s (%s) reads packed %s -> transient natural copy\n",
                ggml_op_desc(dst), dst->name, b->name);
        }
        const size_t nbytes = ggml_nbytes(b);
        char * nat = s.buf[i].alloc(ctx.pool(), nbytes);
        const int64_t nchunk = (int64_t) nbytes / 16;
        const int     grid   = (int) std::min<int64_t>((nchunk + 255) / 256, 65535);
        if (q8pc_is_base(b)) {   // I8PC -> requantized natural Q8_0 copy
            const int64_t N = b->ne[1], K = b->ne[0];
            const int64_t total = N * (K / 32);
            const int     g2    = (int) std::min<int64_t>((total + 255) / 256, 65535);
            q8pc_unpack<<<g2, 256, 0, ctx.stream()>>>((const char *) b->data, (const float *) ((const char *) b->data + N * K),
                                                      nat, (int64_t) b->nb[1], (int) (K / 32), (int) (K / 256), N);
        } else {
            q8ip_permute<false><<<grid, 256, 0, ctx.stream()>>>((const char *) b->data, nat, (int64_t) b->nb[1], (int) (b->ne[0] / 256), nchunk);
        }
        CUDA_CHECK(cudaGetLastError());
        s.copy[i]          = *src;
        s.copy[i].data     = nat + ((const char *) src->data - (const char *) b->data);
        s.copy[i].view_src = nullptr;
        s.orig[i]          = src;
        dst->src[i]        = &s.copy[i];
    }
}

void ggml_cuda_q8ip_restore(ggml_tensor * dst, ggml_cuda_q8ip_subst & s) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (s.orig[i]) {
            dst->src[i] = s.orig[i];
            s.orig[i]   = nullptr;
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// DTREE-0-Q8MF-V5 (GGML_HIP_Q8_MF=6, packed layout only; unpacked weights run V3 MF=3): 4 WGs/CU (<= 128 VGPR, 4
// waves/SIMD) persistent balanced-range grid over HALF-TILE units. The in-place packed layout [tile][k-stage][64 rows]
// [272 B] makes rows 0..31 / 32..63 of a stage two contiguous 8704 B blocks, so a unit (half-tile ht, k-stage ks) is one
// contiguous 8704 B range: 544 dwordx4 chunks, thread t loads chunks t, t + 256 (+ t + 512 for t < 32) straight into an
// identical LDS image (no row math). LDS = 8704 B weights + 4608 B q8_1-style int8 activation stage + 2 KiB (dummy commit
// slot / kh-pair reduction) = 15 KiB -> 4 WGs fit the 64 KiB LDS. Wave w = (rg = w & 1: rows 16 rg.., kh = w >> 1: blocks
// 4 kh..4 kh + 3); 2 x mfma_i32_16x16x16i8 per block. P = 2 register stages in flight (118 VGPR, 0 scratch; P = 3/4 hit the
// 128-VGPR cap and spill 32/124 B/lane, so they are not instantiated).
// Tail: no stream-K fixup pass. At a half-tile flush the kh = 1 waves hand their accumulators to kh = 0 through LDS; a
// half-tile owned entirely by this WG is stored, a split one is added with global_atomic_add_f32 into dst, which the
// pre-pass zeroes. Unit order = (ht, ks) flattened, so each WG's weight range is contiguous per half-tile.
// ---------------------------------------------------------------------------------------------------------------------
static const char q8mf6_build_marker[] __attribute__((used)) =
    "ggml-cuda: V5 4-WG/CU half-tile packed Q8_0 small-M kernel (GGML_HIP_Q8_MF=6)";

#define Q8MF6_HB    (32 * Q8MF_ROWB)   // 8704 B: half a packed stage
#define Q8MF6_RED   2048

template <int P, int AUX>
static __global__ void __launch_bounds__(Q8MF_NT, 4)
q8mf6_mul_mat(const char * __restrict__ W, const uint32_t wbytes, const char * __restrict__ act, const uint32_t abytes,
              const int S, const int U, const int G, const int M, float * __restrict__ out, const int64_t out_s1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    __shared__ __attribute__((aligned(16))) char lds[Q8MF6_HB + Q8MF_A1 + Q8MF6_RED];
    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int w    = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int g    = blockIdx.x;
    const int u0   = (int) (((int64_t) g * U) / G);
    const int u1   = (int) (((int64_t) (g + 1) * U) / G);

    constexpr uint32_t OOB = 0x80000000u;
    const uint32_t vb2 = tid < 32 ? (uint32_t) ((tid + 512) * 16) : OOB;
    uint32_t ab[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int q = tid + i * Q8MF_NT;
        const bool ok = q < 256 ? 2 * (q % 8) < M : (q < 288 && 4 * ((q - 256) % 4) < M);
        ab[i] = ok ? (uint32_t) (q * 16) : OOB;
    }
    const q8mf_v4i wrsrc = q8mf_make_rsrc(W, wbytes);
    const q8mf_v4i arsrc = q8mf_make_rsrc(act, abytes);

    struct regs { q8mf_v4i w[3]; q8mf_v4i a[2]; };
    int ui = u0, iht = u0 / S, iks = u0 - (u0 / S) * S;
    auto issue_next = [&](regs & R) {
        const bool live = ui < u1;
        // unit (ht, ks) at ((ht >> 1) * S + ks) * 17408 + (ht & 1) * 8704
        const uint32_t sw = live ? ((uint32_t) ((iht >> 1) * S + iks) * Q8MF_WSTAGE + (uint32_t) (iht & 1) * Q8MF6_HB) : wbytes;
        const uint32_t sa = live ? (uint32_t) (iks * Q8MF_A1) : abytes;
        R.w[0] = q8mf_buf_x4(wrsrc, (int) (sw + tid * 16), 0, AUX);
        R.w[1] = q8mf_buf_x4(wrsrc, (int) (sw + (tid + 256) * 16), 0, AUX);
        R.w[2] = q8mf_buf_x4(wrsrc, (int) (sw + vb2), 0, AUX);
        R.a[0] = q8mf_buf_x4(arsrc, (int) (sa + ab[0]), 0, 0);
        R.a[1] = q8mf_buf_x4(arsrc, (int) (sa + ab[1]), 0, 0);
        ++ui;
        if (++iks == S) {
            iks = 0;
            ++iht;
        }
    };
    char * red   = lds + Q8MF6_HB + Q8MF_A1;   // dummy commit slot (1 KiB) / kh reduction (2 KiB), never live together
    auto commit = [&](const regs & R) {
        *(q8mf_v4i *) (lds + tid * 16)         = R.w[0];
        *(q8mf_v4i *) (lds + (tid + 256) * 16) = R.w[1];
        *(q8mf_v4i *) (tid < 32 ? lds + (tid + 512) * 16 : red + lane * 16) = R.w[2];
        *(q8mf_v4i *) (lds + Q8MF6_HB + tid * 16) = R.a[0];
        *(q8mf_v4i *) (tid < 32 ? lds + Q8MF6_HB + (tid + 256) * 16 : red + lane * 16) = R.a[1];
    };

    const int rg = w & 1;
    const int kh = w >> 1;
    const int j  = lane & 15;
    const int gq = lane >> 4;
    const int c0 = 4 * gq;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto compute = [&]() {
        const char * lw = lds + (16 * rg + j) * Q8MF_ROWB;
        const char * la = lds + Q8MF6_HB;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            const int i = 4 * kh + ii;   // parity of i == parity of ii
            int w0, w1;
            if (ii % 2 == 0) {
                const int * p = (const int *) (lw + 34 * i + 8 * gq);
                const int d0 = p[0], d1 = p[1], d2 = p[2];
                w0 = (int) __builtin_amdgcn_alignbyte((uint32_t) d1, (uint32_t) d0, 2);
                w1 = (int) __builtin_amdgcn_alignbyte((uint32_t) d2, (uint32_t) d1, 2);
            } else {
                const int * p = (const int *) (lw + 34 * i + 2 + 8 * gq);
                w0 = p[0];
                w1 = p[1];
            }
            const float  dw = __half2float(*(const __half *) (lw + 34 * i));
            const int2   a  = *(const int2 *) (la + i * 512 + lane * 8);
            const float4 da = *(const float4 *) (la + Q8MF_KS * 512 + i * 64 + gq * 16);
            const q8mf_v4i z = {0, 0, 0, 0};
            q8mf_v4i s = __builtin_amdgcn_mfma_i32_16x16x16i8(a.x, w0, z, 0, 0, 0);
            s          = __builtin_amdgcn_mfma_i32_16x16x16i8(a.y, w1, s, 0, 0, 0);
            acc[0] += (float) s[0] * (dw * da.x);
            acc[1] += (float) s[1] * (dw * da.y);
            acc[2] += (float) s[2] * (dw * da.z);
            acc[3] += (float) s[3] * (dw * da.w);
        }
    };

    int cht = u0 / S, cks = u0 - (u0 / S) * S;
    auto end_of_unit = [&](const int uc) {
        if (cks == S - 1 || uc == u1 - 1) {   // uniform
            float * rf = (float *) red;
            if (kh == 1) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    rf[(rg * 4 + r) * 64 + lane] = acc[r];
                }
            }
            Q8MF_LDS_BARRIER();
            if (kh == 0) {
                const int  ts   = cht * S;
                const bool full = ts >= u0 && ts + S <= u1;
                const int  row  = (cht >> 1) * Q8MF_RT + (cht & 1) * 32 + 16 * rg + j;   // packed => N % 64 == 0
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float v = acc[r] + rf[(rg * 4 + r) * 64 + lane];
                    if (c0 + r < M) {
                        float * o = out + (int64_t) (c0 + r) * out_s1 + row;
                        if (full) {
                            *o = v;
                        } else {
                            unsafeAtomicAdd(o, v);
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[r] = 0.0f;
            }
        }
        if (++cks == S) {
            cks = 0;
            ++cht;
        }
    };

    regs R[P];
#pragma unroll
    for (int jj = 0; jj < P; ++jj) {
        issue_next(R[jj]);
        asm volatile("" ::: "memory");
    }
    int u = u0;
    do {
#pragma unroll
        for (int jj = 0; jj < P; ++jj) {
            commit(R[jj]);
            issue_next(R[jj]);
            Q8MF_LDS_BARRIER();
            if (u + jj < u1) {
                compute();
                end_of_unit(u + jj);
            }
            Q8MF_LDS_BARRIER();
        }
        u += P;
    } while (u < u1);
#else
    GGML_UNUSED(W); GGML_UNUSED(wbytes); GGML_UNUSED(act); GGML_UNUSED(abytes); GGML_UNUSED(S); GGML_UNUSED(U);
    GGML_UNUSED(G); GGML_UNUSED(M); GGML_UNUSED(out); GGML_UNUSED(out_s1);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

static void ggml_cuda_mul_mat_q8_mf6(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8mf_cfg & cfg = q8mf_get_cfg();
    constexpr int p = 2;
    {
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8_mf: V5 4-WG/CU half-tile packed path ran (call 1, GGML_HIP_Q8_MF=6, P=%d, slc=%d, %s)\n",
                p, cfg.slc, q8mf6_build_marker);
        }
    }
    cudaStream_t stream = ctx.stream();
    const int     N      = (int) src0->ne[1];
    const int     M      = (int) src1->ne[1];
    const int     nblk   = (int) (src0->ne[0] / 32);
    const int64_t s1     = src1->nb[1] / sizeof(float);
    const int64_t dst_s1 = dst->nb[1]  / sizeof(float);
    const int     S      = nblk / Q8MF_KS;
    const int     U      = (N / Q8MF_RT) * 2 * S;
    const int     ncu    = ggml_cuda_info().devices[ctx.device].nsm;
    GGML_ASSERT(M >= 1 && M <= Q8MF_MAXM && nblk % Q8MF_KS == 0 && N % Q8MF_RT == 0);
    const int G = std::max(1, std::min(cfg.wg > 0 ? cfg.wg : 4 * ncu, U));
    const size_t abytes = (size_t) S * Q8MF_A1;
    ggml_cuda_pool_alloc<char> act(ctx.pool(), abytes);
    float * d = (float *) dst->data;
    {
        const dim3 grid((nblk + 7) / 8, Q8MF_MAXM, 1);
        q8mf_pre_i8<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, M, act.get(), nullptr, 0, d, dst_s1, N);
    }
    const char *   W      = (const char *) src0->data;
    const uint32_t wbytes = (uint32_t) ggml_nbytes(src0);
#define Q8MF6_L(PP, AA) q8mf6_mul_mat<PP, AA><<<G, Q8MF_NT, 0, stream>>>(W, wbytes, act.get(), (uint32_t) abytes, S, U, G, M, d, dst_s1)
    if (cfg.slc) {
        Q8MF6_L(2, 2);
    } else {
        Q8MF6_L(2, 0);
    }
#undef Q8MF6_L
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// AK-ASM-I8GEMM-PERCHANNEL-GEMV: q8pc_mul_mat = q8mf6 (V5 4-WG/CU persistent half-tile grid, P = 2 register stages,
// 2 x mfma_i32_16x16x16i8 per 32-k block, kh-pair LDS reduction, atomic tail on split half-tiles) over the I8PC layout.
// Differences from q8mf6: a unit is 8192 B = 512 dwordx4 (thread t loads chunks t, t + 256; no third chunk), the weight
// image is written to LDS at a 272 B row pitch (chunk q -> row q >> 4, byte 16 (q & 15)) so the 16 rows a wave reads stay
// 4 banks apart, the inner loop has no per-block f16 scale load or multiply (acc += s * da), and the row scale is applied
// once at the flush (s_row from the f32 table at W + N*K). Same activation pre-pass (q8mf_pre_i8) and dst zeroing.
// ---------------------------------------------------------------------------------------------------------------------
template <int P, int AUX>
static __global__ void __launch_bounds__(Q8MF_NT, 4)
q8pc_mul_mat(const char * __restrict__ W, const uint32_t wbytes, const float * __restrict__ wsc,
             const char * __restrict__ act, const uint32_t abytes,
             const int S, const int U, const int G, const int M, float * __restrict__ out, const int64_t out_s1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    __shared__ __attribute__((aligned(16))) char lds[32 * Q8PC_LROW + Q8MF_A1 + Q8MF6_RED];
    constexpr int LW = 32 * Q8PC_LROW;   // 8704 B
    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int w    = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int g    = blockIdx.x;
    const int u0   = (int) (((int64_t) g * U) / G);
    const int u1   = (int) (((int64_t) (g + 1) * U) / G);

    constexpr uint32_t OOB = 0x80000000u;
    uint32_t ab[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int q = tid + i * Q8MF_NT;
        const bool ok = q < 256 ? 2 * (q % 8) < M : (q < 288 && 4 * ((q - 256) % 4) < M);
        ab[i] = ok ? (uint32_t) (q * 16) : OOB;
    }
    const q8mf_v4i wrsrc = q8mf_make_rsrc(W, wbytes);
    const q8mf_v4i arsrc = q8mf_make_rsrc(act, abytes);

    struct regs { q8mf_v4i w[2]; q8mf_v4i a[2]; };
    int ui = u0, iht = u0 / S, iks = u0 - (u0 / S) * S;
    auto issue_next = [&](regs & R) {
        const bool live = ui < u1;
        // unit (ht, ks) at ((ht >> 1) * S + ks) * 16384 + (ht & 1) * 8192
        const uint32_t sw = live ? ((uint32_t) ((iht >> 1) * S + iks) * Q8PC_STAGE + (uint32_t) (iht & 1) * Q8PC_HB) : wbytes;
        const uint32_t sa = live ? (uint32_t) (iks * Q8MF_A1) : abytes;
        R.w[0] = q8mf_buf_x4(wrsrc, (int) (sw + tid * 16), 0, AUX);
        R.w[1] = q8mf_buf_x4(wrsrc, (int) (sw + (tid + 256) * 16), 0, AUX);
        R.a[0] = q8mf_buf_x4(arsrc, (int) (sa + ab[0]), 0, 0);
        R.a[1] = q8mf_buf_x4(arsrc, (int) (sa + ab[1]), 0, 0);
        ++ui;
        if (++iks == S) {
            iks = 0;
            ++iht;
        }
    };
    char * red = lds + LW + Q8MF_A1;   // dummy commit slot / kh reduction, never live together
    // chunk q = row (q >> 4), 16 B column (q & 15); tid and tid + 256 are rows tid >> 4 and 16 + (tid >> 4), same column
    const int wl0 = (tid >> 4) * Q8PC_LROW + (tid & 15) * 16;
    auto commit = [&](const regs & R) {
        *(q8mf_v4i *) (lds + wl0)                  = R.w[0];
        *(q8mf_v4i *) (lds + wl0 + 16 * Q8PC_LROW) = R.w[1];
        *(q8mf_v4i *) (lds + LW + tid * 16) = R.a[0];
        *(q8mf_v4i *) (tid < 32 ? lds + LW + (tid + 256) * 16 : red + lane * 16) = R.a[1];
    };

    const int rg = w & 1;
    const int kh = w >> 1;
    const int j  = lane & 15;
    const int gq = lane >> 4;
    const int c0 = 4 * gq;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto compute = [&]() {
        const char * lw = lds + (16 * rg + j) * Q8PC_LROW + 8 * gq;
        const char * la = lds + LW;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            const int    i  = 4 * kh + ii;
            const int2   wv = *(const int2 *) (lw + 32 * i);
            const int2   a  = *(const int2 *) (la + i * 512 + lane * 8);
            const float4 da = *(const float4 *) (la + Q8MF_KS * 512 + i * 64 + gq * 16);
            const q8mf_v4i z = {0, 0, 0, 0};
            q8mf_v4i s = __builtin_amdgcn_mfma_i32_16x16x16i8(a.x, wv.x, z, 0, 0, 0);
            s          = __builtin_amdgcn_mfma_i32_16x16x16i8(a.y, wv.y, s, 0, 0, 0);
            acc[0] += (float) s[0] * da.x;
            acc[1] += (float) s[1] * da.y;
            acc[2] += (float) s[2] * da.z;
            acc[3] += (float) s[3] * da.w;
        }
    };

    int cht = u0 / S, cks = u0 - (u0 / S) * S;
    auto end_of_unit = [&](const int uc) {
        if (cks == S - 1 || uc == u1 - 1) {   // uniform
            float * rf = (float *) red;
            if (kh == 1) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    rf[(rg * 4 + r) * 64 + lane] = acc[r];
                }
            }
            Q8MF_LDS_BARRIER();
            if (kh == 0) {
                const int   ts   = cht * S;
                const bool  full = ts >= u0 && ts + S <= u1;
                const int   row  = (cht >> 1) * Q8MF_RT + (cht & 1) * 32 + 16 * rg + j;
                const float srow = wsc[row];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float v = (acc[r] + rf[(rg * 4 + r) * 64 + lane]) * srow;
                    if (c0 + r < M) {
                        float * o = out + (int64_t) (c0 + r) * out_s1 + row;
                        if (full) {
                            *o = v;
                        } else {
                            unsafeAtomicAdd(o, v);
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[r] = 0.0f;
            }
        }
        if (++cks == S) {
            cks = 0;
            ++cht;
        }
    };

    regs R[P];
#pragma unroll
    for (int jj = 0; jj < P; ++jj) {
        issue_next(R[jj]);
        asm volatile("" ::: "memory");
    }
    int u = u0;
    do {
#pragma unroll
        for (int jj = 0; jj < P; ++jj) {
            commit(R[jj]);
            issue_next(R[jj]);
            Q8MF_LDS_BARRIER();
            if (u + jj < u1) {
                compute();
                end_of_unit(u + jj);
            }
            Q8MF_LDS_BARRIER();
        }
        u += P;
    } while (u < u1);
#else
    GGML_UNUSED(W); GGML_UNUSED(wbytes); GGML_UNUSED(wsc); GGML_UNUSED(act); GGML_UNUSED(abytes); GGML_UNUSED(S);
    GGML_UNUSED(U); GGML_UNUSED(G); GGML_UNUSED(M); GGML_UNUSED(out); GGML_UNUSED(out_s1);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

static void ggml_cuda_mul_mat_q8_pc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8mf_cfg & cfg = q8mf_get_cfg();
    {
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            GGML_LOG_INFO("ggml_cuda_q8pc: I8PC per-channel int8 packed path ran (call 1, P=2, slc=%d, %s)\n", cfg.slc,
                q8pc_build_marker);
        }
    }
    cudaStream_t stream = ctx.stream();
    const int     N      = (int) src0->ne[1];
    const int     M      = (int) src1->ne[1];
    const int     nblk   = (int) (src0->ne[0] / 32);
    const int64_t s1     = src1->nb[1] / sizeof(float);
    const int64_t dst_s1 = dst->nb[1]  / sizeof(float);
    const int     S      = nblk / Q8MF_KS;
    const int     U      = (N / Q8MF_RT) * 2 * S;
    const int     ncu    = ggml_cuda_info().devices[ctx.device].nsm;
    GGML_ASSERT(M >= 1 && M <= Q8MF_MAXM && nblk % Q8MF_KS == 0 && N % Q8MF_RT == 0);
    const int G = std::max(1, std::min(cfg.wg > 0 ? cfg.wg : 4 * ncu, U));
    const size_t abytes = (size_t) S * Q8MF_A1;
    ggml_cuda_pool_alloc<char> act(ctx.pool(), abytes);
    float * d = (float *) dst->data;
    {
        const dim3 grid((nblk + 7) / 8, Q8MF_MAXM, 1);
        q8mf_pre_i8<<<grid, 256, 0, stream>>>((const float *) src1->data, s1, nblk, M, act.get(), nullptr, 0, d, dst_s1, N);
    }
    const char *   W      = (const char *) src0->data;
    const uint32_t wbytes = (uint32_t) ((int64_t) N * src0->ne[0]);   // int8 region only; the scale table follows it
    const float *  wsc    = (const float *) (W + (int64_t) N * src0->ne[0]);
    if (cfg.slc) {
        q8pc_mul_mat<2, 2><<<G, Q8MF_NT, 0, stream>>>(W, wbytes, wsc, act.get(), (uint32_t) abytes, S, U, G, M, d, dst_s1);
    } else {
        q8pc_mul_mat<2, 0><<<G, Q8MF_NT, 0, stream>>>(W, wbytes, wsc, act.get(), (uint32_t) abytes, S, U, G, M, d, dst_s1);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// DTREE-0-Q8MF-PACKED-CONSUMERS: consumers that read the in-place packed layout directly (no natural copy).
// (a) q8ip_mmvq: n1 (GGML_HIP_Q8IP_MMVQ_MAX=0 -> V3) Q8_0 x q8_1 matvec. A WG (256 threads) owns Q8IPV_R = 8
//     consecutive rows of one 64-row tile; in the packed layout those rows' segment of a k-stage is ONE contiguous,
//     16 B aligned 8 x 272 = 2176 B run (136 dwordx4), so each iteration streams Q8IPV_U = 8 stages = 8 contiguous runs
//     with fully used dwordx4 loads (next group prefetched into VGPRs while the current one is consumed) into LDS together with the group's 2304 B of q8_1 activations (no per-block global y loads), then 2 threads per Q8_0 block (16 quants each, 4 dp4a per column) consume
//     them. 17 KB LDS, N/8 WGs (17408 rows -> 2176 WGs) -> high occupancy, no stream-K / atomics.
// (b) MMQ prefill (n > 16): ggml_cuda_mul_mat_q is called with a NEGATIVE row stride (-K/32) for a packed weight; the
//     Q8_0 tile loader decodes (row, block) from the natural index and reads the packed address (in-stage blocks stay
//     contiguous because MMQ's 8-block k-iteration == one packed stage). See mmq-load-tiles.cuh.
// (c) cuBLAS prefill (n > 16 where MMQ is not chosen - the 27B default on gfx90a): the h20 q8_0 -> f16 src0 convert reads
//     the packed layout directly (h20_q8_0_packed_to_f16_vec, convert.cu) - one pass instead of natural copy + convert.
static const char q8ipc_build_marker[] __attribute__((used)) =
    "ggml-cuda: Q8_0 packed-layout consumers MMVQ+MMQ (DTREE-0-Q8MF-PACKED-CONSUMERS)";

#define Q8IPV_R  8
#define Q8IPV_U  8                             // stages per iteration (8 x 2176 B = 17 KB per WG in flight)
#define Q8IPV_NT 256
#define Q8IPV_CH (Q8IPV_R * Q8MF_ROWB / 16)   // 136 dwordx4 per (stage, 8-row group)
#define Q8IPV_NL ((Q8IPV_U * Q8IPV_CH + Q8IPV_NT - 1) / Q8IPV_NT)   // 5 dwordx4 per thread per iteration
#define Q8IPV_YC (Q8IPV_U * Q8MF_KS * 36 / 16)                       // 144 dwordx4 of q8_1 activations per iteration

// aligned dword read of a 2-byte aligned LDS int pair (Q8_0 qs): two ds_read_b32 + v_alignbyte
static __device__ __forceinline__ int q8ipv_ld(const char * p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t * w = (const uint32_t *) (a & ~(uintptr_t) 3);
    return (int) __builtin_amdgcn_alignbyte(w[1], w[0], (uint32_t) (a & 3));
}

template <int NC>
static __global__ void __launch_bounds__(Q8IPV_NT) q8ip_mmvq(const char * __restrict__ W, const block_q8_1 * __restrict__ y,
        const int S, const int sy, float * __restrict__ dst, const int dst_s1) {
    static_assert(NC == 1, "activations staged in LDS for one column");
    __shared__ int4  lds[Q8IPV_U * Q8IPV_CH + 1];   // +1: the alignbyte read of the last block may touch 4 B past it
    __shared__ int4  ldy[Q8IPV_YC];                  // the group's 64 q8_1 activation blocks (2304 B, contiguous)
    const int ybytes = sy * (int) sizeof(block_q8_1);
    __shared__ float red[NC][Q8IPV_R];
    const int tid  = threadIdx.x;
    const int row0 = blockIdx.x * Q8IPV_R;
    const char * base = W + (int64_t) (row0 >> 6) * S * Q8MF_WSTAGE + (row0 & 63) * Q8MF_ROWB;
    const int r = (tid & 127) >> 4;   // row of the group
    const int b = (tid & 15) >> 1;    // block of the stage row
    const int h = tid & 1;            // half block (16 quants)
    const int g = tid >> 7;           // stage parity
    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = 0.0f;
    }
    // dwordx4 i of a thread covers chunk q = tid + 256 i of the group: (stage u, chunk c) precomputed once (S-independent)
#define Q8IPV_OFF(i) ((tid + (i) * Q8IPV_NT) / Q8IPV_CH * Q8MF_WSTAGE + (tid + (i) * Q8IPV_NT) % Q8IPV_CH * 16)
    const int o0 = Q8IPV_OFF(0), o1 = Q8IPV_OFF(1), o2 = Q8IPV_OFF(2), o3 = Q8IPV_OFF(3), o4 = Q8IPV_OFF(4);
#undef Q8IPV_OFF
    static_assert(Q8IPV_NL == 5, "Q8IPV_ISSUE is unrolled by hand for 5 dwordx4 per thread");
#define Q8IPV_LD1(i, src_, nu_) p##i = *(const int4 *) ((src_) + (tid + (i) * Q8IPV_NT < (nu_) * Q8IPV_CH ? o##i : 0))
#define Q8IPV_ISSUE(ks0_) do {                                                             \
        const int nu_ = min(Q8IPV_U, S - (ks0_));   /* chunks of stages >= nu re-read stage 0 */ \
        const char * src_ = base + (int64_t) (ks0_) * Q8MF_WSTAGE;                         \
        Q8IPV_LD1(0, src_, nu_); Q8IPV_LD1(1, src_, nu_); Q8IPV_LD1(2, src_, nu_);         \
        Q8IPV_LD1(3, src_, nu_); Q8IPV_LD1(4, src_, nu_);                                  \
        const int yo_ = (ks0_) * Q8MF_KS * (int) sizeof(block_q8_1) + tid * 16;            \
        if (tid < Q8IPV_YC && yo_ + 16 <= ybytes) {                                        \
            py = *(const int4 *) ((const char *) y + yo_);                                 \
        }                                                                                  \
    } while (0)
    int4 p0, p1, p2, p3, p4, py = make_int4(0, 0, 0, 0);
    Q8IPV_ISSUE(0);
    for (int ks0 = 0; ks0 < S; ks0 += Q8IPV_U) {
        const int nu = min(Q8IPV_U, S - ks0);
        // slots of stages >= nu hold junk and are never consumed; 1088 = 4 x 256 + 64 -> only the 5th is partial
        lds[tid]                = p0;
        lds[tid + Q8IPV_NT]     = p1;
        lds[tid + 2 * Q8IPV_NT] = p2;
        lds[tid + 3 * Q8IPV_NT] = p3;
        if (tid + 4 * Q8IPV_NT < Q8IPV_U * Q8IPV_CH) {
            lds[tid + 4 * Q8IPV_NT] = p4;
        }
        if (tid < Q8IPV_YC) {
            ldy[tid] = py;
        }
        __syncthreads();
        if (ks0 + Q8IPV_U < S) {
            Q8IPV_ISSUE(ks0 + Q8IPV_U);   // next group in flight while this one is consumed
        }
#pragma unroll
        for (int u0 = 0; u0 < Q8IPV_U; u0 += 2) {
            const int u = u0 + g;
            if (u < nu) {
                const char * blk = (const char *) lds + u * (Q8IPV_CH * 16) + r * Q8MF_ROWB + b * 34;
                const float dx = __half2float(*(const half *) blk);
                const char * qx = blk + 2 + h * 16;
                int xi[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    xi[k] = q8ipv_ld(qx + 4*k);
                }
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    const block_q8_1 * yb = (const block_q8_1 *) ldy + u * Q8MF_KS + b;
                    const int * yq = (const int *) yb->qs + h * 4;   // 36 B blocks: dword aligned only
                    int s = 0;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        s = ggml_cuda_dp4a(xi[k], yq[k], s);
                    }
                    acc[j] += dx * __low2float(yb->ds) * (float) s;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        acc[j] = warp_reduce_sum<16>(acc[j]);
    }
    if (g == 1 && (tid & 15) == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            red[j][r] = acc[j];
        }
    }
    __syncthreads();
    if (g == 0 && (tid & 15) == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            dst[j * dst_s1 + row0 + r] = acc[j] + red[j][r];
        }
    }
#undef Q8IPV_ISSUE
#undef Q8IPV_LD1
}

static int q8ip_mmvq_max() {
    static const int v = std::max(0, std::min(1, q8mf_env_int("GGML_HIP_Q8IP_MMVQ_MAX", 0)));   // n1 only; default 0 (C17: packed n1 MMVQ loses on M=1)
    return v;
}

static void q8ip_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static std::atomic<int64_t> n_calls{0};
    if (n_calls.fetch_add(1) == 0) {
        GGML_LOG_INFO("ggml_cuda_q8ip: packed-layout MMVQ ran (call 1, n=%lld, GGML_HIP_Q8IP_MMVQ_MAX=%d)\n",
            (long long) src1->ne[1], q8ip_mmvq_max());
    }
    cudaStream_t stream = ctx.stream();
    const int64_t K   = src0->ne[0];
    const int     N   = (int) src0->ne[1];
    const int     M   = (int) src1->ne[1];
    const int64_t Kp  = GGML_PAD(K, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> yq(ctx.pool(), (size_t) M * Kp / QK8_1 * sizeof(block_q8_1));
    const int64_t s11 = src1->nb[1] / sizeof(float);
    quantize_row_q8_1_cuda((const float *) src1->data, nullptr, yq.get(), GGML_TYPE_Q8_0, K, s11, s11 * M, s11 * M, Kp, M, 1, 1, stream);
    const int S      = (int) (K / 256);
    const int sy     = (int) (Kp / QK8_1);
    const int dst_s1 = (int) (dst->nb[1] / sizeof(float));
    const block_q8_1 * y = (const block_q8_1 *) yq.get();
    float * d = (float *) dst->data;
    const char * W = (const char *) src0->data;
    const int grid = N / Q8IPV_R;
    switch (M) {
        case 1: q8ip_mmvq<1><<<grid, Q8IPV_NT, 0, stream>>>(W, y, S, sy, d, dst_s1); break;
        default: GGML_ABORT("q8ip_mmvq: bad n %d", M);
    }
    CUDA_CHECK(cudaGetLastError());
}

// prefill (n > Q8MF_MAXM): MMQ reads the packed layout (GGML_HIP_Q8IP_MMQ=0 -> transient natural copy as before)
// prefill on the cuBLAS route (this tree's default for Q8_0 at large n on gfx90a): packed -> natural f16 in one pass
bool ggml_cuda_q8ip_blas_handles(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    static const bool on = q8mf_env_int("GGML_HIP_Q8IP_BLAS", 1) != 0;
    return on && ggml_cuda_h20_enabled() && getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE") == nullptr &&
        ggml_cuda_q8ip_is_packed(src0) && !ggml_cuda_q8pc_is_packed(src0) && src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && dst->op_params[0] != GGML_PREC_F32 && fast_fp16_hardware_available(cc) && ggml_is_contiguous(dst) &&
        src1->ne[1] > Q8MF_MAXM && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        !ggml_cuda_q8ip_mmq_handles(src0, src1, dst, cc);
}

bool ggml_cuda_q8ip_mmq_handles(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    static const bool on = q8mf_env_int("GGML_HIP_Q8IP_MMQ", 1) != 0;
    // I8PC tensors are not Q8_0-packed: prefill gets the requantized natural copy (ggml_cuda_q8ip_substitute)
    return on && ggml_cuda_q8ip_is_packed(src0) && !ggml_cuda_q8pc_is_packed(src0) && src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && src1->ne[1] > Q8MF_MAXM && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_cuda_should_use_mmq(GGML_TYPE_Q8_0, cc, src1->ne[1], /*n_experts =*/ 0);
}

void ggml_cuda_mul_mat_q8_mf(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q8mf_cfg & cfg0 = q8mf_get_cfg();
    if (ggml_cuda_q8pc_is_packed(src0)) {   // AK-ASM-I8GEMM-PERCHANNEL-GEMV: I8PC layout, n 1..16 (mf_handles)
        ggml_cuda_mul_mat_q8_pc(ctx, src0, src1, dst);
        return;
    }
    if (src1->ne[1] <= q8ip_mmvq_max() && ggml_cuda_q8ip_is_packed(src0)) {   // DTREE-0-Q8MF-PACKED-CONSUMERS
        q8ip_mul_mat_vec(ctx, src0, src1, dst);
        return;
    }
    if (cfg0.variant == 6 && ggml_cuda_q8ip_is_packed(src0)) {
        ggml_cuda_mul_mat_q8_mf6(ctx, src0, src1, dst);
        return;
    }
    q8mf_cfg cfg = cfg0;
    if (cfg.variant == 6) {   // V5 reads only the packed layout; natural-layout weights run V3 (MF=3)
        cfg.variant = 3;
    }
    if (cfg.variant == 5) {
        ggml_cuda_mul_mat_q8_mf5(ctx, src0, src1, dst);
        return;
    }
    {
        // Path witness (gate packets grep it; also a content marker of the build).
        static std::atomic<int64_t> n_calls{0};
        if (n_calls.fetch_add(1) == 0) {
            if (cfg.variant <= 2) {
                GGML_LOG_INFO("ggml_cuda_q8_mf: V2 stream-K small-M Q8_0 path ran (call 1, GGML_HIP_Q8_MF=%d, %s)\n",
                    cfg.variant, cfg.variant == 1 ? "int8 mfma_i32_16x16x16i8" : "f16 mfma_f32_32x32x8f16 in-register dequant");
            } else {
                GGML_LOG_INFO("ggml_cuda_q8_mf: V3 2-WG/CU 3-stage stream-K small-M Q8_0 path ran (call 1, GGML_HIP_Q8_MF=%d, %s, "
                    "slc=%d, repack=%d)\n", cfg.variant, cfg.variant == 3 ? "int8 mfma_i32_16x16x16i8" : "f16 mfma_f32_16x16x16f16 in-register dequant",
                    cfg.slc, cfg.repack);
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
    uint32_t       wbytes = (uint32_t) ggml_nbytes(src0);   // = N * nb01 (contiguous): rows >= N are out of range
    // V3 addressing: chunk q of a stage at (q/17)*rowstride + (q%17)*16, stage (tile, ks) at tile*tstride + ks*kstride
    int      rowstride = nb01;
    int      kstride   = Q8MF_ROWB;
    uint32_t tstride   = (uint32_t) Q8MF_RT * (uint32_t) nb01;
    ggml_cuda_pool_alloc<char> ptmp;
    if (v3 && ggml_cuda_q8ip_is_packed(src0)) {   // DTREE-0-Q8MF-REPACK-INPLACE: weight already packed in place
        rowstride = Q8MF_ROWB;
        kstride   = Q8MF_WSTAGE;
        tstride   = (uint32_t) S * Q8MF_WSTAGE;
    } else if (v3 && cfg.repack > 0) {
        size_t pbytes = 0;
        W         = q8mf_get_packed(ctx, src0, cfg.repack, ptmp, pbytes);
        wbytes    = (uint32_t) pbytes;
        rowstride = Q8MF_ROWB;
        kstride   = Q8MF_WSTAGE;
        tstride   = (uint32_t) S * Q8MF_WSTAGE;
    }
    float *        d      = (float *) dst->data;
    switch (cfg.variant * 2 + cfg.slc) {
        case 2: case 3:
            q8mf2_mul_mat<1><<<G, Q8MF_NT, 0, stream>>>(W, nb01, N, wbytes, act.get(), (uint32_t) abytes, S, T, L, M, d, dst_s1, part.get(), cnt.get());
            break;
        case 4: case 5:
            q8mf2_mul_mat<2><<<G, Q8MF_NT, 0, stream>>>(W, nb01, N, wbytes, act.get(), (uint32_t) abytes, S, T, L, M, d, dst_s1, part.get(), cnt.get());
            break;
#define Q8MF3_CASE(c, v, aux) case c: \
            q8mf3_mul_mat<v, aux><<<G, Q8MF_NT, 0, stream>>>(W, rowstride, kstride, tstride, wbytes, act.get(), (uint32_t) abytes, S, T, G, M, N, d, dst_s1, part.get(), cnt.get()); \
            break;
        Q8MF3_CASE(6, 3, 0) Q8MF3_CASE(7, 3, 2) Q8MF3_CASE(8, 4, 0) Q8MF3_CASE(9, 4, 2)
#undef Q8MF3_CASE
        default: GGML_ABORT("q8_mf: bad variant %d", cfg.variant);
    }
    CUDA_CHECK(cudaGetLastError());
}
