// AK-MMID-MF (speculative-decoding.md): expert-grouped, bandwidth-first Q8_0 MUL_MAT_ID for few-token MoE verify on gfx90a
// (MI210). The MoE counterpart of DTREE-0-Q8MF (V5I, mmq-q8-mf.cu). OPT-IN: GGML_HIP_MMID_MF=1, default off.
//
// Problem (Qwen3.6-35B-A3B, 256 experts, top-8, hidden 2048, expert ffn 512, all experts Q8_0): a DF2/MTP verify step at
// np1..4 routes n = 3..12 tokens, i.e. U(n) = 256 (1 - (1 - 8/256)^n) unique experts at ~1.0-1.3 tokens per expert. The op is
// a weight stream whose bytes scale with U(n) (one expert = 3 x 2048 x 512 x 34/32 B per layer), so MFMA tile reuse buys
// ~nothing; what moves it is HBM efficiency (MMVQ_MMID reaches ~640 GB/s at n=1, split16 ~690 GB/s at n=9..16, vs ~1.6 TB/s
// peak and ~1.2 TB/s for V5I on dense Q8_0). MMVQ_MMID also streams each expert once PER (token, slot) pair, not once per
// unique expert (n x 8 expert reads vs U(n): 128 vs 101 at n=16, 512 vs 222 at n=64).
//
// Design (bandwidth first, MFMA only as the dot engine):
//  * prep kernel, one launch: block 0 groups the (token, slot) pairs by expert from ids ON DEVICE (no host sync, CUDA-graph
//    safe): per-expert count + pair list + a compact list of ACTIVE experts (ascending expert id). All other blocks quantise
//    the activation rows once to int8 + f32 per-32 scale (q8_1-style, like MMVQ, so numerics stay on the MMVQ footing).
//  * GEMM kernel grid = (row-tile groups, upper bound of active experts); a block whose y exceeds the active count exits at
//    once, so only the active experts' weights are ever touched, each EXACTLY ONCE for all its tokens.
//  * memory path (V5I's): each wave streams 16 contiguous weight rows of one expert, one stage = 16 rows x 8 Q8_0 blocks
//    (272 B per row = 17 ALIGNED dwordx4), 272 fully used 16 B loads per wave per stage (raw buffer loads, out-of-range
//    lanes load 0 with no traffic), written raw to a WAVE-PRIVATE LDS slot and re-read in MFMA fragment layout. No
//    workgroup barrier in the stream: waves run independently; 2 stages in flight per wave in VGPRs (8.7 KB / wave,
//    35-70 KB / CU at 1-2 WGs per CU, the Little's-law point V3 measured). A block walks several consecutive 64-row tiles
//    of its expert (tpw), so the prologue is paid once per block and the stream is continuous across tiles.
//  * compute: v_mfma_i32_16x16x16i8, srcA = activations (16 token columns), srcB = 16 weight rows, per-block i32 -> f32
//    epilogue acc += s * d_w * d_a (MF=1 fragments of mmq-q8-mf.cu). Columns >= the expert's token count have scale 0.
//    n_tok <= 16: one 16-column tile; n_tok > 16: two tiles (each weight fragment feeds 2 MFMA pairs). An expert carrying
//    more than 32 tokens (heavy skew only: uniform top-8 of 256 puts ~2 tokens per expert at n=64) takes extra passes.
//    MFMA cost per byte is ~3x under the HBM share at 1 tile, so the compute engine is not the limiter at any n here.
//  * output: each (token, slot) column is written by exactly one lane; accumulation order is fixed -> deterministic.
// Byte floor per A3B step-layer at n tokens: 3 x U(n) x 1.11 MB; at 1.1 TB/s the full-model MoE step costs ~0.9 ms per
// added verify row (vs ~1.18 ms measured for split16): the kernel's job is to approach that floor.
// Env: GGML_HIP_MMID_MF=1, GGML_HIP_MMID_MF_MIN (1), GGML_HIP_MMID_MF_CAP (16, <= 64), GGML_HIP_MMID_MF_SLC (0),
//      GGML_HIP_MMID_MF_WGT (0 = 4 x #CU target workgroups).
#include "mmid-mf.cuh"
#include "mmq-q8-mf.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>

#define MMIDMF_MAXN  64                      // max tokens per call (= max pairs per expert: top-k experts are distinct)
#define MMIDMF_MAXE  1024                    // max experts
#define MMIDMF_KS    8                       // Q8_0 blocks per stage (256 k)
#define MMIDMF_ROWB  (MMIDMF_KS * 34)        // 272 B: one row of a stage = 17 aligned dwordx4
#define MMIDMF_WST   (16 * MMIDMF_ROWB)      // 4352 B: one wave stage (16 rows)
#define MMIDMF_NCH   (MMIDMF_WST / 16)       // 272 dwordx4 chunks per wave stage
#define MMIDMF_NLD   5                       // ceil(272 / 64) loads per lane per stage
#define MMIDMF_WSLOT (MMIDMF_WST + 64 * 16)  // wave LDS slot: stage + per-lane dummy for the partial 5th chunk
#define MMIDMF_NT    256                     // threads per GEMM block (4 waves x 16 rows = 64-row tile)
#define MMIDMF_RT    64

struct mmidmf_cfg {
    int on; int nmin; int cap; int slc; int wgt;
};

static int mmidmf_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

static const mmidmf_cfg & mmidmf_get_cfg() {
    static const mmidmf_cfg cfg = [] {
        mmidmf_cfg c;
        c.on   = mmidmf_env_int("GGML_HIP_MMID_MF", 0) == 1 ? 1 : 0;
        c.nmin = std::max(1, mmidmf_env_int("GGML_HIP_MMID_MF_MIN", 1));
        c.cap  = std::min(MMIDMF_MAXN, std::max(1, mmidmf_env_int("GGML_HIP_MMID_MF_CAP", 16)));
        c.slc  = mmidmf_env_int("GGML_HIP_MMID_MF_SLC", 0) != 0 ? 1 : 0;
        c.wgt  = std::max(0, mmidmf_env_int("GGML_HIP_MMID_MF_WGT", 0));
        return c;
    }();
    return cfg;
}

static size_t mmidmf_smem_bytes(const int64_t K, const int nct) {
    return (size_t) (K / 32) * 16 * nct * sizeof(float) + 4 * MMIDMF_WSLOT;
}

bool ggml_cuda_mmid_mf_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    const mmidmf_cfg & cfg = mmidmf_get_cfg();
    if (!cfg.on || !GGML_CUDA_CC_IS_CDNA2(cc) || ids == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    const int64_t K = src0->ne[0], N = src0->ne[1], E = src0->ne[2];
    const int64_t n_used = ids->ne[0], n_tok = ids->ne[1];
    if (n_tok < cfg.nmin || n_tok > cfg.cap || n_used < 1 || n_used > E) {
        return false;
    }
    if (src1->ne[0] != K || src1->ne[2] != n_tok || src1->ne[3] != 1 || !(src1->ne[1] == 1 || src1->ne[1] == n_used) ||
        dst->ne[0] != N || dst->ne[1] != n_used || dst->ne[2] != n_tok || dst->ne[3] != 1 || src0->ne[3] != 1 || ids->ne[2] != 1) {
        return false;
    }
    if (K % 512 != 0 || N % MMIDMF_RT != 0 || E > MMIDMF_MAXE) {   // K % 512: even stage count per tile
        return false;
    }
    if (!ggml_is_contiguous(src0) || ((uintptr_t) src0->data) % 16 != 0 || src0->nb[2] >= (size_t) INT32_MAX - 4096) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 || src1->nb[2] % sizeof(float) != 0 ||
        dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[2] % sizeof(float) != 0 ||
        ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (dst->nb[2] * (size_t) n_tok >= (size_t) INT32_MAX - 4096 || (size_t) src1->ne[1] * n_tok * K >= (size_t) INT32_MAX - 4096) {
        return false;   // raw-buffer byte offsets are 32-bit
    }
    if (ggml_cuda_q8ip_is_packed(src0)) {   // V5I in-place repacked weights are not natural Q8_0 rows
        return false;
    }
    return mmidmf_smem_bytes(K, n_tok > 16 ? 2 : 1) <= 64 * 1024;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(ids); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#endif
}

#if defined(GGML_USE_HIP)

// meta layout (int32): [0] n_active | [1 .. E] active expert ids | [1+E .. 2E] count per expert | [1+2E ..] E x MAXN pair list
// (pair p = token * n_used + slot).
static __global__ void mmidmf_prep(const float * __restrict__ y, const int64_t s11, const int64_t s12, const int ne11,
                                   const int nrows, const int K, int8_t * __restrict__ aq, float * __restrict__ as,
                                   const char * __restrict__ ids, const int64_t ids_nb1, const int n_used, const int n_tok,
                                   const int n_expert, int * __restrict__ meta) {
    if (blockIdx.x == 0) {
        __shared__ int cnt_s[MMIDMF_MAXE];
        const int tid = threadIdx.x;
        for (int e = tid; e < n_expert; e += blockDim.x) {
            cnt_s[e] = 0;
        }
        __syncthreads();
        int * list = meta + 1 + 2 * n_expert;
        for (int p = tid; p < n_tok * n_used; p += blockDim.x) {
            const int t = p / n_used;
            const int s = p - t * n_used;
            const int e = *(const int *) (ids + (int64_t) t * ids_nb1 + (int64_t) s * sizeof(int));
            if (e >= 0 && e < n_expert) {
                const int slot = atomicAdd(&cnt_s[e], 1);
                if (slot < MMIDMF_MAXN) {
                    list[e * MMIDMF_MAXN + slot] = p;
                }
            }
        }
        __syncthreads();
        for (int e = tid; e < n_expert; e += blockDim.x) {
            meta[1 + n_expert + e] = min(cnt_s[e], MMIDMF_MAXN);
        }
        if (tid < warpSize) {   // wave 0: deterministic compaction of the active experts (ascending id)
            int nact = 0;
            for (int base = 0; base < n_expert; base += warpSize) {
                const int  e    = base + tid;
                const bool flag = e < n_expert && cnt_s[e] > 0;
                const unsigned long long m = __ballot(flag);
                const unsigned long long below = tid == 0 ? 0ull : (m & (~0ull >> (64 - tid)));
                if (flag) {
                    meta[1 + nact + __popcll(below)] = e;
                }
                nact += __popcll(m);
            }
            if (tid == 0) {
                meta[0] = nact;
            }
        }
        return;
    }
    // activation quantisation: one 32-lane group per (row, Q8 block); row r = i11 + i12 * ne11
    const int nblk = K / 32;
    const int grp  = (blockIdx.x - 1) * (blockDim.x / 32) + threadIdx.x / 32;
    const int i    = threadIdx.x % 32;
    if (grp >= nrows * nblk) {
        return;
    }
    const int r   = grp / nblk;
    const int b   = grp - r * nblk;
    const int i11 = r % ne11;
    const int i12 = r / ne11;
    const float x = y[i11 * s11 + i12 * s12 + (int64_t) b * 32 + i];
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    }
    const float d = amax / 127.0f;
    const int   q = amax == 0.0f ? 0 : (int) roundf(x / d);
    aq[(int64_t) r * K + b * 32 + i] = (int8_t) q;
    if (i == 0) {
        as[(int64_t) r * nblk + b] = d;
    }
}

typedef int mmidmf_v4i __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ mmidmf_v4i mmidmf_make_rsrc(const void * base, uint32_t num_bytes) {
    union { mmidmf_v4i v; struct { const void * p; uint32_t range; uint32_t cfg; } s; } u;
    u.s.p     = base;
    u.s.range = num_bytes;
    u.s.cfg   = 0x00020000u;   // gfx9 raw buffer dword3, as mmq-q8-mf.cu / mmq-q8-wvsk.cu
    return u.v;
}

// aux: bit1 slc (nontemporal). voffset >= num_records -> returns 0, no memory traffic.
__device__ mmidmf_v4i mmidmf_buf_x4(mmidmf_v4i rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
typedef int mmidmf_v2i __attribute__((ext_vector_type(2)));
__device__ void mmidmf_buf_st(float v, mmidmf_v4i rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.store.f32");
__device__ mmidmf_v2i mmidmf_buf_x2(mmidmf_v4i rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

// same-wave LDS ordering point: compiler barrier + LDS drain (never waits on vmcnt, so the prefetch stays in flight)
#define MMIDMF_WAVE_LDS_SYNC() asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory")

template <int NCT>
struct mmidmf_stage {
    mmidmf_v4i w[MMIDMF_NLD];
    int2       a[MMIDMF_KS][NCT];
};

// Both issue helpers are BRANCH-FREE: a stage past the block's range (u >= U) or a column tile past the pass (ct >= nct)
// is pointed out of range through voffset (raw buffer load -> 0, no memory traffic). A conditional VMEM issue makes the
// waitcnt pass merge the skip path and drain the whole prefetch (vmcnt(0)) at the next commit.
template <int NCT, int SLC>
static __device__ __forceinline__ void mmidmf_issue_w(mmidmf_stage<NCT> & R, const int u, const int U, const int S, const int t0,
                                                      const int w, const int lane, const mmidmf_v4i wrsrc, const int nb01,
                                                      const uint32_t wbytes) {
    const int uu   = min(u, U - 1);
    const int tile = t0 + uu / S;
    const int ks   = uu - (uu / S) * S;
    const int soff = (tile * MMIDMF_RT + 16 * w) * nb01 + ks * MMIDMF_ROWB;   // wave-uniform
#pragma unroll
    for (int i = 0; i < MMIDMF_NLD; ++i) {
        const int q  = lane + 64 * i;
        const int vo = (u < U && (i < MMIDMF_NLD - 1 || q < MMIDMF_NCH)) ? (q / 17) * nb01 + (q % 17) * 16 : (int) wbytes;
        R.w[i] = mmidmf_buf_x4(wrsrc, vo, soff, SLC ? 2 : 0);
    }
}

// activation fragments of stage u (L2-resident int8): lane (column j, k group g) of tile ct, 8 B per Q8 block
template <int NCT>
static __device__ __forceinline__ void mmidmf_issue_a(mmidmf_stage<NCT> & R, const int u, const int U, const int S,
                                                      const mmidmf_v4i arsrc, const int * abase, const uint32_t abytes,
                                                      const int nct) {
    const int uu = min(u, U - 1);
    const int ks = uu - (uu / S) * S;
#pragma unroll
    for (int ct = 0; ct < NCT; ++ct) {
        const bool ok = u < U && ct < nct;
#pragma unroll
        for (int b = 0; b < MMIDMF_KS; ++b) {
            const mmidmf_v2i v = mmidmf_buf_x2(arsrc, ok ? abase[ct] + (ks * MMIDMF_KS + b) * 32 : (int) abytes, 0, 0);
            R.a[b][ct] = make_int2(v[0], v[1]);
        }
    }
}

template <int NCT, int SLC>
static __global__ void __launch_bounds__(MMIDMF_NT, 2)
mmidmf_gemm(const char * __restrict__ x, const int nb01, const int nb02, const int8_t * __restrict__ aq,
            const float * __restrict__ as, const int * __restrict__ meta, float * __restrict__ dst, const int64_t d1,
            const int64_t d2, const int K, const int n_used, const int ne11, const int n_expert, const int tpw, const int ntx,
            const int abytes_total, const int dbytes_total) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    extern __shared__ __align__(16) char mmidmf_smem[];
    __shared__ int pair_s[16 * NCT];
    __shared__ int arow_s[16 * NCT];

    const int nact = meta[0];
    if ((int) blockIdx.y >= nact) {
        return;
    }
    const int e  = meta[1 + blockIdx.y];
    const int c  = meta[1 + n_expert + e];
    const int t0 = blockIdx.x * tpw;
    const int t1 = min(ntx, t0 + tpw);
    if (c <= 0 || t0 >= t1) {
        return;
    }
    const int tid  = threadIdx.x;
    const int w    = __builtin_amdgcn_readfirstlane(tid >> 6);   // wave-uniform: keeps the buffer soffset in an SGPR
    const int lane = tid & 63;
    const int j    = lane & 15;
    const int g    = lane >> 4;
    const int nblk = K >> 5;
    const int S    = K >> 8;
    const int U    = (t1 - t0) * S;
    constexpr int SCW = 16 * NCT;
    float * sc = (float *) mmidmf_smem;                                       // [nblk][SCW] activation scales
    char  * wl = mmidmf_smem + (size_t) nblk * SCW * sizeof(float) + w * MMIDMF_WSLOT;
    const mmidmf_v4i wrsrc  = mmidmf_make_rsrc(x + (int64_t) e * nb02, (uint32_t) nb02);
    const uint32_t   wbytes = (uint32_t) nb02;
    const uint32_t   abytes = (uint32_t) abytes_total;
    const mmidmf_v4i arsrc  = mmidmf_make_rsrc(aq, abytes);
    const uint32_t   dbytes = (uint32_t) dbytes_total;
    const mmidmf_v4i drsrc  = mmidmf_make_rsrc(dst, dbytes);

    // column passes of SCW tokens: one pass unless the expert carries > 16 * NCT tokens (heavy skew), then the expert's
    // rows are streamed again per pass (NCT <= 2 keeps every instance spill-free)
    for (int cb = 0; cb < c; cb += SCW) {
    const int cp  = min(SCW, c - cb);
    const int nct = (cp + 15) >> 4;
    if (tid < SCW) {
        const int p = meta[1 + 2 * n_expert + e * MMIDMF_MAXN + cb + min(tid, cp - 1)];
        const int t = p / n_used;
        const int s = p - t * n_used;
        pair_s[tid] = tid < cp ? p : -1;
        arow_s[tid] = (s % ne11) + t * ne11;
    }
    __syncthreads();

    int abase[NCT];   // byte offsets into aq (< abytes <= INT32_MAX)
#pragma unroll
    for (int ct = 0; ct < NCT; ++ct) {
        abase[ct] = arow_s[ct * 16 + j] * K + 8 * g;
    }
    int coff[NCT][4];   // dst byte offset of this lane's output columns 4g+i (row 0), -1 = no column
#pragma unroll
    for (int ct = 0; ct < NCT; ++ct) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int p = pair_s[ct * 16 + 4 * g + i];
            const int t = p / n_used;
            coff[ct][i] = p >= 0 ? (int) ((t * d2 + (p - t * n_used) * d1) * (int64_t) sizeof(float)) : -1;
        }
    }
    mmidmf_stage<NCT> R0, R1;
    mmidmf_issue_w<NCT, SLC>(R0, 0, U, S, t0, w, lane, wrsrc, nb01, wbytes);
    mmidmf_issue_a<NCT>(R0, 0, U, S, arsrc, abase, abytes, nct);
    mmidmf_issue_w<NCT, SLC>(R1, 1, U, S, t0, w, lane, wrsrc, nb01, wbytes);
    mmidmf_issue_a<NCT>(R1, 1, U, S, arsrc, abase, abytes, nct);
    // scales after the prefetch: waiting for them costs only the (nearly simultaneous) landing of the first two stages
    for (int idx = tid; idx < nblk * SCW; idx += MMIDMF_NT) {
        const int b   = idx / SCW;
        const int col = idx - b * SCW;
        sc[idx] = col < cp ? as[(int64_t) arow_s[col] * nblk + b] : 0.0f;
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");   // LDS-only barrier (no vmcnt drain)

    float acc[NCT][4];
#pragma unroll
    for (int ct = 0; ct < NCT; ++ct) {
        acc[ct][0] = acc[ct][1] = acc[ct][2] = acc[ct][3] = 0.0f;
    }

    // U is even (K % 512 == 0 -> S even): R0 takes the even stages, R1 the odd ones; a tile ends on an odd stage, so only the
    // R1 step can flush (compile-time `flush`), keeping the R0 step free of conditional VMEM.
    auto step = [&](mmidmf_stage<NCT> & R, const int u, const bool flush) {
        // commit this stage's weights to the wave-private LDS slot (partial 5th chunk -> per-lane dummy, branch-free)
#pragma unroll
        for (int i = 0; i < MMIDMF_NLD; ++i) {
            const int q = lane + 64 * i;
            char * dp = (i < MMIDMF_NLD - 1 || q < MMIDMF_NCH) ? wl + q * 16 : wl + MMIDMF_WST + lane * 16;
            *(mmidmf_v4i *) dp = R.w[i];
        }
        MMIDMF_WAVE_LDS_SYNC();
        // weights of stage u+2 go out before the compute; its activations (L2 hits) after it
        mmidmf_issue_w<NCT, SLC>(R, u + 2, U, S, t0, w, lane, wrsrc, nb01, wbytes);
        const int ks = u % S;
        const char * lw = wl + j * MMIDMF_ROWB;
#pragma unroll
        for (int b = 0; b < MMIDMF_KS; ++b) {
            int w0, w1;
            if (b % 2 == 0) {   // even block: qs start at 2 mod 4 -> aligned 12 B window + funnel shift by 2 bytes
                const int * p = (const int *) (lw + 34 * b + 8 * g);
                const int q0 = p[0], q1 = p[1], q2 = p[2];
                w0 = (int) __builtin_amdgcn_alignbyte((uint32_t) q1, (uint32_t) q0, 2);
                w1 = (int) __builtin_amdgcn_alignbyte((uint32_t) q2, (uint32_t) q1, 2);
            } else {            // odd block: qs dword aligned
                const int * p = (const int *) (lw + 34 * b + 2 + 8 * g);
                w0 = p[0];
                w1 = p[1];
            }
            const float dw = __half2float(*(const __half *) (lw + 34 * b));
            const float * scb = sc + (ks * MMIDMF_KS + b) * SCW + 4 * g;
#pragma unroll
            for (int ct = 0; ct < NCT; ++ct) {
                if (ct < nct) {
                    const float4     da = *(const float4 *) (scb + ct * 16);
                    const mmidmf_v4i z  = {0, 0, 0, 0};
                    mmidmf_v4i s = __builtin_amdgcn_mfma_i32_16x16x16i8(R.a[b][ct].x, w0, z, 0, 0, 0);
                    s            = __builtin_amdgcn_mfma_i32_16x16x16i8(R.a[b][ct].y, w1, s, 0, 0, 0);
                    acc[ct][0] += (float) s[0] * (dw * da.x);
                    acc[ct][1] += (float) s[1] * (dw * da.y);
                    acc[ct][2] += (float) s[2] * (dw * da.z);
                    acc[ct][3] += (float) s[3] * (dw * da.w);
                }
            }
        }
        MMIDMF_WAVE_LDS_SYNC();   // fragment reads done before the next commit overwrites the slot
        mmidmf_issue_a<NCT>(R, u + 2, U, S, arsrc, abase, abytes, nct);
        // tile done (uniform): D[col = 4g + i][row = j] -> dst[token][slot][row]. Branch-free raw buffer stores: a lane with
        // no column, a column past the pass, or a step that ends no tile stores out of range (dropped, no traffic).
        const bool fl  = flush && ks == S - 1;
        const int  row = (t0 + u / S) * MMIDMF_RT + 16 * w + j;
#pragma unroll
        for (int ct = 0; ct < NCT; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vo = (fl && coff[ct][i] >= 0) ? coff[ct][i] + row * (int) sizeof(float) : (int) dbytes;
                mmidmf_buf_st(acc[ct][i], drsrc, vo, 0, 0);
                acc[ct][i] = fl ? 0.0f : acc[ct][i];
            }
        }
    };

    for (int u = 0; u < U; u += 2) {
        step(R0, u, false);
        step(R1, u + 1, true);
    }
    __syncthreads();   // pass boundary: pair/row/scale LDS is rewritten by the next pass
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(aq); GGML_UNUSED(as); GGML_UNUSED(meta);
    GGML_UNUSED(dst); GGML_UNUSED(d1); GGML_UNUSED(d2); GGML_UNUSED(K); GGML_UNUSED(n_used); GGML_UNUSED(ne11);
    GGML_UNUSED(n_expert); GGML_UNUSED(tpw); GGML_UNUSED(ntx); GGML_UNUSED(abytes_total); GGML_UNUSED(dbytes_total);
    NO_DEVICE_CODE;
#endif // defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
}

template <int NCT, int SLC>
static void mmidmf_launch(const dim3 grid, const size_t smem, cudaStream_t stream, const char * x, const int nb01,
                          const int nb02, const int8_t * aq, const float * as, const int * meta, float * dst, const int64_t d1,
                          const int64_t d2, const int K, const int n_used, const int ne11, const int n_expert, const int tpw,
                          const int ntx, const int abytes, const int dbytes) {
    mmidmf_gemm<NCT, SLC><<<grid, MMIDMF_NT, smem, stream>>>(x, nb01, nb02, aq, as, meta, dst, d1, d2, K, n_used, ne11,
                                                            n_expert, tpw, ntx, abytes, dbytes);
}

void ggml_cuda_mul_mat_id_mf(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             const ggml_tensor * ids, ggml_tensor * dst) {
    const mmidmf_cfg & cfg = mmidmf_get_cfg();
    cudaStream_t stream = ctx.stream();

    const int K        = (int) src0->ne[0];
    const int N        = (int) src0->ne[1];
    const int n_expert = (int) src0->ne[2];
    const int n_used   = (int) ids->ne[0];
    const int n_tok    = (int) ids->ne[1];
    const int ne11     = (int) src1->ne[1];
    const int nrows    = ne11 * n_tok;
    const int nblk     = K / 32;
    const int nct      = n_tok > 16 ? 2 : 1;   // column tiles per pass (more tokens per expert -> extra passes)

    const size_t off_as   = GGML_PAD((size_t) nrows * K, 256);
    const size_t off_meta = off_as + GGML_PAD((size_t) nrows * nblk * sizeof(float), 256);
    const size_t n_meta   = 1 + 2 * (size_t) n_expert + (size_t) n_expert * MMIDMF_MAXN;
    ggml_cuda_pool_alloc<char> buf(ctx.pool(), off_meta + n_meta * sizeof(int));
    int8_t * aq   = (int8_t *) buf.get();
    float  * as   = (float *) (buf.get() + off_as);
    int    * meta = (int *) (buf.get() + off_meta);

    const int ngrp = nrows * nblk;
    mmidmf_prep<<<1 + (ngrp + 7) / 8, 256, 0, stream>>>((const float *) src1->data, src1->nb[1] / sizeof(float),
        src1->nb[2] / sizeof(float), ne11, nrows, K, aq, as, (const char *) ids->data, ids->nb[1], n_used, n_tok, n_expert, meta);
    CUDA_CHECK(cudaGetLastError());

    const int ncu    = ggml_cuda_info().devices[ctx.device].nsm;
    const int u_max  = std::min(n_expert, n_tok * n_used);
    const int ntx    = N / MMIDMF_RT;
    const int target = cfg.wgt > 0 ? cfg.wgt : 4 * ncu;
    const int tpw    = std::max(1, std::min(ntx, (u_max * ntx + target - 1) / target));
    const dim3 grid((ntx + tpw - 1) / tpw, u_max, 1);
    const size_t smem = mmidmf_smem_bytes(K, nct);

    const char * x    = (const char *) src0->data;
    const int    nb01 = (int) src0->nb[1];
    const int    nb02 = (int) src0->nb[2];
    float      * d    = (float *) dst->data;
    const int64_t d1  = dst->nb[1] / sizeof(float);
    const int64_t d2  = dst->nb[2] / sizeof(float);
    const int abytes  = nrows * K;
    const int dbytes  = (int) (dst->nb[2] * n_tok);

#define MMIDMF_CASE(NCT_)                                                                                                       \
    case NCT_:                                                                                                                  \
        if (cfg.slc) {                                                                                                          \
            mmidmf_launch<NCT_, 1>(grid, smem, stream, x, nb01, nb02, aq, as, meta, d, d1, d2, K, n_used, ne11, n_expert, tpw, ntx, abytes, dbytes); \
        } else {                                                                                                                \
            mmidmf_launch<NCT_, 0>(grid, smem, stream, x, nb01, nb02, aq, as, meta, d, d1, d2, K, n_used, ne11, n_expert, tpw, ntx, abytes, dbytes); \
        }                                                                                                                       \
        break;
    switch (nct) {
        MMIDMF_CASE(1)
        MMIDMF_CASE(2)
        default: GGML_ABORT("mmid-mf: bad column-tile count %d", nct);
    }
#undef MMIDMF_CASE
    CUDA_CHECK(cudaGetLastError());
}

#else  // !GGML_USE_HIP

void ggml_cuda_mul_mat_id_mf(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(ids); GGML_UNUSED(dst);
    GGML_ABORT("mmid-mf: HIP only");
}

#endif // defined(GGML_USE_HIP)
