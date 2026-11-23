#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

// [TAG_GDN_STATE_BF16] recurrent-state precision helpers.
// The recurrent SSM state (curr_state / dst state region) may be stored as F32 (default,
// byte-identical to upstream) or BF16 (runtime-gated by GGML_CUDA_GDN_STATE_BF16 at cache
// build time). All delta-net MATH stays in fp32 in-register; only the HBM load/store of the
// state is narrowed, halving the ~192 MB/dispatch state read+write that dominates GDN decode.
static __device__ __forceinline__ float gdn_load(float x) { return x; }
static __device__ __forceinline__ float gdn_load(const nv_bfloat16 & x) { return __bfloat162float(x); }
template <typename T> static __device__ __forceinline__ T gdn_store(float x);
template <> __device__ __forceinline__ float       gdn_store<float>(float x)       { return x; }
template <> __device__ __forceinline__ nv_bfloat16 gdn_store<nv_bfloat16>(float x) { return __float2bfloat16(x); }

// =====================================================================================================
// [C8-6-GDN-CHUNKED] chunked (WY / UT-transform) gated delta-net prefill for CDNA (gfx90a), chunk 64,
// fp32 MFMA (v_mfma_f32_16x16x4f32) for every intra-chunk matmul. Non-KDA, S_v in {64, 128}.
//
// Per-token recurrence (identical to gated_delta_net_cuda, gamma_t = exp(g_t)):
//     S_t = gamma_t S_{t-1} + k_t delta_t^T,   delta_t = beta_t (v_t - gamma_t S_{t-1}^T k_t),
//     o_t = scale S_t^T q_t                   (S is d_k x d_v; stored transposed, M[col][i] = S[i][col])
// Within a chunk [t0, t0+L) with inclusive log-gate prefix G_t = sum_{s<=t} g_s and D[t,s] = exp(G_t - G_s):
//     A[t,s] = beta_t (k_t . k_s) D[t,s]   (s < t),     T = (I + A)^-1  (unit lower triangular)
//     X      = beta . (V - exp(G) . (K S0))             Delta' = T X        (== the chunk's delta_t rows)
//     O      = scale exp(G) . (Q S0) + Mq Delta',        Mq[t,s] = scale (q_t . k_s) D[t,s]   (s <= t)
//     S_L    = exp(G_{L-1}) S0 + (K . exp(G_{L-1} - G))^T Delta'
// Only bounded differences G_t - G_s (t >= s, <= 0) are exponentiated: no 1/G, no overflow.
// Kernel 1 (gdn_ch_prep, parallel over (chunk, head, seq)) builds T, Mq and G. Kernel 2 (gdn_ch_scan, one
// block per (head, seq, 16*NCT state columns)) walks the chunks in order; the v-columns of S are independent,
// so splitting them across blocks is exact. keep_rs (K > 1): the last K tokens are processed as single-token
// chunks (T = I) so every per-token snapshot the cache keeps is materialised exactly as the recurrent kernel does.
// Decode / verify (n_tokens < GGML_CUDA_GDN_CHUNKED_MIN_TOKENS, default 64) keeps the recurrent kernel.
// Kill switch: GGML_CUDA_GDN_CHUNKED=0. Columns per scan block: GGML_CUDA_GDN_CHUNKED_NC=16|32 (default 16).
// =====================================================================================================
#define GDN_CH_C 64

static std::atomic<uint64_t> g_gdn_chunked_n_calls{0};

uint64_t ggml_cuda_gdn_chunked_n_calls() {
    return g_gdn_chunked_n_calls.load();
}

// effective dispatch threshold (n_tokens >= this -> chunked); INT32_MAX when the kill switch is set
int32_t ggml_cuda_gdn_chunked_min_tokens() {
    static const int32_t v = [] {
        const char * off = getenv("GGML_CUDA_GDN_CHUNKED");
        if (off != nullptr && off[0] == '0') {
            return (int32_t) INT32_MAX;
        }
        const char * m = getenv("GGML_CUDA_GDN_CHUNKED_MIN_TOKENS");
        const int32_t t = m != nullptr ? (int32_t) atoi(m) : 64;
        return t < 2 ? (int32_t) 2 : t;
    }();
    return v;
}

bool ggml_cuda_gdn_chunked_usable(int device) {
#if defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_MMQ_MFMA)
    return GGML_CUDA_CC_IS_CDNA(ggml_cuda_info().devices[device].cc);
#else
    GGML_UNUSED(device);
    return false;
#endif
}

static int gdn_chunked_nc() {
    static const int v = [] {
        const char * s = getenv("GGML_CUDA_GDN_CHUNKED_NC");
        return (s != nullptr && atoi(s) == 32) ? 32 : 16;
    }();
    return v;
}

static bool gdn_chunked_debug() {
    static const bool v = [] {
        const char * s = getenv("GGML_CUDA_GDN_CHUNKED_DEBUG");
        return s != nullptr && s[0] == '1';
    }();
    return v;
}

#if defined(AMD_MFMA_AVAILABLE)
typedef float gdn_floatx4 __attribute__((ext_vector_type(4)));
// D(16x16) += A(16x4) B(4x16): lane supplies a = A[lane%16][lane/16], b = B[lane/16][lane%16];
// lane holds D[4*(lane/16) + v][lane%16], v = 0..3 (the J_MAJOR layout of mma.cuh on CDNA).
static __device__ __forceinline__ gdn_floatx4 gdn_mfma(const float a, const float b, const gdn_floatx4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
}
static __device__ __forceinline__ gdn_floatx4 gdn_zero4() {
    gdn_floatx4 z = {0.0f, 0.0f, 0.0f, 0.0f};
    return z;
}
#endif // defined(AMD_MFMA_AVAILABLE)

static __device__ __forceinline__ void gdn_ch_bounds(const int c, const int Nm, const int nmc, int & t0, int & L) {
    if (c < nmc) {
        t0 = c * GDN_CH_C;
        L  = min(GDN_CH_C, Nm - t0);
    } else {
        t0 = Nm + (c - nmc);
        L  = 1;
    }
}

// grid (n_chunks, H, n_seqs), block 256. Writes per chunk: G[64], T[64x64] (row-major, unit lower), Mq[64x64]
// (lower incl. diagonal; the strictly-upper 16x16 tiles are never written and never read).
template <int S_v>
__global__ void __launch_bounds__(256)
gdn_ch_prep(const float * __restrict__ q, const float * __restrict__ k,
            const float * __restrict__ g, const float * __restrict__ beta,
            float * __restrict__ T_out, float * __restrict__ M_out, float * __restrict__ G_out,
            const int64_t H, const int n_chunks, const int Nm, const int nmc,
            const int64_t sq1, const int64_t sq2, const int64_t sq3,
            const int64_t sb1, const int64_t sb2, const int64_t sb3,
            const uint3 neqk1_magic, const uint3 rq3_magic, const float scale) {
#if defined(AMD_MFMA_AVAILABLE)
    constexpr int AP = GDN_CH_C + 1;
    __shared__ float glog[GDN_CH_C];
    __shared__ float bet[GDN_CH_C];
    __shared__ float A_lds[GDN_CH_C * AP];

    const int c   = blockIdx.x;
    const int h   = blockIdx.y;
    const int seq = blockIdx.z;
    const int tid = threadIdx.x;
    const int w   = tid / 64;
    const int lane = tid % 64;
    const int lr  = lane % 16;
    const int lk  = lane / 16;

    int t0, L;
    gdn_ch_bounds(c, Nm, nmc, t0, L);

    const uint32_t iq1 = fastmodulo((uint32_t) h, neqk1_magic);
    const uint32_t iq3 = fastdiv((uint32_t) seq, rq3_magic);
    const float * kb = k + iq3 * sq3 + iq1 * sq1 + (int64_t) t0 * sq2;
    const float * qb = q + iq3 * sq3 + iq1 * sq1 + (int64_t) t0 * sq2;
    const int64_t gbo = seq * sb3 + h * sb1 + (int64_t) t0 * sb2;
    const int64_t cb  = ((int64_t) seq * H + h) * n_chunks + c;

    if (w == 0) {
        float gv = lane < L ? g[gbo + lane * sb2] : 0.0f;
#pragma unroll
        for (int d = 1; d < 64; d <<= 1) {
            const float o = __shfl_up(gv, d, 64);
            if (lane >= d) {
                gv += o;
            }
        }
        glog[lane] = gv;
        bet[lane]  = lane < L ? beta[gbo + lane * sb2] : 0.0f;
        G_out[cb * GDN_CH_C + lane] = gv;
    }
    __syncthreads();

    // 20 lower-triangular 16x16 tile jobs: 10 of K K^T (-> A) and 10 of Q K^T (-> Mq), contracting S_v.
    for (int j = w; j < 20; j += 4) {
        const bool is_qk = j >= 10;
        const int  jj    = is_qk ? j - 10 : j;
        const int  a     = jj >= 6 ? 3 : (jj >= 3 ? 2 : (jj >= 1 ? 1 : 0));
        const int  b     = jj - a * (a + 1) / 2;
        const int  ra    = 16 * a + lr;
        const int  rb    = 16 * b + lr;
        const float * arow = (is_qk ? qb : kb) + (int64_t) ra * sq2;
        const float * brow = kb + (int64_t) rb * sq2;
        const bool va = ra < L;
        const bool vb = rb < L;
        gdn_floatx4 acc = gdn_zero4();
#pragma unroll
        for (int p = 0; p < S_v / 16; ++p) {
            const int i0 = 16 * p + 4 * lk;
            const float4 av = va ? *(const float4 *) (arow + i0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 bv = vb ? *(const float4 *) (brow + i0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            acc = gdn_mfma(av.x, bv.x, acc);
            acc = gdn_mfma(av.y, bv.y, acc);
            acc = gdn_mfma(av.z, bv.z, acc);
            acc = gdn_mfma(av.w, bv.w, acc);
        }
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int  t  = 16 * a + 4 * lk + v;
            const int  s  = 16 * b + lr;
            const bool in = t < L && s < L;
            if (!is_qk) {
                A_lds[t * AP + s] = (in && s < t) ? bet[t] * acc[v] * expf(glog[t] - glog[s]) : 0.0f;
            } else {
                M_out[cb * (GDN_CH_C * GDN_CH_C) + t * GDN_CH_C + s] = (in && s <= t) ? scale * acc[v] * expf(glog[t] - glog[s]) : 0.0f;
            }
        }
    }
    __syncthreads();

    // T = (I + A)^-1 by forward substitution; lane j owns column j (A_lds reads are wave-uniform broadcasts).
    if (w == 0) {
        const int j = lane;
        float Tc[GDN_CH_C];
#pragma unroll
        for (int t = 0; t < GDN_CH_C; ++t) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < t; ++s) {
                acc = fmaf(A_lds[t * AP + s], Tc[s], acc);
            }
            Tc[t] = (t == j ? 1.0f : 0.0f) - acc;
        }
        float * To = T_out + cb * (GDN_CH_C * GDN_CH_C) + j;
#pragma unroll
        for (int t = 0; t < GDN_CH_C; ++t) {
            To[t * GDN_CH_C] = Tc[t];
        }
    }
#else
    GGML_UNUSED_VARS(q, k, g, beta, T_out, M_out, G_out, H, n_chunks, Nm, nmc, sq1, sq2, sq3, sb1, sb2, sb3,
                     neqk1_magic, rq3_magic, scale);
    NO_DEVICE_CODE;
#endif // defined(AMD_MFMA_AVAILABLE)
}

// grid (H, n_seqs, S_v / (16*NCT)), block 256 (4 waves). Wave w owns chunk rows 16w..16w+15 in the chunk GEMMs and
// state rows [S_v/4 * w, S_v/4 * (w+1)) in the state update. The state slice lives in registers (MFMA accumulators)
// and is mirrored to LDS once per chunk as the B operand of K S0 / Q S0.
template <int S_v, int NCT, typename state_t>
__global__ void __launch_bounds__(256)
gdn_ch_scan(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
            const float * __restrict__ beta,
            const float * __restrict__ T_in, const float * __restrict__ M_in, const float * __restrict__ G_in,
            const state_t * __restrict__ curr_state, state_t * __restrict__ dst, state_t * __restrict__ state,
            const int64_t H, const int64_t n_tokens, const int n_chunks, const int Nm, const int nmc, const int Kt,
            const int64_t sq1, const int64_t sq2, const int64_t sq3,
            const int64_t sv1, const int64_t sv2, const int64_t sv3,
            const int64_t sb1, const int64_t sb2, const int64_t sb3,
            const uint3 neqk1_magic, const uint3 rq3_magic, const float scale, const int64_t state_slot_stride) {
#if defined(AMD_MFMA_AVAILABLE)
    constexpr int NC  = 16 * NCT;
    constexpr int NCP = NC + 4;              // row pitch: rows 4 apart land 16 banks apart
    constexpr int RTW = S_v / 64;            // 16-row state tiles per wave (2 for S_v=128, 1 for 64)
    static_assert(S_v == 64 || S_v == 128, "gdn_ch_scan: S_v must be 64 or 128");
    __shared__ float S_lds[S_v * NCP];
    __shared__ float X_lds[GDN_CH_C * NCP];
    __shared__ float V_lds[GDN_CH_C * NCP];
    __shared__ float gam[GDN_CH_C];
    __shared__ float dec[GDN_CH_C];
    __shared__ float bet[GDN_CH_C];

    const int h        = blockIdx.x;
    const int seq      = blockIdx.y;
    const int col_base = blockIdx.z * NC;
    const int tid  = threadIdx.x;
    const int w    = tid / 64;
    const int lane = tid % 64;
    const int lr   = lane % 16;
    const int lk   = lane / 16;

    const uint32_t iq1 = fastmodulo((uint32_t) h, neqk1_magic);
    const uint32_t iq3 = fastdiv((uint32_t) seq, rq3_magic);
    const float * k_head = k + iq3 * sq3 + iq1 * sq1;
    const float * q_head = q + iq3 * sq3 + iq1 * sq1;
    const float * v_head = v + seq * sv3 + h * sv1 + col_base;
    const int64_t gbo    = seq * sb3 + h * sb1;

    const state_t * s_in  = curr_state + ((int64_t) seq * H + h) * S_v * S_v;
    state_t *       s_out = state + ((int64_t) seq * H + h) * S_v * S_v;
    state_t *       o_out = dst + ((int64_t) seq * n_tokens * H + h) * S_v + col_base;

    // state accumulators: Sacc[r][ct][vv] = S[16*(w*RTW + r) + 4*lk + vv][col_base + 16*ct + lr]
    gdn_floatx4 Sacc[RTW][NCT];
#pragma unroll
    for (int r = 0; r < RTW; ++r) {
#pragma unroll
        for (int ct = 0; ct < NCT; ++ct) {
            const int i0  = 16 * (w * RTW + r) + 4 * lk;
            const int col = 16 * ct + lr;
            const state_t * src = s_in + (int64_t) (col_base + col) * S_v + i0;
#pragma unroll
            for (int vv = 0; vv < 4; ++vv) {
                Sacc[r][ct][vv] = gdn_load(src[vv]);
                S_lds[(i0 + vv) * NCP + col] = Sacc[r][ct][vv];
            }
        }
    }

    for (int c = 0; c < n_chunks; ++c) {
        int t0, L;
        gdn_ch_bounds(c, Nm, nmc, t0, L);
        const int64_t cb = ((int64_t) seq * H + h) * n_chunks + c;
        const float * Tb = T_in + cb * (GDN_CH_C * GDN_CH_C);
        const float * Mb = M_in + cb * (GDN_CH_C * GDN_CH_C);
        const float * kb = k_head + (int64_t) t0 * sq2;
        const float * qb = q_head + (int64_t) t0 * sq2;

        if (tid < GDN_CH_C) {
            const float gl  = G_in[cb * GDN_CH_C + tid];
            const float glL = G_in[cb * GDN_CH_C + L - 1];
            const bool  in  = tid < L;
            gam[tid] = in ? expf(gl) : 0.0f;
            dec[tid] = in ? expf(glL - gl) : 0.0f;
            bet[tid] = in ? beta[gbo + (int64_t) (t0 + tid) * sb2] : 0.0f;
        }
        __syncthreads();

        // ---- step A: KS = K S0, QS = Q S0 (rows 16w.., contract S_v); X = beta (V - gamma KS) ----
        gdn_floatx4 oacc[NCT];
        {
            const int  ta = 16 * w + lr;
            const bool va = ta < L;
            const float * krow = kb + (int64_t) ta * sq2;
            const float * qrow = qb + (int64_t) ta * sq2;
            gdn_floatx4 ks[NCT], qs[NCT];
#pragma unroll
            for (int ct = 0; ct < NCT; ++ct) {
                ks[ct] = gdn_zero4();
                qs[ct] = gdn_zero4();
            }
#pragma unroll
            for (int p = 0; p < S_v / 16; ++p) {
                const int i0 = 16 * p + 4 * lk;
                const float4 kv4 = va ? *(const float4 *) (krow + i0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float4 qv4 = va ? *(const float4 *) (qrow + i0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float ka[4] = {kv4.x, kv4.y, kv4.z, kv4.w};
                const float qa[4] = {qv4.x, qv4.y, qv4.z, qv4.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
#pragma unroll
                    for (int ct = 0; ct < NCT; ++ct) {
                        const float bs = S_lds[(i0 + e) * NCP + 16 * ct + lr];
                        ks[ct] = gdn_mfma(ka[e], bs, ks[ct]);
                        qs[ct] = gdn_mfma(qa[e], bs, qs[ct]);
                    }
                }
            }
#pragma unroll
            for (int ct = 0; ct < NCT; ++ct) {
#pragma unroll
                for (int vv = 0; vv < 4; ++vv) {
                    const int t = 16 * w + 4 * lk + vv;
                    float x = 0.0f;
                    if (t < L) {
                        const float vt = v_head[(int64_t) (t0 + t) * sv2 + 16 * ct + lr];
                        x = bet[t] * (vt - gam[t] * ks[ct][vv]);
                    }
                    X_lds[t * NCP + 16 * ct + lr] = x;
                    oacc[ct][vv] = gam[t] * scale * qs[ct][vv];
                }
            }
        }
        __syncthreads();

        // ---- step B: Delta' = T X (rows 16w.., T lower triangular: contract s < 16(w+1)) ----
        {
            const float * trow = Tb + (16 * w + lr) * GDN_CH_C;
            gdn_floatx4 vn[NCT];
#pragma unroll
            for (int ct = 0; ct < NCT; ++ct) {
                vn[ct] = gdn_zero4();
            }
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                if (p <= w) {
                    const int s0 = 16 * p + 4 * lk;
                    const float4 t4 = *(const float4 *) (trow + s0);
                    const float ta4[4] = {t4.x, t4.y, t4.z, t4.w};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
#pragma unroll
                        for (int ct = 0; ct < NCT; ++ct) {
                            vn[ct] = gdn_mfma(ta4[e], X_lds[(s0 + e) * NCP + 16 * ct + lr], vn[ct]);
                        }
                    }
                }
            }
#pragma unroll
            for (int ct = 0; ct < NCT; ++ct) {
#pragma unroll
                for (int vv = 0; vv < 4; ++vv) {
                    V_lds[(16 * w + 4 * lk + vv) * NCP + 16 * ct + lr] = vn[ct][vv];
                }
            }
        }
        __syncthreads();

        // ---- step C: O = gamma scale QS + Mq Delta' (rows 16w.., contract s < 16(w+1)) ----
        {
            const float * mrow = Mb + (16 * w + lr) * GDN_CH_C;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                if (p <= w) {
                    const int s0 = 16 * p + 4 * lk;
                    const float4 m4 = *(const float4 *) (mrow + s0);
                    const float ma4[4] = {m4.x, m4.y, m4.z, m4.w};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
#pragma unroll
                        for (int ct = 0; ct < NCT; ++ct) {
                            oacc[ct] = gdn_mfma(ma4[e], V_lds[(s0 + e) * NCP + 16 * ct + lr], oacc[ct]);
                        }
                    }
                }
            }
#pragma unroll
            for (int ct = 0; ct < NCT; ++ct) {
#pragma unroll
                for (int vv = 0; vv < 4; ++vv) {
                    const int t = 16 * w + 4 * lk + vv;
                    if (t < L) {
                        o_out[(int64_t) (t0 + t) * S_v * H + 16 * ct + lr] = gdn_store<state_t>(oacc[ct][vv]);
                    }
                }
            }
        }

        // ---- step D: S = gamma_L S0 + (K . dec)^T Delta' (state rows of this wave, contract the chunk) ----
        {
            const float gL = gam[L - 1];
#pragma unroll
            for (int r = 0; r < RTW; ++r) {
                const int ia = 16 * (w * RTW + r) + lr;
#pragma unroll
                for (int ct = 0; ct < NCT; ++ct) {
                    Sacc[r][ct] *= gL;
                }
#pragma unroll
                for (int s0 = 0; s0 < GDN_CH_C; s0 += 4) {
                    const int s = s0 + lk;
                    const float a = s < L ? kb[(int64_t) s * sq2 + ia] * dec[s] : 0.0f;
#pragma unroll
                    for (int ct = 0; ct < NCT; ++ct) {
                        Sacc[r][ct] = gdn_mfma(a, V_lds[s * NCP + 16 * ct + lr], Sacc[r][ct]);
                    }
                }
            }
        }

        // S_lds was last read in step A (two barriers ago): refresh it for the next chunk.
        const int slot = (int) (n_tokens - 1 - (t0 + L - 1));
        const bool snap = Kt > 0 && slot >= 0 && slot < Kt;
#pragma unroll
        for (int r = 0; r < RTW; ++r) {
#pragma unroll
            for (int ct = 0; ct < NCT; ++ct) {
                const int i0  = 16 * (w * RTW + r) + 4 * lk;
                const int col = 16 * ct + lr;
#pragma unroll
                for (int vv = 0; vv < 4; ++vv) {
                    S_lds[(i0 + vv) * NCP + col] = Sacc[r][ct][vv];
                }
                if (snap) {
                    state_t * dsts = s_out + (int64_t) slot * state_slot_stride + (int64_t) (col_base + col) * S_v + i0;
#pragma unroll
                    for (int vv = 0; vv < 4; ++vv) {
                        dsts[vv] = gdn_store<state_t>(Sacc[r][ct][vv]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (Kt == 0) {
#pragma unroll
        for (int r = 0; r < RTW; ++r) {
#pragma unroll
            for (int ct = 0; ct < NCT; ++ct) {
                const int i0  = 16 * (w * RTW + r) + 4 * lk;
                const int col = 16 * ct + lr;
                state_t * dsts = s_out + (int64_t) (col_base + col) * S_v + i0;
#pragma unroll
                for (int vv = 0; vv < 4; ++vv) {
                    dsts[vv] = gdn_store<state_t>(Sacc[r][ct][vv]);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, beta, T_in, M_in, G_in, curr_state, dst, state, H, n_tokens, n_chunks, Nm, nmc, Kt,
                     sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride);
    NO_DEVICE_CODE;
#endif // defined(AMD_MFMA_AVAILABLE)
}

template <int S_v, int NCT, typename state_t>
static void gdn_ch_launch(ggml_backend_cuda_context & ctx,
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        const state_t * s_d, state_t * dst_d, state_t * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs, int Kt,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, uint3 neqk1_magic, uint3 rq3_magic,
        float scale, int64_t state_slot_stride) {
    cudaStream_t stream = ctx.stream();
    const int Nm       = (int) n_tokens - Kt;
    const int nmc      = (Nm + GDN_CH_C - 1) / GDN_CH_C;
    const int n_chunks = nmc + Kt;
    const size_t per_chunk = 2 * GDN_CH_C * GDN_CH_C + GDN_CH_C;
    ggml_cuda_pool_alloc<float> buf(ctx.pool(), (size_t) n_seqs * H * n_chunks * per_chunk);
    float * T_d = buf.get();
    float * M_d = T_d + (size_t) n_seqs * H * n_chunks * GDN_CH_C * GDN_CH_C;
    float * G_d = M_d + (size_t) n_seqs * H * n_chunks * GDN_CH_C * GDN_CH_C;

    gdn_ch_prep<S_v><<<dim3(n_chunks, H, n_seqs), dim3(256), 0, stream>>>(
        q_d, k_d, g_d, b_d, T_d, M_d, G_d, H, n_chunks, Nm, nmc, sq1, sq2, sq3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
    CUDA_CHECK(cudaGetLastError());
    gdn_ch_scan<S_v, NCT, state_t><<<dim3(H, n_seqs, S_v / (16 * NCT)), dim3(256), 0, stream>>>(
        q_d, k_d, v_d, b_d, T_d, M_d, G_d, s_d, dst_d, state_d, H, n_tokens, n_chunks, Nm, nmc, Kt,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride);
    CUDA_CHECK(cudaGetLastError());
}

// Returns true when the chunked path handled the op.
template <typename state_t>
static bool gdn_chunked_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src_q, const ggml_tensor * src_k,
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        const state_t * s_d, state_t * dst_d, state_t * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs, bool kda, int K,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride) {
    // decode / verify fast exit: one compare, no device query (the recurrent path stays exactly as before)
    if (n_tokens < ggml_cuda_gdn_chunked_min_tokens() && !gdn_chunked_debug()) {
        return false;
    }
    const int  Kt     = K > 1 ? K : 0;
    const bool shape  = !kda && (S_v == 64 || S_v == 128);
    const bool tokens = n_tokens >= ggml_cuda_gdn_chunked_min_tokens() && n_tokens > Kt && Kt <= GDN_CH_C &&
                        n_tokens <= INT32_MAX / 2;
    // float4 operand loads need 16-byte aligned q/k rows
    const bool align  = sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 &&
                        ((uintptr_t) src_q->data) % 16 == 0 && ((uintptr_t) src_k->data) % 16 == 0;
    const bool dev    = ggml_cuda_gdn_chunked_usable(ggml_cuda_get_device());
    const bool go     = shape && tokens && align && dev;
    if (gdn_chunked_debug()) {
        GGML_LOG_WARN("[gdn-chunked] S_v=%" PRId64 " H=%" PRId64 " n_tokens=%" PRId64 " n_seqs=%" PRId64 " K=%d kda=%d"
                      " shape=%d tokens=%d align=%d dev=%d -> %s\n", S_v, H, n_tokens, n_seqs, K, (int) kda,
                      (int) shape, (int) tokens, (int) align, (int) dev, go ? "chunked" : "recurrent");
    }
    if (!go) {
        return false;
    }
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const bool  nc32        = gdn_chunked_nc() == 32;
#define GDN_CH_ARGS ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, Kt, \
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride
    if (S_v == 128) {
        if (nc32) { gdn_ch_launch<128, 2, state_t>(GDN_CH_ARGS); } else { gdn_ch_launch<128, 1, state_t>(GDN_CH_ARGS); }
    } else {
        if (nc32) { gdn_ch_launch<64, 2, state_t>(GDN_CH_ARGS); } else { gdn_ch_launch<64, 1, state_t>(GDN_CH_ARGS); }
    }
#undef GDN_CH_ARGS
    // route witness (gate packets grep it; also the build's strings content marker)
    if (g_gdn_chunked_n_calls.fetch_add(1) == 0) {
        GGML_LOG_WARN("ggml_cuda_gdn_chunked: C8-6 chunked WY gated delta-net prefill ran (call 1, chunk 64, fp32 MFMA, nc %d)\n",
                      nc32 ? 32 : 16);
    }
    return true;
}

// [RS-REPLAY] GDN input-log replay (llama.cpp PR #31669, head 93252b852acc): the F32 k/v/gate/beta of the previous
// ubatch's pending tokens, replayed in registers before the new tokens with the exact same update as the main loop
// (minus the attention output), so a rollback only drops pending tokens instead of restoring a snapshot.
// pk == nullptr: no replay. keep: write the state after the replay (the committed state) and not the final state.
struct gdn_rep_args {
    const float *   pk    = nullptr; // [S_k, H_k, n_pend, n_seqs]
    const float *   pv    = nullptr; // [S_v, H_v, n_pend, n_seqs]
    const float *   pgb   = nullptr; // [g0, H_v, 2*n_pend, n_seqs]: gates, then betas (element 0)
    const int32_t * n_rep = nullptr; // [n_seqs]
    int64_t spk1 = 0, spk2 = 0, spk3 = 0;
    int64_t spv1 = 0, spv2 = 0, spv3 = 0;
    int64_t sgb1 = 0, sgb2 = 0, sgb3 = 0;
    int     n_pend = 0;
    int     keep   = 0;
};

template <int S_v, bool KDA, bool keep_rs_t, typename state_t, bool tree_t = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const state_t * curr_state,
                                     state_t *     dst,
                                     state_t *     state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * parent,
                                     const gdn_rep_args rep) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    state_t *     attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = gdn_load(curr_state[i]);
    }

    if (rep.pk != nullptr) {
        // [RS-REPLAY] same update as the main loop below, without the attention output
        const int n_r = rep.n_rep[sequence];
        for (int t = 0; t < n_r; t++) {
            const float * k_t = rep.pk + sequence * rep.spk3 + t * rep.spk2 + iq1 * rep.spk1;
            const float * v_t = rep.pv + sequence * rep.spv3 + t * rep.spv2 + h_idx * rep.spv1;
            const float * g_t = rep.pgb + sequence * rep.sgb3 + t * rep.sgb2 + h_idx * rep.sgb1;
            const float beta_val = rep.pgb[sequence * rep.sgb3 + (rep.n_pend + t) * rep.sgb2 + h_idx * rep.sgb1];

            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_t[r * warp_size + lane];
            }
            if constexpr (!KDA) {
                const float g_val = expf(*g_t);
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<warp_size>(kv_shard);
                float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                }
            } else {
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<warp_size>(kv_shard);
                float delta_col = (v_t[col] - kv_col) * beta_val;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    s_shard[r] = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                }
            }
        }
        if (rep.keep) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * warp_size + lane;
                state[col * S_v + i] = gdn_store<state_t>(s_shard[r]);
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        if constexpr (tree_t) {
            // [AK-DTREE-2] tree walk (record mode): column t continues from its parent's state. Each warp owns one
            // state column and reloads only the shard its own lanes stored into the parent's snapshot slot, so no
            // cross-thread synchronisation is needed. F32 state only: the reload is exact.
            const int p = parent[sequence * n_tokens + t];
            if (p != t - 1) {
                const state_t * src_s = p < 0 ? curr_state
                    : state + ((int64_t) n_tokens - 1 - p) * state_slot_stride + col * S_v;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r] = gdn_load(src_s[r * warp_size + lane]);
                }
            }
        }
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = gdn_store<state_t>(attn_col * scale);
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = gdn_store<state_t>(attn_col * scale);
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                state_t * snap_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    snap_state[col * S_v + i] = gdn_store<state_t>(s_shard[r]);
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if (rep.pk == nullptr || !rep.keep) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * warp_size + lane;
                state[col * S_v + i] = gdn_store<state_t>(s_shard[r]);
            }
        }
    }
}

// =====================================================================================================
// [DTREE-0-GDN-ROWS] few-row (verify) gated delta-net: W = n_tokens in 1..16, one launch per op, all W rows of a
// (head, seq, 32-column slab) in one 512-thread block. The recurrent state is read from HBM once per verify; q/k
// of all W rows, exp(g), beta, v and the parent indices are staged in LDS up front, so the serial per-row chain
// carries no global-memory latency (the recurrent kernel pays one q/k/v/g/beta round trip per row, which is
// the ~0.155 ms/row the 27B verify measured). 16 lanes own one state column (8 rows each for S_v = 128) and
// reduce with 4 DPP butterfly steps (xor 1, 2 inside a quad, half-mirror, mirror: every lane ends with the
// bit-identical sum). The q-side dot (attention output) is off the serial chain: per-row partials stay in
// registers and are reduced after the walk. Snapshots (keep_rs) and the final state are written exactly as the
// recurrent kernel writes them (same slot mapping). Tree batches (AK-DTREE-2 record mode, parent[] I32): column t
// continues from its parent's state; a parent of -1 restores s0 from registers, any other non-chain parent
// reloads the parent's snapshot slot, which this same thread stored earlier (F32: exact), and that reload is
// issued one column ahead so its latency overlaps the previous column's compute. Non-KDA, S_v 64/128 only.
// GGML_HIP_GDN_ROWS=1 enables it (default off = the recurrent kernel, byte-identical dispatch);
// GGML_HIP_GDN_ROWS_MIN (default 1) and GGML_HIP_GDN_ROWS_MAX (default 16, max 16) bound the n_tokens range.
// =====================================================================================================
#define GDN_ROWS_WMAX    16
#define GDN_ROWS_LPC     16
#define GDN_ROWS_THREADS 512
#define GDN_ROWS_CPB     (GDN_ROWS_THREADS / GDN_ROWS_LPC)
#define GDN_ROWS_RSAVE   4

static std::atomic<uint64_t> g_gdn_rows_n_calls{0};

uint64_t ggml_cuda_gdn_rows_n_calls() {
    return g_gdn_rows_n_calls.load();
}

static int gdn_rows_env_int(const char * name, int def) {
    const char * s = getenv(name);
    return s != nullptr && s[0] != '\0' ? atoi(s) : def;
}

// [min, max] n_tokens routed to the rows kernel; {1, 0} (empty) when GGML_HIP_GDN_ROWS is unset/0 or not HIP
static void gdn_rows_range(int & lo, int & hi) {
    static const int v_lo = [] { return std::max(1, gdn_rows_env_int("GGML_HIP_GDN_ROWS_MIN", 1)); }();
    static const int v_hi = [] {
#if defined(GGML_USE_HIP)
        if (gdn_rows_env_int("GGML_HIP_GDN_ROWS", 0) != 1) {
            return 0;
        }
        return std::min(GDN_ROWS_WMAX, gdn_rows_env_int("GGML_HIP_GDN_ROWS_MAX", GDN_ROWS_WMAX));
#else
        return 0;
#endif
    }();
    lo = v_lo;
    hi = v_hi;
}

// sum over the 16 lanes of a DPP row; every lane gets the bit-identical result
static __device__ __forceinline__ float gdn_rows_sum16(float x) {
#if defined(GGML_USE_HIP)
    x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0xB1,  0xF, 0xF, true)); // quad_perm [1,0,3,2]
    x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x4E,  0xF, 0xF, true)); // quad_perm [2,3,0,1]
    x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x141, 0xF, 0xF, true)); // row_half_mirror
    x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x140, 0xF, 0xF, true)); // row_mirror
    return x;
#else
    return warp_reduce_sum<GDN_ROWS_LPC>(x);
#endif
}

// wave-uniform int (the parent index depends on (seq, t) only): scalar register on AMD
static __device__ __forceinline__ int gdn_rows_uniform(int x) {
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_readfirstlane(x);
#else
    return x;
#endif
}

template <typename state_t>
static __device__ __forceinline__ void gdn_rows_ld4(const state_t * p, float * o) {
    if constexpr (std::is_same_v<state_t, float>) {
        const float4 x = *(const float4 *) p;
        o[0] = x.x; o[1] = x.y; o[2] = x.z; o[3] = x.w;
    } else {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            o[e] = gdn_load(p[e]);
        }
    }
}

template <typename state_t>
static __device__ __forceinline__ void gdn_rows_st4(state_t * p, const float * o) {
    if constexpr (std::is_same_v<state_t, float>) {
        *(float4 *) p = make_float4(o[0], o[1], o[2], o[3]);
    } else {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            p[e] = gdn_store<state_t>(o[e]);
        }
    }
}

template <int S_v, bool keep_rs_t, typename state_t, bool tree_t>
__global__ void __launch_bounds__(GDN_ROWS_THREADS, 1)
gdn_rows_cuda(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
              const float * __restrict__ g, const float * __restrict__ beta,
              const state_t * curr_state, state_t * dst, state_t * state,
              const int64_t H, const int n_tokens,
              const int64_t sq1, const int64_t sq2, const int64_t sq3,
              const int64_t sv1, const int64_t sv2, const int64_t sv3,
              const int64_t sb1, const int64_t sb2, const int64_t sb3,
              const uint3 neqk1_magic, const uint3 rq3_magic, const float scale,
              const int64_t state_slot_stride, const int K, const int32_t * __restrict__ parent) {
    constexpr int LPC = GDN_ROWS_LPC;
    constexpr int CPB = GDN_ROWS_CPB;
    constexpr int RPL = S_v / LPC;   // state rows per lane
    constexpr int NCH = RPL / 4;     // float4 chunks per lane: row i = c*4*LPC + 4*lane + e
    static_assert(RPL % 4 == 0, "S_v must be a multiple of 4*LPC");

    __shared__ float q_s[GDN_ROWS_WMAX][S_v];
    __shared__ float k_s[GDN_ROWS_WMAX][S_v];
    __shared__ float v_s[GDN_ROWS_WMAX][CPB];
    __shared__ float eg_s[GDN_ROWS_WMAX];
    __shared__ float b_s[GDN_ROWS_WMAX];
    __shared__ int   p_s[GDN_ROWS_WMAX];
    __shared__ int   r_s[GDN_ROWS_WMAX];   // [DTREE-ROUND-OVERHEAD-2] register save slot of column t, -1 = none

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      tid      = threadIdx.x;
    const int      lane     = tid % LPC;
    const int      cl       = tid / LPC;
    const int      col0     = blockIdx.z * CPB;
    const int      col      = col0 + cl;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_head = ((int64_t) sequence * H + h_idx) * S_v * S_v;   // in == out per-(seq, head) offset
    curr_state += state_head + (int64_t) col * S_v;
    state      += state_head + (int64_t) col * S_v;
    dst        += ((int64_t) sequence * n_tokens * H + h_idx) * S_v + col;

    ggml_cuda_pdl_sync();

    float s[RPL];
#pragma unroll
    for (int c = 0; c < NCH; c++) {
        gdn_rows_ld4(curr_state + c * 4 * LPC + 4 * lane, s + 4 * c);
    }

    const float * qh = q + iq3 * sq3 + iq1 * sq1;
    const float * kh = k + iq3 * sq3 + iq1 * sq1;
    for (int idx = tid; idx < n_tokens * S_v; idx += GDN_ROWS_THREADS) {
        const int t = idx / S_v;
        const int i = idx % S_v;
        q_s[t][i] = qh[t * sq2 + i];
        k_s[t][i] = kh[t * sq2 + i];
    }
    const float * vh = v + sequence * sv3 + h_idx * sv1 + col0;
    for (int idx = tid; idx < n_tokens * CPB; idx += GDN_ROWS_THREADS) {
        const int t = idx / CPB;
        const int c = idx % CPB;
        v_s[t][c] = vh[t * sv2 + c];
    }
    if (tid < n_tokens) {
        const int64_t gb = sequence * sb3 + tid * sb2 + h_idx * sb1;
        eg_s[tid] = expf(g[gb]);
        b_s[tid]  = beta[gb];
        if constexpr (tree_t) {
            p_s[tid] = parent[(int64_t) sequence * n_tokens + tid];
        }
    }
    if constexpr (tree_t) {
        // [DTREE-ROUND-OVERHEAD-2] a column with a non-chain child (a child other than column t+1) keeps its state
        // in one of GDN_ROWS_RSAVE register slots, so the child restores it without a global snapshot reload (on
        // gfx90a vmcnt also counts the snapshot stores, so that reload serialised every pending store into the
        // walk). Columns past the slot budget fall back to the prefetched snapshot reload.
        __syncthreads();
        if (tid == 0) {
            int nslot = 0;
            for (int t = 0; t < n_tokens; t++) {
                bool need = false;
                for (int u = t + 2; u < n_tokens; u++) {
                    need |= p_s[u] == t;
                }
                r_s[t] = need && nslot < GDN_ROWS_RSAVE ? nslot++ : -1;
            }
        }
    }
    __syncthreads();

    float s0[tree_t ? RPL : 1];
    float pre[tree_t ? RPL : 1];
    float sv[tree_t ? GDN_ROWS_RSAVE : 1][tree_t ? RPL : 1];
    if constexpr (tree_t) {
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            s0[r]  = s[r];
            pre[r] = 0.0f;
        }
    }

    float part[GDN_ROWS_WMAX];
#pragma unroll
    for (int t = 0; t < GDN_ROWS_WMAX; t++) {
        if (t >= n_tokens) {
            break;
        }
        if constexpr (tree_t) {
            const int p = gdn_rows_uniform(p_s[t]);
            if (p != t - 1) {
                if (p < 0) {
#pragma unroll
                    for (int r = 0; r < RPL; r++) {
                        s[r] = s0[r];
                    }
                } else {
                    const int ks = gdn_rows_uniform(r_s[p]);
                    if (ks < 0) {
#pragma unroll
                        for (int r = 0; r < RPL; r++) {
                            s[r] = pre[r];
                        }
                    }
#pragma unroll
                    for (int kk = 0; kk < GDN_ROWS_RSAVE; kk++) {
                        if (ks == kk) {
#pragma unroll
                            for (int r = 0; r < RPL; r++) {
                                s[r] = sv[kk][r];
                            }
                        }
                    }
                }
            }
            // prefetch the next column's parent snapshot (stored by this thread at column pn <= t - 1) when that
            // parent has no register slot
            if (t + 1 < n_tokens) {
                const int pn = gdn_rows_uniform(p_s[t + 1]);
                if (pn >= 0 && pn != t && gdn_rows_uniform(r_s[pn]) < 0) {
                    const state_t * src = state + (int64_t) (n_tokens - 1 - pn) * state_slot_stride;
#pragma unroll
                    for (int c = 0; c < NCH; c++) {
                        gdn_rows_ld4(src + c * 4 * LPC + 4 * lane, pre + 4 * c);
                    }
                }
            }
        }

        float kr[RPL];
        float qr[RPL];
#pragma unroll
        for (int c = 0; c < NCH; c++) {
            const float4 k4 = *(const float4 *) &k_s[t][c * 4 * LPC + 4 * lane];
            const float4 q4 = *(const float4 *) &q_s[t][c * 4 * LPC + 4 * lane];
            kr[4 * c + 0] = k4.x; kr[4 * c + 1] = k4.y; kr[4 * c + 2] = k4.z; kr[4 * c + 3] = k4.w;
            qr[4 * c + 0] = q4.x; qr[4 * c + 1] = q4.y; qr[4 * c + 2] = q4.z; qr[4 * c + 3] = q4.w;
        }

        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            kv += s[r] * kr[r];
        }
        kv = gdn_rows_sum16(kv);

        const float g_val = eg_s[t];
        const float delta = (v_s[t][cl] - g_val * kv) * b_s[t];

        float ap = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            s[r] = g_val * s[r] + kr[r] * delta;
            ap  += s[r] * qr[r];
        }
        part[t] = ap;

        if constexpr (tree_t) {
            const int ks = gdn_rows_uniform(r_s[t]);
#pragma unroll
            for (int kk = 0; kk < GDN_ROWS_RSAVE; kk++) {
                if (ks == kk) {
#pragma unroll
                    for (int r = 0; r < RPL; r++) {
                        sv[kk][r] = s[r];
                    }
                }
            }
        }

        if constexpr (keep_rs_t) {
            const int slot = n_tokens - 1 - t;
            if (slot < K) {
                state_t * snap = state + (int64_t) slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < NCH; c++) {
                    gdn_rows_st4(snap + c * 4 * LPC + 4 * lane, s + 4 * c);
                }
            }
        }
    }

    // q-side reductions, off the serial chain (independent: they pipeline)
#pragma unroll
    for (int t = 0; t < GDN_ROWS_WMAX; t++) {
        if (t >= n_tokens) {
            break;
        }
        const float a = gdn_rows_sum16(part[t]);
        if (lane == 0) {
            dst[(int64_t) t * S_v * H] = gdn_store<state_t>(a * scale);
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < NCH; c++) {
            gdn_rows_st4(state + c * 4 * LPC + 4 * lane, s + 4 * c);
        }
    }
}

// Returns true when the rows kernel handled the op.
template <typename state_t>
static bool gdn_rows_try(const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        const state_t * s_d, state_t * dst_d, state_t * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs, bool kda, int K,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride,
        cudaStream_t stream, const int32_t * parent, bool force = false) {
    int lo, hi;
    gdn_rows_range(lo, hi);
    if (force) {
        lo = 1;
        hi = GDN_ROWS_WMAX;   // [GDN-FUSED-VARIANT r2] K=1 route under GGML_HIP_GDN_FUSED=1
    }
    if (n_tokens < lo || n_tokens > hi) {
        return false;
    }
    if (kda || (S_v != 64 && S_v != 128) || n_seqs > 65535) {
        return false;
    }
    // vector state access: 16-byte aligned base and slot stride (always true for ggml/cache buffers)
    if (std::is_same_v<state_t, float> &&
            (((uintptr_t) s_d) % 16 != 0 || ((uintptr_t) state_d) % 16 != 0 || state_slot_stride % 4 != 0)) {
        return false;
    }
    const bool keep_rs = K > 1;
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const dim3  grid(H, n_seqs, S_v / GDN_ROWS_CPB);
    const dim3  block(GDN_ROWS_THREADS, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
#define GDN_ROWS_ARGS lp, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, (int) n_tokens, sq1, sq2, sq3, \
        sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, parent
    if (parent != nullptr) {
        GGML_ASSERT(keep_rs);
        if (S_v == 128) {
            ggml_cuda_kernel_launch(gdn_rows_cuda<128, true, state_t, true>, GDN_ROWS_ARGS);
        } else {
            ggml_cuda_kernel_launch(gdn_rows_cuda<64, true, state_t, true>, GDN_ROWS_ARGS);
        }
    } else if (keep_rs) {
        if (S_v == 128) {
            ggml_cuda_kernel_launch(gdn_rows_cuda<128, true, state_t, false>, GDN_ROWS_ARGS);
        } else {
            ggml_cuda_kernel_launch(gdn_rows_cuda<64, true, state_t, false>, GDN_ROWS_ARGS);
        }
    } else {
        if (S_v == 128) {
            ggml_cuda_kernel_launch(gdn_rows_cuda<128, false, state_t, false>, GDN_ROWS_ARGS);
        } else {
            ggml_cuda_kernel_launch(gdn_rows_cuda<64, false, state_t, false>, GDN_ROWS_ARGS);
        }
    }
#undef GDN_ROWS_ARGS
    // route witness (gate packets grep it; also the build's strings content marker)
    if (g_gdn_rows_n_calls.fetch_add(1) == 0) {
        GGML_LOG_WARN("GDN_ROWS: ggml_cuda_gdn_rows DTREE-0-GDN-ROWS few-row gated delta-net ran (call 1, n_tokens %d, K %d, tree %d,"
                      " GGML_HIP_GDN_ROWS range %d..%d)\n", (int) n_tokens, K, (int) (parent != nullptr), lo, hi);
    }
    return true;
}

// =====================================================================================================
// [GDN-FUSED-VARIANT] fused float2-reduction few-row (verify) gated delta-net, n_tokens 1..16, one launch per op.
// Same contract and slot mapping as the recurrent kernel; non-KDA, non-tree, no replay, S_v 64/128.
// Per row t the old state S feeds ONE pass that accumulates both kv = S^T k_t and sq = S^T q_t; the two partial
// sums are reduced together (a float2 DPP butterfly: two independent chains interleaved), then
//     delta = beta (v - g kv),   attn = scale (g sq + delta (k_t . q_t)),   S <- g S + k_t delta^T
// (attn = scale S_new^T q expanded, so the q-side reduction no longer waits for the update). k_t . q_t is
// computed once per row in the LDS staging phase. 8 lanes own one state column (S_v/8 rows each: 3 DPP steps
// per reduction instead of 4), 256-thread blocks own 32 columns, so a 48-head np1 S_v128 op is 192 blocks.
// State I/O is vectorized for BOTH dtypes: F32 as float4 (2 x float2), BF16 as 8-byte packs of 4 bf16 (the
// recurrent/rows kernels move BF16 state one 2-byte element at a time). K = T snapshots (the bandwidth-bound
// ~3.1 MB/row F32 write) are issued as nontemporal (streaming) stores straight off the chain: the walk reads
// only LDS, so they never stall it, and they do not evict the q/k/v working set from L2.
// GGML_HIP_GDN_FUSED=1 enables it (default off: dispatch byte-identical to C18). Read per call, so a test can
// flip it between two launches in one process. Takes precedence over GGML_HIP_GDN_ROWS for the shapes it covers.
// =====================================================================================================
#define GDN_FU_WMAX    16
#define GDN_FU_LPC     8
#define GDN_FU_THREADS 256
#define GDN_FU_CPB     (GDN_FU_THREADS / GDN_FU_LPC)

static std::atomic<uint64_t> g_gdn_fused_n_calls{0};

uint64_t ggml_cuda_gdn_fused_n_calls() {
    return g_gdn_fused_n_calls.load();
}

static bool gdn_fused_enabled() {
#if defined(GGML_USE_HIP)
    const char * s = getenv("GGML_HIP_GDN_FUSED");
    return s != nullptr && s[0] == '1';
#else
    return false;
#endif
}

// pairwise sum over the 8 lanes of a DPP half-row for two values at once; every lane gets the bit-identical result
static __device__ __forceinline__ float2 gdn_fu_sum8x2(float2 x) {
#if defined(GGML_USE_HIP)
    x.x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x.x), 0xB1,  0xF, 0xF, true)); // quad_perm [1,0,3,2]
    x.y += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x.y), 0xB1,  0xF, 0xF, true));
    x.x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x.x), 0x4E,  0xF, 0xF, true)); // quad_perm [2,3,0,1]
    x.y += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x.y), 0x4E,  0xF, 0xF, true));
    x.x += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x.x), 0x141, 0xF, 0xF, true)); // row_half_mirror
    x.y += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x.y), 0x141, 0xF, 0xF, true));
    return x;
#else
    x.x = warp_reduce_sum<GDN_FU_LPC>(x.x);
    x.y = warp_reduce_sum<GDN_FU_LPC>(x.y);
    return x;
#endif
}

template <typename state_t>
static __device__ __forceinline__ void gdn_fu_ld4(const state_t * p, float * o) {
    if constexpr (std::is_same_v<state_t, float>) {
        const float4 x = *(const float4 *) p;
        o[0] = x.x; o[1] = x.y; o[2] = x.z; o[3] = x.w;
    } else {
        const uint2 x = *(const uint2 *) p;   // 4 bf16 in one 8-byte load
        const state_t * e = (const state_t *) &x;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            o[j] = gdn_load(e[j]);
        }
    }
}

template <typename state_t, bool nt>
static __device__ __forceinline__ void gdn_fu_st4(state_t * p, const float * o) {
    if constexpr (std::is_same_v<state_t, float>) {
        const float4 x = make_float4(o[0], o[1], o[2], o[3]);
#if defined(GGML_USE_HIP)
        if constexpr (nt) {
            typedef float gdn_fu_v4f __attribute__((ext_vector_type(4)));
            const gdn_fu_v4f xv = { o[0], o[1], o[2], o[3] };
            __builtin_nontemporal_store(xv, (gdn_fu_v4f *) p);
            return;
        }
#endif
        *(float4 *) p = x;
    } else {
        uint2 x;
        state_t * e = (state_t *) &x;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            e[j] = gdn_store<state_t>(o[j]);
        }
#if defined(GGML_USE_HIP)
        if constexpr (nt) {
            typedef unsigned int gdn_fu_v2u __attribute__((ext_vector_type(2)));
            const gdn_fu_v2u xv = { x.x, x.y };
            __builtin_nontemporal_store(xv, (gdn_fu_v2u *) p);
            return;
        }
#endif
        *(uint2 *) p = x;
    }
}

template <int S_v, bool keep_rs_t, typename state_t, int TPB, bool NT>
__global__ void __launch_bounds__(TPB, 2)
gdn_fused_cuda(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
               const float * __restrict__ g, const float * __restrict__ beta,
               const state_t * curr_state, state_t * dst, state_t * state,
               const int64_t H, const int n_tokens,
               const int64_t sq1, const int64_t sq2, const int64_t sq3,
               const int64_t sv1, const int64_t sv2, const int64_t sv3,
               const int64_t sb1, const int64_t sb2, const int64_t sb3,
               const uint3 neqk1_magic, const uint3 rq3_magic, const float scale,
               const int64_t state_slot_stride, const int K) {
    constexpr int LPC  = GDN_FU_LPC;
    constexpr int CPB  = TPB / GDN_FU_LPC;
    constexpr int RPL  = S_v / LPC;  // state rows per lane
    constexpr int NCH  = RPL / 4;    // 4-element chunks per lane: row i = c*4*LPC + 4*lane + e
    constexpr int WARP = ggml_cuda_get_physical_warp_size();
    static_assert(RPL % 4 == 0, "S_v must be a multiple of 4*LPC");

    __shared__ float q_s[GDN_FU_WMAX][S_v];
    __shared__ float k_s[GDN_FU_WMAX][S_v];
    __shared__ float v_s[GDN_FU_WMAX][CPB];
    __shared__ float eg_s[GDN_FU_WMAX];
    __shared__ float b_s[GDN_FU_WMAX];
    __shared__ float kq_s[GDN_FU_WMAX];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      tid      = threadIdx.x;
    const int      lane     = tid % LPC;
    const int      cl       = tid / LPC;
    const int      col0     = blockIdx.z * CPB;
    const int      col      = col0 + cl;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_head = ((int64_t) sequence * H + h_idx) * S_v * S_v;
    curr_state += state_head + (int64_t) col * S_v;
    state      += state_head + (int64_t) col * S_v;
    dst        += ((int64_t) sequence * n_tokens * H + h_idx) * S_v + col;

    ggml_cuda_pdl_sync();

    float s[RPL];
#pragma unroll
    for (int c = 0; c < NCH; c++) {
        gdn_fu_ld4(curr_state + c * 4 * LPC + 4 * lane, s + 4 * c);
    }

    const float * qh = q + iq3 * sq3 + iq1 * sq1;
    const float * kh = k + iq3 * sq3 + iq1 * sq1;
    for (int idx = tid; idx < n_tokens * S_v; idx += TPB) {
        const int t = idx / S_v;
        const int i = idx % S_v;
        q_s[t][i] = qh[t * sq2 + i];
        k_s[t][i] = kh[t * sq2 + i];
    }
    const float * vh = v + sequence * sv3 + h_idx * sv1 + col0;
    for (int idx = tid; idx < n_tokens * CPB; idx += TPB) {
        const int t = idx / CPB;
        const int c = idx % CPB;
        v_s[t][c] = vh[t * sv2 + c];
    }
    if (tid < n_tokens) {
        const int64_t gb = sequence * sb3 + tid * sb2 + h_idx * sb1;
        eg_s[tid] = expf(g[gb]);
        b_s[tid]  = beta[gb];
    }
    __syncthreads();
    // k_t . q_t, one wave per row
    {
        const int wv = tid / WARP;
        const int wl = tid % WARP;
        for (int t = wv; t < n_tokens; t += TPB / WARP) {
            float a = 0.0f;
            for (int i = wl; i < S_v; i += WARP) {
                a += k_s[t][i] * q_s[t][i];
            }
            a = warp_reduce_sum<WARP>(a);
            if (wl == 0) {
                kq_s[t] = a;
            }
        }
    }
    __syncthreads();

    float att[GDN_FU_WMAX];
#pragma unroll
    for (int t = 0; t < GDN_FU_WMAX; t++) {
        if (t >= n_tokens) {
            break;
        }
        float kr[RPL];
        float2 acc = make_float2(0.0f, 0.0f);   // (S^T k, S^T q) partials on the OLD state, one pass
#pragma unroll
        for (int c = 0; c < NCH; c++) {
            const float4 k4 = *(const float4 *) &k_s[t][c * 4 * LPC + 4 * lane];
            const float4 q4 = *(const float4 *) &q_s[t][c * 4 * LPC + 4 * lane];
            kr[4 * c + 0] = k4.x; kr[4 * c + 1] = k4.y; kr[4 * c + 2] = k4.z; kr[4 * c + 3] = k4.w;
            acc.x += s[4 * c + 0] * k4.x; acc.y += s[4 * c + 0] * q4.x;
            acc.x += s[4 * c + 1] * k4.y; acc.y += s[4 * c + 1] * q4.y;
            acc.x += s[4 * c + 2] * k4.z; acc.y += s[4 * c + 2] * q4.z;
            acc.x += s[4 * c + 3] * k4.w; acc.y += s[4 * c + 3] * q4.w;
        }
        acc = gdn_fu_sum8x2(acc);

        const float g_val = eg_s[t];
        const float delta = (v_s[t][cl] - g_val * acc.x) * b_s[t];
        att[t] = g_val * acc.y + delta * kq_s[t];

#pragma unroll
        for (int r = 0; r < RPL; r++) {
            s[r] = g_val * s[r] + kr[r] * delta;
        }

        if constexpr (keep_rs_t) {
            const int slot = n_tokens - 1 - t;
            if (slot < K) {
                state_t * snap = state + (int64_t) slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < NCH; c++) {
                    // slot 0 (the final state) is re-read by the next op: keep it cached
                    if (slot == 0) {
                        gdn_fu_st4<state_t, false>(snap + c * 4 * LPC + 4 * lane, s + 4 * c);
                    } else {
                        gdn_fu_st4<state_t, NT>(snap + c * 4 * LPC + 4 * lane, s + 4 * c);
                    }
                }
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < GDN_FU_WMAX; t++) {
            if (t >= n_tokens) {
                break;
            }
            dst[(int64_t) t * S_v * H] = gdn_store<state_t>(att[t] * scale);
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < NCH; c++) {
            gdn_fu_st4<state_t, false>(state + c * 4 * LPC + 4 * lane, s + 4 * c);
        }
    }
}

// Returns true when the fused kernel handled the op (GGML_HIP_GDN_FUSED=1 and a covered shape).
template <typename state_t>
static bool gdn_fused_try(const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        const state_t * s_d, state_t * dst_d, state_t * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs, bool kda, int K,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride,
        cudaStream_t stream) {
    if (!gdn_fused_enabled()) {
        return false;
    }
    if (n_tokens < 1 || n_tokens > GDN_FU_WMAX || kda || (S_v != 64 && S_v != 128) || n_seqs > 65535) {
        return false;
    }
    const size_t esz = sizeof(state_t);
    if (((uintptr_t) s_d) % (4 * esz) != 0 || ((uintptr_t) state_d) % (4 * esz) != 0 || state_slot_stride % 4 != 0) {
        return false;
    }
    const bool keep_rs = K > 1;
    // [r2] K = 1: the DTREE-0 rows kernel (16 lanes/column, 512 threads) measured 6-20 % faster than the fused kernel
    // at T >= 4 in ak-gdn-fused-gpu-20261123; route it there. K > 1: the fused kernel (it won every K = T cell).
    if (!keep_rs && gdn_rows_try<state_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, S_v, H, n_tokens, n_seqs, kda, K,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, stream, nullptr, true)) {
        if (g_gdn_fused_n_calls.fetch_add(1) == 0) {
            GGML_LOG_WARN("GDN_FUSED: ggml_cuda_gdn_fused GDN-FUSED-VARIANT r2 route K=1 -> rows kernel (call 1, n_tokens %d,"
                          " GGML_HIP_GDN_FUSED=1)\n", (int) n_tokens);
        }
        return true;
    }
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
#define GDN_FU_ARGS lp, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, (int) n_tokens, sq1, sq2, sq3, \
        sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K
    if (keep_rs) {
        // [r2] K = T is store-bound (one full state per row): 128-thread blocks (16 columns) double the waves in flight
        // to hide the snapshot store latency. Snapshot slots > 0: nontemporal by default (r1);
        // GGML_HIP_GDN_FUSED_NT=0 selects plain cached 16-byte stores (screen arm).
        constexpr int TPB = GDN_FU_THREADS / 2;
        const char * nte = getenv("GGML_HIP_GDN_FUSED_NT");
        const bool   nt  = !(nte != nullptr && nte[0] == '0');
        const dim3   grid(H, n_seqs, S_v / (TPB / GDN_FU_LPC));
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, dim3(TPB, 1, 1), 0, stream);
        if (S_v == 128) {
            if (nt) { ggml_cuda_kernel_launch(gdn_fused_cuda<128, true, state_t, TPB, true>,  GDN_FU_ARGS); }
            else    { ggml_cuda_kernel_launch(gdn_fused_cuda<128, true, state_t, TPB, false>, GDN_FU_ARGS); }
        } else {
            if (nt) { ggml_cuda_kernel_launch(gdn_fused_cuda<64, true, state_t, TPB, true>,  GDN_FU_ARGS); }
            else    { ggml_cuda_kernel_launch(gdn_fused_cuda<64, true, state_t, TPB, false>, GDN_FU_ARGS); }
        }
    } else {
        const dim3 grid(H, n_seqs, S_v / GDN_FU_CPB);
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, dim3(GDN_FU_THREADS, 1, 1), 0, stream);
        if (S_v == 128) {
            ggml_cuda_kernel_launch(gdn_fused_cuda<128, false, state_t, GDN_FU_THREADS, false>, GDN_FU_ARGS);
        } else {
            ggml_cuda_kernel_launch(gdn_fused_cuda<64, false, state_t, GDN_FU_THREADS, false>, GDN_FU_ARGS);
        }
    }
#undef GDN_FU_ARGS
    // route witness (gate packets grep it; also the build's strings content marker)
    if (g_gdn_fused_n_calls.fetch_add(1) == 0) {
        GGML_LOG_WARN("GDN_FUSED: ggml_cuda_gdn_fused GDN-FUSED-VARIANT float2-reduction gated delta-net ran (call 1, n_tokens %d,"
                      " K %d, state %s, GGML_HIP_GDN_FUSED=1)\n", (int) n_tokens, K, esz == 4 ? "f32" : "bf16");
    }
    return true;
}

template <bool KDA, bool keep_rs_t, typename state_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const state_t * s_d,
        state_t * dst_d, state_t * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream, const int32_t * parent = nullptr,
        const gdn_rep_args & rep = gdn_rep_args()) {
    // prefill with n_tokens >= GGML_CUDA_GDN_CHUNKED_MIN_TOKENS on CDNA takes gdn_chunked_try (above) instead
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            if constexpr (keep_rs_t) {
                if (parent != nullptr) {
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, true, state_t, true>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, parent, rep);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, (const int32_t *) nullptr, rep);
            break;
        case 32:
            if constexpr (keep_rs_t) {
                if (parent != nullptr) {
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, true, state_t, true>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, parent, rep);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, (const int32_t *) nullptr, rep);
            break;
        case 64: {
            if constexpr (keep_rs_t) {
                if (parent != nullptr) {
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, true, state_t, true>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, parent, rep);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, (const int32_t *) nullptr, rep);
            break;
        }
        case 128: {
            if constexpr (keep_rs_t) {
                if (parent != nullptr) {
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, true, state_t, true>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, parent, rep);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, (const int32_t *) nullptr, rep);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// 4-way (kda × keep_rs) dispatch, templated on the recurrent-state dtype.
template <typename state_t>
static void gdn_dispatch(bool kda, bool keep_rs,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const state_t * s_d, state_t * dst_d, state_t * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride, int K, cudaStream_t stream,
        const int32_t * parent = nullptr, const gdn_rep_args & rep = gdn_rep_args()) {
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, parent);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, nullptr, rep);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, parent);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, nullptr, rep);
        }
    }
}

// non-tree route: chunked prefill -> GDN-ROWS -> recurrent kernel. [RS-REPLAY] with a replay, the recurrent kernel
// replays in registers (fused, one state read and one state write). The chunked and GDN-ROWS kernels have no replay
// loop: a non-keep replay at or above the chunked threshold first runs a replay-only recurrent launch (n_tokens 0)
// into a pool scratch state, and the chunked kernel then starts from it (PR #31669's replay-only launch before
// chunked prefill); otherwise a replay ubatch takes the recurrent kernel.
template <typename state_t>
static void gdn_run(ggml_backend_cuda_context & ctx, const ggml_tensor * src_q, const ggml_tensor * src_k,
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        const state_t * s_d, state_t * dst_d, state_t * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs, bool kda, bool keep_rs, int K,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride,
        cudaStream_t stream, const gdn_rep_args & rep) {
    if (rep.pk != nullptr) {
        if (!rep.keep && n_tokens >= ggml_cuda_gdn_chunked_min_tokens()) {
            ggml_cuda_pool_alloc<state_t> s_rep(ctx.pool(), S_v * S_v * H * n_seqs);
            gdn_dispatch<state_t>(kda, false, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, s_rep.get(),
                S_v, H, /*n_tokens=*/0, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
                S_v * S_v * H * n_seqs, 1, stream, nullptr, rep);
            if (gdn_chunked_try<state_t>(ctx, src_q, src_k, q_d, k_d, v_d, g_d, b_d, s_rep.get(), dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, kda, K, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride)) {
                return;
            }
            gdn_dispatch<state_t>(kda, keep_rs, q_d, k_d, v_d, g_d, b_d, s_rep.get(), dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
            return;
        }
        gdn_dispatch<state_t>(kda, keep_rs, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream,
            nullptr, rep);
        return;
    }
    if (gdn_chunked_try<state_t>(ctx, src_q, src_k, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, kda, K, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride)) {
        return;
    }
    if (gdn_fused_try<state_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, kda, K, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
            state_slot_stride, stream)) {
        return;
    }
    if (gdn_rows_try<state_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, kda, K, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
            state_slot_stride, stream, nullptr)) {
        return;
    }
    gdn_dispatch<state_t>(kda, keep_rs, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    // [TAG_GDN_STATE_BF16] state src and dst carry the recurrent state; both must share dtype.
    GGML_ASSERT(src_state->type == dst->type);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_BF16);

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // [RS-REPLAY] replay args (op param 1 = n_pend > 0; src[6] is otherwise the tree parent links)
    gdn_rep_args rep;
    const bool replay = ggml_get_op_params_i32(dst, 1) > 0;
    if (replay) {
        const ggml_tensor * pk = dst->src[6], * pv = dst->src[7], * pgb = dst->src[8], * nr = dst->src[9];
        GGML_ASSERT(K == 1 && cache == nullptr);
        GGML_ASSERT(pk->type == GGML_TYPE_F32 && pv->type == GGML_TYPE_F32 && pgb->type == GGML_TYPE_F32 && nr->type == GGML_TYPE_I32);
        rep.pk = (const float *) pk->data; rep.pv = (const float *) pv->data; rep.pgb = (const float *) pgb->data;
        rep.n_rep = (const int32_t *) nr->data;
        rep.spk1 = pk->nb[1] / sizeof(float);  rep.spk2 = pk->nb[2] / sizeof(float);  rep.spk3 = pk->nb[3] / sizeof(float);
        rep.spv1 = pv->nb[1] / sizeof(float);  rep.spv2 = pv->nb[2] / sizeof(float);  rep.spv3 = pv->nb[3] / sizeof(float);
        rep.sgb1 = pgb->nb[1] / sizeof(float); rep.sgb2 = pgb->nb[2] / sizeof(float); rep.sgb3 = pgb->nb[3] / sizeof(float);
        rep.n_pend = ggml_get_op_params_i32(dst, 1);
        rep.keep   = ggml_get_op_params_i32(dst, 2) != 0;
    }

    if (dst->type == GGML_TYPE_F32) {
        float * dst_d             = (float *) dst->data;
        float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
        int64_t state_slot_stride = S_v * S_v * H * n_seqs;
        if (cache != nullptr) {
            state_d           = cache->data;
            state_slot_stride = cache->slot_stride;
        }

        // [AK-DTREE-2] tree-walk variant: recurrent kernel only (never chunked), F32 state.
        if (!replay && dst->src[6] != nullptr) {
            GGML_ASSERT(keep_rs && K == n_tokens);
            GGML_ASSERT(dst->src[6]->type == GGML_TYPE_I32 && ggml_is_contiguous(dst->src[6]));
            if (gdn_rows_try<float>(q_d, k_d, v_d, g_d, b_d, (const float *) src_state->data, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, kda, K, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
                    state_slot_stride, stream, (const int32_t *) dst->src[6]->data)) {
                return;
            }
            gdn_dispatch<float>(kda, true, q_d, k_d, v_d, g_d, b_d,
                (const float *) src_state->data, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream,
                (const int32_t *) dst->src[6]->data);
            return;
        }
        gdn_run<float>(ctx, src_q, src_k, q_d, k_d, v_d, g_d, b_d, (const float *) src_state->data, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, kda, keep_rs, K, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
            state_slot_stride, stream, rep);
    } else {
        GGML_ASSERT(cache == nullptr);
        GGML_ASSERT(replay || dst->src[6] == nullptr); // [AK-DTREE-2] tree walk is F32-state only
        nv_bfloat16 * dst_d       = (nv_bfloat16 *) dst->data;
        nv_bfloat16 * state_d     = dst_d + S_v * H * n_tokens * n_seqs;
        int64_t state_slot_stride = S_v * S_v * H * n_seqs;

        gdn_run<nv_bfloat16>(ctx, src_q, src_k, q_d, k_d, v_d, g_d, b_d, (const nv_bfloat16 *) src_state->data, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, kda, keep_rs, K, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
            state_slot_stride, stream, rep);
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
