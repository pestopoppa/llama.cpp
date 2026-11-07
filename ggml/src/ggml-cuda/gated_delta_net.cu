#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

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

template <int S_v, bool KDA, bool keep_rs_t, typename state_t>
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
                                     int           K) {
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

    for (int t = 0; t < n_tokens; t++) {
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
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = gdn_store<state_t>(s_shard[r]);
        }
    }
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
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
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
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, state_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
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
        int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
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

    if (dst->type == GGML_TYPE_F32) {
        float * dst_d             = (float *) dst->data;
        float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
        int64_t state_slot_stride = S_v * S_v * H * n_seqs;
        if (cache != nullptr) {
            state_d           = cache->data;
            state_slot_stride = cache->slot_stride;
        }

        if (gdn_chunked_try<float>(ctx, src_q, src_k, q_d, k_d, v_d, g_d, b_d, (const float *) src_state->data, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, kda, K, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride)) {
            return;
        }
        gdn_dispatch<float>(kda, keep_rs, q_d, k_d, v_d, g_d, b_d,
            (const float *) src_state->data, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
    } else {
        GGML_ASSERT(cache == nullptr);
        nv_bfloat16 * dst_d       = (nv_bfloat16 *) dst->data;
        nv_bfloat16 * state_d     = dst_d + S_v * H * n_tokens * n_seqs;
        int64_t state_slot_stride = S_v * S_v * H * n_seqs;

        if (gdn_chunked_try<nv_bfloat16>(ctx, src_q, src_k, q_d, k_d, v_d, g_d, b_d, (const nv_bfloat16 *) src_state->data,
                dst_d, state_d, S_v, H, n_tokens, n_seqs, kda, K, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
                state_slot_stride)) {
            return;
        }
        gdn_dispatch<nv_bfloat16>(kda, keep_rs, q_d, k_d, v_d, g_d, b_d,
            (const nv_bfloat16 *) src_state->data, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
