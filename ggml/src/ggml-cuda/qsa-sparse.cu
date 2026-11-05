// AK-ASM-H12: fused QSA sparse attention (see qsa-sparse.cuh).
//
// Semantics. The composition in qwen4exp build_attn_qsa computes, per query token t, head h:
//   out = softmax_j( scale * q.k_j + base_mask[t, j] + (j in top_k[t] ? 0 : -inf) ) . V
// i.e. attention restricted to the cells named by top_k[t], with the ordinary KQ mask added. top_k
// rows hold distinct cell indices (ggml_top_k), so iterating over them is exactly the same sum.
//
// Kernel. One workgroup of D threads per (query token, kv head, stream[, split]). The G = n_head/n_head_kv
// q heads that share the kv head are processed together, so every gathered K/V row is read once for
// all G heads (GQA packing, after AITER qsa.py _qsa_sparse_paged_gqa_kernel). Cells are processed in
// chunks of D: score phase (thread = cell: gather K row, G dot products), online-softmax phase
// (wave = head), value phase (thread = output dim: gather V row element, G FMAs). Decode/verify
// (few query tokens) splits the chunk loop across workgroups and merges partials in a second kernel.

#include "qsa-sparse.cuh"

#include <cmath>
#include <cstdlib>
#include <cstdio>

// ---------------------------------------------------------------------------------------------
// env

static int qsa_env_int(const char * name, int def) {
    const char * s = getenv(name);
    return s ? atoi(s) : def;
}

static bool qsa_enabled() {
    static const bool v = qsa_env_int("GGML_CUDA_QSA_SPARSE", 1) != 0;
    return v;
}

// 0 = no limit. lets the gate keep the dense FA for big prefill ubatches if those lose
static int qsa_max_tokens() {
    static const int v = qsa_env_int("GGML_CUDA_QSA_SPARSE_MAX_TOKENS", 0);
    return v;
}

static bool qsa_debug() {
    static const bool v = qsa_env_int("GGML_CUDA_QSA_SPARSE_DEBUG", 0) != 0;
    return v;
}

// ---------------------------------------------------------------------------------------------
// pattern match

static const ggml_tensor * qsa_strip_views(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE)) {
        t = t->src[0];
    }
    return t;
}

static const ggml_tensor * qsa_strip_views_cpy(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_CPY || t->op == GGML_OP_CONT)) {
        t = t->src[0];
    }
    return t;
}

static bool qsa_is_fill(const ggml_tensor * t, float value) {
    if (!t || t->op != GGML_OP_FILL) {
        return false;
    }
    const float c = ggml_get_op_params_f32(t, 0);
    if (std::isinf(value)) {
        return std::isinf(c) && ((c < 0) == (value < 0));
    }
    return c == value;
}

static bool qsa_on_device(const ggml_tensor * t) {
    return t && t->data && t->buffer && !ggml_backend_buffer_is_host(t->buffer);
}

struct qsa_match {
    const ggml_tensor * top_k; // I32 [width, n_tps, n_stream, 1] (view)
    const ggml_tensor * mask;  // F16 [n_kv, >=n_tps, 1, n_stream or 1]
};

// FA mask = ADD(VIEW(SET_ROWS(VIEW(FILL(-inf)), FILL(0), VIEW(top_k))), kq_mask), possibly behind a cast/cont
static bool qsa_match_mask(const ggml_tensor * fa, qsa_match & m) {
    const ggml_tensor * mask = fa->src[3];
    if (!mask) {
        return false;
    }
    const ggml_tensor * add = qsa_strip_views_cpy(mask);
    if (!add || add->op != GGML_OP_ADD) {
        return false;
    }
    for (int side = 0; side < 2; ++side) {
        const ggml_tensor * sr   = qsa_strip_views(add->src[side]);
        const ggml_tensor * base = add->src[1 - side];
        if (!sr || sr->op != GGML_OP_SET_ROWS || !base) {
            continue;
        }
        if (!qsa_is_fill(qsa_strip_views(sr->src[0]), -INFINITY)) {
            continue;
        }
        if (!qsa_is_fill(qsa_strip_views(sr->src[1]), 0.0f)) {
            continue;
        }
        // rows of size 1: [1, n_kv, n_batch, n_stream]
        if (sr->ne[0] != 1) {
            continue;
        }
        m.top_k = sr->src[2];
        m.mask  = base;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// kernels

struct qsa_params {
    const char * q;      // F32 [D, n_tps, n_head, n_stream]
    const char * k;      // [D, n_kv, n_head_kv, n_stream]
    const char * v;
    const char * mask;   // F16
    const char * top_k;  // I32
    float      * dst;    // F32 [D, n_head, n_tps, n_stream]
    float      * part_o; // F32 [n_split][rows][D]
    float2     * part_ml;// [n_split][rows]

    int     width;
    int     n_tps;
    int     n_head;
    int     n_head_kv;
    int     n_stream;
    int     n_split;
    int     chunks_per_split;
    int     mask_ne3;
    float   scale;

    int64_t nbq1, nbq2, nbq3;
    int64_t nbk1, nbk2, nbk3;
    int64_t nbv1, nbv2, nbv3;
    int64_t nbm1, nbm3;
    int64_t nbt1, nbt2;
};

template <int D, int G, ggml_type type>
static __device__ __forceinline__ void qsa_score_row(const char * row, const float (*sq)[D], float * s) {
    if constexpr (type == GGML_TYPE_F16) {
        const float4 * r4 = (const float4 *) row; // 8 halves per load
#pragma unroll 2
        for (int i = 0; i < D/8; ++i) {
            const float4 raw = r4[i];
            const half2 * h2 = (const half2 *) &raw;
            float kf[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __half22float2(h2[e]);
                kf[2*e + 0] = f.x;
                kf[2*e + 1] = f.y;
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float4 qa = *(const float4 *) &sq[g][8*i + 0];
                const float4 qb = *(const float4 *) &sq[g][8*i + 4];
                s[g] += qa.x*kf[0] + qa.y*kf[1] + qa.z*kf[2] + qa.w*kf[3]
                      + qb.x*kf[4] + qb.y*kf[5] + qb.z*kf[6] + qb.w*kf[7];
            }
        }
    } else { // Q8_0: 34-byte blocks {half d; int8 qs[32]}, 2-byte aligned
#pragma unroll 1
        for (int b = 0; b < D/QK8_0; ++b) {
            const block_q8_0 * blk = (const block_q8_0 *) (row + b*sizeof(block_q8_0));
            const float d = __half2float(blk->d);
            float sb[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                sb[g] = 0.0f;
            }
            const uint16_t * qs16 = (const uint16_t *) blk->qs;
#pragma unroll 2
            for (int i = 0; i < QK8_0/4; ++i) {
                const uint16_t lo = qs16[2*i + 0];
                const uint16_t hi = qs16[2*i + 1];
                const float k0 = (float) (int8_t) (lo & 0xFF);
                const float k1 = (float) (int8_t) (lo >> 8);
                const float k2 = (float) (int8_t) (hi & 0xFF);
                const float k3 = (float) (int8_t) (hi >> 8);
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float4 qa = *(const float4 *) &sq[g][b*QK8_0 + 4*i];
                    sb[g] += qa.x*k0 + qa.y*k1 + qa.z*k2 + qa.w*k3;
                }
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                s[g] += d*sb[g];
            }
        }
    }
}

template <int D, ggml_type type>
static __device__ __forceinline__ float qsa_v_elem(const char * row, const int t) {
    if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[t]);
    } else {
        const block_q8_0 * blk = (const block_q8_0 *) row + t/QK8_0;
        return __half2float(blk->d) * (float) blk->qs[t % QK8_0];
    }
}

template <int D, int G, ggml_type type>
// min 2 workgroups (8 waves) per CU: <= 128 VGPRs, so a second workgroup hides the gather latency
static __global__ void __launch_bounds__(D, 2) k_qsa_sparse_attn(const qsa_params p) {
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    constexpr int NW = D/WS;

    __shared__ __align__(16) float sq[G][D];
    __shared__ float sp[G][D];
    __shared__ int   sidx[D];
    __shared__ float sM[G];
    __shared__ float sL[G];
    __shared__ float salpha[G];

    const int t     = threadIdx.x;
    const int lane  = t % WS;
    const int warp  = t / WS;
    const int token = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int st    = blockIdx.z / p.n_split;
    const int split = blockIdx.z % p.n_split;

    const char * kbase = p.k + kvh*p.nbk2 + st*p.nbk3;
    const char * vbase = p.v + kvh*p.nbv2 + st*p.nbv3;
    const half * mrow  = (const half *) (p.mask + token*p.nbm1 + (st % p.mask_ne3)*p.nbm3);
    const int  * trow  = (const int  *) (p.top_k + token*p.nbt1 + st*p.nbt2);

#pragma unroll
    for (int g = 0; g < G; ++g) {
        sq[g][t] = *(const float *) (p.q + token*p.nbq1 + (kvh*G + g)*p.nbq2 + st*p.nbq3 + t*sizeof(float));
    }
    if (t < G) {
        sM[t] = -INFINITY;
        sL[t] = 0.0f;
    }

    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        acc[g] = 0.0f;
    }

    __syncthreads();

    const int c_begin = split*p.chunks_per_split*D;
    const int c_end   = min(p.width, (split + 1)*p.chunks_per_split*D);

    for (int c0 = c_begin; c0 < c_end; c0 += D) {
        // ---- score: thread = cell
        {
            const int c = c0 + t;
            float s[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                s[g] = 0.0f;
            }
            int j = -1;
            if (c < c_end) {
                j = trow[c];
                const float m = __half2float(mrow[j]);
                if (m == -INFINITY) {
                    j = -1; // masked by the base KQ mask: no K read needed
                } else {
                    qsa_score_row<D, G, type>(kbase + (int64_t) j*p.nbk1, sq, s);
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        s[g] = s[g]*p.scale + m;
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                sp[g][t] = j >= 0 ? s[g] : -INFINITY;
            }
            sidx[t] = j;
        }
        __syncthreads();

        // ---- online softmax: wave = head
        for (int g = warp; g < G; g += NW) {
            float mx = -INFINITY;
            for (int i = lane; i < D; i += WS) {
                mx = fmaxf(mx, sp[g][i]);
            }
            mx = warp_reduce_max<WS>(mx);
            const float mold = sM[g];
            const float mnew = fmaxf(mold, mx);
            float alpha = 1.0f;
            float sum   = 0.0f;
            if (mnew == -INFINITY) {
                for (int i = lane; i < D; i += WS) {
                    sp[g][i] = 0.0f;
                }
            } else {
                alpha = expf(mold - mnew);
                for (int i = lane; i < D; i += WS) {
                    const float e = expf(sp[g][i] - mnew);
                    sp[g][i] = e;
                    sum += e;
                }
            }
            sum = warp_reduce_sum<WS>(sum);
            if (lane == 0) {
                sM[g]     = mnew;
                sL[g]     = sL[g]*alpha + sum;
                salpha[g] = alpha;
            }
        }
        __syncthreads();

        // ---- value: thread = output dim
#pragma unroll
        for (int g = 0; g < G; ++g) {
            acc[g] *= salpha[g];
        }
        const int nc = min(D, c_end - c0);
        for (int i = 0; i < nc; ++i) {
            const int j = sidx[i]; // uniform across the workgroup
            if (j < 0) {
                continue;
            }
            const float vv = qsa_v_elem<D, type>(vbase + (int64_t) j*p.nbv1, t);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                acc[g] += sp[g][i]*vv;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int h   = kvh*G + g;
        const int64_t row = h + (int64_t) p.n_head*(token + (int64_t) p.n_tps*st);
        if (p.n_split == 1) {
            const float L = sL[g];
            p.dst[row*D + t] = L > 0.0f ? acc[g]/L : 0.0f;
        } else {
            const int64_t n_rows = (int64_t) p.n_head*p.n_tps*p.n_stream;
            p.part_o[(split*n_rows + row)*D + t] = acc[g];
            if (t == 0) {
                p.part_ml[split*n_rows + row] = make_float2(sM[g], sL[g]);
            }
        }
    }
}

template <int D>
static __global__ void __launch_bounds__(D, 1) k_qsa_sparse_combine(
        const float * part_o, const float2 * part_ml, float * dst, const int n_split, const int64_t n_rows) {
    const int64_t row = blockIdx.x;
    const int     t   = threadIdx.x;

    float M = -INFINITY;
    for (int s = 0; s < n_split; ++s) {
        M = fmaxf(M, part_ml[s*n_rows + row].x);
    }
    float L = 0.0f;
    float o = 0.0f;
    if (M != -INFINITY) {
        for (int s = 0; s < n_split; ++s) {
            const float2 ml = part_ml[s*n_rows + row];
            if (ml.x == -INFINITY) {
                continue;
            }
            const float w = expf(ml.x - M);
            L += w*ml.y;
            o += w*part_o[(s*n_rows + row)*D + t];
        }
    }
    dst[row*D + t] = L > 0.0f ? o/L : 0.0f;
}

// ---------------------------------------------------------------------------------------------
// host

template <int D, int G, ggml_type type>
static void qsa_launch(ggml_backend_cuda_context & ctx, qsa_params p, const int n_chunks) {
    cudaStream_t stream = ctx.stream();

    const int nsm         = ggml_cuda_info().devices[ctx.device].nsm;
    const int64_t n_base  = (int64_t) p.n_tps*p.n_head_kv*p.n_stream;
    const int64_t target  = 4*(int64_t) nsm;
    int n_split = 1;
    if (n_base < target) {
        n_split = (int) std::min<int64_t>(n_chunks, (target + n_base - 1)/n_base);
    }
    p.chunks_per_split = (n_chunks + n_split - 1)/n_split;
    n_split = (n_chunks + p.chunks_per_split - 1)/p.chunks_per_split;
    p.n_split = n_split;

    const int64_t n_rows = (int64_t) p.n_head*p.n_tps*p.n_stream;

    ggml_cuda_pool_alloc<float>  part_o (ctx.pool());
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool());
    if (n_split > 1) {
        part_o.alloc(n_split*n_rows*D);
        part_ml.alloc(n_split*n_rows);
        p.part_o  = part_o.get();
        p.part_ml = part_ml.get();
    }

    const dim3 grid(p.n_tps, p.n_head_kv, p.n_stream*n_split);
    k_qsa_sparse_attn<D, G, type><<<grid, D, 0, stream>>>(p);

    if (n_split > 1) {
        k_qsa_sparse_combine<D><<<n_rows, D, 0, stream>>>(p.part_o, p.part_ml, p.dst, n_split, n_rows);
    }
}

template <int D, ggml_type type>
static bool qsa_dispatch_g(ggml_backend_cuda_context & ctx, const qsa_params & p, const int G, const int n_chunks) {
    switch (G) {
        case  1: qsa_launch<D,  1, type>(ctx, p, n_chunks); return true;
        case  2: qsa_launch<D,  2, type>(ctx, p, n_chunks); return true;
        case  4: qsa_launch<D,  4, type>(ctx, p, n_chunks); return true;
        case  8: qsa_launch<D,  8, type>(ctx, p, n_chunks); return true;
        case 12: qsa_launch<D, 12, type>(ctx, p, n_chunks); return true;
        case 16: qsa_launch<D, 16, type>(ctx, p, n_chunks); return true;
        default: return false;
    }
}

bool ggml_cuda_qsa_sparse_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!qsa_enabled()) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    if (dst->src[4] != nullptr) { // sinks
        return false;
    }

    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }

    qsa_match m;
    if (!qsa_match_mask(dst, m)) {
        return false;
    }
    const ggml_tensor * tk   = m.top_k;
    const ggml_tensor * mask = m.mask;

    const int64_t D         = q->ne[0];
    const int64_t n_tps     = q->ne[1];
    const int64_t n_head    = q->ne[2];
    const int64_t n_stream  = q->ne[3];
    const int64_t n_kv      = k->ne[1];
    const int64_t n_head_kv = k->ne[2];

    if (q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (k->type != v->type || (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0)) {
        return false;
    }
    if (D != 256 || k->ne[0] != D || v->ne[0] != D || v->ne[1] != n_kv || v->ne[2] != n_head_kv) {
        return false;
    }
    if (k->ne[3] != n_stream || v->ne[3] != n_stream || n_head % n_head_kv != 0) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type)) {
        return false;
    }
    if (k->type == GGML_TYPE_F16) {
        // 16-byte K row loads
        if (((uintptr_t) k->data % 16) != 0 || k->nb[1] % 16 != 0 || k->nb[2] % 16 != 0 || k->nb[3] % 16 != 0) {
            return false;
        }
    }
    if (tk->type != GGML_TYPE_I32 || tk->nb[0] != sizeof(int32_t) || tk->ne[1] != n_tps || tk->ne[2] != n_stream || tk->ne[3] != 1) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(half) || mask->ne[0] != n_kv || mask->ne[1] < n_tps ||
            mask->ne[2] != 1 || n_stream % mask->ne[3] != 0) {
        return false;
    }
    if (!qsa_on_device(tk) || !qsa_on_device(mask)) {
        return false;
    }

    const int64_t width = tk->ne[0];
    if (width < 1 || width > INT32_MAX/2) {
        return false;
    }
    // nothing to skip: every cell is selected, the dense FA is the better kernel
    if (width >= n_kv) {
        return false;
    }
    const int max_tok = qsa_max_tokens();
    if (max_tok > 0 && n_tps*n_stream > max_tok) {
        return false;
    }

    qsa_params p = {};
    p.q        = (const char *) q->data;
    p.k        = (const char *) k->data;
    p.v        = (const char *) v->data;
    p.mask     = (const char *) mask->data;
    p.top_k    = (const char *) tk->data;
    p.dst      = (float *) dst->data;
    p.width    = (int) width;
    p.n_tps    = (int) n_tps;
    p.n_head   = (int) n_head;
    p.n_head_kv = (int) n_head_kv;
    p.n_stream = (int) n_stream;
    p.mask_ne3 = (int) mask->ne[3];
    p.scale    = scale;
    p.nbq1 = q->nb[1];    p.nbq2 = q->nb[2];    p.nbq3 = q->nb[3];
    p.nbk1 = k->nb[1];    p.nbk2 = k->nb[2];    p.nbk3 = k->nb[3];
    p.nbv1 = v->nb[1];    p.nbv2 = v->nb[2];    p.nbv3 = v->nb[3];
    p.nbm1 = mask->nb[1]; p.nbm3 = mask->nb[3];
    p.nbt1 = tk->nb[1];   p.nbt2 = tk->nb[2];

    const int G        = (int) (n_head/n_head_kv);
    const int n_chunks = (int) ((width + D - 1)/D);

    bool ok;
    if (k->type == GGML_TYPE_F16) {
        ok = qsa_dispatch_g<256, GGML_TYPE_F16>(ctx, p, G, n_chunks);
    } else {
        ok = qsa_dispatch_g<256, GGML_TYPE_Q8_0>(ctx, p, G, n_chunks);
    }

    if (ok) {
        // path witness for the gate: emitted once per process, on the first fused call
        static int n_calls = 0;
        if (n_calls++ == 0) {
            GGML_LOG_INFO("ggml_cuda_qsa_sparse: fused QSA sparse attention ran (call 1)\n");
        }
        if (qsa_debug() && n_calls <= 8) {
            GGML_LOG_INFO("%s: %s n_tps=%lld n_head=%lld/%lld n_kv=%lld width=%lld n_stream=%lld type=%s\n",
                __func__, dst->name, (long long) n_tps, (long long) n_head, (long long) n_head_kv, (long long) n_kv,
                (long long) width, (long long) n_stream, ggml_type_name(k->type));
        }
    }
    if (ok) {
        CUDA_CHECK(cudaGetLastError());
    }
    return ok;
}
