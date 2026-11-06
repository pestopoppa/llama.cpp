#include "ak-fused.cuh"
#include "cpy-utils.cuh"
#include "unary.cuh"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <atomic>
#include <vector>

// ---------------------------------------------------------------------------------------------
// switches
// ---------------------------------------------------------------------------------------------

static bool ak_env_enabled(const char * name) {
    const char * v = getenv(name);
    return v == nullptr || atoi(v) != 0; // default ON, "0" disables
}

// C7: AK-ASM-H14 was DROPPED at its gate; it stays compiled but default OFF (GGML_CUDA_AK_H14=1 opts in).
static bool ak_env_opt_in(const char * name) {
    const char * v = getenv(name);
    return v != nullptr && atoi(v) != 0;
}

static void ak_log_once(std::atomic<bool> & flag, const char * what, const ggml_tensor * t) {
    static const bool do_log = getenv("GGML_CUDA_AK_FUSE_LOG") != nullptr && atoi(getenv("GGML_CUDA_AK_FUSE_LOG")) != 0;
    if (do_log && !flag.exchange(true)) {
        GGML_LOG_INFO("ggml_cuda_ak: %s fused (first hit at node %s)\n", what, t->name);
    }
}

// ---------------------------------------------------------------------------------------------
// H14: rms_norm * w -> rope [-> set_rows]
// ---------------------------------------------------------------------------------------------

struct ak_rope_corr_dims { float v[2]; };
struct ak_mrope_sections { int v[4]; };

// verbatim copy of rope.cu rope_yarn_ramp / rope_yarn<forward=true> so results match the unfused rope
static __device__ float ak_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static __device__ void ak_rope_yarn(
        const float theta_extrap, const float freq_scale, const ak_rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = ak_rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

enum ak_rope_kind { AK_ROPE_NORM = 0, AK_ROPE_NEOX = 1, AK_ROPE_MROPE = 2, AK_ROPE_IMROPE = 3 };
enum ak_out_kind  { AK_OUT_F32 = 0, AK_OUT_SR_F32 = 1, AK_OUT_SR_F16 = 2, AK_OUT_SR_Q8_0 = 3 };

struct ak_h14_params {
    const float * x;
    int64_t s01, s02, s03;   // x strides (elements)
    int ne0, ne1, ne2;       // head dim, heads, tokens
    const float * w;
    float eps;

    void * dst;
    int64_t s1, s2, s3;      // rope dst strides (elements, F32 out)
    const int64_t * row_idx; // set_rows
    int64_t row_bytes;       // set_rows nb[1]

    int kind;
    int n_dims;
    const int32_t * pos;
    float freq_scale, ext_factor, attn_factor, theta_scale;
    ak_rope_corr_dims corr_dims;
    const float * freq_factors;
    ak_mrope_sections sections;
};

template <int out_kind>
static __global__ void ak_h14_rms_rope_kernel(const ak_h14_params p) {
    extern __shared__ float ak_smem[]; // ne0 floats of rotated output + 32 reduce slots
    float * buf   = ak_smem;
    float * s_red = ak_smem + p.ne0;

    const int row = blockIdx.x;
    const int i1  = row % p.ne1;
    const int i2  = (row / p.ne1) % p.ne2;
    const int i3  = row / (p.ne1 * p.ne2);
    const int tid = threadIdx.x;

    const float * x = p.x + i1*p.s01 + i2*p.s02 + i3*p.s03;

    float sumsq = 0.0f;
    for (int c = tid; c < p.ne0; c += blockDim.x) {
        const float v = x[c];
        sumsq += v*v;
    }
    sumsq = block_reduce<block_reduce_method::SUM>(sumsq, s_red);
    const float scale = rsqrtf(sumsq / p.ne0 + p.eps);

    // normalized + weighted value, same op order as rms_norm (scale*x) then mul (*w)
    auto nv = [&](int e) -> float { return (scale * x[e]) * p.w[e]; };

    const int n_half = p.ne0 / 2;
    for (int q = tid; q < n_half; q += blockDim.x) {
        const int i0 = 2*q;
        int ea, eb;
        if (p.kind == AK_ROPE_NORM) {
            ea = i0; eb = i0 + 1;
        } else {
            ea = q;  eb = q + p.n_dims/2;
        }
        if (i0 >= p.n_dims) {
            buf[i0 + 0] = nv(i0 + 0);
            buf[i0 + 1] = nv(i0 + 1);
            continue;
        }

        float theta_base;
        if (p.kind == AK_ROPE_NORM || p.kind == AK_ROPE_NEOX) {
            theta_base = p.pos[i2]*powf(p.theta_scale, i0/2.0f);
        } else {
            const int sect_dims = p.sections.v[0] + p.sections.v[1] + p.sections.v[2] + p.sections.v[3];
            const int sec_w     = p.sections.v[1] + p.sections.v[0];
            const int sector    = (i0 / 2) % sect_dims;
            const int ne02      = p.ne2;
            theta_base = 0.0f;
            if (p.kind == AK_ROPE_IMROPE) {
                if (sector % 3 == 1 && sector < 3 * p.sections.v[1]) {
                    theta_base = p.pos[i2 + ne02 * 1] * powf(p.theta_scale, i0 / 2.0f);
                } else if (sector % 3 == 2 && sector < 3 * p.sections.v[2]) {
                    theta_base = p.pos[i2 + ne02 * 2] * powf(p.theta_scale, i0 / 2.0f);
                } else if (sector % 3 == 0 && sector < 3 * p.sections.v[0]) {
                    theta_base = p.pos[i2] * powf(p.theta_scale, i0 / 2.0f);
                } else {
                    theta_base = p.pos[i2 + ne02 * 3] * powf(p.theta_scale, i0 / 2.0f);
                }
            } else {
                if (sector < p.sections.v[0]) {
                    theta_base = p.pos[i2] * powf(p.theta_scale, i0 / 2.0f);
                } else if (sector >= p.sections.v[0] && sector < sec_w) {
                    theta_base = p.pos[i2 + ne02 * 1] * powf(p.theta_scale, i0 / 2.0f);
                } else if (sector >= sec_w && sector < sec_w + p.sections.v[2]) {
                    theta_base = p.pos[i2 + ne02 * 2] * powf(p.theta_scale, i0 / 2.0f);
                } else if (sector >= sec_w + p.sections.v[2]) {
                    theta_base = p.pos[i2 + ne02 * 3] * powf(p.theta_scale, i0 / 2.0f);
                }
            }
        }
        const float freq_factor = p.freq_factors ? p.freq_factors[i0/2] : 1.0f;

        float cos_theta, sin_theta;
        ak_rope_yarn(theta_base/freq_factor, p.freq_scale, p.corr_dims, i0, p.ext_factor, p.attn_factor, cos_theta, sin_theta);

        const float x0 = nv(ea);
        const float x1 = nv(eb);
        buf[ea] = x0*cos_theta - x1*sin_theta;
        buf[eb] = x0*sin_theta + x1*cos_theta;
    }
    __syncthreads();

    if constexpr (out_kind == AK_OUT_F32) {
        float * d = (float *) p.dst + i1*p.s1 + i2*p.s2 + i3*p.s3;
        for (int c = tid; c < p.ne0; c += blockDim.x) {
            d[c] = buf[c];
        }
    } else {
        char * rowp = (char *) p.dst + p.row_idx[i2]*p.row_bytes;
        if constexpr (out_kind == AK_OUT_SR_F32) {
            float * d = (float *) rowp + (int64_t) i1*p.ne0;
            for (int c = tid; c < p.ne0; c += blockDim.x) {
                d[c] = buf[c];
            }
        } else if constexpr (out_kind == AK_OUT_SR_F16) {
            half * d = (half *) rowp + (int64_t) i1*p.ne0;
            for (int c = tid; c < p.ne0; c += blockDim.x) {
                d[c] = __float2half(buf[c]);
            }
        } else {
            block_q8_0 * d = (block_q8_0 *) rowp + (int64_t) i1*(p.ne0/QK8_0);
            for (int b = tid; b < p.ne0/QK8_0; b += blockDim.x) {
                quantize_f32_q8_0_block(buf + b*QK8_0, d + b);
            }
        }
    }
}

int ggml_cuda_ak_h14_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = ak_env_opt_in("GGML_CUDA_AK_H14");
    static std::atomic<bool> logged_sr{false}, logged_rope{false};
    if (!enabled || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * rms  = cgraph->nodes[i];
    ggml_tensor * mul  = cgraph->nodes[i + 1];
    ggml_tensor * rope = cgraph->nodes[i + 2];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return 0;
    }

    ggml_tensor * view = nullptr;
    ggml_tensor * sr   = nullptr;
    if (i + 4 < cgraph->n_nodes && cgraph->nodes[i + 3]->op == GGML_OP_VIEW && cgraph->nodes[i + 4]->op == GGML_OP_SET_ROWS) {
        view = cgraph->nodes[i + 3];
        sr   = cgraph->nodes[i + 4];
        if (view->view_src != rope || sr->src[0] != view ||
                !ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, { i + 4 })) {
            view = nullptr;
            sr   = nullptr;
        }
    }
    if (!sr && !ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, { i + 2 })) {
        return 0;
    }

    const ggml_tensor * x = rms->src[0];
    if (mul->src[0] != rms) {
        return 0;
    }
    const ggml_tensor * w = mul->src[1];
    if (x->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32 ||
        w->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || !ggml_is_contiguous(w) || ggml_nelements(w) != x->ne[0] ||
        rope->src[0] != mul || rope->src[1]->type != GGML_TYPE_I32 || !ggml_are_same_shape(x, rope)) {
        return 0;
    }
    if (rope->src[2] && (rope->src[2]->type != GGML_TYPE_F32 || !ggml_is_contiguous(rope->src[2]))) {
        return 0;
    }

    const int ne0 = x->ne[0];
    if (ne0 % 2 != 0 || ne0 > 2048) {
        return 0;
    }

    const int32_t * op = (const int32_t *) rope->op_params;
    const int n_dims     = op[1];
    const int mode       = op[2];
    const int n_ctx_orig = op[4];
    int kind;
    if (mode == GGML_ROPE_TYPE_NORMAL) {
        kind = AK_ROPE_NORM;
    } else if (mode == GGML_ROPE_TYPE_NEOX) {
        kind = AK_ROPE_NEOX;
    } else if (mode == GGML_ROPE_TYPE_MROPE) {
        kind = AK_ROPE_MROPE;
    } else if (mode == GGML_ROPE_TYPE_IMROPE) {
        kind = AK_ROPE_IMROPE;
    } else {
        return 0; // vision / unknown
    }
    if (n_dims > ne0 || n_dims % 2 != 0) {
        return 0;
    }

    int out_kind = AK_OUT_F32;
    if (sr) {
        if (x->ne[3] != 1 || !ggml_is_contiguous(view) || view->ne[0] != (int64_t) ne0 * x->ne[1] ||
            sr->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(sr->src[1]) || sr->src[1]->ne[0] != x->ne[2] ||
            ggml_nelements(sr->src[1]) != x->ne[2] || sr->ne[0] != view->ne[0]) {
            return 0;
        }
        if (sr->type == GGML_TYPE_F32) {
            out_kind = AK_OUT_SR_F32;
        } else if (sr->type == GGML_TYPE_F16) {
            out_kind = AK_OUT_SR_F16;
        } else if (sr->type == GGML_TYPE_Q8_0 && ne0 % QK8_0 == 0) {
            out_kind = AK_OUT_SR_Q8_0;
        } else {
            return 0;
        }
    } else if (rope->nb[0] != sizeof(float)) {
        return 0;
    }

    ak_h14_params p{};
    p.x   = (const float *) x->data;
    p.s01 = x->nb[1] / sizeof(float);
    p.s02 = x->nb[2] / sizeof(float);
    p.s03 = x->nb[3] / sizeof(float);
    p.ne0 = ne0;
    p.ne1 = x->ne[1];
    p.ne2 = x->ne[2];
    p.w   = (const float *) w->data;
    memcpy(&p.eps, rms->op_params, sizeof(float));

    if (sr) {
        p.dst       = sr->data;
        p.row_idx   = (const int64_t *) sr->src[1]->data;
        p.row_bytes = sr->nb[1];
    } else {
        p.dst = rope->data;
        p.s1  = rope->nb[1] / sizeof(float);
        p.s2  = rope->nb[2] / sizeof(float);
        p.s3  = rope->nb[3] / sizeof(float);
    }

    float freq_base, beta_fast, beta_slow;
    memcpy(&freq_base,     op +  5, sizeof(float));
    memcpy(&p.freq_scale,  op +  6, sizeof(float));
    memcpy(&p.ext_factor,  op +  7, sizeof(float));
    memcpy(&p.attn_factor, op +  8, sizeof(float));
    memcpy(&beta_fast,     op +  9, sizeof(float));
    memcpy(&beta_slow,     op + 10, sizeof(float));
    memcpy(&p.sections.v,  op + 11, sizeof(int)*4);
    if ((kind == AK_ROPE_MROPE || kind == AK_ROPE_IMROPE) &&
            p.sections.v[0] + p.sections.v[1] + p.sections.v[2] + p.sections.v[3] <= 0) {
        return 0;
    }
    p.kind         = kind;
    p.n_dims       = n_dims;
    p.pos          = (const int32_t *) rope->src[1]->data;
    p.theta_scale  = powf(freq_base, -2.0f / n_dims);
    p.freq_factors = rope->src[2] ? (const float *) rope->src[2]->data : nullptr;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, p.corr_dims.v);

    const int warp = ggml_cuda_info().devices[ctx.device].warp_size;
    int bs = ((ne0/2 + warp - 1) / warp) * warp;
    bs = std::min(bs, 1024);
    const int64_t nrows = ggml_nrows(x);
    const size_t  smem  = (ne0 + 32) * sizeof(float);
    cudaStream_t  st    = ctx.stream();

    switch (out_kind) {
        case AK_OUT_F32:     ak_h14_rms_rope_kernel<AK_OUT_F32>    <<<nrows, bs, smem, st>>>(p); break;
        case AK_OUT_SR_F32:  ak_h14_rms_rope_kernel<AK_OUT_SR_F32> <<<nrows, bs, smem, st>>>(p); break;
        case AK_OUT_SR_F16:  ak_h14_rms_rope_kernel<AK_OUT_SR_F16> <<<nrows, bs, smem, st>>>(p); break;
        case AK_OUT_SR_Q8_0: ak_h14_rms_rope_kernel<AK_OUT_SR_Q8_0><<<nrows, bs, smem, st>>>(p); break;
    }
    CUDA_CHECK(cudaGetLastError());

    if (sr) {
        ak_log_once(logged_sr, "H14 rms_norm+mul+rope+set_rows", sr);
        return 4;
    }
    ak_log_once(logged_rope, "H14 rms_norm+mul+rope", rope);
    return 2;
}

// ---------------------------------------------------------------------------------------------
// H15: concat + conv-state cpy(s) + ssm_conv + silu + l2_norm(q) + l2_norm(k)
// ---------------------------------------------------------------------------------------------

#define AK_H15_MAX_SLOTS 8
#define AK_H15_MAX_DCONV 8
#define AK_H15_MAX_GAP   48

static thread_local const ggml_tensor * ak_h15_pending_conv   = nullptr;
static thread_local const ggml_cgraph * ak_h15_pending_cgraph = nullptr;

struct ak_h15_params {
    const char * s0; int64_t s0_nb0, s0_nb1, s0_nb2; // conv states  [dconv-1, C, ns]
    const char * s1; int64_t s1_nb0, s1_nb1, s1_nb2; // qkv^T       [nt, C, ns]
    const float * w; int64_t w_s1;                   // weight [dconv, C], row stride (elements)
    float * silu;                                    // [C, nt, ns] contiguous
    float * l2q; int q_off, nq; float eps_q;         // channel offset, heads
    float * l2k; int k_off, nk; float eps_k;
    int C, nt, dconv;
    int n_slots;
    int sidx[AK_H15_MAX_SLOTS];
    float * sdst[AK_H15_MAX_SLOTS];
    int64_t sdst_nb1[AK_H15_MAX_SLOTS];               // bytes
};

static __global__ void ak_h15_conv_fused_kernel(const ak_h15_params p) {
    __shared__ float s_red[32];
    const int hk  = blockDim.x;
    const int tid = threadIdx.x;
    const int c   = blockIdx.x*hk + tid;
    const int t   = blockIdx.y;
    const int s   = blockIdx.z;
    const int nst = p.dconv - 1;

    auto X = [&](int idx) -> float {
        if (idx < nst) {
            return *(const float *) (p.s0 + idx*p.s0_nb0 + c*p.s0_nb1 + s*p.s0_nb2);
        }
        return *(const float *) (p.s1 + (idx - nst)*p.s1_nb0 + c*p.s1_nb1 + s*p.s1_nb2);
    };

    const float * w = p.w + (int64_t) c*p.w_s1;
    float sumf = 0.0f;
    for (int j = 0; j < p.dconv; ++j) {
        sumf += X(t + j) * w[j];
    }
    const float y = ggml_cuda_op_silu_single(sumf);
    p.silu[((int64_t) s*p.nt + t)*p.C + c] = y;

    // conv-state snapshot(s): done once per (channel, seq)
    if (t == 0) {
        for (int k = 0; k < p.n_slots; ++k) {
            float * d = (float *) ((char *) p.sdst[k] + s*p.sdst_nb1[k]) + (int64_t) c*nst;
            for (int j = 0; j < nst; ++j) {
                d[j] = X(p.sidx[k] + j);
            }
        }
    }

    // q / k l2 norm: block is exactly one head (uniform branch per block)
    const int c0 = blockIdx.x*hk;
    float * out = nullptr; float eps = 0.0f; int h = 0; int nh = 0;
    if (c0 >= p.q_off && c0 < p.q_off + p.nq*hk) {
        out = p.l2q; eps = p.eps_q; h = (c0 - p.q_off)/hk; nh = p.nq;
    } else if (c0 >= p.k_off && c0 < p.k_off + p.nk*hk) {
        out = p.l2k; eps = p.eps_k; h = (c0 - p.k_off)/hk; nh = p.nk;
    }
    if (out) {
        const float ss    = block_reduce<block_reduce_method::SUM>(y*y, s_red);
        const float scale = rsqrtf(fmaxf(ss, eps * eps));
        out[(((int64_t) s*p.nt + t)*nh + h)*hk + tid] = scale * y;
    }
}

static bool ak_h15_l2_view_ok(const ggml_tensor * view, const ggml_tensor * l2, const ggml_tensor * silu,
                              int64_t & off_ch, int64_t & hk) {
    if (view->view_src != silu || view->type != GGML_TYPE_F32 || view->nb[0] != sizeof(float) ||
        view->nb[1] != (size_t) view->ne[0]*sizeof(float) || view->nb[2] != silu->nb[1] || view->nb[3] != silu->nb[2] ||
        view->ne[2] != silu->ne[1] || view->ne[3] != silu->ne[2] || view->view_offs % (view->ne[0]*sizeof(float)) != 0) {
        return false;
    }
    if (l2->src[0] != view || l2->type != GGML_TYPE_F32 || !ggml_is_contiguous(l2) || !ggml_are_same_shape(l2, view)) {
        return false;
    }
    hk     = view->ne[0];
    off_ch = view->view_offs / sizeof(float);
    return off_ch + view->ne[1]*hk <= silu->ne[0];
}

// GGML_CUDA_AK_FUSE_DEBUG=1: log why H15 did not take a CONCAT it started matching (first 32 rejects per process).
// Reject ids are numbered in source order inside ggml_cuda_ak_h15_try.
static int ak_h15_reject(int id, const ggml_tensor * concat) {
    static const bool dbg = getenv("GGML_CUDA_AK_FUSE_DEBUG") != nullptr && atoi(getenv("GGML_CUDA_AK_FUSE_DEBUG")) != 0;
    static std::atomic<int> n{0};
    if (dbg && n.fetch_add(1) < 32) {
        GGML_LOG_INFO("ggml_cuda_ak: H15 reject %d at %s\n", id, concat->name);
    }
    return 0;
}

int ggml_cuda_ak_h15_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = ak_env_enabled("GGML_CUDA_AK_H15");
    static std::atomic<bool> logged{false};
    if (!enabled) {
        return 0;
    }
    // the fused kernel already ran at the CONCAT position: skip the tail it computed
    // (SSM_CONV, SILU, VIEW, L2_NORM, VIEW, L2_NORM)
    if (cgraph->nodes[i]->op == GGML_OP_SSM_CONV) {
        if (ak_h15_pending_cgraph == cgraph && ak_h15_pending_conv == cgraph->nodes[i]) {
            ak_h15_pending_conv   = nullptr;
            ak_h15_pending_cgraph = nullptr;
            return 5;
        }
        return 0;
    }

    ggml_tensor * concat = cgraph->nodes[i];
    if (concat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(concat, 0) != 0 || concat->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(concat) || concat->ne[3] != 1) {
        return 0;
    }

    int idxs[32];
    ggml_op ops[32];
    int outs[32];
    int n = 0, n_out = 0;
    idxs[n] = i; ops[n++] = GGML_OP_CONCAT;

    // each conv-state snapshot is VIEW(concat) [-> CONT] -> VIEW(state buffer) -> CPY
    // (qwen35/qwen35moe/qwen3next: no CONT; qwen4exp/Q38FN: ggml_cont(tail))
    ggml_tensor * cpys[AK_H15_MAX_SLOTS];
    ggml_tensor * tails[AK_H15_MAX_SLOTS];
    int K = 0;
    int j = i + 1;
    while (K < AK_H15_MAX_SLOTS && j + 2 < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_VIEW) {
        ggml_tensor * tail = cgraph->nodes[j];
        if (tail->view_src != concat) {
            break;
        }
        const bool has_cont = cgraph->nodes[j + 1]->op == GGML_OP_CONT;
        const int  o        = has_cont ? 1 : 0;
        if (j + 2 + o >= cgraph->n_nodes || cgraph->nodes[j + 1 + o]->op != GGML_OP_VIEW || cgraph->nodes[j + 2 + o]->op != GGML_OP_CPY) {
            break;
        }
        ggml_tensor * cpy = cgraph->nodes[j + 2 + o];
        ggml_tensor * csrc = has_cont ? cgraph->nodes[j + 1] : tail;
        if ((has_cont && csrc->src[0] != tail) || cpy->src[0] != csrc || cpy->src[1] != cgraph->nodes[j + 1 + o] ||
                n + 3 + o + 6 > 32) {
            break;
        }
        cpys[K]  = cpy;
        tails[K] = tail;
        K++;
        idxs[n] = j; ops[n++] = GGML_OP_VIEW;
        if (has_cont) {
            idxs[n] = j + 1; ops[n++] = GGML_OP_CONT;
        }
        idxs[n] = j + 1 + o; ops[n++] = GGML_OP_VIEW; outs[n_out++] = j + 1 + o; // view of the persistent state buffer
        idxs[n] = j + 2 + o; ops[n++] = GGML_OP_CPY;  outs[n_out++] = j + 2 + o;
        j += 3 + o;
    }
    if (K == 0) {
        return ak_h15_reject(1, concat);
    }
    const int j_first_tail_search = j;

    // The conv tail is emitted when the GDN op is expanded, after build_rs() of the recurrent state has
    // expanded its own nodes (state zeroing, get_rows) -> allow a short run of independent nodes between
    // the last snapshot CPY and SSM_CONV. They are checked below and left to run in place.
    int jt = -1;
    for (int q = j; q < cgraph->n_nodes && q < j + AK_H15_MAX_GAP; ++q) {
        if (cgraph->nodes[q]->op == GGML_OP_SSM_CONV && cgraph->nodes[q]->src[0] == concat) {
            jt = q;
            break;
        }
    }
    if (jt < 0 || jt + 5 >= cgraph->n_nodes) {
        return ak_h15_reject(2, concat);
    }
    j = jt;

    ggml_tensor * conv  = cgraph->nodes[j];
    ggml_tensor * silu  = cgraph->nodes[j + 1];
    ggml_tensor * vq    = cgraph->nodes[j + 2];
    ggml_tensor * l2q   = cgraph->nodes[j + 3];
    ggml_tensor * vk    = cgraph->nodes[j + 4];
    ggml_tensor * l2k   = cgraph->nodes[j + 5];
    if (conv->op != GGML_OP_SSM_CONV || silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
        vq->op != GGML_OP_VIEW || l2q->op != GGML_OP_L2_NORM || vk->op != GGML_OP_VIEW || l2k->op != GGML_OP_L2_NORM ||
        conv->src[0] != concat || silu->src[0] != conv) {
        return ak_h15_reject(3, concat);
    }
    const ggml_op tail_ops[6] = { GGML_OP_SSM_CONV, GGML_OP_UNARY, GGML_OP_VIEW, GGML_OP_L2_NORM, GGML_OP_VIEW, GGML_OP_L2_NORM };
    for (int k = 0; k < 6; ++k) {
        idxs[n] = j + k; ops[n++] = tail_ops[k];
    }
    outs[n_out++] = j + 1;
    outs[n_out++] = j + 3;
    outs[n_out++] = j + 5;

    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, n, ops, outs, n_out)) {
        return ak_h15_reject(4, concat);
    }

    // geometry
    const ggml_tensor * st  = concat->src[0];
    const ggml_tensor * qkv = concat->src[1];
    const ggml_tensor * wt  = conv->src[1];
    const int64_t dconv = wt->ne[0];
    const int64_t C     = concat->ne[1];
    const int64_t nt    = conv->ne[1];
    const int64_t ns    = conv->ne[2];
    if (st->type != GGML_TYPE_F32 || qkv->type != GGML_TYPE_F32 || wt->type != GGML_TYPE_F32 || wt->nb[0] != sizeof(float) ||
        dconv < 2 || dconv > AK_H15_MAX_DCONV || nt < 1 || nt > 32 ||
        st->ne[0] != dconv - 1 || qkv->ne[0] != nt || st->ne[1] != C || qkv->ne[1] != C || st->ne[2] != ns || qkv->ne[2] != ns ||
        concat->ne[0] != dconv - 1 + nt || concat->ne[2] != ns || wt->ne[1] != C ||
        conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32 || conv->ne[0] != C || !ggml_is_contiguous(silu) ||
        !ggml_are_same_shape(conv, silu)) {
        return ak_h15_reject(5, concat);
    }

    int64_t q_off, hkq, k_off, hkk;
    if (!ak_h15_l2_view_ok(vq, l2q, silu, q_off, hkq) || !ak_h15_l2_view_ok(vk, l2k, silu, k_off, hkk) || hkq != hkk) {
        return ak_h15_reject(6, concat);
    }
    const int64_t hk   = hkq;
    const int     warp = ggml_cuda_info().devices[ctx.device].warp_size;
    if (hk % warp != 0 || hk > 1024 || C % hk != 0) {
        return ak_h15_reject(7, concat);
    }

    ak_h15_params p{};
    p.s0 = (const char *) st->data;  p.s0_nb0 = st->nb[0];  p.s0_nb1 = st->nb[1];  p.s0_nb2 = st->nb[2];
    p.s1 = (const char *) qkv->data; p.s1_nb0 = qkv->nb[0]; p.s1_nb1 = qkv->nb[1]; p.s1_nb2 = qkv->nb[2];
    p.w = (const float *) wt->data;  p.w_s1 = wt->nb[1] / sizeof(float);
    p.silu = (float *) silu->data;
    p.l2q = (float *) l2q->data; p.q_off = q_off; p.nq = vq->ne[1]; memcpy(&p.eps_q, l2q->op_params, sizeof(float));
    p.l2k = (float *) l2k->data; p.k_off = k_off; p.nk = vk->ne[1]; memcpy(&p.eps_k, l2k->op_params, sizeof(float));
    p.C = C; p.nt = nt; p.dconv = dconv;
    p.n_slots = K;
    for (int k = 0; k < K; ++k) {
        const ggml_tensor * src = tails[k];
        const ggml_tensor * dst = cpys[k]->src[1];
        if (src->ne[0] != dconv - 1 || src->ne[1] != C || src->ne[2] != ns || src->nb[0] != sizeof(float) ||
            src->nb[1] != concat->nb[1] || src->nb[2] != concat->nb[2] || src->view_offs % sizeof(float) != 0 ||
            (int64_t) (src->view_offs / sizeof(float)) + dconv - 1 > concat->ne[0] ||
            dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float) || dst->ne[0] != (dconv - 1)*C || dst->ne[1] != ns ||
            ggml_nelements(dst) != ggml_nelements(src)) {
            return ak_h15_reject(8, concat);
        }
        // the snapshot write must not alias anything the other blocks still read
        const char * d0 = (const char *) dst->data;
        const char * d1 = d0 + ggml_nbytes(dst);
        for (const ggml_tensor * r : { st, qkv }) {
            const char * r0 = (const char *) r->data;
            const char * r1 = r0 + ggml_nbytes(r);
            if (d0 < r1 && r0 < d1) {
                return ak_h15_reject(9, concat);
            }
        }
        p.sidx[k]     = src->view_offs / sizeof(float);
        p.sdst[k]     = (float *) dst->data;
        p.sdst_nb1[k] = dst->nb[1];
    }

    // Hazards of running the tail early, before the gap nodes [j_first_tail_search, j):
    //  - a gap node must not consume any fused node;
    //  - a gap node's output must not alias what the fused kernel writes (silu, l2q, l2k, snapshot slots) -- the
    //    allocator may reuse memory across dead tensors (aliasing what it READ is fine: those reads already happened);
    //  - a gap node's inputs must not alias what the fused kernel writes.
    if (j > j_first_tail_search) {
        auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (!a->data || !b->data || ggml_nbytes(a) == 0 || ggml_nbytes(b) == 0) {
                return false;
            }
            const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
            const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
            return a0 < b1 && b0 < a1;
        };
        std::vector<const ggml_tensor *> wr = { silu, l2q, l2k };
        for (int k = 0; k < K; ++k) {
            wr.push_back(cpys[k]->src[1]);
        }
        for (int q = j_first_tail_search; q < j; ++q) {
            const ggml_tensor * g = cgraph->nodes[q];
            for (int si = 0; si < GGML_MAX_SRC; ++si) {
                const ggml_tensor * src = g->src[si];
                if (!src) {
                    continue;
                }
                for (int f = 0; f < n; ++f) {
                    if (src == cgraph->nodes[idxs[f]]) {
                        return ak_h15_reject(10, concat);
                    }
                }
                for (const ggml_tensor * w : wr) {
                    if (overlaps(src, w)) {
                        return ak_h15_reject(11, concat);
                    }
                }
            }
            if (!(g->op == GGML_OP_VIEW || g->op == GGML_OP_RESHAPE || g->op == GGML_OP_PERMUTE || g->op == GGML_OP_TRANSPOSE || g->op == GGML_OP_NONE)) {
                // (former reject 12, removed: a gap node writing over st/qkv/weight is harmless -- the fused kernel has already
                // read them at the CONCAT position, before any gap node runs. The allocator does exactly this in the model:
                // st and qkv die at CONCAT and the recurrent-state get_rows reuses their memory.)
                for (const ggml_tensor * w : wr) {
                    if (overlaps(g, w)) {
                        return ak_h15_reject(13, concat);
                    }
                }
            }
        }
    }

    const dim3 grid(C / hk, nt, ns);
    ak_h15_conv_fused_kernel<<<grid, hk, 0, ctx.stream()>>>(p);
    CUDA_CHECK(cudaGetLastError());

    ak_log_once(logged, "H15 concat+conv_state_cpy+ssm_conv+silu+l2norm(q,k)", conv);
    if (j == j_first_tail_search) {
        return (j + 5) - i;
    }
    // skip CONCAT + snapshot groups now; the gap runs normally; SSM_CONV..L2_NORM are skipped when reached
    ak_h15_pending_conv   = conv;
    ak_h15_pending_cgraph = cgraph;
    return j_first_tail_search - 1 - i;
}
