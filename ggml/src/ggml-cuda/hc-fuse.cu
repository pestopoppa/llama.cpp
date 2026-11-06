// AK-ASM-H13b: hyper-connection seam (qwen4exp build_hc_mix / build_hc_combine) on the device. See hc-fuse.cuh.
// Every fused kernel evaluates the same scalar expressions, in the same order, as the unfused kernels it replaces
// (scale.cu: s*x+b, unary.cu: sigmoid 1/(1+expf(-x)), silu x/(1+expf(-x)), binbcast mul/add, CPU mean_d1: sequential
// sum over ne1 then / ne1).

#include "hc-fuse.cuh"
#include "norm.cuh"

#include <atomic>
#include <cstdlib>

static bool hc_env_flag(const char * name, bool def) {
    const char * e = getenv(name);
    return e ? atoi(e) != 0 : def;
}

bool ggml_cuda_hc_fuse_enabled() {
    static const bool v = hc_env_flag("GGML_CUDA_HC_FUSE", true);
    return v;
}

bool ggml_cuda_hc_rms_fuse_enabled() {
    static const bool v = hc_env_flag("GGML_CUDA_HC_RMS_FUSE", true);  // H13c rms_norm*gamma fusion
    return v;
}

bool ggml_cuda_topk_large_enabled() {
    static const bool v = hc_env_flag("GGML_CUDA_TOPK_LARGE", true);   // H13c TOP_K rows > 1024 on HIP
    return v;
}

bool ggml_cuda_mean_d1_enabled() {
    static const bool v = hc_env_flag("GGML_CUDA_MEAN_D1", true);
    return v;
}

static std::atomic<uint64_t> g_hc_n_calls{0};

uint64_t ggml_cuda_hc_fuse_n_calls() {
    return g_hc_n_calls.load(std::memory_order_relaxed);
}

static __device__ __forceinline__ float hc_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- MEAN_D1: dst[i0, 0, i2, i3] = (sum_{i1} src[i0, i1, i2, i3]) / ne01, arbitrary f32 strides
static __global__ void k_mean_d1(const char * __restrict__ src, float * __restrict__ dst, const int64_t ne00,
                                 const int64_t ne01, const int64_t ne02, const int64_t ne03, const size_t nb00,
                                 const size_t nb01, const size_t nb02, const size_t nb03, const size_t nb0,
                                 const size_t nb2, const size_t nb3) {
    const int64_t n = ne00 * ne02 * ne03;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t i00 = i % ne00;
        const int64_t i02 = (i / ne00) % ne02;
        const int64_t i03 = i / (ne00 * ne02);
        const char * s = src + i00 * nb00 + i02 * nb02 + i03 * nb03;
        float sum = 0.0f;
        for (int64_t i01 = 0; i01 < ne01; ++i01) {
            sum += *(const float *) (s + i01 * nb01);
        }
        *(float *) ((char *) dst + i00 * nb0 + i02 * nb2 + i03 * nb3) = sum / (float) ne01;
    }
}

static unsigned hc_blocks(int64_t n) {
    const int64_t b = (n + 255) / 256;
    return (unsigned) (b < 65535 ? (b > 0 ? b : 1) : 65535);
}

void ggml_cuda_op_mean_d1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    const int64_t n = src->ne[0] * src->ne[2] * src->ne[3];
    k_mean_d1<<<hc_blocks(n), 256, 0, ctx.stream()>>>((const char *) src->data, (float *) dst->data, src->ne[0], src->ne[1],
                                                      src->ne[2], src->ne[3], src->nb[0], src->nb[1], src->nb[2], src->nb[3],
                                                      dst->nb[0], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

// ---- combine: out[e,h,t] = res[e,h,t] + b[e,t] * (s2 * sigmoid(s1 * inj[h,t] + b1) + b2)   (all contiguous)
static __global__ void k_hc_combine(const float * __restrict__ res, const float * __restrict__ bo, const float * __restrict__ inj,
                                    float * __restrict__ out, const int64_t E, const int64_t H, const int64_t T,
                                    const float s1, const float b1, const float s2, const float b2) {
    const int64_t n = E * H * T;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t e = i % E;
        const int64_t h = (i / E) % H;
        const int64_t t = i / (E * H);
        const float   w = s2 * hc_sigmoid(s1 * inj[t * H + h] + b1) + b2;
        out[i] = res[i] + bo[t * E + e] * w;
    }
}

// ---- mix tail: out[e,t] = (sum_h xn[e + h*E, t] * sigmoid(u[e + h*E, t])) / H
static __global__ void k_hc_mix_tail(const float * __restrict__ xn, const float * __restrict__ u, float * __restrict__ out,
                                     const int64_t E, const int64_t H, const int64_t T) {
    const int64_t n = E * T;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t e = i % E;
        const int64_t t = i / E;
        float sum = 0.0f;
        for (int64_t h = 0; h < H; ++h) {
            const int64_t j = t * E * H + h * E + e;
            sum += xn[j] * hc_sigmoid(u[j]);
        }
        out[i] = sum / (float) H;
    }
}

// ---- gate: out = silu(s * x + b)
static __global__ void k_hc_scale_silu(const float * __restrict__ x, float * __restrict__ out, const int64_t n, const float s,
                                       const float b) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const float y = s * x[i] + b;
        out[i] = y / (1.0f + expf(-y));
    }
}

void ggml_cuda_hc_fuse_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_hc_anchor & an) {
    cudaStream_t stream = ctx.stream();
    switch (an.kind) {
        case HC_FUSE_COMBINE: {
            const int64_t E = dst->ne[0], H = dst->ne[1], T = dst->ne[2];
            k_hc_combine<<<hc_blocks(E * H * T), 256, 0, stream>>>((const float *) an.a->data, (const float *) an.b->data,
                (const float *) an.c->data, (float *) dst->data, E, H, T, an.s1, an.b1, an.s2, an.b2);
        } break;
        case HC_FUSE_MIX_TAIL: {
            const int64_t E = dst->ne[0], T = dst->ne[2];
            const int64_t H = ggml_nelements(an.a) / (E * T);
            k_hc_mix_tail<<<hc_blocks(E * T), 256, 0, stream>>>((const float *) an.a->data, (const float *) an.b->data,
                (float *) dst->data, E, H, T);
        } break;
        case HC_FUSE_RMS_GAMMA: {
            // the existing fused rms_norm*mul kernel, given views in the RMS_NORM layout: output [E,hc,T] contiguous is
            // the MUL's [hc*E, T]; gamma [hc*E] is viewed [E,hc,1,1] so it broadcasts over T
            ggml_tensor rms = *an.c;
            ggml_tensor gam = *an.b;
            gam.ne[0] = rms.ne[0]; gam.ne[1] = rms.ne[1]; gam.ne[2] = 1; gam.ne[3] = 1;
            gam.nb[1] = gam.nb[0] * gam.ne[0]; gam.nb[2] = gam.nb[1] * gam.ne[1]; gam.nb[3] = gam.nb[2];
            ggml_tensor mul = *dst;
            for (int d = 0; d < 4; ++d) { mul.ne[d] = rms.ne[d]; }
            mul.nb[0] = sizeof(float);
            for (int d = 1; d < 4; ++d) { mul.nb[d] = mul.nb[d - 1] * mul.ne[d - 1]; }
            mul.src[0] = &rms;
            mul.src[1] = &gam;
            ggml_cuda_op_rms_norm_fused(ctx, &rms, &mul);
        } break;
        case HC_FUSE_SCALE_SILU: {
            const int64_t n = ggml_nelements(dst);
            k_hc_scale_silu<<<hc_blocks(n), 256, 0, stream>>>((const float *) an.a->data, (float *) dst->data, n, an.s1, an.b1);
        } break;
    }
    CUDA_CHECK(cudaGetLastError());
    const uint64_t c = g_hc_n_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (c == 1) {
        GGML_LOG_INFO("ggml_cuda_hc_fuse: hyper-connection fused path ran (call 1) kind=%d node=%s\n", (int) an.kind, dst->name);
    }
}

// ---------------------------------------------------------------------------------------------------- planning
namespace {

struct hc_ctx {
    std::unordered_map<const ggml_tensor *, int> uses;
    std::unordered_set<const ggml_tensor *>      in_graph;
};

bool f32c(const ggml_tensor * t) { return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); }

// an elidable intermediate: a node of this graph, single consumer, not an output
bool inner(const hc_ctx & c, const ggml_tensor * t) {
    if (!t || !c.in_graph.count(t) || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    auto it = c.uses.find(t);
    return it != c.uses.end() && it->second == 1;
}

bool is_unary(const ggml_tensor * t, ggml_unary_op op) { return t && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op; }
bool is_reshape(const ggml_tensor * t) { return t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW); }
void scale_params(const ggml_tensor * t, float & s, float & b) {
    s = ggml_get_op_params_f32(t, 0);
    b = ggml_get_op_params_f32(t, 1);
}

bool try_combine(const hc_ctx & c, const ggml_tensor * add, ggml_cuda_hc_plan & plan) {
    if (add->op != GGML_OP_ADD || !f32c(add)) return false;
    for (int ri = 0; ri < 2; ++ri) {
        const ggml_tensor * res = add->src[ri];
        const ggml_tensor * m   = add->src[1 - ri];
        if (!m || m->op != GGML_OP_MUL || !inner(c, m) || !f32c(res) || !ggml_are_same_shape(res, add)) continue;
        for (int mi = 0; mi < 2; ++mi) {
            const ggml_tensor * rep = m->src[mi];
            const ggml_tensor * wr  = m->src[1 - mi];
            if (mi != 0) continue;  // unfused order is mul(b_rep, w): keep exactly that operand order
            if (!rep || rep->op != GGML_OP_REPEAT || !inner(c, rep) || !is_reshape(wr) || !inner(c, wr)) continue;
            const ggml_tensor * s2 = wr->src[0];
            if (!s2 || s2->op != GGML_OP_SCALE || !inner(c, s2)) continue;
            const ggml_tensor * sg = s2->src[0];
            if (!is_unary(sg, GGML_UNARY_OP_SIGMOID) || !inner(c, sg)) continue;
            const ggml_tensor * s1 = sg->src[0];
            if (!s1 || s1->op != GGML_OP_SCALE || !inner(c, s1)) continue;
            const ggml_tensor * inj = s1->src[0];
            const ggml_tensor * bo  = rep->src[0];
            const int64_t E = add->ne[0], H = add->ne[1], T = add->ne[2];
            if (add->ne[3] != 1 || !f32c(inj) || !f32c(bo) || !f32c(s1) || !f32c(sg) || !f32c(s2)) continue;
            if (ggml_nelements(inj) != H * T || ggml_nelements(bo) != E * T || bo->ne[0] != E || inj->ne[0] != H) continue;
            if (wr->ne[0] != 1 || wr->ne[1] != H || wr->ne[2] != T || !ggml_are_same_shape(rep, add)) continue;
            ggml_cuda_hc_anchor an{HC_FUSE_COMBINE};
            an.a = res; an.b = bo; an.c = inj;
            scale_params(s1, an.s1, an.b1);
            scale_params(s2, an.s2, an.b2);
            plan.anchors[add] = an;
            for (const ggml_tensor * t : { m, rep, wr, s2, sg, s1 }) plan.skip.insert(t);
            return true;
        }
    }
    return false;
}

bool try_mix_tail(const hc_ctx & c, const ggml_tensor * mean, ggml_cuda_hc_plan & plan) {
    if (mean->op != GGML_OP_MEAN_D1 || !f32c(mean) || mean->ne[3] != 1) return false;
    const ggml_tensor * rg = mean->src[0];
    if (!is_reshape(rg) || !inner(c, rg) || !ggml_is_contiguous(rg)) return false;
    const ggml_tensor * g = rg->src[0];
    if (!g || g->op != GGML_OP_MUL || !inner(c, g) || !f32c(g)) return false;
    const ggml_tensor * xn  = g->src[0];
    const ggml_tensor * sig = g->src[1];   // unfused order: mul(xn, gate)
    if (!is_unary(sig, GGML_UNARY_OP_SIGMOID) || !inner(c, sig) || !f32c(sig) || !f32c(xn)) return false;
    const ggml_tensor * u = sig->src[0];
    if (!f32c(u) || !ggml_are_same_shape(xn, g) || !ggml_are_same_shape(u, g)) return false;
    if (rg->ne[0] != mean->ne[0] || rg->ne[2] != mean->ne[2] || ggml_nelements(g) != rg->ne[0] * rg->ne[1] * rg->ne[2]) return false;
    ggml_cuda_hc_anchor an{HC_FUSE_MIX_TAIL};
    an.a = xn; an.b = u;
    plan.anchors[mean] = an;
    for (const ggml_tensor * t : { rg, g, sig }) plan.skip.insert(t);
    return true;
}

bool try_rms_gamma(const hc_ctx & c, const ggml_tensor * mul, ggml_cuda_hc_plan & plan) {
    if (!ggml_cuda_hc_rms_fuse_enabled()) return false;
    if (!mul || mul->op != GGML_OP_MUL || !f32c(mul)) return false;
    for (int ri = 0; ri < 2; ++ri) {
        const ggml_tensor * rs  = mul->src[ri];
        const ggml_tensor * gam = mul->src[1 - ri];
        if (!is_reshape(rs) || !inner(c, rs) || !ggml_is_contiguous(rs)) continue;
        const ggml_tensor * rms = rs->src[0];
        if (!rms || rms->op != GGML_OP_RMS_NORM || !inner(c, rms) || !f32c(rms) || !f32c(rms->src[0])) continue;
        if (rms->ne[3] != 1 || !ggml_are_same_shape(rs, mul)) continue;
        // gamma covers exactly the hc*E columns of one token and broadcasts over tokens
        if (!f32c(gam) || gam->ne[0] != mul->ne[0] || ggml_nelements(gam) != mul->ne[0]) continue;
        if (mul->ne[0] != rms->ne[0] * rms->ne[1] || ggml_nelements(rms) != ggml_nelements(mul)) continue;
        ggml_cuda_hc_anchor an{HC_FUSE_RMS_GAMMA};
        an.b = gam; an.c = rms;
        plan.anchors[mul] = an;
        plan.skip.insert(rs);
        plan.skip.insert(rms);
        return true;
    }
    return false;
}

bool try_scale_silu(const hc_ctx & c, const ggml_tensor * silu, ggml_cuda_hc_plan & plan) {
    if (!is_unary(silu, GGML_UNARY_OP_SILU) || !f32c(silu)) return false;
    const ggml_tensor * s = silu->src[0];
    if (!s || s->op != GGML_OP_SCALE || !inner(c, s) || !f32c(s) || !f32c(s->src[0])) return false;
    if (ggml_nelements(s->src[0]) != ggml_nelements(silu)) return false;
    ggml_cuda_hc_anchor an{HC_FUSE_SCALE_SILU};
    an.a = s->src[0];
    scale_params(s, an.s1, an.b1);
    plan.anchors[silu] = an;
    plan.skip.insert(s);
    return true;
}

} // namespace

void ggml_cuda_hc_fuse_plan(const ggml_cgraph * cgraph, ggml_cuda_hc_plan & plan) {
    plan.skip.clear();
    plan.anchors.clear();
    if (!ggml_cuda_hc_fuse_enabled()) {
        return;
    }
    // cheap pre-check: only graphs that contain a MEAN_D1 (the qwen4exp hc mix) are planned
    bool any = false;
    for (int i = 0; i < cgraph->n_nodes && !any; ++i) {
        any = cgraph->nodes[i]->op == GGML_OP_MEAN_D1;
    }
    if (!any) {
        return;
    }
    hc_ctx c;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        c.in_graph.insert(n);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (n->src[j]) c.uses[n->src[j]]++;
        }
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (plan.skip.count(n)) continue;
        if (!try_combine(c, n, plan) && !try_mix_tail(c, n, plan) && !try_rms_gamma(c, n, plan)) {
            try_scale_silu(c, n, plan);
        }
    }
    // a node must never be both an anchor and elided (the patterns do not overlap); if it ever happens, fuse nothing
    for (const auto & kv : plan.anchors) {
        if (plan.skip.count(kv.first)) {
            plan.skip.clear();
            plan.anchors.clear();
            return;
        }
    }
}
