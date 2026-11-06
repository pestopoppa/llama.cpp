#pragma once

// AK-ASM-H13b planner for the qwen4exp hyper-connection seam fusions (see hc-fuse.cuh). Host-only C++ so the
// planning rules can be unit-tested on the CPU (tests/test-hc-fuse-plan.cpp) without a GPU.

#include "ggml.h"
#include "../ggml-impl.h"

#include <unordered_map>
#include <unordered_set>

enum ggml_cuda_hc_kind { HC_FUSE_COMBINE = 1, HC_FUSE_MIX_TAIL = 2, HC_FUSE_SCALE_SILU = 3, HC_FUSE_RMS_GAMMA = 4 };

struct ggml_cuda_hc_anchor {
    ggml_cuda_hc_kind   kind;
    const ggml_tensor * a = nullptr;  // combine: residual | mix: xn          | silu: x
    const ggml_tensor * b = nullptr;  // combine: block_out [E,1,T] | mix: u (pre-sigmoid) | norm: gamma
    const ggml_tensor * c = nullptr;  // combine: inject [hc,T] | norm: the RMS_NORM node
    float s1 = 1.0f, b1 = 0.0f, s2 = 1.0f, b2 = 0.0f;  // the SCALE op params (scale, bias) of the chain
};

struct ggml_cuda_hc_plan {
    std::unordered_set<const ggml_tensor *>                     skip;
    std::unordered_map<const ggml_tensor *, ggml_cuda_hc_anchor> anchors;
};

namespace ggml_hc_plan_detail {

struct hc_ctx {
    std::unordered_map<const ggml_tensor *, int> uses;
    std::unordered_set<const ggml_tensor *>      in_graph;
    bool rms_enabled = true;
};

inline bool f32c(const ggml_tensor * t) { return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); }

// an elidable intermediate: a node of this graph, single consumer, not an output
inline bool inner(const hc_ctx & c, const ggml_tensor * t) {
    if (!t || !c.in_graph.count(t) || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    auto it = c.uses.find(t);
    return it != c.uses.end() && it->second == 1;
}

inline bool is_unary(const ggml_tensor * t, ggml_unary_op op) { return t && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op; }
inline bool is_reshape(const ggml_tensor * t) { return t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW); }
inline void scale_params(const ggml_tensor * t, float & s, float & b) {
    s = ggml_get_op_params_f32(t, 0);
    b = ggml_get_op_params_f32(t, 1);
}

inline bool try_combine(const hc_ctx & c, const ggml_tensor * add, ggml_cuda_hc_plan & plan) {
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

inline bool try_mix_tail(const hc_ctx & c, const ggml_tensor * mean, ggml_cuda_hc_plan & plan) {
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

inline bool try_rms_gamma(const hc_ctx & c, const ggml_tensor * mul, ggml_cuda_hc_plan & plan) {
    if (!c.rms_enabled) return false;
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

inline bool try_scale_silu(const hc_ctx & c, const ggml_tensor * silu, ggml_cuda_hc_plan & plan) {
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

} // namespace ggml_hc_plan_detail

// Build the hc fusion plan for one graph evaluation. `cgraph` may be a view of a larger graph (a scheduler split).
inline void ggml_hc_fuse_plan_build(const ggml_cgraph * cgraph, bool rms_enabled, ggml_cuda_hc_plan & plan) {
    using namespace ggml_hc_plan_detail;
    plan.skip.clear();
    plan.anchors.clear();
    // cheap pre-check: only graphs that contain a MEAN_D1 (the qwen4exp hc mix) are planned
    bool any = false;
    for (int i = 0; i < cgraph->n_nodes && !any; ++i) {
        any = cgraph->nodes[i]->op == GGML_OP_MEAN_D1;
    }
    if (!any) {
        return;
    }
    hc_ctx c;
    c.rms_enabled = rms_enabled;
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
