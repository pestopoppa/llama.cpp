#pragma once

// AK-BW-GDN-GATE-FOLD planner (GGML_HIP_GDN_GATE_FOLD=1, default off). Host-only C++.
//
// The qwen35 (27B) / qwen35moe linear-attention layer builds the GATED_DELTA_NET gate inputs as
//     beta = SIGMOID(RESHAPE(MUL_MAT(ssm_beta,  x)))                              -> GDN src[4]
//     g    = RESHAPE(MUL(SOFTPLUS(ADD(RESHAPE(MUL_MAT(ssm_alpha, x)), ssm_dt)), ssm_a))   -> GDN src[3]
// which on the MI210 verify path is 2 x (q8sk_quantize + q8sk_zero + q8sk_mul_mat) + sigmoid + add + softplus + mul
// = 10 launches per layer (C18 27B trace: 0.63 ms mm + ~0.87 ms quantize/zero + 0.88 ms gate ops per 40.07 ms verify).
// The fold replaces them by ONE launch at the chain's last sink (beta or g, whichever the graph computes last): both
// projections read the f32 activations directly (no int8 activation quantization, no split-K atomics, no zero-init),
// and the epilogue applies sigmoid / softplus(+dt)*ssm_a and writes the exact tensors GDN reads. The GDN op itself is
// untouched, so every GDN route (fused K=T, rows, recurrent, chunked) consumes the folded inputs unchanged.
//
// Safety (same C7R rules as hc-fuse-plan.h): only the intermediates (both MUL_MATs, the ADD, the SOFTPLUS and the
// reshapes between them) are elided, each must have exactly one use in the whole graph and not be an output; the sinks
// beta and g are WRITTEN by the fold with their exact unfused meaning, so they may have any number of readers. The
// anchor is kept only if, from the chain start to the anchor, no computed node overwrote the activation x (or any other
// input), and neither sink overlaps an input or the other sink; otherwise the chain runs unfused (fixed point).

#include "hc-fuse-plan.h"

#include <cstdio>
#include <string>

struct ggml_cuda_gdn_gate_anchor {
    const ggml_tensor * x     = nullptr;  // shared activation [K, ntok] f32
    const ggml_tensor * wb    = nullptr;  // ssm_beta  Q8_0 [K, H]
    const ggml_tensor * wa    = nullptr;  // ssm_alpha Q8_0 [K, H]
    const ggml_tensor * dt    = nullptr;  // ssm_dt  f32 [H]
    const ggml_tensor * a     = nullptr;  // ssm_a   f32 [H]  (= -exp(A_log))
    const ggml_tensor * beta  = nullptr;  // SIGMOID node (written)
    const ggml_tensor * g     = nullptr;  // MUL node (written)
    int64_t             H     = 0;
    int64_t             ntok  = 0;
};

struct ggml_cuda_gdn_gate_plan {
    std::unordered_set<const ggml_tensor *>                                  skip;
    std::unordered_map<const ggml_tensor *, ggml_cuda_gdn_gate_anchor>        anchors;
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> chains;
    int n_matched = 0;
    int n_unsafe  = 0;
    std::vector<std::string> unsafe_why;   // one line per refused anchor: the failing check + the tensors involved
};

#define GDN_GATE_FOLD_MAXT 16

namespace ggml_gdn_gate_plan_detail {

using namespace ggml_hc_plan_detail;

// follow RESHAPE/VIEW nodes of contiguous data at offset 0; the reshapes passed are recorded
inline const ggml_tensor * strip(const ggml_tensor * t, std::vector<const ggml_tensor *> & rs) {
    while (t && is_reshape(t) && ggml_is_contiguous(t) && t->src[0] && t->view_offs == 0) {
        rs.push_back(t);
        t = t->src[0];
    }
    return t;
}

inline bool q8w(const ggml_tensor * w, int64_t K) {
    return w && w->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(w) && w->ne[0] == K && w->ne[2] == 1 && w->ne[3] == 1;
}

inline bool try_gdn(const hc_ctx & c, const ggml_tensor * gdn, ggml_cuda_gdn_gate_plan & plan) {
    if (gdn->op != GGML_OP_GATED_DELTA_NET || !gdn->src[2] || !gdn->src[3] || !gdn->src[4]) return false;
    if (gdn->src[3]->ne[0] != 1) return false;   // KDA (per-channel gate) is not this pattern
    const int64_t H = gdn->src[2]->ne[1];

    std::vector<const ggml_tensor *> rs;
    const ggml_tensor * g  = strip(gdn->src[3], rs);
    const ggml_tensor * bs = strip(gdn->src[4], rs);
    if (!g || g->op != GGML_OP_MUL || !f32c(g) || !is_unary(bs, GGML_UNARY_OP_SIGMOID) || !f32c(bs)) return false;
    if (!c.pos.count(g) || !c.pos.count(bs)) return false;

    const ggml_tensor * sp = g->src[0];
    const ggml_tensor * aw = g->src[1];   // unfused order: mul(softplus, ssm_a)
    if (!is_unary(sp, GGML_UNARY_OP_SOFTPLUS) || !inner(c, sp) || !f32c(sp) || !f32c(aw) || ggml_nelements(aw) != H) return false;
    const ggml_tensor * add = sp->src[0];
    if (!add || add->op != GGML_OP_ADD || !inner(c, add) || !f32c(add)) return false;
    const ggml_tensor * dt = add->src[1];
    if (!f32c(dt) || ggml_nelements(dt) != H || add->ne[0] != H) return false;
    if (aw->ne[0] != H) return false;

    std::vector<const ggml_tensor *> chain_rs;
    const ggml_tensor * mma = strip(add->src[0], chain_rs);
    const ggml_tensor * mmb = strip(bs->src[0], chain_rs);
    if (!mma || !mmb || mma->op != GGML_OP_MUL_MAT || mmb->op != GGML_OP_MUL_MAT) return false;
    if (!inner(c, mma) || !inner(c, mmb)) return false;
    for (const ggml_tensor * r : chain_rs) {
        if (!inner(c, r)) return false;
    }
    const ggml_tensor * x = mma->src[1];
    if (x != mmb->src[1] || !x || x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    const int64_t K = x->ne[0], ntok = x->ne[1];
    if (K % 32 != 0 || ntok < 1 || ntok > GDN_GATE_FOLD_MAXT || x->nb[1] % sizeof(float) != 0) return false;
    if (!q8w(mma->src[0], K) || !q8w(mmb->src[0], K) || mma->src[0]->ne[1] != H || mmb->src[0]->ne[1] != H) return false;
    if (!f32c(mma) || !f32c(mmb) || mma->ne[0] != H || mmb->ne[0] != H || mma->ne[1] != ntok || mmb->ne[1] != ntok) return false;
    if (ggml_nelements(g) != H * ntok || ggml_nelements(bs) != H * ntok || ggml_nelements(add) != H * ntok) return false;
    // the sinks are read by GDN with the [1, H, T, S] contiguous layout = mm output order (h fastest, then token)
    if (gdn->src[3]->data != g->data || gdn->src[4]->data != bs->data) return false;

    const int pg = c.pos.at(g), pb = c.pos.at(bs);
    const ggml_tensor * anchor = pg > pb ? g : bs;
    const ggml_tensor * other  = pg > pb ? bs : g;
    if (plan.anchors.count(anchor) || plan.skip.count(anchor)) return false;

    ggml_cuda_gdn_gate_anchor an;
    an.x = x; an.wb = mmb->src[0]; an.wa = mma->src[0]; an.dt = dt; an.a = aw; an.beta = bs; an.g = g;
    an.H = H; an.ntok = ntok;
    std::vector<const ggml_tensor *> chain = { mma, add, sp, mmb, other };
    for (const ggml_tensor * r : chain_rs) chain.push_back(r);
    plan.anchors[anchor] = an;
    plan.chains[anchor]  = chain;
    for (const ggml_tensor * t : chain) plan.skip.insert(t);
    plan.n_matched++;
    return true;
}

inline std::string tdesc(const ggml_tensor * t, const hc_ctx & c) {
    char b[256];
    const auto ip = c.pos.find(t);
    snprintf(b, sizeof(b), "'%s' %s pos %d [%p, +%zu)", t->name, ggml_op_desc(t), ip == c.pos.end() ? -1 : ip->second,
             t->data, ggml_nbytes(t));
    return b;
}

// why (optional): the failing check and the tensors involved, for the unsafe log
inline bool anchor_safe(const hc_ctx & c, const ggml_cuda_gdn_gate_plan & plan, const ggml_tensor * dst,
                        const ggml_cuda_gdn_gate_anchor & an, std::string * why = nullptr) {
    auto fail = [&](const char * check, const ggml_tensor * a, const ggml_tensor * b) {
        if (why) {
            *why = std::string(check) + ": anchor " + tdesc(dst, c) + (a ? "; " + tdesc(a, c) : "") + (b ? "; " + tdesc(b, c) : "");
        }
        return false;
    };
    const auto ia = c.pos.find(dst);
    const auto ic = plan.chains.find(dst);
    if (ia == c.pos.end() || ic == plan.chains.end()) return fail("anchor-not-in-graph", nullptr, nullptr);
    int chain_start = ia->second;
    for (const ggml_tensor * t : ic->second) {
        const auto ip = c.pos.find(t);
        if (ip == c.pos.end()) return fail("chain-node-not-in-graph", t, nullptr);
        chain_start = std::min(chain_start, ip->second);
    }
    const ggml_tensor * outs[2] = { an.beta, an.g };
    const ggml_tensor * ins[5]  = { an.x, an.wb, an.wa, an.dt, an.a };
    for (const ggml_tensor * o : outs) {
        if (!o->data) return fail("sink-no-data", o, nullptr);
    }
    if (overlaps(an.beta, an.g)) return fail("sinks-overlap", an.beta, an.g);
    for (const ggml_tensor * x : ins) {
        if (!x->data) return fail("input-no-data", x, nullptr);
        for (const ggml_tensor * o : outs) {
            // an exact same-index alias of g with ssm_dt / ssm_a is harmless: the fold thread that writes g[t*H + h] reads
            // dt[h] / a[h] just before, and only ntok == 1 (index t*H + h == h) can produce it. ggml-alloc makes it when the
            // [H, 1, 1] ADD(alpha, dt) runs in place on an input-leaf dt (host debug, test-gdn-gate-fold ntok=1 graph);
            // served models keep dt / a in the weight buffer, which is never an in-place target.
            const bool same_index = o == an.g && (x == an.dt || x == an.a) && an.ntok == 1 && o->data == x->data &&
                                    ggml_nbytes(o) == ggml_nbytes(x);
            if (overlaps(o, x) && !same_index) return fail("sink-overlaps-input", o, x);
        }
        const ggml_tensor * r  = root_of(x);
        const auto          ip = c.pos.find(r);
        const int           i0 = std::max(chain_start + 1, ip == c.pos.end() ? 0 : ip->second + 1);
        for (int i = i0; i < ia->second; ++i) {
            const ggml_tensor * w = c.graph->nodes[i];
            if (w == r || is_noop(w) || plan.skip.count(w) || !w->data) continue;
            if (overlaps(w, x)) return fail("input-overwritten-before-anchor", x, w);
        }
    }
    return true;
}

} // namespace ggml_gdn_gate_plan_detail

inline void ggml_gdn_gate_plan_build(const ggml_cgraph * cgraph, ggml_cuda_gdn_gate_plan & plan) {
    using namespace ggml_gdn_gate_plan_detail;
    plan.skip.clear();
    plan.anchors.clear();
    plan.chains.clear();
    plan.n_matched = 0;
    plan.n_unsafe  = 0;
    plan.unsafe_why.clear();
    bool any = false;
    for (int i = 0; i < cgraph->n_nodes && !any; ++i) {
        any = cgraph->nodes[i]->op == GGML_OP_GATED_DELTA_NET;
    }
    if (!any) {
        return;
    }
    hc_ctx c;
    c.graph = cgraph;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        c.pos[n] = i;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (n->src[j]) c.uses[n->src[j]]++;
        }
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        try_gdn(c, cgraph->nodes[i], plan);
    }
    for (const auto & kv : plan.anchors) {
        if (plan.skip.count(kv.first)) {   // never both elided and an anchor; fuse nothing if it happens
            plan.skip.clear();
            plan.anchors.clear();
            plan.chains.clear();
            return;
        }
    }
    for (bool changed = true; changed;) {
        changed = false;
        std::vector<const ggml_tensor *> order;
        order.reserve(plan.anchors.size());
        for (const auto & kv : plan.anchors) order.push_back(kv.first);
        std::sort(order.begin(), order.end(), [&](const ggml_tensor * x, const ggml_tensor * y) { return c.pos[x] < c.pos[y]; });
        for (const ggml_tensor * dst : order) {
            std::string why;
            if (anchor_safe(c, plan, dst, plan.anchors.at(dst), &why)) continue;
            plan.unsafe_why.push_back(why);
            for (const ggml_tensor * t : plan.chains[dst]) plan.skip.erase(t);
            plan.chains.erase(dst);
            plan.anchors.erase(dst);
            plan.n_unsafe++;
            changed = true;
        }
    }
}
