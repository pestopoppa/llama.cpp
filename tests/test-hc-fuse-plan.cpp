// AK-ASM-H13b regression test (C7R, 2026-11-06): the hc seam fusion planner (ggml/src/ggml-cuda/hc-fuse-plan.h)
// must never change the result of a graph, whatever ggml-alloc does with the buffers and however the scheduler
// splits the graph.
//
// The fused kernels run on the GPU only, but the planner is host C++ and the defect class is about BUFFER LIFETIMES, so
// it reproduces on the CPU: build the qwen4exp hyper-connection graph exactly as src/models/qwen4exp.cpp does, split and
// allocate it the way ggml_backend_sched does for `-ngl 99 --n-cpu-moe N` (GPU splits | CPU expert splits, input copies
// at every split boundary, one gallocr buffer per backend, tensor memory reused after its last *unfused* consumer),
// plan every GPU split, and execute node by node on the CPU backend with the plan applied -- elided nodes skipped,
// anchors evaluated by a scalar CPU mirror of the hc-fuse.cu kernels (all reads before any write, as concurrent GPU
// threads may). The result must match the same allocation run unfused.
//
// Cases:
//   ngl99       - no CPU experts: every split is a GPU split
//   n-cpu-moe   - the routed experts of every layer run in a CPU split (decode-size batches)
//   op-offload  - the routed experts run on the GPU from copies of their CPU-resident weights, every expert matmul
//                 starting a new split: what -ngl 99 --n-cpu-moe 48 -ub 512 does at prompt/PPL batch sizes (CUDA
//                 offload_op), i.e. the C7 Q38FN gate and bisect configuration. On c420b9d1a this case fails: the
//                 hc_inject MUL_MAT is allocated over the block_out that the fused COMBINE still reads
//   cross-split - an elidable SCALE has a second consumer in a LATER GPU split (uses must be counted over the whole
//                 graph, like upstream ggml_node_get_use_count, not over the split)
// Every case must also still fuse each of the 4 kinds at least once (the fix keeps H13's speed for the safe cases),
// and no fused kernel may write a dst that partially overlaps one of its own inputs (concurrent GPU threads would race).
//
// HC_TEST_DIMS="E,H,T,LR,NL" overrides the shapes (the Q38FN shapes are 2560,4,512,320,<n>).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../ggml/src/ggml-cuda/hc-fuse-plan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

int64_t E = 320, H = 4, T = 64, LR = 40;
int     NL = 3;
constexpr int64_t NEXP = 16;  // router width (stand-in)

struct hc_w {
    ggml_tensor * norm, * down, * up, * inject;
};

struct layer_w {
    hc_w          attn, ffn;
    ggml_tensor * wa1, * wa2;            // attention stand-in
    ggml_tensor * wr;                    // router
    ggml_tensor * eg, * eu, * ed;        // routed experts (CPU with --n-cpu-moe)
    ggml_tensor * sg, * su, * sd;        // shared expert (GPU)
};

ggml_tensor * mk(ggml_context * ctx, int64_t a, int64_t b = 1) {
    return b == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b);
}

hc_w mk_hc(ggml_context * ctx, bool inject) {
    return { mk(ctx, H * E), mk(ctx, H * E, LR), mk(ctx, LR, H * E), inject ? mk(ctx, H * E, H) : nullptr };
}

void fill(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(ggml_nelements(t));
    for (auto & x : v) x = d(rng);
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

// --- the qwen4exp graph pieces, verbatim from src/models/qwen4exp.cpp (build_hc_mix / build_hc_combine)
ggml_tensor * hc_mix(ggml_context * ctx0, ggml_tensor * x, const hc_w & w, ggml_tensor ** inject) {
    const int64_t nt = x->ne[2];
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, 1e-6f);
    xn = ggml_reshape_2d(ctx0, xn, H * E, nt);
    xn = ggml_mul(ctx0, xn, w.norm);
    ggml_tensor * lo = ggml_mul_mat(ctx0, w.down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) H));
    ggml_tensor * gate  = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, w.up, lo));
    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, E, H, nt);
    ggml_tensor * mixed = ggml_mean_d1(ctx0, gated);
    mixed = ggml_reshape_2d(ctx0, mixed, E, nt);
    if (inject) {
        *inject = ggml_mul_mat(ctx0, w.inject, xn);
    }
    return mixed;
}

ggml_tensor * hc_combine(ggml_context * ctx0, ggml_tensor * residual, ggml_tensor * block_out, ggml_tensor * inject) {
    const int64_t nt = residual->ne[2];
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) H));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, H, nt);
    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, E, 1, nt);
    b = ggml_repeat_4d(ctx0, b, E, H, nt, 1);
    return ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
}

struct built {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_tensor *   out = nullptr;
    std::unordered_set<const ggml_tensor *> cpu_nodes;     // nodes the scheduler would put on the CPU backend
    std::unordered_set<const ggml_tensor *> offload_w;     // CPU-resident expert weights of ops offloaded to the GPU
};

built build_graph(const std::vector<layer_w> & L, const hc_w & head, ggml_tensor * wout, ggml_tensor * inp,
                  int moe_mode, bool cross_split) {
    built b;
    ggml_init_params ip = { 16384 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false), nullptr, true };
    b.ctx = ggml_init(ip);
    ggml_context * ctx0 = b.ctx;
    b.gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor * res = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inp, E, 1, T), E, H, T, 1);
    for (int il = 0; il < NL; ++il) {
        const layer_w & l = L[il];
        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = hc_mix(ctx0, res, l.attn, &inject);
        ggml_build_forward_expand(b.gf, cur);
        cur = ggml_mul_mat(ctx0, l.wa2, ggml_silu(ctx0, ggml_mul_mat(ctx0, l.wa1, cur)));  // attention stand-in
        res = hc_combine(ctx0, res, cur, inject);

        if (cross_split && il == 0) {
            // SILU(SCALE(x)) whose SCALE has a second consumer in the next GPU split, behind a CPU split
            ggml_tensor * sc = ggml_scale(ctx0, res, 0.5f);
            ggml_tensor * sl = ggml_silu(ctx0, sc);
            ggml_tensor * c  = ggml_cpy(ctx0, sl, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, E, H, T));  // CPU split
            ggml_build_forward_expand(b.gf, c);
            b.cpu_nodes.insert(c);
            res = ggml_add(ctx0, res, ggml_scale(ctx0, ggml_add(ctx0, c, sc), 0.1f));  // GPU: reads the SCALE again
        }

        cur = hc_mix(ctx0, res, l.ffn, &inject);
        // MoE stand-in: router (GPU) -> routed experts (CPU with --n-cpu-moe) -> + shared expert (GPU)
        ggml_tensor * probs = ggml_soft_max(ctx0, ggml_mul_mat(ctx0, l.wr, cur));
        ggml_tensor * wsum  = ggml_sum_rows(ctx0, ggml_sqr(ctx0, probs));  // [1, T] per-token routing weight
        ggml_tensor * eg    = ggml_mul_mat(ctx0, l.eg, cur);
        ggml_tensor * eu    = ggml_mul_mat(ctx0, l.eu, cur);
        ggml_tensor * ea    = ggml_mul(ctx0, ggml_silu(ctx0, eg), eu);
        ggml_tensor * ed    = ggml_mul_mat(ctx0, l.ed, ea);
        ggml_tensor * moe   = ggml_mul(ctx0, ed, wsum);
        if (moe_mode == 1) {
            for (ggml_tensor * t : { eg, eu, ea, ea->src[0], ed, moe }) b.cpu_nodes.insert(t);
        } else if (moe_mode == 2) {
            // op offload (CUDA offload_op, batch >= 32): the expert matmuls run on the GPU, each reading a copy of its
            // CPU-resident weight, and each starts a new split (ggml_backend_sched need_new_split on a foreign weight)
            for (ggml_tensor * w : { l.eg, l.eu, l.ed }) b.offload_w.insert(w);
        }
        ggml_tensor * sh = ggml_mul_mat(ctx0, l.sd, ggml_mul(ctx0, ggml_silu(ctx0, ggml_mul_mat(ctx0, l.sg, cur)),
                                                             ggml_mul_mat(ctx0, l.su, cur)));
        cur = ggml_add(ctx0, moe, sh);
        res = hc_combine(ctx0, res, cur, inject);

    }
    ggml_tensor * cur = hc_mix(ctx0, res, head, nullptr);
    b.out = ggml_mul_mat(ctx0, wout, cur);
    ggml_set_output(b.out);
    ggml_build_forward_expand(b.gf, b.out);
    return b;
}

bool is_noop(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE;
}

const ggml_tensor * root_of(const ggml_tensor * t) {
    while (t->view_src) t = t->view_src;
    return t;
}

int g_alias_violations = 0;

float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// scalar CPU mirror of hc-fuse.cu ggml_cuda_hc_fuse_run (same reads, same writes, same index maps); every read
// happens before the first write, as concurrent GPU threads may order them
void run_anchor(ggml_tensor * dst, const ggml_cuda_hc_anchor & an) {
    std::vector<float> tmp(ggml_nelements(dst));
    switch (an.kind) {
        case HC_FUSE_COMBINE: {
            const int64_t e_ = dst->ne[0], h_ = dst->ne[1], t_ = dst->ne[2];
            const float * res = (const float *) an.a->data;
            const float * bo  = (const float *) an.b->data;
            const float * inj = (const float *) an.c->data;
            for (int64_t i = 0; i < e_ * h_ * t_; ++i) {
                const int64_t e = i % e_, h = (i / e_) % h_, t = i / (e_ * h_);
                const float w = an.s2 * sigm(an.s1 * inj[t * h_ + h] + an.b1) + an.b2;
                tmp[i] = res[i] + bo[t * e_ + e] * w;
            }
        } break;
        case HC_FUSE_MIX_TAIL: {
            const int64_t e_ = dst->ne[0], t_ = dst->ne[2];
            const int64_t h_ = ggml_nelements(an.a) / (e_ * t_);
            const float * xn = (const float *) an.a->data;
            const float * u  = (const float *) an.b->data;
            for (int64_t i = 0; i < e_ * t_; ++i) {
                const int64_t e = i % e_, t = i / e_;
                float sum = 0.0f;
                for (int64_t h = 0; h < h_; ++h) {
                    const int64_t j = t * e_ * h_ + h * e_ + e;
                    sum += xn[j] * sigm(u[j]);
                }
                tmp[i] = sum / (float) h_;
            }
        } break;
        case HC_FUSE_RMS_GAMMA: {
            const ggml_tensor * rms = an.c;
            const ggml_tensor * x   = rms->src[0];
            float eps;
            memcpy(&eps, rms->op_params, sizeof(float));
            const int64_t ne0 = x->ne[0], nr = ggml_nrows(x), cols = ne0 * x->ne[1];
            const float * xs  = (const float *) x->data;
            const float * gam = (const float *) an.b->data;
            for (int64_t r = 0; r < nr; ++r) {
                double ss = 0.0;
                for (int64_t k = 0; k < ne0; ++k) ss += (double) xs[r * ne0 + k] * xs[r * ne0 + k];
                const float sc = 1.0f / std::sqrt((float) (ss / ne0) + eps);
                for (int64_t k = 0; k < ne0; ++k) {
                    const int64_t flat = r * ne0 + k;
                    tmp[flat] = xs[flat] * sc * gam[flat % cols];
                }
            }
        } break;
        case HC_FUSE_SCALE_SILU: {
            const float * x = (const float *) an.a->data;
            for (size_t i = 0; i < tmp.size(); ++i) {
                const float y = an.s1 * x[i] + an.b1;
                tmp[i] = y / (1.0f + std::exp(-y));
            }
        } break;
    }
    memcpy(dst->data, tmp.data(), tmp.size() * sizeof(float));
}

struct split_t {
    int i0, i1, backend;  // backend 0 = "GPU" (planned), 1 = CPU
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> inputs;  // (source, copy in this split's buffer)
};

// ggml_backend_sched, reduced to what matters here: splits = runs of nodes on one backend; a split reads every tensor
// produced on another backend through a copy allocated at the split start; one gallocr buffer per backend, sized and
// laid out by ggml_gallocr over the graph copy [input deps, input copies, split nodes]...
struct sched_emu {
    std::vector<split_t> splits;
    ggml_gallocr_t       galloc = nullptr;
    ggml_context *       ctx    = nullptr;
};

void sched_build(built & b, sched_emu & s) {
    ggml_init_params ip = { 8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32768, false), nullptr, true };
    s.ctx = ggml_init(ip);
    const int n = ggml_graph_n_nodes(b.gf);
    std::unordered_map<const ggml_tensor *, int> home;  // producing backend of every node
    auto backend_of = [&](ggml_tensor * t) {
        return b.cpu_nodes.count(root_of(t)) || b.cpu_nodes.count(t) ? 1 : 0;
    };
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(b.gf, i);
        home[t] = backend_of(t);
    }
    auto reads_offload_w = [&](ggml_tensor * t) {
        for (int k = 0; k < GGML_MAX_SRC; ++k) if (t->src[k] && b.offload_w.count(t->src[k])) return true;
        return false;
    };
    for (int i = 0; i < n;) {
        int j = i + 1;
        while (j < n && home[ggml_graph_node(b.gf, j)] == home[ggml_graph_node(b.gf, i)] && !reads_offload_w(ggml_graph_node(b.gf, j))) ++j;
        s.splits.push_back({ i, j, home[ggml_graph_node(b.gf, i)], {} });
        i = j;
    }
    ggml_cgraph *    gc = ggml_new_graph_custom(s.ctx, 32768, false);
    std::vector<int> ids;
    for (split_t & sp : s.splits) {
        std::unordered_map<ggml_tensor *, ggml_tensor *> cpy;
        for (int i = sp.i0; i < sp.i1; ++i) {
            ggml_tensor * node = ggml_graph_node(b.gf, i);
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                ggml_tensor * src = node->src[k];
                const bool offw = src && b.offload_w.count(src);
                if (!src || (!offw && (!home.count(src) || home[src] == sp.backend))) continue;  // GPU weights/inputs: no copy
                auto it = cpy.find(src);
                if (it == cpy.end()) {
                    ggml_tensor * c = ggml_dup_tensor(s.ctx, src);
                    ggml_tensor * dep = ggml_view_tensor(s.ctx, src);
                    dep->src[0] = src;
                    ggml_graph_add_node(gc, dep); ids.push_back(offw ? 1 : home[src]);
                    ggml_graph_add_node(gc, c);   ids.push_back(sp.backend);
                    sp.inputs.push_back({ src, c });
                    it = cpy.emplace(src, c).first;
                }
                node->src[k] = it->second;
            }
        }
        for (int i = sp.i0; i < sp.i1; ++i) {
            ggml_graph_add_node(gc, ggml_graph_node(b.gf, i));
            ids.push_back(sp.backend);
        }
    }
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_cpu_buffer_type(), ggml_backend_cpu_buffer_type() };
    s.galloc = ggml_gallocr_new_n(bufts, 2);
    std::vector<int> leaf_ids(1, 0);  // the graph copy has no leafs (weights and the input live in their own buffer)
    if (!ggml_gallocr_reserve_n(s.galloc, gc, ids.data(), leaf_ids.data()) || !ggml_gallocr_alloc_graph(s.galloc, gc)) {
        fprintf(stderr, "gallocr failed\n");
        exit(2);
    }
}

std::vector<float> run(ggml_backend_t cpu, built & b, sched_emu & s, ggml_tensor * inp, const std::vector<float> & x, bool plan_on,
                       int counts[5]) {
    // poison every graph buffer: a read of memory nobody wrote this run must not see the previous run's values
    for (const split_t & sp : s.splits) {
        for (const auto & io : sp.inputs) memset(io.second->data, 0xff, ggml_nbytes(io.second));
        for (int i = sp.i0; i < sp.i1; ++i) {
            ggml_tensor * node = ggml_graph_node(b.gf, i);
            if (!node->view_src && node->data) memset(node->data, 0xff, ggml_nbytes(node));
        }
    }
    ggml_backend_tensor_set(inp, x.data(), 0, ggml_nbytes(inp));
    for (int k = 0; k < 5; ++k) counts[k] = 0;
    for (const split_t & sp : s.splits) {
        for (const auto & io : sp.inputs) memcpy(io.second->data, io.first->data, ggml_nbytes(io.first));
        ggml_cgraph       view = ggml_graph_view(b.gf, sp.i0, sp.i1);
        ggml_cuda_hc_plan plan;
        if (plan_on && sp.backend == 0) {
            ggml_hc_fuse_plan_build(&view, true, plan);
            if (getenv("HC_TEST_VERBOSE")) {
                for (int i = sp.i0; i < sp.i1; ++i) {
                    ggml_tensor * node = ggml_graph_node(b.gf, i);
                    const auto it = plan.anchors.find(node);
                    printf("  [%4d] %-10s %-6s %s\n", i, ggml_op_desc(node), plan.skip.count(node) ? "skip" : "",
                           it != plan.anchors.end() ? std::to_string((int) it->second.kind).c_str() : "");
                }
            }
        }
        for (int i = sp.i0; i < sp.i1; ++i) {
            ggml_tensor * node = ggml_graph_node(b.gf, i);
            if (is_noop(node) || plan.skip.count(node)) continue;
            auto it = plan.anchors.find(node);
            if (it != plan.anchors.end()) {
                const ggml_cuda_hc_anchor & an = it->second;
                const ggml_tensor * ins[3] = { an.a, an.b, an.kind == HC_FUSE_RMS_GAMMA ? an.c->src[0] : an.c };
                for (const ggml_tensor * in : ins) {
                    if (!in || !in->data) continue;
                    const char * d0 = (const char *) node->data, * d1 = d0 + ggml_nbytes(node);
                    const char * i0 = (const char *) in->data,   * i1 = i0 + ggml_nbytes(in);
                    const bool same = d0 == i0 && d1 == i1 && an.kind != HC_FUSE_MIX_TAIL;  // same-index in-place
                    if (d0 < i1 && i0 < d1 && !same) {
                        g_alias_violations++;
                    }
                }
                run_anchor(node, an);
                counts[it->second.kind]++;
                continue;
            }
            ggml_cgraph one = ggml_graph_view(b.gf, i, i + 1);
            ggml_backend_graph_compute(cpu, &one);
        }
    }
    std::vector<float> out(ggml_nelements(b.out));
    ggml_backend_tensor_get(b.out, out.data(), 0, ggml_nbytes(b.out));
    return out;
}

} // namespace

int main() {
    if (const char * d = getenv("HC_TEST_DIMS")) {
        long long e, h, t, lr; int nl;
        if (sscanf(d, "%lld,%lld,%lld,%lld,%d", &e, &h, &t, &lr, &nl) == 5) { E = e; H = h; T = t; LR = lr; NL = nl; }
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, getenv("HC_TEST_THREADS") ? atoi(getenv("HC_TEST_THREADS")) : 2);

    ggml_init_params ip = { (size_t) (32 * NL + 16) * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(ip);
    std::vector<layer_w> L;
    for (int il = 0; il < NL; ++il) {
        layer_w l;
        l.attn = mk_hc(wctx, true);
        l.ffn  = mk_hc(wctx, true);
        l.wa1 = mk(wctx, E, 2 * E); l.wa2 = mk(wctx, 2 * E, E);
        l.wr  = mk(wctx, E, NEXP);
        l.eg  = mk(wctx, E, E / 2); l.eu = mk(wctx, E, E / 2); l.ed = mk(wctx, E / 2, E);
        l.sg  = mk(wctx, E, E / 2); l.su = mk(wctx, E, E / 2); l.sd = mk(wctx, E / 2, E);
        L.push_back(l);
    }
    hc_w          head = mk_hc(wctx, false);
    ggml_tensor * wout = mk(wctx, E, 32);
    ggml_tensor * inp  = mk(wctx, E, T);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, cpu);
    {
        std::mt19937 rng(1234);
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t == inp) continue;
            if (t->ne[1] == 1) {
                fill(t, rng, 0.5f, 1.5f);
            } else {
                const float s = 1.5f / std::sqrt((float) t->ne[0]);
                fill(t, rng, -s, s);
            }
        }
    }
    std::vector<float> x(E * T);
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto & v : x) v = d(rng);
    }
    printf("test-hc-fuse-plan: E=%lld H=%lld T=%lld LR=%lld NL=%d\n", (long long) E, (long long) H, (long long) T, (long long) LR, NL);

    bool all_ok = true;
    const char * names[4] = { "ngl99", "n-cpu-moe", "op-offload", "cross-split" };
    for (int c = 0; c < 4; ++c) {
        built     b = build_graph(L, head, wout, inp, c == 3 ? 1 : c, c == 3);
        sched_emu s;
        sched_build(b, s);
        int nofuse[5], fused[5];
        const std::vector<float> ref = run(cpu, b, s, inp, x, false, nofuse);
        g_alias_violations = 0;
        const std::vector<float> got = run(cpu, b, s, inp, x, true, fused);
        double max_ref = 0.0, max_err = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            max_ref = std::max(max_ref, (double) std::fabs(ref[i]));
            max_err = std::max(max_err, (double) std::fabs(ref[i] - got[i]));
        }
        const double rel     = max_err / (max_ref > 0 ? max_ref : 1.0);
        const bool   all4    = fused[1] > 0 && fused[2] > 0 && fused[3] > 0 && fused[4] > 0;
        const bool   ok      = std::isfinite(rel) && rel < 1e-4 && all4 && g_alias_violations == 0;
        int n_gpu = 0;
        for (const auto & sp : s.splits) n_gpu += sp.backend == 0;
        printf("%-12s %s  rel_err=%.3e  splits=%zu (gpu %d)  fused: combine=%d mix_tail=%d scale_silu=%d rms_gamma=%d%s\n", names[c],
               ok ? "OK  " : "FAIL", rel, s.splits.size(), n_gpu, fused[HC_FUSE_COMBINE], fused[HC_FUSE_MIX_TAIL],
               fused[HC_FUSE_SCALE_SILU], fused[HC_FUSE_RMS_GAMMA], all4 ? "" : "  <- a fusion kind never ran");
        if (g_alias_violations) {
            printf("             %d fused dst(s) partially overlap their own input\n", g_alias_violations);
        }
        all_ok = all_ok && ok;
        ggml_gallocr_free(s.galloc);
        ggml_free(s.ctx);
        ggml_free(b.ctx);
    }

    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(cpu);
    printf("%s\n", all_ok ? "test-hc-fuse-plan: PASS" : "test-hc-fuse-plan: FAIL");
    return all_ok ? 0 : 1;
}
