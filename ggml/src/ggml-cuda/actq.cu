// AK-BW-27B-ACTQUANT-FUSE bit 1: sibling MUL_MATs share one quantized activation. Design in actq.cuh.
//
// Safety argument (why a later sibling may skip its pre-pass):
//  * a group is formed only by the node being dispatched (cgraph->nodes[i] == dst) on stream 0 and contains only the
//    MUL_MATs that follow it with nothing but view / no-op nodes in between, so no kernel writes src1 (or the group's
//    dsts) between the leader's pre-pass and the siblings' GEMMs;
//  * a sibling reuses only when it is dispatched as exactly the expected node (same graph, same evaluate pass, same node
//    index, same dst / src1 data and shape, same layout kind) on stream 0, so stream order covers every dependency;
//  * the buffer is persistent per backend context (allocated outside capture, never freed or regrown), so captured
//    HIP graphs keep a valid pointer; a group that needs more than its size, or a capture without the buffer, falls
//    back to the normal path;
//  * zeroing a sibling's dst ahead is harmless even if that sibling then takes another route: every MUL_MAT route
//    writes its whole dst.
#include "actq.cuh"

#include <atomic>
#include <mutex>
#include <unordered_map>

#define ACTQ_BUF_BYTES ((size_t) 4 << 20)

namespace {
struct actq_node_ctx {
    const ggml_cgraph * g     = nullptr;
    int                 i     = -1;
    int                 sno   = -1;
    uint64_t            epoch = 0;
};
struct actq_group {
    const void *        ctx   = nullptr;
    const ggml_cgraph * g     = nullptr;
    uint64_t            epoch = 0;
    int                 kind  = 0;
    int                 n     = 0;
    int                 idx[ACTQ_MAXG] = {};
    const void *        dst[ACTQ_MAXG] = {};
    const void *        y     = nullptr;
    int64_t             ne[4] = {};
    size_t              nb[4] = {};
};
thread_local actq_node_ctx t_node;
thread_local actq_group    t_grp;
thread_local uint64_t      t_epoch = 0;

std::mutex                                   g_buf_mu;
std::unordered_map<const void *, char *>     g_buf;   // backend ctx -> persistent buffer (never freed)
std::atomic<int64_t>                         g_n_groups{0};
std::atomic<int64_t>                         g_n_reuse{0};
}

void ggml_cuda_actq_graph_begin() {
    ++t_epoch;
    t_node = {};
    t_grp  = {};
}

void ggml_cuda_actq_set_node(const ggml_cgraph * g, int i, int stream_no) {
    t_node.g     = g;
    t_node.i     = i;
    t_node.sno   = stream_no;
    t_node.epoch = t_epoch;
}

static bool actq_same_src1(const ggml_tensor * a, const void * y, const int64_t * ne, const size_t * nb) {
    if (a == nullptr || a->type != GGML_TYPE_F32 || a->data != y) {
        return false;
    }
    for (int k = 0; k < 4; ++k) {
        if (a->ne[k] != ne[k] || a->nb[k] != nb[k]) {
            return false;
        }
    }
    return true;
}

static bool actq_node_live(const ggml_tensor * dst) {
    return (ggml_cuda_actq_mode() & 2) && t_node.g != nullptr && t_node.i >= 0 && t_node.sno == 0 &&
        t_node.epoch == t_epoch && t_node.i < t_node.g->n_nodes && t_node.g->nodes[t_node.i] == dst;
}

const char * ggml_cuda_actq_reuse(ggml_backend_cuda_context & ctx, int kind, const ggml_tensor * src1,
                                  const ggml_tensor * dst) {
    if (!actq_node_live(dst) || t_grp.n == 0 || t_grp.kind != kind || t_grp.ctx != (const void *) &ctx ||
        t_grp.g != t_node.g || t_grp.epoch != t_epoch || !actq_same_src1(src1, t_grp.y, t_grp.ne, t_grp.nb)) {
        return nullptr;
    }
    for (int k = 1; k < t_grp.n; ++k) {
        if (t_grp.idx[k] == t_node.i && t_grp.dst[k] == dst->data) {
            std::lock_guard<std::mutex> lk(g_buf_mu);
            const auto it = g_buf.find((const void *) &ctx);
            if (it == g_buf.end()) {
                return nullptr;
            }
            const int64_t r = g_n_reuse.fetch_add(1) + 1;
            if (r == 1) {
                GGML_LOG_INFO("ggml_cuda_actq: sibling activation reuse ran (call 1, kind=%d, GGML_HIP_ACTQ_FUSE=%d)\n",
                    kind, ggml_cuda_actq_mode());
            }
            return it->second;
        }
    }
    return nullptr;
}

char * ggml_cuda_actq_begin(ggml_backend_cuda_context & ctx, int kind, const ggml_tensor * src1, const ggml_tensor * dst,
                            size_t bytes, actq_zero & z, bool (*sib_ok)(const ggml_tensor * node)) {
    t_grp = {};
    if (!actq_node_live(dst) || bytes > ACTQ_BUF_BYTES || src1->type != GGML_TYPE_F32) {
        return nullptr;
    }
    const ggml_cgraph * g = t_node.g;
    actq_group grp;
    grp.ctx   = (const void *) &ctx;
    grp.g     = g;
    grp.epoch = t_epoch;
    grp.kind  = kind;
    grp.y     = src1->data;
    for (int k = 0; k < 4; ++k) {
        grp.ne[k] = src1->ne[k];
        grp.nb[k] = src1->nb[k];
    }
    grp.idx[0] = t_node.i;
    grp.dst[0] = dst->data;
    grp.n      = 1;
    for (int j = t_node.i + 1; j < g->n_nodes && grp.n < ACTQ_MAXG; ++j) {
        const ggml_tensor * nd = g->nodes[j];
        if (ggml_is_empty(nd) || nd->op == GGML_OP_RESHAPE || nd->op == GGML_OP_TRANSPOSE || nd->op == GGML_OP_VIEW ||
            nd->op == GGML_OP_PERMUTE || nd->op == GGML_OP_NONE) {
            continue;
        }
        if (nd->op != GGML_OP_MUL_MAT || nd->type != GGML_TYPE_F32 || nd->src[0] == nullptr ||
            !actq_same_src1(nd->src[1], grp.y, grp.ne, grp.nb) || nd->ne[1] != dst->ne[1] ||
            nd->ne[2] != 1 || nd->ne[3] != 1 || nd->nb[0] != sizeof(float) || nd->nb[1] % sizeof(float) != 0 ||
            nd->data == nullptr || !sib_ok(nd)) {
            break;
        }
        grp.idx[grp.n] = j;
        grp.dst[grp.n] = nd->data;
        ++grp.n;
    }
    if (grp.n < 2) {
        return nullptr;
    }
    char * buf = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_buf_mu);
        const auto it = g_buf.find((const void *) &ctx);
        if (it != g_buf.end()) {
            buf = it->second;
        } else {
            cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
            CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &st));
            if (st != cudaStreamCaptureStatusNone) {
                return nullptr;   // never allocate during capture; the next uncaptured pass allocates
            }
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaMalloc((void **) &buf, ACTQ_BUF_BYTES));
            g_buf.emplace((const void *) &ctx, buf);
        }
    }
    z   = {};
    z.M = (int) dst->ne[1];
    for (int k = 0; k < grp.n; ++k) {
        const ggml_tensor * d = k == 0 ? dst : g->nodes[grp.idx[k]];
        z.p[k]  = (float *) d->data;
        z.N[k]  = (int) d->ne[0];
        z.s1[k] = (int) (d->nb[1] / sizeof(float));
    }
    z.n   = grp.n;
    t_grp = grp;
    const int64_t c = g_n_groups.fetch_add(1) + 1;
    if (c == 1 || (c & (c - 1)) == 0) {
        GGML_LOG_INFO("ggml_cuda_actq: sibling group formed (groups=%lld, kind=%d, size=%d, K=%lld, M=%lld)\n",
            (long long) c, kind, grp.n, (long long) src1->ne[0], (long long) src1->ne[1]);
    }
    return buf;
}
