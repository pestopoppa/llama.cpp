#pragma once
// AK-BW-27B-ACTQUANT-FUSE: fewer activation-quantize launches on the gfx90a small-M Q8_0 paths (q8sk / q8mf6).
// GGML_HIP_ACTQ_FUSE bit mask, default 0 = the C21 launch sequence unchanged:
//   bit 0 (1): q8sk folds its dst zero-init (q8sk_zero) into the q8sk_quantize launch (one launch fewer per call).
//   bit 1 (2): sibling MUL_MATs that read the same src1 and run back to back in the graph share one quantized activation:
//              the first sibling's pre-pass quantizes once into a persistent per-device buffer and zeroes every sibling's
//              dst; the later siblings launch only their GEMM kernel. Outputs are bit-identical to the knob-off path
//              (same quantization arithmetic, same GEMM kernel, same atomic order per dst).
#include "common.cuh"

#include <cstdlib>

#define ACTQ_MAXG 4

static inline int ggml_cuda_actq_mode() {
    static const int mode = [] {
        const char * s = getenv("GGML_HIP_ACTQ_FUSE");
        return s != nullptr ? atoi(s) : 0;
    }();
    return mode;
}

// dst zero-init set carried by a quantize pre-pass: n dsts, each [M][N[g]] floats with row stride s1[g] floats.
struct actq_zero {
    float * p[ACTQ_MAXG];
    int     N[ACTQ_MAXG];
    int     s1[ACTQ_MAXG];
    int     n;
    int     M;
};

// grid-stride zero of every dst in z, executed by all threads of the launch (blockIdx.y-major linearisation)
static __device__ __forceinline__ void actq_zero_dsts(const actq_zero & z) {
    const int nthr = gridDim.x * gridDim.y * blockDim.x;
    const int tid  = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    for (int g = 0; g < z.n; ++g) {
        const int tot = z.M * z.N[g];
        for (int i = tid; i < tot; i += nthr) {
            z.p[g][(int64_t) (i / z.N[g]) * z.s1[g] + i % z.N[g]] = 0.0f;
        }
    }
}

// ---- bit 1: sibling sharing (host side, actq.cu) --------------------------------------------------------------------
// kinds of shared activation layout
#define ACTQ_KIND_Q8MF 1   // q8mf_pre_i8 stage-major act (q8mf6)
#define ACTQ_KIND_Q8SK 2   // q8sk yq [16][nblk*32] int8 + yd [16][nblk] f32

// graph-eval hooks (ggml-cuda.cu): begin of every evaluate pass, and the node about to be dispatched (-1 = none)
void ggml_cuda_actq_graph_begin();
void ggml_cuda_actq_set_node(const ggml_cgraph * g, int i, int stream_no);

// Later sibling: returns the shared quantized activation already written by this node's group leader (skip the
// pre-pass and the dst zero-init), or nullptr (run the normal path).
const char * ggml_cuda_actq_reuse(ggml_backend_cuda_context & ctx, int kind, const ggml_tensor * src1,
                                  const ggml_tensor * dst);

// Group leader: if dst is the node being dispatched and the next non-view nodes are MUL_MATs on the same src1 that
// sib_ok accepts, returns a persistent per-context buffer of >= bytes to quantize into and fills z with the zero-init
// set (own dst first, then each sibling's); otherwise nullptr (run the normal path).
char * ggml_cuda_actq_begin(ggml_backend_cuda_context & ctx, int kind, const ggml_tensor * src1, const ggml_tensor * dst,
                            size_t bytes, actq_zero & z, bool (*sib_ok)(const ggml_tensor * node));
