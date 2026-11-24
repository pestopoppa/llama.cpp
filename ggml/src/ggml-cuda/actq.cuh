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
