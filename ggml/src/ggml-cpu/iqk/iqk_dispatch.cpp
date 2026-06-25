//
// iqk port (Stage 1): dispatch hook. Routes dense GGML_OP_MUL_MAT for the quant
// families iqk really implements (Q4_K/Q5_K/Q6_K/Q2_K/Q3_K via kquants,
// Q8_0/Q4_0/Q5_0/Q6_0/Q4_1/Q5_1 via legacy) to ik's iqk_mul_mat_4d. Mirrors
// ik_llama's own integration (ggml.c:17003): quantize src1 F32 -> Q8_2_X4 into
// params->wdata cooperatively, barrier, then call the iqk GEMM. Falls through to
// v6's native kernel for everything else. Runtime gate: env GGML_IQK=1.
//
#include "iqk_config.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"

#if defined(IQK_IMPLEMENT) && defined(GGML_USE_IQK_MULMAT)

#include "iqk_mul_mat.h"
#include <cstdlib>
#include <cstdio>

// activation quantizer (iqk_quantize_min.cpp)
extern "C" void quantize_row_q8_2_x4(const float * x, void * vy, int64_t k);

namespace {
inline bool iqk_enabled() {
    // C++ magic-static: thread-safe one-time init
    static const bool e = []() { const char * s = getenv("GGML_IQK"); return s && atoi(s) != 0; }();
    return e;
}
inline bool iqk_typeA_supported(int t) {
    switch (t) {
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_1:   // note: Q6_0 is ik-only, not in v6
            return true;
        default: return false;
    }
}
} // namespace

extern "C" bool ggml_iqk_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!iqk_enabled()) return false;

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (dst->type  != GGML_TYPE_F32) return false;
    if (src1->type != GGML_TYPE_F32) return false;          // Stage 1: F32 activations only
    if (!iqk_typeA_supported((int) src0->type)) return false;

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    if (ne00 != ne10)   return false;
    if (ne00 % 32 != 0) return false;                       // QK8_2 = 32

    // ggml guarantees ne0==ne01, ne1==ne11, ne2==ne12, ne3==ne13 and unpermuted src0/src1
    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    const int ith = params->ith, nth = params->nth;

    // Quantize src1 (F32) -> Q8_2_X4 into params->wdata, cooperatively by row.
    // row_size = (ne10/QK8_2) * sizeof(block_q8_2) = (ne10/32) * 36.
    const size_t row_size = (size_t)(ne10 / 32) * 36;
    const size_t nbw1 = row_size, nbw2 = nbw1 * ne11, nbw3 = nbw2 * ne12;
    if (params->wsize < (size_t) ne13 * nbw3) return false; // safety: Q8_2_X4 <= Q8_K so should always fit

    char * wdata = (char *) params->wdata;
    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                quantize_row_q8_2_x4(
                    (const float *)((const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                    wdata + i13*nbw3 + i12*nbw2 + i11*nbw1, ne10);
            }
        }
    }
    ggml_barrier(params->threadpool);

    const bool ok = iqk_mul_mat_4d(ne01, ne11, ne00,
            ne02, ne03, ne12, ne13,
            nb02, nb03, nbw2, nbw3,
            nb2 / sizeof(float), nb3 / sizeof(float),
            (int) src0->type, src0->data, nb01,
            GGML_TYPE_Q8_2_X4, wdata, row_size,
            (float *) dst->data, nb1 / sizeof(float), ith, nth);
    if (ok && ith == 0) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=%d ne00=%lld)\n", (int) src0->type, (long long) ne00); }
    }
    return ok;
}

#else  // iqk not implemented / disabled — no-op hook

extern "C" bool ggml_iqk_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    (void) params; (void) dst;
    return false;
}

#endif
