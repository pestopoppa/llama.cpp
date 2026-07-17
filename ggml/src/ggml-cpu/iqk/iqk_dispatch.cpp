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
#include <cstring>
#include <cstdint>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

// iqk port DEBUG: SIGSEGV backtrace handler (no gdb on host). Installed once when
// GGML_IQK_DEBUG_SEGV=1. Prints the crashing call stack to stderr.
namespace {
void iqk_segv_handler(int sig) {
    void * bt[64];
    int n = backtrace(bt, 64);
    fprintf(stderr, "\n[iqk] SIGSEGV (%d) — backtrace (%d frames):\n", sig, n);
    backtrace_symbols_fd(bt, n, STDERR_FILENO);
    _exit(139);
}
struct IqkSegvInstaller {
    IqkSegvInstaller() { if (const char * s = getenv("GGML_IQK_DEBUG_SEGV"); s && atoi(s)) signal(SIGSEGV, iqk_segv_handler); }
};
IqkSegvInstaller iqk_segv_installer_;
} // namespace

// activation quantizer (iqk_quantize_min.cpp)
extern "C" void quantize_row_q8_2_x4(const float * x, void * vy, int64_t k);

namespace {
inline bool iqk_enabled() {
    // C++ magic-static: thread-safe one-time init
    static const bool e = []() { const char * s = getenv("GGML_IQK"); return s && atoi(s) != 0; }();
    return e;
}
inline bool iqk_q8_0_enabled() {
    static const bool e = []() {
        const char * s = getenv("GGML_IQK_Q8_0");
        return s && atoi(s) != 0;
    }();
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
    if (src0->type == GGML_TYPE_Q8_0 && !iqk_q8_0_enabled()) return false;

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

// ---------------------------------------------------------------------------
// iqk port (Stage 2): MoE expert GEMM hook for GGML_OP_MUL_MAT_ID.
// Mirrors the dense hook but: (a) OWNS the Q8_2_X4 src1 quantization (v6's stock
// vec_dot_type for Q4_K is Q8_K — wrong for iqk), (b) builds the per-expert row
// mapping with v6's stride (ids->ne[0]*ids->ne[1], NOT ik's ne12), (c) runs
// iqk_mul_mat_moe per expert. Returns true if handled, else false -> v6 native.
// Stage 2 covers the kquants families (Q8_2_X4 activation); the repack guard
// already keeps these expert weights out of CPU_REPACK so this hook isn't starved.
// ---------------------------------------------------------------------------
namespace { struct iqk_mmid { int32_t i1; int32_t i2; }; } // layout-matches mmid_row_mapping

extern "C" bool ggml_iqk_try_mul_mat_id(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!iqk_enabled()) return false;
    const struct ggml_tensor * src0 = dst->src[0]; // expert weights [ne00, ne01, n_expert]
    const struct ggml_tensor * src1 = dst->src[1]; // F32 activations
    const struct ggml_tensor * ids  = dst->src[2];
    if (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) return false;
    const int tA = (int) src0->type;
    // Same families as the dense hook (kquants + legacy Q8_0/Q4_0/...); all use the
    // Q8_2_X4 activation. iqk_mul_mat_moe returns false for any it can't handle -> native.
    if (!iqk_typeA_supported(tA)) return false;
    if (tA == GGML_TYPE_Q8_0 && !iqk_q8_0_enabled()) return false;

    const int64_t ne01 = src0->ne[1], ne02 = src0->ne[2];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    if (ne13 != 1)        return false;
    if (src0->ne[0] != ne10) return false;
    if (ne10 % 256 != 0)  return false; // QK_K

    const int ith = params->ith, nth = params->nth;
    const int64_t n_ids = ids->ne[0];   // n_expert_used
    const int     n_as  = (int) ne02;   // n_expert

    // Carve wdata in v6's layout: activation region (sized for vec_dot_type=Q8_K, which is
    // >= our Q8_2_X4), then matrix_row_counts[n_as], then matrix_rows[n_as*n_ids*ids->ne[1]].
    const size_t act_row = (size_t)(ne10 / 32) * 36; // sizeof(block_q8_2) per 32 elems
    char * base = (char *) params->wdata;
    char * qact = base;
    char * wc   = base + GGML_PAD(ggml_row_size(GGML_TYPE_Q8_K, ggml_nelements(src1)), sizeof(int64_t));
    int64_t * matrix_row_counts = (int64_t *) wc;
    wc += GGML_PAD(n_as * sizeof(int64_t), sizeof(int64_t));
    iqk_mmid * matrix_rows = (iqk_mmid *) wc;
    wc += (size_t) n_as * n_ids * ids->ne[1] * sizeof(iqk_mmid);
    if (params->wsize < (size_t)(wc - base)) return false; // safety (should always fit: Q8_2_X4 <= Q8_K)

    // 1) quantize src1 F32 -> Q8_2_X4, row layout [i12*ne11 + i11]
    const size_t nbw1 = act_row, nbw2 = nbw1 * ne11;
    for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
            quantize_row_q8_2_x4((const float *)((const char *) src1->data + i12*src1->nb[2] + i11*src1->nb[1]),
                                 qact + i12*nbw2 + i11*nbw1, ne10);
        }
    }
    // 2) build per-expert row mapping (ith==0), v6 stride = n_ids*ids->ne[1]
    if (ith == 0) {
        memset(matrix_row_counts, 0, n_as * sizeof(int64_t));
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int64_t id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *)((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                matrix_rows[(size_t) i02 * n_ids * ids->ne[1] + matrix_row_counts[i02]] =
                    iqk_mmid{ (int32_t) id, (int32_t) iid1 };
                matrix_row_counts[i02] += 1;
            }
        }
    }
    ggml_barrier(params->threadpool);

    // 3) per-expert GEMM via iqk
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;
        const char * A = (const char *) src0->data + (size_t) cur_a * src0->nb[2];
        const iqk_mmid * rmap = matrix_rows + (size_t) cur_a * n_ids * ids->ne[1];
        if (!iqk_mul_mat_moe(ne01, cne1, ne10, (int) ne11,
                tA, A, src0->nb[1],
                GGML_TYPE_Q8_2_X4, qact, act_row,
                (float *) dst->data, dst->nb[1], dst->nb[2],
                rmap, ith, nth)) {
            return false; // gating should preclude; native re-runs from scratch on false
        }
    }
    if (ith == 0) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=%d n_as=%d)\n", tA, n_as); }
    }
    return true;
}

#else  // iqk not implemented / disabled — no-op hook

extern "C" bool ggml_iqk_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    (void) params; (void) dst;
    return false;
}

extern "C" bool ggml_iqk_try_mul_mat_id(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    (void) params; (void) dst;
    return false;
}

#endif
