//
// iqk port (Stage 1): dispatch hook. Routes dense GGML_OP_MUL_MAT for the quant
// families iqk really implements to ik's iqk_mul_mat_4d. Activations use the
// format required by each weight family: Q8_2_X4 for Q4_K/Q5_K/Q6_K and legacy
// quants, Q8_K for IQ quants. Q2_K/Q3_K deliberately fall through before
// activation quantization. Runtime gate: env GGML_IQK=1.
//
#include "iqk_config.h"
#include "../ggml-cpu-knobs.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "quants.h"

#if defined(IQK_IMPLEMENT) && defined(GGML_USE_IQK_MULMAT)

#include "iqk_mul_mat.h"
#include <atomic>
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
// INF-70 B3-k: flat (expert, row) slab partition of the single-token mul_mat_id — on by
// default; GGML_MMID_SLAB=0 restores D1's per-expert 1/nth row stripes (same-binary A/B).
inline bool iqk_mmid_slab_enabled() {
    static const bool e = []() { const char * s = getenv("GGML_MMID_SLAB"); return s == nullptr || atoi(s) != 0; }();
    return e;
}
inline bool iqk_q8_0_enabled() {
    static const bool e = []() {
        const char * s = getenv("GGML_IQK_Q8_0");
        return s && atoi(s) != 0;
    }();
    return e;
}

// INF-70 SYNC-15: GGML_QSPLIT / GGML_QSPLIT_MIN — the activation-quantization split.
//
// D1's batch-1 fast path has EVERY thread quantize the WHOLE activation row into its own
// private wdata slice so the publishing barrier can be dropped.  That is a good trade only
// while the row is cheap to quantize: the cost is paid nth times over (once on each thread,
// all on the critical path) to save one ~3.1 us barrier.  For ne00 = 10240 with a Q8_K
// activation the whole-row quantization measures ~15 us, so the fast path pays ~15 us where
// a block-split + barrier would pay ~15/nth + 3.1 us.
//
// GGML_QSPLIT=1 turns the lever on (default OFF -> D1's behaviour, exactly).  GGML_QSPLIT_MIN
// is the smallest ne00 (in elements) at which a SINGLE activation row is split across the team
// at block granularity and published with one barrier; with more than one row the stock path
// already pays that barrier, so the split is unconditional there.
//
// Bit-identity: both Q8_K and Q8_2_X4 quantize block-locally (256- and 32-element blocks,
// the latter packed in groups of 4).  Segments start at multiples of 256 / 128 elements and
// the final segment carries any remainder, so every block is produced by the identical
// computation over the identical inputs and lands at the identical byte offset.
// INF-70 HARNESS-1: per-graph snapshot instead of a first-use latch (see ggml-cpu-knobs.h).
inline bool iqk_qsplit_enabled() {
    return ggml_cpu_knobs_cur.qsplit != 0;
}
// Only consulted when there is exactly ONE activation row, i.e. when the split has to BUY the
// barrier that D1's private-row path avoids.  With more than one row the stock path already
// pays that barrier, so the split is free there and the threshold does not apply.
inline int64_t iqk_qsplit_min() {
    return ggml_cpu_knobs_cur.qsplit_min;
}

#ifdef GGML_CPU_PROF
// INF-70 SYNC-15 instrumentation (profiling builds only, and only with GGML_IQK_PROF=1):
// separate the activation-quantization phase from the GEMM phase, bucketed by whether the
// node is "skinny" (fewer output rows than threads).  Thread 0 only, so no shared traffic.
std::atomic<uint64_t> iqk_prof_quant_ns{0}, iqk_prof_gemm_ns{0}, iqk_prof_n{0};
std::atomic<uint64_t> iqk_prof_sk_quant_ns{0}, iqk_prof_sk_gemm_ns{0}, iqk_prof_sk_n{0};
inline bool iqk_prof_enabled() {
    static const bool e = []() { const char * s = getenv("GGML_IQK_PROF"); return s && atoi(s) != 0; }();
    return e;
}
struct IqkProfDump {
    ~IqkProfDump() {
        if (!iqk_prof_enabled()) return;
        fprintf(stderr, "[iqk_prof] mul_mat calls(thread0)=%llu  quant=%.3f ms  gemm=%.3f ms\n",
                (unsigned long long) iqk_prof_n.load(),
                iqk_prof_quant_ns.load()/1e3, iqk_prof_gemm_ns.load()/1e3);
        fprintf(stderr, "[iqk_prof]   of which SKINNY (ne01 < nth): calls=%llu  quant=%.3f ms  gemm=%.3f ms\n",
                (unsigned long long) iqk_prof_sk_n.load(),
                iqk_prof_sk_quant_ns.load()/1e3, iqk_prof_sk_gemm_ns.load()/1e3);
    }
};
IqkProfDump iqk_prof_dump_;
#endif
constexpr bool iqk_typeA_supported(int t) {
    switch (t) {
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_1:   // note: Q6_0 is ik-only, not in v6
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS:
            return true;
        default: return false;
    }
}

// Keep rejected types out of both dense and MoE hooks: eligibility is checked
// before iqk_activation_type() and iqk_quantize_activation() in each path.
static_assert(!iqk_typeA_supported(GGML_TYPE_Q2_K));
static_assert(!iqk_typeA_supported(GGML_TYPE_Q3_K));
static_assert(iqk_typeA_supported(GGML_TYPE_IQ4_XS));

constexpr bool iqk_weight_uses_q8_k(int t) {
    // Q8_K is enabled only for the validated IQ families. Q2_K/Q3_K are
    // rejected by iqk_typeA_supported() before this helper is reached.
    switch (t) {
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}
static_assert(!iqk_weight_uses_q8_k(GGML_TYPE_Q2_K));
static_assert(!iqk_weight_uses_q8_k(GGML_TYPE_Q3_K));
static_assert(iqk_weight_uses_q8_k(GGML_TYPE_IQ2_XXS));
static_assert(iqk_weight_uses_q8_k(GGML_TYPE_IQ3_XXS));
static_assert(iqk_weight_uses_q8_k(GGML_TYPE_IQ4_XS));

constexpr bool iqk_shape_supported(int weight_type, int64_t n_rows) {
    // The imported IQ3_XXS kernel exceeds the backend NMSE limit for some tiny
    // output matrices. Production model matrices are much larger; retain the
    // native fallback for narrow utility/test shapes.
    //
    // Q4_K, Q4_0, and Q6_K use the same Q8_2_X4 activation path.  Deterministic
    // AutoKernel holdouts showed that their approximate fast paths can exceed
    // the independent host-double error-ratio bound for 16-row utility/test
    // shapes even when the generic comparison's broad tolerance accepts them.
    // Do not make correctness-sensitive narrow shapes depend on that path:
    // fall through to the native CPU kernel.  The 32-row cutoff retains the
    // intended model-matrix dispatch while making the boundary deterministic.
    if (weight_type == GGML_TYPE_IQ3_XXS || weight_type == GGML_TYPE_Q4_K ||
            weight_type == GGML_TYPE_Q4_0 || weight_type == GGML_TYPE_Q6_K) {
        return n_rows >= 32;
    }
    return true;
}

static_assert(!iqk_shape_supported(GGML_TYPE_Q4_0, 16));
static_assert(iqk_shape_supported(GGML_TYPE_Q4_0, 32));
static_assert(!iqk_shape_supported(GGML_TYPE_Q4_K, 16));
static_assert(iqk_shape_supported(GGML_TYPE_Q4_K, 32));
static_assert(!iqk_shape_supported(GGML_TYPE_Q6_K, 16));
static_assert(iqk_shape_supported(GGML_TYPE_Q6_K, 32));

constexpr bool iqk_mmid_shape_supported(int weight_type, int64_t n_tokens) {
    // IQ3_XXS MoE is seed-sensitive near the independent NMSE boundary when
    // more than one token is routed in a call.  The deterministic holdout found
    // n=15 above the bound and n=16 only 2.7% below it, while all measured n=1
    // decode cases were effectively exact.  Preserve that proven decode path;
    // route wider IQ3_XXS MMID calls through the native CPU implementation.
    return weight_type != GGML_TYPE_IQ3_XXS || n_tokens == 1;
}

static_assert(iqk_mmid_shape_supported(GGML_TYPE_IQ3_XXS, 1));
static_assert(!iqk_mmid_shape_supported(GGML_TYPE_IQ3_XXS, 15));
static_assert(!iqk_mmid_shape_supported(GGML_TYPE_IQ3_XXS, 16));
static_assert(iqk_mmid_shape_supported(GGML_TYPE_IQ3_S, 16));

inline int iqk_activation_type(int weight_type) {
    return iqk_weight_uses_q8_k(weight_type) ? GGML_TYPE_Q8_K : GGML_TYPE_Q8_2_X4;
}

inline size_t iqk_activation_row_size(int activation_type, int64_t n) {
    return activation_type == GGML_TYPE_Q8_K
        ? ggml_row_size(GGML_TYPE_Q8_K, n)
        : (size_t) (n / 32) * 36;
}

inline void iqk_quantize_activation(int activation_type, const float * src, void * dst, int64_t n) {
    if (activation_type == GGML_TYPE_Q8_K) {
        quantize_row_q8_K(src, dst, n);
    } else {
        quantize_row_q8_2_x4(src, dst, n);
    }
}

inline bool iqk_first_engagement(std::atomic<uint64_t> & logged_types, int weight_type) {
    GGML_ASSERT(weight_type >= 0 && weight_type < 64);
    const uint64_t bit = UINT64_C(1) << weight_type;
    return (logged_types.fetch_or(bit, std::memory_order_relaxed) & bit) == 0;
}
} // namespace

extern "C" bool ggml_iqk_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!iqk_enabled()) return false;
    if (params->use_ref) return false;

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (dst->type  != GGML_TYPE_F32) return false;
    if (src1->type != GGML_TYPE_F32) return false;          // Stage 1: F32 activations only
    if (!iqk_typeA_supported((int) src0->type)) return false;
    if (!iqk_shape_supported((int) src0->type, src0->ne[1])) return false;
    if (src0->type == GGML_TYPE_Q8_0 && !iqk_q8_0_enabled()) return false;

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    if (ne00 != ne10)   return false;
    const int activation_type = iqk_activation_type((int) src0->type);
    if (activation_type == GGML_TYPE_Q8_K ? ne00 % QK_K != 0 : ne00 % 32 != 0) return false;

    // ggml guarantees ne0==ne01, ne1==ne11, ne2==ne12, ne3==ne13 and unpermuted src0/src1
    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    const int ith = params->ith, nth = params->nth;

    // Quantize src1 into the activation format required by this weight family.
    const size_t row_size = iqk_activation_row_size(activation_type, ne10);
    const size_t nbw1 = row_size, nbw2 = nbw1 * ne11, nbw3 = nbw2 * ne12;
    if (params->wsize < (size_t) ne13 * nbw3) return false;

    char * wdata = (char *) params->wdata;

    // INF-70 D1: batch-1 fast path — at a single activation row the stock code has thread 0
    // quantize it while nth-1 threads wait at a full-team barrier. Instead every thread
    // quantizes the row into its own private slice and reads its own copy; iqk_mul_mat_4d
    // partitions only over src0 rows / tiles and reads all of B, so the result is identical
    // and no barrier is needed. wdata is sized nth * row bytes by ggml_graph_plan.
    const int64_t rows_total = ne11 * ne12 * ne13;

    // INF-70 SYNC-15: activation-quantization grain split.  See iqk_qsplit_min() above.
    // Both the D1 private-row fast path (rows_total == 1) and the stock row-parallel path
    // (i11 += nth) leave the quantization of a wide activation row on far fewer threads than
    // the team has: the fast path replicates the WHOLE row on all nth threads, and the stock
    // path hands one whole row to each of rows_total threads while nth - rows_total idle.
    // When there are fewer activation rows than threads, split the (row, block) space instead.
    // Default OFF (GGML_QSPLIT unset).
    const int64_t grain      = (activation_type == GGML_TYPE_Q8_K) ? QK_K : 128;
    const int64_t ngrain_row = (ne10 + grain - 1) / grain;
    const bool iqk_qsplit =
        iqk_qsplit_enabled() && nth > 1 && rows_total < nth &&
        (rows_total > 1 || ne10 >= iqk_qsplit_min());

    const bool iqk_batch1 =
        rows_total == 1 && !iqk_qsplit &&
        params->wsize >= (size_t) nth * nbw3;

#ifdef GGML_CPU_PROF
    const bool  prof   = iqk_prof_enabled() && ith == 0;
    const int64_t p_t0 = prof ? ggml_time_us() : 0;
#endif

    if (iqk_batch1) {
        wdata += (size_t) ith * nbw3;
        iqk_quantize_activation(activation_type, (const float *) src1->data, wdata, ne10);
    } else if (iqk_qsplit) {
        // Grain: Q8_K quantizes 256 elements per block; Q8_2_X4 quantizes 32 and packs them in
        // groups of 4, so a segment must start on a 128-element boundary for the packed layout
        // to be byte-identical to the whole-row call.  Every block is therefore produced by the
        // identical computation over the identical inputs and lands at the identical offset.
        const int64_t units = rows_total * ngrain_row;
        const int64_t u0 = ((int64_t) ith * units) / nth;
        const int64_t u1 = ((int64_t) (ith + 1) * units) / nth;
        for (int64_t u = u0; u < u1; ) {
            const int64_t r  = u / ngrain_row;
            const int64_t gs = u - r * ngrain_row;
            int64_t ge = u1 - r * ngrain_row;
            if (ge > ngrain_row) ge = ngrain_row;
            const int64_t e0 = gs * grain;
            const int64_t e1 = (ge * grain < ne10) ? ge * grain : ne10;

            const int64_t i13 = r / (ne11 * ne12);
            const int64_t rem = r - i13 * ne11 * ne12;
            const int64_t i12 = rem / ne11;
            const int64_t i11 = rem - i12 * ne11;

            iqk_quantize_activation(activation_type,
                    (const float *)((const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11) + e0,
                    wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + iqk_activation_row_size(activation_type, e0),
                    e1 - e0);

            u = r * ngrain_row + ge;
        }
        ggml_barrier(params->threadpool);
    } else {
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    iqk_quantize_activation(
                        activation_type,
                        (const float *)((const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                        wdata + i13*nbw3 + i12*nbw2 + i11*nbw1, ne10);
                }
            }
        }
        ggml_barrier(params->threadpool);
    }

#ifdef GGML_CPU_PROF
    const int64_t p_t1 = prof ? ggml_time_us() : 0;
#endif

    const bool ok = iqk_mul_mat_4d(ne01, ne11, ne00,
            ne02, ne03, ne12, ne13,
            nb02, nb03, nbw2, nbw3,
            nb2 / sizeof(float), nb3 / sizeof(float),
            (int) src0->type, src0->data, nb01,
            activation_type, wdata, row_size,
            (float *) dst->data, nb1 / sizeof(float), ith, nth);
#ifdef GGML_CPU_PROF
    if (prof) {
        const int64_t p_t2 = ggml_time_us();
        iqk_prof_quant_ns.fetch_add((uint64_t)(p_t1 - p_t0), std::memory_order_relaxed);
        iqk_prof_gemm_ns.fetch_add((uint64_t)(p_t2 - p_t1), std::memory_order_relaxed);
        iqk_prof_n.fetch_add(1, std::memory_order_relaxed);
        if (ne01 < nth) {
            iqk_prof_sk_quant_ns.fetch_add((uint64_t)(p_t1 - p_t0), std::memory_order_relaxed);
            iqk_prof_sk_gemm_ns.fetch_add((uint64_t)(p_t2 - p_t1), std::memory_order_relaxed);
            iqk_prof_sk_n.fetch_add(1, std::memory_order_relaxed);
        }
    }
#endif
    if (ok && ith == 0) {
        static std::atomic<uint64_t> logged_types{0};
        if (iqk_first_engagement(logged_types, (int) src0->type)) {
            fprintf(stderr, "[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=%d activation=%d ne00=%lld)\n",
                    (int) src0->type, activation_type, (long long) ne00);
        }
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
    if (params->use_ref) return false;
    const struct ggml_tensor * src0 = dst->src[0]; // expert weights [ne00, ne01, n_expert]
    const struct ggml_tensor * src1 = dst->src[1]; // F32 activations
    const struct ggml_tensor * ids  = dst->src[2];
    if (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) return false;
    const int tA = (int) src0->type;
    // Same families as the dense hook. iqk_mul_mat_moe returns false for any it
    // cannot handle, in which case the native path reruns the operation.
    if (!iqk_typeA_supported(tA)) return false;
    if (!iqk_shape_supported(tA, src0->ne[1])) return false;
    if (!iqk_mmid_shape_supported(tA, ids->ne[1])) return false;
    if (tA == GGML_TYPE_Q8_0 && !iqk_q8_0_enabled()) return false;

    const int64_t ne01 = src0->ne[1], ne02 = src0->ne[2];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    if (ne13 != 1)        return false;
    if (src0->ne[0] != ne10) return false;
    const int activation_type = iqk_activation_type(tA);
    if (activation_type == GGML_TYPE_Q8_K ? ne10 % QK_K != 0 : ne10 % 32 != 0) return false;

    const int ith = params->ith, nth = params->nth;
    const int64_t n_ids = ids->ne[0];   // n_expert_used
    const int     n_as  = (int) ne02;   // n_expert

    // Carve wdata in the native layout: a Q8_K-sized activation region followed
    // by matrix_row_counts[n_as] and matrix_rows[n_as*n_ids*ids->ne[1]].
    const size_t act_row = iqk_activation_row_size(activation_type, ne10);
    char * base = (char *) params->wdata;
    char * qact = base;
    char * wc   = base + GGML_PAD(ggml_row_size(GGML_TYPE_Q8_K, ggml_nelements(src1)), sizeof(int64_t));
    int64_t * matrix_row_counts = (int64_t *) wc;
    wc += GGML_PAD(n_as * sizeof(int64_t), sizeof(int64_t));
    iqk_mmid * matrix_rows = (iqk_mmid *) wc;
    wc += (size_t) n_as * n_ids * ids->ne[1] * sizeof(iqk_mmid);
    if (params->wsize < (size_t)(wc - base)) return false; // safety (should always fit: Q8_2_X4 <= Q8_K)

    // 1) quantize src1, row layout [i12*ne11 + i11]
    const size_t nbw1 = act_row, nbw2 = nbw1 * ne11;

    // INF-70 D1: single-token fast path — no internal barrier. Every thread quantizes the
    // whole activation block into its own private slice, builds the trivial row->expert map
    // on the stack (experts ascending, ids ascending within an expert — exactly the shared
    // pass's order) and calls iqk_mul_mat_moe only for the used experts, skipping the
    // n_as-entry scan. iqk_mul_mat_moe splits only over src0 rows and reads all of B, so
    // per-thread copies of B are bit-identical to the shared one.
    constexpr int IQK_MMID_B1_MAX_IDS = 64;

    const size_t b1_region = nbw2 * ne12;

    const bool mmid_batch1 =
        ids->ne[1] == 1 && n_ids > 0 && n_ids <= IQK_MMID_B1_MAX_IDS &&
        params->wsize >= (size_t) nth * b1_region;

    if (mmid_batch1) {
        char * const qact_priv = base + (size_t) ith * b1_region;

        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                iqk_quantize_activation(
                        activation_type,
                        (const float *)((const char *) src1->data + i12*src1->nb[2] + i11*src1->nb[1]),
                        qact_priv + i12*nbw2 + i11*nbw1, ne10);
            }
        }

        int32_t    b1_expert[IQK_MMID_B1_MAX_IDS];
        iqk_mmid   b1_row   [IQK_MMID_B1_MAX_IDS];
        int        b1_n = 0;

        for (int64_t id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *)((const char *) ids->data + id*ids->nb[0]);
            if (i02 < 0 || i02 >= n_as) {
                // inactive SER route: nobody else writes this dst row
                if (ith == 0) {
                    memset((char *) dst->data + id*dst->nb[1], 0, dst->ne[0]*sizeof(float));
                }
                continue;
            }
            int pos = b1_n;
            while (pos > 0 && b1_expert[pos-1] > i02) {
                b1_expert[pos] = b1_expert[pos-1];
                b1_row   [pos] = b1_row   [pos-1];
                --pos;
            }
            b1_expert[pos] = i02;
            b1_row   [pos] = iqk_mmid{ (int32_t) id, 0 };
            ++b1_n;
        }

        bool engaged_b1 = false;

        // ------------------------------------------------------------------------------
        // INF-70 B3-k: flat slab partition. iqk_mul_mat_moe splits EVERY expert 1/nth by
        // rows, so at 48 threads each thread streams ten 14–54-row stripes (19–26 KB each)
        // per op — B2 measured that split at ~11 ms/token of overhead beyond the bytes.
        // Instead treat the used experts' rows as ONE flat space of n_groups*ne01 rows
        // (experts ascending — exactly the order the loop below iterates) and give thread
        // ith the contiguous range [total*ith/nth, total*(ith+1)/nth): ~133 rows (181 KB of
        // IQ4_XS up/gate) or ~533 rows (256 KB of Q5_1 down) of at most two adjacent expert
        // slabs, balanced to ±1 row. Bit-identical: iqk_mul_mat_moe_rows runs the same
        // kernel over the same rows and every kernel it can reach accumulates each row
        // independently; dst rows stay disjoint per (id, row). Taken only for distinct
        // experts (cne1 == 1 everywhere — the top-k decode case) on the direct-kernel path;
        // anything else keeps the per-expert loop.
        // ------------------------------------------------------------------------------
        int  grp_start[IQK_MMID_B1_MAX_IDS];
        int  n_groups = 0;
        bool slab_ok  = iqk_mmid_slab_enabled() && b1_n > 0;
        for (int i = 0; i < b1_n; ) {
            int j = i;
            while (j < b1_n && b1_expert[j] == b1_expert[i]) ++j;
            if (j - i != 1) slab_ok = false;
            grp_start[n_groups++] = i;
            i = j;
        }
        const int64_t slab_gran = slab_ok ? iqk_mul_mat_moe_row_granularity(tA) : 1;
        if (slab_ok && (slab_gran <= 0 || ne01 % slab_gran != 0 || iqk_dequant_type(tA, 1) != tA)) {
            slab_ok = false;
        }

        if (slab_ok) {
            const int64_t units_per_group = ne01 / slab_gran;
            const int64_t units_total     = units_per_group * n_groups;
            const int64_t u0 = (units_total * ith) / nth;
            const int64_t u1 = (units_total * (ith + 1)) / nth;

            // every thread makes at least one call (possibly empty) so a kernel-selection
            // failure is observed by all threads alike (native re-runs from scratch on false)
            int64_t u = u0;
            do {
                const int     g  = (int) (u / units_per_group);
                const int64_t g0 = (int64_t) g * units_per_group;
                const int64_t ue = u1 < g0 + units_per_group ? u1 : g0 + units_per_group;
                const char *  A  = (const char *) src0->data + (size_t) b1_expert[grp_start[g]] * src0->nb[2];
                if (!iqk_mul_mat_moe_rows(ne01, /*Ny =*/ 1, ne10, (int) ne11,
                        tA, A, src0->nb[1],
                        activation_type, qact_priv, act_row,
                        (float *) dst->data, dst->nb[1], dst->nb[2],
                        b1_row + grp_start[g], (u - g0) * slab_gran, (ue - u) * slab_gran)) {
                    return false; // gating should preclude; native re-runs from scratch on false
                }
                u = ue;
            } while (u < u1);
            engaged_b1 = true;
        } else {
            for (int i = 0; i < b1_n; ) {
                int j = i;
                while (j < b1_n && b1_expert[j] == b1_expert[i]) ++j;
                const int64_t cne1 = j - i;
                const char * A = (const char *) src0->data + (size_t) b1_expert[i] * src0->nb[2];
                engaged_b1 = true;
                if (!iqk_mul_mat_moe(ne01, cne1, ne10, (int) ne11,
                        tA, A, src0->nb[1],
                        activation_type, qact_priv, act_row,
                        (float *) dst->data, dst->nb[1], dst->nb[2],
                        b1_row + i, ith, nth)) {
                    return false; // gating should preclude; native re-runs from scratch on false
                }
                i = j;
            }
        }
        if (engaged_b1 && ith == 0) {
            static std::atomic<uint64_t> logged_types_b1{0};
            if (iqk_first_engagement(logged_types_b1, tA)) {
                fprintf(stderr, "[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=%d activation=%d n_as=%d)\n",
                        tA, activation_type, n_as);
            }
        }
        return true;
    }

    for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
            iqk_quantize_activation(
                    activation_type,
                    (const float *)((const char *) src1->data + i12*src1->nb[2] + i11*src1->nb[1]),
                    qact + i12*nbw2 + i11*nbw1, ne10);
        }
    }
    // 2) Zero inactive SER rows and build the valid per-expert row mapping.
    for (int64_t iid1 = ith; iid1 < ids->ne[1]; iid1 += nth) {
        for (int64_t id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *)((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
            if (i02 < 0 || i02 >= n_as) {
                memset((char *) dst->data + id*dst->nb[1] + iid1*dst->nb[2], 0, dst->ne[0]*sizeof(float));
            }
        }
    }

    // v6 stride = n_ids*ids->ne[1]
    if (ith == 0) {
        memset(matrix_row_counts, 0, n_as * sizeof(int64_t));
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int64_t id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *)((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                if (i02 < 0 || i02 >= n_as) {
                    continue;
                }
                matrix_rows[(size_t) i02 * n_ids * ids->ne[1] + matrix_row_counts[i02]] =
                    iqk_mmid{ (int32_t) id, (int32_t) iid1 };
                matrix_row_counts[i02] += 1;
            }
        }
    }
    ggml_barrier(params->threadpool);

    // 3) per-expert GEMM via iqk
    bool engaged = false;
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;
        engaged = true;
        const char * A = (const char *) src0->data + (size_t) cur_a * src0->nb[2];
        const iqk_mmid * rmap = matrix_rows + (size_t) cur_a * n_ids * ids->ne[1];
        if (!iqk_mul_mat_moe(ne01, cne1, ne10, (int) ne11,
                tA, A, src0->nb[1],
                activation_type, qact, act_row,
                (float *) dst->data, dst->nb[1], dst->nb[2],
                rmap, ith, nth)) {
            return false; // gating should preclude; native re-runs from scratch on false
        }
    }
    if (engaged && ith == 0) {
        static std::atomic<uint64_t> logged_types{0};
        if (iqk_first_engagement(logged_types, tA)) {
            fprintf(stderr, "[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=%d activation=%d n_as=%d)\n",
                    tA, activation_type, n_as);
        }
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
