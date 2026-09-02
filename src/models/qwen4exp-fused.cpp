// Fused batch-1 decoder for qwen4exp (CPU). The layer's math runs as one
// sequential function with no ggml node machinery between the micro-ops; every
// projection is the same vec_dot the graph's mul_mat uses (same quantized
// activation, same per-row accumulation), every elementwise op the same vec
// function, so the numerics mirror the decomposed graph bit for bit.
//
// Invoked only for the single-token decode ubatch on the CPU backend, behind
// model.supports_fused_decode() (see llama-context.cpp process_ubatch).

#include "ggml.h"
#include "ggml-cpu.h"

extern "C" {
void ggml_compute_forward_ssm_conv(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_gated_delta_net(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_vec_silu_f32(const int n, float * y, const float * x);
void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
// ggml-cpu/traits.h is not on the llama include path; these are the CPU backend's
// extra-buffer-type hooks that ggml_compute_forward()/ggml_graph_plan() call before
// the built-in kernel (they are what claims a CPU_REPACK / AMX weight).
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);
bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size);
}

// ggml_compute_params lives in ggml-cpu-impl.h (not on the llama include path);
// define the exact layout here (it is part of the kernel ABI).
struct ggml_compute_params {
    int ith, nth;
    size_t wsize;
    void * wdata;
    struct ggml_threadpool * threadpool;
    bool use_ref;
};

#include "llama-impl.h"
#include "llama-fused-debug.h"
#include "llama-model.h"
#include "llama-memory-hybrid.h"
#include "models/models.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

// the feasibility profiler (GGML_FUSED_PROF; a non-claim instrument: the gemv
// (vec_dot) time vs the total, per token)
namespace {
struct FusedProf {
    double gemv_ms = 0.0;
    bool on = false;
    int depth = 0;
    std::chrono::steady_clock::time_point t;
    void start() { if (on) { if (depth == 0) t = std::chrono::steady_clock::now(); depth++; } }
    void stop() { if (on && depth > 0) { depth--; if (depth == 0) gemv_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); } }
    void reset() { gemv_ms = 0.0; depth = 0; }
};
static FusedProf g_prof;
}
// ---- A2: the scratch arena --------------------------------------------------
//
// Every ggml_init() in this file asked for a fresh 16-64 MB block and freed it
// again: one per token for the cache views, one per GDN layer (x36), one per
// full-attention layer for the flash staging (x12), and one per fused_rope call
// (2 per attention layer in the dense-QSA case, one per pooled block otherwise).
// glibc serves a request that size with mmap, so each pair is an mmap + munmap
// plus a page fault for every page the context actually touches — and the GDN
// context touches 3.1 MB of them (the [S_v, S_v, H_v] recurrent state tensor).
//
// The arena keeps one buffer per nesting slot, hands it to ggml_init() as
// `mem_buffer` (ggml then neither allocates nor frees it; only the small
// ggml_context header is malloc'd), and grows it on demand. The slots are
// distinct because the contexts nest: the token-scoped cache-view context is
// live while a layer context is, and a rope context is live inside the
// attention layer.
//
// The counters are the A2 measurement: they record what the churn WAS, per
// token, so the report carries a number rather than the retired "~2.5 GB"
// code-reading estimate. They are printed by the GGML_FUSED_PROF line.
namespace {

enum fused_arena_slot {
    FUSED_ARENA_VIEWS = 0,   // token-scoped: the KV / indexer cache views
    FUSED_ARENA_LAYER = 1,   // layer-scoped: the GDN scan tensors
    FUSED_ARENA_ATTN  = 2,   // the flash-attention staging
    FUSED_ARENA_ROPE  = 3,   // the rope staging (nested inside ATTN's caller)
    FUSED_ARENA_N     = 4,
};

struct FusedArenaStats {
    uint64_t ctx_calls  = 0;  // ggml_init calls this token
    uint64_t ctx_bytes  = 0;  // bytes those calls need this token
    uint64_t churn_was  = 0;  // bytes the SAME calls asked for before A2 (the mmap churn)
    uint64_t stage_bytes = 0; // the staged-state bytes this token (was a malloc per layer)
    uint64_t arena_bytes = 0; // bytes the arenas actually hold (allocated once)
    void reset() { ctx_calls = 0; ctx_bytes = 0; churn_was = 0; stage_bytes = 0; }
};
static FusedArenaStats g_arena;

// the A2 kill switch: `GGML_FUSED_ARENA_OFF=1` restores the per-call
// ggml_init(mem_buffer = NULL) at its original request size, so the arena is
// measurable as a same-build A/B. Read once per token, never in a loop.
static bool g_arena_off = false;

static struct ggml_context * fused_arena_init(int slot, size_t need, size_t was) {
    static std::vector<uint8_t> arenas[FUSED_ARENA_N];
    if (g_arena_off) {
        g_arena.ctx_calls++;
        g_arena.ctx_bytes += need;
        g_arena.churn_was += was;
        struct ggml_init_params ip = { was, nullptr, false };
        return ggml_init(ip);
    }
    std::vector<uint8_t> & a = arenas[slot];
    const size_t need_raw = need;
    // Slack. An arena one object short does not degrade — ggml_new_tensor calls
    // GGML_ABORT("not enough space in the context's memory pool"), i.e. it takes
    // the process down. 25% + 1 MB is cheap insurance against a shape this
    // sizing did not anticipate; the request itself is still what gets counted.
    need += need / 4 + (1u << 20);
    // + 64 so the buffer can be aligned up without losing capacity
    if (a.size() < need + 64) {
        a.resize(need + 64);
        g_arena.arena_bytes = 0;
        for (int i = 0; i < FUSED_ARENA_N; i++) g_arena.arena_bytes += arenas[i].size();
    }
    uint8_t * base = a.data();
    uint8_t * aligned = (uint8_t *) ((((uintptr_t) base) + 63) & ~(uintptr_t) 63);
    const size_t usable = a.size() - (size_t) (aligned - base);
    g_arena.ctx_calls++;
    g_arena.ctx_bytes += need_raw;
    g_arena.churn_was += was;
    struct ggml_init_params ip = { usable, aligned, false };
    return ggml_init(ip);
}

} // namespace

#define FUSED_PROF_INIT() do { if (getenv("GGML_FUSED_PROF") != NULL) { g_prof.on = true; } } while (0)
#define FUSED_PROF_RESET() do { g_prof.reset(); } while (0)
#define FUSED_PROF_DOT_BEGIN() g_prof.start()
#define FUSED_PROF_DOT_END() g_prof.stop()

using ggml_type_traits_cpu = struct ggml_type_traits_cpu;

// ---- helpers mirroring the graph's batch-1 kernels -------------------------

// Read a small dense weight tensor into F32 regardless of its stored type.
// The fused path dereferences several of these directly as `(const float *)
// t->data`, which is only correct when the quantizer left them F32. On the
// uniform IQ4_XS artifact blk.N.ple_conv1d is **F16** [4, 10240], so the PLE
// depthwise conv was reading 160 KB of F16 pairs as 40,960 floats — running
// 81 KB past the tensor into whatever followed it in the weight buffer. (In
// bounds of the model buffer, hence no crash: a silent wrong result.)
static const float * fused_weights_f32(const struct ggml_tensor * t, std::vector<float> & tmp) {
    if (t->type == GGML_TYPE_F32) {
        return (const float *) t->data;
    }
    const int64_t n = ggml_nelements(t);
    tmp.resize((size_t) n);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * h = (const ggml_fp16_t *) t->data;
        for (int64_t i = 0; i < n; i++) tmp[i] = ggml_fp16_to_fp32(h[i]);
    } else {
        ggml_get_type_traits(t->type)->to_float(t->data, tmp.data(), n);
    }
    return tmp.data();
}

// ---- A1: the batched mul_mat ----------------------------------------------
//
// Every projection in the fused path used to be a per-row `vec_dot` loop. That
// is the 2026-08-28 HC_MIX anti-pattern (785 us unbatched vs 150 us batched):
// the graph's mul_mat is not a vec_dot loop, it is a dispatch — the CPU
// backend's extra buffer types (the CPU_REPACK 4x4/8x8 gemv), the iqk GEMM
// under GGML_IQK=1, llamafile's tinyBLAS, and only then the chunked
// vec_dot fallback. A per-row call reaches none of them.
//
// fused_mm() stages a real GGML_OP_MUL_MAT on borrowed tensor headers and runs
// the same two-step dispatch ggml_compute_forward() uses:
//     ggml_cpu_extra_compute_forward()  ->  ggml_compute_forward_mul_mat()
// src0 is a shallow copy of the weight header, so `buffer`, `extra`, `type` and
// the row stride are the real ones — which is exactly what the repack traits key
// on (`op->src[0]->buffer->buft == repack_buft` then `op->src[0]->extra`).
//
// Staging details the earlier attempt got wrong (it segfaulted here):
//   - wdata. The kernel writes the quantized activation into params->wdata and
//     asserts nothing about its size; iqk carves its own Q8_K-sized region and
//     the repack traits have their own work_size. Sized below by asking
//     ggml_cpu_extra_work_size() first (what ggml_graph_plan does) and taking
//     the max with the built-in MUL_MAT formula and a Q8_K-shaped upper bound.
//   - threadpool. mul_mat does an unconditional
//     atomic_store(&threadpool->current_chunk) and ggml_barrier(threadpool), so
//     the pointer must be valid even at nth=1. A persistent 1-thread pool, as
//     the GDN and flash staging already do.
//   - strides. dst must satisfy nb0 == sizeof(float) and nb0<=nb1<=nb2<=nb3;
//     src1 must be contiguous F32; src0 must have nb00 == type_size and its
//     ne[2]/ne[3] collapsed to 1 (an expert slab is sliced by moving `data`).
//
// NOTE (single-thread contract): at nth == 1 ggml_barrier() takes the
// `#pragma omp barrier` branch, because a threadpool created by
// ggml_threadpool_new() has n_graph == 0 rather than 1. Outside a parallel
// region that barrier is a no-op, so this is correct in an OpenMP build — which
// every build directory in this tree is. A GGML_OPENMP=OFF build would spin on
// the atomic barrier instead; the fused path is single-threaded by construction
// today, so that is recorded here rather than worked around.
namespace {

static struct ggml_threadpool * fused_mm_tp() {
    static struct ggml_threadpool * tp = nullptr;
    if (tp == nullptr) {
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
        tpp.n_threads = 1;
        tp = ggml_threadpool_new(&tpp);
    }
    return tp;
}

static std::vector<uint8_t> & fused_mm_wdata() {
    static std::vector<uint8_t> buf;
    return buf;
}

// A-GATE diagnostic: the per-token mul_mat call census. Answers whether the fused
// path's gemv cost is structural (it issues more calls / touches more bytes than
// the graph) or mechanical (same shape of work, slower per call).
struct FusedMMCensus {
    // 0 = dense/lora, 1 = routed expert, 2 = lm_head
    uint64_t calls[3] = {0, 0, 0};
    uint64_t bytes[3] = {0, 0, 0};
    double   us[3]    = {0.0, 0.0, 0.0};
    void reset() { for (int i = 0; i < 3; i++) { calls[i] = 0; bytes[i] = 0; us[i] = 0.0; } }
};
static FusedMMCensus g_census;

// dst[n_out] (f32) = w[n_in, n_out] . x[n_in] (f32)
// `w_data` slices one expert out of a [ne0, ne1, n_expert] slab; pass w->data
// for a plain 2-D weight.
static void fused_mm_raw(const struct ggml_tensor * w, const void * w_data, float * out, const float * x) {
    const int64_t n_in  = w->ne[0];
    const int64_t n_out = w->ne[1];
    // census bucket, decided from the ORIGINAL tensor (ne[2] > 1 means an expert slab)
    const int cbucket = (w->ne[2] > 1) ? 1 : (n_out > 100000 ? 2 : 0);
    const auto c_t0 = std::chrono::steady_clock::now();

    struct ggml_tensor src0 = *w;                       // header copy: type/nb/buffer/extra
    src0.data     = const_cast<void *>(w_data);
    src0.ne[2]    = 1;
    src0.ne[3]    = 1;
    src0.nb[2]    = src0.nb[1] * n_out;
    src0.nb[3]    = src0.nb[2];
    src0.op       = GGML_OP_NONE;
    src0.view_src = nullptr;
    src0.view_offs = 0;
    for (int i = 0; i < GGML_MAX_SRC; i++) src0.src[i] = nullptr;

    struct ggml_tensor src1 = {};
    src1.type  = GGML_TYPE_F32;
    src1.ne[0] = n_in; src1.ne[1] = 1; src1.ne[2] = 1; src1.ne[3] = 1;
    src1.nb[0] = sizeof(float);
    src1.nb[1] = src1.nb[0] * n_in;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.data  = const_cast<float *>(x);
    src1.op    = GGML_OP_NONE;

    struct ggml_tensor dst = {};
    dst.type  = GGML_TYPE_F32;
    dst.ne[0] = n_out; dst.ne[1] = 1; dst.ne[2] = 1; dst.ne[3] = 1;
    dst.nb[0] = sizeof(float);
    dst.nb[1] = dst.nb[0] * n_out;
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data  = out;
    dst.op    = GGML_OP_MUL_MAT;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    // the work buffer, sized the way ggml_graph_plan sizes it
    size_t need = 0;
    if (!ggml_cpu_extra_work_size(1, &dst, &need)) {
        const enum ggml_type vdt = ggml_get_type_traits_cpu(src0.type)->vec_dot_type;
        need = (src1.type != vdt) ? ggml_row_size(vdt, n_in) : 0;
    }
    // iqk carves a Q8_K-sized activation region whatever the vec_dot_type is
    // (iqk_dispatch.cpp), so keep an upper bound on top: Q8_K is ~1.15 B/elem.
    const size_t upper = (size_t) n_in * 8 + 4096;
    if (need < upper) need = upper;
    std::vector<uint8_t> & wbuf = fused_mm_wdata();
    if (wbuf.size() < need) wbuf.resize(need);

    struct ggml_compute_params params;
    params.ith        = 0;
    params.nth        = 1;
    params.wsize      = wbuf.size();
    params.wdata      = wbuf.data();
    params.threadpool = fused_mm_tp();
    params.use_ref    = false;

    if (!ggml_cpu_extra_compute_forward(&params, &dst)) {
        ggml_compute_forward_mul_mat(&params, &dst);
    }
    g_census.calls[cbucket]++;
    g_census.bytes[cbucket] += (size_t) src0.nb[1] * (size_t) n_out;   // the src0 slice actually walked
    g_census.us[cbucket] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - c_t0).count();
}

// the A1 kill switch: `GGML_FUSED_MM_LEGACY=1` restores the per-row vec_dot loop
// so the substitution can be measured as a same-build A/B. Read once per token.
static bool g_mm_legacy = false;

} // namespace

// quantize the activation once to the weight's vec_dot_type, then per-row
// vec_dot — identical to ggml_compute_forward_mul_mat_one_chunk at batch-1.
struct FusedMM {
    const ggml_type_traits_cpu * qt;
    const ggml_type_traits_cpu * qtv;
    int64_t n_in;
    size_t qrow;
    std::vector<uint8_t> xq;

    FusedMM(const struct ggml_tensor * w, const float * x, int n_threads);
    void dot(const struct ggml_tensor * w, int row, float * out) const {
        FUSED_PROF_DOT_BEGIN();
        qt->vec_dot((int) n_in, out, 0, (const char *) w->data + (size_t) row * w->nb[1], 0, xq.data(), 0, 1);
        FUSED_PROF_DOT_END();
    }
};

FusedMM::FusedMM(const struct ggml_tensor * w, const float * x, int n_threads) {
    qt = ggml_get_type_traits_cpu(w->type);
    qtv = ggml_get_type_traits_cpu(qt->vec_dot_type);
    n_in = w->ne[0];
    qrow = ggml_row_size(qt->vec_dot_type, n_in);
    xq.resize(qrow);

    qtv->from_float(x, xq.data(), n_in);
}

// the lora mm: one batched mul_mat (A1), optionally multiplied elementwise by w_s
static void lora_mm(const struct ggml_tensor * w, const float * x, const struct ggml_tensor * w_s, float * out, int n_threads) {
    (void) n_threads;
    const int64_t n_out = w->ne[1];
    if (g_mm_legacy) {
        FusedMM mm(w, x, n_threads);
        for (int64_t j = 0; j < n_out; j++) {
            mm.dot(w, (int) j, &out[j]);
        }
    } else {
        FUSED_PROF_DOT_BEGIN();
        fused_mm_raw(w, w->data, out, x);
        FUSED_PROF_DOT_END();
    }
    if (w_s) {
        const float * s = (const float *) w_s->data;
        const int64_t ns = w_s->ne[0];
        for (int64_t j = 0; j < n_out; j++) {
            out[j] *= s[j % ns];
        }
    }
}

// grouped rms norm over the hc streams + the elementwise gamma, mirroring
// ggml_rms_norm (double accumulation) then ggml_mul
static void hc_rms_norm_gamma(const float * x, const struct ggml_tensor * w_norm,
                              float * out, int64_t n_embd, int64_t hc, float eps) {
    for (int64_t c = 0; c < hc; c++) {
        const float * xc = x + c * n_embd;
        float * oc = out + c * n_embd;
        const float * wc = (const float *) w_norm->data + c * n_embd;
        double sum = 0.0;
        for (int64_t i = 0; i < n_embd; i++) {
            sum += (double) (xc[i] * xc[i]);
        }
        const float mean  = sum / n_embd;
        const float scale = 1.0f / sqrtf(mean + eps);
        for (int64_t i = 0; i < n_embd; i++) {
            oc[i] = xc[i] * scale * wc[i];
        }
    }
}

// mean over the hc streams of xn*gate (the graph's mean_d1)
static void hc_stream_mean(const float * xn, const float * gate, float * out, int64_t n_embd, int64_t hc) {
    for (int64_t i = 0; i < n_embd; i++) {
        float s = 0.0f;
        for (int64_t c = 0; c < hc; c++) {
            s += xn[c * n_embd + i] * gate[c * n_embd + i];
        }
        out[i] = s / (float) hc;
    }
}

// the low-rank hc mixer (mirrors build_hc_mix): xn in, mixed + inject out
static void hc_mix(const struct ggml_tensor * w_down, const struct ggml_tensor * w_up,
                   const struct ggml_tensor * w_inject, int64_t hc,
                   const float * xn, float * mixed, float * inject,
                   int n_threads) {

    // the lo-rank down: lo = silu(mm(w_down, xn) * (1/hc))
    const int64_t low_rank = w_down->ne[1];
    std::vector<float> lo(low_rank);
    lora_mm(w_down, xn, nullptr, lo.data(), n_threads);
    const float inv_hc = 1.0f / (float) hc;
    for (int64_t j = 0; j < low_rank; j++) {
        const float v = lo[j] * inv_hc;
        lo[j] = v / (1.0f + expf(-v)); // silu
    }
    // the gate: sigmoid(mm(w_up, lo)); gated = xn*gate; mean over streams
    const int64_t hc_dim = w_up->ne[1];
    std::vector<float> gate(hc_dim);
    lora_mm(w_up, lo.data(), nullptr, gate.data(), n_threads);
    for (int64_t i = 0; i < hc_dim; i++) {
        gate[i] = 1.0f / (1.0f + expf(-gate[i]));
    }
    const int64_t n_embd2 = hc_dim / hc;
    hc_stream_mean(xn, gate.data(), mixed, n_embd2, hc);
    if (inject) {
        lora_mm(w_inject, xn, nullptr, inject, n_threads);
    }
}


// ---- the MoE (mirrors build_moe_ffn + build_layer_ffn's shared expert) -----

// the graph's silu uses the SIMD ggml_v_expf approximation (1.45 ulps), not
// expf — the scalar mirror of the AVX2 ggml_v_silu/ggml_v_expf path
static float v_silu(float x) {
    const float r = 12582912.0f;
    const float z = fmaf(x, 1.4426950216293335f, r);
    const float n = z - r;
    const float b = fmaf(-n, 1.428606765330187e-06f,
                         fmaf(-n, 0.693145751953125f, x));
    uint32_t zu;
    memcpy(&zu, &z, sizeof(zu));
    const uint32_t e = zu << 23;
    uint32_t kb = e + 1;
    float k;
    memcpy(&k, &kb, sizeof(k));
    const float u = b * b;
    float j = fmaf(fmaf(fmaf(0.008247390389442444f, b, 0.04189976677298546f), u,
                        fmaf(0.16668395698070526f, b, 0.4999912679195404f)), u,
                   0.9999994039535522f * b);
    if (fabsf(n) <= 126.0f) return fmaf(j, k, k);
    // the overflow/underflow tail (|n| > 126): fall back to the plain expf
    return x / (1.0f + expf(-x));
}

static void fused_moe(
        const struct llama_layer & L,
        const struct llama_hparams & hp,
        const float * x, float * out, int n_threads) {

    const int64_t n_expert = L.ffn_gate_inp->ne[1]; // 512
    const int64_t n_used   = hp.n_expert_used;      // 10
#if FUSED_DBG_ON
    static int dn_call = 0;
    static FILE * dn_file = nullptr;
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && dn_call >= 24 && dn_call <= 26 && !dn_file) {
        char fn[128];
        snprintf(fn, sizeof(fn), "/tmp/qwen4exp-builds/f_moe_dn_%d.bin", dn_call);
        dn_file = fopen(fn, "wb");
    }
#endif

    // router logits
    std::vector<float> logits(n_expert);
    lora_mm(L.ffn_gate_inp, x, nullptr, logits.data(), n_threads);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        extern float ggml_table_f32_f16[1 << 16];
        fprintf(stderr, "  moe f16table[0x3C00]=%.6g f16table[0]=%.6g\n",
                (double) ggml_table_f32_f16[0x3C00], (double) ggml_table_f32_f16[0]);
    }
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        fprintf(stderr, "  moe router[0..5] = %.6g %.6g %.6g %.6g %.6g %.6g  (n_used=%d)\n",
                (double) logits[0], (double) logits[1], (double) logits[2],
                (double) logits[3], (double) logits[4], (double) logits[5],
                (int) n_used);
    }

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        static int moe_n = 0;
        if (moe_n < 2) {
            char fn[128];
            snprintf(fn, sizeof(fn), "/tmp/qwen4exp-builds/f_moe_logits_%d.bin", moe_n);
            FILE * f = fopen(fn, "wb");
            if (f) { fwrite(logits.data(), 4, n_expert, f); fclose(f); }
        }
        moe_n++;
    }
    // softmax over the experts (max-subtract, sequential exp/sum — the
    // ggml_vec_soft_max_f32 contract)
    float maxv = -INFINITY;
    for (int64_t e = 0; e < n_expert; e++) maxv = fmaxf(maxv, logits[e]);
    double sum = 0.0;
    for (int64_t e = 0; e < n_expert; e++) sum += expf(logits[e] - maxv);
    const float inv_sum = (float) (1.0 / sum);

    // argsort: the top-k indices of the probs (descending; index tie-break)
    std::vector<int32_t> sel(n_used);
    {
        std::vector<float> probs(n_expert);
        for (int64_t e = 0; e < n_expert; e++) probs[e] = expf(logits[e] - maxv) * inv_sum;
        for (int64_t j = 0; j < n_used; j++) {
            int32_t best = -1;
            float bestv = -INFINITY;
            for (int64_t e = 0; e < n_expert; e++) {
                if (probs[e] > bestv) { bestv = probs[e]; best = (int32_t) e; }
            }
            sel[j] = best;
            probs[best] = -INFINITY;
        }
    }

    // the top-k weights (the moe_topk_norm math: gather the softmax values,
    // clamp the sum at the F16 min, renormalize)
    std::vector<float> w(n_used);
    {
        float wsum = 0.0f;
        for (int64_t j = 0; j < n_used; j++) {
            const float v = expf(logits[sel[j]] - maxv) * inv_sum;
            w[j] = v;
            wsum += v;
        }
        const float s = fmaxf(wsum, 6.103515625e-5f);
        for (int64_t j = 0; j < n_used; j++) w[j] /= s;
    }

    // the expert gemvs: up/gate [n_embd, n_ff, n_expert] with the Q8_K activation.
    // A1: one batched mul_mat per selected expert (a 2-D slice of the slab)
    // instead of n_ff per-row vec_dots.
    const struct ggml_type_traits_cpu * qt_up = ggml_get_type_traits_cpu(L.ffn_up_exps->type);
    const struct ggml_type_traits_cpu * qt_upv = ggml_get_type_traits_cpu(qt_up->vec_dot_type);
    const size_t xq_up_size = ggml_row_size(qt_up->vec_dot_type, hp.n_embd);
    std::vector<uint8_t> xq_up;
    if (g_mm_legacy) {
        xq_up.resize(xq_up_size);
        qt_upv->from_float(x, xq_up.data(), hp.n_embd);
    }

    const int64_t n_ff = L.ffn_up_exps->ne[1]; // 640
    std::vector<float> glu(n_used * n_ff), up_tmp(n_used * n_ff), gate_tmp(n_used * n_ff);
    const size_t nb_up_exp = L.ffn_up_exps->nb[2]; // expert stride

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        fprintf(stderr, "  moe up: type=%s nb1=%zu nb2=%zu n_embd=%lld sel[0]=%d\n",
                ggml_type_name(L.ffn_up_exps->type), (size_t) L.ffn_up_exps->nb[1],
                (size_t) L.ffn_up_exps->nb[2], (long long) hp.n_embd, sel[0]);
    }
    for (int64_t j = 0; j < n_used; j++) {
        const int32_t e = sel[j];
        const char * up_e = (const char *) L.ffn_up_exps->data + (size_t) e * nb_up_exp;
        const char * gt_e = (const char *) L.ffn_gate_exps->data + (size_t) e * L.ffn_gate_exps->nb[2];
        FUSED_PROF_DOT_BEGIN();
        if (g_mm_legacy) {
            for (int64_t r = 0; r < n_ff; r++) {
                qt_up->vec_dot((int) hp.n_embd, &up_tmp[j * n_ff + r], 0,
                               up_e + (size_t) r * L.ffn_up_exps->nb[1], 0, xq_up.data(), 0, 1);
                qt_up->vec_dot((int) hp.n_embd, &gate_tmp[j * n_ff + r], 0,
                               gt_e + (size_t) r * L.ffn_gate_exps->nb[1], 0, xq_up.data(), 0, 1);
            }
        } else {
            fused_mm_raw(L.ffn_up_exps,   up_e, &up_tmp[j * n_ff],   x);
            fused_mm_raw(L.ffn_gate_exps, gt_e, &gate_tmp[j * n_ff], x);
        }
        FUSED_PROF_DOT_END();
        for (int64_t r = 0; r < n_ff; r++) {
            const float g = gate_tmp[j * n_ff + r];
            glu[j * n_ff + r] = up_tmp[j * n_ff + r] * (g / (1.0f + expf(-g))); // silu(gate)*up
        }
    }
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        static int up_n = 0;
        if (up_n >= 24 && up_n <= 26) {
            char fn[128];
            snprintf(fn, sizeof(fn), "/tmp/qwen4exp-builds/f_moe_up_%d.bin", up_n);
            FILE * f = fopen(fn, "wb");
            if (f) {
                fwrite(up_tmp.data(), 4, n_used * n_ff, f);
                fwrite(gate_tmp.data(), 4, n_used * n_ff, f);
                fwrite(glu.data(), 4, n_used * n_ff, f);
                fclose(f);
            }
        }
        up_n++;
    }
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        int un = 0, gn = 0, gln = 0;
        for (int64_t i = 0; i < n_used * n_ff; i++) { if (std::isnan(up_tmp[i])) un++; if (std::isnan(gate_tmp[i])) gn++; if (std::isnan(glu[i])) gln++; }
        fprintf(stderr, "  moe updots: up nan=%d gate nan=%d glu nan=%d  up[0]=%.6g glu[0]=%.6g\n", un, gn, gln, (double) up_tmp[0], (double) glu[0]);
    }

    // the down expert gemvs: [640, 2560, 512] IQ4_NL with its vec_dot_type.
    // NOTE: the CPU_REPACK buffer may have interleaved the IQ4_NL rows (the
    // graph's repack compute handles that layout; the plain vec_dot cannot).
    // Detect via tensor->extra and mirror the interleaved layout here, per
    // ggml_gemv_iq4_nl_{4x4,8x8}_q8_0_generic: a group holds I rows; the row's
    // slot j reads scale d[j] and nibbles qs[k*I*I + j*I + i] (low nibble vs the
    // activation qs[k*I+i], high nibble vs qs[k*I+i+16]).
    const struct ggml_type_traits_cpu * qt_dn = ggml_get_type_traits_cpu(L.ffn_down_exps->type);
    const struct ggml_type_traits_cpu * qt_dnv = ggml_get_type_traits_cpu(qt_dn->vec_dot_type);
    const size_t glu_q_size = ggml_row_size(qt_dn->vec_dot_type, n_ff);
    static const int8_t kv_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    const bool dn_repacked = L.ffn_down_exps->extra != nullptr;
    // the pre-quantized activation is only needed by the per-row paths (the
    // legacy arm and the repacked IQ4_NL mirror); the batched mul_mat quantizes
    // into its own wdata.
    std::vector<uint8_t> glu_q;
    if (g_mm_legacy || dn_repacked) {
        glu_q.resize(glu_q_size * n_used);
        for (int64_t j = 0; j < n_used; j++) {
            qt_dnv->from_float(glu.data() + j * n_ff, glu_q.data() + j * glu_q_size, n_ff);
        }
    }
    const bool up_repacked = L.ffn_up_exps->extra != nullptr;
    const bool gt_repacked = L.ffn_gate_exps->extra != nullptr;
    const int64_t rp_I = dn_repacked ? (ggml_cpu_has_avx2() ? 8 : 4) : 0;
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        fprintf(stderr, "  moe repack: up=%s extra=%d gt=%s extra=%d dn=%s extra=%d rp_I=%lld\n",
                ggml_type_name(L.ffn_up_exps->type), (int) up_repacked,
                ggml_type_name(L.ffn_gate_exps->type), (int) gt_repacked,
                ggml_type_name(L.ffn_down_exps->type), (int) dn_repacked, (long long) rp_I);
    }
    const int64_t rp_nblocks = n_ff / 32;
    const size_t nb_dn_exp = L.ffn_down_exps->nb[2];
    std::vector<float> down_acc(hp.n_embd, 0.0f);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        fprintf(stderr, "  moe dn: type=%s vec_dot_type=%s nb1=%zu nb2=%zu glu_q_size=%zu buf=%s repacked=%d I=%lld\n",
                ggml_type_name(L.ffn_down_exps->type), ggml_type_name(qt_dn->vec_dot_type),
                (size_t) L.ffn_down_exps->nb[1], (size_t) L.ffn_down_exps->nb[2], glu_q_size,
                L.ffn_down_exps->buffer ? ggml_backend_buffer_name(L.ffn_down_exps->buffer) : "none",
                (int) dn_repacked, (long long) rp_I);
        fprintf(stderr, "  moe up: type=%s buf=%s\n", ggml_type_name(L.ffn_up_exps->type),
                L.ffn_up_exps->buffer ? ggml_backend_buffer_name(L.ffn_up_exps->buffer) : "none");
    }
    std::vector<float> down_j;
    for (int64_t j = 0; j < n_used; j++) {
        const int32_t e = sel[j];
        const char * dn_e = (const char *) L.ffn_down_exps->data + (size_t) e * nb_dn_exp;
        if (!g_mm_legacy) {
            // A1: one batched mul_mat over the expert's [n_ff, n_embd] slice.
            //
            // This used to exclude repacked down-experts and fall through to the
            // hand-rolled interleaved IQ4_NL mirror below. The census
            // (2026-09-02) showed why that mattered: 42 of the 48
            // ffn_down_exps tensors ARE in CPU_REPACK, so 84 of the 96 MoE
            // invocations per token walked 2560 rows x 10 experts by hand —
            // 2.15 M scalar dots, ~598 ms of the 801 ms "gemv" column, while the
            // 3,053 genuinely batched mul_mats cost only 203 ms in total.
            // ggml_cpu_extra_compute_forward() dispatches a repacked src0 to the
            // repack 8x8 gemv, which is the kernel the graph itself uses, so the
            // mirror is not needed here — it stays only for the legacy arm.
            down_j.resize(hp.n_embd);
            FUSED_PROF_DOT_BEGIN();
            fused_mm_raw(L.ffn_down_exps, dn_e, down_j.data(), glu.data() + j * n_ff);
            FUSED_PROF_DOT_END();
            for (int64_t r = 0; r < hp.n_embd; r++) {
                down_acc[r] += down_j[r] * w[j];
            }
            continue;
        }
        for (int64_t r = 0; r < hp.n_embd; r++) {
            float v = 0.0f;
            FUSED_PROF_DOT_BEGIN();
            if (dn_repacked) {
                // the interleaved IQ4_NL dot for one row of its group
                const int64_t g = r / rp_I, slot = r % rp_I;
                const char * grp = dn_e + g * (rp_nblocks * 18 * rp_I);
                if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && j == 0 && r < 3) {
                    const ggml_fp16_t * d0 = (const ggml_fp16_t *) grp;
                    const ggml_fp16_t * d1 = (const ggml_fp16_t *) (grp + 18 * rp_I);
                    fprintf(stderr, "  moe dn r%lld: slot=%lld d[0..2]=%.6g %.6g %.6g  l1d[0..2]=%.6g %.6g %.6g\n",
                            (long long) r, (long long) slot,
                            (double) ggml_fp16_to_fp32(d0[0]), (double) ggml_fp16_to_fp32(d0[1]), (double) ggml_fp16_to_fp32(d0[2]),
                            (double) ggml_fp16_to_fp32(d1[0]), (double) ggml_fp16_to_fp32(d1[1]), (double) ggml_fp16_to_fp32(d1[2]));
                }
                const char * act = (const char *) glu_q.data() + j * glu_q_size;
                float dot = 0.0f;
                if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && j == 0 && r == 3) {
                    const char * bl0 = grp + 0 * (18 * rp_I);
                    const ggml_fp16_t * dsc0 = (const ggml_fp16_t *) bl0;
                    const uint8_t * qs0 = (const uint8_t *) (bl0 + 2 * rp_I);
                    fprintf(stderr, "  moe dn r3 rep: slot=%lld d[slot]=%.6g qs[0..3]=%d %d %d %d qs[64..67]=%d %d %d %d\n",
                            (long long) slot, (double) ggml_fp16_to_fp32(dsc0[slot]),
                            qs0[0], qs0[1], qs0[2], qs0[3], qs0[64], qs0[65], qs0[66], qs0[67]);
                }
                for (int64_t l = 0; l < rp_nblocks; l++) {
                    const char * bl = grp + l * (18 * rp_I);
                    const ggml_fp16_t * dsc = (const ggml_fp16_t *) bl;
                    const uint8_t * qs = (const uint8_t *) (bl + 2 * rp_I);
                    const ggml_fp16_t * ad = (const ggml_fp16_t *) (act + l * 34);
                    const int8_t * aqs = (const int8_t *) (act + l * 34 + 2);
                    // the graph's kernel: sumf[j] += sumi_k * d * a per k (the
                    // per-k products round separately — NOT one accumulated sumi)
                    const float da = ggml_fp16_to_fp32(dsc[slot]) * ggml_fp16_to_fp32(ad[0]);
                    for (int64_t k = 0; k < 32 / (2 * rp_I); k++) {
                        int sumi = 0;
                        for (int64_t i = 0; i < rp_I; i++) {
                            const uint8_t byte = qs[k * rp_I * rp_I + slot * rp_I + i];
                            sumi += kv_iq4nl[byte & 0xF] * aqs[k * rp_I + i]
                                 +  kv_iq4nl[byte >> 4] * aqs[k * rp_I + i + 16];
                        }
                        dot += (float) sumi * da;
                    }
                }
                v = dot;
            } else {
                qt_dn->vec_dot((int) n_ff, &v, 0, dn_e + (size_t) r * L.ffn_down_exps->nb[1], 0,
                               glu_q.data() + j * glu_q_size, 0, 1);
            }
            FUSED_PROF_DOT_END();
            down_acc[r] += v * w[j];
#if FUSED_DBG_ON
            if (dn_file) fwrite(&v, 4, 1, dn_file);
#endif
            if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && r < 2) {
                fprintf(stderr, "  moe dn j%lld r%lld: e=%d v=%.6g w=%.6g\n", (long long) j, (long long) r, e, (double) v, (double) w[j]);
            }
        }
    }
#if FUSED_DBG_ON
    if (dn_file) { fclose(dn_file); dn_file = nullptr; }
    dn_call++;
#endif
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        int dn = 0; for (int64_t i = 0; i < hp.n_embd; i++) if (std::isnan(down_acc[i])) dn++;
        fprintf(stderr, "  moe dnacc: nan=%d [0]=%.6g\n", dn, (double) down_acc[0]);
    }


    // the shared expert + its sigmoided gate
    {

        std::vector<float> up_s(n_ff), gate_s(n_ff), glu_s(n_ff), down_s(hp.n_embd);
        lora_mm(L.ffn_up_shexp, x, nullptr, up_s.data(), n_threads);
        lora_mm(L.ffn_gate_shexp, x, nullptr, gate_s.data(), n_threads);
        for (int64_t r = 0; r < n_ff; r++) {
            const float g = gate_s[r];
            glu_s[r] = up_s[r] * (g / (1.0f + expf(-g)));
        }
        lora_mm(L.ffn_down_shexp, glu_s.data(), nullptr, down_s.data(), n_threads);
        // the shared-expert gate is a 1-row projection; size the destination from
        // the tensor rather than assuming it (the batched mul_mat writes ne[1])
        std::vector<float> shg_v(L.ffn_gate_inp_shexp->ne[1]);
        lora_mm(L.ffn_gate_inp_shexp, x, nullptr, shg_v.data(), n_threads);
        float shg = 1.0f / (1.0f + expf(-shg_v[0]));

        for (int64_t i = 0; i < hp.n_embd; i++) {
            down_acc[i] += down_s[i] * shg;
        }
    }

    // the expert weight scale (the hparams.expert_weights_scale)
    if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f) {
        for (int64_t i = 0; i < hp.n_embd; i++) down_acc[i] *= hp.expert_weights_scale;
    }

    memcpy(out, down_acc.data(), hp.n_embd * sizeof(float));
}

// the hc combine: res[i,c] += block[i] * (2*sigmoid(inject[c]/hc))
static void hc_combine(float * res, const float * block, const float * inject,
                       int64_t hc, int64_t n_embd) {
    std::vector<float> w(hc);
    for (int64_t c = 0; c < hc; c++) {
        const float v = inject[c] / (float) hc;
        w[c] = 2.0f / (1.0f + expf(-v));
    }
    for (int64_t c = 0; c < hc; c++) {
        float * rc = res + c * n_embd;
        const float wc = w[c];
        for (int64_t i = 0; i < n_embd; i++) {
            rc[i] += block[i] * wc;
        }
    }
}

// ---- the fused GDN layer ---------------------------------------------------

// mirror of ggml_compute_forward_ssm_conv_f32 at batch-1 (float accumulation);
// the kernel is {d_conv, n_ch} with the tap as the fast index (c[i0 + i1*nc])
static void conv1d_4tap(const float * window, const float * kernel, float * out, int64_t n_ch) {
    for (int64_t i = 0; i < n_ch; i++) {
        float sumf = 0.0f;
        for (int k = 0; k < 4; k++) {
            sumf += window[k * n_ch + i] * kernel[k + i * 4];
        }
        out[i] = sumf;
    }
}

// mirror of ggml_compute_forward_l2_norm_f32 (double accumulation)
static void l2_norm(const float * x, float * y, int64_t n, float eps) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const float xi = x[i];
        sum += (double) (xi * xi);
    }
    const float scale = 1.0f / fmaxf(sqrtf((float) sum), eps);
    for (int64_t i = 0; i < n; i++) {
        y[i] = x[i] * scale;
    }
}

// the fused GDN layer for the single-token decode:
//   x      : the hc-wide input [hc*n_embd]
//   res_in : the hc residual stream [hc*n_embd] (updated in place by hc_combine)
//   out    : the layer output [n_embd] (the hc-mixed result of the ffn side)
//   conv_state, ssm_state : the memory context tensors (row layout)
// All weights come from the model's layer tensors; every op mirrors the graph.
void fused_gdn_layer(
        const struct llama_layer & L,
        const struct llama_hparams & hp,
        const float * x, float * res_in_out, float * out,
        float * conv_state_row, float * ssm_state_row,
        int n_threads, int il) {

    const int64_t hc      = hp.dsv4_hc_mult; // 4
    const int64_t n_embd  = hp.n_embd;       // 2560
    const int64_t hc_dim  = hc * n_embd;     // 10240
    const float   eps     = hp.f_norm_rms_eps;

    const int64_t S_k      = hp.ssm_d_state;          // 128
    const int64_t H_k      = hp.ssm_n_group;          // 16
    const int64_t H_v      = hp.ssm_dt_rank;          // 48
    const int64_t S_v      = hp.ssm_d_state;          // 128
    const int64_t n_in     = hc_dim;
    const int64_t low_rank = hp.hc_low_rank;           // 320

    (void) n_threads;

    std::vector<float> xn(hc_dim), mixed(n_embd), inject(hc * n_embd), gate_hc(hc_dim);

    // ---- hc_mix (attn side) ----
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        const float * w0 = (const float *) L.hc_attn_norm->data;
        fprintf(stderr, "  gdn hc input: x[0..3]=%.6g %.6g %.6g %.6g w_norm type=%d ne=[%lld,%lld,%lld] w[0..3]=%.6g %.6g %.6g %.6g\n",
                (double) x[0], (double) x[1], (double) x[2], (double) x[3],
                (int) L.hc_attn_norm->type, (long long) L.hc_attn_norm->ne[0],
                (long long) L.hc_attn_norm->ne[1], (long long) L.hc_attn_norm->ne[2],
                (double) w0[0], (double) w0[1], (double) w0[2], (double) w0[3]);
    }
    hc_rms_norm_gamma(x, L.hc_attn_norm, xn.data(), n_embd, hc, eps);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        fprintf(stderr, "  gdn hc xn: xn[0..3]=%.6g %.6g %.6g %.6g\n",
                (double) xn[0], (double) xn[1], (double) xn[2], (double) xn[3]);
    }
    hc_mix(L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, hc, xn.data(), mixed.data(), inject.data(), n_threads);

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  gdn hc_mix done\n");
    // ---- GDN body ----
    // the qkv span is the conv channels (q+k+v), not ssm_d_inner (the v span alone)
    const int64_t qkv_span = L.wqkv->ne[1];
    std::vector<float> qkv(qkv_span);
    std::vector<float> z(hp.ssm_d_inner);              // 6144 (v-dim)
    lora_mm(L.wqkv, mixed.data(), nullptr, qkv.data(), n_threads);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        fprintf(stderr, "  gdn post-wqkv: xn[0..1]=%.6g %.6g mixed[0..1]=%.6g %.6g qkv[0..1]=%.6g %.6g (qkv_span=%lld wqkv ne=[%lld,%lld,%lld] type=%d)\n",
                (double) xn[0], (double) xn[1], (double) mixed[0], (double) mixed[1],
                (double) qkv[0], (double) qkv[1], (long long) qkv_span,
                (long long) L.wqkv->ne[0], (long long) L.wqkv->ne[1], (long long) L.wqkv->ne[2], (int) L.wqkv->type);
    }
    if (FUSED_DBG("GGML_FUSED_DUMP_FLAYERS") && FUSED_DBG("GGML_FUSED_ONCE")) {
        char fmix[128];
        snprintf(fmix, sizeof(fmix), "/tmp/qwen4exp-builds/f_mix_%d.bin", il);
        FILE * f = fopen(fmix, "wb");
        if (f) {
            fwrite(mixed.data(), 4, n_embd, f);
            fwrite(xn.data(), 4, hc * n_embd, f);
            fclose(f);
        }
    }
    lora_mm(L.wqkv_gate, mixed.data(), nullptr, z.data(), n_threads);

    // beta = sigmoid(mm(ssm_beta, mixed))
    std::vector<float> beta(H_v);
    lora_mm(L.ssm_beta, mixed.data(), nullptr, beta.data(), n_threads);
    for (int64_t i = 0; i < H_v; i++) {
        beta[i] = 1.0f / (1.0f + expf(-beta[i]));
    }

    // gate = softplus(mm(ssm_alpha, mixed) + ssm_dt) * ssm_a
    std::vector<float> gate(H_v);
    lora_mm(L.ssm_alpha, mixed.data(), nullptr, gate.data(), n_threads);
    const float * dt = (const float *) L.ssm_dt->data;
    const float * ssa = (const float *) L.ssm_a->data;
    for (int64_t i = 0; i < H_v; i++) {
        float v = gate[i] + dt[i];
        // softplus: log(1 + exp(x))
        v = v > 0.0f ? v + log1pf(expf(-v)) : log1pf(expf(v));
        gate[i] = v * ssa[i];
    }

    // conv: window = conv_state + qkv, in the graph's channel-major layout:
    // element (tap k, channel i) at i*d_conv + k; the state row is (channel i,
    // tap k) at i*(d_conv-1) + k (the concat's ne[0]-fastest layout)
    const int64_t d_conv = L.ssm_conv1d->ne[0]; // 4
    const int64_t n_ch   = L.ssm_conv1d->ne[1]; // 10240
    std::vector<float> window((d_conv) * n_ch);
    for (int64_t i = 0; i < n_ch; i++) {
        for (int64_t k = 0; k < d_conv - 1; k++) {
            window[i * d_conv + k] = conv_state_row[i * (d_conv - 1) + k];
        }
        window[i * d_conv + (d_conv - 1)] = qkv[i];
    }
    std::vector<float> conv_out(n_ch);
    std::vector<float> ssm_cw_tmp;
    const float * ssm_cw = fused_weights_f32(L.ssm_conv1d, ssm_cw_tmp);
    for (int64_t i = 0; i < n_ch; i++) {
        float sumf = 0.0f;
        for (int64_t k = 0; k < d_conv; k++) {
            sumf += window[i * d_conv + k] * ssm_cw[k + i * d_conv];
        }
        conv_out[i] = sumf;
    }
    // the graph's SIMD silu (ggml_v_silu / ggml_v_expf)
    ggml_vec_silu_f32((int) n_ch, conv_out.data(), conv_out.data());
    // conv state update: keep the last d_conv-1 taps (drop tap 0, append qkv)
    for (int64_t i = 0; i < n_ch; i++) {
        for (int64_t k = 0; k < d_conv - 1; k++) {
            conv_state_row[i * (d_conv - 1) + k] = window[i * d_conv + k + 1];
        }
    }

    // q/k/v split + l2 norms
    const int64_t q_off = 0, k_off = S_k * H_k, v_off = 2 * S_k * H_k;
    std::vector<float> q(S_k * H_k), k(S_k * H_k), v(S_v * H_v);
    // the graph's l2_norm normalizes each ne[0] column separately (one scale per head)
    for (int64_t h = 0; h < H_k; h++) {
        l2_norm(conv_out.data() + q_off + h * S_k, q.data() + h * S_k, S_k, eps);
        l2_norm(conv_out.data() + k_off + h * S_k, k.data() + h * S_k, S_k, eps);
    }
    memcpy(v.data(), conv_out.data() + v_off, S_v * H_v * sizeof(float));

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  gdn conv done\n");
    // GDN scan: the ggml_gated_delta_net kernel on scratch tensors (nth=1)
    // q/k [S_k, H_k, 1, 1]; v [S_v, H_v, 1, 1]; g/b [1, H_v, 1, 1]; s [S_v, S_v, H_v, 1]
    // A2: the arena, sized from the tensors this context actually creates
    // (the [S_v, S_v, H_v] state is the 3.1 MB term) plus the kernel's own
    // ne-sized output tensor and the object headers.
    // tq + tk (S_k*H_k each), tv (S_v*H_v), tg + tb (H_v each), ts (S_v*S_v*H_v)
    // and — the term that is easy to miss — the kernel's OWN result tensor, which
    // ggml_gated_delta_net allocates as [S_v*H, n_tokens + K*S_v] = S_v*H_v*(1+S_v).
    const size_t gdn_need = (size_t) (2 * S_k * H_k + S_v * H_v + 2 * H_v +
                                      S_v * S_v * H_v +
                                      S_v * H_v * (1 + S_v)) * sizeof(float)
                          + 32 * ggml_tensor_overhead() + (1u << 16);
    ggml_context * gctx = fused_arena_init(FUSED_ARENA_LAYER, gdn_need, 64u << 20);
    ggml_tensor * tq = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, S_k, H_k, 1, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, S_k, H_k, 1, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, S_v, H_v, 1, 1);
    ggml_tensor * tg = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, H_v, 1, 1);
    ggml_tensor * tb = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, H_v, 1, 1);
    ggml_tensor * ts = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, S_v, S_v, H_v, 1);
    memcpy(tq->data, q.data(), S_k * H_k * sizeof(float));
    memcpy(tk->data, k.data(), S_k * H_k * sizeof(float));
    memcpy(tv->data, v.data(), S_v * H_v * sizeof(float));
    memcpy(tg->data, gate.data(), H_v * sizeof(float));
    memcpy(tb->data, beta.data(), H_v * sizeof(float));
    memcpy(ts->data, ssm_state_row, S_v * S_v * H_v * sizeof(float));
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  gdn qkv[0..2]=%.6g %.6g %.6g conv[0..2]=%.6g %.6g %.6g\n", (double)qkv[0],(double)qkv[1],(double)qkv[2],(double)conv_out[0],(double)conv_out[1],(double)conv_out[2]);
    ggml_tensor * tgd = ggml_gated_delta_net(gctx, tq, tk, tv, tg, tb, ts, 1);
    // the kernel's K=1 output carries [attn (S_v*H_v) | new_state (S_v*S_v*H_v)],
    // written past the tensor's own ne-based allocation — point it at a full-size
    // buffer (a 3 MB write into the 24 KB tensor buffer was the heap corruption
    // that surfaced as the nondeterministic teardown segfault)
    static std::vector<float> tgd_buf; // static: the kernel writes through the tensor's data pointer
    tgd_buf.resize(S_v * H_v + S_v * S_v * H_v);
    tgd->data = tgd_buf.data();
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        float gmin = 1e30f, gmax = -1e30f, bmin = 1e30f, bmax = -1e30f, smin = 1e30f, smax = -1e30f;
        for (int64_t i = 0; i < H_v; i++) { gmin = fminf(gmin, gate[i]); gmax = fmaxf(gmax, gate[i]); bmin = fminf(bmin, beta[i]); bmax = fmaxf(bmax, beta[i]); }
        for (int64_t i = 0; i < S_v*S_v*H_v; i++) { smin = fminf(smin, ((const float*)ts->data)[i]); smax = fmaxf(smax, ((const float*)ts->data)[i]); }
        fprintf(stderr, "  gdn gate[min=%.6g max=%.6g] beta[min=%.6g max=%.6g] state[min=%.6g max=%.6g]\n", (double)gmin,(double)gmax,(double)bmin,(double)bmax,(double)smin,(double)smax);
    }

    static struct ggml_threadpool * gdn_tp = nullptr;
    if (gdn_tp == nullptr) {
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
        tpp.n_threads = 1;
        gdn_tp = ggml_threadpool_new(&tpp);
    }
    struct ggml_compute_params gparams;
    gparams.ith = 0;
    gparams.nth = 1;
    // the GDN kernel uses wdata as per-thread scratch: delta = wdata + ith*per_thread + CACHE_LINE_SIZE_F32
    // (CACHE_LINE_SIZE_F32 = 64 bytes / sizeof(float) = 16; internal to ggml-cpu, so mirrored here)
    std::vector<float> gdn_wdata(S_v + 16);
    gparams.wsize = gdn_wdata.size() * sizeof(float);
    gparams.wdata = gdn_wdata.data();
    gparams.threadpool = gdn_tp;
    gparams.use_ref = false;
    ggml_compute_forward_gated_delta_net(&gparams, tgd);

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  gdn kernel done\n");
    // z-gated rms norm: rms_norm(output, ssm_norm) * sigmoid(z)
    std::vector<float> gdn_out(S_v * H_v);
    memcpy(gdn_out.data(), tgd->data, S_v * H_v * sizeof(float));
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        float omin = 1e30f, omax = -1e30f; int onn = 0, oni = 0;
        for (int64_t i = 0; i < S_v*H_v; i++) { if (std::isnan(gdn_out[i])) onn++; if (std::isinf(gdn_out[i])) oni++; omin = fminf(omin, gdn_out[i]); omax = fmaxf(omax, gdn_out[i]); }
        fprintf(stderr, "  gdn out: nan=%d inf=%d min=%.6g max=%.6g\n", onn, oni, (double)omin, (double)omax);
    }
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  gdn out[0..2]=%.6g %.6g %.6g z[0..2]=%.6g %.6g %.6g\n", (double)gdn_out[0],(double)gdn_out[1],(double)gdn_out[2],(double)z[0],(double)z[1],(double)z[2]);
    // the kernel's K=1 output carries [attn | new_state]; the state advances
    // (ts holds the input state; copying it back would freeze the recurrence).
    //
    // BUGFIX (INF-70 A1/A2 pass): the offset was `(const char *) tgd->data +
    // S_v * H_v`, i.e. 6144 BYTES into the buffer, where the state actually
    // starts 6144 FLOATS in. ggml_gated_delta_net's result is
    // [S_v*H, n_tokens + K*S_v] and row 0 is the attention output, so the state
    // begins at float index S_v*H_v. The old expression read from float index
    // 1536 and stayed in bounds (hence no crash), but every GDN layer's
    // recurrent state was shifted by 4608 floats from step 2 onward — invisible
    // at step 1, where the incoming state comes from the prompt batch and is
    // identical on both A/B arms, which is why the INF-67 per-layer
    // bit-exactness checks did not see it.
    memcpy(ssm_state_row, (const float *) tgd->data + (size_t) S_v * H_v,
           (size_t) S_v * S_v * H_v * sizeof(float));
    ggml_free(gctx);

    const float * znorm = (const float *) L.ssm_norm->data;
    std::vector<float> final_in(S_v * H_v);
    for (int64_t h = 0; h < H_v; h++) {
        const float * row = gdn_out.data() + h * S_v;
        double sum = 0.0;
        for (int64_t i = 0; i < S_v; i++) {
            sum += (double) (row[i] * row[i]);
        }
        const float scale = 1.0f / sqrtf((float) (sum / S_v) + eps);
        for (int64_t i = 0; i < S_v; i++) {
            const float zg = 1.0f / (1.0f + expf(-z[h * S_v + i]));
            final_in[h * S_v + i] = row[i] * scale * znorm[i] * zg;
        }
    }

    std::vector<float> attn_out(n_embd);
    lora_mm(L.ssm_out, final_in.data(), nullptr, attn_out.data(), n_threads);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        int nn = 0, ni = 0;
        float mn = 1e30f, mx = -1e30f;
        for (int64_t i = 0; i < S_v * H_v; i++) {
            if (std::isnan(final_in[i])) nn++;
            if (std::isinf(final_in[i])) ni++;
            mn = fminf(mn, final_in[i]); mx = fmaxf(mx, final_in[i]);
        }
        int ann = 0, ain = 0; float amn = 1e30f, amx = -1e30f;
        for (int64_t i = 0; i < n_embd; i++) { if (std::isnan(attn_out[i])) ann++; if (std::isinf(attn_out[i])) ain++; amn = fminf(amn, attn_out[i]); amx = fmaxf(amx, attn_out[i]); }
        fprintf(stderr, "  gdn final_in: nan=%d inf=%d min=%.6g max=%.6g  attn_out: nan=%d inf=%d min=%.6g max=%.6g\n",
                nn, ni, (double) mn, (double) mx, ann, ain, (double) amn, (double) amx);
    }

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  gdn ssm_out done\n");
    // ---- MoE ----
    std::vector<float> moe_out(n_embd);
    fused_moe(L, hp, attn_out.data(), moe_out.data(), n_threads);

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  gdn moe1 done\n");
    // ---- hc_combine (attn side): res = res + repeat(attn_out) * (2*sigmoid(inject/hc)) ----
    hc_combine(res_in_out, attn_out.data(), inject.data(), hc, n_embd);

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  gdn comb1 done\n");
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        int mn = 0; for (int64_t i = 0; i < n_embd; i++) if (std::isnan(moe_out[i])) mn++;
        int xn2 = 0; for (int64_t i = 0; i < hc*n_embd; i++) if (std::isnan(xn[i])) xn2++;
        fprintf(stderr, "  gdn moe1 out nan=%d xn(attn-side) nan=%d\n", mn, xn2);
    }
    // ---- hc_mix (ffn side) + MoE again ----
    hc_rms_norm_gamma(res_in_out, L.hc_ffn_norm, xn.data(), n_embd, hc, eps);
    hc_mix(L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, hc, xn.data(), mixed.data(), inject.data(), n_threads);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && il == 12) {
        FILE * f = fopen("/tmp/qwen4exp-builds/f_gdn_ffnmixed_12.bin", "wb");
        if (f) { fwrite(mixed.data(), 4, n_embd, f); fclose(f); }
        FILE * f2 = fopen("/tmp/qwen4exp-builds/f_gdn_ffnxn_12.bin", "wb");
        if (f2) { fwrite(xn.data(), 4, hc * n_embd, f2); fclose(f2); }
    }
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        int mm = 0; for (int64_t i = 0; i < n_embd; i++) if (std::isnan(mixed[i])) mm++;
        fprintf(stderr, "  gdn ffn mixed nan=%d\n", mm);
    }
    fused_moe(L, hp, mixed.data(), moe_out.data(), n_threads);
    hc_combine(res_in_out, moe_out.data(), inject.data(), hc, n_embd);

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  gdn moe2 done\n");
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        int rn = 0, mn = 0, in = 0;
        for (int64_t i = 0; i < hc*n_embd; i++) { if (std::isnan(res_in_out[i])) rn++; }
        for (int64_t i = 0; i < n_embd; i++) { if (std::isnan(moe_out[i])) mn++; }
        for (int64_t i = 0; i < hc; i++) { if (std::isnan(inject[i])) in++; }
        fprintf(stderr, "  gdn final: res nan=%d moe_out nan=%d inject nan=%d\n", rn, mn, in);
    }
    memcpy(out, res_in_out, n_embd * sizeof(float));
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        char fconv[128];
        snprintf(fconv, sizeof(fconv), "/tmp/qwen4exp-builds/f_conv_%d.bin", il);
        FILE * fc = fopen(fconv, "wb");
        if (fc) {
            fwrite(qkv.data(), 4, qkv_span, fc);
            fwrite(window.data(), 4, (d_conv) * n_ch, fc);
            fwrite(conv_out.data(), 4, n_ch, fc);
            fclose(fc);
        }
        char fnd[128];
        snprintf(fnd, sizeof(fnd), "/tmp/qwen4exp-builds/f_nodes_%d.bin", il);
        FILE * f = fopen(fnd, "wb");
        if (f) {
            auto wr = [&](const char * nm, const float * d, size_t n) {
                size_t tag = strlen(nm);
                fwrite(&tag, 4, 1, f);
                fwrite(nm, 1, tag, f);
                uint32_t pad = (4 - (tag % 4)) % 4;
                for (uint32_t z = 0; z < pad; z++) fputc(0, f);
                fwrite(d, 4, n, f);
            };
            wr("conv_output_silu", conv_out.data(), n_ch);
            wr("final_output", final_in.data(), S_v * H_v);
            wr("linear_attn_out", attn_out.data(), n_embd);
            wr("ffn_out", moe_out.data(), n_embd);
            wr("hc_combine", res_in_out, hc * n_embd);
            fclose(f);
        }
    }
}


// ---- the head + the decode loop --------------------------------------------

// the final hc_mix (the output norm; no inject) + the output projection
static void fused_head(const struct llama_model_qwen4exp & model,
                       const struct llama_hparams & hp,
                       const float * res_hc, float * logits, int n_threads) {
    const int64_t hc = hp.dsv4_hc_mult;
    const int64_t n_embd = hp.n_embd;
    const float eps = hp.f_norm_rms_eps;

    std::vector<float> xn(hc * n_embd), mixed(n_embd);

    hc_rms_norm_gamma(res_hc, model.hc_head_norm, xn.data(), n_embd, hc, eps);
    hc_mix(model.hc_head_down, model.hc_head_up, nullptr, hc, xn.data(), mixed.data(), nullptr, n_threads);
    lora_mm(model.output, mixed.data(), model.output_s, logits, n_threads);
}

// (the INF-67 `fused_decode_token` skeleton lived here; it was never called —
//  llama_model_qwen4exp::fused_decode below is the live entry point. Removed in
//  INF-70 A3.)

// ---- the fused full-attention layer ----------------------------------------

extern "C" {
void ggml_compute_forward_rope(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_flash_attn_ext(const struct ggml_compute_params * params, struct ggml_tensor * dst);
}

// the interleaved mrope parameters (the graph context's rope members, which
// come from the llama_context_params in llm_graph_context's ctor)
struct FusedRopeParams {
    int n_ctx_orig;
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
};

// apply the interleaved mrope via the graph's own rope tensor + kernel. x is the
// flat [n_head][n_embd_head] vector; the rope stages it as the graph's
// [n_embd_head, n_head, n_stream] tensor (the same bytes) and rotates the first
// n_rot dims of each head.
static void fused_rope(const FusedRopeParams & rp, float * x, int64_t n_rot,
                       int64_t n_embd_head, int64_t n_head, int32_t pos, int * sections,
                       const int64_t n_stream) {
    // A2: the arena (this was a 16 MB mmap per call, and the QSA block path calls
    // it once per pooled block per index head)
    const size_t rope_need = (size_t) (2 * n_embd_head * n_head * n_stream) * sizeof(float)
                           + (size_t) (4 * n_stream) * sizeof(int32_t)
                           + 8 * ggml_tensor_overhead() + (1u << 14);
    ggml_context * gctx = fused_arena_init(FUSED_ARENA_ROPE, rope_need, 16u << 20);
    ggml_tensor * a = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_embd_head, n_head, n_stream);
    memcpy(a->data, x, n_embd_head * n_head * n_stream * sizeof(float));
    // IMROPE: 4 positions per token (text tokens: [pos, pos, pos, 0])
    ggml_tensor * b = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 4 * n_stream);
    for (int64_t s = 0; s < n_stream; s++) {
        ((int32_t *) b->data)[0 * n_stream + s] = pos;
        ((int32_t *) b->data)[1 * n_stream + s] = pos;
        ((int32_t *) b->data)[2 * n_stream + s] = pos;
        ((int32_t *) b->data)[3 * n_stream + s] = 0;
    }
    ggml_tensor * rope = ggml_rope_multi(gctx, a, b, nullptr,
            (int) n_rot, sections, LLAMA_ROPE_TYPE_IMROPE,
            rp.n_ctx_orig, rp.freq_base, rp.freq_scale,
            rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
    // the rope kernel uses wdata for the mrope cache (ne[0] floats per thread)
    std::vector<float> rw(n_embd_head);
    struct ggml_compute_params gparams;
    gparams.ith = 0; gparams.nth = 1;
    gparams.wsize = rw.size() * sizeof(float);
    gparams.wdata = rw.data();
    gparams.threadpool = nullptr; gparams.use_ref = false;
    ggml_compute_forward_rope(&gparams, rope);
    memcpy(x, rope->data, n_embd_head * n_head * n_stream * sizeof(float));
    ggml_free(gctx);
}

// ---- the fused full-attention layer ----------------------------------------

// the QSA-masked attention for one token via the graph's own flash kernel.
// Bit-exact path: builds the flash_attn_ext op on scratch tensors and runs the
// graph's kernel on it (same kernel, same inputs => same bits).
static void fused_attn_flash(
        const float * q, const float * k, const float * v,
        const int64_t n_embd_head, const int64_t n_head, const int64_t n_head_kv,
        const int64_t n_kv, const float * selected_cells, float * out) {
    // A2: the arena, sized from n_kv (the K/V F16 stages dominate)
    const size_t fa_need = (size_t) (n_embd_head * n_head) * sizeof(float) * 2
                         + (size_t) (2 * n_embd_head * n_head_kv * n_kv) * sizeof(uint16_t)
                         + (size_t) n_kv * sizeof(uint16_t)
                         + 16 * ggml_tensor_overhead() + (1u << 16);
    ggml_context * gctx = fused_arena_init(FUSED_ARENA_ATTN, fa_need, 64u << 20);
    // the graph casts the F32 activations to F16 for the flash and permutes the
    // heads/kv dims (build_attn_mha) before the op; mirror that exactly. The Q
    // stays F32 (the CPU kernel reads Q as const float* unconditionally — the
    // earlier F16 Q staging was the NaN source, 4096 = 16 heads x 256).
    ggml_tensor * tq = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n_embd_head, n_head, 1, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, n_embd_head, n_head_kv, n_kv, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, n_embd_head, n_head_kv, n_kv, 1);
    ggml_tensor * tm = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, n_kv, 1, 1, 1);
    memcpy(tq->data, q, n_embd_head * n_head * sizeof(float));
    for (int64_t i = 0; i < n_embd_head * n_head_kv * n_kv; i++) ((ggml_fp16_t *) tk->data)[i] = ggml_fp32_to_fp16(k[i]);
    for (int64_t i = 0; i < n_embd_head * n_head_kv * n_kv; i++) ((ggml_fp16_t *) tv->data)[i] = ggml_fp32_to_fp16(v[i]);
    for (int64_t j = 0; j < n_kv; j++) {
        ((ggml_fp16_t *) tm->data)[j] = selected_cells[j] == 0.0f
            ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
    }
    ggml_tensor * pq = ggml_permute(gctx, tq, 0, 2, 1, 3);
    ggml_tensor * pk = ggml_permute(gctx, tk, 0, 2, 1, 3);
    ggml_tensor * pv = ggml_permute(gctx, tv, 0, 2, 1, 3);
    const float scale = 1.0f / sqrtf((float) n_embd_head);
    ggml_tensor * tfa = ggml_flash_attn_ext(gctx, pq, pk, pv, tm, scale, 0.0f, 0.0f);
    // the flash kernel's decode-path scratch: per-thread Q/V copies (DK + 2*DV +
    // CACHE_LINE_SIZE_F32) plus the [q_head][kv_chunk][M, S, VKQ] partials
    // (flash_attn_ext_f16: partials_base = wdata + nth*(DK + 2*DV + 16))
    const int64_t fw = 16 + (n_embd_head + 2 * n_embd_head + 16) + n_head * (2 + n_embd_head);
    std::vector<float> fw_buf(fw);
    static struct ggml_threadpool * fa_tp = nullptr;
    if (fa_tp == nullptr) {
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
        tpp.n_threads = 1;
        fa_tp = ggml_threadpool_new(&tpp);
    }
    struct ggml_compute_params gparams;
    gparams.ith = 0; gparams.nth = 1;
    gparams.wsize = fw_buf.size() * sizeof(float);
    gparams.wdata = fw_buf.data();
    gparams.threadpool = fa_tp; gparams.use_ref = false;
    ggml_compute_forward_flash_attn_ext(&gparams, tfa);
    memcpy(out, tfa->data, n_embd_head * n_head * sizeof(float));
    ggml_free(gctx);
}

// the QSA-masked attention for one token via the graph's own flash kernel.
//   q/k/v : [n_embd_head, n_head(_kv), 1] rotated + roped (q/k) and raw (v)
//   k_all/v_all : the attention cache slices [n_embd_head, n_head_kv, n_kv]
//   selected : the per-cell visibility (0 visible, -inf blocked), F16 [n_kv]
//   kq_scale : 1/sqrt(n_embd_head)
static void fused_attn_qsa(
        const struct llama_hparams & hp,
        const float * q, const float * k, const float * v,
        const float * k_all, const float * v_all, const int n_kv,
        const float * selected_cells, float * out, int n_head, int n_head_kv,
        const int n_threads) {
    (void) n_threads;
    const int64_t n_embd_head = hp.n_embd_head_k();
    const float kq_scale = hp.f_attention_scale == 0.0f ? 1.0f / sqrtf((float) n_embd_head) : hp.f_attention_scale;
    const int n_repeat = n_head / n_head_kv;
    const int64_t k_cell = n_embd_head * n_head_kv;

    // the manual attention mirroring the graph's MHA: every query head scores
    // its own distribution against its kv head's cells. k_all/v_all are the
    // flat staged caches in the measured layout: element (dim i, kv head h,
    // cell j) at i + h*n_embd_head + j*k_cell.
    std::vector<float> scores(n_kv);
    std::vector<float> out_h(n_embd_head * n_head, 0.0f);

    for (int h = 0; h < n_head; h++) {
        const int hk = h / n_repeat;
        // max over the attended cells for the numeric stability
        float mx = -INFINITY;
        for (int j = 0; j < n_kv; j++) {
            if (selected_cells[j] == 0.0f) {
                float s = 0.0f;
                for (int i = 0; i < n_embd_head; i++) {
                    s += q[h * n_embd_head + i] * k_all[(size_t) i + (size_t) hk * n_embd_head + (size_t) j * k_cell];
                }
                scores[j] = s * kq_scale;
                mx = fmaxf(mx, scores[j]);
            }
        }
        double sum = 0.0;
        for (int j = 0; j < n_kv; j++) {
            scores[j] = selected_cells[j] == 0.0f ? expf(scores[j] - mx) : 0.0f;
            sum += scores[j];
        }
        const float inv = (float) (1.0 / sum);
        for (int j = 0; j < n_kv; j++) {
            scores[j] *= inv;
        }
        for (int i = 0; i < n_embd_head; i++) {
            float acc = 0.0f;
            for (int j = 0; j < n_kv; j++) {
                acc += scores[j] * v_all[(size_t) i + (size_t) hk * n_embd_head + (size_t) j * k_cell];
            }
            out_h[(size_t) h * n_embd_head + i] = acc;
        }
    }
    memcpy(out, out_h.data(), n_embd_head * n_head * sizeof(float));
    (void) k; (void) v;
}

// the fused full-attention layer for the single-token decode:
//   x/res_in_out : the hc-wide input / residual stream
//   pos          : the current token position
//   tk/tv/ti     : the attention/indexer cache views (written in place at the
//                  current cell; read for the attention over the used cells)
//   n_used       : the used attention cells (incl. the current token's cell)
//   n_visible    : the cells this token may attend to (pos < pos)
//   idx_n_used   : the used indexer cells
// Returns false when a cache type is unsupported (falls back to the graph).
bool fused_full_attn_layer(
        const struct llama_layer & L,
        const struct llama_hparams & hp,
        const FusedRopeParams & rp,
        int32_t pos,
        const float * x, float * res_in_out, float * out,
        struct ggml_tensor * tk, struct ggml_tensor * tv, struct ggml_tensor * ti,
        int n_used, int n_visible, int idx_n_used,
        int n_threads, int il) {

    const int64_t hc = hp.dsv4_hc_mult;
    const int64_t n_embd = hp.n_embd;
    const int64_t n_embd_head = hp.n_embd_head_k(); // 256
    const int64_t n_head = hp.n_head();             // 24
    const int64_t n_head_kv = hp.n_head_kv();       // 2
    const float eps = hp.f_norm_rms_eps;
    const int n_rot = (int) hp.n_rot();

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && il == 3) {
        fprintf(stderr, "  attn hparams: f_attention_scale=%.6f n_embd_head=%lld n_head=%lld n_head_kv=%lld n_rot=%d indexer_top_k=%d\n",
                (double) hp.f_attention_scale, (long long) n_embd_head, (long long) n_head, (long long) n_head_kv, n_rot, (int) hp.indexer_top_k);
    }

    if (tk->type != GGML_TYPE_F32 && tk->type != GGML_TYPE_F16) {
        LLAMA_LOG_INFO("%s: fused attn declined: cache types %s/%s/%s\n", __func__,
                ggml_type_name(tk->type), ggml_type_name(tv->type), ggml_type_name(ti->type));
        return false;
    }
    // the cache may be F16 (the graph dequantizes it for the attention; the
    // fused path stages a converted F32 view and casts back on the writes)
    const bool k16 = tk->type == GGML_TYPE_F16;
    const bool i16 = ti->type == GGML_TYPE_F16;
    const int64_t k_cell = (int64_t) n_embd_head * n_head_kv;
    // the k/v cache stage is built AFTER the current cell's write below, so the
    // attention sees the freshly written k/v (the graph's cpy_k/cpy_v precede the
    // attention in the same graph; a pre-write stage would hold the stale cell).
    // The indexer stage is independent and built here.
    std::vector<float> k_stage, v_stage, idx_stage;
    const float * k_cache, * v_cache, * idx_cache;
    if (i16) {
        idx_stage.resize(hp.indexer_head_size * idx_n_used);
        for (int64_t i = 0; i < hp.indexer_head_size * idx_n_used; i++) idx_stage[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) ti->data)[i]);
        idx_cache = idx_stage.data();
    } else {
        idx_cache = (const float *) ti->data;
    }
    auto write_cache_f32 = [](void * dst, bool is16, const float * src, size_t n) {
        if (is16) {
            for (size_t i = 0; i < n; i++) ((ggml_fp16_t *) dst)[i] = ggml_fp32_to_fp16(src[i]);
        } else {
            memcpy(dst, src, n * sizeof(float));
        }
    };

    std::vector<float> xn(hc * n_embd), mixed(n_embd), inject(hc * n_embd);
    hc_rms_norm_gamma(x, L.hc_attn_norm, xn.data(), n_embd, hc, eps);
    hc_mix(L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, hc, xn.data(), mixed.data(), inject.data(), n_threads);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        char fn1[128], fn2[128], fn3[128];
        snprintf(fn1, sizeof(fn1), "/tmp/qwen4exp-builds/f_attn_mixed_%d.bin", il);
        snprintf(fn2, sizeof(fn2), "/tmp/qwen4exp-builds/f_attn_xn_%d.bin", il);
        snprintf(fn3, sizeof(fn3), "/tmp/qwen4exp-builds/f_attn_inject_%d.bin", il);
        FILE * f = fopen(fn1, "wb"); if (f) { fwrite(mixed.data(), 4, n_embd, f); fclose(f); }
        FILE * f2 = fopen(fn2, "wb"); if (f2) { fwrite(xn.data(), 4, hc * n_embd, f2); fclose(f2); }
        FILE * f3 = fopen(fn3, "wb"); if (f3) { fwrite(inject.data(), 4, hc, f3); fclose(f3); }
    }

    // the interleaved mrope sections (the graph's rope_sections)
    int sections[4] = { (int) hp.rope_sections[0], (int) hp.rope_sections[1], (int) hp.rope_sections[2], (int) hp.rope_sections[3] };

    // ---- QSA indexer (short-context dense fallback: n_visible <= width) ----
    const int64_t r = hp.dsv4_compress_ratios[il] ? hp.dsv4_compress_ratios[il] : 4;
    const int64_t width = std::min<int64_t>(idx_n_used, (int64_t) hp.indexer_top_k + r - 1);
    const bool dense = n_visible <= width;

    std::vector<float> k_raw(hp.indexer_head_size);
    lora_mm(L.index_k_proj, mixed.data(), nullptr, k_raw.data(), n_threads);
    if (idx_n_used > 0) {
        // write the new key at the current cell (the last used one)
        write_cache_f32((char *) ti->data + (size_t) (idx_n_used - 1) * hp.indexer_head_size * ggml_type_size(ti->type),
                        i16, k_raw.data(), hp.indexer_head_size);
    }

    std::vector<float> selected(n_used, -INFINITY);
    if (dense) {
        // all used cells attend, including the current token's own cell (the graph's
        // causal mask at the decode covers every cell up to and including the current)
        for (int64_t j = 0; j < n_used; j++) selected[j] = 0.0f;
    } else {
        // pooled block scores (mean over the r members, norm, rope, query, relu-sum)
        const int64_t n_blocks = (n_visible + r - 1) / r;
        const int64_t idx_dim = hp.indexer_head_size;
        const int64_t n_idx_h = hp.indexer_n_head;
        std::vector<float> pooled(idx_dim * n_blocks), pooled_n(idx_dim * n_blocks);
        for (int64_t b = 0; b < n_blocks; b++) {
            for (int64_t i = 0; i < idx_dim; i++) {
                double s = 0.0;
                for (int64_t m = 0; m < r; m++) {
                    const int64_t cell = b * r + m;
                    if (cell < n_visible) s += idx_cache[(size_t) cell * idx_dim + i];
                }
                pooled[(size_t) b * idx_dim + i] = (float) (s / r);
            }
            // rms norm over the block key (the index_k_norm gamma folded to 1+w)
            const float * pk = pooled.data() + (size_t) b * idx_dim;
            const float * wn = (const float *) L.index_k_norm->data;
            double ss = 0.0;
            for (int64_t i = 0; i < idx_dim; i++) ss += (double) (pk[i] * pk[i]);
            const float sc = 1.0f / sqrtf((float) (ss / idx_dim) + eps);
            for (int64_t i = 0; i < idx_dim; i++) pooled_n[(size_t) b * idx_dim + i] = pk[i] * sc * wn[i];
        }
        // the block rope (position = b*r) + the query + the query rope
        std::vector<float> blk_score(n_blocks);
        {
            // the pooled blocks rotate at their own block positions (the graph's blk_pos)
            std::vector<float> pooled_rope(idx_dim * n_blocks);
            for (int64_t b = 0; b < n_blocks; b++) {
                const int64_t bp = b * r;
                memcpy(pooled_rope.data() + b * idx_dim, pooled_n.data() + b * idx_dim, idx_dim * sizeof(float));
                fused_rope(rp, pooled_rope.data() + b * idx_dim, n_rot, idx_dim, 1, (int32_t) bp, sections, 1);
            }
            std::vector<float> q_idx(n_idx_h * idx_dim);
            lora_mm(L.index_q_proj, mixed.data(), nullptr, q_idx.data(), n_threads);
            {
                const float * wn = (const float *) L.index_q_norm->data;
                for (int64_t h = 0; h < n_idx_h; h++) {
                    const float * qh = q_idx.data() + h * idx_dim;
                    double ss = 0.0;
                    for (int64_t i = 0; i < idx_dim; i++) ss += (double) (qh[i] * qh[i]);
                    const float sc = 1.0f / sqrtf((float) (ss / idx_dim) + eps);
                    for (int64_t i = 0; i < idx_dim; i++) q_idx[h * idx_dim + i] = qh[i] * sc * wn[i];
                    // the query rotates at the current position (the graph's inp_pos)
                    fused_rope(rp, q_idx.data() + h * idx_dim, n_rot, idx_dim, 1, (int32_t) pos, sections, 1);
                }
            }
            // the rectified scores: sum over the heads of relu(pooled dot q)
            for (int64_t b = 0; b < n_blocks; b++) {
                float s = 0.0f;
                for (int64_t h = 0; h < n_idx_h; h++) {
                    float d = 0.0f;
                    for (int64_t i = 0; i < idx_dim; i++) {
                        d += pooled_rope[(size_t) b * idx_dim + i] * q_idx[(size_t) h * idx_dim + i];
                    }
                    s += fmaxf(d, 0.0f);
                }
                blk_score[b] = s;
            }
        }
        // pick the width best blocks (the tail cells of the last blocks follow)
        std::vector<int> picked;
        {
            std::vector<float> sc = blk_score;
            for (int64_t k = 0; k < width; k++) {
                int best = -1; float bestv = -INFINITY;
                for (int64_t b = 0; b < n_blocks; b++) {
                    if (sc[b] > bestv) { bestv = sc[b]; best = (int) b; }
                }
                if (best < 0) break;
                picked.push_back(best);
                sc[best] = -INFINITY;
            }
        }
        for (int64_t j = 0; j < n_visible; j++) {
            const int64_t b = j / r;
            bool ok = false;
            for (size_t k = 0; k < picked.size(); k++) if (picked[k] == (int) b) { ok = true; break; }
            if (ok) selected[j] = 0.0f;
        }
    }

    // ---- the q/k/v projections + norms + rope ----
    std::vector<float> qfull(hp.n_embd_head_v() * 2 * n_head);
    lora_mm(L.wq, mixed.data(), L.wq_s, qfull.data(), n_threads);
    const int64_t hdim2 = n_embd_head * n_head;
    std::vector<float> q(hdim2), gate(hdim2);
    {
        const float * qn = (const float *) L.attn_q_norm->data;
        for (int64_t h = 0; h < n_head; h++) {
            const float * qh = qfull.data() + h * 2 * n_embd_head;
            double ss = 0.0;
            for (int64_t i = 0; i < n_embd_head; i++) ss += (double) (qh[i] * qh[i]);
            const float sc = 1.0f / sqrtf((float) (ss / n_embd_head) + eps);
            for (int64_t i = 0; i < n_embd_head; i++) q[h * n_embd_head + i] = qh[i] * sc * qn[i];
            memcpy(gate.data() + h * n_embd_head, qh + n_embd_head, n_embd_head * sizeof(float));
        }
    }
    std::vector<float> k(n_embd_head * n_head_kv), v(n_embd_head * n_head_kv);
    {
        std::vector<float> kraw(n_embd_head * n_head_kv), vraw(n_embd_head * n_head_kv);
        lora_mm(L.wk, mixed.data(), L.wk_s, kraw.data(), n_threads);
        lora_mm(L.wv, mixed.data(), L.wv_s, vraw.data(), n_threads);
        const float * kn = (const float *) L.attn_k_norm->data;
        for (int64_t h = 0; h < n_head_kv; h++) {
            double ss = 0.0;
            for (int64_t i = 0; i < n_embd_head; i++) ss += (double) (kraw[h * n_embd_head + i] * kraw[h * n_embd_head + i]);
            const float sc = 1.0f / sqrtf((float) (ss / n_embd_head) + eps);
            for (int64_t i = 0; i < n_embd_head; i++) {
                k[h * n_embd_head + i] = kraw[h * n_embd_head + i] * sc * kn[i];
                v[h * n_embd_head + i] = vraw[h * n_embd_head + i];
            }
        }
    }

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        FILE * fq = fopen("/tmp/qwen4exp-builds/f_attn_qpre_3.bin", "wb");
        if (fq) { fwrite(q.data(), 4, hdim2, fq); fclose(fq); }
        FILE * fk = fopen("/tmp/qwen4exp-builds/f_attn_kpre_3.bin", "wb");
        if (fk) { fwrite(k.data(), 4, n_embd_head * n_head_kv, fk); fclose(fk); }
    }
    fused_rope(rp, q.data(), n_rot, n_embd_head, n_head, pos, sections, 1);
    fused_rope(rp, k.data(), n_rot, n_embd_head, n_head_kv, pos, sections, 1);

    // ---- KV write + the attention ----
    if (n_used > 0) {
        // write the new k/v at the current cell (the last used one)
        for (int64_t h = 0; h < n_head_kv; h++) {
            write_cache_f32((char *) tk->data + ((size_t) (n_used - 1) * k_cell + (size_t) h * n_embd_head) * ggml_type_size(tk->type),
                            k16, k.data() + h * n_embd_head, n_embd_head);
            write_cache_f32((char *) tv->data + ((size_t) (n_used - 1) * k_cell + (size_t) h * n_embd_head) * ggml_type_size(tv->type),
                            k16, v.data() + h * n_embd_head, n_embd_head);
        }
    }
    // the k/v stage for the attention: built after the write (see above)
    if (k16) {
        k_stage.resize(k_cell * n_used);
        for (int64_t i = 0; i < k_cell * n_used; i++) k_stage[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) tk->data)[i]);
        v_stage.resize(k_cell * n_used);
        for (int64_t i = 0; i < k_cell * n_used; i++) v_stage[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) tv->data)[i]);
        k_cache = k_stage.data();
        v_cache = v_stage.data();
    } else {
        k_cache = (const float *) tk->data;
        v_cache = (const float *) tv->data;
    }

    std::vector<float> attn_out(hdim2);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        fprintf(stderr, "  attn: n_used=%d n_visible=%d q[0..2]=%.6g %.6g %.6g\n", n_used, n_visible,
                (double) q[0], (double) q[1], (double) q[2]);
        char fnq[128], fnk[128], fnv[128], fng[128];
        snprintf(fnq, sizeof(fnq), "/tmp/qwen4exp-builds/f_attn_q_%d.bin", il);
        snprintf(fnk, sizeof(fnk), "/tmp/qwen4exp-builds/f_attn_k_%d.bin", il);
        snprintf(fnv, sizeof(fnv), "/tmp/qwen4exp-builds/f_attn_v_%d.bin", il);
        snprintf(fng, sizeof(fng), "/tmp/qwen4exp-builds/f_attn_gate_%d.bin", il);
        FILE * fq = fopen(fnq, "wb"); if (fq) { fwrite(q.data(), 4, hdim2, fq); fclose(fq); }
        FILE * fk = fopen(fnk, "wb"); if (fk) { fwrite(k.data(), 4, n_embd_head * n_head_kv, fk); fclose(fk); }
        FILE * fv = fopen(fnv, "wb"); if (fv) { fwrite(v.data(), 4, n_embd_head * n_head_kv, fv); fclose(fv); }
        FILE * fg = fopen(fng, "wb"); if (fg) { fwrite(gate.data(), 4, hdim2, fg); fclose(fg); }
    }
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        char fqnm[128];
        snprintf(fqnm, sizeof(fqnm), "/tmp/qwen4exp-builds/f_attn_q_in_%d.bin", il);
        FILE * fq = fopen(fqnm, "wb");
        if (fq) { fwrite(q.data(), 4, hdim2, fq); fclose(fq); }
    }
    if (n_visible > 0) {
        std::vector<float> k_all(n_embd_head * n_head_kv * n_used), v_all(n_embd_head * n_head_kv * n_used);
        memcpy(k_all.data(), k_cache, n_embd_head * n_head_kv * n_used * sizeof(float));
        memcpy(v_all.data(), v_cache, n_embd_head * n_head_kv * n_used * sizeof(float));
        // a fused-side reference of the first head's scores (head 0, kv 0)
        if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && il == 3) {
            fprintf(stderr, "  attn ref h0: k_all[0..3]=%.6g %.6g %.6g %.6g k_all[512..515]=%.6g %.6g %.6g %.6g\n",
                    (double) k_all[0], (double) k_all[1], (double) k_all[2], (double) k_all[3],
                    (double) k_all[512], (double) k_all[513], (double) k_all[514], (double) k_all[515]);
        }
        if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
            char fk[128], fv[128], fii[128];
            snprintf(fk, sizeof(fk), "/tmp/qwen4exp-builds/f_attn_kall_%d.bin", il);
            snprintf(fv, sizeof(fv), "/tmp/qwen4exp-builds/f_attn_vall_%d.bin", il);
            snprintf(fii, sizeof(fii), "/tmp/qwen4exp-builds/f_attn_info_%d.txt", il);
            FILE * fa = fopen(fk, "wb");
            if (fa) { fwrite(k_all.data(), 4, k_all.size(), fa); fclose(fa); }
            FILE * fb = fopen(fv, "wb");
            if (fb) { fwrite(v_all.data(), 4, v_all.size(), fb); fclose(fb); }
            FILE * fi = fopen(fii, "w");
            if (fi) { fprintf(fi, "%lld %lld %lld %d\n", (long long) n_embd_head, (long long) n_head, (long long) n_head_kv, n_used); fclose(fi); }
        }
        // the QSA-masked attention via the graph's flash kernel (bit-exact)
        if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
            int kn = 0, vn = 0;
            for (int64_t i = 0; i < (int64_t) k_all.size(); i++) { if (std::isnan(k_all[i])) kn++; if (std::isnan(v_all[i])) vn++; }
            fprintf(stderr, "  attn flash in: k nan=%d v nan=%d k[0..2]=%.6g %.6g %.6g v[0..2]=%.6g %.6g %.6g mask[0..3]=%.6g %.6g %.6g %.6g\n",
                    kn, vn, (double) k_all[0], (double) k_all[1], (double) k_all[2],
                    (double) v_all[0], (double) v_all[1], (double) v_all[2],
                    (double) selected[0], (double) selected[1], (double) selected[2], (double) selected[3]);
        }
        if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
            FILE * fq = fopen("/tmp/qwen4exp-builds/flash_q.bin", "wb");
            fwrite(q.data(), 4, n_embd_head * n_head, fq); fclose(fq);
            FILE * fk = fopen("/tmp/qwen4exp-builds/flash_k.bin", "wb");
            fwrite(k_all.data(), 4, n_embd_head * n_head_kv * n_used, fk); fclose(fk);
            FILE * fv = fopen("/tmp/qwen4exp-builds/flash_v.bin", "wb");
            fwrite(v_all.data(), 4, n_embd_head * n_head_kv * n_used, fv); fclose(fv);
            FILE * fm = fopen("/tmp/qwen4exp-builds/flash_m.bin", "wb");
            fwrite(selected.data(), 4, n_used, fm); fclose(fm);
            FILE * fi = fopen("/tmp/qwen4exp-builds/flash_info.txt", "w");
            fprintf(fi, "%lld %lld %lld %lld\n", (long long) n_embd_head, (long long) n_head, (long long) n_head_kv, (long long) n_used);
            fclose(fi);
            fprintf(stderr, "  attn flash dumped\n");
        }
        // the graph runs the non-flash MHA (flash_attn=false), and the flash
        // kernel is NaN on this model's real activations; the manual path
        // mirrors the graph's math (NMSE-ok)
        fused_attn_flash(q.data(), k_all.data(), v_all.data(),
                         n_embd_head, n_head, n_head_kv, n_used,
                         selected.data(), attn_out.data());
    }

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        int ann = 0; for (int64_t i = 0; i < hdim2; i++) if (std::isnan(attn_out[i])) ann++;
        fprintf(stderr, "  attn out: nan=%d [0..2]=%.6g %.6g %.6g\n", ann, (double) attn_out[0], (double) attn_out[1], (double) attn_out[2]);
    }
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        char fna[128], fns[128];
        snprintf(fna, sizeof(fna), "/tmp/qwen4exp-builds/f_attn_out_pre_%d.bin", il);
        snprintf(fns, sizeof(fns), "/tmp/qwen4exp-builds/f_attn_selected_%d.bin", il);
        FILE * fa = fopen(fna, "wb"); if (fa) { fwrite(attn_out.data(), 4, hdim2, fa); fclose(fa); }
        FILE * fs = fopen(fns, "wb"); if (fs) { fwrite(selected.data(), 4, n_used, fs); fclose(fs); }
    }
    // the gate + the output projection
    for (int64_t i = 0; i < hdim2; i++) {
        const float g = gate[i];
        attn_out[i] *= 1.0f / (1.0f + expf(-g));
    }
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        char fng[128];
        snprintf(fng, sizeof(fng), "/tmp/qwen4exp-builds/f_attn_gated_%d.bin", il);
        FILE * fg2 = fopen(fng, "wb"); if (fg2) { fwrite(attn_out.data(), 4, hdim2, fg2); fclose(fg2); }
    }
    std::vector<float> layer_out(n_embd);
    lora_mm(L.wo, attn_out.data(), L.wo_s, layer_out.data(), n_threads);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        char fno[128];
        snprintf(fno, sizeof(fno), "/tmp/qwen4exp-builds/f_attn_out_%d.bin", il);
        FILE * fa = fopen(fno, "wb"); if (fa) { fwrite(layer_out.data(), 4, n_embd, fa); fclose(fa); }
        char fgm[128];
        snprintf(fgm, sizeof(fgm), "/tmp/qwen4exp-builds/f_attn_gsig_%d.bin", il);
        FILE * fgs = fopen(fgm, "wb"); if (fgs) {
            for (int64_t i = 0; i < hdim2; i++) {
                const float g = gate[i];
                const float sg = 1.0f / (1.0f + expf(-g));
                fwrite(&sg, 4, 1, fgs);
            }
            fclose(fgs);
        }
    }

    // the MoE + the hc combine (same as the GDN layers)
    std::vector<float> moe_out(n_embd);
    fused_moe(L, hp, layer_out.data(), moe_out.data(), n_threads);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        char fnm[128];
        snprintf(fnm, sizeof(fnm), "/tmp/qwen4exp-builds/f_attn_moe1_%d.bin", il);
        FILE * fm = fopen(fnm, "wb"); if (fm) { fwrite(moe_out.data(), 4, n_embd, fm); fclose(fm); }
    }
    hc_combine(res_in_out, layer_out.data(), inject.data(), hc, n_embd);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && il == 7) {
        FILE * f = fopen("/tmp/qwen4exp-builds/f_attn_res7.bin", "wb");
        if (f) { fwrite(res_in_out, 4, hc * n_embd, f); fclose(f); }
    }
    hc_rms_norm_gamma(res_in_out, L.hc_ffn_norm, xn.data(), n_embd, hc, eps);
    hc_mix(L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, hc, xn.data(), mixed.data(), inject.data(), n_threads);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        char fnm2[128], fnm3[128];
        snprintf(fnm2, sizeof(fnm2), "/tmp/qwen4exp-builds/f_attn_ffnmixed_%d.bin", il);
        FILE * fm2 = fopen(fnm2, "wb"); if (fm2) { fwrite(mixed.data(), 4, n_embd, fm2); fclose(fm2); }
        snprintf(fnm3, sizeof(fnm3), "/tmp/qwen4exp-builds/f_attn_ffnxn_%d.bin", il);
        FILE * fm3 = fopen(fnm3, "wb"); if (fm3) { fwrite(xn.data(), 4, hc * n_embd, fm3); fclose(fm3); }
    }
    fused_moe(L, hp, mixed.data(), moe_out.data(), n_threads);
    hc_combine(res_in_out, moe_out.data(), inject.data(), hc, n_embd);

    memcpy(out, res_in_out, n_embd * sizeof(float));
    return true;
}

// ---- the fused PLE (n-gram hash embedding) ---------------------------------

// the host-side n-gram hash + the 51GB-table gather + the key/value
// projections + the gate + the dilated depthwise conv, mirroring build_ple.
//   tok : the current token; prev : the n_gram-1 predecessors (oldest first)
//   hidden : the hc-wide input; res : the wide residual (updated in place)
//   ple_conv_state : the [hist, hc_dim] conv history (updated in place)
void fused_ple(
        const struct llama_model_qwen4exp & model,
        const struct llama_hparams & hp,
        const struct llama_layer & L,
        int32_t tok, const int32_t * prev,
        const float * hidden, float * res,
        float * ple_conv_state, int n_threads) {

    const int64_t hc = hp.dsv4_hc_mult;
    const int64_t n_embd = hp.n_embd;
    const int64_t hc_dim = hc * n_embd;
    const float eps = hp.f_norm_rms_eps;

    // ---- the host-side hash (mirrors llm_graph_input_ple::set_input) ----
    const int64_t n_gram = hp.ple_ngram_size;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t n_heads = hp.ple_n_heads;
    const int64_t eos = hp.ple_eos_token_id;
    const int64_t n_prev = n_gram - 1;

    int64_t ctxv[8];
    ctxv[0] = tok;
    bool cut = false;
    for (int64_t s = 1; s < n_gram; s++) {
        const int64_t t = cut ? -1 : prev[n_prev - s];
        cut = cut || t < 0 || t == eos;
        ctxv[s] = cut ? eos : t;
    }
    std::vector<int32_t> rows(n_heads);
    for (int64_t n = 2; n <= n_gram; n++) {
        uint64_t mixed = (uint64_t) ctxv[0] * hp.ple_layer_multipliers[0];
        for (int64_t j = 1; j < n; j++) {
            mixed ^= (uint64_t) ctxv[j] * hp.ple_layer_multipliers[j];
        }
        const int64_t base = (n - 2) * per_gram;
        for (int64_t g = 0; g < per_gram; g++) {
            const int64_t h_i = base + g;
            rows[h_i] = (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
        }
    }

    // ---- the gather from the per-layer token embedding table + the f32 cast
    //      (the table is IQ4_NL in the on-disk file) ----
    const struct ggml_tensor * table = model.per_layer_tok_embd;
    const int64_t head_dim = hp.ple_head_dim;
    std::vector<float> emb(n_heads * head_dim);
    {
        const struct ggml_type_traits * qtt = ggml_get_type_traits(table->type);
        const size_t row_bytes = ggml_row_size(table->type, head_dim);
        if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        fprintf(stderr, "  ple table: name=%s type=%d ne=[%lld,%lld] row_bytes=%zu buf=%s buft=%s to_float=%p\n",
                table->name, (int) table->type, (long long) table->ne[0], (long long) table->ne[1], row_bytes,
                table->buffer ? ggml_backend_buffer_name(table->buffer) : "-",
                table->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(table->buffer)) : "-",
                (const void *) qtt->to_float);
        // dump the first row raw (the dequant reference for the graph's gather)
        const char * r0 = (const char *) table->data + (size_t) rows[0] * row_bytes;
        fprintf(stderr, "  ple row0 raw first 36 bytes: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                (unsigned char) r0[0], (unsigned char) r0[1], (unsigned char) r0[2], (unsigned char) r0[3],
                (unsigned char) r0[4], (unsigned char) r0[5], (unsigned char) r0[6], (unsigned char) r0[7],
                (unsigned char) r0[8], (unsigned char) r0[9], (unsigned char) r0[10], (unsigned char) r0[11],
                (unsigned char) r0[12], (unsigned char) r0[13], (unsigned char) r0[14], (unsigned char) r0[15],
                (unsigned char) r0[16], (unsigned char) r0[17], (unsigned char) r0[18], (unsigned char) r0[19],
                (unsigned char) r0[20], (unsigned char) r0[21], (unsigned char) r0[22], (unsigned char) r0[23],
                (unsigned char) r0[24], (unsigned char) r0[25], (unsigned char) r0[26], (unsigned char) r0[27],
                (unsigned char) r0[28], (unsigned char) r0[29], (unsigned char) r0[30], (unsigned char) r0[31],
                (unsigned char) r0[32], (unsigned char) r0[33], (unsigned char) r0[34], (unsigned char) r0[35]);
        }
        for (int64_t h = 0; h < n_heads; h++) {
            const char * row = (const char *) table->data + (size_t) rows[h] * row_bytes;
            if (table->type == GGML_TYPE_F32) {
                memcpy(emb.data() + h * head_dim, row, head_dim * sizeof(float));
            } else {
                qtt->to_float(row, emb.data() + h * head_dim, head_dim);
            }
        }
    }

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        fprintf(stderr, "  ple rows[0..2]=%d %d %d head_dim=%lld\n", rows[0], rows[1], rows[2], (long long)head_dim);
        fprintf(stderr, "  ple ctxv=[%lld %lld %lld] prev=[%d %d] n_gram=%lld per_gram=%lld n_heads=%lld\n",
                (long long) ctxv[0], (long long) ctxv[1], (long long) ctxv[2],
                (int) prev[0], (int) prev[1], (long long) n_gram, (long long) per_gram, (long long) n_heads);
        FILE * f = fopen("/tmp/qwen4exp-builds/f_ple_rows.bin", "wb");
        if (f) { fwrite(rows.data(), 4, rows.size(), f); fclose(f); }
        FILE * f2 = fopen("/tmp/qwen4exp-builds/f_ple_emb.bin", "wb");
        if (f2) { fwrite(emb.data(), 4, n_heads * head_dim, f2); fclose(f2); }
        int nn = 0; for (int64_t i = 0; i < n_heads*head_dim; i++) if (std::isnan(emb[i])) nn++;
        fprintf(stderr, "  ple emb nan=%d [0..2]=%.6g %.6g %.6g\n", nn, (double)emb[0], (double)emb[1], (double)emb[2]);
    }
    // ---- the key/value projections ----
    std::vector<float> key(hc_dim), value(n_embd);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "  ple w: key type=%s buf=%s extra=%p nb1=%zu nb2=%zu ne=[%lld,%lld,%lld] | val type=%s buf=%s extra=%p\n",
            ggml_type_name(L.ple_key->type), L.ple_key->buffer ? ggml_backend_buffer_name(L.ple_key->buffer) : "-",
            (const void *) L.ple_key->extra, (size_t) L.ple_key->nb[1], (size_t) L.ple_key->nb[2],
            (long long) L.ple_key->ne[0], (long long) L.ple_key->ne[1], (long long) L.ple_key->ne[2],
            ggml_type_name(L.ple_value->type), L.ple_value->buffer ? ggml_backend_buffer_name(L.ple_value->buffer) : "-",
            (const void *) L.ple_value->extra);
    lora_mm(L.ple_key, emb.data(), nullptr, key.data(), n_threads);
    lora_mm(L.ple_value, emb.data(), nullptr, value.data(), n_threads);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        FILE * f = fopen("/tmp/qwen4exp-builds/f_ple_key.bin", "wb");
        if (f) { fwrite(key.data(), 4, key.size(), f); fclose(f); }
        FILE * f2 = fopen("/tmp/qwen4exp-builds/f_ple_val.bin", "wb");
        if (f2) { fwrite(value.data(), 4, value.size(), f2); fclose(f2); }
        fprintf(stderr, "  ple key[0..2]=%.6g %.6g %.6g value[0..2]=%.6g %.6g %.6g\n",
                (double) key[0], (double) key[1], (double) key[2],
                (double) value[0], (double) value[1], (double) value[2]);
    }

    // the grouped norms (the key + the query=hidden) + the gate
    std::vector<float> key_n(hc_dim), query_n(hc_dim);
    {
        const float * kn = (const float *) L.ple_norm_key->data;
        for (int64_t c = 0; c < hc; c++) {
            const float * xc = key.data() + c * n_embd;
            double ss = 0.0;
            for (int64_t i = 0; i < n_embd; i++) ss += (double) (xc[i] * xc[i]);
            const float sc = 1.0f / sqrtf((float) (ss / n_embd) + eps);
            for (int64_t i = 0; i < n_embd; i++) key_n[c * n_embd + i] = xc[i] * sc * kn[c * n_embd + i];
        }
        const float * qn = (const float *) L.ple_norm_query->data;
        for (int64_t c = 0; c < hc; c++) {
            const float * xc = hidden + c * n_embd;
            double ss = 0.0;
            for (int64_t i = 0; i < n_embd; i++) ss += (double) (xc[i] * xc[i]);
            const float sc = 1.0f / sqrtf((float) (ss / n_embd) + eps);
            if (FUSED_DBG("GGML_FUSED_DECODE_TRACE"))
            fprintf(stderr, "  fused query c=%lld: ss/n=%.6e sc=%.6f first=%.6g qn0=%.6g\n",
                    (long long) c, ss / n_embd, (double) sc, (double) (xc[0] * sc * qn[c * n_embd]), (double) qn[c * n_embd]);
            for (int64_t i = 0; i < n_embd; i++) query_n[c * n_embd + i] = xc[i] * sc * qn[c * n_embd + i];
        }
    }

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        FILE * f = fopen("/tmp/qwen4exp-builds/f_ple_keyn.bin", "wb");
        if (f) { fwrite(key_n.data(), 4, key_n.size(), f); fclose(f); }
        FILE * f2 = fopen("/tmp/qwen4exp-builds/f_ple_queryn.bin", "wb");
        if (f2) { fwrite(query_n.data(), 4, query_n.size(), f2); fclose(f2); }
    }

    // the per-stream score + the signed-square-root gate
    std::vector<float> gate(hc);
    {
        const float inv = 1.0f / sqrtf((float) n_embd);
        for (int64_t c = 0; c < hc; c++) {
            float s = 0.0f;
            for (int64_t i = 0; i < n_embd; i++) s += key_n[c * n_embd + i] * query_n[c * n_embd + i];
            s *= inv;
            const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
            gate[c] = 1.0f / (1.0f + expf(-(s >= 0.0f ? mag : -mag)));
        }
    }

    // the gated value + the conv norm
    std::vector<float> gated(hc_dim), norm_conv(hc_dim);
    for (int64_t c = 0; c < hc; c++) {
        for (int64_t i = 0; i < n_embd; i++) gated[c * n_embd + i] = value[i] * gate[c];
    }
    {
        const float * cn = (const float *) L.ple_norm_conv->data;
        for (int64_t c = 0; c < hc; c++) {
            const float * xc = gated.data() + c * n_embd;
            double ss = 0.0;
            for (int64_t i = 0; i < n_embd; i++) ss += (double) (xc[i] * xc[i]);
            const float sc = 1.0f / sqrtf((float) (ss / n_embd) + eps);
            for (int64_t i = 0; i < n_embd; i++) norm_conv[c * n_embd + i] = xc[i] * sc * cn[c * n_embd + i];
        }
    }

    // the dilated depthwise causal conv over the [hist, hc_dim] state:
    // out[c] = sum_k w[k,c] * x[c, hist - (K-1-k)*dil]; the state shifts
    const int64_t kern = hp.ple_conv_kernel;
    const int64_t dil = hp.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;
    // BUGFIX (INF-70): ple_conv1d is F16 on the uniform IQ4_XS artifact
    std::vector<float> ple_cw_tmp;
    const float * w = fused_weights_f32(L.ple_conv1d, ple_cw_tmp);
    std::vector<float> conv_out(hc_dim);
    {
        // the window: state + the current norm_conv, in the graph's channel-major
        // layout: element (tap pos, channel c) at c*(hist+1) + pos; the state row
        // is (channel c, tap j) at c*hist + j (the concat's ne[0]-fastest layout)
        std::vector<float> window((hist + 1) * hc_dim);
        for (int64_t c = 0; c < hc_dim; c++) {
            for (int64_t j = 0; j < hist; j++) {
                window[c * (hist + 1) + j] = ple_conv_state[c * hist + j];
            }
            window[c * (hist + 1) + hist] = norm_conv[c];
        }
        if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
            FILE * f = fopen("/tmp/qwen4exp-builds/f_ple_normconv.bin", "wb");
            if (f) { fwrite(norm_conv.data(), 4, hc_dim, f); fclose(f); }
            FILE * f2 = fopen("/tmp/qwen4exp-builds/f_ple_win.bin", "wb");
            if (f2) { fwrite(window.data(), 4, window.size(), f2); fclose(f2); }
        }
        for (int64_t c = 0; c < hc_dim; c++) {
            float sumf = 0.0f;
            for (int64_t k = 0; k < kern; k++) {
                const int64_t pos = hist - (kern - 1 - k) * dil;
                sumf += window[c * (hist + 1) + pos] * w[(size_t) k + c * 4];
            }
            conv_out[c] = sumf;
        }
        // the graph's SIMD silu (ggml_v_silu / ggml_v_expf), not the scalar expf
        ggml_vec_silu_f32((int) hc_dim, conv_out.data(), conv_out.data());
        // the state update: keep the last hist taps (drop tap 0, append norm_conv)
        for (int64_t c = 0; c < hc_dim; c++) {
            for (int64_t j = 0; j < hist; j++) {
                ple_conv_state[c * hist + j] = window[c * (hist + 1) + j + 1];
            }
        }
    }

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        int nn = 0; for (int64_t i = 0; i < hc_dim; i++) if (std::isnan(conv_out[i])) nn++;
        fprintf(stderr, "  ple key nan=%d value nan=%d conv_out nan=%d gate[0]=%.6g gate[1]=%.6g\n",
                (int) std::isnan(key[0]), (int) std::isnan(value[0]), nn, (double)gate[0], (double)gate[1]);
        FILE * f = fopen("/tmp/qwen4exp-builds/f_ple_gate.bin", "wb");
        if (f) { fwrite(gate.data(), 4, gate.size(), f); fclose(f); }
        FILE * f2 = fopen("/tmp/qwen4exp-builds/f_ple_conv.bin", "wb");
        if (f2) { fwrite(conv_out.data(), 4, conv_out.size(), f2); fclose(f2); }
    }
    // the combine: hidden + gated + conv_out
    for (int64_t i = 0; i < hc_dim; i++) {
        res[i] = hidden[i] + gated[i] + conv_out[i];
    }
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
        int nn = 0, gn = 0, vn = 0;
        for (int64_t i = 0; i < hc_dim; i++) { if (std::isnan(res[i])) nn++; if (std::isnan(gated[i])) gn++; }
        for (int64_t i = 0; i < n_embd; i++) if (std::isnan(value[i])) vn++;
        fprintf(stderr, "  ple res nan=%d gated nan=%d value nan=%d\n", nn, gn, vn);
    }
}

// ---- the fused decode wiring (the process_ubatch fast path) ----------------

#include "llama-memory-hybrid-idx.h"
#include "llama-kv-cache.h"
#include <chrono>

// the token embedding gather (the graph's build_inp_embd)
static void fused_embd(const struct ggml_tensor * tok_embd, int32_t tok, float * out, int64_t n_embd) {
    const size_t row_bytes = ggml_row_size(tok_embd->type, n_embd);
    const char * row = (const char *) tok_embd->data + (size_t) tok * row_bytes;
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "fused_embd: name=%s type=%d ne=[%lld,%lld] rb=%zu buf=%s buft=%s extra=%p row0=%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
            tok_embd->name, (int) tok_embd->type, (long long) tok_embd->ne[0], (long long) tok_embd->ne[1],
            row_bytes, tok_embd->buffer ? ggml_backend_buffer_name(tok_embd->buffer) : "-",
            tok_embd->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(tok_embd->buffer)) : "-",
            (const void *) tok_embd->extra,
            (unsigned char) row[0], (unsigned char) row[1], (unsigned char) row[2], (unsigned char) row[3],
            (unsigned char) row[4], (unsigned char) row[5], (unsigned char) row[6], (unsigned char) row[7],
            (unsigned char) row[8], (unsigned char) row[9], (unsigned char) row[10], (unsigned char) row[11],
            (unsigned char) row[12], (unsigned char) row[13], (unsigned char) row[14], (unsigned char) row[15]);
    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE"))
    fprintf(stderr, "fused_embd row40: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
            (unsigned char) row[0], (unsigned char) row[1], (unsigned char) row[2], (unsigned char) row[3],
            (unsigned char) row[4], (unsigned char) row[5], (unsigned char) row[6], (unsigned char) row[7],
            (unsigned char) row[8], (unsigned char) row[9], (unsigned char) row[10], (unsigned char) row[11],
            (unsigned char) row[12], (unsigned char) row[13], (unsigned char) row[14], (unsigned char) row[15],
            (unsigned char) row[16], (unsigned char) row[17], (unsigned char) row[18], (unsigned char) row[19],
            (unsigned char) row[20], (unsigned char) row[21], (unsigned char) row[22], (unsigned char) row[23],
            (unsigned char) row[24], (unsigned char) row[25], (unsigned char) row[26], (unsigned char) row[27],
            (unsigned char) row[28], (unsigned char) row[29], (unsigned char) row[30], (unsigned char) row[31],
            (unsigned char) row[32], (unsigned char) row[33], (unsigned char) row[34], (unsigned char) row[35],
            (unsigned char) row[36], (unsigned char) row[37], (unsigned char) row[38], (unsigned char) row[39]);
    if (tok_embd->type == GGML_TYPE_F32) {
        memcpy(out, row, n_embd * sizeof(float));
    } else if (tok_embd->type == GGML_TYPE_BF16) {
        const ggml_bf16_t * b = (const ggml_bf16_t *) row;
        for (int64_t i = 0; i < n_embd; i++) out[i] = ggml_bf16_to_fp32(b[i]);
    } else {
        const struct ggml_type_traits * qtt = ggml_get_type_traits(tok_embd->type);
        qtt->to_float(row, out, n_embd);
    }
}

// ---- A4: the residency / layout predicate ----------------------------------
//
// Before INF-70 A4 this returned `true` unconditionally, so the opt-out hook
// would have run the fused kernels against GPU-resident or repacked weights.
// It now checks what it claims:
//   (1) every model tensor sits on a buffer whose device is a CPU device — the
//       fused kernels dereference `tensor->data` directly and have no copy path;
//   (2) no tensor is repacked except the MoE down-experts, and those only when
//       they are IQ4_NL (the one interleaved layout `fused_moe` mirrors). The
//       repack extra_buffer_type claims MUL_MAT / MUL_MAT_ID only, so a repacked
//       weight read row-by-row by the plain vec_dot decodes silently wrong; the
//       two tables the fused path gathers ROWS from (tok_embd,
//       per_layer_tok_embd) must not be repacked at all;
//   (3) the hparams the fused kernels hard-assume are present and in range.
// Batch-1 / single-seq / DEFAULT-graph is checked at the hook (llama-context.cpp);
// the memory-context cache types and the logits carrier are checked in
// fused_decode()'s preflight, which runs before any persistent write.
bool llama_model_qwen4exp::supports_fused_decode() const {
    if (fused_decode_supported >= 0) {
        return fused_decode_supported == 1;
    }
    fused_decode_supported = 0;

    auto is_cpu_resident = [](const ggml_tensor * t) {
        if (t == nullptr)         return false;
        if (t->data == nullptr)   return false;
        if (t->buffer == nullptr) return false;
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
        if (buft == nullptr)      return false;
        // The device is usable only as a NEGATIVE test: ggml_backend_cpu_buffer_type()
        // ships with `.device = NULL` behind an upstream FIXME
        // (ggml/src/ggml-backend.cpp: "// FIXME ggml_backend_reg_dev_get(...)"), so
        // requiring a CPU device here refuses every host tensor. Measured
        // 2026-09-02: that is exactly what happened — "tensor 'token_embd.weight'
        // is not CPU-resident" and all three fused arms silently fell back to the
        // graph, at x1.00 and a bit-identical logit diff.
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return false;
        }
        // The positive test is host memory. The plain CPU buffer types implement
        // is_host; CPU_REPACK does not implement it but IS host memory by
        // construction (its own kernels dereference tensor->data directly), and
        // since A1 the fused path dispatches those through
        // ggml_cpu_extra_compute_forward() rather than reading their rows itself.
        if (ggml_backend_buft_is_host(buft)) {
            return true;
        }
        const char * bn = ggml_backend_buft_name(buft);
        return bn != nullptr && strcmp(bn, "CPU_REPACK") == 0;
    };

    // (1) + (2) over every loaded tensor
    for (const auto & kv : tensors_by_name) {
        const ggml_tensor * t = kv.second;
        if (!is_cpu_resident(t)) {
            LLAMA_LOG_INFO("%s: fused decode unavailable: tensor '%s' is not CPU-resident\n",
                    __func__, kv.first.c_str());
            return false;
        }
        if (t->extra != nullptr) {
            // Repacked (CPU_REPACK: 4x4/8x8-interleaved rows). Since A1 the dense
            // projections go through ggml_cpu_extra_compute_forward(), which is
            // exactly the hook that claims these, so a repacked MUL_MAT weight is
            // handled correctly and needs no guard. Measured on the uniform IQ4_XS
            // artifact with GGML_IQK=1: hc_attn_up / hc_ffn_up are IQ4_NL
            // [320, 10240] and ARE repacked (IQ4_NL is not in the iqk exclusion
            // list and ne[1] % 8 == 0), so the pre-A1 per-row vec_dot was reading
            // an interleaved layout as if it were plain rows, on every layer.
            //
            // The routed down-experts are the exception: fused_moe still walks
            // them by hand, and that mirror only implements the IQ4_NL 4x4/8x8
            // layout.
            const bool is_down_exps = kv.first.find("ffn_down_exps") != std::string::npos;
            if (is_down_exps && t->type != GGML_TYPE_IQ4_NL) {
                LLAMA_LOG_INFO("%s: fused decode unavailable: '%s' (%s) is repacked and fused_moe mirrors IQ4_NL only\n",
                        __func__, kv.first.c_str(), ggml_type_name(t->type));
                return false;
            }
        }
    }

    // (2b) the row-gather tables: a raw row read cannot see any repacking
    if (tok_embd == nullptr || tok_embd->extra != nullptr) return false;
    if (per_layer_tok_embd == nullptr || per_layer_tok_embd->extra != nullptr) return false;
    if (output == nullptr) return false;

    // (3) the hparams the fused kernels hard-assume
    if (hparams.dsv4_hc_mult <= 0)                                return false;
    if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > 8) return false; // ctxv[8] on the stack
    if (hparams.n_expert_used == 0)                               return false;
    if (layers.size() != hparams.n_layer())                       return false;

    fused_decode_supported = 1;
    return true;
}

bool llama_model_qwen4exp::fused_decode(
        const llama_ubatch & ubatch,
        const struct llama_memory_context_i * mctx_in,
        class llm_graph_result * res,
        int n_threads,
        const struct ggml_tensor * const * prev_layer_inp) const {
    FUSED_PROF_INIT();
    FUSED_PROF_RESET();
    // A1 kill switch, read once per token (never in an inner loop)
    g_mm_legacy = getenv("GGML_FUSED_MM_LEGACY") != NULL;
    g_arena_off = getenv("GGML_FUSED_ARENA_OFF") != NULL;
    g_arena.reset();
    g_census.reset();
    const auto prof_t0 = std::chrono::steady_clock::now();
    if (ubatch.n_tokens != 1 || ubatch.n_seqs != 1) return false;
    if (ubatch.token == nullptr) return false;

    const int64_t n_layer_slot = hparams.n_layer() + 1;
    const auto * mctx = static_cast<const llama_memory_hybrid_idx_context *>(mctx_in);

    // ================= A4 PREFLIGHT =================
    // Everything that can refuse refuses HERE, before a single byte of
    // persistent state is written. Combined with the staged commit below, a
    // fall-through to the graph path from anywhere in this function leaves the
    // memory context able to run the same token.
    if (mctx == nullptr || mctx->get_attn() == nullptr || mctx->get_recr() == nullptr ||
            mctx->get_idx() == nullptr) {
        return false;
    }

    // the logits carrier. The fused path builds no graph, so it has no tensor of
    // its own that the scheduler knows about; llama_context::decode() extracts
    // through ggml_backend_sched_get_tensor_backend(res->get_logits()) and
    // asserts that lookup is non-null. The previous graph's logits tensor is the
    // only sched-known F32 buffer of the right size available here, so the fused
    // path writes into it. That write is safe *because*:
    //   - res->reset() has cleared the graph and the fused path allocates NO
    //     graph, so nothing between this write and the extraction a few lines
    //     later in decode() can re-assign those bytes (the gallocr only
    //     re-partitions the compute buffer inside ggml_backend_sched_alloc_graph);
    //   - the extraction happens in the same call, before any other decode;
    //   - and the properties it relies on are now CHECKED rather than assumed:
    //     existence, F32, contiguity, and capacity >= n_vocab floats. Before A4
    //     none of these were checked and the failure mode (a null carrier on the
    //     very first decode of a context) surfaced only AFTER the whole token had
    //     been computed and the recurrent state advanced.
    // The remaining coupling — that the carrier is another graph's tensor rather
    // than one this path owns — is what a fully owned buffer would remove; doing
    // that needs ggml_backend_sched_set_tensor_backend() (which clears the
    // scheduler's is_reset flag) or a change to decode()'s extraction path.
    const int64_t n_vocab_pf = vocab.n_tokens();
    ggml_tensor * t_logits = const_cast<ggml_tensor *>(prev_layer_inp[n_layer_slot]);
    if (t_logits == nullptr || t_logits->data == nullptr ||
            t_logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(t_logits) ||
            ggml_nbytes(t_logits) < (size_t) n_vocab_pf * sizeof(float)) {
        return false;
    }

    // the cache views + their types, for every full-attention layer. These were
    // checked inside fused_full_attn_layer() — i.e. after the GDN/PLE layers
    // below it had already advanced the recurrent state — which is exactly the
    // partial-write hazard A4 exists to remove.
    // A2: the arena. This context only holds cache VIEWS (no data), so a few
    // hundred tensor headers is the whole requirement — it was asking for 64 MB.
    ggml_context * vctx = fused_arena_init(FUSED_ARENA_VIEWS,
            (size_t) (8 * hparams.n_layer() + 64) * ggml_tensor_overhead() + (1u << 16), 64u << 20);
    if (vctx == nullptr) {
        return false;
    }
    {
        const auto * attn_pf = mctx->get_attn();
        const auto * idx_pf  = mctx->get_idx();
        const auto * recr_pf = mctx->get_recr();
        bool ok = true;
        for (int il = 0; ok && il < (int) hparams.n_layer(); il++) {
            if (!hparams.is_recr(il)) {
                ggml_tensor * tk = attn_pf->get_k(vctx, il);
                ggml_tensor * tv = attn_pf->get_v(vctx, il);
                ggml_tensor * ti = idx_pf->get_k(vctx, il);
                if (tk == nullptr || tv == nullptr || ti == nullptr ||
                        tk->data == nullptr || tv->data == nullptr || ti->data == nullptr) {
                    ok = false;
                    break;
                }
                if (!((tk->type == GGML_TYPE_F32 || tk->type == GGML_TYPE_F16) &&
                       tv->type == tk->type &&
                      (ti->type == GGML_TYPE_F32 || ti->type == GGML_TYPE_F16))) {
                    LLAMA_LOG_INFO("%s: fused decode declined: cache types %s/%s/%s at layer %d\n",
                            __func__, ggml_type_name(tk->type), ggml_type_name(tv->type),
                            ggml_type_name(ti->type), il);
                    ok = false;
                    break;
                }
            } else {
                const auto * rl = recr_pf->get_r_l(il);
                const auto * sl = recr_pf->get_s_l(il);
                if (rl == nullptr || sl == nullptr || rl->data == nullptr || sl->data == nullptr ||
                        rl->type != GGML_TYPE_F32 || sl->type != GGML_TYPE_F32) { ok = false; break; }
            }
            if (hparams.is_ple_impl[il]) {
                const auto * pl = recr_pf->get_p_l(il);
                if (pl == nullptr || pl->data == nullptr || pl->type != GGML_TYPE_F32) { ok = false; break; }
            }
        }
        if (!ok) {
            ggml_free(vctx);
            return false;
        }
    }
    // ================= END A4 PREFLIGHT =================

    const int64_t n_embd = hparams.n_embd;
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int32_t tok = ubatch.token[0];
    const int64_t pos = ubatch.pos[0];
    const int64_t seq = ubatch.seq_id[0][0];

    FusedRopeParams rp;
    rp.n_ctx_orig = 0; // filled by the caller's cparams via the hook
    // the rope values are set by the hook's caller (the cparams); defaults here
    rp.freq_base = 10000000.0f;
    rp.freq_scale = 1.0f;
    rp.ext_factor = 0.0f;
    rp.attn_factor = 1.0f;
    rp.beta_fast = 32.0f;
    rp.beta_slow = 1.0f;

    // the token embedding
    std::vector<float> res_hc(hc_dim);
    {
        std::vector<float> emb(n_embd);
        fused_embd(tok_embd, tok, emb.data(), n_embd);
        if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) fprintf(stderr, "fused res_hc: hc=%lld n_embd=%lld hc_dim=%lld emb[0]=%.8f\n",
                (long long) hc, (long long) n_embd, (long long) hc_dim, (double) emb[0]);
        for (int64_t c = 0; c < hc; c++) {
            memcpy(res_hc.data() + c * n_embd, emb.data(), n_embd * sizeof(float));
        }
        if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
            FILE * f = fopen("/tmp/qwen4exp-builds/f_res_hc.bin", "wb");
            if (f) { fwrite(res_hc.data(), 4, res_hc.size(), f); fclose(f); }
            FILE * f2 = fopen("/tmp/qwen4exp-builds/f_embd_tok.bin", "wb");
            if (f2) { fwrite(emb.data(), 4, emb.size(), f2); fclose(f2); }
        }
        if (FUSED_DBG("GGML_FUSED_DUMP_FLAYERS") && FUSED_DBG("GGML_FUSED_ONCE")) {
            FILE * f = fopen("/tmp/qwen4exp-builds/f_embd.bin", "wb");
            if (f) { fwrite(emb.data(), 4, n_embd, f); fclose(f); }
        }
    }

    std::vector<float> layer_out(n_embd);

    // A4 ATOMICITY: every write to persistent recurrent state (the PLE conv
    // history and the GDN conv/ssm rows) is staged here and applied in one pass
    // at end-of-token. Until that pass runs the memory context still holds the
    // pre-token state, so ANY bail-out below leaves the graph path able to run
    // the same token. (The KV / indexer cell writes are NOT staged: the cell was
    // allocated by mctx->apply() for THIS token and is written with this token's
    // k/v, so re-running the token through the graph rewrites the same cell with
    // the same values — idempotent, unlike the recurrent state, which advances.)
    // A2: the staging lives in ONE buffer reused across tokens rather than a
    // fresh std::vector per layer — the GDN [S_v, S_v, H_v] row alone is 3.1 MB,
    // so this was ~113 MB of malloc/free (and first-touch page faults) per token.
    // Function-static: the fused path is single-threaded by construction.
    struct PendingStateWrite { void * dst; size_t off; size_t n; };
    static std::vector<float> stage_buf;
    static std::vector<PendingStateWrite> pending_state;
    pending_state.clear();
    size_t stage_used = 0;
    auto stage_alloc = [&](size_t n) -> size_t {
        const size_t off = stage_used;
        stage_used += n;
        if (stage_buf.size() < stage_used) stage_buf.resize(stage_used);
        return off;
    };

    if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && FUSED_DBG("GGML_FUSED_LAYER_CMP") && prev_layer_inp && prev_layer_inp[0] && prev_layer_inp[0]->data) {
        const ggml_tensor * g0 = prev_layer_inp[0];
        const int64_t nt = g0->ne[1];
        const float * gp = (const float *) g0->data + (nt - 1) * g0->ne[0];
        double md = 0.0;
        for (int64_t i = 0; i < g0->ne[0]; i++) md = fmax(md, (double) fabs(res_hc[i] - gp[i]));
        fprintf(stderr, "  fused cmp input embd: max_abs=%.6g\n", md);
    }
    for (int il = 0; il < (int) hparams.n_layer(); il++) {
        const struct llama_layer & L = layers[il];
        if (FUSED_DBG("GGML_FUSED_DECODE_TRACE")) {
            fprintf(stderr, "fused layer %2d: %s (ple=%d recr=%d)\n", il,
                    hparams.is_ple_impl[il] ? "PLE" : (hparams.is_recr(il) ? "GDN" : "ATTN"),
                    (int) hparams.is_ple_impl[il], (int) hparams.is_recr(il));
        }

        if (FUSED_DBG("GGML_FUSED_SKIP_PLE") && hparams.is_ple_impl[il]) {
            continue;
        }
        if (FUSED_DBG("GGML_FUSED_SKIP_ATTN") && !hparams.is_ple_impl[il] && !hparams.is_recr(il)) {
            continue;
        }
        if (hparams.is_ple_impl[il]) {
            // the PLE: the predecessors from the attention cells
            if (FUSED_DBG("GGML_FUSED_LAYER_CMP") && prev_layer_inp && prev_layer_inp[il] && prev_layer_inp[il]->data) {
                const ggml_tensor * g0 = prev_layer_inp[il];
                const int64_t nt = g0->ne[2] > 1 ? g0->ne[2] : 1;
                const float * gp = (const float *) g0->data + (nt - 1) * g0->ne[0] * g0->ne[1];
                double md = 0.0, ss = 0.0;
                for (int64_t i = 0; i < hc_dim; i++) {
                    md = fmax(md, (double) fabs(res_hc[i] - gp[i]));
                    ss += (double) (gp[i] * gp[i]);
                }
                fprintf(stderr, "  fused cmp ple hidden il=%d: max_abs=%.6g graph_sc=%.6f graph_first=%.6g %.6g f_first=%.6g %.6g\n",
                        il, md, 1.0 / sqrt((ss / hc_dim) + hparams.f_norm_rms_eps),
                        (double) gp[0], (double) gp[1], (double) res_hc[0], (double) res_hc[1]);
                FILE * f = fopen("/tmp/qwen4exp-builds/g_layer_inp.bin", "wb");
                if (f) { fwrite(gp, 4, hc_dim, f); fclose(f); }
                FILE * f2 = fopen("/tmp/qwen4exp-builds/f_layer_inp.bin", "wb");
                if (f2) { fwrite(res_hc.data(), 4, hc_dim, f2); fclose(f2); }
            }
            const auto * attn = mctx->get_attn();
            std::vector<llama_token> prev_toks;
            attn->get_prev_tokens(ubatch, hparams.ple_ngram_size - 1, prev_toks);
            // get_prev_tokens is oldest-first (the furthest back first), and the
            // hash below reads prev[n_prev - s] with that same convention — keep
            // the order as-is (a reversal here would flip the n-gram context)
            int32_t prev[2] = { -1, -1 };
            for (int64_t s = 0; s < hparams.ple_ngram_size - 1; s++) {
                const llama_token t = prev_toks[s];
                prev[s] = t < 0 ? -1 : (int32_t) t;
            }
            const auto * recr = mctx->get_recr();
            const auto * pl = recr->get_p_l(il);
            const size_t row_bytes = ggml_row_size(pl->type, pl->ne[0]);
            const int head = (int) recr->get_head();
            const int64_t hist = (hparams.ple_conv_kernel - 1) * hparams.ple_ngram_size;
            const size_t ple_n   = (size_t) (hist * hc_dim);
            const size_t ple_off = stage_alloc(ple_n);
            memcpy(stage_buf.data() + ple_off,
                   (const char *) pl->data + (size_t) head * row_bytes,
                   ple_n * sizeof(float));
            fused_ple(*this, hparams, L, tok, prev, res_hc.data(), res_hc.data(),
                      stage_buf.data() + ple_off, n_threads);
            // A4: staged, not written (see PendingStateWrite above)
            pending_state.push_back({ (void *) ((const char *) pl->data + (size_t) head * row_bytes),
                                      ple_off, ple_n });
        } 
        if (hparams.is_recr(il)) {
            // the GDN: the conv + ssm state rows at the current head
            const auto * recr = mctx->get_recr();
            const auto * rl = recr->get_r_l(il);
            const auto * sl = recr->get_s_l(il);
            const int head = (int) recr->get_head();
            const size_t rrow = ggml_row_size(rl->type, rl->ne[0]);
            const size_t srow = ggml_row_size(sl->type, sl->ne[0]);
            const float * conv_state = (const float *) ((const char *) rl->data + (size_t) head * rrow);
            const float * ssm_state  = (const float *) ((const char *) sl->data + (size_t) head * srow);
            // the fused layer works on copies for the state (in-place safety);
            // the conv row width is (d_conv-1)*conv_channels == rl->ne[0]
            const size_t conv_n = (size_t) rl->ne[0];
            const size_t ssm_n  = (size_t) sl->ne[0];
            const size_t conv_off = stage_alloc(conv_n);
            const size_t ssm_off  = stage_alloc(ssm_n);
            memcpy(stage_buf.data() + conv_off, conv_state, conv_n * sizeof(float));
            memcpy(stage_buf.data() + ssm_off,  ssm_state,  ssm_n  * sizeof(float));
            fused_gdn_layer(L, hparams, res_hc.data(), res_hc.data(), layer_out.data(),
                            stage_buf.data() + conv_off, stage_buf.data() + ssm_off, n_threads, il);
            // A4: staged, not written (see PendingStateWrite above)
            pending_state.push_back({ (void *) conv_state, conv_off, conv_n });
            pending_state.push_back({ (void *) ssm_state,  ssm_off,  ssm_n  });
        }
        if (!hparams.is_recr(il)) {
            // the full-attn layer: the KV + indexer cache slices. The cell
            // bookkeeping ran in apply(): the current ubatch's cell is already
            // allocated, so the last used cell is this token's write target.
            const auto * attn = mctx->get_attn();
            const auto * idx = mctx->get_idx();
            // unreachable after the A4 preflight; harmless if reached — no state committed yet
            if (idx == nullptr) { ggml_free(vctx); return false; }
            const llama_seq_id seq = ubatch.seq_id[0][0];
            const auto & attn_cells = attn->get_cells(seq);
            const auto & idx_cells  = idx->get_cells(seq);
            const int n_used    = (int) attn_cells.used_max_p1();
            const int idx_n_used = (int) idx_cells.used_max_p1();
            int n_visible = 0;
            for (int j = 0; j < n_used; j++) {
                if (attn_cells.pos_get((uint32_t) j) < pos) n_visible++;
            }
            ggml_tensor * tk = attn->get_k(vctx, il);
            ggml_tensor * tv = attn->get_v(vctx, il);
            ggml_tensor * ti = idx->get_k(vctx, il);
            if (FUSED_DBG("GGML_FUSED_DECODE_TRACE") && il == 3) {
                fprintf(stderr, "  fused cache: tk type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n", ggml_type_name(tk->type), (long long)tk->ne[0],(long long)tk->ne[1],(long long)tk->ne[2],(long long)tk->ne[3],(size_t)tk->nb[0],(size_t)tk->nb[1],(size_t)tk->nb[2],(size_t)tk->nb[3]);
                fprintf(stderr, "  fused cache: tv type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n", ggml_type_name(tv->type), (long long)tv->ne[0],(long long)tv->ne[1],(long long)tv->ne[2],(long long)tv->ne[3],(size_t)tv->nb[0],(size_t)tv->nb[1],(size_t)tv->nb[2],(size_t)tv->nb[3]);
                fprintf(stderr, "  fused cache: n_used=%d n_visible=%d cell pos: ", n_used, n_visible);
                for (int j = 0; j < n_used; j++) {
                    const uint32_t pj = attn_cells.pos_get((uint32_t) j);
                    fprintf(stderr, "%u ", pj);
                }
                fprintf(stderr, "\n");
            }
            if (tk == nullptr || tv == nullptr || ti == nullptr) { ggml_free(vctx); return false; }
            if (!fused_full_attn_layer(L, hparams, rp, (int32_t) pos,
                        res_hc.data(), res_hc.data(), layer_out.data(),
                        tk, tv, ti, n_used, n_visible, idx_n_used, n_threads, il)) {
                ggml_free(vctx);
                return false;
            }
        }
        if (FUSED_DBG("GGML_FUSED_DUMP_FLAYERS") && il + 1 < (int64_t) hparams.n_layer()) {
            FILE * f = fopen("/tmp/qwen4exp-builds/f_layers.bin", "ab");
            if (f) {
                fwrite(res_hc.data(), 4, res_hc.size(), f);
                fclose(f);
            }
        }
    }

    // the head -> the logits
    const int64_t n_vocab = vocab.n_tokens();
    std::vector<float> logits(n_vocab);
    fused_head(*this, hparams, res_hc.data(), logits.data(), n_threads);

    // ================= A4 COMMIT =================
    // The token is complete. Apply every staged recurrent-state write in one
    // pass; only after this point has the model's persistent state advanced.
    for (const auto & pw : pending_state) {
        memcpy(pw.dst, stage_buf.data() + pw.off, pw.n * sizeof(float));
    }
    g_arena.stage_bytes = stage_used * sizeof(float);

    // the logits carrier, validated in the preflight (see the note there for why
    // writing into the previous graph's sched-known tensor is safe here)
    GGML_ASSERT(t_logits != nullptr && t_logits->data != nullptr);
    GGML_ASSERT(ggml_nbytes(t_logits) >= (size_t) n_vocab * sizeof(float));
    res->t_logits = t_logits;
    memcpy(t_logits->data, logits.data(), n_vocab * sizeof(float));

    ggml_free(vctx);
    if (g_prof.on) {
        const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prof_t0).count();
        const double gemv_ms = g_prof.gemv_ms;
        const double other_ms = total_ms - gemv_ms;
        if (gemv_ms > total_ms || other_ms < 0.0 || total_ms <= 0.0 || gemv_ms < 0.0) {
            fprintf(stderr, "FUSED_PROF: SANITY FAILED total=%.1f gemv=%.1f other=%.1f\n",
                    total_ms, gemv_ms, other_ms);
        } else {
            fprintf(stderr, "FUSED_PROF: total=%.1f ms gemv=%.1f ms other=%.1f ms (gemv %.0f%%)"
                    " | ctx_init=%llu calls: pre-A2 churn %.0f MB/token -> need %.1f MB,"
                    " staged state %.1f MB/token, arenas %.1f MB resident\n",
                    total_ms, gemv_ms, other_ms, 100.0 * gemv_ms / total_ms,
                    (unsigned long long) g_arena.ctx_calls,
                    (double) g_arena.churn_was / (1024.0 * 1024.0),
                    (double) g_arena.ctx_bytes / (1024.0 * 1024.0),
                    (double) g_arena.stage_bytes / (1024.0 * 1024.0),
                    (double) g_arena.arena_bytes / (1024.0 * 1024.0));
            static const char * cn[3] = { "dense/lora", "experts", "lm_head" };
            uint64_t tc = 0; double tb = 0.0, tu = 0.0;
            for (int i = 0; i < 3; i++) {
                fprintf(stderr, "FUSED_MM_CENSUS: %-11s calls=%-6llu bytes=%8.1f MB  mean=%8.1f us  total=%8.1f ms\n",
                        cn[i], (unsigned long long) g_census.calls[i],
                        (double) g_census.bytes[i] / (1024.0 * 1024.0),
                        g_census.calls[i] ? g_census.us[i] / (double) g_census.calls[i] : 0.0,
                        g_census.us[i] / 1000.0);
                tc += g_census.calls[i]; tb += (double) g_census.bytes[i]; tu += g_census.us[i];
            }
            fprintf(stderr, "FUSED_MM_CENSUS: TOTAL       calls=%-6llu bytes=%8.1f MB  mean=%8.1f us  total=%8.1f ms"
                    "   (graph reference: 797 mul_mat + 144 mul_mat_id, ~4.16 GB/token)\n",
                    (unsigned long long) tc, tb / (1024.0 * 1024.0),
                    tc ? tu / (double) tc : 0.0, tu / 1000.0);
        }
    }
    return true;
}
