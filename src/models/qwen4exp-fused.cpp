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

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-memory-hybrid.h"
#include "models/models.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace llama_model_qwen4exp_fused {

using ggml_type_traits_cpu = struct ggml_type_traits_cpu;

// ---- helpers mirroring the graph's batch-1 kernels -------------------------

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
        qt->vec_dot((int) n_in, out, 0, (const char *) w->data + (size_t) row * w->nb[1], 0, xq.data(), 0, 1);
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

// the lora mm: the per-row dots, optionally multiplied elementwise by w_s
static void lora_mm(const struct ggml_tensor * w, const float * x, const struct ggml_tensor * w_s, float * out, int n_threads) {
    FusedMM mm(w, x, n_threads);
    const int64_t n_out = w->ne[1];
    (void) n_threads;
    for (int64_t j = 0; j < n_out; j++) {
        mm.dot(w, (int) j, &out[j]);
        if (w_s) {
            const float s = ((const float *) w_s->data)[j % w_s->ne[0]];
            out[j] *= s;
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
static void hc_mix(const struct llama_layer & L, int64_t hc, float eps,
                   const float * xn, float * mixed, float * inject,
                   int n_threads) {
    // the lo-rank down: lo = silu(mm(w_down, xn) * (1/hc))
    const int64_t low_rank = L.hc_attn_down->ne[1];
    std::vector<float> lo(low_rank);
    lora_mm(L.hc_attn_down, xn, nullptr, lo.data(), n_threads);
    const float inv_hc = 1.0f / (float) hc;
    for (int64_t j = 0; j < low_rank; j++) {
        const float v = lo[j] * inv_hc;
        lo[j] = v / (1.0f + expf(-v)); // silu
    }
    // the gate: sigmoid(mm(w_up, lo)); gated = xn*gate; mean over streams
    const int64_t hc_dim = L.hc_attn_up->ne[1];
    std::vector<float> gate(hc_dim);
    FusedMM mm_up(L.hc_attn_up, lo.data(), n_threads);
    for (int64_t i = 0; i < hc_dim; i++) {
        mm_up.dot(L.hc_attn_up, (int) i, &gate[i]);
        gate[i] = 1.0f / (1.0f + expf(-gate[i]));
    }
    const int64_t n_embd2 = hc_dim / hc;
    hc_stream_mean(xn, gate.data(), mixed, n_embd2, hc);
    if (inject) {
        lora_mm(L.hc_attn_inject, xn, nullptr, inject, n_threads);
    }
}

} // namespace llama_model_qwen4exp_fused
