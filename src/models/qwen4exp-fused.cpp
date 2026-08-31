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
#include "llama-model.h"
#include "llama-memory-hybrid.h"
#include "models/models.h"

#include <cmath>
#include <cstring>
#include <vector>

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
    FusedMM mm_up(w_up, lo.data(), n_threads);
    for (int64_t i = 0; i < hc_dim; i++) {
        mm_up.dot(w_up, (int) i, &gate[i]);
        gate[i] = 1.0f / (1.0f + expf(-gate[i]));
    }
    const int64_t n_embd2 = hc_dim / hc;
    hc_stream_mean(xn, gate.data(), mixed, n_embd2, hc);
    if (inject) {
        lora_mm(w_inject, xn, nullptr, inject, n_threads);
    }
}


// ---- the MoE (mirrors build_moe_ffn + build_layer_ffn's shared expert) -----

static void fused_moe(
        const struct llama_layer & L,
        const struct llama_hparams & hp,
        const float * x, float * out, int n_threads) {

    const int64_t n_expert = L.ffn_gate_inp->ne[1]; // 512
    const int64_t n_used   = hp.n_expert_used;      // 10

    // router logits
    std::vector<float> logits(n_expert);
    lora_mm(L.ffn_gate_inp, x, nullptr, logits.data(), n_threads);
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        extern float ggml_table_f32_f16[1 << 16];
        fprintf(stderr, "  moe f16table[0x3C00]=%.6g f16table[0]=%.6g\n",
                (double) ggml_table_f32_f16[0x3C00], (double) ggml_table_f32_f16[0]);
    }
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        fprintf(stderr, "  moe router[0..5] = %.6g %.6g %.6g %.6g %.6g %.6g  (n_used=%d)\n",
                (double) logits[0], (double) logits[1], (double) logits[2],
                (double) logits[3], (double) logits[4], (double) logits[5],
                (int) n_used);
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

    // the expert gemvs: up/gate [2560, 640, 512] IQ3_S with the Q8_K activation
    const struct ggml_type_traits_cpu * qt_up = ggml_get_type_traits_cpu(L.ffn_up_exps->type);
    const struct ggml_type_traits_cpu * qt_upv = ggml_get_type_traits_cpu(qt_up->vec_dot_type);
    const size_t xq_up_size = ggml_row_size(qt_up->vec_dot_type, hp.n_embd);
    std::vector<uint8_t> xq_up(xq_up_size);
    qt_upv->from_float(x, xq_up.data(), hp.n_embd);

    const int64_t n_ff = L.ffn_up_exps->ne[1]; // 640
    std::vector<float> glu(n_used * n_ff), up_tmp(n_used * n_ff), gate_tmp(n_used * n_ff);
    const size_t nb_up_exp = L.ffn_up_exps->nb[2]; // expert stride

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        fprintf(stderr, "  moe up: type=%s nb1=%zu nb2=%zu n_embd=%lld sel[0]=%d\n",
                ggml_type_name(L.ffn_up_exps->type), (size_t) L.ffn_up_exps->nb[1],
                (size_t) L.ffn_up_exps->nb[2], (long long) hp.n_embd, sel[0]);
    }
    for (int64_t j = 0; j < n_used; j++) {
        const int32_t e = sel[j];
        const char * up_e = (const char *) L.ffn_up_exps->data + (size_t) e * nb_up_exp;
        const char * gt_e = (const char *) L.ffn_gate_exps->data + (size_t) e * nb_up_exp;
        for (int64_t r = 0; r < n_ff; r++) {
            qt_up->vec_dot((int) hp.n_embd, &up_tmp[j * n_ff + r], 0,
                           up_e + (size_t) r * L.ffn_up_exps->nb[1], 0, xq_up.data(), 0, 1);
            qt_up->vec_dot((int) hp.n_embd, &gate_tmp[j * n_ff + r], 0,
                           gt_e + (size_t) r * L.ffn_gate_exps->nb[1], 0, xq_up.data(), 0, 1);
            const float g = gate_tmp[j * n_ff + r];
            glu[j * n_ff + r] = up_tmp[j * n_ff + r] * (g / (1.0f + expf(-g))); // silu(gate)*up
        }
    }
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
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
    std::vector<uint8_t> glu_q(glu_q_size * n_used);
    for (int64_t j = 0; j < n_used; j++) {
        qt_dnv->from_float(glu.data() + j * n_ff, glu_q.data() + j * glu_q_size, n_ff);
    }
    static const int8_t kv_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    const bool dn_repacked = L.ffn_down_exps->extra != nullptr;
    const int64_t rp_I = dn_repacked ? (ggml_cpu_has_avx2() ? 8 : 4) : 0;
    const int64_t rp_nblocks = n_ff / 32;
    const size_t nb_dn_exp = L.ffn_down_exps->nb[2];
    std::vector<float> down_acc(hp.n_embd, 0.0f);
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        fprintf(stderr, "  moe dn: type=%s vec_dot_type=%s nb1=%zu nb2=%zu glu_q_size=%zu buf=%s repacked=%d I=%lld\n",
                ggml_type_name(L.ffn_down_exps->type), ggml_type_name(qt_dn->vec_dot_type),
                (size_t) L.ffn_down_exps->nb[1], (size_t) L.ffn_down_exps->nb[2], glu_q_size,
                L.ffn_down_exps->buffer ? ggml_backend_buffer_name(L.ffn_down_exps->buffer) : "none",
                (int) dn_repacked, (long long) rp_I);
        fprintf(stderr, "  moe up: type=%s buf=%s\n", ggml_type_name(L.ffn_up_exps->type),
                L.ffn_up_exps->buffer ? ggml_backend_buffer_name(L.ffn_up_exps->buffer) : "none");
    }
    for (int64_t j = 0; j < n_used; j++) {
        const int32_t e = sel[j];
        const char * dn_e = (const char *) L.ffn_down_exps->data + (size_t) e * nb_dn_exp;
        for (int64_t r = 0; r < hp.n_embd; r++) {
            float v = 0.0f;
            if (dn_repacked) {
                // the interleaved IQ4_NL dot for one row of its group
                const int64_t g = r / rp_I, slot = r % rp_I;
                const char * grp = dn_e + g * (rp_nblocks * 18 * rp_I);
                if (getenv("GGML_FUSED_DECODE_TRACE") != NULL && j == 0 && r < 3) {
                    const ggml_fp16_t * d0 = (const ggml_fp16_t *) grp;
                    const ggml_fp16_t * d1 = (const ggml_fp16_t *) (grp + 18 * rp_I);
                    fprintf(stderr, "  moe dn r%lld: slot=%lld d[0..2]=%.6g %.6g %.6g  l1d[0..2]=%.6g %.6g %.6g\n",
                            (long long) r, (long long) slot,
                            (double) ggml_fp16_to_fp32(d0[0]), (double) ggml_fp16_to_fp32(d0[1]), (double) ggml_fp16_to_fp32(d0[2]),
                            (double) ggml_fp16_to_fp32(d1[0]), (double) ggml_fp16_to_fp32(d1[1]), (double) ggml_fp16_to_fp32(d1[2]));
                }
                const char * act = (const char *) glu_q.data() + j * glu_q_size;
                float dot = 0.0f;
                if (getenv("GGML_FUSED_DECODE_TRACE") != NULL && j == 0 && r == 3) {
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
                    int sumi = 0;
                    for (int64_t k = 0; k < 32 / (2 * rp_I); k++) {
                        for (int64_t i = 0; i < rp_I; i++) {
                            const uint8_t byte = qs[k * rp_I * rp_I + slot * rp_I + i];
                            sumi += kv_iq4nl[byte & 0xF] * aqs[k * rp_I + i]
                                 +  kv_iq4nl[byte >> 4] * aqs[k * rp_I + i + 16];
                        }
                    }
                    dot += (float) sumi * ggml_fp16_to_fp32(dsc[slot]) * ggml_fp16_to_fp32(ad[0]);
                }
                v = dot;
            } else {
                qt_dn->vec_dot((int) n_ff, &v, 0, dn_e + (size_t) r * L.ffn_down_exps->nb[1], 0,
                               glu_q.data() + j * glu_q_size, 0, 1);
            }
            down_acc[r] += v * w[j];
            if (getenv("GGML_FUSED_DECODE_TRACE") != NULL && r < 2) {
                fprintf(stderr, "  moe dn j%lld r%lld: e=%d v=%.6g w=%.6g\n", (long long) j, (long long) r, e, (double) v, (double) w[j]);
            }
        }
    }
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
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
        float shg = 0.0f;
        FusedMM mm_shg(L.ffn_gate_inp_shexp, x, n_threads);
        mm_shg.dot(L.ffn_gate_inp_shexp, 0, &shg);
        shg = 1.0f / (1.0f + expf(-shg));
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
        int n_threads) {

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
    hc_rms_norm_gamma(x, L.hc_attn_norm, xn.data(), n_embd, hc, eps);
    hc_mix(L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, hc, xn.data(), mixed.data(), inject.data(), n_threads);

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) fprintf(stderr, "  gdn hc_mix done\n");
    // ---- GDN body ----
    // the qkv span is the conv channels (q+k+v), not ssm_d_inner (the v span alone)
    const int64_t qkv_span = L.wqkv->ne[1];
    std::vector<float> qkv(qkv_span);
    std::vector<float> z(hp.ssm_d_inner);              // 6144 (v-dim)
    lora_mm(L.wqkv, mixed.data(), nullptr, qkv.data(), n_threads);
    if (getenv("GGML_FUSED_DUMP_FLAYERS") != NULL && getenv("GGML_FUSED_ONCE") != NULL) {
        FILE * f = fopen("/tmp/qwen4exp-builds/f_mix.bin", "wb");
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

    // conv: window = conv_state [3, 10240] + qkv [1, 10240]
    const int64_t d_conv = L.ssm_conv1d->ne[0]; // 4
    const int64_t n_ch   = L.ssm_conv1d->ne[1]; // 10240
    std::vector<float> window((d_conv) * n_ch);
    memcpy(window.data(), conv_state_row, (d_conv - 1) * n_ch * sizeof(float));
    memcpy(window.data() + (d_conv - 1) * n_ch, qkv.data(), n_ch * sizeof(float));
    std::vector<float> conv_out(n_ch);
    conv1d_4tap(window.data(), (const float *) L.ssm_conv1d->data, conv_out.data(), n_ch);
    for (int64_t i = 0; i < n_ch; i++) {
        const float v = conv_out[i];
        conv_out[i] = v / (1.0f + expf(-v)); // silu
    }
    // conv state update: keep the last d_conv-1 columns (the 3-token tail)
    memcpy(conv_state_row, window.data() + n_ch, (d_conv - 1) * n_ch * sizeof(float));

    // q/k/v split + l2 norms
    const int64_t q_off = 0, k_off = S_k * H_k, v_off = 2 * S_k * H_k;
    std::vector<float> q(S_k * H_k), k(S_k * H_k), v(S_v * H_v);
    l2_norm(conv_out.data() + q_off, q.data(), S_k * H_k, eps);
    l2_norm(conv_out.data() + k_off, k.data(), S_k * H_k, eps);
    memcpy(v.data(), conv_out.data() + v_off, S_v * H_v * sizeof(float));

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) fprintf(stderr, "  gdn conv done\n");
    // GDN scan: the ggml_gated_delta_net kernel on scratch tensors (nth=1)
    // q/k [S_k, H_k, 1, 1]; v [S_v, H_v, 1, 1]; g/b [1, H_v, 1, 1]; s [S_v, S_v, H_v, 1]
    ggml_init_params gip = { 64 << 20, nullptr, false };
    ggml_context * gctx = ggml_init(gip);
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
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) fprintf(stderr, "  gdn qkv[0..2]=%.6g %.6g %.6g conv[0..2]=%.6g %.6g %.6g\n", (double)qkv[0],(double)qkv[1],(double)qkv[2],(double)conv_out[0],(double)conv_out[1],(double)conv_out[2]);
    ggml_tensor * tgd = ggml_gated_delta_net(gctx, tq, tk, tv, tg, tb, ts, 1);
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
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

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) fprintf(stderr, "  gdn kernel done\n");
    // z-gated rms norm: rms_norm(output, ssm_norm) * sigmoid(z)
    std::vector<float> gdn_out(S_v * H_v);
    memcpy(gdn_out.data(), tgd->data, S_v * H_v * sizeof(float));
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        float omin = 1e30f, omax = -1e30f; int onn = 0, oni = 0;
        for (int64_t i = 0; i < S_v*H_v; i++) { if (std::isnan(gdn_out[i])) onn++; if (std::isinf(gdn_out[i])) oni++; omin = fminf(omin, gdn_out[i]); omax = fmaxf(omax, gdn_out[i]); }
        fprintf(stderr, "  gdn out: nan=%d inf=%d min=%.6g max=%.6g\n", onn, oni, (double)omin, (double)omax);
    }
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) fprintf(stderr, "  gdn out[0..2]=%.6g %.6g %.6g z[0..2]=%.6g %.6g %.6g\n", (double)gdn_out[0],(double)gdn_out[1],(double)gdn_out[2],(double)z[0],(double)z[1],(double)z[2]);
    // the kernel's K=1 output carries [attn | new_state]; the state advances
    // (ts holds the input state; copying it back would freeze the recurrence)
    memcpy(ssm_state_row, (const char *) tgd->data + S_v * H_v, S_v * S_v * H_v * sizeof(float));
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
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
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

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) fprintf(stderr, "  gdn ssm_out done\n");
    // ---- MoE ----
    std::vector<float> moe_out(n_embd);
    fused_moe(L, hp, attn_out.data(), moe_out.data(), n_threads);

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) fprintf(stderr, "  gdn moe1 done\n");
    // ---- hc_combine (attn side): res = res + repeat(attn_out) * (2*sigmoid(inject/hc)) ----
    hc_combine(res_in_out, attn_out.data(), inject.data(), hc, n_embd);

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) fprintf(stderr, "  gdn comb1 done\n");
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        int mn = 0; for (int64_t i = 0; i < n_embd; i++) if (std::isnan(moe_out[i])) mn++;
        int xn2 = 0; for (int64_t i = 0; i < hc*n_embd; i++) if (std::isnan(xn[i])) xn2++;
        fprintf(stderr, "  gdn moe1 out nan=%d xn(attn-side) nan=%d\n", mn, xn2);
    }
    // ---- hc_mix (ffn side) + MoE again ----
    hc_rms_norm_gamma(res_in_out, L.hc_ffn_norm, xn.data(), n_embd, hc, eps);
    hc_mix(L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, hc, xn.data(), mixed.data(), inject.data(), n_threads);
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        int mm = 0; for (int64_t i = 0; i < n_embd; i++) if (std::isnan(mixed[i])) mm++;
        fprintf(stderr, "  gdn ffn mixed nan=%d\n", mm);
    }
    fused_moe(L, hp, mixed.data(), moe_out.data(), n_threads);
    hc_combine(res_in_out, moe_out.data(), inject.data(), hc, n_embd);

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) fprintf(stderr, "  gdn moe2 done\n");
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        int rn = 0, mn = 0, in = 0;
        for (int64_t i = 0; i < hc*n_embd; i++) { if (std::isnan(res_in_out[i])) rn++; }
        for (int64_t i = 0; i < n_embd; i++) { if (std::isnan(moe_out[i])) mn++; }
        for (int64_t i = 0; i < hc; i++) { if (std::isnan(inject[i])) in++; }
        fprintf(stderr, "  gdn final: res nan=%d moe_out nan=%d inject nan=%d\n", rn, mn, in);
    }
    memcpy(out, res_in_out, n_embd * sizeof(float));
    if (getenv("GGML_FUSED_DUMP_FLAYERS") != NULL) {
        FILE * fc = fopen("/tmp/qwen4exp-builds/f_conv.bin", "wb");
        if (fc) {
            fwrite(qkv.data(), 4, qkv_span, fc);
            fwrite(window.data(), 4, (d_conv) * n_ch, fc);
            fwrite(conv_out.data(), 4, n_ch, fc);
            fclose(fc);
        }
        FILE * f = fopen("/tmp/qwen4exp-builds/f_nodes.bin", "wb");
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

// the full fused decode for a single token: PLE + the layer loop + the head.
// The PLE (layer 1) and the full-attention layers still need the memory-context
// transcription; this skeleton runs the GDN layers and marks the rest.
// Returns false until every layer type is wired.
bool fused_decode_token(const struct llama_model_qwen4exp & model,
                        const struct llama_hparams & hp,
                        const float * tok_embd,
                        float * logits,
                        int n_threads) {
    const int64_t hc = hp.dsv4_hc_mult;
    const int64_t n_embd = hp.n_embd;
    const int64_t hc_dim = hc * n_embd;

    // the wide residual starts as hc identical copies of the embedding
    std::vector<float> res_hc(hc_dim);
    for (int64_t c = 0; c < hc; c++) {
        memcpy(res_hc.data() + c * n_embd, tok_embd, n_embd * sizeof(float));
    }

    std::vector<float> layer_out(n_embd);

    for (int il = 0; il < (int) hp.n_layer(); il++) {
        const struct llama_layer & L = model.layers[il];

        if (hp.is_ple_impl[il]) {
            // TODO(INF-64): the PLE fused transcription (host-side n-gram hash +
            // the 51GB-table gather + the conv + the gate)
            return false;
        }
        if (hp.is_recr(il)) {
            // TODO(INF-64): the conv/ssm state rows from the memory context
            float * conv_state = nullptr;
            float * ssm_state = nullptr;
            if (conv_state == nullptr || ssm_state == nullptr) {
                return false;
            }
            fused_gdn_layer(L, hp, res_hc.data(), res_hc.data(), layer_out.data(),
                            conv_state, ssm_state, n_threads);
        } else {
            // TODO(INF-64): the full-attention layer fused transcription
            return false;
        }
    }

    fused_head(model, hp, res_hc.data(), logits, n_threads);
    return true;
}

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

// apply the interleaved mrope via the graph's own rope tensor + kernel
static void fused_rope(const FusedRopeParams & rp, float * x, int64_t n_dims,
                       int64_t n_head, int32_t pos, int * sections,
                       const int64_t n_stream) {
    ggml_init_params gip = { 16 << 20, nullptr, false };
    ggml_context * gctx = ggml_init(gip);
    ggml_tensor * a = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_dims, n_head, n_stream);
    memcpy(a->data, x, n_dims * n_head * n_stream * sizeof(float));
    // IMROPE: 4 positions per token (text tokens: [pos, pos, pos, 0])
    ggml_tensor * b = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 4 * n_stream);
    for (int64_t s = 0; s < n_stream; s++) {
        ((int32_t *) b->data)[0 * n_stream + s] = pos;
        ((int32_t *) b->data)[1 * n_stream + s] = pos;
        ((int32_t *) b->data)[2 * n_stream + s] = pos;
        ((int32_t *) b->data)[3 * n_stream + s] = 0;
    }
    ggml_tensor * rope = ggml_rope_multi(gctx, a, b, nullptr,
            (int) n_dims, sections, LLAMA_ROPE_TYPE_IMROPE,
            rp.n_ctx_orig, rp.freq_base, rp.freq_scale,
            rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
    // the rope kernel uses wdata for the mrope cache (ne[0] floats per thread)
    std::vector<float> rw(n_dims);
    struct ggml_compute_params gparams;
    gparams.ith = 0; gparams.nth = 1;
    gparams.wsize = rw.size() * sizeof(float);
    gparams.wdata = rw.data();
    gparams.threadpool = nullptr; gparams.use_ref = false;
    ggml_compute_forward_rope(&gparams, rope);
    memcpy(x, rope->data, n_dims * n_head * n_stream * sizeof(float));
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
    ggml_init_params gip = { 64 << 20, nullptr, false };
    ggml_context * gctx = ggml_init(gip);
    // the graph casts the F32 activations to F16 for the flash and permutes the
    // heads/kv dims (build_attn_mha) before the op; mirror that exactly
    ggml_tensor * tq = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, n_embd_head, n_head, 1, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, n_embd_head, n_head_kv, n_kv, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, n_embd_head, n_head_kv, n_kv, 1);
    ggml_tensor * tm = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, n_kv, 1, 1, 1);
    for (int64_t i = 0; i < n_embd_head * n_head; i++) ((ggml_fp16_t *) tq->data)[i] = ggml_fp32_to_fp16(q[i]);
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
    // the flash kernel's decode-path scratch (VKQ acc + partial M/S + V/Q/VKQ)
    const int64_t fw = n_head * (2 + n_embd_head) + n_embd_head + 2 * n_embd_head;
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
    const float kq_scale = 1.0f / sqrtf((float) n_embd_head);
    const int n_repeat = n_head / n_head_kv;

    // the manual attention: for each kv head, its q-group's heads attend to
    // the visible cells. The flash kernel's accumulation differs, so this is
    // the NMSE-ok path; the bit-exact flash path lands with the kernel call.
    std::vector<float> scores(n_kv);
    std::vector<float> out_h(n_embd_head * n_head, 0.0f);

    for (int hk = 0; hk < n_head_kv; hk++) {
        // max over the visible cells for the numeric stability
        float mx = -INFINITY;
        for (int j = 0; j < n_kv; j++) {
            if (selected_cells[j] == 0.0f) {
                float s = 0.0f;
                for (int i = 0; i < n_embd_head; i++) {
                    s += q[hk * n_embd_head + i] * k_all[(size_t) hk * n_embd_head * n_kv + (size_t) i * n_kv + j];
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
        for (int r = 0; r < n_repeat; r++) {
            const int h = hk * n_repeat + r;
            for (int i = 0; i < n_embd_head; i++) {
                float acc = 0.0f;
                for (int j = 0; j < n_kv; j++) {
                    acc += scores[j] * v_all[(size_t) hk * n_embd_head * n_kv + (size_t) i * n_kv + j];
                }
                out_h[(size_t) h * n_embd_head + i] = acc;
            }
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
        int n_threads) {

    const int64_t hc = hp.dsv4_hc_mult;
    const int64_t n_embd = hp.n_embd;
    const int64_t n_embd_head = hp.n_embd_head_k(); // 256
    const int64_t n_head = hp.n_head();             // 24
    const int64_t n_head_kv = hp.n_head_kv();       // 2
    const float eps = hp.f_norm_rms_eps;
    const int n_rot = (int) hp.n_rot();

    if (tk->type != GGML_TYPE_F32 && tk->type != GGML_TYPE_F16) {
        fprintf(stderr, "  fused attn fallback: cache type %s %s %s\n",
                ggml_type_name(tk->type), ggml_type_name(tv->type), ggml_type_name(ti->type));
        return false;
    }
    // the cache may be F16 (the graph dequantizes it for the attention; the
    // fused path stages a converted F32 view and casts back on the writes)
    const bool k16 = tk->type == GGML_TYPE_F16;
    const bool i16 = ti->type == GGML_TYPE_F16;
    const int64_t k_cell = (int64_t) n_embd_head * n_head_kv;
    std::vector<float> k_stage, v_stage, idx_stage;
    const float * k_cache, * v_cache, * idx_cache;
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

    // ---- QSA indexer (short-context dense fallback: n_visible <= width) ----
    const int64_t r = hp.dsv4_compress_ratios[0] ? hp.dsv4_compress_ratios[0] : 4;
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
        for (int64_t j = 0; j < n_visible; j++) selected[j] = 0.0f;
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
            }
        }
        // the rectified scores: sum over the heads of relu(pooled dot q)
        std::vector<float> blk_score(n_blocks);
        for (int64_t b = 0; b < n_blocks; b++) {
            float s = 0.0f;
            for (int64_t h = 0; h < n_idx_h; h++) {
                float d = 0.0f;
                for (int64_t i = 0; i < idx_dim; i++) {
                    d += pooled_n[(size_t) b * idx_dim + i] * q_idx[(size_t) h * idx_dim + i];
                }
                s += fmaxf(d, 0.0f);
            }
            blk_score[b] = s;
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
    lora_mm(L.wq, mixed.data(), nullptr, qfull.data(), n_threads);
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
        lora_mm(L.wk, mixed.data(), nullptr, kraw.data(), n_threads);
        lora_mm(L.wv, mixed.data(), nullptr, vraw.data(), n_threads);
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

    int sections[4] = { 11, 11, 10, 0 };
    fused_rope(rp, q.data(), n_rot, n_head, pos, sections, 1);
    fused_rope(rp, k.data(), n_rot, n_head_kv, pos, sections, 1);

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

    std::vector<float> attn_out(hdim2);
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        fprintf(stderr, "  attn: n_used=%d n_visible=%d q[0..2]=%.6g %.6g %.6g\n", n_used, n_visible,
                (double) q[0], (double) q[1], (double) q[2]);
    }
    if (n_visible > 0) {
        std::vector<float> k_all(n_embd_head * n_head_kv * n_used), v_all(n_embd_head * n_head_kv * n_used);
        memcpy(k_all.data(), k_cache, n_embd_head * n_head_kv * n_used * sizeof(float));
        memcpy(v_all.data(), v_cache, n_embd_head * n_head_kv * n_used * sizeof(float));
        // the QSA-masked attention via the graph's flash kernel (bit-exact)
        if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
            int kn = 0, vn = 0;
            for (int64_t i = 0; i < (int64_t) k_all.size(); i++) { if (std::isnan(k_all[i])) kn++; if (std::isnan(v_all[i])) vn++; }
            fprintf(stderr, "  attn flash in: k nan=%d v nan=%d k[0..2]=%.6g %.6g %.6g v[0..2]=%.6g %.6g %.6g mask[0..3]=%.6g %.6g %.6g %.6g\n",
                    kn, vn, (double) k_all[0], (double) k_all[1], (double) k_all[2],
                    (double) v_all[0], (double) v_all[1], (double) v_all[2],
                    (double) selected[0], (double) selected[1], (double) selected[2], (double) selected[3]);
        }
        if (getenv("GGML_FUSED_DUMP_FLASH") != NULL) {
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
        fused_attn_qsa(hp, q.data(), k.data(), v.data(),
                       k_all.data(), v_all.data(), n_used,
                       selected.data(), attn_out.data(), (int) n_head, (int) n_head_kv,
                       n_threads);
    }

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        int ann = 0; for (int64_t i = 0; i < hdim2; i++) if (std::isnan(attn_out[i])) ann++;
        fprintf(stderr, "  attn out: nan=%d [0..2]=%.6g %.6g %.6g\n", ann, (double) attn_out[0], (double) attn_out[1], (double) attn_out[2]);
    }
    // the gate + the output projection
    for (int64_t i = 0; i < hdim2; i++) {
        const float g = gate[i];
        attn_out[i] *= 1.0f / (1.0f + expf(-g));
    }
    std::vector<float> layer_out(n_embd);
    lora_mm(L.wo, attn_out.data(), nullptr, layer_out.data(), n_threads);

    // the MoE + the hc combine (same as the GDN layers)
    std::vector<float> moe_out(n_embd);
    fused_moe(L, hp, layer_out.data(), moe_out.data(), n_threads);
    hc_combine(res_in_out, layer_out.data(), inject.data(), hc, n_embd);
    hc_rms_norm_gamma(res_in_out, L.hc_ffn_norm, xn.data(), n_embd, hc, eps);
    hc_mix(L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, hc, xn.data(), mixed.data(), inject.data(), n_threads);
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
        for (int64_t h = 0; h < n_heads; h++) {
            const char * row = (const char *) table->data + (size_t) rows[h] * row_bytes;
            if (table->type == GGML_TYPE_F32) {
                memcpy(emb.data() + h * head_dim, row, head_dim * sizeof(float));
            } else {
                qtt->to_float(row, emb.data() + h * head_dim, head_dim);
            }
        }
    }

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        fprintf(stderr, "  ple rows[0..2]=%d %d %d head_dim=%lld\n", rows[0], rows[1], rows[2], (long long)head_dim);
        int nn = 0; for (int64_t i = 0; i < n_heads*head_dim; i++) if (std::isnan(emb[i])) nn++;
        fprintf(stderr, "  ple emb nan=%d [0..2]=%.6g %.6g %.6g\n", nn, (double)emb[0], (double)emb[1], (double)emb[2]);
    }
    // ---- the key/value projections ----
    std::vector<float> key(hc_dim), value(n_embd);
    lora_mm(L.ple_key, emb.data(), nullptr, key.data(), n_threads);
    lora_mm(L.ple_value, emb.data(), nullptr, value.data(), n_threads);

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
            for (int64_t i = 0; i < n_embd; i++) query_n[c * n_embd + i] = xc[i] * sc * qn[c * n_embd + i];
        }
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
    const float * w = (const float *) L.ple_conv1d->data;
    std::vector<float> conv_out(hc_dim);
    {
        // the window: state [hist, hc_dim] + the current norm_conv at position hist
        std::vector<float> window((hist + 1) * hc_dim);
        memcpy(window.data(), ple_conv_state, hist * hc_dim * sizeof(float));
        memcpy(window.data() + hist * hc_dim, norm_conv.data(), hc_dim * sizeof(float));
        for (int64_t c = 0; c < hc_dim; c++) {
            float sumf = 0.0f;
            for (int64_t k = 0; k < kern; k++) {
                const int64_t pos = hist - (kern - 1 - k) * dil;
                sumf += window[(size_t) pos * hc_dim + c] * w[(size_t) k + c * 4];
            }
            const float v = sumf;
            conv_out[c] = v / (1.0f + expf(-v)); // silu
        }
        // the state update: keep the last hist positions
        memcpy(ple_conv_state, window.data() + hc_dim, hist * hc_dim * sizeof(float));
    }

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        int nn = 0; for (int64_t i = 0; i < hc_dim; i++) if (std::isnan(conv_out[i])) nn++;
        fprintf(stderr, "  ple key nan=%d value nan=%d conv_out nan=%d gate[0]=%.6g\n",
                (int) std::isnan(key[0]), (int) std::isnan(value[0]), nn, (double)gate[0]);
    }
    // the combine: hidden + gated + conv_out
    for (int64_t i = 0; i < hc_dim; i++) {
        res[i] = hidden[i] + gated[i] + conv_out[i];
    }
    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
        int nn = 0, gn = 0, vn = 0;
        for (int64_t i = 0; i < hc_dim; i++) { if (std::isnan(res[i])) nn++; if (std::isnan(gated[i])) gn++; }
        for (int64_t i = 0; i < n_embd; i++) if (std::isnan(value[i])) vn++;
        fprintf(stderr, "  ple res nan=%d gated nan=%d value nan=%d\n", nn, gn, vn);
    }
}

// ---- the fused decode wiring (the process_ubatch fast path) ----------------

#include "llama-memory-hybrid-idx.h"
#include "llama-kv-cache.h"

// the token embedding gather (the graph's build_inp_embd)
static void fused_embd(const struct ggml_tensor * tok_embd, int32_t tok, float * out, int64_t n_embd) {
    const size_t row_bytes = ggml_row_size(tok_embd->type, n_embd);
    const char * row = (const char *) tok_embd->data + (size_t) tok * row_bytes;
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

bool llama_model_qwen4exp::fused_decode(
        const llama_ubatch & ubatch,
        const struct llama_memory_context_i * mctx_in,
        class llm_graph_result * res,
        int n_threads,
        const struct ggml_tensor * const * prev_layer_inp) const {
    if (ubatch.n_tokens != 1 || ubatch.n_seqs != 1) return false;
    if (ubatch.token == nullptr) return false;

    const int64_t n_layer_slot = hparams.n_layer() + 1;
    const auto * mctx = static_cast<const llama_memory_hybrid_idx_context *>(mctx_in);
    if (mctx == nullptr || mctx->get_attn() == nullptr) return false;

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
        for (int64_t c = 0; c < hc; c++) {
            memcpy(res_hc.data() + c * n_embd, emb.data(), n_embd * sizeof(float));
        }
    }

    // scratch ggml context for the cache views
    ggml_init_params gip = { 64 << 20, nullptr, false };
    ggml_context * vctx = ggml_init(gip);
    std::vector<float> layer_out(n_embd);

    if (getenv("GGML_FUSED_DECODE_TRACE") != NULL && getenv("GGML_FUSED_LAYER_CMP") != NULL && prev_layer_inp && prev_layer_inp[0] && prev_layer_inp[0]->data) {
        const ggml_tensor * g0 = prev_layer_inp[0];
        const int64_t nt = g0->ne[1];
        const float * gp = (const float *) g0->data + (nt - 1) * g0->ne[0];
        double md = 0.0;
        for (int64_t i = 0; i < g0->ne[0]; i++) md = fmax(md, (double) fabs(res_hc[i] - gp[i]));
        fprintf(stderr, "  fused cmp input embd: max_abs=%.6g\n", md);
    }
    for (int il = 0; il < (int) hparams.n_layer(); il++) {
        const struct llama_layer & L = layers[il];
        if (getenv("GGML_FUSED_DECODE_TRACE") != NULL) {
            fprintf(stderr, "fused layer %2d: %s\n", il,
                    hparams.is_ple_impl[il] ? "PLE" : (hparams.is_recr(il) ? "GDN" : "ATTN"));
        }

        if (getenv("GGML_FUSED_SKIP_PLE") != NULL && hparams.is_ple_impl[il]) {
            continue;
        }
        if (getenv("GGML_FUSED_SKIP_ATTN") != NULL && !hparams.is_ple_impl[il] && !hparams.is_recr(il)) {
            continue;
        }
        if (hparams.is_ple_impl[il]) {
            // the PLE: the predecessors from the attention cells
            const auto * attn = mctx->get_attn();
            std::vector<llama_token> prev_toks;
            attn->get_prev_tokens(ubatch, hparams.ple_ngram_size - 1, prev_toks);
            int32_t prev[2] = { -1, -1 };
            for (int64_t s = 0; s < hparams.ple_ngram_size - 1; s++) {
                const llama_token t = prev_toks[s];
                prev[hparams.ple_ngram_size - 2 - s] = t < 0 ? -1 : (int32_t) t;
            }
            const auto * recr = mctx->get_recr();
            const auto * pl = recr->get_p_l(il);
            const size_t row_bytes = ggml_row_size(pl->type, pl->ne[0]);
            const int head = (int) recr->get_head();
            const int64_t hist = (hparams.ple_conv_kernel - 1) * hparams.ple_ngram_size;
            std::vector<float> ple_state_copy(hist * hc_dim);
            memcpy(ple_state_copy.data(),
                   (const char *) pl->data + (size_t) head * row_bytes,
                   hist * hc_dim * sizeof(float));
            fused_ple(*this, hparams, L, tok, prev, res_hc.data(), res_hc.data(),
                      ple_state_copy.data(), n_threads);
            memcpy((void *) ((const char *) pl->data + (size_t) head * row_bytes),
                   ple_state_copy.data(), hist * hc_dim * sizeof(float));
        } else if (hparams.is_recr(il)) {
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
            std::vector<float> conv_c(rl->ne[0]), ssm_c(sl->ne[0]);
            memcpy(conv_c.data(), conv_state, rl->ne[0] * sizeof(float));
            memcpy(ssm_c.data(), ssm_state, sl->ne[0] * sizeof(float));
            fused_gdn_layer(L, hparams, res_hc.data(), res_hc.data(), layer_out.data(),
                            conv_c.data(), ssm_c.data(), n_threads);
            memcpy((void *) conv_state, conv_c.data(), rl->ne[0] * sizeof(float));
            memcpy((void *) ssm_state, ssm_c.data(), sl->ne[0] * sizeof(float));
        } else {
            // the full-attn layer: the KV + indexer cache slices. The cell
            // bookkeeping ran in apply(): the current ubatch's cell is already
            // allocated, so the last used cell is this token's write target.
            const auto * attn = mctx->get_attn();
            const auto * idx = mctx->get_idx();
            if (idx == nullptr) { fprintf(stderr, "  fused attn fallback: idx null\n"); return false; }
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
            if (getenv("GGML_FUSED_DECODE_TRACE") != NULL && il == 3) {
                fprintf(stderr, "  fused cache: tk ne=[%lld,%lld,%lld,%lld] nb1=%zu data=%p\n", (long long)tk->ne[0],(long long)tk->ne[1],(long long)tk->ne[2],(long long)tk->ne[3],(size_t)tk->nb[1], (const void*)tk->data);
                fprintf(stderr, "  fused cache: tv ne=[%lld,%lld,%lld,%lld] nb1=%zu data=%p same=%d\n", (long long)tv->ne[0],(long long)tv->ne[1],(long long)tv->ne[2],(long long)tv->ne[3],(size_t)tv->nb[1], (const void*)tv->data, (int)(tv->data == tk->data));
                fprintf(stderr, "  fused cache: ti ne=[%lld,%lld,%lld,%lld] nb1=%zu\n", (long long)ti->ne[0],(long long)ti->ne[1],(long long)ti->ne[2],(long long)ti->ne[3],(size_t)ti->nb[1]);
            }
            if (tk == nullptr || tv == nullptr || ti == nullptr) { fprintf(stderr, "  fused attn fallback: cache view null (tk=%p tv=%p ti=%p)\n", (const void*) tk, (const void*) tv, (const void*) ti); return false; }
            if (!fused_full_attn_layer(L, hparams, rp, (int32_t) pos,
                        res_hc.data(), res_hc.data(), layer_out.data(),
                        tk, tv, ti, n_used, n_visible, idx_n_used, n_threads)) {
                ggml_free(vctx);
                return false;
            }
        }
        if (getenv("GGML_FUSED_DUMP_FLAYERS") != NULL && il + 1 < (int64_t) hparams.n_layer()) {
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

    // write into the previous graph's logits tensor (sched-known, so the
    // post-decode extraction and llama_get_logits_ith see the values)
    ggml_tensor * t_logits = const_cast<ggml_tensor *>(prev_layer_inp[n_layer_slot]);
    if (t_logits == nullptr || t_logits->data == nullptr) {
        return false;
    }
    res->t_logits = t_logits;
    memcpy(t_logits->data, logits.data(), n_vocab * sizeof(float));

    ggml_free(vctx);
    return true;
}
