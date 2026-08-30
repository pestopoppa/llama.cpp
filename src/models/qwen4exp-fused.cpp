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

    // the down expert gemvs: [640, 2560, 512] IQ4_NL with its vec_dot_type
    const struct ggml_type_traits_cpu * qt_dn = ggml_get_type_traits_cpu(L.ffn_down_exps->type);
    const struct ggml_type_traits_cpu * qt_dnv = ggml_get_type_traits_cpu(qt_dn->vec_dot_type);
    const size_t glu_q_size = ggml_row_size(qt_dn->vec_dot_type, n_ff);
    std::vector<uint8_t> glu_q(glu_q_size * n_used);
    for (int64_t j = 0; j < n_used; j++) {
        qt_dnv->from_float(glu.data() + j * n_ff, glu_q.data() + j * glu_q_size, n_ff);
    }
    const size_t nb_dn_exp = L.ffn_down_exps->nb[2];
    std::vector<float> down_acc(hp.n_embd, 0.0f);
    for (int64_t j = 0; j < n_used; j++) {
        const int32_t e = sel[j];
        const char * dn_e = (const char *) L.ffn_down_exps->data + (size_t) e * nb_dn_exp;
        for (int64_t r = 0; r < hp.n_embd; r++) {
            float v = 0.0f;
            qt_dn->vec_dot((int) n_ff, &v, 0, dn_e + (size_t) r * L.ffn_down_exps->nb[1], 0,
                           glu_q.data() + j * glu_q_size, 0, 1);
            down_acc[r] += v * w[j];
        }
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

// mirror of ggml_compute_forward_ssm_conv_f32 at batch-1 (float accumulation)
static void conv1d_4tap(const float * window, const float * kernel, float * out, int64_t n_ch) {
    for (int64_t i = 0; i < n_ch; i++) {
        float sumf = 0.0f;
        for (int k = 0; k < 4; k++) {
            sumf += window[k * n_ch + i] * kernel[k * n_ch + i];
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

    // ---- GDN body ----
    std::vector<float> qkv(hp.ssm_d_inner);            // 10240
    std::vector<float> z(hp.ssm_d_inner);              // 6144 (v-dim)
    lora_mm(L.wqkv, mixed.data(), nullptr, qkv.data(), n_threads);
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
    ggml_tensor * tgd = ggml_gated_delta_net(gctx, tq, tk, tv, tg, tb, ts, 1);

    static struct ggml_threadpool * gdn_tp = nullptr;
    if (gdn_tp == nullptr) {
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
        tpp.n_threads = 1;
        gdn_tp = ggml_threadpool_new(&tpp);
    }
    struct ggml_compute_params gparams;
    gparams.ith = 0;
    gparams.nth = 1;
    gparams.wsize = 0;
    gparams.wdata = nullptr;
    gparams.threadpool = gdn_tp;
    gparams.use_ref = false;
    ggml_compute_forward_gated_delta_net(&gparams, tgd);

    // z-gated rms norm: rms_norm(output, ssm_norm) * sigmoid(z)
    std::vector<float> gdn_out(S_v * H_v);
    memcpy(gdn_out.data(), tgd->data, S_v * H_v * sizeof(float));
    memcpy(ssm_state_row, ts->data, S_v * S_v * H_v * sizeof(float));
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
        const float zg = 1.0f / (1.0f + expf(-z[h * S_v]));
        for (int64_t i = 0; i < S_v; i++) {
            final_in[h * S_v + i] = row[i] * scale * znorm[h * S_v + i] * zg;
        }
    }

    std::vector<float> attn_out(n_embd);
    lora_mm(L.ssm_out, final_in.data(), nullptr, attn_out.data(), n_threads);

    // ---- MoE ----
    std::vector<float> moe_out(n_embd);
    fused_moe(L, hp, attn_out.data(), moe_out.data(), n_threads);

    // ---- hc_combine (attn side): res = res + repeat(attn_out) * (2*sigmoid(inject/hc)) ----
    hc_combine(res_in_out, attn_out.data(), inject.data(), hc, n_embd);

    // ---- hc_mix (ffn side) + MoE again ----
    hc_rms_norm_gamma(res_in_out, L.hc_ffn_norm, xn.data(), n_embd, hc, eps);
    hc_mix(L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, hc, xn.data(), mixed.data(), inject.data(), n_threads);
    fused_moe(L, hp, mixed.data(), moe_out.data(), n_threads);
    hc_combine(res_in_out, moe_out.data(), inject.data(), hc, n_embd);

    memcpy(out, res_in_out, n_embd * sizeof(float));
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
    ggml_tensor * b = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_stream);
    ((int32_t *) b->data)[0] = pos;
    ggml_tensor * rope = ggml_rope_multi(gctx, a, b, nullptr,
            (int) n_dims, sections, LLAMA_ROPE_TYPE_IMROPE,
            rp.n_ctx_orig, rp.freq_base, rp.freq_scale,
            rp.ext_factor, rp.attn_factor, rp.beta_fast, rp.beta_slow);
    struct ggml_compute_params gparams;
    gparams.ith = 0; gparams.nth = 1; gparams.wsize = 0; gparams.wdata = nullptr;
    gparams.threadpool = nullptr; gparams.use_ref = false;
    ggml_compute_forward_rope(&gparams, rope);
    memcpy(x, rope->data, n_dims * n_head * n_stream * sizeof(float));
    ggml_free(gctx);
}
