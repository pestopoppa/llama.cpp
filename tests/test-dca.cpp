// Dual Chunk Attention (DCA): the llama graph vs a double-precision reference.
//
// DCA (ChunkLlama, arXiv 2402.17463; Qwen2.5-1M) ropes every key at its chunk-local
// position k = p mod c (c = chunk_size - local_size) and attends each query with one
// of three remapped positions, chosen by the chunk distance of the key:
//   same chunk        (intra): q = p mod c
//   previous chunk    (succ) : q = min(p mod c + c, chunk_size)
//   any earlier chunk (inter): q = min(2c - 1, chunk_size)  (= chunk_size unless local >= c)
// with one softmax over all causal keys (equal to vLLM's LSE merge of the three
// partial attentions). With dca_orig_ctx > 0 the logits get the temperature
// s = max(1, 0.1 ln((p+1)/orig) + 1).
//
// Tests (synthetic models from the test-llama-archs fixture, CPU):
//  1. numeric, qwen2 (NEOX rope): prefill in misaligned ubatches, then token-by-token
//     decode; the eval callback captures layer 0's unroped q/k/v (dca_q_in/k_in/v_in)
//     and the attention output (dca_out); a double-precision reference recomputes DCA
//     from the captured q/k/v and must match. A plain-rope reference must NOT match
//     (the guard that positions beyond chunk_len are actually remapped).
//  2. equivalence, qwen2 / qwen35 / qwen35moe: DCA with chunk_len > n_ctx is plain
//     attention, so the logits must equal the non-DCA run.
//  3. smoke, qwen35 / qwen35moe (IMROPE hybrids): DCA with a small chunk runs, gives
//     finite logits and differs from the plain run.

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// norm weights near 1, output scales at 1 and q/k projections of std 0.15 give O(1) attention logits, so
// the softmax is peaked enough for the rope geometry to matter; everything else ~1e-2
static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    const char * n = tensor->name;
    const bool is_norm = strstr(n, "norm") != nullptr;
    const bool is_qk   = strstr(n, "attn_q.weight") || strstr(n, "attn_k.weight") || strstr(n, "attn_qkv.weight");
    const bool is_scl  = strstr(n, ".scale") != nullptr; // per-tensor output scales (build_lora_mm's w_s): keep at 1
    std::normal_distribution<float> dis_n(is_norm ? 1.0f : 0.0f, is_norm ? 0.05f : (is_qk ? 0.15f : 1.0e-2f));
    auto dis = [&](std::mt19937 & g) { return is_scl ? 1.0f : dis_n(g); };

    if (getenv("DCA_TEST_VERBOSE")) {
        fprintf(stderr, "init %-40s %s norm=%d qk=%d\n", n, ggml_type_name(tensor->type), (int) is_norm, (int) is_qk);
    }
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// ---- the fixture of tests/test-llama-archs.cpp (verbatim) ----
static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const uint32_t n_layer_nextn = 0) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 128;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    }

    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    if (n_layer_nextn > 0) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, n_layer_nextn);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_LAGUNA) {
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,               10000.0f);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT_SWA,         n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));
    }

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_DEEPSEEK4 ? n_head : uint32_t(1));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(2)); // sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

// ---- capture ----------------------------------------------------------------

struct capture {
    // per name: the rows of every graph evaluation, appended in token order
    std::map<std::string, std::vector<float>> data;
    std::map<std::string, std::vector<int64_t>> shape; // ne0, ne1 of one token's slice
};

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    static const char * names[] = { "dca_q_in-0", "dca_k_in-0", "dca_v_in-0", "dca_out-0" };
    bool want = false;
    for (const char * nm : names) {
        want = want || strcmp(t->name, nm) == 0;
    }
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    auto * cap = (capture *) user_data;
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    // read with strides: q/k/v are [D, H, T] (possibly views), out is [D*H, T]
    const bool is_3d = t->ne[2] > 1 || strcmp(t->name, "dca_out-0") != 0;
    const int64_t ne0 = t->ne[0], ne1 = is_3d ? t->ne[1] : 1, nt = is_3d ? t->ne[2] : t->ne[1];
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    auto & dst = cap->data[t->name];
    for (int64_t it = 0; it < nt; it++) {
        for (int64_t i1 = 0; i1 < ne1; i1++) {
            for (int64_t i0 = 0; i0 < ne0; i0++) {
                const size_t off = is_3d ? i0*t->nb[0] + i1*t->nb[1] + it*t->nb[2] : i0*t->nb[0] + it*t->nb[1];
                float v;
                memcpy(&v, raw.data() + off, sizeof(float));
                dst.push_back(v);
            }
        }
    }
    cap->shape[t->name] = { ne0, ne1 };
    return true;
}

// ---- reference --------------------------------------------------------------

// ggml NEOX rope: pair (i, i + n_rot/2), theta_i = p * base^(-2i/n_rot)
static void rope_neox(const float * x, double * out, int64_t D, int n_rot, double p, double base) {
    for (int64_t i = 0; i < D; i++) {
        out[i] = x[i];
    }
    const int half = n_rot/2;
    for (int i = 0; i < half; i++) {
        const double th = p * std::pow(base, -2.0*i/n_rot);
        const double c = std::cos(th), s = std::sin(th);
        const double x0 = x[i], x1 = x[i + half];
        out[i]        = x0*c - x1*s;
        out[i + half] = x0*s + x1*c;
    }
}

struct dca_cfg {
    bool   on;
    int    chunk_size;
    int    local_size;
    int    orig_ctx;
};

// out[t][h][d] for every captured token; q [T][Hq][D], k/v [T][Hkv][D]
static std::vector<double> reference(const capture & cap, int64_t T, const dca_cfg & dc, int n_rot, double base, double kq_scale) {
    const auto & q = cap.data.at("dca_q_in-0");
    const auto & k = cap.data.at("dca_k_in-0");
    const auto & v = cap.data.at("dca_v_in-0");
    const int64_t D   = cap.shape.at("dca_q_in-0")[0];
    const int64_t Hq  = cap.shape.at("dca_q_in-0")[1];
    const int64_t Hkv = cap.shape.at("dca_k_in-0")[1];
    GGML_ASSERT((int64_t) q.size() == T*Hq*D && (int64_t) k.size() == T*Hkv*D);
    const int c  = dc.chunk_size - dc.local_size;
    const int cs = dc.chunk_size;

    auto kpos = [&](int64_t p) -> double { return dc.on ? (double) (p % c) : (double) p; };
    auto qpos = [&](int64_t pq, int64_t pk) -> double {
        if (!dc.on) {
            return (double) pq;
        }
        const int64_t cq = pq / c, ck = pk / c;
        if (cq == ck) {
            return (double) (pq % c);
        }
        if (cq == ck + 1) {
            return (double) std::min<int64_t>(pq % c + c, cs);
        }
        return (double) std::min(2*c - 1, cs);
    };

    // keys roped once (chunk-local)
    std::vector<double> kr(T*Hkv*D);
    for (int64_t t = 0; t < T; t++) {
        for (int64_t h = 0; h < Hkv; h++) {
            rope_neox(&k[(t*Hkv + h)*D], &kr[(t*Hkv + h)*D], D, n_rot, kpos(t), base);
        }
    }

    std::vector<double> out(T*Hq*D, 0.0);
    std::vector<double> qr(D), logit(T);
    for (int64_t t = 0; t < T; t++) {
        const double s = (dc.on && dc.orig_ctx > 0) ? std::max(1.0, 0.1*std::log((t + 1.0)/dc.orig_ctx) + 1.0) : 1.0;
        for (int64_t h = 0; h < Hq; h++) {
            const int64_t hk = h / (Hq/Hkv);
            double mx = -INFINITY;
            for (int64_t j = 0; j <= t; j++) {
                rope_neox(&q[(t*Hq + h)*D], qr.data(), D, n_rot, qpos(t, j), base);
                double dot = 0.0;
                for (int64_t d = 0; d < D; d++) {
                    dot += qr[d]*kr[(j*Hkv + hk)*D + d];
                }
                logit[j] = dot*kq_scale*s;
                mx = std::max(mx, logit[j]);
            }
            double sum = 0.0;
            for (int64_t j = 0; j <= t; j++) {
                logit[j] = std::exp(logit[j] - mx);
                sum += logit[j];
            }
            for (int64_t j = 0; j <= t; j++) {
                const double w = logit[j]/sum;
                for (int64_t d = 0; d < D; d++) {
                    out[(t*Hq + h)*D + d] += w*v[(j*Hkv + hk)*D + d];
                }
            }
        }
    }
    return out;
}

static double rel_err(const std::vector<double> & ref, const std::vector<float> & x) {
    GGML_ASSERT(ref.size() == x.size());
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        num += (ref[i] - x[i])*(ref[i] - x[i]);
        den += ref[i]*ref[i];
    }
    return std::sqrt(num/den);
}

// ---- driver -----------------------------------------------------------------

static llama_model_ptr make_model(llm_arch arch, bool moe, size_t seed) {
    gguf_context_ptr gctx = get_gguf_ctx(arch, moe);
    llama_model_params mp = llama_model_default_params();
    // DCA_TEST_DEVICE=<name> (e.g. ROCm0) runs the model on that device; default CPU
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (const char * name = getenv("DCA_TEST_DEVICE")) {
        dev = ggml_backend_dev_by_name(name);
        if (!dev) {
            throw std::runtime_error(std::string("no device ") + name);
        }
    }
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    mp.devices = devs;
    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gctx.get(), set_tensor_data, &tmp, mp));
    if (!model) {
        throw std::runtime_error("failed to create the model");
    }
    return model;
}

// prefill n_prompt tokens (n_ubatch-sized ubatches), then decode the rest one by one;
// returns the logits of every decoded token (and of the last prompt token)
static std::vector<float> run(llama_model * model, const std::vector<llama_token> & toks, size_t n_prompt,
                              const dca_cfg & dc, capture * cap, uint32_t n_ubatch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 128;
    cp.n_ubatch = n_ubatch;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    if (dc.on) {
        cp.dca_chunk_size = dc.chunk_size;
        cp.dca_local_size = dc.local_size;
        cp.dca_orig_ctx   = dc.orig_ctx;
    }
    if (cap) {
        cp.cb_eval = capture_cb;
        cp.cb_eval_user_data = cap;
    }
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) {
        throw std::runtime_error("context init failed");
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> logits;

    llama_batch batch = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < n_prompt; i++) {
        common_batch_add(batch, toks[i], (llama_pos) i, {0}, i + 1 == n_prompt);
    }
    if (llama_decode(ctx.get(), batch)) {
        throw std::runtime_error("prefill failed");
    }
    const float * l = llama_get_logits_ith(ctx.get(), -1);
    logits.insert(logits.end(), l, l + n_vocab);
    for (size_t i = n_prompt; i < toks.size(); i++) {
        common_batch_clear(batch);
        common_batch_add(batch, toks[i], (llama_pos) i, {0}, true);
        if (llama_decode(ctx.get(), batch)) {
            throw std::runtime_error("decode failed");
        }
        l = llama_get_logits_ith(ctx.get(), -1);
        logits.insert(logits.end(), l, l + n_vocab);
    }
    llama_batch_free(batch);
    return logits;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        m = std::max(m, (double) std::fabs(a[i] - b[i]));
    }
    return m;
}

int main(int argc, char ** argv) {
    common_init();
    size_t seed = argc > 1 ? std::stoull(argv[1]) : 42;
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 127);
    std::vector<llama_token> toks(96);
    for (auto & t : toks) {
        t = dis(gen);
    }
    const size_t n_prompt = 60;

    bool ok = true;

    // 1. numeric: qwen2, chunk_size 24, local 8 -> chunk_len 16; 96 tokens = 6 chunks;
    //    ubatch 13 so the prefill ubatches straddle chunk boundaries; temperature from 40
    {
        llama_model_ptr model = make_model(LLM_ARCH_QWEN2, false, seed);
        const int n_embd_head = 128; // fixture: n_embd 256 / n_head 2
        for (int orig : { 0, 40 }) {
            const dca_cfg dc = { true, 24, 8, orig };
            capture cap;
            run(model.get(), toks, n_prompt, dc, &cap, 13);
            const int64_t T = (int64_t) toks.size();
            const auto & got = cap.data.at("dca_out-0");
            const double base = 10000.0; // the fixture's rope base (model default)
            const double err_dca   = rel_err(reference(cap, T, dc, n_embd_head, base, 1.0/std::sqrt((double) n_embd_head)), got);
            const dca_cfg plain = { false, 0, 0, 0 };
            const double err_plain = rel_err(reference(cap, T, plain, n_embd_head, base, 1.0/std::sqrt((double) n_embd_head)), got);
            // DCA_TEST_DUMP=<dir>: write the captured tensors for tests/dca_reference.py --check-dump <dir>
            if (const char * dir = getenv("DCA_TEST_DUMP")) {
                const std::string d = std::string(dir) + "/orig" + std::to_string(orig);
                const std::string mk = "mkdir -p '" + d + "'";
                if (system(mk.c_str()) != 0) {
                    throw std::runtime_error("mkdir failed: " + d);
                }
                for (const char * nm : { "dca_q_in-0", "dca_k_in-0", "dca_v_in-0", "dca_out-0" }) {
                    const auto & x = cap.data.at(nm);
                    FILE * f = fopen((d + "/" + nm + ".f32").c_str(), "wb");
                    GGML_ASSERT(f);
                    fwrite(x.data(), sizeof(float), x.size(), f);
                    fclose(f);
                }
                FILE * f = fopen((d + "/meta.txt").c_str(), "w");
                GGML_ASSERT(f);
                fprintf(f, "T %lld\nHq %lld\nHkv %lld\nD %lld\nchunk_size %d\nlocal_size %d\norig_ctx %d\nbase %.1f\nn_rot %d\n",
                        (long long) T, (long long) cap.shape.at("dca_q_in-0")[1], (long long) cap.shape.at("dca_k_in-0")[1],
                        (long long) cap.shape.at("dca_q_in-0")[0], dc.chunk_size, dc.local_size, dc.orig_ctx, base, n_embd_head);
                fclose(f);
            }
            const bool pass = (int64_t) got.size() == T*256 && err_dca < 1e-5 && err_plain > 1e-2;
            printf("numeric  qwen2   chunk=24 local=8 orig=%-2d  rel_err vs DCA ref=%.2e  vs plain-rope ref=%.2e  %s\n",
                   orig, err_dca, err_plain, pass ? "OK" : "FAIL");
            ok = ok && pass;
        }
    }

    // 2./3. equivalence with a chunk larger than the context, and a small-chunk smoke run
    struct arch_case { llm_arch arch; bool moe; };
    for (const arch_case ac : { arch_case{ LLM_ARCH_QWEN2, false }, arch_case{ LLM_ARCH_QWEN35, false }, arch_case{ LLM_ARCH_QWEN35MOE, true } }) {
        llama_model_ptr model = make_model(ac.arch, ac.moe, seed);
        const dca_cfg plain = { false, 0, 0, 0 };
        const dca_cfg big   = { true, 1024, 0, 0 };
        const dca_cfg small = { true, 24, 8, 0 };
        const auto l_plain = run(model.get(), toks, n_prompt, plain, nullptr, 13);
        const auto l_big   = run(model.get(), toks, n_prompt, big,   nullptr, 13);
        const auto l_small = run(model.get(), toks, n_prompt, small, nullptr, 13);
        const double d_big   = max_abs_diff(l_plain, l_big);
        const double d_small = max_abs_diff(l_plain, l_small);
        bool finite = true;
        for (float x : l_small) {
            finite = finite && std::isfinite(x);
        }
        double scale = 0.0;
        for (float x : l_plain) {
            scale = std::max(scale, (double) std::fabs(x));
        }
        const bool pass = d_big == 0.0 && finite && d_small > 1e-3*scale;
        printf("equiv    %-9s chunk>ctx max|d|=%.2e (must be 0)  chunk=24: max|d|=%.2e of max|l|=%.2e finite=%d  %s\n",
               llm_arch_name(ac.arch), d_big, d_small, scale, (int) finite, pass ? "OK" : "FAIL");
        ok = ok && pass;
    }

    return ok ? 0 : 1;
}
