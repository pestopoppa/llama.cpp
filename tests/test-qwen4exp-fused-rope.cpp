// qwen4exp fused decode vs graph decode under the runtime rope parameters.
//
// The fused single-token decode (src/models/qwen4exp-fused.cpp) ropes the query,
// the key and the indexer itself. It must use the same rope geometry as the graph
// that prefilled the cache: freq_base, freq_scale, YaRN ext/attn factors, beta_*
// and n_ctx_orig_yarn. This test builds a tiny synthetic qwen4exp model, prefills
// the same prompt through the graph in two contexts, then decodes token by token
// with the fused path disabled (GGML_FUSED_DECODE_OFF=1) in one context and
// enabled in the other, and compares
//   - the roped K rows and the roped indexer rows that each decode step wrote
//     into the cache (the direct rope observable: a rope-geometry mismatch is an
//     O(1) relative error there), and
//   - the logits of every step.
// The fixture's weights are ~1e-2, where the fused transcription matches the
// graph to ~1e-11 nmse on the logits; at that scale the attention softmax is
// near uniform, so the logits alone cannot see the rope, hence the cache rows.
//
// It runs three rope configurations: native (no scaling), YaRN (freq_scale 1/4,
// n_ctx_orig_yarn 32) and a freq_base override. Two guards keep it from passing
// vacuously: the fused path must have served every decode step
// (llama_n_fused_decode), and the graph's own K rows under each non-native
// configuration must differ from the native ones.

#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    // repro knobs for the fused-vs-graph divergence at non-production weight scales (see the INF-64 note);
    // unset = the fixture's N(0, 1e-2) for every tensor
    static const char * e_nm = getenv("QFR_NORM_MEAN");
    static const char * e_ns = getenv("QFR_NORM_STD");
    static const char * e_ws = getenv("QFR_W_STD");
    const bool is_norm = strstr(tensor->name, "norm") != nullptr;
    const float mean = is_norm && e_nm ? (float) atof(e_nm) : 0.0f;
    const float sd   = is_norm ? (e_ns ? (float) atof(e_ns) : (e_ws ? (float) atof(e_ws) : 1.0e-2f))
                               : (e_ws ? (float) atof(e_ws) : 1.0e-2f);
    std::normal_distribution<float> dis(mean, sd);

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

// the qwen4exp subset of test-llama-archs.cpp's fixture (same hparams), plus the
// production rope base (1e7) so that the native configuration is meaningful
static gguf_context_ptr get_gguf_ctx_qwen4exp() {
    const llm_arch arch = LLM_ARCH_QWEN4EXP;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx   = 128;
    const uint32_t n_vocab = 128;
    const uint32_t n_embd  = 256;
    const uint32_t n_head  = 2;
    const uint32_t n_ff    = 384;
    const uint32_t n_layer = 2;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,   uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head);
    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,  8.0f);
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,       1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,   1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,   1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,     uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,    uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,  n_ctx/8);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,   uint32_t(8));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,   std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE,            1.0e7f);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,           "no_vocab");
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
    ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP, uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_COUNT,              uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,         uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,       uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,        uint32_t(2)); // sigmoid
    ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,        1.0f);
    ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,         uint32_t(1));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           uint32_t(2));
    return ret;
}

struct rope_cfg {
    const char * name;
    llama_rope_scaling_type type;
    float freq_base;   // 0 = from model
    float freq_scale;  // 0 = from model
    uint32_t yarn_orig_ctx;
};


static std::vector<float> decode_one(llama_context * lctx, llama_token tok, llama_pos pos, uint32_t n_vocab) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, {0}, true);
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("decode failed");
    }
    const float * l = llama_get_logits_ith(lctx, 0);
    std::vector<float> ret(l, l + n_vocab);
    llama_batch_free(batch);
    return ret;
}

static void prefill(llama_context * lctx, const std::vector<llama_token> & toks) {
    llama_batch batch = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        common_batch_add(batch, toks[i], (llama_pos) i, {0}, i + 1 == toks.size());
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("prefill failed");
    }
    llama_batch_free(batch);
}

// the cache rows [row0, row1) of every layer the cache holds, as f32
static std::vector<float> cache_rows(const llama_kv_cache * kv, uint32_t n_layer, int64_t row0, int64_t row1) {
    std::vector<float> ret;
    for (uint32_t il = 0; il < n_layer; il++) {
        if (!kv->has_layer((int32_t) il)) {
            continue;
        }
        const ggml_tensor * t = kv->get_k_layer_raw((int32_t) il);
        const size_t rs = ggml_row_size(t->type, t->ne[0]);
        GGML_ASSERT(t->nb[1] == rs && t->ne[1] >= row1);
        std::vector<uint8_t> raw(rs * (row1 - row0));
        ggml_backend_tensor_get(t, raw.data(), rs * row0, raw.size());
        std::vector<float> f(t->ne[0] * (row1 - row0));
        ggml_get_type_traits(t->type)->to_float(raw.data(), f.data(), (int64_t) f.size());
        ret.insert(ret.end(), f.begin(), f.end());
    }
    return ret;
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & x, double * max_abs = nullptr) {
    GGML_ASSERT(ref.size() == x.size() && !ref.empty());
    double num = 0.0, den = 0.0, ma = 0.0;
    for (size_t j = 0; j < ref.size(); j++) {
        const double d = (double) ref[j] - (double) x[j];
        num += d * d;
        den += (double) ref[j] * ref[j];
        ma = std::max(ma, std::fabs(d));
    }
    if (max_abs) {
        *max_abs = ma;
    }
    return num / den;
}

struct cfg_result {
    int64_t n_fused = 0;
    double logits_nmse = 0.0;
    double k_nmse = 0.0;
    double idx_nmse = 0.0;
    std::vector<float> ref_k; // the graph's K rows of the decoded tokens
};

static cfg_result run_cfg(llama_model * model, const rope_cfg & rc, const std::vector<llama_token> & toks, size_t n_prompt) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 128;
    cp.n_ubatch = 64;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.rope_scaling_type = rc.type;
    cp.rope_freq_base = rc.freq_base;
    cp.rope_freq_scale = rc.freq_scale;
    cp.yarn_orig_ctx = rc.yarn_orig_ctx;

    llama_context_ptr ref(llama_init_from_model(model, cp));
    llama_context_ptr fus(llama_init_from_model(model, cp));
    if (!ref || !fus) {
        throw std::runtime_error("context init failed");
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_layer = llama_model_n_layer(model);
    const std::vector<llama_token> prompt(toks.begin(), toks.begin() + n_prompt);

    setenv("GGML_FUSED_DECODE_OFF", "1", 1);
    prefill(ref.get(), prompt);
    prefill(fus.get(), prompt);

    cfg_result r;
    const int64_t n_fused0 = llama_n_fused_decode(fus.get());
    for (size_t i = n_prompt; i < toks.size(); i++) {
        setenv("GGML_FUSED_DECODE_OFF", "1", 1);
        const std::vector<float> a = decode_one(ref.get(), toks[i], (llama_pos) i, n_vocab);
        unsetenv("GGML_FUSED_DECODE_OFF");
        const std::vector<float> b = decode_one(fus.get(), toks[i], (llama_pos) i, n_vocab);
        r.logits_nmse = std::max(r.logits_nmse, nmse(a, b));
    }
    setenv("GGML_FUSED_DECODE_OFF", "1", 1);
    r.n_fused = llama_n_fused_decode(fus.get()) - n_fused0;
    if (llama_n_fused_decode(ref.get()) != 0) {
        throw std::runtime_error("the reference context ran the fused path");
    }

    // one sequence, cells allocated in order: the decoded tokens sit in rows [n_prompt, n_tok)
    auto * mref = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ref.get()));
    auto * mfus = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(fus.get()));
    if (!mref || !mfus || !mref->get_mem_idx() || !mfus->get_mem_idx()) {
        throw std::runtime_error("expected the hybrid memory with an indexer cache");
    }
    const int64_t r0 = (int64_t) n_prompt, r1 = (int64_t) toks.size();
    r.ref_k = cache_rows(mref->get_mem_attn(), n_layer, r0, r1);
    r.k_nmse = nmse(r.ref_k, cache_rows(mfus->get_mem_attn(), n_layer, r0, r1));
    r.idx_nmse = nmse(cache_rows(mref->get_mem_idx(), n_layer, r0, r1),
                      cache_rows(mfus->get_mem_idx(), n_layer, r0, r1));
    return r;
}

int main(int argc, char ** argv) {
    common_init();
    size_t seed = 1234;
    if (argc > 1) {
        seed = std::stoull(argv[1]);
    }
    // the fused path transcribes the graph's ops with the same kernels, but its
    // gemv reduction order differs, so it is close, not bit-identical; the K and
    // indexer rows are stored as f16, so their tolerance is f16 rounding
    const double tol_logits = 1e-6;
    const double tol_cache  = 1e-5;

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    gguf_context_ptr gctx = get_gguf_ctx_qwen4exp();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[2] = { cpu, nullptr };
    mp.devices = devs;
    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gctx.get(), set_tensor_data, &tmp, mp));
    if (!model) {
        fprintf(stderr, "failed to create the model\n");
        return 1;
    }

    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 127);
    std::vector<llama_token> toks(96);
    for (auto & t : toks) {
        t = dis(gen);
    }
    const size_t n_prompt = 40;

    const rope_cfg cfgs[] = {
        { "native",         LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED, 0.0f,   0.0f,  0  },
        { "yarn-f4-orig32", LLAMA_ROPE_SCALING_TYPE_YARN,        0.0f,   0.25f, 32 },
        { "freq-base-1e4",  LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED, 1.0e4f, 0.0f,  0  },
    };

    bool ok = true;
    std::vector<float> ref_k_native;
    for (const rope_cfg & rc : cfgs) {
        cfg_result r;
        try {
            r = run_cfg(model.get(), rc, toks, n_prompt);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: %s\n", rc.name, e.what());
            ok = false;
            continue;
        }
        double sens = -1.0;
        if (ref_k_native.empty()) {
            ref_k_native = r.ref_k;
        } else {
            sens = nmse(ref_k_native, r.ref_k);
        }
        const int64_t n_expected = (int64_t) (toks.size() - n_prompt);
        const bool pass = r.n_fused == n_expected
            && r.k_nmse <= tol_cache && r.idx_nmse <= tol_cache && r.logits_nmse <= tol_logits
            && (sens < 0.0 || sens > 1e-2);
        printf("%-15s fused=%" PRId64 "/%" PRId64 " K_nmse=%.2e idx_nmse=%.2e logits_nmse=%.2e graphK_vs_native=%.2e  %s\n",
               rc.name, r.n_fused, n_expected, r.k_nmse, r.idx_nmse, r.logits_nmse, sens, pass ? "OK" : "FAIL");
        ok = ok && pass;
    }
    return ok ? 0 : 1;
}
