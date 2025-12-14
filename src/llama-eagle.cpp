#include "llama-eagle.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>

// EAGLE GGUF keys
#define EAGLE_KEY_EMBEDDING_LENGTH        "eagle.embedding_length"
#define EAGLE_KEY_FEED_FORWARD_LENGTH     "eagle.feed_forward_length"
#define EAGLE_KEY_HEAD_COUNT              "eagle.attention.head_count"
#define EAGLE_KEY_HEAD_COUNT_KV           "eagle.attention.head_count_kv"
#define EAGLE_KEY_BLOCK_COUNT             "eagle.block_count"
#define EAGLE_KEY_VOCAB_SIZE              "eagle.vocab_size"
#define EAGLE_KEY_CONTEXT_LENGTH          "eagle.context_length"
#define EAGLE_KEY_RMS_NORM_EPS            "eagle.attention.layer_norm_rms_epsilon"
#define EAGLE_KEY_ROPE_FREQ_BASE          "eagle.rope.freq_base"

// EAGLE tensor names
#define EAGLE_TENSOR_EMBED_TOKENS         "eagle.embed_tokens.weight"
#define EAGLE_TENSOR_FC_WEIGHT            "eagle.fc.weight"
#define EAGLE_TENSOR_FC_BIAS              "eagle.fc.bias"
#define EAGLE_TENSOR_ATTN_NORM            "eagle.blk.0.attn_norm.weight"
#define EAGLE_TENSOR_ATTN_Q               "eagle.blk.0.attn_q.weight"
#define EAGLE_TENSOR_ATTN_K               "eagle.blk.0.attn_k.weight"
#define EAGLE_TENSOR_ATTN_V               "eagle.blk.0.attn_v.weight"
#define EAGLE_TENSOR_ATTN_OUTPUT          "eagle.blk.0.attn_output.weight"
#define EAGLE_TENSOR_FFN_NORM             "eagle.blk.0.ffn_norm.weight"
#define EAGLE_TENSOR_FFN_GATE             "eagle.blk.0.ffn_gate.weight"
#define EAGLE_TENSOR_FFN_UP               "eagle.blk.0.ffn_up.weight"
#define EAGLE_TENSOR_FFN_DOWN             "eagle.blk.0.ffn_down.weight"

llama_eagle_head::~llama_eagle_head() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

// Helper to get uint32 from GGUF
static uint32_t get_u32(const gguf_context * gguf_ctx, const char * key, uint32_t default_val = 0) {
    int idx = gguf_find_key(gguf_ctx, key);
    if (idx < 0) {
        return default_val;
    }
    return gguf_get_val_u32(gguf_ctx, idx);
}

// Helper to get float from GGUF
static float get_f32(const gguf_context * gguf_ctx, const char * key, float default_val = 0.0f) {
    int idx = gguf_find_key(gguf_ctx, key);
    if (idx < 0) {
        return default_val;
    }
    return gguf_get_val_f32(gguf_ctx, idx);
}

struct llama_eagle_head * llama_eagle_load(
        const char * path_gguf,
        const struct llama_model * model_base) {

    LLAMA_LOG_INFO("%s: loading EAGLE head from '%s'\n", __func__, path_gguf);

    // Open GGUF file
    ggml_context * ctx_meta = nullptr;
    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };

    gguf_context * gguf_ctx = gguf_init_from_file(path_gguf, params);
    if (!gguf_ctx) {
        LLAMA_LOG_ERROR("%s: failed to load GGUF file '%s'\n", __func__, path_gguf);
        return nullptr;
    }

    // Check architecture
    int arch_idx = gguf_find_key(gguf_ctx, "general.architecture");
    if (arch_idx >= 0) {
        const char * arch = gguf_get_val_str(gguf_ctx, arch_idx);
        if (strcmp(arch, "eagle") != 0) {
            LLAMA_LOG_ERROR("%s: expected 'eagle' architecture, got '%s'\n", __func__, arch);
            gguf_free(gguf_ctx);
            ggml_free(ctx_meta);
            return nullptr;
        }
    }

    // Create EAGLE head
    auto * eagle = new llama_eagle_head();

    // Load hyperparameters
    eagle->n_embd       = get_u32(gguf_ctx, EAGLE_KEY_EMBEDDING_LENGTH, 4096);
    eagle->n_ff         = get_u32(gguf_ctx, EAGLE_KEY_FEED_FORWARD_LENGTH, 11008);
    eagle->n_head       = get_u32(gguf_ctx, EAGLE_KEY_HEAD_COUNT, 32);
    eagle->n_head_kv    = get_u32(gguf_ctx, EAGLE_KEY_HEAD_COUNT_KV, eagle->n_head);
    eagle->n_layer      = get_u32(gguf_ctx, EAGLE_KEY_BLOCK_COUNT, 1);
    eagle->n_vocab      = get_u32(gguf_ctx, EAGLE_KEY_VOCAB_SIZE, 32000);
    eagle->n_ctx        = get_u32(gguf_ctx, EAGLE_KEY_CONTEXT_LENGTH, 4096);
    eagle->rms_norm_eps = get_f32(gguf_ctx, EAGLE_KEY_RMS_NORM_EPS, 1e-6f);
    eagle->rope_freq_base = get_f32(gguf_ctx, EAGLE_KEY_ROPE_FREQ_BASE, 10000.0f);

    LLAMA_LOG_INFO("%s: EAGLE head params:\n", __func__);
    LLAMA_LOG_INFO("%s:   n_embd       = %u\n", __func__, eagle->n_embd);
    LLAMA_LOG_INFO("%s:   n_ff         = %u\n", __func__, eagle->n_ff);
    LLAMA_LOG_INFO("%s:   n_head       = %u\n", __func__, eagle->n_head);
    LLAMA_LOG_INFO("%s:   n_head_kv    = %u\n", __func__, eagle->n_head_kv);
    LLAMA_LOG_INFO("%s:   n_layer      = %u\n", __func__, eagle->n_layer);
    LLAMA_LOG_INFO("%s:   n_vocab      = %u\n", __func__, eagle->n_vocab);

    // Verify compatibility with base model
    if (model_base) {
        const auto & hparams = model_base->hparams;
        if (eagle->n_embd != hparams.n_embd) {
            LLAMA_LOG_ERROR("%s: EAGLE n_embd (%u) != base model n_embd (%u)\n",
                    __func__, eagle->n_embd, hparams.n_embd);
            delete eagle;
            gguf_free(gguf_ctx);
            ggml_free(ctx_meta);
            return nullptr;
        }
        // Note: vocab size might differ slightly, but should be close
        // We'll use base model's vocab for tokenization
    }

    // Calculate memory needed for tensors
    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    size_t total_size = 0;
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * meta = ggml_get_tensor(ctx_meta, name);
        if (meta) {
            total_size += ggml_nbytes(meta);
        }
    }

    LLAMA_LOG_INFO("%s: EAGLE tensors: %d, total size: %.2f MB\n",
            __func__, n_tensors, total_size / (1024.0 * 1024.0));

    // Create context for EAGLE tensors
    struct ggml_init_params ctx_params = {
        /*.mem_size   = */ ggml_tensor_overhead() * (n_tensors + 1),
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    eagle->ctx = ggml_init(ctx_params);
    if (!eagle->ctx) {
        LLAMA_LOG_ERROR("%s: failed to create ggml context\n", __func__);
        delete eagle;
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return nullptr;
    }

    // Create tensors
    auto create_tensor = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * meta = ggml_get_tensor(ctx_meta, name);
        if (!meta) {
            LLAMA_LOG_WARN("%s: tensor '%s' not found\n", __func__, name);
            return nullptr;
        }
        ggml_tensor * tensor = ggml_dup_tensor(eagle->ctx, meta);
        ggml_set_name(tensor, name);
        return tensor;
    };

    // Create all EAGLE tensors
    eagle->tok_embd      = create_tensor(EAGLE_TENSOR_EMBED_TOKENS);
    eagle->fc_weight     = create_tensor(EAGLE_TENSOR_FC_WEIGHT);
    eagle->fc_bias       = create_tensor(EAGLE_TENSOR_FC_BIAS);
    eagle->layer.attn_norm = create_tensor(EAGLE_TENSOR_ATTN_NORM);
    eagle->layer.wq      = create_tensor(EAGLE_TENSOR_ATTN_Q);
    eagle->layer.wk      = create_tensor(EAGLE_TENSOR_ATTN_K);
    eagle->layer.wv      = create_tensor(EAGLE_TENSOR_ATTN_V);
    eagle->layer.wo      = create_tensor(EAGLE_TENSOR_ATTN_OUTPUT);
    eagle->layer.ffn_norm = create_tensor(EAGLE_TENSOR_FFN_NORM);
    eagle->layer.ffn_gate = create_tensor(EAGLE_TENSOR_FFN_GATE);
    eagle->layer.ffn_up  = create_tensor(EAGLE_TENSOR_FFN_UP);
    eagle->layer.ffn_down = create_tensor(EAGLE_TENSOR_FFN_DOWN);

    // Check required tensors
    if (!eagle->fc_weight || !eagle->layer.wq || !eagle->layer.wk ||
        !eagle->layer.wv || !eagle->layer.wo || !eagle->layer.ffn_gate ||
        !eagle->layer.ffn_up || !eagle->layer.ffn_down) {
        LLAMA_LOG_ERROR("%s: missing required EAGLE tensors\n", __func__);
        delete eagle;
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return nullptr;
    }

    // Allocate buffer for tensors (CPU backend)
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        LLAMA_LOG_ERROR("%s: failed to create CPU backend\n", __func__);
        delete eagle;
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return nullptr;
    }

    eagle->buffer = ggml_backend_alloc_ctx_tensors(eagle->ctx, backend);
    if (!eagle->buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate tensor buffer\n", __func__);
        ggml_backend_free(backend);
        delete eagle;
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return nullptr;
    }

    // Load tensor data from file
    FILE * fp = fopen(path_gguf, "rb");
    if (!fp) {
        LLAMA_LOG_ERROR("%s: failed to open file for reading\n", __func__);
        ggml_backend_free(backend);
        delete eagle;
        gguf_free(gguf_ctx);
        ggml_free(ctx_meta);
        return nullptr;
    }

    // Get data offset in GGUF file
    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Load each tensor
    auto load_tensor = [&](ggml_tensor * tensor) -> bool {
        if (!tensor) return true; // Skip optional tensors

        int tensor_idx = gguf_find_tensor(gguf_ctx, ggml_get_name(tensor));
        if (tensor_idx < 0) {
            LLAMA_LOG_WARN("%s: tensor '%s' not found in GGUF\n", __func__, ggml_get_name(tensor));
            return true; // Not a fatal error for optional tensors
        }

        size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx, tensor_idx);
        size_t file_offset = data_offset + tensor_offset;

        fseek(fp, file_offset, SEEK_SET);

        size_t nbytes = ggml_nbytes(tensor);
        std::vector<uint8_t> buf(nbytes);
        size_t nread = fread(buf.data(), 1, nbytes, fp);
        if (nread != nbytes) {
            LLAMA_LOG_ERROR("%s: failed to read tensor '%s' (read %zu of %zu bytes)\n",
                    __func__, ggml_get_name(tensor), nread, nbytes);
            return false;
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
        return true;
    };

    bool success = true;
    success = success && load_tensor(eagle->tok_embd);
    success = success && load_tensor(eagle->fc_weight);
    success = success && load_tensor(eagle->fc_bias);
    success = success && load_tensor(eagle->layer.attn_norm);
    success = success && load_tensor(eagle->layer.wq);
    success = success && load_tensor(eagle->layer.wk);
    success = success && load_tensor(eagle->layer.wv);
    success = success && load_tensor(eagle->layer.wo);
    success = success && load_tensor(eagle->layer.ffn_norm);
    success = success && load_tensor(eagle->layer.ffn_gate);
    success = success && load_tensor(eagle->layer.ffn_up);
    success = success && load_tensor(eagle->layer.ffn_down);

    fclose(fp);
    ggml_backend_free(backend);
    gguf_free(gguf_ctx);
    ggml_free(ctx_meta);

    if (!success) {
        LLAMA_LOG_ERROR("%s: failed to load EAGLE tensors\n", __func__);
        delete eagle;
        return nullptr;
    }

    eagle->loaded = true;
    LLAMA_LOG_INFO("%s: EAGLE head loaded successfully\n", __func__);

    return eagle;
}

void llama_eagle_free(struct llama_eagle_head * eagle) {
    delete eagle;
}

bool llama_eagle_is_compatible(
        const struct llama_eagle_head * eagle,
        const struct llama_model * model) {
    if (!eagle || !model) {
        return false;
    }

    const auto & hparams = model->hparams;

    // Check embedding size match
    if (eagle->n_embd != hparams.n_embd) {
        LLAMA_LOG_WARN("%s: n_embd mismatch: EAGLE=%u, model=%u\n",
                __func__, eagle->n_embd, hparams.n_embd);
        return false;
    }

    // Vocab size should be close (within tolerance)
    const uint32_t model_vocab = model->vocab.n_tokens();
    int vocab_diff = (int)eagle->n_vocab - (int)model_vocab;
    if (abs(vocab_diff) > 128) {
        LLAMA_LOG_WARN("%s: vocab size mismatch: EAGLE=%u, model=%u\n",
                __func__, eagle->n_vocab, model_vocab);
        return false;
    }

    return true;
}

int32_t llama_eagle_generate_draft(
        struct llama_context * ctx,
        struct llama_eagle_head * eagle,
        const float * hidden_states,
        llama_token last_token,
        int32_t n_draft,
        llama_token * draft_tokens) {
    // TODO: Implement EAGLE draft generation in Phase 3
    // For now, return 0 (no draft tokens)
    (void)ctx;
    (void)eagle;
    (void)hidden_states;
    (void)last_token;
    (void)n_draft;
    (void)draft_tokens;

    LLAMA_LOG_WARN("%s: EAGLE draft generation not yet implemented\n", __func__);
    return 0;
}
