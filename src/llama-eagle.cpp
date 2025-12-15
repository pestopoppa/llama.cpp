#include "llama-eagle.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include "gguf.h"

#include <cmath>
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

// Helper: Build EAGLE forward pass graph for a single token
// Returns logits, hidden state, and intermediate tensors for debugging
struct eagle_output {
    ggml_tensor * logits;       // [n_vocab]
    ggml_tensor * hidden_out;   // [n_embd] - for next iteration
    ggml_tensor * fc_out;       // [n_embd] - FC layer output (debug)
    ggml_tensor * attn_out;     // [n_embd] - attention output (debug)
    ggml_tensor * ffn_norm_out; // [n_embd] - FFN norm output (debug)
};

static eagle_output eagle_build_graph(
        ggml_context * ctx0,
        const llama_eagle_head * eagle,
        const llama_model * model,
        ggml_tensor * hidden_state,  // [n_embd]
        ggml_tensor * token_embd,    // [n_embd] - embedded token
        llama_pos pos) {

    const uint32_t n_embd = eagle->n_embd;
    const uint32_t n_head = eagle->n_head;
    const uint32_t n_head_kv = eagle->n_head_kv;
    const uint32_t n_embd_head = n_embd / n_head;
    const uint32_t n_ff = eagle->n_ff;

    // Step 1: Concatenate token_embd and hidden_state → [2*n_embd]
    // EAGLE paper: concat(e_t, h_t) where e_t = token embedding, h_t = hidden state
    ggml_tensor * concat = ggml_concat(ctx0, token_embd, hidden_state, 0);

    // Step 2: Fusion layer (fc): [2*n_embd] → [n_embd]
    ggml_tensor * cur = ggml_mul_mat(ctx0, eagle->fc_weight, concat);
    ggml_tensor * fc_output = cur;  // Save for debug output
    ggml_set_name(fc_output, "fc_out");
    if (eagle->fc_bias) {
        // Cast bias to f32 if needed (EAGLE weights are F16)
        ggml_tensor * bias = eagle->fc_bias;
        if (bias->type != GGML_TYPE_F32) {
            bias = ggml_cast(ctx0, bias, GGML_TYPE_F32);
        }
        cur = ggml_add(ctx0, cur, bias);
    }

    // Step 3: Single decoder layer
    ggml_tensor * inpL = cur;

    // 3a. Attention norm (if present - EAGLE-1 may skip this)
    if (eagle->layer.attn_norm) {
        cur = ggml_rms_norm(ctx0, inpL, eagle->rms_norm_eps);
        ggml_tensor * norm = eagle->layer.attn_norm;
        if (norm->type != GGML_TYPE_F32) {
            norm = ggml_cast(ctx0, norm, GGML_TYPE_F32);
        }
        cur = ggml_mul(ctx0, cur, norm);
    }

    // 3b. Self-attention with Q/K/V projections
    // For single token without KV cache, attention weights are trivially 1.0
    // but we still need proper Q/K projections as the trained weights encode information
    {
        // Q/K/V projections
        ggml_tensor * Qcur = ggml_mul_mat(ctx0, eagle->layer.wq, cur);
        ggml_tensor * Kcur = ggml_mul_mat(ctx0, eagle->layer.wk, cur);
        ggml_tensor * Vcur = ggml_mul_mat(ctx0, eagle->layer.wv, cur);

        // Qcur shape: [n_head * head_dim] = [n_embd]
        // Kcur shape: [n_head_kv * head_dim]
        // Vcur shape: [n_head_kv * head_dim]

        // NOTE: RoPE is skipped for MVP single-token inference
        // For single token at relative position 0, RoPE rotation has no effect
        // on the attention output (Q @ K^T would just be rotated the same amount)
        // TODO: Add RoPE when implementing KV cache

        // For single token, attention score is Q @ K^T / sqrt(d)
        // With one token, this is a scalar per head, and softmax([x]) = 1.0
        // So attention output = 1.0 * V = V
        // But with GQA, we need to properly expand V to match the output dimension

        // TEMPORARY: RESTORE ORIGINAL BROKEN GQA TILING FOR SANITY CHECK
        // This uses ggml_repeat which TILES [A,B,A,B...] instead of INTERLEAVING [A,A,A,A,B,B,B,B...]
        // This is WRONG for GQA, but we're testing if the EAGLE checkpoint was trained with this bug
        if (n_head != n_head_kv) {
            // Reshape [n_head_kv * head_dim] → [head_dim, n_head_kv]
            Vcur = ggml_reshape_2d(ctx0, Vcur, n_embd_head, n_head_kv);

            // Create target [head_dim, n_head] and use ggml_repeat
            // This will TILE the KV heads: [H0,H1,H2,H3,H4,H5,H6,H7, H0,H1,H2,H3,H4,H5,H6,H7, ...]
            // Which is WRONG for GQA but matches potential training bug
            ggml_tensor * Vexp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_head, n_head);
            Vcur = ggml_repeat(ctx0, Vcur, Vexp);

            // Flatten to [n_embd]
            Vcur = ggml_reshape_1d(ctx0, Vcur, n_embd);
        }

        // Output projection: wo @ V
        // wo shape: [n_embd, n_embd], V shape: [n_embd]
        cur = ggml_mul_mat(ctx0, eagle->layer.wo, Vcur);
    }
    ggml_tensor * attn_output = cur;  // Save for debug
    ggml_set_name(attn_output, "attn_out");

    // Residual connection
    cur = ggml_add(ctx0, cur, inpL);
    ggml_tensor * ffn_inp = cur;

    // 3c. FFN norm
    cur = ggml_rms_norm(ctx0, ffn_inp, eagle->rms_norm_eps);
    if (eagle->layer.ffn_norm) {
        ggml_tensor * norm = eagle->layer.ffn_norm;
        if (norm->type != GGML_TYPE_F32) {
            norm = ggml_cast(ctx0, norm, GGML_TYPE_F32);
        }
        cur = ggml_mul(ctx0, cur, norm);
    }
    ggml_tensor * ffn_norm_output = cur;  // Save for debug
    ggml_set_name(ffn_norm_output, "ffn_norm_out");

    // 3d. FFN with SwiGLU
    {
        ggml_tensor * gate = ggml_mul_mat(ctx0, eagle->layer.ffn_gate, cur);
        ggml_tensor * up = ggml_mul_mat(ctx0, eagle->layer.ffn_up, cur);

        // SiLU activation on gate
        gate = ggml_silu(ctx0, gate);

        // Element-wise multiply
        ggml_tensor * ffn_out = ggml_mul(ctx0, gate, up);

        // Down projection
        cur = ggml_mul_mat(ctx0, eagle->layer.ffn_down, ffn_out);
    }

    // Residual connection
    cur = ggml_add(ctx0, cur, ffn_inp);

    // Save the hidden state output (before LM head) for next iteration
    ggml_tensor * hidden_out = cur;

    // Step 4: Project through LM head
    // IMPORTANT: Use EAGLE's embed_tokens as tied lm_head (matches training)
    // The target model's lm_head may be quantized differently than during EAGLE training
    ggml_tensor * logits = cur;
    if (eagle->tok_embd) {
        // Use EAGLE's embed_tokens (transposed) as lm_head - this matches training
        // tok_embd: [n_embd, n_vocab], cur: [n_embd]
        // logits = tok_embd^T @ cur = [n_vocab]
        logits = ggml_mul_mat(ctx0, eagle->tok_embd, cur);
    } else if (model->output) {
        // Fallback to target model's output (may be quantized)
        logits = ggml_mul_mat(ctx0, model->output, cur);
        if (model->output_b) {
            logits = ggml_add(ctx0, logits, model->output_b);
        }
    } else if (model->tok_embd) {
        // Tied embeddings from target model
        logits = ggml_mul_mat(ctx0, model->tok_embd, cur);
    }

    return { logits, hidden_out, fc_output, attn_output, ffn_norm_output };
}

int32_t llama_eagle_generate_draft(
        struct llama_context * ctx,
        struct llama_eagle_head * eagle,
        const float * hidden_states,
        llama_token last_token,
        int32_t n_draft,
        llama_token * draft_tokens) {

    if (!ctx || !eagle || !hidden_states || !draft_tokens || n_draft <= 0) {
        return 0;
    }

    if (!eagle->loaded) {
        LLAMA_LOG_ERROR("%s: EAGLE head not loaded\n", __func__);
        return 0;
    }

    const llama_model & model_ref = ctx->get_model();
    const llama_model * model = &model_ref;

    // Check compatibility
    if (!llama_eagle_is_compatible(eagle, model)) {
        LLAMA_LOG_ERROR("%s: EAGLE head incompatible with model\n", __func__);
        return 0;
    }

    const uint32_t n_embd = eagle->n_embd;
    const uint32_t n_vocab = model->vocab.n_tokens();

    // Debug: Print FC weight shape and first values
    static bool printed_shapes = false;
    if (!printed_shapes) {
        LLAMA_LOG_INFO("%s: EAGLE shapes debug:\n", __func__);
        LLAMA_LOG_INFO("%s:   fc_weight: [%lld, %lld]\n", __func__,
                       (long long)eagle->fc_weight->ne[0], (long long)eagle->fc_weight->ne[1]);

        // Print first few FC weight values
        std::vector<ggml_fp16_t> fc_vals(10);
        ggml_backend_tensor_get(eagle->fc_weight, fc_vals.data(), 0, 10 * sizeof(ggml_fp16_t));
        LLAMA_LOG_INFO("%s:   fc_weight first 5: %.4f %.4f %.4f %.4f %.4f\n", __func__,
                       ggml_fp16_to_fp32(fc_vals[0]), ggml_fp16_to_fp32(fc_vals[1]),
                       ggml_fp16_to_fp32(fc_vals[2]), ggml_fp16_to_fp32(fc_vals[3]),
                       ggml_fp16_to_fp32(fc_vals[4]));
        LLAMA_LOG_INFO("%s:   wq: [%lld, %lld]\n", __func__,
                       (long long)eagle->layer.wq->ne[0], (long long)eagle->layer.wq->ne[1]);
        LLAMA_LOG_INFO("%s:   wk: [%lld, %lld]\n", __func__,
                       (long long)eagle->layer.wk->ne[0], (long long)eagle->layer.wk->ne[1]);
        LLAMA_LOG_INFO("%s:   wv: [%lld, %lld]\n", __func__,
                       (long long)eagle->layer.wv->ne[0], (long long)eagle->layer.wv->ne[1]);
        LLAMA_LOG_INFO("%s:   wo: [%lld, %lld]\n", __func__,
                       (long long)eagle->layer.wo->ne[0], (long long)eagle->layer.wo->ne[1]);
        if (model->output) {
            LLAMA_LOG_INFO("%s:   lm_head: [%lld, %lld]\n", __func__,
                           (long long)model->output->ne[0], (long long)model->output->ne[1]);
        }
        printed_shapes = true;
    }

    // Create backend for computation
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        LLAMA_LOG_ERROR("%s: failed to create backend\n", __func__);
        return 0;
    }

    int32_t n_generated = 0;
    llama_token cur_token = last_token;
    std::vector<float> cur_hidden(n_embd);
    std::memcpy(cur_hidden.data(), hidden_states, n_embd * sizeof(float));

    // Buffer for logits
    std::vector<float> logits(n_vocab);

    for (int32_t i = 0; i < n_draft; i++) {
        // Create fresh compute context for each draft token
        // (Less efficient but simpler - can optimize later)
        const size_t ctx_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead();
        struct ggml_init_params ctx_params = {
            /*.mem_size   = */ ctx_size,
            /*.mem_buffer = */ nullptr,
            /*.no_alloc   = */ true,
        };

        ggml_context * ctx0 = ggml_init(ctx_params);
        if (!ctx0) {
            LLAMA_LOG_ERROR("%s: failed to create compute context\n", __func__);
            break;
        }

        // Create input tensors
        ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_embd);
        ggml_tensor * inp_embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_embd);

        // Build graph
        eagle_output out = eagle_build_graph(ctx0, eagle, model, inp_hidden, inp_embd, i);
        ggml_tensor * out_logits = out.logits;
        ggml_tensor * out_hidden = out.hidden_out;
        ggml_tensor * out_fc = out.fc_out;
        ggml_tensor * out_attn = out.attn_out;
        ggml_tensor * out_ffn_norm = out.ffn_norm_out;

        // Create compute graph - need all outputs for debug
        ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, out_logits);
        ggml_build_forward_expand(gf, out_hidden);
        ggml_build_forward_expand(gf, out_fc);
        ggml_build_forward_expand(gf, out_attn);
        ggml_build_forward_expand(gf, out_ffn_norm);

        // Allocate buffers
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx0, backend);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute buffer\n", __func__);
            ggml_free(ctx0);
            break;
        }

        // Set input data
        ggml_backend_tensor_set(inp_hidden, cur_hidden.data(), 0, n_embd * sizeof(float));

        // Get token embedding
        if (eagle->tok_embd) {
            // Extract embedding for cur_token from eagle's embeddings
            // tok_embd shape is [n_embd, n_vocab] in ggml
            // For token T, we want row T which is at offset T * n_embd
            const size_t embd_offset = cur_token * n_embd * sizeof(ggml_fp16_t);
            std::vector<ggml_fp16_t> embd_f16(n_embd);
            ggml_backend_tensor_get(eagle->tok_embd, embd_f16.data(), embd_offset, n_embd * sizeof(ggml_fp16_t));

            // Convert to f32
            std::vector<float> embd_f32(n_embd);
            for (uint32_t j = 0; j < n_embd; j++) {
                embd_f32[j] = ggml_fp16_to_fp32(embd_f16[j]);
            }

            // Debug: Print first embedding stats and first values
            static bool printed_embd = false;
            if (!printed_embd && i == 0) {
                float e_min = embd_f32[0], e_max = embd_f32[0], e_sum = 0;
                for (uint32_t j = 0; j < n_embd; j++) {
                    e_sum += embd_f32[j];
                    if (embd_f32[j] < e_min) e_min = embd_f32[j];
                    if (embd_f32[j] > e_max) e_max = embd_f32[j];
                }
                LLAMA_LOG_INFO("%s: embd stats: min=%.4f max=%.4f mean=%.4f\n",
                               __func__, e_min, e_max, e_sum/n_embd);
                LLAMA_LOG_INFO("%s: embd first 5: %.6f %.6f %.6f %.6f %.6f\n",
                               __func__, embd_f32[0], embd_f32[1], embd_f32[2], embd_f32[3], embd_f32[4]);
                LLAMA_LOG_INFO("%s: hidden first 5: %.6f %.6f %.6f %.6f %.6f\n",
                               __func__, cur_hidden[0], cur_hidden[1], cur_hidden[2], cur_hidden[3], cur_hidden[4]);

                // Save hidden state to file for Python comparison
                FILE * f_hidden = fopen("/tmp/eagle_hidden.bin", "wb");
                if (f_hidden) {
                    fwrite(cur_hidden.data(), sizeof(float), n_embd, f_hidden);
                    fclose(f_hidden);
                    LLAMA_LOG_INFO("%s: saved hidden state to /tmp/eagle_hidden.bin\n", __func__);
                }
                printed_embd = true;
            }

            ggml_backend_tensor_set(inp_embd, embd_f32.data(), 0, n_embd * sizeof(float));
        } else {
            // Use base model's embeddings
            // This is more complex - skip for MVP
            LLAMA_LOG_WARN("%s: EAGLE without tok_embd not yet supported\n", __func__);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx0);
            break;
        }

        // Compute
        ggml_backend_graph_compute(backend, gf);

        // Debug: Print intermediate values
        static bool printed_fc = false;
        if (!printed_fc && i == 0) {
            auto print_stats = [n_embd](const char* name, ggml_tensor* t) {
                std::vector<float> data(n_embd);
                ggml_backend_tensor_get(t, data.data(), 0, n_embd * sizeof(float));
                float mn = data[0], mx = data[0];
                for (uint32_t j = 0; j < n_embd; j++) {
                    if (data[j] < mn) mn = data[j];
                    if (data[j] > mx) mx = data[j];
                }
                LLAMA_LOG_INFO("  %s: min=%.4f max=%.4f first5=[%.4f %.4f %.4f %.4f %.4f]\n",
                               name, mn, mx, data[0], data[1], data[2], data[3], data[4]);
            };

            LLAMA_LOG_INFO("%s: Intermediate values:\n", __func__);
            print_stats("FC output", out_fc);
            print_stats("Attn out", out_attn);
            print_stats("FFN norm", out_ffn_norm);
            print_stats("Hidden out", out_hidden);
            printed_fc = true;
        }

        // Debug: Print logits stats for first iteration
        static bool printed_logits = false;
        if (!printed_logits && i == 0) {
            std::vector<float> dbg_logits(n_vocab);
            ggml_backend_tensor_get(out_logits, dbg_logits.data(), 0, n_vocab * sizeof(float));
            float l_min = dbg_logits[0], l_max = dbg_logits[0], l_sum = 0;
            for (uint32_t j = 0; j < n_vocab; j++) {
                l_sum += dbg_logits[j];
                if (dbg_logits[j] < l_min) l_min = dbg_logits[j];
                if (dbg_logits[j] > l_max) l_max = dbg_logits[j];
            }
            // Find top 5 tokens
            std::vector<std::pair<float, int>> top5;
            for (uint32_t j = 0; j < n_vocab; j++) {
                if (top5.size() < 5 || dbg_logits[j] > top5.back().first) {
                    top5.push_back({dbg_logits[j], (int)j});
                    std::sort(top5.begin(), top5.end(), std::greater<>());
                    if (top5.size() > 5) top5.resize(5);
                }
            }
            LLAMA_LOG_INFO("%s: logits stats: min=%.4f max=%.4f mean=%.4f\n",
                           __func__, l_min, l_max, l_sum/n_vocab);
            LLAMA_LOG_INFO("%s: top 5 tokens: %d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f)\n",
                           __func__,
                           top5[0].second, top5[0].first,
                           top5[1].second, top5[1].first,
                           top5[2].second, top5[2].first,
                           top5[3].second, top5[3].first,
                           top5[4].second, top5[4].first);
            // Check specific tokens (common continuations for "Hello")
            LLAMA_LOG_INFO("%s: logit[14924]=%f (likely target), logit[323]=%f (likely top)\n",
                           __func__, dbg_logits[14924], dbg_logits[323]);
            printed_logits = true;
        }

        // Get logits
        ggml_backend_tensor_get(out_logits, logits.data(), 0, n_vocab * sizeof(float));

        // Get hidden state for next iteration (BEFORE buffer free!)
        ggml_backend_tensor_get(out_hidden, cur_hidden.data(), 0, n_embd * sizeof(float));

        // Cleanup this iteration's context
        ggml_backend_buffer_free(buf);
        ggml_free(ctx0);

        // Sample (greedy argmax for MVP)
        llama_token sampled = 0;
        float max_logit = logits[0];
        for (uint32_t j = 1; j < n_vocab; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                sampled = j;
            }
        }

        draft_tokens[n_generated++] = sampled;
        cur_token = sampled;

        // Early stop on EOS
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (sampled == llama_vocab_eos(vocab)) {
            break;
        }
    }

    ggml_backend_free(backend);

    return n_generated;
}
