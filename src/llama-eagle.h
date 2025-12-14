#pragma once

#include "llama.h"
#include "ggml.h"

#include <string>
#include <memory>

struct llama_model;
struct llama_context;

// EAGLE draft head - lightweight transformer layer for speculative decoding
// EAGLE-1 architecture:
//   - embed_tokens: token embeddings (same vocab as base model)
//   - fc: fusion layer (combines hidden state + embedded token)
//   - Single decoder layer (attention + FFN + norms)
//   - Uses base model's LM head for output
struct llama_eagle_head {
    // Hyperparameters
    uint32_t n_embd       = 0;  // embedding size (must match base model)
    uint32_t n_ff         = 0;  // feed-forward size
    uint32_t n_head       = 0;  // attention heads
    uint32_t n_head_kv    = 0;  // KV heads (for GQA)
    uint32_t n_layer      = 1;  // number of layers (usually 1 for EAGLE-1)
    uint32_t n_vocab      = 0;  // vocab size (must match base model)
    uint32_t n_ctx        = 0;  // context length
    float    rms_norm_eps = 1e-6f;
    float    rope_freq_base = 10000.0f;

    // Token embeddings - may be shared with base model
    ggml_tensor * tok_embd = nullptr;

    // Fusion layer (fc) - combines hidden state with embedded token
    // Input: [hidden_state, embedded_token] (2 * n_embd)
    // Output: n_embd
    ggml_tensor * fc_weight = nullptr;
    ggml_tensor * fc_bias   = nullptr;

    // Single decoder layer (layer 0)
    struct {
        // Attention norm (may be nullptr for EAGLE)
        ggml_tensor * attn_norm = nullptr;

        // Self-attention projections
        ggml_tensor * wq = nullptr;  // Q projection
        ggml_tensor * wk = nullptr;  // K projection
        ggml_tensor * wv = nullptr;  // V projection
        ggml_tensor * wo = nullptr;  // Output projection

        // FFN norm
        ggml_tensor * ffn_norm = nullptr;

        // FFN (SwiGLU)
        ggml_tensor * ffn_gate = nullptr;  // gate projection
        ggml_tensor * ffn_up   = nullptr;  // up projection
        ggml_tensor * ffn_down = nullptr;  // down projection
    } layer;

    // GGML context for tensors
    ggml_context * ctx = nullptr;

    // Backend buffer
    ggml_backend_buffer_t buffer = nullptr;

    // Whether tensors are loaded
    bool loaded = false;

    ~llama_eagle_head();
};

// Load EAGLE head from GGUF file
// Returns nullptr on failure
// The EAGLE head must be compatible with the base model (same hidden size, vocab)
LLAMA_API struct llama_eagle_head * llama_eagle_load(
        const char * path_gguf,
        const struct llama_model * model_base);

// Free EAGLE head
LLAMA_API void llama_eagle_free(struct llama_eagle_head * eagle);

// Check if EAGLE head is compatible with base model
LLAMA_API bool llama_eagle_is_compatible(
        const struct llama_eagle_head * eagle,
        const struct llama_model * model);

// Generate draft tokens using EAGLE head
// Input:
//   - ctx: target model context (for LM head)
//   - eagle: EAGLE head
//   - hidden_states: penultimate layer hidden states from target model [n_embd]
//   - last_token: last generated token
//   - n_draft: number of draft tokens to generate
// Output:
//   - draft_tokens: array to store draft tokens (must have space for n_draft)
// Returns: number of draft tokens generated
LLAMA_API int32_t llama_eagle_generate_draft(
        struct llama_context * ctx,
        struct llama_eagle_head * eagle,
        const float * hidden_states,
        llama_token last_token,
        int32_t n_draft,
        llama_token * draft_tokens);
