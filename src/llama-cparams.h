#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool kv_hadamard;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    // MoE self-drafting: override n_expert_used
    // 0 = use model default, 1+ = force exactly N active experts
    int32_t moe_n_expert_override;

    // MoE-Spec budget (arXiv:2602.16052) for spec-dec verification batches:
    // top-B aggregate-routing-score expert shortlist, applied per-batch before per-token argsort_top_k.
    // 0 = off (default); 1..n_expert-1 = budget. Fires only when n_tokens >= moe_spec_min_batch.
    int32_t moe_spec_budget;
    int32_t moe_spec_min_batch;  // default 4; minimum batch size to trigger budgeting

    // TIDE early exit: number of layers to compute (0 = all layers)
    // When > 0 and < n_layer, the model exits early after this many layers.
    // Used by TIDE router for per-token adaptive layer exit.
    int32_t n_layer_exit;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
