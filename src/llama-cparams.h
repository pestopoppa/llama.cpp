#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    bool     rs_replay;       // roll back by replaying kept tokens instead of keeping n_rs_seq snapshots
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;

    // Dual Chunk Attention (0 = off); see llama_context_params
    uint32_t dca_chunk_size;
    uint32_t dca_local_size;
    uint32_t dca_orig_ctx;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    // MoE-Spec budget (arXiv:2602.16052) for spec-dec verification batches:
    // top-B aggregate-routing-score expert shortlist, applied per-batch before per-token argsort_top_k.
    // 0 = off (default); 1..n_expert-1 = budget. Fires only when n_tokens >= moe_spec_min_batch.
    int32_t moe_spec_budget;
    int32_t moe_spec_min_batch;  // default 4; minimum batch size to trigger budgeting

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;

    // [GPU-RESIDENT-ROUND-2d] LLAMA_DEVICE_LATTICE=1 inputs of the device DFlash2 lattice selection (owned by the context)
    struct llama_dsel_inputs * dsel = nullptr;

    // [GPU-RESIDENT-ROUND-2c] LLAMA_DEVICE_DFLASH_FEED=1: persistent device tensor [n_embd, n_ubatch] the DFlash encoder
    // writes and the KV-injection decode reads, so the fused target features never round-trip through the host
    struct ggml_tensor * dflash_feed = nullptr;

    // [GPU-RESIDENT-ROUND-2e] device target->drafter feature gather (owned by the drafter context)
    struct llama_dgather_inputs * dgather = nullptr;
};

// [GPU-RESIDENT-ROUND-2d] per decode: inverse temperature per draft block (1 = greedy) and Gumbel noise
// [top_k, block_tokens, n_blocks] (0 = greedy); empty vectors mean greedy for every block
// [GPU-RESIDENT-ROUND-2e] the target context's device layer-input tensors (one per DFlash target layer, in feature
// order) and the target batch rows of the next encoder chunk
struct llama_dgather_inputs {
    std::vector<struct ggml_tensor *> feats;
    std::vector<int32_t>              rows;
};

struct llama_dsel_inputs {
    std::vector<float> inv_temp;
    std::vector<float> noise;
    // [GPU-RESIDENT-ROUND-2f phase 3.2] the target context's verify-token buffer F32 [T + 1, n_stream] (nullptr = off)
    struct ggml_tensor * vtok_dst = nullptr;
    int64_t              vtok_T   = 0;
    // [GPU-RESIDENT-ROUND-2f phase 3.4c] LLAMA_DEVICE_DRAFT_IN=1: the target context's KV cache (control block + mirrors)
    const class llama_kv_cache * tgt_kv = nullptr;
    // [GPU-RESIDENT-ROUND-2f phase 3.4d] LLAMA_DEVICE_DSEL_RNG=1: persistent per-seq selection state on the drafter's
    // device: rng_key I32 [2, n_seq] (Philox key lo/hi), rng_st F32 [2, n_seq] (draft counter, -1 = greedy; inv_temp).
    // The Gumbel noise is generated in the graph (GGML_OP_PHILOX_GUMBEL) and the counter advances on the device;
    // the host mirrors (*_m, predicted after every draft) are written only when the wanted state (*_w) disagrees.
    struct ggml_tensor * rng_key = nullptr;
    struct ggml_tensor * rng_st  = nullptr;
    std::vector<int32_t> key_w, key_m;
    std::vector<float>   st_w,  st_m;
    bool     rng_any_temp = false;
    uint64_t rng_hits = 0, rng_writes = 0;
};
