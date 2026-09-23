#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include "gguf.h"
#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cmath>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// DeepSeek-V4.1-Flash (arch "deepseek41").
//
// The serving artifact (antirez/deepseek-v4.1-flash-gguf, converted for antirez/ds4 @ ds4.1flash)
// publishes its hyperparameters under the raw HuggingFace `text_config` field names. Most of those
// are mapped back onto the canonical llm_kv ids by LLM_KV_ALIASES in llama-arch.cpp; the ones
// handled here are the keys whose stored GGUF type does not match what the canonical reader wants,
// plus the V4.1-only keys.
//
// Everything this function does NOT do is deliberate. See the flagged deltas at the bottom.

// engram.encoding is "e4m3_e8m0_32_row<N>": N bytes per packed table row.
static uint32_t dsv41_engram_row_bytes(const std::string & encoding) {
    const std::string prefix = "e4m3_e8m0_32_row";

    if (encoding.compare(0, prefix.size(), prefix) != 0) {
        throw std::runtime_error("deepseek41: unsupported engram.encoding '" + encoding + "'");
    }

    const std::string digits = encoding.substr(prefix.size());
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("deepseek41: malformed engram.encoding '" + encoding + "'");
    }

    return (uint32_t) std::stoul(digits);
}

// Read a scalar that this converter may have written as either UINT32 or FLOAT32: the canonical
// key first, then the raw-HF field `hf_field` under the arch prefix. These three keys are kept out
// of LLM_KV_ALIASES precisely because their stored type is not the canonical reader's type, and
// GKV::get_kv() throws on a mismatch even when the key is optional.
static bool dsv41_get_scalar_f32(llama_model_loader & ml, enum llm_kv kid, const char * hf_field, float & result) {
    std::string key = ml.kv_name(kid);
    if (gguf_find_key(ml.metadata, key.c_str()) < 0) {
        key = ml.get_arch_name() + "." + hf_field;
    }

    const int id = gguf_find_key(ml.metadata, key.c_str());
    if (id < 0) {
        return false;
    }

    switch (gguf_get_kv_type(ml.metadata, id)) {
        case GGUF_TYPE_FLOAT32: result = gguf_get_val_f32(ml.metadata, id); return true;
        case GGUF_TYPE_UINT32:  result = (float) gguf_get_val_u32(ml.metadata, id); return true;
        default:
            throw std::runtime_error("deepseek41: key " + key + " is neither f32 nor u32");
    }
}

static bool dsv41_get_scalar_u32(llama_model_loader & ml, enum llm_kv kid, const char * hf_field, uint32_t & result) {
    std::string key = ml.kv_name(kid);
    if (gguf_find_key(ml.metadata, key.c_str()) < 0) {
        key = ml.get_arch_name() + "." + hf_field;
    }

    const int id = gguf_find_key(ml.metadata, key.c_str());
    if (id < 0) {
        return false;
    }

    switch (gguf_get_kv_type(ml.metadata, id)) {
        case GGUF_TYPE_UINT32:  result = gguf_get_val_u32(ml.metadata, id); return true;
        case GGUF_TYPE_FLOAT32: result = (uint32_t) gguf_get_val_f32(ml.metadata, id); return true;
        default:
            throw std::runtime_error("deepseek41: key " + key + " is neither u32 nor f32");
    }
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    // --- keys shared with V4, reached through LLM_KV_ALIASES ---------------------------------
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp, hparams.n_layer_all);
    hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);

    // V4.1 has no token-hash MoE routing layers and the key is absent; V4 requires it.
    hparams.dsv4_hash_layer_count = 0;
    ml.get_key(LLM_KV_HASH_LAYER_COUNT, hparams.dsv4_hash_layer_count, false);

    // V4.1 carries no nextn/MTP blocks in this GGUF (the upstream checkpoint has 3). The generic
    // reader already left n_layer_nextn at 0 because no key is present; assert that, rather than
    // letting a later converter silently change the trunk depth under the tensor loop below.
    if (hparams.n_layer_nextn != 0) {
        throw std::runtime_error("deepseek41: nextn/MTP blocks are not supported yet (DS41-B6)");
    }

    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    // --- keys whose stored type does not match the canonical reader --------------------------
    // rope_theta and compress_rope_theta are UINT32 in this file; the canonical readers want f32
    // and GKV::get_kv() throws on a type mismatch even for a non-required key, so they must not
    // go through LLM_KV_ALIASES.
    dsv41_get_scalar_f32(ml, LLM_KV_ROPE_FREQ_BASE, "rope_theta",
            hparams.rope_freq_base_train);
    dsv41_get_scalar_f32(ml, LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, "compress_rope_theta",
            hparams.dsv4_compress_rope_base);
    // ...and rope_scaling.original_max_position_embeddings is FLOAT32 where we want a uint32.
    dsv41_get_scalar_u32(ml, LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, "rope_scaling.original_max_position_embeddings",
            hparams.n_ctx_orig_yarn);

    // scoring_func is the string "sqrtsoftplus"; the canonical key is the gating-func enum.
    {
        std::string scoring_func;
        if (ml.get_key(LLM_KV_EXPERT_GATING_FUNC, scoring_func, false)) {
            if (scoring_func == "sqrtsoftplus") {
                hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS;
            } else {
                throw std::runtime_error("deepseek41: unsupported scoring_func '" + scoring_func + "'");
            }
        } else {
            ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
        }
    }

    // The file has rope_scaling.{factor,beta_fast,beta_slow,original_max_position_embeddings} but
    // no rope_scaling.rope_type, so the generic reader defaulted to "linear". The checkpoint is
    // YaRN (config.json: rope_scaling.rope_type == "yarn").
    hparams.rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_YARN;

    // --- compression ratios -------------------------------------------------------------------
    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("deepseek41: compress_ratios is shorter than block_count");
    }
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    // V4 allows {0, 4, 128}; V4.1 uses {0, 1, 2}. Accept the union — the V4 values stay valid so
    // a V4 GGUF loaded through this path is not silently reinterpreted.
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        switch (hparams.dsv4_compress_ratios[il]) {
            case 0: case 1: case 2: case 4: case 128:
                break;
            default:
                throw std::runtime_error("deepseek41: unsupported compression ratio " +
                        std::to_string(hparams.dsv4_compress_ratios[il]) +
                        " on layer " + std::to_string(il));
        }
    }

    // --- V4.1-only keys ------------------------------------------------------------------------
    {
        std::vector<uint32_t> kv_source_ids;
        std::vector<uint32_t> index_source_ids;

        ml.get_arr(LLM_KV_DSV41_KV_SOURCE_LAYERS,    kv_source_ids);
        ml.get_arr(LLM_KV_DSV41_INDEX_SOURCE_LAYERS, index_source_ids);

        hparams.dsv41_is_kv_source.fill(false);
        hparams.dsv41_is_index_source.fill(false);

        for (uint32_t il : kv_source_ids) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error("deepseek41: kv_source_layer_ids out of range");
            }
            hparams.dsv41_is_kv_source[il] = true;
        }

        for (uint32_t il : index_source_ids) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error("deepseek41: index_source_layer_ids out of range");
            }
            hparams.dsv41_is_index_source[il] = true;
        }
    }

    ml.get_key(LLM_KV_DSV41_CANDIDATE_SOURCE_LAYER, hparams.dsv41_candidate_source_layer, false);
    ml.get_key(LLM_KV_DSV41_CANDIDATE_TOPK_BLOCKS,  hparams.dsv41_candidate_topk_blocks,  false);
    ml.get_key(LLM_KV_DSV41_CANDIDATE_BLOCK_SIZE,   hparams.dsv41_candidate_block_size,   false);

    {
        std::vector<uint32_t> engram_layer_ids;
        std::vector<uint32_t> engram_rows;
        std::string           engram_encoding;

        hparams.dsv41_engram_rows.fill(0);

        if (ml.get_arr(LLM_KV_DSV41_ENGRAM_LAYERS, engram_layer_ids, false)) {
            ml.get_arr(LLM_KV_DSV41_ENGRAM_ROWS, engram_rows);
            ml.get_key(LLM_KV_DSV41_ENGRAM_ENCODING, engram_encoding);

            if (engram_rows.size() != engram_layer_ids.size()) {
                throw std::runtime_error("deepseek41: engram.rows and engram.layer_ids differ in length");
            }

            hparams.dsv41_engram_row_bytes = dsv41_engram_row_bytes(engram_encoding);

            for (size_t i = 0; i < engram_layer_ids.size(); ++i) {
                const uint32_t il = engram_layer_ids[i];
                if (il >= hparams.n_layer_all) {
                    throw std::runtime_error("deepseek41: engram.layer_ids out of range");
                }
                hparams.dsv41_engram_rows[il] = engram_rows[i];
            }

            // The n-gram hash spec itself: token_map, primes, multipliers, pad_id. It is not
            // hparams (the token_map is one entry per vocabulary token), so it hangs off the
            // model. DS41-B8.
            uint32_t n_vocab = 0;
            ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab);

            dsv41_engram.load(ml, n_vocab);

            if (dsv41_engram.empty()) {
                throw std::runtime_error("deepseek41: engram tables are declared but the "
                                         "engram.* hash metadata is missing");
            }

            if (dsv41_engram.layer_ids != engram_layer_ids) {
                throw std::runtime_error("deepseek41: engram.layer_ids disagrees with itself");
            }

            for (size_t i = 0; i < engram_layer_ids.size(); ++i) {
                if (dsv41_engram.rows[i] != hparams.dsv41_engram_rows[engram_layer_ids[i]]) {
                    throw std::runtime_error("deepseek41: engram.rows disagrees with itself");
                }
            }
        }
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head    = hparams.n_embd_head_k();
    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t o_groups       = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank    = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult        = hparams.dsv4_hc_mult;
    const int64_t hc_dim         = hc_mult * n_embd;
    const int64_t hc_mix_dim     = (2 + hc_mult) * hc_mult;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // V4 collapses the hc_mult output streams with a learned head; V4.1's published GGUF has no
    // output_hc_* tensors. Optional here; the graph must not assume they exist (DS41-B11).
    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult}, TENSOR_NOT_REQUIRED);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult},         TENSOR_NOT_REQUIRED);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1},               TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        // the file stores wo_a as (n_head*n_embd_head/o_groups, o_lora_rank*o_groups); reshape on load
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        // V4 keys the compressor off the per-layer compression ratio. V4.1 shares one compressor
        // across a run of layers, so the weights exist only on kv_source_layer_ids and the
        // indexer weights only on index_source_layer_ids.
        if (hparams.dsv41_is_kv_source[i]) {
            const int64_t ratio = hparams.dsv4_compress_ratios[i];
            const int64_t coff  = ratio == 4 ? 2 : 1;

            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, coff * n_embd_head}, 0);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, 0);
            // the last KV-source layer (20 in this checkpoint) has no compressor gate
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, coff * n_embd_head}, TENSOR_NOT_REQUIRED);
            // V4's attn_compressor_ape (absolute positional embedding over the compressed block)
            // has no V4.1 counterpart; V4.1 uses compress_rope_theta instead.
        }

        if (hparams.dsv41_is_index_source[i]) {
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, 0);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, 0);

            // the indexer key is projected from the compressed KV latent, so indexer_attn_k and
            // indexer.k_norm exist only where an index source is also a KV source
            if (hparams.dsv41_is_kv_source[i]) {
                layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", i), {n_embd_head, n_embd_indexer}, 0);
                layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {n_embd_indexer}, 0);
            }
        }

        // Engram tables. engram_embd is the packed FP8 n-gram table stored as I8 rows of
        // `dsv41_engram_row_bytes` bytes (256 e4m3 values + 8 e8m0 block scales); engram_q_norm /
        // engram_k_norm are the per-stream learned gate vectors, NOT norms. Nothing dereferences
        // them yet: the FP8-row get_rows/dequant op is DS41-B7.
        if (const uint64_t engram_rows = hparams.dsv41_engram_rows[i]) {
            const int64_t row_bytes = hparams.dsv41_engram_row_bytes;

            // (max_ngram_size - 1) * engram_n_heads * engram_head_dim is not published as GGUF
            // metadata (it lives only in the embedded deepseek41.config blob), so take the
            // Engram projection's input width from the tensor itself.
            const llama_model_loader::llama_tensor_weight * w_kv = ml.get_weight(tn(LLM_TENSOR_ENGRAM_KV, "weight", i).str().c_str());
            if (w_kv == nullptr) {
                throw std::runtime_error("deepseek41: engram.layer_ids names layer " + std::to_string(i) +
                        " but blk." + std::to_string(i) + ".engram_kv.weight is missing");
            }
            const int64_t n_hash_embd = w_kv->tensor->ne[0];

            layer.engram_embd   = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD,   "weight", i), {row_bytes, (int64_t) engram_rows}, 0);
            layer.engram_kv     = create_tensor(tn(LLM_TENSOR_ENGRAM_KV,     "weight", i), {n_hash_embd, n_embd * (hc_mult + 1)}, 0);
            layer.engram_q_norm = create_tensor(tn(LLM_TENSOR_ENGRAM_Q_NORM, "weight", i), {n_embd, hc_mult}, 0);
            layer.engram_k_norm = create_tensor(tn(LLM_TENSOR_ENGRAM_K_NORM, "weight", i), {n_embd, hc_mult}, 0);
        }

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,       "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,    "bias",   i), {n_expert}, 0);
        // the vision router bias rides along in this text-only GGUF; registered with GGML_OP_NONE
        // so the loader accounts for it and skips it instead of failing the tensor count
        create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
    }
}

//
// DS41-B8 -- Engram conditional memory.
//
// Normative: DeepSeek-V4.1-Flash/inference/engram.py and inference/model.py:296-365
// (ParallelEngramEmbedding + Engram). Working cross-check: antirez/ds4 @ 0aaea5a,
// ds4_engram.c and ds4_deepseek41_cuda.cuh:179-202 (the gate kernel).
//
// Per BATCH, on the host: map token ids through engram.token_map, take the 4-token suffix
// newest-first with a sticky blocked flag, and turn it into 24 table-row ids per token per
// engram layer (llm_graph_input_dsv41_engram::set_input). There is no prefetch depth at
// decode: the newest element of the n-gram is the token just sampled, so the row ids for step
// t cannot be known before step t-1 has produced its token.
//
// Per TOKEN, in the graph: gather those 24 rows (256 dequantized floats each), project them
// through engram_kv to hc_mult keys plus one shared value, and add the value into every
// hyper-connection stream, scaled by a per-(token, stream) sigmoid gate.
//

llm_graph_input_dsv41_engram * llama_model_deepseek4::graph::build_inp_dsv41_engram(const llama_model & model) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);

    llama_dsv41_engram_state * state = mctx_cur ? mctx_cur->get_engram_state() : nullptr;

    if (state == nullptr) {
        // V4, or a V4.1 GGUF with no engram tables
        return nullptr;
    }

    const llama_dsv41_engram_spec * spec = &model.dsv41_engram;

    GGML_ASSERT(!spec->empty());

    auto inp = std::make_unique<llm_graph_input_dsv41_engram>(spec, state, (uint32_t) n_tokens);

    inp->ids.resize(spec->n_engram);

    for (uint32_t e = 0; e < spec->n_engram; ++e) {
        inp->ids[e] = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, spec->n_col, n_tokens);
        ggml_set_input(inp->ids[e]);
        ggml_format_name(inp->ids[e], "engram_ids_l%u", spec->layer_ids[e]);
    }

    inp->gate_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(inp->gate_mask);
    ggml_set_name(inp->gate_mask, "engram_gate_mask");

    return (llm_graph_input_dsv41_engram *) res->add_input(std::move(inp));
}

ggml_tensor * llama_model_deepseek4::graph::build_engram(
        const llama_model & model,
        llm_graph_input_dsv41_engram * inp,
        ggml_tensor * h,
        int il) const {
    if (inp == nullptr) {
        return h;
    }

    const llama_dsv41_engram_spec & spec = model.dsv41_engram;

    const int e = spec.index_of_layer((uint32_t) il);
    if (e < 0) {
        return h;
    }

    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.engram_embd   != nullptr);
    GGML_ASSERT(layer.engram_kv     != nullptr);
    GGML_ASSERT(layer.engram_q_norm != nullptr);
    GGML_ASSERT(layer.engram_k_norm != nullptr);

    const int64_t hc  = hparams.dsv4_hc_mult;
    const float   eps = hparams.f_norm_rms_eps;

    GGML_ASSERT(h->ne[0] == n_embd && h->ne[1] == hc && h->ne[2] == n_tokens);

    // --- the lookup -----------------------------------------------------------------------
    // engram_embd is the packed FP8 table, I8 [33*k, rows]. Only the gathered rows are read;
    // the 189 GiB table is never scanned and never forced resident, and there is no LRU --
    // antirez has none either (ds4_engram.c:276-277).
    ggml_tensor * rows = ggml_gather_rows_e4m3_e8m0(ctx0, layer.engram_embd, inp->ids[e]);
    cb(rows, "engram_rows", il);

    // [head_dim, n_col, n_tokens] -> [n_col*head_dim, n_tokens]; engram.py:353 does the same
    // flatten(-2), so column c occupies [c*head_dim, (c+1)*head_dim).
    ggml_tensor * flat = ggml_reshape_2d(ctx0, rows, rows->ne[0]*rows->ne[1], n_tokens);

    GGML_ASSERT(flat->ne[0] == layer.engram_kv->ne[0]);
    GGML_ASSERT(layer.engram_kv->ne[1] == n_embd*(hc + 1));

    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.engram_kv, flat); // [n_embd*(hc+1), n_tokens]
    cb(kv, "engram_kv", il);

    // engram.py:354 splits [hc_mult*dim | dim]: hc_mult per-stream keys, then one shared value.
    ggml_tensor * key = ggml_cont(ctx0, ggml_view_3d(ctx0, kv,
                n_embd, hc, n_tokens,
                kv->nb[0]*n_embd, kv->nb[1], 0));

    ggml_tensor * val = ggml_cont(ctx0, ggml_view_2d(ctx0, kv,
                n_embd, n_tokens,
                kv->nb[1], kv->nb[0]*n_embd*hc));

    // --- the gate -------------------------------------------------------------------------
    // engram.py:356-362:
    //   weight = q_weight * k_weight                       (only ever used as the product)
    //   rstd   = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)   per (token, stream)
    //   dot    = (h * weight * key).sum(-1) * rstd * dim^-0.5
    //   gate   = sigmoid(copysign(sqrt(clamp_min(|dot|, 1e-6)), dot))
    // The two rsqrt terms are exactly what ggml_rms_norm applies, so folding them in keeps the
    // normalization per (token, stream) over dim and NOT jointly across streams.
    ggml_tensor * hn = ggml_rms_norm(ctx0, h,   eps);
    ggml_tensor * kn = ggml_rms_norm(ctx0, key, eps);

    ggml_tensor * w = ggml_mul(ctx0, layer.engram_q_norm, layer.engram_k_norm); // [n_embd, hc]

    ggml_tensor * dot = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_mul(ctx0, hn, kn), w));
    dot = ggml_scale(ctx0, dot, 1.0f/sqrtf((float) n_embd)); // [1, hc, n_tokens]
    cb(dot, "engram_dot", il);

    // signed sqrt. ggml_sgn(0) is 0 where torch.copysign(y, +0.0) is +y, so a dot of exactly
    // +0.0f gives sigmoid(0) here and sigmoid(1e-3) in the reference -- a 2.5e-4 gate delta on
    // a measure-zero input. Flagged in NOTES.md rather than papered over.
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, mag, ggml_sgn(ctx0, dot)));

    // engram.py:363-364: a dead/image position passes through untouched.
    gate = ggml_mul(ctx0, gate, inp->gate_mask);
    cb(gate, "engram_gate", il);

    // --- the write ------------------------------------------------------------------------
    // engram.py:365: h + gate * value, the SAME value added to every stream.
    ggml_tensor * v = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, val, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);

    ggml_tensor * out = ggml_add(ctx0, h, ggml_mul(ctx0, v, gate));
    cb(out, "engram_out", il);

    return out;
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        // This GGUF carries no nextn blocks (load_arch_hparams asserts n_layer_nextn == 0), and
        // the upstream V4.1 "mtp.*" namespace is DSpark, not a NextN MTP head (model.py:1100-1101).
        GGML_ABORT("deepseek41: native MTP is not implemented (DS41-B6)");
    }

    return std::make_unique<graph>(*this, params);
}

//
// DeepSeek-V4.1 graph. Cross-referenced to SPEC.md throughout; read that first.
//

ggml_tensor * llama_model_deepseek41::graph::build_hc_stream0(ggml_tensor * x) const {
    // make_identity_pre_mix (model.py:1159-1163) is one-hot on stream 0, so hc_pre degenerates to
    // picking that stream. Exact, not an approximation -- ds4.c:41277-41278 uses the same {1,0,0,0}.
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == (int64_t) hparams.dsv4_hc_mult);

    return ggml_view_2d(ctx0, x, n_embd, x->ne[2], x->nb[2], 0);
}

ggml_tensor * llama_model_deepseek41::graph::build_hc_mean(ggml_tensor * x) const {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx0, acc, ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }

    return ggml_scale(ctx0, acc, 1.0f/hc);
}

// SPEC.md 7.2: V4.1 applies no YaRN magnitude attenuation, unlike V4's dsv4_rope_attn_factor().
static constexpr float DSV41_ROPE_ATTN_FACTOR = 1.0f;

//
// SPEC.md sections 4-6: the compressed-KV group, the indexer, and the two-level candidate stage.
//

// A finite stand-in for +/-infinity. The block max-pool below computes max(a,b) as
// a + relu(b - a), and -inf minus -inf is NaN, so masked scores are floored first. The reference
// kernels do the same (kernel.py:355 uses -1e30, ds4_cuda.cu:10399 folds the sink the same way).
static constexpr float DSV41_BIG = 1e30f;

struct dsv41_state_tensors {
    ggml_tensor * kv;
    ggml_tensor * score;
};

static size_t dsv41_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv41_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv41_elem_offset(t, i0));
}

// Identical in shape to deepseek4.cpp's static dsv4_build_state_restore / _snapshot, which are not
// visible from this translation unit.
static dsv41_state_tensors dsv41_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il) {
    dsv41_state_tensors restored = {
        state->get_kv_all(ctx, il),
        state->get_score_all(ctx, il),
    };

    if (inp.state_restore_src_idxs == nullptr || inp.state_restore_dst_idxs == nullptr) {
        return restored;
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);

    return restored;
}

static dsv41_state_tensors dsv41_build_state_snapshot(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        ggml_tensor * source_kv,
        ggml_tensor * source_score,
        int32_t il) {
    if (inp.state_snapshot_src_idxs == nullptr || inp.state_snapshot_dst_idxs == nullptr ||
            source_kv == nullptr || source_score == nullptr) {
        return {};
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, source_kv, inp.state_snapshot_src_idxs);
    ggml_tensor * kv = state->cpy_kv(ctx, kv_rows, inp.state_snapshot_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, source_score, inp.state_snapshot_src_idxs);
    ggml_tensor * score = state->cpy_score(ctx, score_rows, inp.state_snapshot_dst_idxs, il);

    return { kv, score };
}

// The layer whose compressed plane (or indexer-key plane) this layer reads. Sources own a plane;
// every other layer of the same ratio run reads the most recent preceding source's
// (model.py:746-748 / :763 and model.py:537-548, SPEC.md 4.4 / 5.1; ds4.c:40623's owner index).
static int dsv41_source_layer(const llama_hparams & hp, int il, bool index_key) {
    const uint32_t ratio = hp.dsv4_compress_ratios[il];

    for (int j = il; j >= 0; --j) {
        if (hp.dsv4_compress_ratios[j] != ratio) {
            break;
        }
        // only a KV source can produce index keys, because they are projected from its latent
        if (hp.dsv41_is_kv_source[j] && (!index_key || hp.dsv41_is_index_source[j])) {
            return j;
        }
    }

    return -1;
}

ggml_tensor * llama_model_deepseek41::graph::build_v41_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_blocks = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    // non-overlapping: exactly `ratio` source rows per block (SPEC.md 4.2). V4's ratio-4 path
    // reads 2*ratio and doubles the projection width; V4.1 does neither.
    GGML_ASSERT(state_read_idxs->ne[0] == ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == n_embd_head);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, ratio, n_blocks);
    cb(kv, name, il);

    ggml_tensor * comp = nullptr;
    if (score_state == nullptr) {
        // ratio 1, ungated: model.py:461-462 returns norm(wkv(x)) with no pooling at all.
        GGML_ASSERT(ratio == 1);
        comp = kv;
    } else {
        GGML_ASSERT(score_state->ne[0] == n_embd_head);

        ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
        score = ggml_reshape_3d(ctx0, score, n_embd_head, ratio, n_blocks);

        // put the group axis on ne0 so soft_max normalises over the `ratio` timesteps of each
        // channel independently (model.py:475, ds4_deepseek41_cuda.cuh:219-231)
        ggml_tensor * values  = ggml_cont(ctx0, ggml_permute(ctx0, kv,    1, 0, 2, 3));
        ggml_tensor * scores  = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));
        ggml_tensor * weights = ggml_soft_max(ctx0, scores);

        comp = ggml_mul(ctx0, values, weights);
        comp = ggml_sum_rows(ctx0, comp);
        comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    }
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    // returned PRE-RoPE on purpose: the indexer key is projected from this (model.py:434, :530-531)
    return comp;
}

ggml_tensor * llama_model_deepseek41::graph::build_v41_candidate_mask(
        ggml_tensor * scores,
        ggml_tensor * n_visible,
        int il) const {
    const int64_t block  = hparams.dsv41_candidate_block_size;
    const int64_t n_comp = scores->ne[0];
    const int64_t nt     = scores->ne[1];
    const int64_t ns     = scores->ne[3];
    const int64_t n_blk  = (n_comp + block - 1)/block;

    GGML_ASSERT(block > 0);
    GGML_ASSERT(n_visible && "deepseek41: the candidate stage needs the per-token visible count");
    GGML_ASSERT(scores->ne[2] == 1);

    ggml_tensor * s = ggml_clamp(ctx0, scores, -DSV41_BIG, INFINITY);

    const int64_t pad = n_blk*block - n_comp;
    if (pad > 0) {
        ggml_tensor * tail = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, pad, nt, 1, ns);
        tail = ggml_fill(ctx0, tail, -DSV41_BIG);
        s = ggml_concat(ctx0, s, tail, 0);
    }
    s = ggml_reshape_4d(ctx0, ggml_cont(ctx0, s), block, n_blk, nt, ns);

    // block score = max over the block (model.py:599, ds4_deepseek41_cuda.cuh:267-269)
    ggml_tensor * pooled = ggml_cont(ctx0,
            ggml_view_4d(ctx0, s, 1, n_blk, nt, ns, s->nb[1], s->nb[2], s->nb[3], 0));
    for (int64_t j = 1; j < block; ++j) {
        ggml_tensor * sj = ggml_cont(ctx0,
                ggml_view_4d(ctx0, s, 1, n_blk, nt, ns, s->nb[1], s->nb[2], s->nb[3], j*s->nb[0]));
        pooled = ggml_add(ctx0, pooled, ggml_relu(ctx0, ggml_sub(ctx0, sj, pooled)));
    }
    pooled = ggml_reshape_4d(ctx0, pooled, n_blk, nt, 1, ns);
    cb(pooled, "cand_block_scores", il);

    // pin the block holding this query's newest visible position (model.py:604-605; ds4's
    // `if (visible && col == (visible - 1u)/8u) best = INFINITY`).
    //   keep  <=>  blk*block <= n_visible-1  and  n_visible <= (blk+1)*block
    ggml_tensor * blk = ggml_reshape_4d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_blk, 1.0f), n_blk, 1, 1, 1);
    ggml_tensor * lo  = ggml_scale(ctx0, blk, (float) block);

    ggml_tensor * nv = ggml_repeat_4d(ctx0,
            ggml_reshape_4d(ctx0, n_visible, 1, nt, 1, ns), n_blk, nt, 1, ns);

    ggml_tensor * a = ggml_sub(ctx0, ggml_scale_bias(ctx0, nv, 1.0f, -0.5f), lo);
    ggml_tensor * b = ggml_add(ctx0, ggml_scale(ctx0, nv, -1.0f),
            ggml_scale_bias(ctx0, lo, 1.0f, (float) block + 0.5f));

    ggml_tensor * pin = ggml_mul(ctx0, ggml_step(ctx0, a), ggml_step(ctx0, b));
    pooled = ggml_add(ctx0, pooled, ggml_scale(ctx0, pin, DSV41_BIG));
    cb(pooled, "cand_block_scores_pinned", il);

    const int64_t n_keep = std::min<int64_t>((int64_t) hparams.dsv41_candidate_topk_blocks, n_blk);
    GGML_ASSERT(n_keep > 0);

    ggml_tensor * top = ggml_cont(ctx0, ggml_top_k(ctx0, pooled, n_keep));

    // scatter zeros onto a -inf plane, exactly as build_top_k_mask does for positions
    ggml_tensor * all = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_blk, nt, 1, ns), -INFINITY);
    all = ggml_view_4d(ctx0, all, 1, all->ne[0], all->ne[1], all->ne[3],
            all->nb[0], all->nb[1], all->nb[2], 0);

    ggml_tensor * top3 = ggml_view_4d(ctx0, top, top->ne[0], top->ne[1], top->ne[3], 1,
            top->nb[1], top->nb[2], top->ne[3]*top->nb[3], 0);

    ggml_tensor * zeros = ggml_fill(ctx0,
            ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top3->ne[0], top3->ne[1], top3->ne[2]), 0.0f);

    ggml_tensor * keep = ggml_set_rows(ctx0, all, zeros, top3);
    keep = ggml_view_4d(ctx0, keep, keep->ne[1], keep->ne[2], 1, keep->ne[3],
            keep->nb[2], keep->nb[3], keep->nb[3], 0);
    keep = ggml_cont(ctx0, keep);

    // repeat_interleave(block) then crop back to n_comp (model.py:610)
    keep = ggml_reshape_4d(ctx0, keep, 1, n_blk, nt, ns);
    keep = ggml_repeat_4d(ctx0, keep, block, n_blk, nt, ns);
    keep = ggml_reshape_4d(ctx0, keep, n_blk*block, nt, 1, ns);
    keep = ggml_view_4d(ctx0, keep, n_comp, nt, 1, ns, keep->nb[1], keep->nb[2], keep->nb[3], 0);
    cb(keep, "cand_mask", il);

    // NOTE (SPEC.md 6): model.py:609 additionally drops blocks whose pooled score is still -inf.
    // That is redundant here: an unreachable position is already -inf in `scores` from the
    // visibility mask, and the final attention mask is built from inp_comp.kq_mask independently,
    // so such a position can never contribute even if its block survives this stage.
    return keep;
}

ggml_tensor * llama_model_deepseek41::graph::build_v41_lid_top_k(
        const llama_model * model,
        const llm_graph_input_dsv4::comp_input & inp_idx,
        const llama_kv_cache_dsv4_comp_context * idx_ctx,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int idx_src_il,
        int il) const {
    const auto & layer = model->layers[il];

    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_idx.kq_mask);
    GGML_ASSERT(inp_idx.k_rot);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);
    GGML_ASSERT(layer.indexer_attn_q_b && layer.indexer_proj);

    // model.py:550-551 -- q from the shared, normed q-LoRA activation, roped at the layer's own
    // (compressed) theta. attn_factor 1.0, SPEC.md 7.2.
    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    ggml_tensor * indexer_q_nope = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            0);
    ggml_tensor * indexer_q_pe = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));

    indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, DSV41_ROPE_ATTN_FACTOR, beta_fast, beta_slow);
    cb(indexer_q_pe, "lid_q_pe", il);

    indexer_q = ggml_concat(ctx0, indexer_q_nope, indexer_q_pe, 0);
    indexer_q = llama_mul_mat_hadamard(ctx0, indexer_q, inp_idx.k_rot);
    cb(indexer_q, "lid_q_rot", il);

    // model.py:555 -- softmax_scale * n_heads^-0.5 == 1/sqrt(index_head_dim*index_n_heads)
    ggml_tensor * indexer_weights = build_lora_mm(layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    // keys come from the group's owner, not from this layer (SPEC.md 5.1)
    ggml_tensor * indexer_k = idx_ctx->get_k(ctx0, idx_src_il);
    const int64_t n_lid = inp_idx.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexer_k->ne[3];
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    // Always the explicit path: the candidate stage has to see the masked scores, and the fused
    // lightning-indexer op folds the mask and the top-k together. SPEC.md 6 / NOTES.md.
    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);

    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
    cb(indexer_kq, "lid_kq", il);

    ggml_tensor * indexer_score = ggml_relu(ctx0, indexer_kq);
    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
    indexer_score = ggml_sum_rows(ctx0, indexer_score);
    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
    cb(indexer_score, "lid_score", il);

    ggml_tensor * lid_mask = inp_idx.kq_mask;
    if (lid_mask->type != GGML_TYPE_F32) {
        lid_mask = ggml_cast(ctx0, lid_mask, GGML_TYPE_F32);
    }
    indexer_score = ggml_add(ctx0, indexer_score, lid_mask);
    cb(indexer_score, "lid_score_masked", il);

    // --- two-level candidate selection (SPEC.md 6) ------------------------------------------
    const int cand_src = (int) hparams.dsv41_candidate_source_layer;
    if (hparams.dsv41_candidate_block_size != 0) {
        if (il == cand_src) {
            shared_candidates = build_v41_candidate_mask(indexer_score, inp_idx.n_visible, il);
        } else if (il > cand_src && shared_candidates) {
            indexer_score = ggml_add(ctx0, indexer_score, shared_candidates);
            cb(indexer_score, "lid_score_candidates", il);
        }
    }

    const uint32_t n_top_k = indexer_score->ne[0] < hparams.indexer_top_k ?
        indexer_score->ne[0] : hparams.indexer_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_deepseek41::graph::build_compressed_attention(
        const llama_model * model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        float kq_scale,
        int il) const {
    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    if (ratio == 0) {
        return nullptr;
    }

    const auto & layer    = model->layers[il];
    const auto & geom     = inp_dsv4->mctx->get_geom();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const bool is_csa = (uint32_t) ratio == geom.ratio_csa;
    GGML_ASSERT((is_csa || (uint32_t) ratio == geom.ratio_hca) &&
            "deepseek41: a compression ratio with no cache group");

    const auto & inp_comp   = is_csa ? inp_dsv4->get_csa()          : inp_dsv4->get_hca();
    const auto & inp_idx    = is_csa ? inp_dsv4->get_lid()          : inp_dsv4->get_lid_b();
    const auto * comp_ctx   = is_csa ? inp_dsv4->mctx->get_csa()    : inp_dsv4->mctx->get_hca();
    const auto * idx_ctx    = is_csa ? inp_dsv4->mctx->get_lid()    : inp_dsv4->mctx->get_lid_b();
    const auto * comp_state = is_csa ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();

    GGML_ASSERT(comp_ctx && idx_ctx && comp_state);

    const int64_t n_embd_head    = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_embd_indexer = hparams.indexer_head_size;

    const int src_il     = dsv41_source_layer(hparams, il, false);
    const int idx_src_il = dsv41_source_layer(hparams, il, true);
    GGML_ASSERT(src_il >= 0 && idx_src_il >= 0 &&
            "deepseek41: a compressed layer with no preceding KV source in its ratio run");

    // =========================================================================================
    // 1. A KV source compresses its own KV and publishes it, plus the indexer key derived from
    //    the PRE-RoPE latent (model.py:739-763, :537-548; SPEC.md 4 and 5.1).
    // =========================================================================================
    if (hparams.dsv41_is_kv_source[il] && inp_comp.state_write_idxs) {
        GGML_ASSERT(layer.attn_comp_wkv && layer.attn_comp_norm);
        // SPEC.md 4.1: ratio 1 is ungated and has no attn_compressor_gate tensor at all.
        GGML_ASSERT((ratio == 1) == (layer.attn_comp_wgate == nullptr));

        ggml_tensor * comp_kv    = build_lora_mm(layer.attn_comp_wkv, cur);
        ggml_tensor * comp_score = layer.attn_comp_wgate ? build_lora_mm(layer.attn_comp_wgate, cur) : nullptr;
        // SPEC.md 4.2: NO absolute-position term. V4 adds attn_comp_ape[pos%ratio] to the gate
        // (deepseek4.cpp:995-997); V4.1 has no such tensor and model.py:465 adds nothing.
        cb(comp_kv, "comp_kv", il);

        const dsv41_state_tensors restored = dsv41_build_state_restore(ctx0, inp_comp, comp_state, il);

        ggml_tensor * base_kv = dsv41_view_2d(ctx0, restored.kv, restored.kv->ne[0], comp_state->get_n_rows(), 0);
        ggml_tensor * source_kv = ggml_concat(ctx0, base_kv, comp_kv, 1);

        ggml_tensor * source_score = nullptr;
        if (comp_score) {
            ggml_tensor * base_score = dsv41_view_2d(ctx0, restored.score, restored.score->ne[0],
                    comp_state->get_n_rows(), 0);
            source_score = ggml_concat(ctx0, base_score, comp_score, 1);
        }

        ggml_tensor * latent = build_v41_compressed_kv_from_state(
                source_kv, source_score,
                inp_comp.state_read_idxs, inp_comp.state_write_pos,
                layer.attn_comp_norm, ratio, n_embd_head, "comp_latent", il);

        const int64_t n_blocks = inp_comp.state_write_pos->ne[0];

        // --- indexer key, from the latent BEFORE it is rotated (model.py:544-547) ------------
        if (hparams.dsv41_is_index_source[il]) {
            GGML_ASSERT(layer.indexer_attn_k && layer.indexer_k_norm);

            ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k,
                    ggml_reshape_2d(ctx0, ggml_cont(ctx0, latent), n_embd_head, n_blocks));
            ik = build_norm(ik, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
            ik = ggml_reshape_3d(ctx0, ik, n_embd_indexer, 1, n_blocks);
            cb(ik, "lid_k_new", il);

            ggml_tensor * ik_nope = ggml_view_3d(ctx0, ik, n_embd_indexer - n_embd_head_rope, 1, n_blocks,
                    ggml_row_size(ik->type, n_embd_indexer),
                    ggml_row_size(ik->type, n_embd_indexer),
                    0);
            ggml_tensor * ik_pe = ggml_view_3d(ctx0, ik, n_embd_head_rope, 1, n_blocks,
                    ggml_row_size(ik->type, n_embd_indexer),
                    ggml_row_size(ik->type, n_embd_indexer),
                    ggml_row_size(ik->type, n_embd_indexer - n_embd_head_rope));
            ik_pe = ggml_rope_ext(ctx0, ik_pe, inp_comp.state_write_pos, nullptr, n_embd_head_rope,
                    rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
                    ext_factor, DSV41_ROPE_ATTN_FACTOR, beta_fast, beta_slow);
            ik = ggml_concat(ctx0, ik_nope, ik_pe, 0);

            if (inp_idx.k_rot) {
                ik = llama_mul_mat_hadamard(ctx0, ik, inp_idx.k_rot);
            }

            ggml_build_forward_expand(gf, idx_ctx->cpy_k(ctx0, ik, inp_comp.state_write_idxs, il));
        }

        // --- rotate the latent itself and write it (model.py:753-761) ------------------------
        ggml_tensor * comp_nope = ggml_view_3d(ctx0, latent, n_embd_head_nope, 1, n_blocks,
                ggml_row_size(latent->type, n_embd_head),
                ggml_row_size(latent->type, n_embd_head),
                0);
        ggml_tensor * comp_pe = ggml_view_3d(ctx0, latent, n_embd_head_rope, 1, n_blocks,
                ggml_row_size(latent->type, n_embd_head),
                ggml_row_size(latent->type, n_embd_head),
                ggml_row_size(latent->type, n_embd_head_nope));
        comp_pe = ggml_rope_ext(ctx0, comp_pe, inp_comp.state_write_pos, nullptr, n_embd_head_rope,
                rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
                ext_factor, DSV41_ROPE_ATTN_FACTOR, beta_fast, beta_slow);

        ggml_tensor * comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
        if (inp_comp.k_rot) {
            comp = llama_mul_mat_hadamard(ctx0, comp, inp_comp.k_rot);
        }
        cb(comp, "comp_latent_rot", il);

        ggml_build_forward_expand(gf, comp_ctx->cpy_k(ctx0, comp, inp_comp.state_write_idxs, il));

        // --- carry the partial group, and the rollback snapshot ------------------------------
        if (comp_score) {
            ggml_tensor * snap_kv    = ggml_concat(ctx0, restored.kv, comp_kv, 1);
            ggml_tensor * snap_score = ggml_concat(ctx0, restored.score, comp_score, 1);

            const dsv41_state_tensors snapshot = dsv41_build_state_snapshot(
                    ctx0, inp_comp, comp_state, snap_kv, snap_score, il);
            if (snapshot.kv) {
                ggml_build_forward_expand(gf, snapshot.kv);
            }
            if (snapshot.score) {
                ggml_build_forward_expand(gf, snapshot.score);
            }

            ggml_tensor * persist_kv = ggml_get_rows(ctx0, comp_kv, inp_comp.state_persist_src_idxs);
            ggml_tensor * persist_score = ggml_get_rows(ctx0, comp_score, inp_comp.state_persist_src_idxs);

            ggml_build_forward_expand(gf,
                    comp_state->cpy_kv(ctx0, persist_kv, inp_comp.state_persist_dst_idxs, il));
            ggml_build_forward_expand(gf,
                    comp_state->cpy_score(ctx0, persist_score, inp_comp.state_persist_dst_idxs, il));
        }
        // ratio 1 completes a group at every token, so there is never a partial group to carry.
    }

    // =========================================================================================
    // 1b. Nothing compressed is visible yet -- model.py:729-731, `compress_len == 0` short-
    //     circuits to an empty selection and model.py:776 then leaves `kv` as the window alone.
    //     The compressor above still ran, so its partial-group state keeps accumulating.
    //     This happens on every prefill shorter than `ratio` and on the first decode steps.
    // =========================================================================================
    if (inp_comp.kq_mask == nullptr) {
        return build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);
    }

    // =========================================================================================
    // 2. Index sources publish a selection; the layers after them reuse it (SPEC.md 5.4).
    // =========================================================================================
    if (hparams.dsv41_is_index_source[il]) {
        shared_topk = build_v41_lid_top_k(model, inp_idx, idx_ctx, qr, cur, inp_pos, idx_src_il, il);
    }

    ggml_tensor * top_k = shared_topk;
    GGML_ASSERT(top_k && "deepseek41: a compressed layer before the first index source");
    GGML_ASSERT(inp_idx.kq_mask == nullptr || inp_idx.kq_mask->ne[0] == inp_comp.kq_mask->ne[0]);

    // =========================================================================================
    // 3. One attention over [sliding window | selected compressed rows] (model.py:774-780).
    // =========================================================================================
    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q,  k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;
    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "v41_raw_k", il);

    // consumers read the source layer's plane, never their own (SPEC.md 4.4)
    ggml_tensor * comp_k = comp_ctx->get_k(ctx0, src_il);
    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= comp_k->ne[2]);

    comp_k = ggml_view_4d(ctx0, comp_k,
            comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
            comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
    cb(comp_k, "v41_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
    cb(k_all, "v41_k_all", il);

    ggml_tensor * raw_mask  = inp_attn->get_kq_mask();
    ggml_tensor * comp_mask = build_top_k_mask(inp_comp.kq_mask, top_k, "v41_top_k_mask", il);

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
    cb(kq_mask, "v41_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks,
            nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_v41", il);

    return out;
}

ggml_tensor * llama_model_deepseek41::graph::build_attention_v41(
        const llama_model * model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model->layers[il];

    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head/n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    // SPEC.md 7.2: compressed layers rotate at compress_rope_theta WITH YaRN; ratio-0 layers at
    // rope_theta with YaRN off. Identical rule to V4 (deepseek4.cpp:926-933); the one difference
    // is the attenuation factor, which is a literal 1.0 here.
    const bool    use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float   freq_base_l       = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float   freq_scale_l      = use_compress_rope ? freq_scale : 1.0f;
    const float   ext_factor_l      = use_compress_rope ? ext_factor : 0.0f;
    const float   beta_fast_l       = use_compress_rope ? beta_fast : 0.0f;
    const float   beta_slow_l       = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l      = use_compress_rope ? n_ctx_orig : 0;

    // --- query: wq_a -> q_norm(LoRA rank) -> wq_b -> RoPE (model.py:770-772) ------------------
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    // SPEC.md 3.1: NO query head norm. V4 does ggml_rms_norm(q) here (deepseek4.cpp:943);
    // model.py:770-772 and ds4.c:40700-40709 both go straight from wq_b to RoPE.
    cb(q, "q", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            ggml_row_size(q->type, n_embd_head_nope));
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, DSV41_ROPE_ATTN_FACTOR, beta_fast_l, beta_slow_l);
    cb(q_pe, "q_pe", il);
    q = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(q, "q_rope", il);

    // --- sliding-window latent: one shared K row per position (model.py:700-720) --------------
    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            0);
    ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head_nope));
    kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, DSV41_ROPE_ATTN_FACTOR, beta_fast_l, beta_slow_l);
    cb(kv_pe, "kv_pe", il);
    kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
    cb(kv, "kv", il);

    // --- attention -----------------------------------------------------------------------------
    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * out = build_compressed_attention(model, inp_dsv4, q, kv, qr, cur, inp_pos, kq_scale, il);
    if (out == nullptr) {
        // ratio 0 (layers 0 and 1): pure SWA-128 over this layer's own window (SPEC.md 3.2).
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);
    }

    // --- de-rotate the output (model.py:781, SPEC.md 3.4) --------------------------------------
    // The cache holds one shared rotated latent for all 64 heads, so the query's own rotation is
    // removed from the output before the grouped projection. Same parameters as the forward
    // rotation of this layer.
    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            0);
    ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            ggml_row_size(out->type, n_embd_head_nope));
    out_pe = ggml_rope_ext_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, DSV41_ROPE_ATTN_FACTOR, beta_fast_l, beta_slow_l);
    out = ggml_concat(ctx0, out_nope, out_pe, 0);
    cb(out, "attn_derope", il);

    // --- grouped low-rank output projection (model.py:785-788, SPEC.md 3.5) --------------------
    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llama_model_deepseek41::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const int64_t hc = hparams.dsv4_hc_mult;

    ggml_tensor * inp         = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    ggml_build_forward_expand(gf, inp_dsv4->get_raw()->self_kq_mask);

    // DS41-B8. Returns nullptr when this GGUF carries no Engram tables; build_engram is then the
    // identity on every layer.
    llm_graph_input_dsv41_engram * inp_engram = build_inp_dsv41_engram(model);

    // model.py:1258 -- broadcast the embedding into all hc streams
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // model.py:1260 -- the collapse coefficients entering layer 0 are one-hot on stream 0.
    // nullptr stands for that one-hot; every later layer carries its predecessor's ffn_pre.
    ggml_tensor * pre_mix = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // model.py:1262-1263 -- Engram is injected into the hc-expanded stream at the TOP of the
        // layer body, BEFORE the MTP target capture at model.py:1265-1266. Identity on a layer
        // with no Engram tables (DS41-B8).
        inpL = build_engram(model, inp_engram, inpL, il);

        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            // model.py:1265-1266 -- the MTP target layers read the MEAN of the streams
            res->t_layer_inp[il] = build_hc_mean(inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post     = nullptr;
        ggml_tensor * comb     = nullptr;

        // model.py:982 -- computed here, but the collapse below uses the INCOMING pre_mix
        ggml_tensor * attn_pre = build_hc_mixes(inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il);
        // ggml_cont for two reasons: the mixes are a strided view into a 24-wide row, and the
        // carried copy has to outlive the tensor it views so the allocator keeps it alive across
        // the sublayer (and, for ffn_pre, across the whole next layer).
        attn_pre = ggml_cont(ctx0, attn_pre);
        cb(attn_pre, "hc_attn_mix", il);

        ggml_tensor * cur = pre_mix ? build_hc_pre(inpL, pre_mix, il) : build_hc_stream0(inpL);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention_v41(&model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;

        // model.py:989 -- again computed before the sublayer that will consume it
        ggml_tensor * ffn_pre = build_hc_mixes(inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &post, &comb, il);
        ffn_pre = ggml_cont(ctx0, ffn_pre);
        cb(ffn_pre, "hc_ffn_mix", il);

        // model.py:990 -- the FFN collapses with attn_pre, computed BEFORE attention ran
        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // SPEC.md 8. V4.1 has no token-hash routed layers, so there is no tid2eid branch.
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);

        // model.py:994 -- carried to the next block's attention, and, after the last layer, to
        // the head collapse
        pre_mix = ffn_pre;
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = build_hc_mean(inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    GGML_ASSERT(pre_mix && "deepseek41 needs at least one layer to produce the head mix");

    ggml_tensor * flat     = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;

    if (cparams.embeddings_nextn) {
        ggml_tensor * h_nextn = cparams.embeddings_nextn_masked ? flat_out : inpL;
        cb(h_nextn, "h_nextn", -1);
        res->t_h_nextn = h_nextn;
    }

    if (inp_out_ids) {
        inpL    = ggml_reshape_3d(ctx0, flat_out, n_embd, hc, n_outputs);
        // the carried mix is per token, so it has to follow the same row selection
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
    }

    // model.py:1268 -- SPEC.md 2.2. There is no output_hc_* head in V4.1; the collapse reuses the
    // last block's FFN mix. il = -1 keeps this off the fused hc_pre path.
    ggml_tensor * cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
