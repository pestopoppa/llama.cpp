#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include "gguf.h"

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
    GGML_UNUSED(params);

    // Deliberately NOT falling through to the V4 graph. V4 dispatches on compression ratio 4
    // (CSA) / 128 (HCA); with V4.1's ratios of 1 and 2 every layer would take the raw-attention
    // path, the shared compressor/indexer state would never be built, and the Engram tables loaded
    // above would never be read -- a silently wrong model rather than a failure. The graph deltas
    // are DS41-B7 (Engram FP8-row op), DS41-B11 and DS41-B12.
    GGML_ABORT("deepseek41: the V4.1 graph is not implemented yet (DS41-B7/B11/B12); this build "
               "loads the model but cannot decode with it");
}
