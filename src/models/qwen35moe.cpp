#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_qwen35moe::llm_build_qwen35moe(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    const int n_main_layers = n_layer - (int)hparams.nextn_predict_layers;

    // === MTP-only evaluation mode ===
    // Skip main transformer, only run MTP head on a single token.
    // Used for draft generation in speculation loop.
    if (gtype == LLM_GRAPH_TYPE_MTP_EVAL) {
        GGML_ASSERT(hparams.nextn_predict_layers > 0);
        GGML_ASSERT(mtp_hidden_valid);

        ggml_tensor * inpL = build_inp_embd(model.tok_embd);
        cb(inpL, "model.input_embed", -1);

        build_mtp_head(inpL, n_main_layers);
        return;
    }

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    // Save initial embedding for MTP
    ggml_tensor * mtp_embd = (hparams.nextn_predict_layers > 0) ? inpL : nullptr;

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Main transformer layers (exclude MTP layers)

    for (int il = 0; il < n_main_layers; ++il) {
        // Attention-only draft: skip recurrent layers entirely (pass input through)
        if (cparams.skip_recurrent && hparams.is_recurrent(il)) {
            continue;
        }

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_main_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // MOE FFN layer
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Save the last hidden state (pre-norm) for MTP
    ggml_tensor * main_hidden = cur;

    // Filter MTP embedding by output ids
    if (mtp_embd && inp_out_ids) {
        mtp_embd = ggml_get_rows(ctx0, mtp_embd, inp_out_ids);
    }

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // Export pre-norm hidden state for MTP caching (used by NEXT decode step)
    if (hparams.nextn_predict_layers > 0) {
        res->t_mtp_hidden_out = main_hidden;
        ggml_build_forward_expand(gf, main_hidden);
    }

    // === In-graph MTP-1 Forward Pass ===
    // Predicts the next token from CACHED hidden state (previous step) + current token embedding
    // Used for acceptance rate measurement; for speculation, use LLM_GRAPH_TYPE_MTP_EVAL instead
    // Skip when batch has multiple tokens (cached hidden is [n_embd, 1], can't broadcast)
    if (hparams.nextn_predict_layers > 0 && mtp_hidden_valid && n_outputs <= 1) {
        build_mtp_head(mtp_embd, n_main_layers);
    }
}

void llm_build_qwen35moe::build_mtp_head(ggml_tensor * mtp_embd, int n_main_layers) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int mtp_il = n_main_layers; // MTP layer index
    const auto & mtp_layer = model.layers[mtp_il];

    // Get cached hidden state from previous decode step
    auto * inp_mtp = build_inp_mtp_hidden();
    ggml_tensor * cached_hidden = inp_mtp->hidden_prev;

    // Step 1: Normalize embedding and hidden state, concatenate, project
    ggml_tensor * embd_normed = build_norm(mtp_embd,     mtp_layer.nextn.enorm, nullptr, LLM_NORM_RMS, mtp_il);
    cb(embd_normed, "mtp_enorm", mtp_il);

    ggml_tensor * h_normed    = build_norm(cached_hidden, mtp_layer.nextn.hnorm, nullptr, LLM_NORM_RMS, mtp_il);
    cb(h_normed, "mtp_hnorm", mtp_il);

    // Concatenate along embedding dimension: [2*n_embd, n_mtp_tokens]
    ggml_tensor * combined = ggml_concat(ctx0, embd_normed, h_normed, 0);
    cb(combined, "mtp_combined", mtp_il);

    // Project down: [2*n_embd, n_mtp_tokens] -> [n_embd, n_mtp_tokens]
    ggml_tensor * mtp_cur = build_lora_mm(mtp_layer.nextn.eh_proj, combined);
    cb(mtp_cur, "mtp_projected", mtp_il);

    const int64_t n_mtp_tokens = mtp_cur->ne[1];

    // Step 2: MTP transformer layer — full attention (gated Q) + MoE
    ggml_tensor * mtp_attn_in = build_norm(mtp_cur, mtp_layer.attn_norm, nullptr, LLM_NORM_RMS, mtp_il);
    cb(mtp_attn_in, "mtp_attn_norm", mtp_il);

    // Gated Q + K + V projections
    ggml_tensor * Qcur_full = build_lora_mm(mtp_layer.wq, mtp_attn_in);
    cb(Qcur_full, "mtp_Qcur_full", mtp_il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_mtp_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "mtp_Qcur", mtp_il);

    Qcur = build_norm(Qcur, mtp_layer.attn_q_norm, nullptr, LLM_NORM_RMS, mtp_il);
    cb(Qcur, "mtp_Qcur_normed", mtp_il);

    ggml_tensor * Kcur = build_lora_mm(mtp_layer.wk, mtp_attn_in);
    cb(Kcur, "mtp_Kcur", mtp_il);

    ggml_tensor * Vcur = build_lora_mm(mtp_layer.wv, mtp_attn_in);
    cb(Vcur, "mtp_Vcur", mtp_il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_mtp_tokens);
    Kcur = build_norm(Kcur, mtp_layer.attn_k_norm, nullptr, LLM_NORM_RMS, mtp_il);
    cb(Kcur, "mtp_Kcur_normed", mtp_il);

    // Gate (second half of Q projection)
    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_mtp_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_mtp_tokens);
    cb(gate, "mtp_gate", mtp_il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_mtp_tokens);

    // RoPE: skip for single-token MTP (Q·K^T is scalar, RoPE is identity)
    // Only apply during prefill when all positions are available
    if (n_mtp_tokens == n_tokens && gtype != LLM_GRAPH_TYPE_MTP_EVAL) {
        int sections[4];
        std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);
        ggml_tensor * inp_pos = build_inp_pos();

        Qcur = ggml_rope_multi(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        Kcur = ggml_rope_multi(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    }

    cb(Qcur, "mtp_Qcur_rope", mtp_il);
    cb(Kcur, "mtp_Kcur_rope", mtp_il);

    // No-cache attention (MTP layer has no KV cache)
    auto * mtp_attn_inp = build_attn_inp_no_cache();

    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f / sqrtf(float(n_embd_head))
        : hparams.f_attention_scale;

    ggml_tensor * mtp_attn_out = build_attn(mtp_attn_inp,
                nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, mtp_il);
    cb(mtp_attn_out, "mtp_attn_pregate", mtp_il);

    // Apply output gate
    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    mtp_attn_out = ggml_mul(ctx0, mtp_attn_out, gate_sigmoid);
    cb(mtp_attn_out, "mtp_attn_gated", mtp_il);

    // Output projection
    mtp_attn_out = build_lora_mm(mtp_layer.wo, mtp_attn_out);
    cb(mtp_attn_out, "mtp_attn_output", mtp_il);

    // Residual
    mtp_cur = ggml_add(ctx0, mtp_attn_out, mtp_cur);
    cb(mtp_cur, "mtp_attn_residual", mtp_il);

    // Post-attention norm + MoE FFN
    ggml_tensor * mtp_ffn_residual = mtp_cur;

    ggml_tensor * mtp_post_norm = build_norm(mtp_cur, mtp_layer.attn_post_norm, nullptr, LLM_NORM_RMS, mtp_il);
    cb(mtp_post_norm, "mtp_attn_post_norm", mtp_il);

    mtp_cur = build_layer_ffn(mtp_post_norm, mtp_il);
    cb(mtp_cur, "mtp_ffn_out", mtp_il);

    // FFN residual
    mtp_cur = ggml_add(ctx0, mtp_cur, mtp_ffn_residual);
    cb(mtp_cur, "mtp_post_moe", mtp_il);

    // Step 3: Final norm + shared LM head
    ggml_tensor * mtp_out_norm;
    if (mtp_layer.nextn.shared_head_norm) {
        mtp_out_norm = build_norm(mtp_cur, mtp_layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, -1);
    } else {
        mtp_out_norm = build_norm(mtp_cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    }
    cb(mtp_out_norm, "mtp_result_norm", -1);

    // Shared LM head (same as main model)
    ggml_tensor * mtp_logits = build_lora_mm(model.output, mtp_out_norm);
    cb(mtp_logits, "mtp_result_output", -1);

    // In MTP_EVAL mode, this IS the main output. In normal mode, it's auxiliary.
    if (gtype == LLM_GRAPH_TYPE_MTP_EVAL) {
        res->t_logits = mtp_logits;
    } else {
        res->t_logits_mtp = mtp_logits;
    }
    ggml_build_forward_expand(gf, mtp_logits);
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen35moe::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llm_build_qwen35moe::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_qwen35moe ::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llm_build_qwen35moe ::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur);
    alpha = ggml_cont_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    // Update convolution state cache
    // Extract the last (conv_kernel_size - 1) states from conv_input
    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
        ggml_view_1d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels * n_seqs,
                     kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    if (!cparams.freeze_recurrent) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));
    }

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        // TODO: try to avoid these explicit repeats by utilizing op broadcast
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // Choose between build_delta_net_chunking, build_delta_net_recurrent, and build_delta_net_autoregressive based on n_tokens
    std::pair<ggml_tensor *, ggml_tensor *> attn_out; // pair of (output, new_state)
    if (n_seq_tokens == 1) {
        attn_out = build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il);
    } else {
        attn_out = build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, il);
    }
    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    // Update the recurrent states (skip when freeze_recurrent is active)
    if (!cparams.freeze_recurrent) {
        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));
    }

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_build_qwen35moe ::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used, LLM_FFN_SILU,
            true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps);
    cb(moe_out, "ffn_moe_out", il);

    // Add shared experts if present - following Qwen3Next reference implementation
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, NULL,
                model.layers[il].ffn_gate_shexp, NULL, NULL,
                model.layers[il].ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // Apply shared expert gating as in the reference implementation
        // The shared expert has its own gate that is sigmoided
        // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        // Apply sigmoid to the gate
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);


        // Apply the gate to the shared expert output
        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}
