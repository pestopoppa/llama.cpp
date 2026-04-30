#include "models.h"

#include "llama-memory-recurrent.h"

#include <cmath>

// Ant Group Ring-mini-linear-2.0 / Ring-flash-linear-2.0 — hybrid Lightning Attention + softmax MoE.
//   - linear-attention layers use the existing GGML_OP_GATED_LINEAR_ATTN op with a
//     constant per-(layer, head) decay tensor (Lightning Attention = degenerate-`g` GLA).
//   - softmax-attention layers use standard build_attn() with partial NeoX RoPE.
//   - decay form (matches BailingMoeV2LinearAttention.build_slope_tensor):
//       slope_base       = 2^(-(2^-(log2(n_head) - 3)))                  // ALiBi-style base
//       slope_alibi[h]   = slope_base^(h+1)                              // per-head ALiBi slope
//       layer_factor[il] = 1 - (il - 1) / (n_layer - 1) + 1e-5           // per-layer scale
//       g[il, h]         = exp(-slope_alibi[h] * layer_factor[il])       // multiplicative decay

llm_build_ring_linear::llm_build_ring_linear(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {

    const int64_t n_embd_head = hparams.n_embd_head_k();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());

    const int64_t head_dim     = n_embd_head;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // ALiBi slope base for the per-head Lightning Attention decay.
    const float slope_base    = powf(2.0f, -powf(2.0f, -(log2f((float)n_head) - 3.0f)));
    const float ln_slope_base = logf(slope_base);

    // GroupRMSNorm groups for the post-attention output norm on linear layers.
    const uint32_t group_norm_size = hparams.n_norm_groups ? hparams.n_norm_groups : 4;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto * inp = build_inp_mem_hybrid();   // recurrent state for linear layers, KV cache for softmax layers

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf((float)head_dim);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        // Pre-attention RMSNorm
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            // ===== Lightning Attention path (linear) =====
            // Linear layers have num_kv_heads == num_attention_heads (no GQA on linear path),
            // so wqkv shape is [n_embd, 3*n_head*head_dim]. build_qkv with n_head_kv=n_head
            // produces Q/K/V views of shape [head_dim, n_head, n_tokens].

            auto qkv = build_qkv(layer, cur, head_dim, n_head, /*n_head_kv=*/n_head, il);
            ggml_tensor * Qcur = qkv.q;
            ggml_tensor * Kcur = qkv.k;
            ggml_tensor * Vcur = qkv.v;

            // QK norm (use_qk_norm=True for Ring-mini)
            Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

            // Partial NeoX RoPE on Q and K (n_rot < head_dim per partial_rotary_factor).
            // Lightning Attention is mathematically position-implicit, but the reference
            // BailingMoeV2LinearAttention.forward applies RoPE here, so we mirror it.
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            // The GLA op asserts contiguity on k/v/q/g/state — materialize the QKV views.
            Qcur = ggml_cont(ctx0, Qcur);
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);

            // Build the per-head decay tensor `g` for the GLA op:
            //   slope_alibi[h] = slope_base^(h+1)  =  slope_base * exp(h * ln(slope_base))
            //   final_g[h]     = exp(-slope_alibi[h] * layer_factor)
            const float layer_factor = 1.0f - float(il - 1) / float(n_layer - 1) + 1e-5f;
            ggml_tensor * h_idx        = ggml_arange(ctx0, 0.0f, (float)n_head, 1.0f);
            ggml_tensor * h_log_slope  = ggml_scale(ctx0, h_idx, ln_slope_base);
            ggml_tensor * slope_h      = ggml_exp(ctx0, h_log_slope);                          // slope_base^h
            ggml_tensor * neg_slope_h  = ggml_scale(ctx0, slope_h, -slope_base * layer_factor);// -slope_base^(h+1) * layer_factor
            ggml_tensor * decay_per_h  = ggml_exp(ctx0, neg_slope_h);                          // exp(-...)
            decay_per_h = ggml_reshape_3d(ctx0, decay_per_h, 1, n_head, 1);

            // Broadcast the per-head decay to [head_dim, n_head, n_tokens] — same shape as Q/K/V.
            ggml_tensor * decay_target = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head, n_tokens);
            ggml_tensor * decay_g      = ggml_repeat(ctx0, decay_per_h, decay_target);
            decay_g = ggml_cont(ctx0, decay_g);

            // Recurrent state input
            auto * inp_rs = inp->get_recr();
            const auto * mctx_rs = static_cast<const llama_memory_recurrent_context *>(inp_rs->mctx);
            const auto kv_head = mctx_rs->get_head();
            ggml_tensor * ssm_states_all = mctx_rs->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);

            // Call the GLA op (note: argument order is K, V, Q, g, state, scale).
            ggml_tensor * gla_out = ggml_gated_linear_attn(ctx0, Kcur, Vcur, Qcur, decay_g, state, kq_scale);

            // The GLA op concatenates token output and new state along the same buffer:
            //   [0, n_embd*n_tokens)              = token output
            //   [n_embd*n_tokens, ..)             = new state of size n_embd*head_dim*n_seqs
            ggml_tensor * o_flat   = ggml_view_1d(ctx0, gla_out, n_embd * n_tokens, 0);
            ggml_tensor * new_state = ggml_view_1d(ctx0, gla_out, n_embd * head_dim * n_seqs,
                                                   n_embd * n_tokens * sizeof(float));

            // Persist the new recurrent state into the per-layer state cache.
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                                 kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            // Reshape token output for the post-attn GroupRMSNorm + gate.
            // BailingMoeV2GroupRMSNorm splits the last axis (n_embd) into group_norm_size
            // groups of size n_embd/group_norm_size each, then RMS-normalizes each group
            // independently (NO mean subtraction). Implementation: reshape n_embd → (group_size, n_groups),
            // call ggml_rms_norm (which normalizes along ne[0] = group_size), reshape back.
            // NOTE: ggml_group_norm is LayerNorm (subtracts mean) and groups along ne[2] —
            // wrong on both axes; do not use here.
            const int64_t group_size = n_embd / group_norm_size;
            ggml_tensor * o_grouped = ggml_reshape_4d(ctx0, o_flat,
                                                     group_size, (int64_t) group_norm_size,
                                                     n_seq_tokens, n_seqs);
            ggml_tensor * o_norm    = ggml_rms_norm(ctx0, o_grouped, hparams.f_norm_rms_eps);
            ggml_tensor * o_normed  = ggml_reshape_3d(ctx0, o_norm, n_embd, n_seq_tokens, n_seqs);
            o_normed = ggml_mul(ctx0, o_normed, layer.attn_out_norm);

            // Sigmoid gate computed from the LAYER INPUT (not the attention output), per the
            // BailingMoeV2LinearAttention reference: `o = o * sigmoid(g_proj(hidden_states))`.
            ggml_tensor * gate_proj = build_lora_mm(layer.wqkv_gate, cur);
            gate_proj = ggml_reshape_3d(ctx0, gate_proj, n_embd, n_seq_tokens, n_seqs);
            ggml_tensor * gate_sig = ggml_sigmoid(ctx0, gate_proj);

            ggml_tensor * gated = ggml_mul(ctx0, o_normed, gate_sig);

            // Output projection
            gated = ggml_reshape_2d(ctx0, gated, n_embd, n_tokens);
            cur = build_lora_mm(layer.wo, gated);
            cb(cur, "lin_attn_out", il);

        } else {
            // ===== Standard softmax attention path =====
            auto qkv = build_qkv(layer, cur, head_dim, n_head, n_head_kv, il);
            ggml_tensor * Qcur = qkv.q;
            ggml_tensor * Kcur = qkv.k;
            ggml_tensor * Vcur = qkv.v;

            // QK norm
            Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

            // Partial NeoX RoPE
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp->get_attn(),
                             layer.wo, layer.wo_b, layer.wo_s,
                             Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "softmax_attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual
        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpSA);
        cb(sa_out, "sa_out", il);

        // Pre-FFN RMSNorm
        cur = build_norm(sa_out, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // FFN: dense for layers < n_layer_dense_lead, MoE otherwise (Bailing pattern).
        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                            layer.ffn_up,   NULL, NULL,
                            layer.ffn_gate, NULL, NULL,
                            layer.ffn_down, NULL, NULL,
                            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   NULL, NULL,
                layer.ffn_gate_shexp, NULL, NULL,
                layer.ffn_down_shexp, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, sa_out);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
