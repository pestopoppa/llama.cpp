#include "models.h"

// DFlash block diffusion drafter graph builder.
//
// When cross-attention data is available (via llama_set_cross_data):
//   1. Apply fc.weight projection + hidden_norm to conditioning data
//   2. Compute K/V from both conditioning + noise tokens, concatenate
//   3. Compute Q from noise tokens only
//   4. Non-causal attention (all tokens attend to all K/V)
//
// When no cross-attention data: falls back to standard self-attention.

llm_build_dflash::llm_build_dflash(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // DFlash conditioning: project cross-attention data through fc + hidden_norm
    ggml_tensor * target_hidden = nullptr;
    if (cross && !cross->v_embd.empty() && model.dflash_fc) {
        const int64_t n_cross_embd = cross->n_embd;  // should be n_taps * n_embd_target
        const int64_t n_cross_tokens = cross->n_enc;

        // Create input tensor from cross data
        ggml_tensor * cross_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_cross_embd, n_cross_tokens);
        ggml_set_name(cross_inp, "dflash_cross_inp");
        ggml_set_input(cross_inp);

        // fc projection: [n_taps * n_embd, n_tokens] -> [n_embd, n_tokens]
        target_hidden = ggml_mul_mat(ctx0, model.dflash_fc, cross_inp);
        cb(target_hidden, "dflash_fc_out", -1);

        // hidden_norm: RMS normalization
        if (model.dflash_hidden_norm) {
            target_hidden = build_norm(target_hidden, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
            cb(target_hidden, "dflash_hidden_norm_out", -1);
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // attention: cross-attention when conditioning available, else self-attention
        {
            // Q always from noise hidden states
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur;
            ggml_tensor * Vcur;

            if (target_hidden) {
                // Cross-attention: K/V from concatenated [target_hidden; cur]
                ggml_tensor * Kcur_ctx   = build_lora_mm(model.layers[il].wk, target_hidden);
                ggml_tensor * Kcur_noise = build_lora_mm(model.layers[il].wk, cur);
                Kcur = ggml_concat(ctx0, Kcur_ctx, Kcur_noise, 1); // concat along token dim

                ggml_tensor * Vcur_ctx   = build_lora_mm(model.layers[il].wv, target_hidden);
                ggml_tensor * Vcur_noise = build_lora_mm(model.layers[il].wv, cur);
                Vcur = ggml_concat(ctx0, Vcur_ctx, Vcur_noise, 1); // concat along token dim
            } else {
                // Standard self-attention (fallback)
                Kcur = build_lora_mm(model.layers[il].wk, cur);
                Vcur = build_lora_mm(model.layers[il].wv, cur);
            }
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);

            const int64_t n_kv_tokens = Kcur->ne[1]; // n_tokens or n_ctx+n_tokens
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_kv_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_kv_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            // TODO: RoPE for K needs proper position handling for cross-attention
            // Context tokens: positions 0..n_ctx-1, noise tokens: n_ctx..n_ctx+n_noise-1
            // For now, apply RoPE with inp_pos (only correct for self-attention fallback)
            if (!target_hidden) {
                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );
            }
            // When target_hidden is set, skip RoPE on K for now (positions need special handling)
            // TODO: create concatenated position tensor for cross-attention K

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        // Handle last layer
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (standard Qwen3 SwiGLU)
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;

        // Capture layer output for hidden state extraction
        if (res->t_hidden_states.size() <= static_cast<size_t>(il)) {
            res->t_hidden_states.resize(il + 1, nullptr);
        }
        res->t_hidden_states[il] = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // Mark hidden state tensors as graph outputs to prevent buffer reuse
    for (auto * t_hs : res->t_hidden_states) {
        if (t_hs) {
            ggml_build_forward_expand(gf, t_hs);
        }
    }
}
