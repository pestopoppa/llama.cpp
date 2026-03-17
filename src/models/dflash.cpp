#include "models.h"

// DFlash block diffusion drafter graph builder.
//
// Two modes:
//   1. Self-attention (no cross data): standard Qwen3 with KV cache
//   2. Cross-attention (cross data set): fc conditioning + K/V concatenation,
//      direct MHA without KV cache, non-causal (fully permissive) mask

llm_build_dflash::llm_build_dflash(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const bool has_cross = cross && !cross->v_embd.empty() && model.dflash_fc;

    // Conditioning projection: cross data → fc → hidden_norm
    ggml_tensor * target_hidden = nullptr;
    ggml_tensor * pos_k = nullptr; // K position tensor for cross-attention RoPE
    int64_t n_ctx_tokens = 0;
    if (has_cross) {
        n_ctx_tokens = cross->n_enc;

        // Register cross data input handler
        auto inp_cross = std::make_unique<llm_graph_input_dflash_cross>(cross, n_tokens);
        inp_cross->cross_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cross->n_embd, n_ctx_tokens);
        ggml_set_name(inp_cross->cross_inp, "dflash_cross_inp");
        ggml_set_input(inp_cross->cross_inp);

        // K position tensor for RoPE: [0..n_ctx+n_noise-1]
        inp_cross->pos_k = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx_tokens + n_tokens);
        ggml_set_name(inp_cross->pos_k, "dflash_pos_k");
        ggml_set_input(inp_cross->pos_k);

        ggml_tensor * cross_inp = inp_cross->cross_inp;
        pos_k = inp_cross->pos_k;
        res->add_input(std::move(inp_cross));

        // fc projection: [n_taps * n_embd, n_tokens] → [n_embd, n_tokens]
        target_hidden = ggml_mul_mat(ctx0, model.dflash_fc, cross_inp);
        cb(target_hidden, "dflash_fc_out", -1);

        if (model.dflash_hidden_norm) {
            target_hidden = build_norm(target_hidden, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
            cb(target_hidden, "dflash_hidden_norm_out", -1);
        }
    }

    // Attention input: KV cache for self-attention, none for cross-attention
    llm_graph_input_attn_kv * inp_attn_kv = nullptr;
    if (!has_cross) {
        inp_attn_kv = build_attn_inp_kv();
    }

    // For cross-attention: non-causal mask (all zeros = fully permissive)
    // NULL mask = no masking applied in build_attn_mha
    ggml_tensor * cross_kq_mask = nullptr;
    // DFlash uses is_causal=False, so we pass nullptr for the mask

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur;
            ggml_tensor * Vcur;

            if (has_cross && target_hidden) {
                // Cross-attention: K/V from [target_hidden; cur]
                ggml_tensor * Kcur_ctx   = build_lora_mm(model.layers[il].wk, target_hidden);
                ggml_tensor * Kcur_noise = build_lora_mm(model.layers[il].wk, cur);
                Kcur = ggml_concat(ctx0, Kcur_ctx, Kcur_noise, 1);

                ggml_tensor * Vcur_ctx   = build_lora_mm(model.layers[il].wv, target_hidden);
                ggml_tensor * Vcur_noise = build_lora_mm(model.layers[il].wv, cur);
                Vcur = ggml_concat(ctx0, Vcur_ctx, Vcur_noise, 1);
            } else {
                Kcur = build_lora_mm(model.layers[il].wk, cur);
                Vcur = build_lora_mm(model.layers[il].wv, cur);
            }
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            const int64_t n_kv_tokens = Kcur->ne[1];
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_kv_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_kv_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);

            // RoPE on Q (noise positions)
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            if (has_cross && pos_k) {
                // Cross-attention: RoPE on K with full [0..n_ctx+n_noise-1] positions
                Kcur = ggml_rope_ext(ctx0, Kcur, pos_k, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            } else {
                // Self-attention: RoPE on K with same positions as Q
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (has_cross) {
                // Direct MHA with non-causal mask — no KV cache
                cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, cross_kq_mask, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il);
                cb(cur, "kqv_out", il);

                // O projection
                cur = build_lora_mm(model.layers[il].wo, cur);
            } else {
                cur = build_attn(inp_attn_kv,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il);
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;

        if (res->t_hidden_states.size() <= static_cast<size_t>(il)) {
            res->t_hidden_states.resize(il + 1, nullptr);
        }
        res->t_hidden_states[il] = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    for (auto * t_hs : res->t_hidden_states) {
        if (t_hs) {
            ggml_build_forward_expand(gf, t_hs);
        }
    }
}
