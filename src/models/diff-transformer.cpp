#include "models.h"

// Differential Transformer V2 (Microsoft, 2026)
//
// Key difference from standard Llama: Q heads are doubled (2h), K/V unchanged.
// After a single attention call, even/odd head outputs are split and the
// differential is computed: output = attn_even - sigmoid(lambda) * attn_odd.
// Lambda is a per-token, per-head learned scalar via W_lambda projection.
//
// Reference: arXiv:2410.05258, Microsoft Diff Attn V2 blog (2026)

llm_build_diff_transformer::llm_build_diff_transformer(
        const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // pre-attention RMSNorm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // save norm output for lambda computation
        ggml_tensor * attn_norm_out = cur;

        // --- Differential Attention V2 ---
        {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // Q projection: output is (2*n_head*d_head, n_tokens) — DOUBLED
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            // Reshape: Q to (d_head, 2*n_head, n_tokens), K/V standard
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, 2 * n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv,  n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv,  n_tokens);

            // Apply RoPE to all 2*n_head Q heads and n_head_kv K heads
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Use build_attn with wo=nullptr to get attention output without output projection.
            // build_attn handles KV cache store/retrieve and mask application internally.
            ggml_tensor * attn_full = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,  // wo=NULL, wo_b=NULL, wo_s=NULL — we apply wo after differential
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(attn_full, "attn_full", il);

            // attn_full shape: (d_head * 2*n_head, n_tokens) as 2D
            // Reshape to 3D: (d_head, 2*n_head, n_tokens)
            attn_full = ggml_reshape_3d(ctx0, attn_full, n_embd_head, 2 * n_head, n_tokens);

            // Split even/odd heads via strided views
            const size_t elem_size     = ggml_type_size(attn_full->type);
            const size_t head_bytes    = n_embd_head * elem_size;
            const size_t two_head_nb1  = 2 * head_bytes;              // stride: skip every other head
            const size_t token_nb2     = 2 * n_head * head_bytes;     // stride: all heads per token

            // Even heads (indices 0, 2, 4, ...)
            ggml_tensor * attn_even = ggml_view_3d(ctx0, attn_full,
                    n_embd_head, n_head, n_tokens,
                    two_head_nb1, token_nb2, 0);
            attn_even = ggml_cont(ctx0, attn_even);
            cb(attn_even, "attn_even", il);

            // Odd heads (indices 1, 3, 5, ...)
            ggml_tensor * attn_odd = ggml_view_3d(ctx0, attn_full,
                    n_embd_head, n_head, n_tokens,
                    two_head_nb1, token_nb2, head_bytes);
            attn_odd = ggml_cont(ctx0, attn_odd);
            cb(attn_odd, "attn_odd", il);

            // Compute lambda = sigmoid(attn_norm_out @ W_lambda) -> (n_head, n_tokens)
            ggml_tensor * lambda_val = build_lora_mm(model.layers[il].w_lambda, attn_norm_out);
            lambda_val = ggml_sigmoid(ctx0, lambda_val);
            cb(lambda_val, "lambda", il);

            // Reshape lambda to (1, n_head, n_tokens) for broadcasting across d_head
            lambda_val = ggml_reshape_3d(ctx0, lambda_val, 1, n_head, n_tokens);

            // Differential: output = attn_even - sigmoid(lambda) * attn_odd
            ggml_tensor * scaled_odd = ggml_mul(ctx0, attn_odd, lambda_val);
            cb(scaled_odd, "scaled_odd", il);

            cur = ggml_sub(ctx0, attn_even, scaled_odd);
            cb(cur, "diff_out", il);

            // Flatten to 2D and apply output projection
            cur = ggml_cont_2d(ctx0, cur, n_embd_head * n_head, n_tokens);

            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_out", il);
        }

        // Handle last layer early exit
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FFN (standard SwiGLU, identical to Llama)
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

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
