#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <cmath>

//
// DeepSeek-V4.1-Flash DSpark drafter (INF-77 DS41-B13).
//
// Normative: DeepSeek-V4.1-Flash/inference/model.py
//   :1020-1029  get_dspark_topk_idxs   -- the in-block mask
//   :1031-1076  DSparkAttention        -- compress_ratio == 0, window ring fed by main_kv
//   :1100-1135  DSparkBlock            -- a plain Block with DSparkAttention, plus main_proj/norm
//   :1137-1156  forward_head           -- one head call for all block positions, then the
//                                         sequential Markov-bias chain and the confidence head
//   :1275-1282  forward_spec           -- the whole draft pass
//
// Shape of this graph, mirroring llama_model_dflash::graph_dsv4 (src/models/dflash.cpp):
//
//   * an EMBD ubatch is the ring write. The driver has already pushed the target's
//     hc-mean-of-attention-input at layers 37/38/39 through the ENCODER graph, which is
//     main_norm(main_proj(main_hidden)) (model.py:1130). Here each stage projects that main_x
//     through its own wkv, rotates the trailing rope dims, and stores it in the window cache --
//     model.py:1039-1041 and :1065. One row per TARGET token, at that token's own position, so
//     the "catch up the ring after accepting several tokens" problem the reference would have
//     (:1065 hardcodes seqlen == 1) does not arise here.
//
//   * a TOKEN ubatch is the draft block: [id_last, NOISE x (B-1)] at positions P+1..P+B
//     (model.py:1131-1132). It runs the 3 DSpark stages and the shared head. The Markov chain
//     and the confidence head are build_dspark_markov_head() in dflash.cpp, which already
//     matches model.py:1145-1155 (bias from output_ids[:,i], conf from the PRE-norm collapsed
//     hidden) and is reused verbatim.
//
// The four deltas against graph_dsv4 (which is V4's DSpark) are spelled out in DESIGN.md
// section 3 and flagged inline below: LAGGED hyper-connection mix, head collapse without
// output_hc_*, no query head-norm, and the drafts' KV kept out of the ring.
//

// model.py:1058 -- softmax_scale = head_dim ** -0.5, and V4.1 applies no YaRN magnitude
// attenuation (deepseek41.cpp DSV41_ROPE_ATTN_FACTOR).
static constexpr float DSPARK41_ROPE_ATTN_FACTOR = 1.0f;

ggml_tensor * llama_model_dflash::graph_dsv41::build_dspark_kv(
        const llama_model & model,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    // model.py:1039-1040 / :1060-1061 -- kv_norm(wkv(x)), rope on the trailing rope dims only.
    // Every DSpark stage has compress_ratio 0 (asserted at model.py:1034), so this is the plain
    // rope_theta path with YaRN off, exactly like a ratio-0 backbone layer.
    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "dspark_kv_norm", il);

    ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            0);
    ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head_nope));
    kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
            freq_base, 1.0f, 0.0f, DSPARK41_ROPE_ATTN_FACTOR, 0.0f, 0.0f);
    cb(kv_pe, "dspark_kv_pe", il);

    kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
    cb(kv, "dspark_kv", il);

    return kv;
}

// Not used by the current mask strategy -- see the note on bidirectionality in the block loop --
// but kept because the exact-mask variant in DESIGN.md section 3.4 needs precisely this block.
ggml_tensor * llama_model_dflash::graph_dsv41::build_dspark_block_mask(int64_t n_block, ggml_type type) const {
    GGML_ASSERT(n_block > 0);

    // model.py:1027 -- `window_size + torch.arange(block_size)` is handed to EVERY draft
    // position, so the in-block half of the mask is all-visible: zeros, no causal triangle.
    ggml_tensor * mask = ggml_new_tensor_4d(ctx0, type, n_block, n_block, 1, 1);
    mask = ggml_scale(ctx0, mask, 0.0f);

    return mask;
}

llama_model_dflash::graph_dsv41::graph_dsv41(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    //
    // mode 0: ENCODER -- main_norm(main_proj(main_hidden)), model.py:1130.
    //
    // This has to be an encoder graph and not a wider embd branch of the decoder, because the
    // batch allocator is sized at hparams.n_embd_inp_enc() ONLY on the encode path
    // (llama-context.cpp:1709); llama_context::decode sizes it at n_embd_inp()
    // (:2020, = n_embd = 5120 here). Feeding a 15360-wide embd batch to llama_decode therefore
    // does not merely trip the width assert in llm_graph_input_embd::set_input -- the ubatch copy
    // upstream of it (llama-batch.cpp:778) would stride by 5120 and hand the graph a silently
    // wrong reinterpretation of the buffer. The assert was the loud symptom; this is the bug.
    //
    if (params.gtype == LLM_GRAPH_TYPE_ENCODER) {
        GGML_ASSERT(model.dspark_main_proj && model.dspark_main_norm &&
                "deepseek41-dspark encoder needs blk.0.dspark_main_proj / dspark_main_norm");

        const int64_t n_embd_inp = hparams.n_embd_inp_enc();

        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp->embd);

        // llm_graph_input_embd::set_input asserts its constructor width against this tensor
        // (llama-graph.cpp:74), so tie the two together here rather than trusting they match.
        GGML_ASSERT(inp->embd->ne[0] == n_embd_inp);

        ggml_tensor * cur = inp->embd;
        cb(cur, "inp_main_hidden", -1);

        res->add_input(std::move(inp));

        cur = build_lora_mm(model.dspark_main_proj, cur);
        cb(cur, "dspark_main_proj", -1);

        cur = build_norm(cur, model.dspark_main_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "dspark_main_norm", -1);

        GGML_ASSERT(cur->ne[0] == n_embd && "main_proj must produce the draft hidden width");

        // read back with llama_get_embeddings_nextn(); hparams.n_embd_out() is n_embd here, which
        // is the width llama_context::encode copies (llama-context.cpp:1844).
        ggml_set_output(cur);
        res->t_embd    = cur;
        res->t_h_nextn = cur;

        ggml_build_forward_expand(gf, cur);
        return;
    }

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head/n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t hc               = hparams.dsv4_hc_mult;

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);
    GGML_ASSERT(hc > 0);

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    //
    // mode 1: EMBD ubatch -- write the window ring (model.py:1039-1041, :1065)
    //
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        // The driver hands in main_x at the DRAFT hidden width -- the encoder graph above has
        // already applied main_proj/main_norm. This batch goes through llama_decode, whose batch
        // allocator is sized at n_embd_inp() (llama-context.cpp:2020), so any other width here is
        // silently mis-strided by llama-batch.cpp:778 before the graph ever sees it.
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        GGML_ASSERT(inp->embd->ne[0] == n_embd && "ring-write embd width must be the draft n_embd");

        ggml_tensor * main_x = inp->embd;
        cb(main_x, "inp_g_embeddings", -1);

        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * kv = build_dspark_kv(model, main_x, inp_pos, il);

            if (inp_attn->self_k_rot_swa) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa);
            }

            ggml_build_forward_expand(gf,
                    inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = main_x;

        ggml_build_forward_expand(gf, main_x);
        return;
    }

    //
    // mode 2: TOKEN ubatch -- the draft block
    //

    // model.py:1119-1120, :1276 -- embed and head are the BACKBONE's, shared through ctx_other.
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DSpark decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    // model.py:1134 -- x.unsqueeze(2).repeat(1, 1, hc_mult, 1)
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // DELTA 1 (DESIGN.md 3.1). model.py:1277 make_identity_pre_mix: the mix entering stage 0 is
    // one-hot on stream 0, and each sublayer's mix is consumed by the NEXT one. nullptr stands
    // for the one-hot; V4's graph_dsv4 applies each mix eagerly, which is a different model.
    ggml_tensor * pre_mix = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post     = nullptr;
        ggml_tensor * comb     = nullptr;

        ggml_tensor * attn_pre = build_hc_mixes(inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il);
        // the mixes are a strided view into one projection and the carried copy has to outlive
        // the sublayer that produced it -- same reasoning as deepseek41.cpp's layer loop
        attn_pre = ggml_cont(ctx0, attn_pre);
        cb(attn_pre, "hc_attn_mix", il);

        ggml_tensor * cur;
        if (pre_mix) {
            cur = build_hc_pre(inpL, pre_mix, il);
        } else {
            // one-hot on stream 0 degenerates to picking that stream (deepseek41.cpp
            // build_hc_stream0); exact, not an approximation
            GGML_ASSERT(inpL->ne[0] == n_embd && inpL->ne[1] == hc);
            cur = ggml_view_2d(ctx0, inpL, n_embd, inpL->ne[2], inpL->nb[2], 0);
        }
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        //
        // --- attention (model.py:1051-1076) --------------------------------------------------
        //
        {
            const int64_t nt = cur->ne[1];

            // DELTA 3 (DESIGN.md 3.3). model.py:1056-1057 goes wq_a -> q_norm (over the LoRA
            // rank) -> wq_b -> RoPE. There is NO per-head RMS norm after wq_b; V4's
            // build_attention_impl inserts one (deepseek4.cpp:957), which is why this attention
            // is written here instead of calling it.
            ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
            cb(qr, "qr", il);

            qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "qr_norm", il);

            ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
            q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
            cb(q, "q", il);

            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, nt,
                    ggml_row_size(q->type, n_embd_head),
                    ggml_row_size(q->type, n_embd_head)*n_head,
                    0);
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, nt,
                    ggml_row_size(q->type, n_embd_head),
                    ggml_row_size(q->type, n_embd_head)*n_head,
                    ggml_row_size(q->type, n_embd_head_nope));
            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, DSPARK41_ROPE_ATTN_FACTOR, 0.0f, 0.0f);
            cb(q_pe, "q_pe", il);
            q = ggml_concat(ctx0, q_nope, q_pe, 0);
            cb(q, "q_rope", il);

            ggml_tensor * kv = build_dspark_kv(model, cur, inp_pos, il);

            if (inp_attn->self_k_rot_swa) {
                q  = llama_mul_mat_hadamard(ctx0, q,  inp_attn->self_k_rot_swa);
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa);
            }

            ggml_build_forward_expand(gf, q);
            ggml_build_forward_expand(gf, kv);

            // DELTA 4 (DESIGN.md 3.4). model.py:1066-1067 CONCATENATES the block's own KV onto
            // the window rather than storing it, and get_dspark_topk_idxs (:1027) hands every
            // draft position all `block_size` block columns -- bidirectional, no causal triangle.
            //
            // Here the block's KV does go into the cache, and bidirectionality comes from the
            // driver's llama_set_causal_attn(ctx_dft, false): with causal off, set_input_kq_mask
            // keeps every in-sequence cell inside the SWA window, so each draft position sees the
            // whole ring AND all block positions, including strictly later ones. The rows are
            // then un-committed by the driver (llama_memory_seq_rm at the end of draft()), which
            // is what keeps the ring equal to the reference's "main_kv only" invariant.
            //
            // One measured divergence, recorded in NOTES.md: the reference gives every draft
            // position the FULL filled window, while the SWA mask cuts at a fixed distance, so
            // the draft at block offset j loses the j+1 oldest of 128 slots (at most 5/128 = 3.9%
            // for B = 5, always the oldest). It can cost acceptance, never correctness -- the
            // target verifies every token regardless.
            ggml_build_forward_expand(gf,
                    inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));

            ggml_tensor * k       = inp_attn->mctx->get_swa()->get_k(ctx0, il);
            ggml_tensor * kq_mask = inp_attn->get_kq_mask_swa();

            const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

            ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask, layer.attn_sinks,
                    nullptr, kq_scale, il);

            if (inp_attn->self_k_rot_swa) {
                out = llama_mul_mat_hadamard(ctx0, out, inp_attn->self_k_rot_swa);
            }
            cb(out, "attn_raw", il);

            // model.py:1071 -- apply_rotary_emb(o[..., -rd:], freqs_cis, True): the cache holds
            // one shared rotated latent for all heads, so the query's own rotation comes back off
            // the output before the grouped projection.
            out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
            ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, nt,
                    ggml_row_size(out->type, n_embd_head),
                    ggml_row_size(out->type, n_embd_head)*n_head,
                    0);
            ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, nt,
                    ggml_row_size(out->type, n_embd_head),
                    ggml_row_size(out->type, n_embd_head)*n_head,
                    ggml_row_size(out->type, n_embd_head_nope));
            out_pe = ggml_rope_ext_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, DSPARK41_ROPE_ATTN_FACTOR, 0.0f, 0.0f);
            out = ggml_concat(ctx0, out_nope, out_pe, 0);
            cb(out, "attn_derope", il);

            // model.py:1073-1076 -- wo_a is block-diagonal over o_groups, then wo_b
            out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
            out = ggml_permute(ctx0, out, 0, 2, 1, 3);
            ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
            cb(oa, "attn_wo_a", il);
            oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
            oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

            cur = build_lora_mm(layer.wo_b, oa);
            cb(cur, "attn_out", il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;

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

        // model.py:1104 -- the DSpark stage's FFN is a MoE with its own expert count
        // (dspark_n_routed_experts / dspark_n_activated_experts), which the converter writes as
        // this model's expert_count / expert_used_count.
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
        cb(inpL, "l_out", il);

        pre_mix = ffn_pre;
    }

    // DELTA 2 (DESIGN.md 3.2). model.py:1143 -- forward_head collapses with the CARRIED mix.
    // The DSpark checkpoint ships no output_hc_*, so V4's build_hc_head() is not available and
    // would be the wrong operator anyway. il = -1 keeps this off the fused hc_pre path.
    GGML_ASSERT(pre_mix && "DSpark v41 needs at least one stage to produce the head mix");

    ggml_tensor * cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_head", -1);

    // model.py:1154 -- the confidence head scores the PRE-norm collapsed hidden state.
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DSpark decoder requires the target model's output projection");
        output = model_other->output;
    }

    cur = build_lora_mm(output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // model.py:1145-1155 -- the sequential Markov-bias chain and the confidence head. Shared
    // verbatim with the V4 DSpark graph; see build_dspark_markov_head in dflash.cpp.
    GGML_ASSERT(model.dspark_markov_w1 && "DSpark v41 draft is missing markov_w1");
    llama_model_dflash_build_dspark_markov_head(*this, model, inp_tokens);
}
