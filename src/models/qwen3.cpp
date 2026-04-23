#include "models.h"
#include <cstdio>
#include <cstring>
#include <vector>

// TIDE: projection matrix for early exit (loaded once, used in graph)
// Non-static so qwen35moe.cpp and other models can extern-reference them
bool              tide_proj_loaded = false;
std::vector<float> tide_proj_data;
int               tide_proj_exit_layer = 0;
int               tide_proj_n_embd = 0;

// Load projection from .npy file (raw float32, n_embd×n_embd matrix)
__attribute__((constructor))
static void tide_try_load_projection() {
    const char * path = getenv("TIDE_PROJECTION_PATH");
    const char * layer_str = getenv("TIDE_EXIT_LAYER");
    const char * embd_str = getenv("TIDE_N_EMBD");
    if (!path || !layer_str || !embd_str) return;

    int n_embd = atoi(embd_str);
    int exit_layer = atoi(layer_str);
    if (n_embd <= 0 || exit_layer <= 0) return;

    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "TIDE: cannot open %s\n", path); return; }

    // Skip .npy header (find first \n after MAGIC)
    // Simple: seek to data start. .npy v1 header is 128 bytes typically.
    // For raw binary files, no header to skip.
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    long expected = (long)n_embd * n_embd * sizeof(float);

    // Check if it's a .npy file (starts with \x93NUMPY)
    fseek(f, 0, SEEK_SET);
    char magic[6];
    if (fread(magic, 1, 6, f) == 6 && magic[0] == (char)0x93 && memcmp(magic+1, "NUMPY", 5) == 0) {
        // .npy format: skip header
        fseek(f, 0, SEEK_SET);
        unsigned char header[10];
        fread(header, 1, 10, f);
        uint16_t header_len = *(uint16_t*)(header + 8);
        fseek(f, 10 + header_len, SEEK_SET);
    } else {
        // Raw binary
        fseek(f, 0, SEEK_SET);
    }

    tide_proj_data.resize(n_embd * n_embd);
    size_t read = fread(tide_proj_data.data(), sizeof(float), n_embd * n_embd, f);
    fclose(f);

    if ((int)read != n_embd * n_embd) {
        fprintf(stderr, "TIDE: read %zu floats, expected %d\n", read, n_embd * n_embd);
        tide_proj_data.clear();
        return;
    }

    tide_proj_exit_layer = exit_layer;
    tide_proj_n_embd = n_embd;
    tide_proj_loaded = true;
    fprintf(stderr, "TIDE: loaded projection for layer %d (%dx%d, %.1f MB)\n",
            exit_layer, n_embd, n_embd, (float)(n_embd * n_embd * 4) / 1e6);
}

llm_build_qwen3::llm_build_qwen3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // TIDE early exit
    const int64_t n_layer_exit = cparams.n_layer_exit;
    const int64_t n_layer_eff = (n_layer_exit > 0 && n_layer_exit < n_layer) ? n_layer_exit : n_layer;

    for (int il = 0; il < n_layer_eff; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            if (model.layers[il].wo_s) {
                cur = ggml_mul(ctx0, cur, model.layers[il].wo_s);
            }
        }
        if (il == n_layer_eff - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    // TIDE: if early exit is active and projection matrix is loaded,
    // use projection instead of output_norm (projection maps h_L → result_norm space)
    if (n_layer_exit > 0 && tide_proj_loaded && tide_proj_exit_layer == n_layer_exit) {
        // Load projection as ggml tensor and apply: cur = cur @ projection
        ggml_tensor * proj = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                tide_proj_n_embd, tide_proj_n_embd);
        ggml_set_name(proj, "tide_projection");
        memcpy(proj->data, tide_proj_data.data(), tide_proj_data.size() * sizeof(float));

        cur = ggml_mul_mat(ctx0, proj, cur);
        cb(cur, "tide_projected", -1);
    } else {
        // Normal path: output norm
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
