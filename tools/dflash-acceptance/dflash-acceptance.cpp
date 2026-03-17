// DFlash Block Diffusion — Acceptance Rate Test Tool
//
// Loads a target model and a DFlash drafter model, runs inference, and provides
// a framework for measuring DFlash draft acceptance rate.
//
// Usage:
//   llama-dflash-acceptance \
//     -m <target_model.gguf> \
//     -md <dflash_drafter.gguf> \
//     -p "prompt text" \
//     -n <n_tokens>

#include "common.h"
#include "arg.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 64;
    params.prompt = "Write a Python function to compute fibonacci numbers:";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    // Initialize backends
    ggml_backend_load_all();

    printf("DFlash Acceptance Rate Test\n");
    printf("===========================\n\n");

    // Load target model
    printf("Loading target model: %s\n", params.model.path.c_str());

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model_tgt = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model_tgt) {
        fprintf(stderr, "ERROR: failed to load target model\n");
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    llama_context * ctx_tgt = llama_init_from_model(model_tgt, ctx_params);
    if (!ctx_tgt) {
        fprintf(stderr, "ERROR: failed to create target context\n");
        llama_model_free(model_tgt);
        return 1;
    }

    // Load DFlash drafter model
    printf("Loading DFlash drafter: %s\n", params.speculative.mparams_dft.path.c_str());

    llama_model * model_dft = llama_model_load_from_file(
        params.speculative.mparams_dft.path.c_str(), model_params);
    if (!model_dft) {
        fprintf(stderr, "ERROR: failed to load DFlash drafter model\n");
        fprintf(stderr, "       (use -md <path> to specify the DFlash drafter GGUF)\n");
        llama_free(ctx_tgt);
        llama_model_free(model_tgt);
        return 1;
    }

    llama_context * ctx_dft = llama_init_from_model(model_dft, ctx_params);
    if (!ctx_dft) {
        fprintf(stderr, "ERROR: failed to create DFlash context\n");
        llama_model_free(model_dft);
        llama_free(ctx_tgt);
        llama_model_free(model_tgt);
        return 1;
    }

    const int n_layer_tgt = llama_model_n_layer(model_tgt);
    const int n_embd_tgt  = llama_model_n_embd(model_tgt);
    const int n_layer_dft = llama_model_n_layer(model_dft);
    const int n_embd_dft  = llama_model_n_embd(model_dft);

    printf("\nTarget model:  %d layers, %d embd\n", n_layer_tgt, n_embd_tgt);
    printf("DFlash drafter: %d layers, %d embd\n", n_layer_dft, n_embd_dft);

    // Tokenize prompt
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true);
    printf("Prompt: %zu tokens\n\n", tokens.size());

    // === Phase 1: Target model prefill ===
    printf("Phase 1: Target model prefill...\n");

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, {0}, i == tokens.size() - 1);
    }

    if (llama_decode(ctx_tgt, batch) != 0) {
        fprintf(stderr, "ERROR: target model prefill failed\n");
        llama_batch_free(batch);
        goto cleanup;
    }

    {
        // Get first generated token from target
        llama_sampler * smpl = llama_sampler_init_greedy();
        llama_token token_tgt = llama_sampler_sample(smpl, ctx_tgt, -1);
        llama_sampler_free(smpl);

        char buf[128];
        int n = llama_token_to_piece(vocab, token_tgt, buf, sizeof(buf), 0, true);
        printf("  First target token: %d (%.*s)\n", token_tgt, n, buf);

        // === Phase 2: Autoregressive decode loop ===
        printf("\nPhase 2: Autoregressive decode...\n");

        int n_generated = 0;
        llama_token prev_token = token_tgt;

        for (int i = 0; i < params.n_predict; i++) {
            common_batch_clear(batch);
            common_batch_add(batch, prev_token, tokens.size() + i, {0}, true);

            if (llama_decode(ctx_tgt, batch) != 0) {
                fprintf(stderr, "ERROR: target decode failed at step %d\n", i);
                break;
            }

            smpl = llama_sampler_init_greedy();
            prev_token = llama_sampler_sample(smpl, ctx_tgt, -1);
            llama_sampler_free(smpl);
            n_generated++;

            // Print token
            n = llama_token_to_piece(vocab, prev_token, buf, sizeof(buf), 0, true);
            if (n > 0) {
                printf("%.*s", n, buf);
                fflush(stdout);
            }

            if (llama_vocab_is_eog(vocab, prev_token)) {
                printf("\n[EOS]");
                break;
            }
        }

        printf("\n\n");
        printf("=== Results ===\n");
        printf("Generated: %d tokens (target only, no DFlash drafting yet)\n", n_generated);
        printf("\n");
        printf("=== TODO: DFlash Integration ===\n");
        printf("1. Extract hidden states from target at layers [1, 12, 23, 34, 45]\n");
        printf("   Using t_hidden_states[] from Qwen3 graph builder\n");
        printf("2. Concatenate and project through fc (%d x %d) + hidden_norm\n",
               n_embd_dft, 5 * n_embd_tgt);
        printf("3. Run DFlash drafter with conditioning\n");
        printf("4. Compare draft tokens vs target tokens\n");
        printf("5. Measure acceptance rate (paper target: tau=6.49)\n");
    }

cleanup:
    llama_batch_free(batch);
    llama_free(ctx_dft);
    llama_model_free(model_dft);
    llama_free(ctx_tgt);
    llama_model_free(model_tgt);

    return 0;
}
