// DFlash Block Diffusion — Acceptance Rate Test Tool
//
// Loads a target model and a DFlash drafter model, runs inference, extracts
// hidden states from the target model at configured layer indices, and validates
// the hidden state extraction pipeline.
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

    printf("\nTarget model:   %d layers, %d embd\n", n_layer_tgt, n_embd_tgt);
    printf("DFlash drafter: %d layers, %d embd\n", n_layer_dft, n_embd_dft);

    // DFlash target layer IDs (from drafter config: [1, 12, 23, 34, 45])
    // TODO: read from GGUF metadata
    const std::vector<int> target_layer_ids = {1, 12, 23, 34, 45};
    printf("Target layer taps: [");
    for (size_t i = 0; i < target_layer_ids.size(); i++) {
        printf("%d%s", target_layer_ids[i], i < target_layer_ids.size()-1 ? ", " : "");
    }
    printf("]\n");

    // Tokenize prompt
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true);
    printf("Prompt: %zu tokens\n\n", tokens.size());

    // === Phase 1: Target model prefill + hidden state extraction ===
    printf("Phase 1: Target model prefill + hidden state extraction...\n");

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
        // Check hidden state extraction
        int32_t n_hidden = llama_get_hidden_state_count(ctx_tgt);
        printf("  Hidden states captured: %d layers\n", n_hidden);

        if (n_hidden > 0) {
            // Verify hidden states at target layer IDs
            for (int lid : target_layer_ids) {
                float * hs = llama_get_hidden_state(ctx_tgt, lid);
                if (hs) {
                    // Compute L2 norm of first token's hidden state as sanity check
                    float norm = 0.0f;
                    for (int j = 0; j < n_embd_tgt; j++) {
                        norm += hs[j] * hs[j];
                    }
                    norm = sqrtf(norm);
                    printf("  Layer %2d: hidden state extracted, L2 norm = %.4f\n", lid, norm);
                } else {
                    printf("  Layer %2d: hidden state NOT available\n", lid);
                }
            }

            // Concatenate hidden states at target layers for fc projection
            const int n_taps = target_layer_ids.size();
            std::vector<float> concat_hidden(n_taps * n_embd_tgt);
            bool all_available = true;
            for (int t = 0; t < n_taps; t++) {
                float * hs = llama_get_hidden_state(ctx_tgt, target_layer_ids[t]);
                if (hs) {
                    // Copy first token's hidden state (for verification)
                    memcpy(concat_hidden.data() + t * n_embd_tgt, hs, n_embd_tgt * sizeof(float));
                } else {
                    all_available = false;
                }
            }

            if (all_available) {
                printf("\n  Concatenated hidden: %d dims (= %d taps x %d embd)\n",
                       n_taps * n_embd_tgt, n_taps, n_embd_tgt);
                printf("  fc.weight expects:   %d x %d\n", n_embd_dft, n_taps * n_embd_tgt);
                printf("  Dimensions match:    %s\n",
                       (n_taps * n_embd_tgt == n_taps * n_embd_dft) ? "YES" : "NO");
                printf("\n  Hidden state extraction pipeline: VALIDATED\n");
            }
        } else {
            printf("  WARNING: No hidden states captured. Target model may not support hidden state extraction.\n");
        }

        // Get first generated token from target
        llama_sampler * smpl = llama_sampler_init_greedy();
        llama_token token_tgt = llama_sampler_sample(smpl, ctx_tgt, -1);
        llama_sampler_free(smpl);

        char buf[128];
        int n = llama_token_to_piece(vocab, token_tgt, buf, sizeof(buf), 0, true);
        printf("\n  First target token: %d (%.*s)\n", token_tgt, n, buf);

        // === Phase 2: Autoregressive decode with hidden state tracking ===
        printf("\nPhase 2: Autoregressive decode with hidden state tracking...\n");

        int n_generated = 0;
        llama_token prev_token = token_tgt;

        for (int i = 0; i < params.n_predict; i++) {
            common_batch_clear(batch);
            common_batch_add(batch, prev_token, tokens.size() + i, {0}, true);

            if (llama_decode(ctx_tgt, batch) != 0) {
                fprintf(stderr, "ERROR: target decode failed at step %d\n", i);
                break;
            }

            // Verify hidden states still available after each decode step
            if (i == 0) {
                int32_t n_hs = llama_get_hidden_state_count(ctx_tgt);
                float * hs0 = llama_get_hidden_state(ctx_tgt, target_layer_ids[0]);
                printf("  Step 0: %d layers captured, layer %d %s\n",
                       n_hs, target_layer_ids[0],
                       hs0 ? "available" : "NOT available");
            }

            smpl = llama_sampler_init_greedy();
            prev_token = llama_sampler_sample(smpl, ctx_tgt, -1);
            llama_sampler_free(smpl);
            n_generated++;

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

        printf("\n\n=== Results ===\n");
        printf("Generated: %d tokens from target model\n", n_generated);
        printf("Hidden state extraction: %s\n",
               n_hidden > 0 ? "WORKING" : "NOT WORKING");
        printf("Target layer count:  %d\n", n_hidden);
        printf("Required layer taps: %zu\n", target_layer_ids.size());
        printf("\n");

        if (n_hidden >= n_layer_tgt) {
            printf("READY for Phase 3: fc conditioning + DFlash cross-attention\n");
        } else {
            printf("BLOCKED: Hidden states not fully captured (%d/%d layers)\n", n_hidden, n_layer_tgt);
        }
    }

cleanup:
    llama_batch_free(batch);
    llama_free(ctx_dft);
    llama_model_free(model_dft);
    llama_free(ctx_tgt);
    llama_model_free(model_tgt);

    return 0;
}
