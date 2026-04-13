// test-am-beta-injection.cpp
// Tests that per-token attention bias (beta) from Attention Matching
// affects generation output when injected into the KV cache.
//
// Usage: ./test-am-beta-injection -m <model.gguf> [-t threads]
//
// Test plan:
// 1. Fill KV cache with a prompt
// 2. Generate N tokens (baseline)
// 3. Reset, fill same prompt
// 4. Set large positive beta on early tokens (bias attention toward them)
// 5. Generate N tokens (biased)
// 6. Compare: outputs should differ, proving beta injection works

#include "llama.h"
#include "common.h"
#include "arg.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    params.prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\nThe capital of France is<|im_end|>\n"
                    "<|im_start|>assistant\n";
    params.n_predict = 32;
    params.n_ctx = 512;
    params.cpuparams.n_threads = 48;
    params.warmup = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (!model || !ctx) {
        fprintf(stderr, "Failed to init model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Tokenize prompt
    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true);
    const int n_prompt = tokens.size();
    printf("Prompt tokens: %d\n", n_prompt);

    auto generate = [&](const char * label) -> std::string {
        // Decode prompt
        llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Failed to decode prompt\n");
            return "";
        }

        // Generate
        std::string result;
        llama_token token = -1;
        auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        for (int i = 0; i < params.n_predict; i++) {
            token = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, token)) break;

            std::string piece = common_token_to_piece(vocab, token);
            result += piece;

            llama_batch next = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, next)) break;
        }

        llama_sampler_free(smpl);
        printf("[%s] Generated: %s\n", label, result.c_str());
        return result;
    };

    // Test 1: Baseline generation
    printf("\n=== Test 1: Baseline (all beta = 0) ===\n");
    std::string baseline = generate("baseline");

    // Test 2: Set large positive beta on the first few tokens
    printf("\n=== Test 2: Beta injection (beta = 5.0 on first 5 tokens) ===\n");
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, false); // clear metadata, keep data buffers

    // Re-decode prompt to refill KV
    llama_batch batch2 = llama_batch_get_one(tokens.data(), n_prompt);
    if (llama_decode(ctx, batch2)) {
        fprintf(stderr, "Failed to re-decode prompt\n");
        return 1;
    }

    // Inject large beta on first 5 positions to heavily bias attention toward them
    int n_set = 0;
    for (int p = 0; p < std::min(5, n_prompt); p++) {
        if (llama_memory_set_beta(mem, 0, p, 5.0f)) {
            n_set++;
        }
    }
    printf("Set beta=5.0 on %d positions\n", n_set);

    std::string biased = generate("biased");

    // Test 3: Negative beta (suppress early tokens)
    printf("\n=== Test 3: Negative beta (beta = -5.0 on first 5 tokens) ===\n");
    llama_memory_clear(mem, false);
    llama_batch batch3 = llama_batch_get_one(tokens.data(), n_prompt);
    if (llama_decode(ctx, batch3)) {
        fprintf(stderr, "Failed to re-decode prompt\n");
        return 1;
    }
    for (int p = 0; p < std::min(5, n_prompt); p++) {
        llama_memory_set_beta(mem, 0, p, -5.0f);
    }
    std::string suppressed = generate("suppressed");

    // Results
    printf("\n=== Results ===\n");
    printf("Baseline:   %s\n", baseline.c_str());
    printf("Biased(+5): %s\n", biased.c_str());
    printf("Suppressed(-5): %s\n", suppressed.c_str());

    bool outputs_differ = (baseline != biased) || (baseline != suppressed);
    printf("\nOutputs differ: %s\n", outputs_differ ? "YES (beta injection working)" : "NO (beta may not be affecting attention)");

    return outputs_differ ? 0 : 1;
}
