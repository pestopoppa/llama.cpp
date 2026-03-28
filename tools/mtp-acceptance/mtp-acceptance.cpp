// MTP acceptance rate diagnostic
//
// Loads a model with MTP (nextn_predict_layers > 0), processes a prompt,
// then generates tokens while measuring how often the MTP-predicted
// next token matches the target model's actual output.
//
// Usage:
//   llama-mtp-acceptance -m model.gguf -p "prompt text" -n 128 -t 192

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static llama_token argmax(const float * logits, int n_vocab) {
    float max_val = logits[0];
    llama_token max_id = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_id = i;
        }
    }
    return max_id;
}

static void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nexample usage:\n");
    LOG("\n  %s -m model.gguf -p \"prompt text\" -n 128 -t 192\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 128;
    params.prompt = "The quick brown fox jumps over the lazy dog. In the realm of artificial intelligence,";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    common_init();

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize prompt
    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true);
    const int n_prompt = (int) tokens.size();

    LOG_INF("prompt tokens: %d\n", n_prompt);
    LOG_INF("generation tokens: %d\n", params.n_predict);
    LOG_INF("vocab size: %d\n", n_vocab);

    // Process prompt
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            common_batch_add(batch, tokens[i], i, { 0 }, i == n_prompt - 1);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("prompt decode failed\n");
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_batch_free(batch);
    }

    // Check if MTP logits are available
    const float * mtp_logits_check = llama_get_logits_mtp(ctx);
    if (!mtp_logits_check) {
        LOG_ERR("model has no MTP logits (nextn_predict_layers = 0?)\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    LOG_INF("MTP logits available - starting generation\n\n");

    // Generate tokens and measure acceptance
    int n_accepted = 0;
    int n_total = 0;
    int n_top5_match = 0;

    // Get first token from main logits
    const float * main_logits = llama_get_logits_ith(ctx, -1);
    llama_token cur_token = argmax(main_logits, n_vocab);

    // Get top-5 from logits
    auto get_top5 = [&](const float * lgt) -> std::vector<llama_token> {
        std::vector<llama_token> top5(5, -1);
        std::vector<float> top5_val(5, -1e30f);
        for (int i = 0; i < n_vocab; i++) {
            for (int j = 0; j < 5; j++) {
                if (lgt[i] > top5_val[j]) {
                    for (int k = 4; k > j; k--) {
                        top5[k] = top5[k-1];
                        top5_val[k] = top5_val[k-1];
                    }
                    top5[j] = i;
                    top5_val[j] = lgt[i];
                    break;
                }
            }
        }
        return top5;
    };

    llama_pos pos = n_prompt;

    // Escape newlines for display
    auto escape = [](const std::string & s) -> std::string {
        std::string r;
        for (char c : s) {
            if (c == '\n') r += "\\n";
            else if (c == '\r') r += "\\r";
            else if (c == '\t') r += "\\t";
            else r += c;
        }
        return r;
    };

    printf("%-6s %-20s %-20s %-20s %s\n", "Step", "Generated", "MTP Predicted", "Actual Next", "Match");
    printf("%-6s %-20s %-20s %-20s %s\n", "----", "---------", "-------------", "-----------", "-----");

    // With cached hidden state approach:
    // On step N, we decode cur_token (= T_N). MTP uses embed(T_N) + cached_hidden(N-1).
    // MTP(hidden_{N-1}, embed(T_N)) → predicts T_{N+1} (same as main model's output).
    // So we compare MTP prediction against actual_next on the SAME step.

    for (int i = 0; i < params.n_predict; i++) {
        // Decode cur_token
        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, cur_token, pos, { 0 }, true);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("decode failed at step %d\n", i);
            llama_batch_free(batch);
            break;
        }
        llama_batch_free(batch);
        pos++;

        // Get actual next token from main logits
        main_logits = llama_get_logits_ith(ctx, -1);
        llama_token actual_next = argmax(main_logits, n_vocab);

        // Get MTP prediction from THIS step (same-step comparison)
        const float * mtp_logits = llama_get_logits_mtp_ith(ctx, -1);
        llama_token mtp_predicted = mtp_logits ? argmax(mtp_logits, n_vocab) : -1;

        if (mtp_predicted >= 0) {
            n_total++;
            bool exact_match = (mtp_predicted == actual_next);
            bool top5_match = false;

            std::vector<llama_token> mtp_top5 = get_top5(mtp_logits);
            for (auto t : mtp_top5) {
                if (t == actual_next) { top5_match = true; break; }
            }

            if (exact_match) n_accepted++;
            if (top5_match)  n_top5_match++;

            std::string s_gen = common_token_to_piece(ctx, cur_token, false);
            std::string s_mtp = common_token_to_piece(ctx, mtp_predicted, false);
            std::string s_act = common_token_to_piece(ctx, actual_next, false);

            printf("%-6d %-20s %-20s %-20s %s\n",
                i,
                ("\"" + escape(s_gen) + "\"").c_str(),
                ("\"" + escape(s_mtp) + "\"").c_str(),
                ("\"" + escape(s_act) + "\"").c_str(),
                exact_match ? "YES" : (top5_match ? "top5" : "no"));

            // Print running stats every 20 tokens
            if (n_total > 0 && n_total % 20 == 0) {
                printf("--- running: %d/%d = %.1f%% exact, %d/%d = %.1f%% top5 ---\n",
                    n_accepted, n_total, 100.0f * n_accepted / n_total,
                    n_top5_match, n_total, 100.0f * n_top5_match / n_total);
            }
        } else {
            std::string s_gen = common_token_to_piece(ctx, cur_token, false);
            std::string s_act = common_token_to_piece(ctx, actual_next, false);
            printf("%-6d %-20s %-20s %-20s %s\n",
                i,
                ("\"" + escape(s_gen) + "\"").c_str(),
                "(no cache yet)",
                ("\"" + escape(s_act) + "\"").c_str(),
                "skip");
        }

        // Move to next token
        cur_token = actual_next;

        // Stop on EOS
        if (llama_vocab_is_eog(vocab, cur_token)) {
            LOG_INF("\nEOS reached at step %d\n", i);
            break;
        }
    }

    printf("\n=== MTP Acceptance Results ===\n");
    printf("Total tokens:    %d\n", n_total);
    printf("Exact match:     %d/%d = %.1f%%\n", n_accepted, n_total,
           n_total > 0 ? 100.0f * n_accepted / n_total : 0.0f);
    printf("Top-5 match:     %d/%d = %.1f%%\n", n_top5_match, n_total,
           n_total > 0 ? 100.0f * n_top5_match / n_total : 0.0f);

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
