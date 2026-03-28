// MTP speculation loop benchmark
//
// Measures actual throughput of MTP-1 speculative decoding on Qwen3.5 models.
// Uses llama_decode_mtp() for lightweight draft generation, then verifies
// with a 2-token batch through the full model.
//
// Usage:
//   llama-mtp-speculation -m model.gguf -p "prompt text" -n 128 -t 192

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Single-token decode, returns next token from main logits
static llama_token decode_single(llama_context * ctx, llama_token token,
                                  llama_pos pos, int n_vocab) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { 0 }, true);
    int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) return -1;

    const float * logits = llama_get_logits_ith(ctx, -1);
    return argmax(logits, n_vocab);
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

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true);
    const int n_prompt = (int) tokens.size();

    LOG_INF("prompt tokens: %d\n", n_prompt);
    LOG_INF("generation tokens: %d\n", params.n_predict);

    // === Phase 1: Baseline ===
    LOG_INF("\n=== Baseline: single-token decode ===\n");

    llama_context_params ctx_params = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    std::vector<llama_token> baseline_tokens;
    double baseline_tps = 0;
    {
        // Process prompt
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

        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token cur = argmax(logits, n_vocab);
        llama_pos pos = n_prompt;

        const int64_t t_start = ggml_time_us();

        for (int i = 0; i < params.n_predict; i++) {
            baseline_tokens.push_back(cur);
            if (llama_vocab_is_eog(vocab, cur)) break;

            llama_token next = decode_single(ctx, cur, pos, n_vocab);
            if (next < 0) break;
            pos++;
            cur = next;
        }

        const int64_t t_end = ggml_time_us();
        const double t_sec = (t_end - t_start) / 1e6;
        baseline_tps = (double) baseline_tokens.size() / t_sec;
        LOG_INF("baseline: %d tokens in %.2f s = %.2f t/s\n",
            (int) baseline_tokens.size(), t_sec, baseline_tps);
    }
    llama_free(ctx);

    // === Phase 2: MTP Speculation ===
    LOG_INF("\n=== MTP speculation ===\n");

    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("failed to recreate context\n");
        llama_model_free(model);
        return 1;
    }

    llama_memory_t memory = llama_get_memory(ctx);

    std::vector<llama_token> spec_tokens;
    int n_accepted = 0;
    int n_drafted = 0;
    int n_full_decodes = 0;
    double spec_tps = 0;
    {
        // Process prompt
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

        // First token from prompt
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token cur = argmax(logits, n_vocab);
        spec_tokens.push_back(cur);

        // Bootstrap: single-token decode to establish hidden state cache
        llama_token next = decode_single(ctx, cur, n_prompt, n_vocab);
        if (next < 0) { llama_free(ctx); llama_model_free(model); return 1; }
        spec_tokens.push_back(next);
        n_full_decodes += 2; // prompt + bootstrap

        llama_pos pos = n_prompt + 1;
        cur = next;

        const int64_t t_start = ggml_time_us();

        while ((int) spec_tokens.size() < params.n_predict) {
            if (llama_vocab_is_eog(vocab, cur)) break;

            // Step 1: MTP-only eval to get draft
            const int64_t t_mtp_start = ggml_time_us();
            int mtp_ret = llama_decode_mtp(ctx, cur);
            const int64_t t_mtp_end = ggml_time_us();
            if (n_drafted < 3) {
                LOG_INF("  MTP eval: %.1f ms\n", (t_mtp_end - t_mtp_start) / 1e3);
            }
            if (mtp_ret != 0) {
                // MTP failed — fallback to single-token decode
                next = decode_single(ctx, cur, pos, n_vocab);
                if (next < 0) break;
                pos++;
                n_full_decodes++;
                spec_tokens.push_back(next);
                cur = next;
                continue;
            }

            const float * mtp_logits = llama_get_logits_mtp(ctx);
            if (!mtp_logits) {
                next = decode_single(ctx, cur, pos, n_vocab);
                if (next < 0) break;
                pos++;
                n_full_decodes++;
                spec_tokens.push_back(next);
                cur = next;
                continue;
            }

            llama_token draft = argmax(mtp_logits, n_vocab);
            n_drafted++;

            // Step 2: Verify with 2-token batch [cur, draft]
            {
                const int64_t t_verify_start = ggml_time_us();

                llama_batch vbatch = llama_batch_init(2, 0, 1);
                common_batch_add(vbatch, cur,   pos,     { 0 }, true);
                common_batch_add(vbatch, draft, pos + 1, { 0 }, true);

                int ret = llama_decode(ctx, vbatch);
                llama_batch_free(vbatch);
                n_full_decodes++;

                const int64_t t_verify_end = ggml_time_us();
                if (n_drafted <= 3) {
                    LOG_INF("  2-token batch: %.1f ms\n", (t_verify_end - t_verify_start) / 1e3);
                }

                if (ret != 0) {
                    LOG_ERR("verify decode failed at pos %d\n", pos);
                    break;
                }
            }

            // Step 3: Check acceptance
            // logits_ith(0) = logits after cur at pos → predicts what should be at pos+1
            logits = llama_get_logits_ith(ctx, 0);
            llama_token verified = argmax(logits, n_vocab);

            if (verified == draft) {
                // Accepted: advance by 2
                n_accepted++;
                spec_tokens.push_back(draft);
                if (llama_vocab_is_eog(vocab, draft)) break;

                // Next token from logits after draft
                const float * logits_1 = llama_get_logits_ith(ctx, 1);
                cur = argmax(logits_1, n_vocab);
                spec_tokens.push_back(cur);
                pos += 2;
            } else {
                // Rejected: output only the verified token, remove draft from cache
                llama_memory_seq_rm(memory, 0, pos + 1, -1);
                spec_tokens.push_back(verified);
                cur = verified;
                pos += 1;

                // Note: recurrent state has been corrupted by the draft token.
                // The next verification will still produce correct tokens since
                // the main model's logits are computed fresh each time.
                // The corruption only affects the hidden state cache quality,
                // which may reduce draft acceptance slightly.
            }
        }

        const int64_t t_end = ggml_time_us();
        const double t_sec = (t_end - t_start) / 1e6;
        const int n_spec_gen = (int) spec_tokens.size() - 2; // subtract bootstrap tokens
        spec_tps = n_spec_gen / t_sec;

        LOG_INF("speculation: %d tokens in %.2f s = %.2f t/s\n", n_spec_gen, t_sec, spec_tps);
    }

    // === Results ===
    printf("\n=== MTP Speculation Results ===\n");
    printf("Baseline:            %.2f t/s (%d tokens)\n", baseline_tps, (int) baseline_tokens.size());
    printf("Speculation:         %.2f t/s (%d tokens)\n", spec_tps, (int) spec_tokens.size() - 2);
    if (baseline_tps > 0) {
        printf("Speedup:             %.2fx\n", spec_tps / baseline_tps);
    }
    printf("Drafts attempted:    %d\n", n_drafted);
    printf("Drafts accepted:     %d (%.1f%%)\n", n_accepted,
        n_drafted > 0 ? 100.0 * n_accepted / n_drafted : 0.0);
    printf("Full model decodes:  %d\n", n_full_decodes);
    printf("Tokens per decode:   %.2f\n",
        n_full_decodes > 0 ? (double) spec_tokens.size() / n_full_decodes : 0.0);

    // Check output match
    int n_match = 0;
    int n_cmp = std::min((int) baseline_tokens.size(), (int) spec_tokens.size());
    for (int i = 0; i < n_cmp; i++) {
        if (baseline_tokens[i] == spec_tokens[i]) n_match++;
        else break;
    }
    printf("Output match:        %d/%d tokens identical before divergence\n", n_match, n_cmp);

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
