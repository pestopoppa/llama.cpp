// Test-Time Training (TTT) implementation for llama.cpp
// Based on "End-to-End Test-Time Training for Long Context" (arXiv:2512.23675)
//
// TTT adapts model weights on context using gradient descent during inference.
// This enables continuous context compression without periodic compaction halts.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

// Custom param filter: only train FFN/MLP layers (per TTT-E2E paper)
// This reduces overhead by ~75% while maintaining adaptation quality
static bool param_filter_ffn(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(userdata);
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name[0] == '\0') {
        return false;  // Empty name
    }

    // Match FFN layer patterns across different model architectures
    // Qwen/Llama: ffn_gate, ffn_up, ffn_down, gate_proj, up_proj, down_proj
    // GPT-style: mlp, fc1, fc2
    return strstr(name, "ffn") != nullptr ||
           strstr(name, "mlp") != nullptr ||
           strstr(name, "gate_proj") != nullptr ||
           strstr(name, "up_proj") != nullptr ||
           strstr(name, "down_proj") != nullptr ||
           strstr(name, "fc1") != nullptr ||
           strstr(name, "fc2") != nullptr;
}

// Filter that trains all parameters (for comparison/debugging)
static bool param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(userdata);
    GGML_UNUSED(tensor);
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    // Default TTT parameters
    params.n_predict = 128;  // Generate 128 tokens after adaptation

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // Force settings required for training
    if (params.use_mmap) {
        LOG_INF("%s: disabling mmap (weights must be writable for TTT)\n", __func__);
        params.use_mmap = false;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: forcing F32 K cache (required for gradient computation)\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: forcing F32 V cache (required for gradient computation)\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Load model
    LOG_INF("%s: loading model from %s\n", __func__, params.model.path.c_str());
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // Print system info
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // Tokenize context (the text we'll adapt the model on)
    LOG_INF("%s: tokenizing context...\n", __func__);
    std::vector<llama_token> context_tokens = common_tokenize(ctx, params.prompt, true);
    LOG_INF("%s: context has %zu tokens\n", __func__, context_tokens.size());

    if (context_tokens.empty()) {
        LOG_ERR("%s: empty context - nothing to adapt on\n", __func__);
        return 1;
    }

    // The dataset API requires tokens.size() >= n_ctx + stride + 1 to create at least one data point
    // where stride = min(n_ctx/2, tokens.size())
    // For worst case: tokens.size() >= 1.5*n_ctx + 1
    const int32_t n_ctx = llama_n_ctx(ctx);
    const int32_t stride = std::min((int32_t)(n_ctx / 2), (int32_t)context_tokens.size());
    const int32_t min_tokens = n_ctx + stride + 1;

    if ((int32_t)context_tokens.size() < min_tokens) {
        // Calculate max n_ctx that would work with available tokens
        // tokens >= n_ctx + n_ctx/2 + 1 = 1.5*n_ctx + 1
        // n_ctx <= (tokens - 1) / 1.5
        const int32_t max_ctx = (int32_t)((context_tokens.size() - 1) / 1.5);
        LOG_ERR("%s: context has %zu tokens but needs at least %d for n_ctx=%d\n",
                __func__, context_tokens.size(), min_tokens, n_ctx);
        LOG_ERR("%s: either provide more context text, or use -c %d to reduce context size\n",
                __func__, max_ctx > 0 ? max_ctx : 8);
        return 1;
    }

    // Create dataset from context
    // Using half context size for chunk size (per finetune.cpp pattern)
    const int32_t chunk_size = stride;
    LOG_INF("%s: creating dataset with chunk_size=%d\n", __func__, chunk_size);

    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, context_tokens, chunk_size);
    const int64_t n_data = ggml_opt_dataset_ndata(dataset);
    LOG_INF("%s: dataset has %lld data points\n", __func__, (long long)n_data);

    // Configure optimizer
    struct lr_opt & lr = params.lr;

    // TTT-specific defaults if not set
    if (lr.lr0 <= 0.0f) {
        lr.lr0 = 1e-5f;  // Conservative learning rate for TTT
    }
    if (lr.epochs <= 0) {
        lr.epochs = 1;  // Single epoch for TTT (adapt once on context)
    }

    LOG_INF("%s: TTT configuration:\n", __func__);
    LOG_INF("  - optimizer: %s\n", ggml_opt_optimizer_name(params.optimizer));
    LOG_INF("  - learning rate: %.2g\n", (double)lr.lr0);
    LOG_INF("  - weight decay: %.2g\n", (double)lr.wd);
    LOG_INF("  - epochs: %d\n", (int)lr.epochs);
    LOG_INF("  - layer filter: FFN only\n");

    // Initialize TTT with FFN-only filter
    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ 0,
        /*param_filter    =*/ param_filter_ffn,  // Only FFN layers
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ common_opt_lr_pars,
        /*get_opt_pars_ud =*/ &params.lr,
        /*optimizer_type  =*/ params.optimizer,
    };

    LOG_INF("%s: initializing TTT optimizer...\n", __func__);
    llama_opt_init(ctx, model, lopt_params);

    // Run TTT adaptation
    LOG_INF("%s: running TTT adaptation...\n", __func__);

    ggml_opt_result_t result = ggml_opt_result_init();

    const auto t_start = std::chrono::high_resolution_clock::now();

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        LOG_INF("%s: epoch %d/%d\n", __func__, lr.epoch + 1, (int)lr.epochs);

        // Run single epoch on all data (no train/eval split for TTT)
        llama_opt_epoch(ctx, dataset, result, nullptr, n_data,
                        ggml_opt_epoch_callback_progress_bar, nullptr);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result);
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    const double t_adapt_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    LOG_INF("%s: TTT adaptation complete in %.2f ms\n", __func__, t_adapt_ms);
    LOG_INF("%s: adaptation speed: %.2f ms per 1K tokens\n", __func__,
            t_adapt_ms / (context_tokens.size() / 1000.0));

    ggml_opt_result_free(result);
    ggml_opt_dataset_free(dataset);

    // Optional: Save adapted model
    if (!params.out_file.empty()) {
        LOG_INF("%s: saving adapted model to %s\n", __func__, params.out_file.c_str());
        llama_model_save_to_file(model, params.out_file.c_str());
    }

    // Generate with adapted weights
    if (params.n_predict > 0) {
        LOG_INF("%s: generating %d tokens with adapted weights...\n", __func__, params.n_predict);

        // Initialize sampler
        auto * smpl = common_sampler_init(model, params.sampling);
        if (smpl == nullptr) {
            LOG_ERR("%s: failed to initialize sampler\n", __func__);
            llama_backend_free();
            return 1;
        }

        // Clear KV cache for generation
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_clear(mem, true);

        // Use last part of context as prompt for generation
        const int n_prompt = std::min((int)context_tokens.size(), (int)(llama_n_ctx(ctx) / 4));
        std::vector<llama_token> prompt_tokens(
            context_tokens.end() - n_prompt,
            context_tokens.end()
        );

        // Encode prompt
        llama_batch batch = llama_batch_init(prompt_tokens.size(), 0, 1);
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            common_batch_add(batch, prompt_tokens[i], i, { 0 }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: failed to decode prompt\n", __func__);
            llama_batch_free(batch);
            common_sampler_free(smpl);
            llama_backend_free();
            return 1;
        }

        // Generate tokens
        LOG_INF("\n--- Generated output ---\n");

        int n_cur = batch.n_tokens;
        for (int i = 0; i < params.n_predict; i++) {
            llama_token new_token = common_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(llama_model_get_vocab(model), new_token)) {
                LOG_INF("\n[end of generation]\n");
                break;
            }

            // Print token
            char buf[256];
            int n = llama_token_to_piece(llama_model_get_vocab(model), new_token, buf, sizeof(buf), 0, true);
            if (n > 0) {
                printf("%.*s", n, buf);
                fflush(stdout);
            }

            // Prepare next batch
            common_batch_clear(batch);
            common_batch_add(batch, new_token, n_cur++, { 0 }, true);

            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: failed to decode\n", __func__);
                break;
            }

            common_sampler_accept(smpl, new_token, true);
        }
        printf("\n");

        llama_batch_free(batch);
        common_sampler_free(smpl);
    }

    // Print final stats
    LOG_INF("\n");
    LOG_INF("%s: TTT stats:\n", __func__);
    LOG_INF("  - context tokens: %zu\n", context_tokens.size());
    LOG_INF("  - adaptation time: %.2f ms\n", t_adapt_ms);
    LOG_INF("  - ms per 1K tokens: %.2f\n", t_adapt_ms / (context_tokens.size() / 1000.0));

    llama_backend_free();

    return 0;
}
