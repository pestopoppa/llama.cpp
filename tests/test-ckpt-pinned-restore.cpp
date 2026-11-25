// CKPT-COPY-1: context-checkpoint save -> greedy generate -> restore from the checkpoint -> greedy generate again.
// The checkpoint is taken exactly as the server's create_checkpoint takes it (llama_state_seq_get_data_ext with
// LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY), the restore exactly as the server's rollback (llama_state_seq_set_data_ext, then
// seq_rm of the generated tail). PASS = both greedy continuations are token-identical. Run once with LLAMA_CKPT_PINNED=1
// and once without: the printed blob hash must match across the two runs (byte-identical checkpoint content).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <cinttypes>
#include <cstdio>
#include <vector>

static uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static bool gen(llama_context * ctx, llama_sampler * smpl, llama_token first, int pos, int n, llama_tokens & out) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    llama_token tok = first;
    for (int i = 0; i < n; ++i) {
        out.push_back(tok);
        common_batch_clear(batch);
        common_batch_add(batch, tok, pos + i, {0}, true);
        if (llama_decode(ctx, batch)) {
            llama_batch_free(batch);
            return false;
        }
        tok = llama_sampler_sample(smpl, ctx, -1);
    }
    llama_batch_free(batch);
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.n_predict < 0) {
        params.n_predict = 64;
    }
    common_init();
    ggml_backend_load_all();
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    llama_tokens prompt = common_tokenize(ctx, params.prompt.empty() ? std::string("The quick brown fox") : params.prompt, true);
    const int n_prompt = (int) prompt.size();
    llama_batch batch = llama_batch_init(std::max<int>(n_prompt, params.n_batch), 0, 1);
    for (int i0 = 0; i0 < n_prompt; i0 += params.n_batch) {
        common_batch_clear(batch);
        for (int i = i0; i < std::min(n_prompt, i0 + params.n_batch); ++i) {
            common_batch_add(batch, prompt[i], i, {0}, i == n_prompt - 1);
        }
        if (llama_decode(ctx, batch)) {
            LOG_ERR("prompt decode failed\n");
            return 1;
        }
    }
    llama_batch_free(batch);

    auto smpl = llama_sampler_ptr{llama_sampler_init_greedy()};
    const llama_token first = llama_sampler_sample(smpl.get(), ctx, -1);

    const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    int64_t t0 = ggml_time_us();
    std::vector<uint8_t> ckpt(llama_state_seq_get_size_ext(ctx, 0, fl));
    const size_t nw = llama_state_seq_get_data_ext(ctx, ckpt.data(), ckpt.size(), 0, fl);
    const int64_t save_us = ggml_time_us() - t0;
    if (nw != ckpt.size() || nw == 0) {
        LOG_ERR("checkpoint save failed (%zu of %zu)\n", nw, ckpt.size());
        return 1;
    }

    llama_tokens a, b;
    if (!gen(ctx, smpl.get(), first, n_prompt, params.n_predict, a)) {
        LOG_ERR("generate A failed\n");
        return 1;
    }
    // restore first, then drop the generated tail, in the server's order (rollback: load_tgt, then seq_rm [n_past, end))
    t0 = ggml_time_us();
    const size_t nr = llama_state_seq_set_data_ext(ctx, ckpt.data(), ckpt.size(), 0, fl);
    const int64_t restore_us = ggml_time_us() - t0;
    if (nr != ckpt.size()) {
        LOG_ERR("checkpoint restore failed (%zu of %zu)\n", nr, ckpt.size());
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt, -1)) {
        LOG_ERR("seq_rm failed\n");
        return 1;
    }
    llama_sampler_reset(smpl.get());
    if (!gen(ctx, smpl.get(), first, n_prompt, params.n_predict, b)) {
        LOG_ERR("generate B failed\n");
        return 1;
    }
    const bool same = a == b;
    uint64_t th = 1469598103934665603ULL;
    for (auto t : a) {
        th = (th ^ (uint64_t) (uint32_t) t) * 1099511628211ULL;
    }
    printf("CKPT_RESTORE_TEST n_prompt=%d n_predict=%d ckpt_bytes=%zu ckpt_fnv=%016" PRIx64 " tok_fnv=%016" PRIx64
           " save_us=%lld restore_us=%lld identical=%d %s\n",
           n_prompt, params.n_predict, ckpt.size(), fnv1a(ckpt.data(), ckpt.size()), th, (long long) save_us,
           (long long) restore_us, same ? 1 : 0, same ? "PASS" : "FAIL");
    fflush(stdout);
    return same ? 0 : 2;
}
