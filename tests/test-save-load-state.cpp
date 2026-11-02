#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <random>
#include <vector>

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch & get() { return batch; }
    const llama_batch & get() const { return batch; }
};

static llama_tokens generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    llama_tokens result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        LOG("%d ", next_token);
        result.push_back(next_token);

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), next_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static llama_tokens test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOG("\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return {};
    }

    LOG("\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static bool test_seq_rm_isolated(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOG("\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        llama_batch_ptr batch(n_tokens, 0, 1);
        for (size_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch.get(), tokens[i], i, { seq_id }, false);
        }

        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return false;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return false;
    }

    if (state_before != state_after) {
        LOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return false;
    }

    LOG("PASS\n");
    return true;
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate tokens
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 6: restore recurrent checkpoint into a different live sequence while
// the source advances.  A/A controls use the same token/chunk shape.  The
// stale-state mutation must exceed the observed A/A floor; otherwise the
// diagnostic is inconclusive, never a positive result.
static bool test_partial_restore_busy_source_one(
        struct llama_model * model, const struct common_params & params,
        const llama_tokens & tokens, uint32_t n_rs_seq) {
    if (tokens.size() < 6) {
        LOG_ERR("Test 6 INVALID: need at least six prompt tokens\n");
        return false;
    }
    auto p = common_context_params_to_llama(params);
    p.n_ctx = 512;
    p.n_seq_max = 3;
    p.kv_unified = true;
    p.n_rs_seq = n_rs_seq;
    auto ctx = llama_context_ptr{llama_init_from_model(model, p)};
    auto ref = llama_context_ptr{llama_init_from_model(model, p)};
    auto aa = llama_context_ptr{llama_init_from_model(model, p)};
    if (!ctx || !ref || !aa) return false;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int end = std::min<int>(32, int(tokens.size()) - 2);
    if (n_vocab <= 0 || end < 3) return false;

    auto decode_one = [&](llama_context * c, llama_seq_id id, int pos) {
        llama_batch_ptr batch(1, 0, 1);
        common_batch_add(batch.get(), tokens[pos], pos, {id}, true);
        return llama_decode(c, batch.get()) == 0;
    };
    auto logits = [&](llama_context * c) {
        const float * ptr = llama_get_logits_ith(c, -1);
        return ptr ? std::vector<float>(ptr, ptr + n_vocab) : std::vector<float>{};
    };
    auto max_diff = [&](const std::vector<float> & a, const std::vector<float> & b) {
        if (a.size() != size_t(n_vocab) || b.size() != a.size()) return INFINITY;
        float d = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
            d = std::max(d, std::fabs(a[i] - b[i]));
        }
        return d;
    };
    auto state = [&](llama_seq_id id, llama_state_seq_flags flags) {
        const size_t n = llama_state_seq_get_size_ext(ctx.get(), id, flags);
        std::vector<uint8_t> out(n);
        if (!n || llama_state_seq_get_data_ext(ctx.get(), out.data(), n, id, flags) != n) out.clear();
        return out;
    };
    for (int i = 0; i < end - 1; ++i) {
        if (!decode_one(ctx.get(), 0, i) || !decode_one(ref.get(), 0, i) || !decode_one(aa.get(), 0, i)) return false;
    }
    // Deliberate stale-state mutation: checkpoint before the final prefix token.
    const auto stale = state(0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (stale.empty()) return false;
    if (!decode_one(ctx.get(), 0, end - 1) || !decode_one(ref.get(), 0, end - 1) ||
        !decode_one(aa.get(), 0, end - 1)) return false;
    const auto partial = state(0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (partial.empty()) return false;
    if (!decode_one(ctx.get(), 0, end) || !decode_one(ref.get(), 0, end) ||
        !decode_one(aa.get(), 0, end)) return false;
    const auto source_before = state(0, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (source_before.empty()) return false;
    const auto lref = logits(ref.get());
    const auto floor = max_diff(lref, logits(aa.get()));
    if (!std::isfinite(floor)) return false;
    auto mem = llama_get_memory(ctx.get());
    auto restore = [&](llama_seq_id id, const std::vector<uint8_t> & bytes) {
        if (!llama_memory_seq_cp_ext(mem, 0, id, 0, end, LLAMA_MEMORY_SEQ_CP_FLAGS_ATTN_ONLY)) return false;
        return llama_state_seq_set_data_ext(ctx.get(), bytes.data(), bytes.size(), id,
                                          LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == bytes.size();
    };
    if (!restore(1, partial)) return false;
    if (llama_memory_seq_pos_max(mem, 0) != end || llama_memory_seq_pos_max(mem, 1) != end - 1 ||
        state(0, LLAMA_STATE_SEQ_FLAGS_NONE) != source_before) return false;
    if (!decode_one(ctx.get(), 1, end)) return false;
    const float actual = max_diff(lref, logits(ctx.get()));
    if (!std::isfinite(actual) || actual > floor || state(0, LLAMA_STATE_SEQ_FLAGS_NONE) != source_before) {
        LOG_ERR("Test 6 FAIL: restored mismatch=%g control=%g\n", (double) actual, (double) floor);
        return false;
    }
    // The same attention prefix but stale recurrent state must not look like the
    // control; refusal is also an acceptable detection of this invalid state.
    bool mutation_detected = !restore(2, stale);
    if (!mutation_detected) {
        mutation_detected = !decode_one(ctx.get(), 2, end);
        if (!mutation_detected) {
            const float changed = max_diff(lref, logits(ctx.get()));
            mutation_detected = std::isfinite(changed) && changed > floor;
        }
    }
    if (!mutation_detected || state(0, LLAMA_STATE_SEQ_FLAGS_NONE) != source_before) {
        LOG_ERR("Test 6 INVALID: stale-state mutation not detected or source changed\n");
        return false;
    }
    if (!decode_one(ctx.get(), 0, end + 1) || !decode_one(ref.get(), 0, end + 1) ||
        !decode_one(aa.get(), 0, end + 1)) return false;
    const auto lnext = logits(ref.get());
    const auto next_floor = max_diff(lnext, logits(aa.get()));
    const auto next_actual = max_diff(lnext, logits(ctx.get()));
    if (!std::isfinite(next_actual) || !std::isfinite(next_floor) || next_actual > next_floor) return false;
    LOG("Test 6 PASS n_rs_seq=%u restored=%g A/A=%g source=%g source_A/A=%g mutation=detected\n",
        n_rs_seq, (double) actual, (double) floor, (double) next_actual, (double) next_floor);
    return true;
}

static bool test_partial_restore_busy_source(
        struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    if (!llama_model_is_hybrid(model)) {
        LOG("Test 6 SKIP: hybrid model required\n");
        return true;
    }
    return test_partial_restore_busy_source_one(model, params, tokens, 0) &&
           test_partial_restore_busy_source_one(model, params, tokens, 8);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    if (result_baseline.empty()) {
        return 1;
    }

    // Test 2: sequence removal isolation
    if (!test_seq_rm_isolated(model, params, tokens)) {
        return 1;
    }

    // Test 3: state load
    if (!test_state_load(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 4: seq copy (host)
    if (!test_seq_cp_host(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 5: seq copy (device)
    if (!test_seq_cp_device(model, params, tokens, result_baseline)) {
        return 1;
    }

    if (!test_partial_restore_busy_source(model, params, tokens)) {
        LOG_ERR("Test 6: PARTIAL_ONLY busy-source restore failed\n");
        return 1;
    }

    LOG("\nAll tests passed.\n");

    return 0;
}
