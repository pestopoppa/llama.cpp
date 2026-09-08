#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// Public-API observable-equivalence gate for stateful target verification
// rollback. Native NextN dispatch is proven separately by the server gate.

static int env_int(const char * name, int fallback, int minimum) {
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char * end = nullptr; errno = 0; long value = std::strtol(raw, &end, 10);
    if (errno || !end || *end || value < minimum || value > 1000000) {
        std::fprintf(stderr, "invalid %s=%s\n", name, raw); std::exit(2);
    }
    return (int) value;
}

static std::string json_quote(const std::string & value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { out.push_back('\\'); out.push_back((char) c); }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) { char buf[7]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
        else out.push_back((char) c);
    }
    return out + "\"";
}

static double env_real(const char * name, double fallback) {
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char * end = nullptr; errno = 0; double value = std::strtod(raw, &end);
    if (errno || !end || *end || !std::isfinite(value) || value < 0) {
        std::fprintf(stderr, "invalid %s=%s\n", name, raw); std::exit(2);
    }
    return value;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t batch) {
    auto cp = common_context_params_to_llama(params);
    cp.n_seq_max = 1;
    cp.n_rs_seq = std::max<uint32_t>(8, batch + 1);
    cp.n_batch = std::max(cp.n_batch, batch + 1);
    cp.n_ubatch = std::max(cp.n_ubatch, batch + 1);
    return llama_init_from_model(model, cp);
}

static bool decode_range(llama_context * ctx, const std::vector<llama_token> & toks,
                         uint32_t begin, uint32_t count, bool serial) {
    for (uint32_t off = 0; off < count;) {
        const uint32_t n = serial ? 1 : count;
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (uint32_t j = 0; j < n; ++j) {
            common_batch_add(batch, toks[begin + off + j], begin + off + j, {0}, j + 1 == n);
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        if (!ok) return false;
        off += n;
    }
    return true;
}

static double compare_logits(llama_context * a, llama_context * b, int n_vocab, bool & finite) {
    const float * x = llama_get_logits_ith(a, 0);
    const float * y = llama_get_logits_ith(b, 0);
    if (!x || !y) { finite = false; return 0; }
    double result = 0;
    for (int i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { finite = false; return 0; }
        const double diff = std::fabs((double) x[i] - y[i]);
        if (!std::isfinite(diff)) { finite = false; return 0; }
        result = std::max(result, diff);
    }
    return result;
}

static llama_token argmax(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, 0);
    return (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    params.prompt = "The quick brown fox jumps over the lazy dog. Stateful decoding must remain correct after speculative rejection.";
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    ggml_backend_load_all();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    if (!model) return 1;
    char arch_buf[128] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf)) < 0) {
        std::fprintf(stderr, "REFUSE: missing general.architecture\n"); return 2;
    }
    const std::string architecture(arch_buf);
    if (architecture != "glm5next" && architecture != "glm5-next") {
        std::fprintf(stderr, "REFUSE: architecture=%s is not a GLM5Next alias\n", arch_buf); return 2;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        std::fprintf(stderr, "REFUSE: target is not recurrent/hybrid\n"); return 2;
    }

    const int draft_max = env_int("GLM53_TEST_DRAFT_MAX", 5, 1);
    const int cycles = env_int("GLM53_TEST_CYCLES", 2, 2);
    const int horizon = env_int("GLM53_TEST_CONTINUATION", 4, 2);
    // Same predeclared F32 tolerance used by test-recurrent-state-rollback.cpp.
    // Override only before execution, never in response to an observed failure.
    const double tolerance = env_real("GLM53_TEST_LOGIT_TOLERANCE", 1e-5);
    const char * output = std::getenv("GLM53_TEST_JSONL");
    if (!output || !*output) { std::fprintf(stderr, "GLM53_TEST_JSONL is required\n"); return 2; }
    std::ofstream jsonl(output, std::ios::out | std::ios::trunc);
    if (!jsonl) { std::fprintf(stderr, "cannot open %s\n", output); return 2; }

    const uint32_t prefix = 16;
    const uint32_t need = prefix + 1 + draft_max + horizon;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        std::fprintf(stderr, "REFUSE: model vocabulary size is zero\n"); return 2;
    }
    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens.resize(need);
        for (uint32_t i = 0; i < need; ++i) {
            tokens[i] = n_vocab > 1 ? (llama_token) (1 + i % (n_vocab - 1)) : 0;
        }
    } else {
        tokens = common_tokenize(init->context(), params.prompt, true, true);
        if (tokens.empty()) return 2;
        tokens.resize(need, tokens.back());
    }
    if (llama_n_ctx(init->context()) < need || llama_n_batch(init->context()) < (uint32_t) draft_max) {
        std::fprintf(stderr, "REFUSE: context/batch capacity below test shape\n"); return 2;
    }

    for (int cycle = 1; cycle <= cycles; ++cycle) {
        for (int k = 0; k <= draft_max; ++k) {
            llama_context * rollback = make_ctx(params, model, draft_max + 1);
            llama_context * replay = make_ctx(params, model, draft_max + 1);
            llama_context * used = make_ctx(params, model, draft_max + 1);
            if (!rollback || !replay || !used) return 2;
            if (!decode_range(rollback, tokens, 0, prefix, false) ||
                    !decode_range(replay, tokens, 0, prefix, false)) return 2;

            // A real speculative target verification batch contains the sampled anchor
            // followed by every draft token. The anchor is always retained, including k=0.
            if (!decode_range(rollback, tokens, prefix, draft_max + 1, false)) return 2;
            if (!llama_memory_seq_rm(llama_get_memory(rollback), 0, prefix + 1 + k, -1)) {
                std::fprintf(stderr, "rollback refused at k=%d\n", k); return 2;
            }
            // Fresh context replays the same anchor and accepted draft prefix.
            if (!decode_range(replay, tokens, prefix, 1 + k, false)) return 2;

            // Restore the accepted-prefix checkpoint into a context that already
            // owns recurrent snapshots. This catches stale state_read/state_drop
            // and pool/index invalidation which a fresh destination misses.
            std::vector<llama_token> noise(tokens);
            for (auto & tok : noise) tok = (tok + 1) % n_vocab;
            if (!decode_range(used, noise, 0, prefix + draft_max, false) ||
                    !llama_memory_seq_rm(llama_get_memory(used), 0, prefix, -1)) return 2;
            common_prompt_checkpoint accepted;
            accepted.update_tgt(replay, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
            accepted.load_tgt(used, 0, LLAMA_STATE_SEQ_FLAGS_NONE);

            double max_diff = 0.0;
            double used_max_diff = 0.0;
            bool logits_finite = true;
            std::vector<llama_token> roll_next, replay_next, used_next;
            for (int step = 0; step < horizon; ++step) {
                const uint32_t pos = prefix + 1 + k + step;
                if (!decode_range(rollback, tokens, pos, 1, true) ||
                        !decode_range(replay, tokens, pos, 1, true)) return 2;
                max_diff = std::max(max_diff, compare_logits(rollback, replay, n_vocab, logits_finite));
                if (!decode_range(used, tokens, pos, 1, true)) return 2;
                used_max_diff = std::max(used_max_diff, compare_logits(used, replay, n_vocab, logits_finite));
                roll_next.push_back(argmax(rollback, n_vocab));
                replay_next.push_back(argmax(replay, n_vocab));
                used_next.push_back(argmax(used, n_vocab));
            }
            const bool tokens_equal = roll_next == replay_next;
            const bool used_tokens_equal = used_next == replay_next;
            jsonl << "{\"schema\":\"epyc.glm53.mtp_rollback_case.v1\",\"cycle\":" << cycle
                  << ",\"producer\":\"test-glm5next-mtp-rollback/public-api-v1\",\"model_path\":" << json_quote(params.model.path)
                  << ",\"architecture\":\"" << architecture << "\",\"recipe_id\":\"glm53-target-verify-seq-rm-v1\""
                  << ",\"forced_accepted_prefix\":" << k << ",\"drafted_tokens\":" << draft_max
                  << ",\"accepted_prefix_tokens\":[";
            for (int i = 0; i < k; ++i) { if (i) jsonl << ','; jsonl << tokens[prefix + 1 + i]; }
            jsonl << "],\"replay_prefix_tokens\":[";
            for (int i = 0; i < k; ++i) { if (i) jsonl << ','; jsonl << tokens[prefix + 1 + i]; }
            jsonl << "],\"continuation_steps\":" << horizon
                  << ",\"continuation_tokens_equal\":" << (tokens_equal ? "true" : "false")
                  << ",\"max_abs_logit_diff\":" << max_diff
                  << ",\"used_context_max_abs_logit_diff\":" << used_max_diff
                  << ",\"used_context_tokens_equal\":" << (used_tokens_equal ? "true" : "false")
                  << ",\"logits_finite\":" << (logits_finite ? "true" : "false")
                  << ",\"used_context_restore\":true"
                  << ",\"declared_logit_tolerance\":" << tolerance
                  << ",\"state_scope\":\"observable continuation after real target suffix seq_rm; raw unused storage excluded\"}\n";
            llama_free(rollback); llama_free(replay); llama_free(used);
            if (!logits_finite || !tokens_equal || !used_tokens_equal || max_diff > tolerance || used_max_diff > tolerance) {
                std::fprintf(stderr, "FAIL cycle=%d k=%d tokens_equal=%d max_abs=%g tol=%g\n", cycle, k, tokens_equal, max_diff, tolerance);
                return 3;
            }
        }
    }
    std::fprintf(stderr, "PASS: %d cycles x %d prefix lengths, horizon=%d, tolerance=%g\n", cycles, draft_max + 1, horizon, tolerance);
    return 0;
}
