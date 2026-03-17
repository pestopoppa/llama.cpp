#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <queue>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_TREE,
    COMMON_SPECULATIVE_TYPE_DFLASH
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"tree",          COMMON_SPECULATIVE_TYPE_TREE},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void accept(uint16_t n_accepted) = 0;
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < params.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < params.p_min) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

// --- speculation_tree method implementations ---

std::vector<std::vector<int32_t>> speculation_tree::get_paths() const {
    // find leaf nodes (nodes that are not parents of any other node)
    std::vector<bool> is_parent(n_nodes, false);
    for (int32_t i = 0; i < n_nodes; i++) {
        if (parent[i] >= 0) {
            is_parent[parent[i]] = true;
        }
    }

    std::vector<std::vector<int32_t>> paths;
    for (int32_t i = 0; i < n_nodes; i++) {
        if (is_parent[i]) {
            continue; // not a leaf
        }
        // trace back to root
        std::vector<int32_t> path;
        for (int32_t j = i; j >= 0; j = parent[j]) {
            path.push_back(j);
        }
        std::reverse(path.begin(), path.end());
        paths.push_back(std::move(path));
    }
    return paths;
}

llama_tokens speculation_tree::get_best_path() const {
    if (n_nodes == 0) {
        return {};
    }

    auto paths = get_paths();
    if (paths.empty()) {
        return {};
    }

    // find path with highest cumulative log prob at its leaf
    int best_idx = 0;
    float best_prob = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < paths.size(); i++) {
        float p = log_probs[paths[i].back()];
        if (p > best_prob) {
            best_prob = p;
            best_idx = (int) i;
        }
    }

    llama_tokens result;
    for (int32_t node_idx : paths[best_idx]) {
        result.push_back(tokens[node_idx]);
    }
    return result;
}

llama_tokens speculation_tree::get_greedy_path() const {
    if (n_nodes == 0) {
        return {};
    }

    // The greedy path follows primary children (k==0 at each expansion step).
    // Node 0 is always the greedy root. For each parent, the primary child is
    // the first node in the array with parent[i] == cur (added first during expansion).
    llama_tokens result;
    int32_t cur = 0;
    result.push_back(tokens[cur]);

    while (true) {
        int32_t first_child = -1;
        for (int32_t i = cur + 1; i < n_nodes; i++) {
            if (parent[i] == cur) {
                first_child = i;
                break;
            }
        }
        if (first_child < 0) {
            break;
        }
        result.push_back(tokens[first_child]);
        cur = first_child;
    }

    return result;
}

// --- DFlash speculation state ---
// DFlash block diffusion drafting with target model conditioning.
// Reads hidden states from the target context, conditions the DFlash drafter,
// and generates draft tokens autoregressively.

struct common_speculative_state_dflash : public common_speculative_state_draft {
    // DFlash target layer IDs (from drafter model config)
    // TODO: read from GGUF metadata
    static constexpr int dflash_taps[] = {1, 12, 23, 34, 45};
    static constexpr int n_taps = 5;

    common_speculative_state_dflash(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state_draft(type, ctx_tgt, ctx_dft, replacements) {}

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        // Step 1: Extract hidden states from the target context
        const int n_embd = llama_model_n_embd(llama_get_model(ctx_tgt));
        const int32_t n_hidden = llama_get_hidden_state_count(ctx_tgt);

        if (n_hidden > 0) {
            // Concatenate hidden states at target layer IDs
            std::vector<float> concat_hidden(n_taps * n_embd);
            bool all_available = true;

            for (int t = 0; t < n_taps; t++) {
                float * hs = llama_get_hidden_state(ctx_tgt, dflash_taps[t]);
                if (hs) {
                    memcpy(concat_hidden.data() + t * n_embd, hs, n_embd * sizeof(float));
                } else {
                    all_available = false;
                    break;
                }
            }

            if (all_available) {
                // Set conditioning on DFlash drafter
                llama_set_cross_data(ctx_dft, n_taps * n_embd, 1, concat_hidden.data());
            }
        }

        // Step 2: Generate draft tokens using the conditioned drafter
        // Fall through to base class draft which handles the AR generation loop
        common_speculative_state_draft::draft(params, prompt_tgt, id_last, result);
    }
};

// --- Tree speculation state ---
// Wraps common_speculative_state_draft and adds p_split tree branching.
// During draft(), builds a tree by exploring multiple candidate continuations.
// Returns best-path tokens (flat, backward-compatible with linear verification).
// The full tree is available via common_speculative_get_tree() for multi-path verification.

struct common_speculative_state_tree : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true;
    std::unordered_map<std::string, std::string> vocab_map;

    speculation_tree tree; // built during draft()

    // max branches to spawn per node (limits fan-out)
    static constexpr int MAX_BRANCHES_PER_NODE = 7;

    common_speculative_state_tree(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };
            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        if (!vocab_cmpt) {
            LOG_WRN("tree spec: target and draft vocabs not compatible - tokens will be translated\n");
            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_tree() override {
        llama_perf_context_print(ctx_dft);
        llama_free(ctx_dft);
        common_sampler_free(smpl);
        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        auto * mem_dft = llama_get_memory(ctx_dft);

        // --- Prompt reuse logic (same as linear draft) ---
        int reuse_i = 0;
        int reuse_n = 0;
        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!vocab_cmpt) {
            std::string text = common_detokenize(ctx_tgt, prompt_tgt, true);
            for (const auto & pair : vocab_map) {
                size_t pos = text.find(pair.first);
                while (pos != std::string::npos) {
                    text.replace(pos, pair.first.length(), pair.second);
                    pos = text.find(pair.first, pos + pair.second.length());
                }
            }
            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            const auto * vocab_tgt = llama_model_get_vocab(llama_get_model(ctx_tgt));
            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0);
            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            for (const auto & pair : vocab_map) {
                size_t pos = text.find(pair.first);
                while (pos != std::string::npos) {
                    text.replace(pos, pair.first.length(), pair.second);
                    pos = text.find(pair.first, pos + pair.second.length());
                }
            }
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = vocab_cmpt ? prompt_tgt : prompt_cnv;
        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }
            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // check for reusable previous draft
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);
                    if (params.n_max <= (int) result.size()) break;
                }
                return;
            }

            if (reuse_i > 0) {
                llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);
                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }
            if (reuse_n < (int) prompt_dft.size()) {
                llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // eval new prompt tokens
        common_batch_clear(batch);
        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);
            prompt_dft.push_back(prompt_cur[i]);
        }
        if (batch.n_tokens > 0) {
            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        // eval id_last
        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past, { 0 }, true);
        prompt_dft.push_back(id_last);
        llama_decode(ctx_dft, batch);
        common_sampler_reset(smpl);

        // --- DySpec: heap-based dynamic tree construction ---
        // Globally prioritizes the most promising frontier nodes by cumulative log probability.
        // Wave-based: pop up to WAVE_BATCH_SIZE nodes per llama_decode call.
        tree = speculation_tree{}; // reset

        // Budget = n_max tree nodes (each node = one drafted token)
        const int budget = params.n_max;
        const int max_depth = params.n_max; // depth cap to prevent degenerate narrow trees

        // Wave batch size: how many frontier nodes to expand per llama_decode call
        static constexpr int WAVE_BATCH_SIZE = 8;

        // Sample root candidates
        common_sampler_sample(smpl, ctx_dft, 0, true);
        const auto * cur_p = common_sampler_get_candidates(smpl, true);

        if (cur_p->size == 0) {
            return;
        }

        // Branching factor at root: up to MAX_BRANCHES_PER_NODE candidates above p_split
        // At deeper levels, branching decreases (top-3 instead of top-7)
        auto branching_factor = [&](int depth) -> int {
            if (depth == 0) return MAX_BRANCHES_PER_NODE; // 7 at root
            if (depth <= 2) return 5;
            if (depth <= 4) return 3;
            return 2; // deep nodes: only top-2
        };

        // Seed tree with root candidates
        llama_seq_id next_seq_id = 1;

        // Priority queue: (cumulative_log_prob, node_idx, depth, seq_id, sampler)
        // We need a max-heap (highest probability first)
        struct frontier_node {
            float    cum_log_prob;
            int32_t  node_idx;
            int32_t  depth;
            llama_seq_id seq_id;
            common_sampler * smpl;
            bool owns_smpl;

            // max-heap: higher probability = higher priority
            bool operator<(const frontier_node & other) const {
                return cum_log_prob < other.cum_log_prob;
            }
        };

        std::priority_queue<frontier_node> heap;

        // Add root candidates to tree and heap
        const int n_root_candidates = std::min(branching_factor(0), (int) cur_p->size);
        for (int k = 0; k < n_root_candidates; k++) {
            if (k > 0 && cur_p->data[k].p < params.p_split) break;

            const llama_token tok = cur_p->data[k].id;
            const float log_p = std::log(std::max(cur_p->data[k].p, 1e-10f));

            int32_t node = tree.n_nodes;
            tree.parent.push_back(-1); // root nodes have no parent
            tree.tokens.push_back(tok);
            tree.log_probs.push_back(log_p);
            tree.n_nodes++;

            llama_seq_id seq;
            common_sampler * node_smpl;
            bool owns;

            if (k == 0) {
                // Primary root: use seq_id 0 and main sampler
                seq = 0;
                node_smpl = smpl;
                owns = false;
                common_sampler_accept(smpl, tok, true);
            } else {
                // Alternative roots: clone KV and sampler
                seq = next_seq_id++;
                llama_memory_seq_cp(mem_dft, 0, seq, 0, -1);
                node_smpl = common_sampler_clone(smpl);
                common_sampler_accept(node_smpl, tok, true);
                owns = true;
            }

            heap.push({log_p, node, 0, seq, node_smpl, owns});
        }

        // Wave-based expansion: pop batch_size nodes, decode, expand children
        while (tree.n_nodes < budget && !heap.empty()) {
            // Pop up to WAVE_BATCH_SIZE highest-probability frontier nodes
            std::vector<frontier_node> wave;
            while ((int) wave.size() < WAVE_BATCH_SIZE && !heap.empty()) {
                auto fn = heap.top();
                heap.pop();

                // Skip if depth limit reached
                if (fn.depth >= max_depth - 1) {
                    if (fn.owns_smpl) common_sampler_free(fn.smpl);
                    if (fn.seq_id != 0) llama_memory_seq_rm(mem_dft, fn.seq_id, 0, -1);
                    continue;
                }

                wave.push_back(fn);
            }

            if (wave.empty()) break;

            // Batch decode: one forward pass for all wave nodes
            common_batch_clear(batch);
            for (size_t w = 0; w < wave.size(); w++) {
                auto & fn = wave[w];
                common_batch_add(batch, tree.tokens[fn.node_idx],
                    n_past + fn.depth + 1, { fn.seq_id }, true);
            }

            llama_decode(ctx_dft, batch);

            // Expand children for each wave node
            for (size_t w = 0; w < wave.size(); w++) {
                auto & fn = wave[w];

                common_sampler_sample(fn.smpl, ctx_dft, (int) w, true);
                const auto * cands = common_sampler_get_candidates(fn.smpl, true);

                if (cands->size == 0) {
                    if (fn.owns_smpl) common_sampler_free(fn.smpl);
                    if (fn.seq_id != 0) llama_memory_seq_rm(mem_dft, fn.seq_id, 0, -1);
                    continue;
                }

                const int n_children = std::min(branching_factor(fn.depth + 1), (int) cands->size);

                for (int k = 0; k < n_children && tree.n_nodes < budget; k++) {
                    // p_split threshold for non-primary children
                    if (k > 0 && cands->data[k].p < params.p_split) break;
                    // p_min early stop for primary child
                    if (k == 0 && cands->data[0].p < params.p_min) {
                        // Add leaf but don't push to heap (won't be expanded further)
                        tree.parent.push_back(fn.node_idx);
                        tree.tokens.push_back(cands->data[0].id);
                        tree.log_probs.push_back(fn.cum_log_prob +
                            std::log(std::max(cands->data[0].p, 1e-10f)));
                        tree.n_nodes++;
                        break; // don't spawn alternatives from low-confidence node
                    }

                    const llama_token child_tok = cands->data[k].id;
                    const float child_log_p = fn.cum_log_prob +
                        std::log(std::max(cands->data[k].p, 1e-10f));

                    int32_t child_node = tree.n_nodes;
                    tree.parent.push_back(fn.node_idx);
                    tree.tokens.push_back(child_tok);
                    tree.log_probs.push_back(child_log_p);
                    tree.n_nodes++;

                    llama_seq_id child_seq;
                    common_sampler * child_smpl;
                    bool child_owns;

                    if (k == 0) {
                        // Primary child: inherit parent's seq_id and sampler
                        child_seq = fn.seq_id;
                        child_smpl = fn.smpl;
                        child_owns = fn.owns_smpl;
                        common_sampler_accept(fn.smpl, child_tok, true);
                        fn.owns_smpl = false; // transferred to child
                    } else {
                        // Alternative child: clone KV and sampler from parent
                        if (next_seq_id >= 32) break; // cap total seq_ids
                        child_seq = next_seq_id++;
                        llama_memory_seq_cp(mem_dft, fn.seq_id, child_seq,
                            0, -1);
                        child_smpl = common_sampler_clone(fn.smpl);
                        common_sampler_accept(child_smpl, child_tok, true);
                        child_owns = true;
                    }

                    heap.push({child_log_p, child_node, fn.depth + 1,
                               child_seq, child_smpl, child_owns});
                }

                // Clean up parent if ownership wasn't transferred (no primary child)
                if (fn.owns_smpl) {
                    common_sampler_free(fn.smpl);
                }
                if (fn.seq_id != 0 && fn.owns_smpl) {
                    // seq_id was not inherited by any child — clean up
                    // (if owns_smpl is still true, primary child path wasn't taken)
                    llama_memory_seq_rm(mem_dft, fn.seq_id, 0, -1);
                }
            }
        }

        // Clean up any remaining heap entries (free samplers only — KV cleaned below)
        while (!heap.empty()) {
            auto fn = heap.top();
            heap.pop();
            if (fn.owns_smpl) common_sampler_free(fn.smpl);
        }

        // Return greedy path tokens (top-1 at each depth, matches linear speculation)
        // This guarantees path 0 is never worse than linear — tree can only improve
        // via alternative paths verified in multi-path target verification.
        result = tree.get_greedy_path();

        if ((int) result.size() > params.n_max) {
            result.resize(params.n_max);
        }

        // Clean up ALL tree KV state:
        // - Remove tree expansion positions from seq 0 (keep only prompt + id_last)
        // - Remove all alternative seq_ids entirely
        // This ensures KV cache is consistent with prompt_dft for the next call
        llama_memory_seq_rm(mem_dft, 0, n_past + 1, -1);
        for (llama_seq_id s = 1; s < next_seq_id; s++) {
            llama_memory_seq_rm(mem_dft, s, 0, -1);
        }
        // prompt_dft already has prompt + id_last from before tree construction
        // Do NOT append tree tokens — they're not in the KV cache for seq 0

        LOG_DBG("tree/dyspec: built tree with %d nodes, %zu paths, returning %zu tokens\n",
                tree.n_nodes, tree.get_paths().size(), result.size());

        // Vocab translation if needed
        if (!vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            for (const auto & pair : vocab_map) {
                size_t pos = detokenized.find(pair.second);
                while (pos != std::string::npos) {
                    detokenized.replace(pos, pair.second.length(), pair.first);
                    pos = detokenized.find(pair.second, pos + pair.first.length());
                }
            }
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t) params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(map, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_TREE:          return "tree";
        case COMMON_SPECULATIVE_TYPE_DFLASH:        return "dflash";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    auto * mem = llama_get_memory(ctx_tgt);
    if (mem == nullptr) {
        return false;
    }

    bool res = true;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        // partial seq_rm failed — this is expected for recurrent/hybrid models
        // check if we can use checkpoint/restore instead
        if (llama_memory_has_recurrent(mem)) {
            LOG_INF("%s: partial seq_rm not supported, but recurrent checkpointing available — speculation enabled\n", __func__);
            // speculation is compatible via checkpoint/restore path
        } else {
            LOG_WRN("%s: the target context does not support partial sequence removal or recurrent checkpointing\n", __func__);
            res = false;
            goto done;
        }
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx_tgt);

    return res;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    if (params.model_dft) {
        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.mparams_dft.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            // shared instance for all speculative decoding contexts
            if (!params.ngram_mod) {
                params.ngram_mod = std::make_shared<common_ngram_mod>(params.ngram_size_n, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                        params.ngram_size_n, params.ngram_mod->size(),
                        (float)(params.ngram_mod->size_bytes())/1024/1024);

                if (params.ngram_size_n < 16) {
                    LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, params.ngram_size_n);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft) {
            // Check if the draft model is a DFlash drafter (by architecture name)
            bool is_dflash = false;
            if (ctx_dft) {
                char arch_buf[64] = {};
                llama_model_meta_val_str(llama_get_model(ctx_dft), "general.architecture", arch_buf, sizeof(arch_buf));
                is_dflash = (std::string(arch_buf) == "dflash");
            }

            if (is_dflash) {
                // DFlash block diffusion drafting
                configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DFLASH, params));
            } else if (params.p_split > 0.0f) {
                // tree speculation: draft model with p_split branching
                configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_TREE, params));
            } else {
                configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
            }
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        params.lookup_cache_static, params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_TREE: {
                impls.push_back(std::make_unique<common_speculative_state_tree>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_state_dflash>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                LOG_INF("%s: DFlash speculation enabled — conditioning from target hidden states\n", __func__);
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls = */ std::move(impls)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // We have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}

const speculation_tree * common_speculative_get_tree(const common_speculative * spec) {
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return nullptr;
    }

    if (spec->curr_impl->type == COMMON_SPECULATIVE_TYPE_TREE) {
        auto * tree_state = static_cast<const common_speculative_state_tree *>(spec->curr_impl);
        if (tree_state->tree.n_nodes > 0) {
            return &tree_state->tree;
        }
    }

    return nullptr;
}
