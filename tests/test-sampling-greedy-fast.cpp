// GPU-ZOS-2.1 (C2a `LLAMA_ZOS2=greedy-fast`): the greedy fast-path row kernel must select exactly the token the
// sampling chain selects whenever it engages, and must refuse (LLAMA_TOKEN_NULL -> chain fallback) on ties, NaNs,
// non-finite maxima and every non-greedy sampler configuration.
//
// The chain is built here from the same params, in the same order and with the same constructor arguments as
// common_sampler_init() (common/sampling.cpp); DRY uses llama_sampler_init_dry_testing as tests/test-sampling.cpp
// does, since there is no vocab without a model. Model-free; CPU only.

#include "common.h"
#include "llama.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static int n_fail   = 0;
static int n_checks = 0;

#define CHECK(cond, ...) do { \
    n_checks++; \
    if (!(cond)) { \
        n_fail++; \
        fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

// mirror of common_sampler_init()'s chain construction (mirostat == 0 branch, plus the mirostat branches so that
// the eligibility check can be exercised against a chain that really is non-greedy)
static llama_sampler * build_chain(const common_params_sampling & params, int32_t n_vocab) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());

    if (params.has_logit_bias()) {
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, params.logit_bias.size(), params.logit_bias.data()));
    }

    if (params.mirostat == 0) {
        bool use_adaptive_p = false;
        for (const auto type : params.samplers) {
            switch (type) {
                case COMMON_SAMPLER_TYPE_DRY:
                    // llama_sampler_init_dry returns the empty "?dry" sampler when DRY is disabled (dry_multiplier 0,
                    // the default), and the _testing wrapper cannot handle that case, so mirror it directly
                    if (params.dry_multiplier != 0.0f && params.dry_base >= 1.0f && params.dry_penalty_last_n != 0) {
                        llama_sampler_chain_add(chain, llama_sampler_init_dry_testing(4096, params.dry_multiplier, params.dry_base,
                                    params.dry_allowed_length, params.dry_penalty_last_n, { { 1 } }));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    llama_sampler_chain_add(chain, llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    llama_sampler_chain_add(chain, llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    llama_sampler_chain_add(chain, llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    llama_sampler_chain_add(chain, llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    use_adaptive_p = true;
                    break;
                default:
                    break; // INFILL needs a vocab; the eligibility test below only checks that it is refused
            }
        }
        if (use_adaptive_p) {
            llama_sampler_chain_add(chain, llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(chain, llama_sampler_init_mirostat(n_vocab, params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(chain, llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    }

    return chain;
}

// what common_sampler_sample() does on the raw-logits path: an id-ordered full-vocab row, then the chain
static llama_token chain_select(llama_sampler * chain, const std::vector<float> & logits) {
    std::vector<llama_token_data> cur(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        cur[i] = llama_token_data{ (llama_token) i, logits[i], 0.0f };
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(chain, &cur_p);
    assert(cur_p.selected >= 0 && (size_t) cur_p.selected < cur_p.size);
    const llama_token id = cur_p.data[cur_p.selected].id;
    llama_sampler_accept(chain, id);
    return id;
}

static common_params_sampling greedy_params() {
    common_params_sampling p; // the full default chain: penalties, dry, top-n-sigma, top-k 40, typical, top-p 0.95, min-p 0.05, xtc, temp
    p.temp = 0.0f;
    p.seed = 1234;
    return p;
}

static std::vector<float> random_row(std::mt19937 & rng, int32_t n_vocab, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> row(n_vocab);
    for (auto & v : row) {
        v = nd(rng);
    }
    return row;
}

// run one row through both; `expect_engage`: 1 must engage, 0 must refuse, -1 either
static void check_row(const char * name, llama_sampler * chain, const common_params_sampling & p,
                      const std::vector<float> & row, int expect_engage) {
    const llama_token fast = common_sampler_greedy_fast_select(p, row.data(), (int32_t) row.size());
    const llama_token ref  = chain_select(chain, row);

    if (expect_engage == 1) {
        CHECK(fast != LLAMA_TOKEN_NULL, "%s: fast path did not engage (chain picked %d)", name, ref);
    } else if (expect_engage == 0) {
        CHECK(fast == LLAMA_TOKEN_NULL, "%s: fast path engaged (%d) where it must fall back", name, fast);
    }
    if (fast != LLAMA_TOKEN_NULL) {
        CHECK(fast == ref, "%s: fast %d != chain %d", name, fast, ref);
    }
}

static void test_random_rows(const common_params_sampling & p, const char * name, int32_t n_vocab, int n_rows, uint32_t seed) {
    std::mt19937 rng(seed);
    llama_sampler * chain = build_chain(p, n_vocab);
    for (int r = 0; r < n_rows; ++r) {
        auto row = random_row(rng, n_vocab, 1.0f + (float) (r % 7));
        check_row(name, chain, p, row, 1);
    }
    llama_sampler_free(chain);
}

static void test_edge_rows() {
    const int32_t n_vocab = 2048;
    std::mt19937 rng(42);

    // ties at the maximum -> fallback
    {
        auto p = greedy_params();
        llama_sampler * chain = build_chain(p, n_vocab);
        auto row = random_row(rng, n_vocab, 1.0f);
        row[17] = 50.0f; row[1900] = 50.0f;
        check_row("tie-at-max", chain, p, row, 0);
        // a tie below the maximum does not matter
        row[1900] = 49.0f; row[3] = 49.0f;
        check_row("tie-below-max", chain, p, row, 1);
        // one-ulp separation engages and matches
        row[3] = std::nextafter(50.0f, 0.0f);
        check_row("one-ulp-gap", chain, p, row, 1);
        // maximum at the first and at the last index
        auto row0 = random_row(rng, n_vocab, 1.0f); row0[0] = 30.0f;
        check_row("max-at-0", chain, p, row0, 1);
        auto rowl = random_row(rng, n_vocab, 1.0f); rowl[n_vocab - 1] = 30.0f;
        check_row("max-at-last", chain, p, rowl, 1);
        llama_sampler_free(chain);
    }

    // NaN anywhere -> fallback (the chain's behaviour with NaN is not reproduced)
    {
        auto p = greedy_params();
        llama_sampler * chain = build_chain(p, n_vocab);
        auto row = random_row(rng, n_vocab, 1.0f);
        row[100] = 40.0f;
        row[5] = NAN;
        const llama_token fast = common_sampler_greedy_fast_select(p, row.data(), n_vocab);
        CHECK(fast == LLAMA_TOKEN_NULL, "nan-below-max: fast engaged (%d)", fast);
        row[5] = 0.0f; row[100] = NAN;
        CHECK(common_sampler_greedy_fast_select(p, row.data(), n_vocab) == LLAMA_TOKEN_NULL, "nan-at-max: fast engaged");
        llama_sampler_free(chain);
    }

    // +inf logit or all -inf -> fallback
    {
        auto p = greedy_params();
        auto row = random_row(rng, n_vocab, 1.0f);
        row[7] = INFINITY;
        CHECK(common_sampler_greedy_fast_select(p, row.data(), n_vocab) == LLAMA_TOKEN_NULL, "+inf logit: fast engaged");
        std::vector<float> ninf(n_vocab, -INFINITY);
        CHECK(common_sampler_greedy_fast_select(p, ninf.data(), n_vocab) == LLAMA_TOKEN_NULL, "all -inf: fast engaged");
    }

    // -inf bias on the argmax (ignore_eos style: EOG ids at -inf) -> the runner-up, as the chain
    {
        auto p = greedy_params();
        auto row = random_row(rng, n_vocab, 1.0f);
        row[11] = 40.0f; row[12] = 39.0f; row[13] = 38.0f;
        p.logit_bias = { { 11, -INFINITY }, { 13, -INFINITY } };
        llama_sampler * chain = build_chain(p, n_vocab);
        check_row("-inf-bias-on-argmax", chain, p, row, 1);
        CHECK(common_sampler_greedy_fast_select(p, row.data(), n_vocab) == 12, "-inf bias: expected 12");
        llama_sampler_free(chain);
    }

    // a positive bias promotes a non-max token; duplicate entries add in order; out-of-range tokens are ignored
    {
        auto p = greedy_params();
        auto row = random_row(rng, n_vocab, 1.0f);
        row[200] = 20.0f; row[300] = 10.0f;
        p.logit_bias = { { 300, 6.0f }, { -5, 100.0f }, { n_vocab + 3, 100.0f }, { 300, 4.5f } };
        llama_sampler * chain = build_chain(p, n_vocab);
        check_row("bias-promotes", chain, p, row, 1);
        CHECK(common_sampler_greedy_fast_select(p, row.data(), n_vocab) == 300, "bias-promotes: expected 300");
        llama_sampler_free(chain);
    }

    // a bias that creates an exact tie at the maximum -> fallback
    {
        auto p = greedy_params();
        auto row = random_row(rng, n_vocab, 1.0f);
        row[200] = 20.0f; row[300] = 10.0f;
        p.logit_bias = { { 300, 10.0f } };
        llama_sampler * chain = build_chain(p, n_vocab);
        check_row("bias-makes-tie", chain, p, row, 0);
        llama_sampler_free(chain);
    }

    // +inf bias -> fallback
    {
        auto p = greedy_params();
        auto row = random_row(rng, n_vocab, 1.0f);
        p.logit_bias = { { 9, INFINITY } };
        CHECK(common_sampler_greedy_fast_select(p, row.data(), n_vocab) == LLAMA_TOKEN_NULL, "+inf bias: fast engaged");
    }

    // a NaN bias -> fallback
    {
        auto p = greedy_params();
        auto row = random_row(rng, n_vocab, 1.0f);
        p.logit_bias = { { 9, NAN } };
        CHECK(common_sampler_greedy_fast_select(p, row.data(), n_vocab) == LLAMA_TOKEN_NULL, "nan bias: fast engaged");
    }
}

static void test_eligibility() {
    const int32_t n_vocab = 512;
    std::mt19937 rng(7);
    auto row = random_row(rng, n_vocab, 2.0f);
    row[77] = 25.0f; // a clear unique maximum: only the params decide

    struct variant { const char * name; void (*mut)(common_params_sampling &); };
    const variant refused[] = {
        { "temp>0",            [](common_params_sampling & p) { p.temp = 0.2f; } },
        { "temp NaN",          [](common_params_sampling & p) { p.temp = NAN; } },
        { "dynatemp",          [](common_params_sampling & p) { p.dynatemp_range = 0.5f; } },
        { "mirostat 1",        [](common_params_sampling & p) { p.mirostat = 1; } },
        { "mirostat 2",        [](common_params_sampling & p) { p.mirostat = 2; } },
        { "n_probs",           [](common_params_sampling & p) { p.n_probs = 5; } },
        { "repeat penalty",    [](common_params_sampling & p) { p.penalty_repeat = 1.1f; } },
        { "freq penalty",      [](common_params_sampling & p) { p.penalty_freq = 0.1f; } },
        { "presence penalty",  [](common_params_sampling & p) { p.penalty_present = 0.1f; } },
        { "dry",               [](common_params_sampling & p) { p.dry_multiplier = 0.8f; } },
        { "top-n-sigma",       [](common_params_sampling & p) { p.top_n_sigma = 1.0f; } },
        { "typical",           [](common_params_sampling & p) { p.typ_p = 0.9f; } },
        { "xtc",               [](common_params_sampling & p) { p.xtc_probability = 0.5f; } },
        { "top-k 0",           [](common_params_sampling & p) { p.top_k = 0; } },
        { "no temperature",    [](common_params_sampling & p) { p.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P }; } },
        { "no top-k",          [](common_params_sampling & p) { p.samplers = { COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE }; } },
        { "adaptive-p",        [](common_params_sampling & p) { p.samplers.push_back(COMMON_SAMPLER_TYPE_ADAPTIVE_P); } },
        { "infill",            [](common_params_sampling & p) { p.samplers.push_back(COMMON_SAMPLER_TYPE_INFILL); } },
        { "top-p NaN",         [](common_params_sampling & p) { p.top_p = NAN; } },
    };
    for (const auto & v : refused) {
        auto p = greedy_params();
        v.mut(p);
        CHECK(common_sampler_greedy_fast_select(p, row.data(), n_vocab) == LLAMA_TOKEN_NULL, "%s: must be refused", v.name);
    }

    const variant accepted[] = {
        { "defaults temp 0",    [](common_params_sampling &)     { } },
        { "temp -1",            [](common_params_sampling & p) { p.temp = -1.0f; } },
        { "penalty_last_n 0",   [](common_params_sampling & p) { p.penalty_repeat = 1.3f; p.penalty_last_n = 0; } },
        { "top-p 1 min-p 0",    [](common_params_sampling & p) { p.top_p = 1.0f; p.min_p = 0.0f; } },
        { "top-p 0.1 min-p 0.9",[](common_params_sampling & p) { p.top_p = 0.1f; p.min_p = 0.9f; } },
        { "top-k 1",            [](common_params_sampling & p) { p.top_k = 1; } },
        { "temp first, top-k last", [](common_params_sampling & p) {
              p.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TOP_K }; } },
    };
    for (const auto & v : accepted) {
        auto p = greedy_params();
        v.mut(p);
        llama_sampler * chain = build_chain(p, n_vocab);
        check_row(v.name, chain, p, row, 1);
        llama_sampler_free(chain);
    }

    // the refused-but-buildable variants: whatever the chain does, the fast path never claims a token
    for (const auto & v : refused) {
        auto p = greedy_params();
        v.mut(p);
        if (std::isnan(p.temp) || std::isnan(p.top_p)) {
            continue;
        }
        bool has_infill = false;
        for (auto t : p.samplers) { has_infill |= t == COMMON_SAMPLER_TYPE_INFILL; }
        if (has_infill) {
            continue;
        }
        llama_sampler * chain = build_chain(p, n_vocab);
        check_row(v.name, chain, p, row, 0);
        llama_sampler_free(chain);
    }
}

int main(void) {
    // the full default chain at temp 0, at two vocab sizes (one the size of the :8083 target's vocab), and a
    // long-lived chain whose penalties / dist state advance across rows as in a real request
    test_random_rows(greedy_params(), "default-chain-v4096", 4096, 400, 1);
    test_random_rows(greedy_params(), "default-chain-v248320", 248320, 12, 2);
    {
        auto p = greedy_params();
        p.logit_bias = { { 5, -INFINITY }, { 6, -INFINITY }, { 4095, 0.25f } };
        test_random_rows(p, "default-chain-eog-bias", 4096, 200, 3);
    }
    {
        auto p = greedy_params();
        p.top_k = 0;    // refused: the chain then has no sorted top-k
        std::mt19937 rng(9);
        auto row = random_row(rng, 1024, 1.0f);
        CHECK(common_sampler_greedy_fast_select(p, row.data(), 1024) == LLAMA_TOKEN_NULL, "top_k 0 must be refused");
    }

    test_edge_rows();
    test_eligibility();

    printf("test-sampling-greedy-fast: %d checks, %d failures\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
