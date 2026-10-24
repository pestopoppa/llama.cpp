#pragma once

#include "llama.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;
struct common_speculative_token_dist;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(const struct llama_model * model, struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
// if out_dists is non-null, one entry is appended per returned token, holding a sorted
// (descending probability) snapshot of the post-chain candidate distribution at the target
// idx used to accept that token -- i.e. the same distribution common_sampler_get_candidates()
// would report had that token been sampled outside of speculative verification. Passing
// nullptr (the default) skips the snapshot entirely, so non-probability callers pay nothing.
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false, std::vector<std::vector<llama_token_data>> * out_dists = nullptr);

// maximal-coupling verification for stochastic speculative decoding
//
// out_dists: see the overload above -- same contract, one snapshot per returned token.
std::vector<llama_token> common_sampler_sample_and_accept_n(
        struct common_sampler * gsmpl,
        struct llama_context * ctx,
        const std::vector<int> & idxs,
        const llama_tokens & draft,
        const std::vector<common_speculative_token_dist> & dists,
        bool grammar_first = false,
        std::vector<std::vector<llama_token_data>> * out_dists = nullptr);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(
        struct common_sampler * gsmpl,
        struct llama_context * ctx,
        const llama_tokens & draft,
        const std::vector<common_speculative_token_dist> & dists,
        bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

// GPU-ZOS-2 C2a (`LLAMA_ZOS2=greedy-fast`, experimental): an exact greedy fast path for the
// verification rows of speculative decoding. Returns exactly what
//
//      common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, false, nullptr)
//
// returns, and leaves the grammar, the reasoning budget, the penalty/DRY history and the `prev`
// ring in the same state (every returned token goes through common_sampler_accept, as there).
//
// Per row, when the sampler is provably greedy (common_sampler_greedy_fast_select below, plus no
// grammar, no reasoning budget and no backend-sampled row), the token is the unique argmax of
// logit + logit_bias over the raw logits row, found in one scan instead of running the chain.
// Any other row -- a tied maximum, a NaN, a non-finite maximum, a backend-sampled row, or an
// ineligible sampler -- is sampled by common_sampler_sample, i.e. the champion path.
//
// Differences that cannot change a returned token: on a fast row the chain does not run, so the
// candidate buffer (common_sampler_get_candidates) is stale -- callers must not request n_probs --
// the `dist` sampler's RNG is not advanced (at temp <= 0 with top-k > 0 its draw never changes
// the selected token), and the chain's perf counters do not count the row.
//
// n_fast / n_fallback (optional) are incremented by the number of rows taken by the fast path and
// by the chain respectively.
std::vector<llama_token> common_sampler_sample_and_accept_n_greedy_fast(
        struct common_sampler * gsmpl,
        struct llama_context * ctx,
        const std::vector<int> & idxs,
        const llama_tokens & draft,
        int64_t * n_fast     = nullptr,
        int64_t * n_fallback = nullptr);

// GPU-ZOS-2 C2a row kernel, model-free (exposed for tests/test-sampling-greedy-fast.cpp).
// Returns the token the sampling chain built from `params` would select on this raw logits row,
// or LLAMA_TOKEN_NULL when that cannot be decided exactly without running the chain:
//   - params not provably greedy: requires mirostat 0, temp <= 0 with no dynamic temperature, a
//     TEMPERATURE and a top-k (> 0) sampler in the chain, n_probs 0, and every other sampler a
//     no-op (penalties neutral, dry_multiplier 0, top_n_sigma <= 0, typ_p >= 1, xtc_probability
//     <= 0) or max-preserving (top-p, min-p); infill and adaptive-p are never eligible;
//   - the row has a NaN, a tied maximum, or a non-finite maximum (after logit_bias).
// logit_bias is applied exactly as llama_sampler_logit_bias does on an id-ordered full-vocab row:
// entries in order, `logit += bias`, out-of-range tokens ignored. The caller guarantees there is
// no grammar and no reasoning-budget sampler.
llama_token common_sampler_greedy_fast_select(const struct common_params_sampling & params, const float * logits, int32_t n_vocab);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
