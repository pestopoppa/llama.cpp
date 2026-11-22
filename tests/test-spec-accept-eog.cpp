// SPEC-STALE-1 (a): every speculative accept loop stops at an accepted EOG that is not the last draft token
// (upstream #29638 for the plain loop, philippesic ed1254a for the rejection loop; ours also covers the
// greedy-fast loop). A trailing EOG (checkpoint replay draft) is still accepted and followed by the bonus row.
//
// Every row's target pick is forced with a +1000 logit bias, so the expected result is a pure function of the
// draft. Needs a model (any vocab with an EOG); skips without one.
//   expected: {eog,eog} -> 1, {eog,eog,eog} -> 1, {eog} -> 2, {t,t} -> 3 (control, non-EOG), {t,eog,eog} -> 2

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "get-model.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_fail = 0;

#define CHECK_EQ(a, b, what) do { \
    if ((a) != (b)) { fprintf(stderr, "FAIL %s: got %d want %d\n", what, (int) (a), (int) (b)); n_fail++; } \
    else            { fprintf(stderr, "ok   %s: %d\n", what, (int) (a)); } } while (0)

enum loop_kind { LOOP_PLAIN, LOOP_GREEDY_FAST, LOOP_REJECTION };

static const char * kind_name(loop_kind k) {
    return k == LOOP_PLAIN ? "plain" : k == LOOP_GREEDY_FAST ? "greedy_fast" : "rejection";
}

// decode a fresh batch of draft.size()+1 rows (all with logits) and run one accept loop on it
static size_t run_accept(llama_model * model, llama_context * ctx, loop_kind kind, llama_token forced,
                         const llama_tokens & draft, const llama_tokens & expect_prefix) {
    llama_memory_clear(llama_get_memory(ctx), true);

    const int n = (int) draft.size() + 1;
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        common_batch_add(batch, i == 0 ? llama_vocab_bos(llama_model_get_vocab(model)) : draft[i - 1], i, { 0 }, true);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        exit(1);
    }
    llama_batch_free(batch);

    common_params_sampling sp;
    sp.temp = 0.0f;
    sp.logit_bias.push_back({ forced, 1000.0f });
    common_sampler * smpl = common_sampler_init(model, sp);

    std::vector<int> idxs(n);
    for (int i = 0; i < n; ++i) {
        idxs[i] = i;
    }

    std::vector<llama_token> res;
    if (kind == LOOP_PLAIN) {
        res = common_sampler_sample_and_accept_n(smpl, ctx, idxs, draft);
    } else if (kind == LOOP_GREEDY_FAST) {
        res = common_sampler_sample_and_accept_n_greedy_fast(smpl, ctx, idxs, draft, nullptr, nullptr);
    } else {
        std::vector<common_speculative_token_dist> dists(draft.size());
        for (size_t i = 0; i < draft.size(); ++i) {
            dists[i].ids   = { draft[i] };
            dists[i].probs = { 1.0f };
        }
        res = common_sampler_sample_and_accept_n(smpl, ctx, idxs, draft, dists, false);
    }
    common_sampler_free(smpl);

    for (size_t i = 0; i < res.size() && i < expect_prefix.size(); ++i) {
        if (res[i] != expect_prefix[i]) {
            fprintf(stderr, "FAIL %s: token %zu = %d want %d\n", kind_name(kind), i, res[i], expect_prefix[i]);
            n_fail++;
        }
    }
    return res.size();
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }
    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 64;
    cparams.n_batch = 64;
    cparams.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cparams);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token eog = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : llama_vocab_eot(vocab);
    if (eog == LLAMA_TOKEN_NULL || !llama_vocab_is_eog(vocab, eog)) {
        fprintf(stderr, "model has no EOG token\n");
        return 1;
    }
    llama_token t = 100;
    while (llama_vocab_is_eog(vocab, t)) {
        t++;
    }

    char name[128];
    for (loop_kind k : { LOOP_PLAIN, LOOP_GREEDY_FAST, LOOP_REJECTION }) {
        snprintf(name, sizeof(name), "%s {eog,eog}", kind_name(k));
        CHECK_EQ(run_accept(model, ctx, k, eog, { eog, eog }, { eog }), 1, name);
        snprintf(name, sizeof(name), "%s {eog,eog,eog}", kind_name(k));
        CHECK_EQ(run_accept(model, ctx, k, eog, { eog, eog, eog }, { eog }), 1, name);
        snprintf(name, sizeof(name), "%s {eog} (trailing EOG + bonus)", kind_name(k));
        CHECK_EQ(run_accept(model, ctx, k, eog, { eog }, { eog, eog }), 2, name);
        snprintf(name, sizeof(name), "%s {t,t} (non-EOG control)", kind_name(k));
        CHECK_EQ(run_accept(model, ctx, k, t, { t, t }, { t, t, t }), 3, name);
    }

    // 48-EOG draft tail (upstream #28232 ngram case): the first accepted EOG ends the round
    {
        const llama_tokens tail(48, eog);
        for (loop_kind k : { LOOP_PLAIN, LOOP_GREEDY_FAST, LOOP_REJECTION }) {
            snprintf(name, sizeof(name), "%s 48-EOG tail", kind_name(k));
            CHECK_EQ(run_accept(model, ctx, k, eog, tail, { eog }), 1, name);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "%s (%d failures)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
