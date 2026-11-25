// Q8-SERVED-DEV-INJECT-1: the device-resident DFlash2 KV injection (LLAMA_DFLASH_DEV_INJECT=1) must produce the
// same draft tokens, the same accepted counts and the same greedy output as the host round-trip path.
//
//   test-dflash-dev-inject -m <target.gguf> -md <dflash2-drafter.gguf> [speculative-simple args] \
//       [-p "prompt one|||prompt two|||..."] [-n 64]
//
// Takes the speculative-simple argument set. For every prompt the process runs the greedy np1 speculative loop
// twice from a fresh target context and a fresh draft context: once with LLAMA_DFLASH_DEV_INJECT unset (OFF) and
// once with it =1 (ON). It records every round's draft and accepted count plus the committed token stream.
// Exit 0 iff every prompt matches round for round AND the ON arm actually engaged dev inject on ctx_dft.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct run_trace {
    std::vector<llama_tokens> drafts;
    std::vector<int>          accepted;
    llama_tokens              output;
    bool                      engaged = false;
    double                    ms      = 0.0;
    bool                      ok      = false;
};

static run_trace run_once(common_params & params, llama_model * model_tgt, const std::string & prompt, bool on) {
    run_trace tr;
    if (on) {
        setenv("LLAMA_DFLASH_DEV_INJECT", "1", 1);
    } else {
        unsetenv("LLAMA_DFLASH_DEV_INJECT");
    }

    llama_context * ctx_tgt = llama_init_from_model(model_tgt, common_context_params_to_llama(params));
    if (!ctx_tgt) {
        fprintf(stderr, "FAIL: target context\n");
        return tr;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    common_params params_dft = common_base_params_to_speculative(params);
    common_speculative_init_result_ptr spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    if (!spec_init || !spec_init->context()) {
        fprintf(stderr, "FAIL: draft context\n");
        llama_free(ctx_tgt);
        return tr;
    }
    common_params_speculative pspec = params.speculative;
    pspec.draft.ctx_tgt = ctx_tgt;
    pspec.draft.ctx_dft = spec_init->context();
    llama_context * ctx_dft = pspec.draft.ctx_dft;

    const auto seq_rm_tgt   = common_context_can_seq_rm(ctx_tgt);
    const bool use_ckpt_tgt = seq_rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt, true, true);

    common_params_sampling sp = params.sampling;
    sp.temp = 0.0f; // greedy
    common_sampler_ptr smpl(common_sampler_init(model_tgt, sp));

    common_speculative * spec = common_speculative_init(pspec, 1);
    if (!spec) {
        fprintf(stderr, "FAIL: speculative init\n");
        llama_free(ctx_tgt);
        return tr;
    }
    tr.engaged = llama_get_dflash_dev_inject(ctx_dft);

    const llama_seq_id seq_id = 0;
    {
        llama_batch bp = llama_batch_init(inp.size(), 0, 1);
        for (size_t i = 0; i + 1 < inp.size(); ++i) {
            common_batch_add(bp, inp[i], i, { seq_id }, false);
        }
        llama_decode(ctx_tgt, bp);
        if (!common_speculative_process(spec, bp)) {
            fprintf(stderr, "FAIL: speculative prompt process\n");
        }
        llama_batch_free(bp);
    }

    llama_token  id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past    = inp.size() - 1;
    int n_predict = 0;
    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    llama_tokens draft;
    std::vector<common_speculative_token_dist> dists;
    common_prompt_checkpoint ckpt;

    const int64_t t0 = ggml_time_us();
    bool has_eos = false;
    while (true) {
        if (draft.empty()) {
            ckpt.update_pos(prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));
            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            n_draft_max = std::max(0, std::min(n_draft_max, params.n_predict - n_predict - 1));
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting        = */ true,
                /* .n_max           = */ n_draft_max,
                /* .n_max_ngram_mod = */ -1,
                /* .n_past          = */ n_past,
                /* .id_last         = */ id_last,
                /* .prompt          = */ &prompt_tgt,
                /* .result          = */ &draft,
                /* .dists           = */ &dists,
                /* .temperature     = */ 0.0f,
                /* .seed            = */ common_sampler_get_seed(smpl.get()),
            };
            common_speculative_draft(spec);
            tr.drafts.push_back(draft);
            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (use_ckpt_dft) {
                GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            }
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
        }

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
        }
        llama_decode(ctx_tgt, batch_tgt);
        if (!common_speculative_process(spec, batch_tgt)) {
            fprintf(stderr, "FAIL: speculative batch process\n");
            break;
        }

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        GGML_ASSERT(!ids.empty());

        if (use_ckpt_tgt && ids.size() - 1 < draft.size()) {
            tr.accepted.push_back(-(int) ids.size()); // partial: replayed from the checkpoint
            draft = std::move(ids);
            dists.clear();
            GGML_ASSERT(ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            prompt_tgt.resize(ckpt.n_tokens);
            smpl  = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);
        tr.accepted.push_back((int) ids.size() - 1);
        n_past    += ids.size() - 1;
        n_predict += ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            tr.output.push_back(id_last);
            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }
        }
        draft.clear();
        dists.clear();
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
        if (n_predict > params.n_predict || has_eos) {
            break;
        }
    }
    tr.ms = (ggml_time_us() - t0) / 1000.0;
    tr.ok = true;

    llama_batch_free(batch_tgt);
    common_speculative_free(spec);
    spec_init.reset();
    llama_free(ctx_tgt);
    return tr;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 2;
    }
    if (params.n_predict < 0) {
        params.n_predict = 64;
    }

    std::vector<std::string> prompts;
    if (params.prompt.empty()) {
        prompts = {
            "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
            "Explain in three sentences why the sky is blue.",
            "Write a short C function that reverses a singly linked list.",
            "List the first ten prime numbers and then sum them.",
        };
    } else {
        std::string s = params.prompt;
        size_t p;
        while ((p = s.find("|||")) != std::string::npos) {
            prompts.push_back(s.substr(0, p));
            s = s.substr(p + 3);
        }
        prompts.push_back(s);
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    auto init_tgt = common_init_from_params(params);
    llama_model * model_tgt = init_tgt->model();
    if (!model_tgt) {
        fprintf(stderr, "FAIL: target model load\n");
        return 1;
    }

    // LLAMA_TEST_INJECT_AA=1: A/A control - the second arm also runs with dev inject OFF, so any divergence is
    // drafter/target nondeterminism, not the inject path (diagnostic for the 20261125 devhandoff r3/r4 smokes)
    const char * aa_env = getenv("LLAMA_TEST_INJECT_AA");
    const bool aa = aa_env != nullptr && atoi(aa_env) != 0;
    fprintf(stderr, "mode: %s\n", aa ? "AA (OFF vs OFF)" : "AB (OFF vs ON)");
    // LLAMA_TEST_INJECT_SINGLE=off|on (+ LLAMA_TEST_INJECT_PROMPT=k): run ONE arm on ONE prompt as the process's first
    // context and print its full trace; the caller compares traces across processes. Isolates the inject path from the
    // in-process second-context nondeterminism seen in the 20261125 A/A diagnostic (GPU-27B-SECOND-CONTEXT-NONDET).
    if (const char * single = getenv("LLAMA_TEST_INJECT_SINGLE")) {
        const bool on1 = std::string(single) == "on";
        const char * pk = getenv("LLAMA_TEST_INJECT_PROMPT");
        const size_t k = pk ? (size_t) atoi(pk) : 0;
        if (k >= prompts.size()) {
            fprintf(stderr, "FAIL: prompt index %zu out of range\n", k);
            return 2;
        }
        const run_trace tr = run_once(params, model_tgt, prompts[k], on1);
        auto fmt = [](const llama_tokens & t) {
            std::string s;
            for (llama_token x : t) { s += std::to_string(x); s += ','; }
            return s;
        };
        std::string acc;
        for (int a : tr.accepted) { acc += std::to_string(a); acc += ','; }
        fprintf(stderr, "single: arm=%s prompt=%zu ok=%d engaged=%d rounds=%zu ms=%.1f\n", on1 ? "on" : "off", k,
                (int) tr.ok, (int) tr.engaged, tr.drafts.size(), tr.ms);
        for (size_t r = 0; r < tr.drafts.size(); ++r) {
            fprintf(stderr, "single-draft: %zu %s\n", r, fmt(tr.drafts[r]).c_str());
        }
        fprintf(stderr, "single-accepted: %s\n", acc.c_str());
        fprintf(stderr, "single-output: %s\n", fmt(tr.output).c_str());
        llama_backend_free();
        const bool good = tr.ok && tr.engaged == on1;
        fprintf(stderr, "%s\n", good ? "SINGLE-DONE" : "FAIL");
        return good ? 0 : 1;
    }
    int n_fail = 0;
    bool engaged_all = true;
    double ms_off = 0, ms_on = 0;
    size_t rounds_total = 0;
    for (size_t k = 0; k < prompts.size(); ++k) {
        const run_trace off = run_once(params, model_tgt, prompts[k], false);
        const run_trace on  = run_once(params, model_tgt, prompts[k], !aa);
        engaged_all = engaged_all && (aa ? !on.engaged : on.engaged) && !off.engaged;
        size_t n_acc_off = 0;
        for (int a : off.accepted) n_acc_off += a > 0 ? a : 0;
        const bool same = off.ok && on.ok && off.drafts == on.drafts && off.accepted == on.accepted && off.output == on.output;
        size_t first_div = 0;
        while (first_div < std::min(off.drafts.size(), on.drafts.size()) && off.drafts[first_div] == on.drafts[first_div]) {
            ++first_div;
        }
        fprintf(stderr, "prompt %zu: rounds off=%zu on=%zu  accepted_tokens off=%zu  out off=%zu on=%zu  "
                "ms off=%.1f on=%.1f  engaged on=%d off=%d  %s (first draft divergence at round %zu)\n",
                k, off.drafts.size(), on.drafts.size(), n_acc_off, off.output.size(), on.output.size(),
                off.ms, on.ms, (int) on.engaged, (int) off.engaged, same ? "MATCH" : "MISMATCH", first_div);
        // attribution: which of drafts / accepts / output differ, and where (first_div == rounds means none)
        size_t first_out = 0;
        while (first_out < std::min(off.output.size(), on.output.size()) && off.output[first_out] == on.output[first_out]) {
            ++first_out;
        }
        size_t first_acc = 0;
        while (first_acc < std::min(off.accepted.size(), on.accepted.size()) && off.accepted[first_acc] == on.accepted[first_acc]) {
            ++first_acc;
        }
        fprintf(stderr, "prompt %zu detail: drafts_eq=%d accepted_eq=%d output_eq=%d first_accept_div=%zu first_output_div=%zu\n",
                k, (int) (off.drafts == on.drafts), (int) (off.accepted == on.accepted), (int) (off.output == on.output),
                first_acc, first_out);
        if (first_div < std::min(off.drafts.size(), on.drafts.size())) {
            auto fmt = [](const llama_tokens & t) {
                std::string s;
                for (llama_token x : t) { s += std::to_string(x); s += ' '; }
                return s;
            };
            fprintf(stderr, "prompt %zu round %zu draft off: [ %s] accepted=%d\n", k, first_div,
                    fmt(off.drafts[first_div]).c_str(), first_div < off.accepted.size() ? off.accepted[first_div] : 0);
            fprintf(stderr, "prompt %zu round %zu draft on:  [ %s] accepted=%d\n", k, first_div,
                    fmt(on.drafts[first_div]).c_str(), first_div < on.accepted.size() ? on.accepted[first_div] : 0);
        }
        n_fail += same ? 0 : 1;
        ms_off += off.ms;
        ms_on  += on.ms;
        rounds_total += off.drafts.size();
    }
    fprintf(stderr, "summary: prompts=%zu mismatches=%d engaged=%d rounds=%zu ms/round off=%.3f on=%.3f\n",
            prompts.size(), n_fail, (int) engaged_all, rounds_total,
            rounds_total ? ms_off / rounds_total : 0.0, rounds_total ? ms_on / rounds_total : 0.0);

    llama_backend_free();
    if (n_fail == 0 && engaged_all) {
        fprintf(stderr, "PASS\n");
        return 0;
    }
    fprintf(stderr, "FAIL\n");
    return 1;
}
