// AK-GAP-1-SAMPLE: the device greedy accept (LLAMA_DEVICE_ACCEPT=1) must return the same tokens and the same
// accepted counts as the host path that scans the downloaded logits.
//
//   test-device-accept <model.gguf> [n_gen=64] [n_draft=8] [ngl=0]
//
// The process sets LLAMA_DEVICE_ACCEPT=1 itself. For every step it first runs the device path (which must not
// touch the logits), then reads the logits (forcing the deferred download) and recomputes the host answer:
//   plain decode : common_sampler_sample_device_greedy  vs  first-max argmax over llama_get_logits_ith
//   chain verify : common_sampler_sample_and_accept_n_greedy_fast  vs  the host accept loop over the same rows
// Drafts are the host-greedy continuation with one token corrupted at a step-dependent position, so the
// accepted count covers 0..n_draft.
//   server-style : the gate request (temp 0, top_k 1, ignore_eos EOG bans) on the device path alone must not
//                  flush the deferred logits (except on a banned device pick) and must match the host scan.
// Exit 0 iff every step matches and the deferred path actually engaged.

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static llama_token host_argmax(llama_context * ctx, int idx, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, idx);
    int best = 0;
    for (int t = 1; t < n_vocab; ++t) {
        if (l[t] > l[best]) {
            best = t;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [n_gen] [n_draft] [ngl]\n", argv[0]);
        return 2;
    }
    setenv("LLAMA_DEVICE_ACCEPT", "1", 1);

    const int n_gen   = argc > 2 ? atoi(argv[2]) : 64;
    const int n_draft = argc > 3 ? atoi(argv[3]) : 8;
    const int ngl     = argc > 4 ? atoi(argv[4]) : 0;

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "FAIL: model load\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 4096;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.n_seq_max = 2; // seq 1 = scratch sequence for building the drafts
    llama_context * ctx = llama_init_from_model(model, cparams);

    common_params_sampling sp;
    sp.temp = 0.0f;
    common_sampler * smpl = common_sampler_init(model, sp);

    const char * prompt = "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n";
    std::vector<llama_token> toks(512);
    const int n_p = llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), toks.size(), true, false);
    toks.resize(n_p);

    llama_batch batch = llama_batch_init(512, 0, 1);
    common_batch_clear(batch);
    for (int i = 0; i < n_p; ++i) {
        common_batch_add(batch, toks[i], i, {0}, i == n_p - 1);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "FAIL: prompt decode\n");
        return 1;
    }

    int fails = 0;
    int n_pos = n_p;

    // phase 1: plain greedy decode
    std::vector<llama_token> gen;
    int idx = n_p - 1;
    for (int s = 0; s < n_gen; ++s) {
        const llama_token d = common_sampler_sample_device_greedy(smpl, ctx, idx);
        const llama_token h = host_argmax(ctx, idx, n_vocab);
        if (d != h) {
            fprintf(stderr, "decode step %d: device %d != host %d\n", s, d, h);
            fails++;
        }
        common_sampler_accept(smpl, h, true);
        gen.push_back(h);
        common_batch_clear(batch);
        common_batch_add(batch, h, n_pos++, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "FAIL: decode\n");
            return 1;
        }
        idx = 0;
    }
    printf("plain decode: %d steps, %d mismatches\n", n_gen, fails);

    // phase 2: chain verify. last = token whose logits are pending at idx 0 (sampled but not yet decoded)
    llama_token last = common_sampler_sample(smpl, ctx, 0, false);
    common_sampler_accept(smpl, last, true);

    int vfails = 0;
    int hist[64] = {0};
    const int n_rounds = 24;
    for (int r = 0; r < n_rounds; ++r) {
        // host-greedy continuation of `last` for n_draft tokens, built on a scratch sequence 1
        llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, -1, -1);
        llama_tokens draft;
        llama_token cur = last;
        int p = n_pos;
        for (int k = 0; k < n_draft; ++k) {
            common_batch_clear(batch);
            common_batch_add(batch, cur, p++, {1}, true);
            llama_decode(ctx, batch);
            cur = host_argmax(ctx, 0, n_vocab);
            draft.push_back(cur);
        }
        llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);
        const int bad = r % (n_draft + 1); // == n_draft -> no corruption, full accept
        if (bad < n_draft) {
            draft[bad] = (draft[bad] + 1) % n_vocab;
        }

        // verify batch: [last, draft...] all with outputs, on seq 0
        common_batch_clear(batch);
        common_batch_add(batch, last, n_pos, {0}, true);
        for (int k = 0; k < n_draft; ++k) {
            common_batch_add(batch, draft[k], n_pos + 1 + k, {0}, true);
        }
        llama_decode(ctx, batch);
        std::vector<int> idxs(n_draft + 1);
        for (int k = 0; k <= n_draft; ++k) {
            idxs[k] = k;
        }

        common_sampler * s_dev = common_sampler_clone(smpl);
        int64_t n_fast = 0, n_fb = 0;
        const auto dev = common_sampler_sample_and_accept_n_greedy_fast(s_dev, ctx, idxs, draft, &n_fast, &n_fb);
        common_sampler_free(s_dev);

        std::vector<llama_token> host;
        for (int k = 0; k <= n_draft; ++k) {
            const llama_token id = host_argmax(ctx, k, n_vocab);
            host.push_back(id);
            if (k == n_draft || draft[k] != id) {
                break;
            }
        }
        if (dev != host || n_fb != 0) {
            fprintf(stderr, "verify round %d: device n=%zu host n=%zu fast=%lld fallback=%lld\n", r, dev.size(), host.size(),
                    (long long) n_fast, (long long) n_fb);
            vfails++;
        }
        hist[host.size() - 1]++;

        for (auto t : host) {
            common_sampler_accept(smpl, t, true);
        }
        // keep the accepted prefix in the KV: last + accepted drafts; the bonus becomes the next `last`
        const int n_acc = (int) host.size() - 1;
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_pos + 1 + n_acc, -1);
        n_pos += 1 + n_acc;
        last = host.back();
    }
    printf("chain verify: %d rounds, %d mismatches; accepted-count histogram:", n_rounds, vfails);
    for (int k = 0; k <= n_draft; ++k) {
        printf(" %d:%d", k, hist[k]);
    }
    printf("\n");


    // phase 3: server-style plain decode (AK-GAP-1-SAMPLE fix 1). The server's /completion request of the gate
    // (temperature 0, top_k 1, default chain incl. penalty_last_n 64, ignore_eos -> -inf bias on every EOG token)
    // must take the device row WITHOUT reading the logits: 3a decodes with the device path only and asserts zero
    // deferred-download flushes, 3b replays the same steps on the host (first max over non-banned tokens) and
    // must give the same tokens. One extra banned token (the 6th phase-1 token) forces the banned-pick fallback.
    common_params_sampling sps;
    sps.temp  = 0.0f;
    sps.top_k = 1;
    for (int t = 0; t < n_vocab; ++t) {
        if (llama_vocab_is_eog(vocab, t)) {
            sps.logit_bias.push_back({t, -INFINITY});
        }
    }
    const llama_token extra_ban = gen.size() > 5 ? gen[5] : gen.back();
    sps.logit_bias.push_back({extra_ban, -INFINITY});
    auto banned = [&](llama_token t) {
        for (const auto & lb : sps.logit_bias) {
            if (lb.token == t) {
                return true;
            }
        }
        return false;
    };
    auto run_prompt = [&]() {
        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch_clear(batch);
        for (int i = 0; i < n_p; ++i) {
            common_batch_add(batch, toks[i], i, {0}, i == n_p - 1);
        }
        return llama_decode(ctx, batch) == 0;
    };
    std::vector<llama_token> dev_seq, host_seq;
    int64_t sk0 = 0, fl0 = 0, sk1 = 0, fl1 = 0;
    int n_banned_hits = 0;
    {
        common_sampler * s3 = common_sampler_init(model, sps);
        if (!run_prompt()) {
            fprintf(stderr, "FAIL: prompt decode (3a)\n");
            return 1;
        }
        llama_device_accept_stats(ctx, &sk0, &fl0);
        int ix = n_p - 1;
        for (int s = 0; s < n_gen; ++s) {
            if (banned(llama_get_argmax_ith(ctx, ix))) {
                n_banned_hits++; // this step legitimately falls back to the host scan (one flush)
            }
            const llama_token d = common_sampler_sample_device_greedy(s3, ctx, ix);
            common_sampler_accept(s3, d, true);
            dev_seq.push_back(d);
            common_batch_clear(batch);
            common_batch_add(batch, d, n_p + s, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "FAIL: decode (3a)\n");
                return 1;
            }
            ix = 0;
        }
        llama_device_accept_stats(ctx, &sk1, &fl1);
        common_sampler_free(s3);
    }
    {
        if (!run_prompt()) {
            fprintf(stderr, "FAIL: prompt decode (3b)\n");
            return 1;
        }
        int ix = n_p - 1;
        for (int s = 0; s < n_gen; ++s) {
            const float * l = llama_get_logits_ith(ctx, ix);
            int best = -1;
            for (int t = 0; t < n_vocab; ++t) {
                if (!banned(t) && (best < 0 || l[t] > l[best])) {
                    best = t;
                }
            }
            host_seq.push_back(best);
            common_batch_clear(batch);
            common_batch_add(batch, dev_seq[s], n_p + s, {0}, true); // teacher-forced on the device tokens
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "FAIL: decode (3b)\n");
                return 1;
            }
            ix = 0;
        }
    }
    int sfails = 0;
    for (int s = 0; s < n_gen; ++s) {
        if (dev_seq[s] != host_seq[s]) {
            fprintf(stderr, "server-style step %d: device %d != host %d\n", s, dev_seq[s], host_seq[s]);
            sfails++;
        }
    }
    const int64_t flushed_3a = fl1 - fl0;
    const int64_t skipped_3a = sk1 - sk0;
    printf("server-style decode: %d steps, %d mismatches, lazy_skipped=%lld flushed=%lld (banned-pick fallbacks %d)\n",
           n_gen, sfails, (long long) skipped_3a, (long long) flushed_3a, n_banned_hits);
    if (skipped_3a < n_gen || flushed_3a != n_banned_hits) {
        fprintf(stderr, "server-style decode: deferred download not honoured (expected flushed == %d)\n", n_banned_hits);
        sfails++;
    }
    fails += sfails;

    common_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx); // prints the llama_device_accept witness with the lazy_skipped / flushed counts
    llama_model_free(model);
    llama_backend_free();

    if (fails || vfails) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
