// AK-GAP-1-STOCH-ACCEPT: the device top-K candidate path (GGML_HIP_DEVICE_ACCEPT_STOCH=K) must keep the target
// sampling distribution of a temp > 0 chain exactly.
//
//   test-device-accept-stoch <model.gguf> [threads=4] [ngl=0] [draws=20000]
//
// The process sets GGML_HIP_DEVICE_ACCEPT_STOCH=32 itself (before any graph is built). One llama_decode carries
// 4 sequences (np4, kv_unified = false) with 8 output rows each (a 7-token draft + bonus row, as in a verify batch)
// at n_ubatch = 16, so the outputs span several ubatches (C17-FIX-LAZY multi-ubatch shape) and are reordered.
//   1. candidates : every output row's device top-K ids/logits equal the true top-K of the downloaded logits row
//                   (logit values bit-equal to logits[id], min candidate >= max non-candidate).
//   2. identity   : two samplers with the same seed and the same chain (temp 1, top_k 20, top_p 0.95, min_p 0.05),
//                   A eligible for the device path, B forced to the full-row path (n_probs = 1, which does not change
//                   the chain), draw `draws` times per row; the token streams must be identical.
//   3. law        : Pearson chi-square of A's counts on one row against the exact chain probabilities read from B's
//                   candidates; p-value (Wilson-Hilferty) must exceed 1e-5 (the SPEC-PIT.1 threshold).
//   4. witness    : the device path engaged (A's draws were served from candidates: the logits read in step 1 happens
//                   after them, so the full-row download cannot have been used by A).
// Exit 0 iff all pass.

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

static const char * PROMPTS[4] = {
    "The history of the printing press begins in",
    "To compute the eigenvalues of a symmetric matrix one",
    "def fibonacci(n):\n    if n < 2:\n        return",
    "Il pleuvait sur la ville quand",
};

static double chi2_pvalue(double x, int dof) {   // Wilson-Hilferty upper tail
    if (dof <= 0) {
        return 1.0;
    }
    const double k = dof;
    const double z = (std::cbrt(x / k) - (1.0 - 2.0 / (9.0 * k))) / std::sqrt(2.0 / (9.0 * k));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [threads] [ngl] [draws]\n", argv[0]);
        return 2;
    }
    setenv("GGML_HIP_DEVICE_ACCEPT_STOCH", getenv("STOCH_TEST_K") ? getenv("STOCH_TEST_K") : "32", 1);
    const int nth   = argc > 2 ? atoi(argv[2]) : 4;
    const int ngl   = argc > 3 ? atoi(argv[3]) : 0;
    const int draws = argc > 4 ? atoi(argv[4]) : 20000;
    const int K     = atoi(getenv("GGML_HIP_DEVICE_ACCEPT_STOCH")) == 1 ? 20 : atoi(getenv("GGML_HIP_DEVICE_ACCEPT_STOCH"));
    const int NOUT  = 8;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "FAIL: model load\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 256; cp.n_ubatch = 16; cp.n_seq_max = 4; cp.kv_unified = false;
    cp.n_threads = nth; cp.n_threads_batch = nth;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "FAIL: context\n");
        return 1;
    }

    llama_batch b = llama_batch_init(256, 0, 1);
    std::vector<int> rows;   // batch indices with logits
    for (int s = 0; s < 4; ++s) {
        std::vector<llama_token> t(128);
        int n = llama_tokenize(vocab, PROMPTS[s], (int) strlen(PROMPTS[s]), t.data(), (int) t.size(), true, false);
        t.resize(std::max(n, 0));
        while ((int) t.size() < NOUT + 2) {
            t.push_back(t.back());
        }
        for (size_t i = 0; i < t.size(); ++i) {
            const int k = b.n_tokens++;
            b.token[k] = t[i]; b.pos[k] = (llama_pos) i; b.n_seq_id[k] = 1; b.seq_id[k][0] = s;
            b.logits[k] = i + NOUT >= t.size();
            if (b.logits[k]) {
                rows.push_back(k);
            }
        }
    }
    if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "FAIL: decode\n");
        return 1;
    }

    common_params_sampling sp;
    sp.temp = 1.0f; sp.top_k = 20; sp.top_p = 0.95f; sp.min_p = 0.05f; sp.seed = 1234; sp.n_probs = 0;
    common_params_sampling sp_full = sp;
    sp_full.n_probs = 1;   // ineligible for the device path; the chain is unchanged

    bool ok = true;
    int  n_ident_rows = 0;
    long n_mismatch   = 0;
    int  chi_row      = -1;
    double best_chi_p = 1.0;

    // 2 + 3: identity and law (device path first, before anything reads the logits)
    std::vector<std::map<llama_token, long>> counts(rows.size());
    std::vector<std::vector<std::pair<llama_token, float>>> exact(rows.size());
    for (size_t r = 0; r < rows.size(); ++r) {
        common_sampler * A = common_sampler_init(model, sp);
        common_sampler * B = common_sampler_init(model, sp_full);
        for (int d = 0; d < draws; ++d) {
            const llama_token ta = common_sampler_sample(A, ctx, rows[r], false);
            const llama_token tb = common_sampler_sample(B, ctx, rows[r], false);
            counts[r][ta]++;
            if (ta != tb) {
                n_mismatch++;
            }
            if (d == 0) {
                const auto * cp_b = common_sampler_get_candidates(B, true);
                for (size_t i = 0; i < cp_b->size; ++i) {
                    exact[r].emplace_back(cp_b->data[i].id, cp_b->data[i].p);
                }
            }
        }
        n_ident_rows++;
        common_sampler_free(A);
        common_sampler_free(B);
    }
    if (n_mismatch != 0) {
        fprintf(stderr, "FAIL identity: %ld of %ld draws differ between device-candidate and full-row paths\n",
                n_mismatch, (long) draws * (long) rows.size());
        ok = false;
    }

    // law: chi-square on every row with >= 2 support tokens; report the worst
    double worst_p = 1.0;
    for (size_t r = 0; r < rows.size(); ++r) {
        double ps = 0.0;
        for (auto & e : exact[r]) {
            ps += e.second;
        }
        int dof = -1;
        double chi = 0.0;
        long seen = 0;
        for (auto & e : exact[r]) {
            const double expct = draws * e.second / ps;
            const long   obs   = counts[r].count(e.first) ? counts[r][e.first] : 0;
            seen += obs;
            if (expct > 0) {
                chi += (obs - expct) * (obs - expct) / expct;
                dof++;
            }
        }
        if (seen != draws) {
            fprintf(stderr, "FAIL law: row %zu drew %ld tokens outside the exact support\n", r, (long) draws - seen);
            ok = false;
        }
        if (dof >= 1) {
            const double p = chi2_pvalue(chi, dof);
            if (p < worst_p) {
                worst_p = p; chi_row = (int) r; best_chi_p = p;
            }
        }
    }
    if (chi_row >= 0 && worst_p <= 1e-5) {
        fprintf(stderr, "FAIL law: chi-square p = %.3g on row %d\n", worst_p, chi_row);
        ok = false;
    }

    // 4: composition with SPEC-RNG.1 (rng_poskey, used by the V2 RRS tree walk). The poskey dist and the served
    // rejection step draw from (seed, absolute position), so the candidate path must give the same tokens, the same
    // accept decisions and the same residual resamples as the full-row path while the position advances.
    long n_pk_mismatch = 0;
    long n_pk_steps    = 0;
    {
        common_params_sampling spk = sp;           spk.rng_poskey = true;
        common_params_sampling spk_full = sp_full; spk_full.rng_poskey = true;
        for (size_t r = 0; r < rows.size(); ++r) {
            common_sampler * A = common_sampler_init(model, spk);
            common_sampler * B = common_sampler_init(model, spk_full);
            // a fixed draft distribution: uniform over the row's first 4 device candidates
            const int32_t * ids = nullptr;
            const float *   vals = nullptr;
            const int32_t   n = llama_get_topk_ith(ctx, rows[r], &ids, &vals);
            common_speculative_token_dist q;
            for (int i = 0; i < std::min<int32_t>(n, 4); ++i) {
                q.ids.push_back(ids[i]);
                q.probs.push_back(1.0f / std::min<int32_t>(n, 4));
            }
            for (int d = 0; d < 2000 && n > 0; ++d) {
                const llama_token ta = common_sampler_sample(A, ctx, rows[r], false);
                const llama_token tb = common_sampler_sample(B, ctx, rows[r], false);
                bool acc_a = false, acc_b = false;
                const llama_token draft_tok = q.ids[d % q.ids.size()];
                const llama_token ya = common_sampler_spec_reject_step(A, ctx, rows[r], draft_tok, q, &acc_a);
                const llama_token yb = common_sampler_spec_reject_step(B, ctx, rows[r], draft_tok, q, &acc_b);
                if (ta != tb || ya != yb || acc_a != acc_b) {
                    n_pk_mismatch++;
                }
                n_pk_steps++;
                common_sampler_accept(A, ya, true);   // advance the absolute position on both
                common_sampler_accept(B, yb, true);
            }
            common_sampler_free(A);
            common_sampler_free(B);
        }
    }
    if (n_pk_mismatch != 0) {
        fprintf(stderr, "FAIL poskey: %ld of %ld steps differ between device-candidate and full-row paths\n",
                n_pk_mismatch, n_pk_steps);
        ok = false;
    }
    printf("poskey+reject_step steps=%ld mismatches=%ld\n", n_pk_steps, n_pk_mismatch);

    // 1: candidates vs the downloaded logits (forces the deferred download)
    int n_cand_rows = 0;
    for (size_t r = 0; r < rows.size(); ++r) {
        const int32_t * ids = nullptr;
        const float *   vals = nullptr;
        const int32_t n = llama_get_topk_ith(ctx, rows[r], &ids, &vals);
        const float * lg = llama_get_logits_ith(ctx, rows[r]);
        if (n != K || lg == nullptr) {
            fprintf(stderr, "FAIL candidates: row %zu has %d candidates (want %d)\n", r, n, K);
            ok = false;
            continue;
        }
        std::vector<char> in(n_vocab, 0);
        float mn = INFINITY;
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0 || ids[i] >= n_vocab || in[ids[i]] || vals[i] != lg[ids[i]]) {
                fprintf(stderr, "FAIL candidates: row %zu cand %d id %d val %.7g logit %.7g\n", r, i, ids[i], vals[i],
                        (ids[i] >= 0 && ids[i] < n_vocab) ? lg[ids[i]] : NAN);
                ok = false;
                break;
            }
            in[ids[i]] = 1;
            mn = std::min(mn, vals[i]);
        }
        float mx_out = -INFINITY;
        for (int t = 0; t < n_vocab; ++t) {
            if (!in[t]) {
                mx_out = std::max(mx_out, lg[t]);
            }
        }
        if (mx_out > mn) {
            fprintf(stderr, "FAIL candidates: row %zu non-candidate logit %.7g > min candidate %.7g\n", r, mx_out, mn);
            ok = false;
        }
        n_cand_rows++;
    }

    printf("rows=%zu (np4, n_ubatch=16) candidate_rows_ok=%d identity_rows=%d draws/row=%d mismatches=%ld "
           "worst_chi2_p=%.4g (row %d)\n", rows.size(), n_cand_rows, n_ident_rows, draws, n_mismatch, best_chi_p, chi_row);
    printf("%s\n", ok ? "PASS" : "FAIL");

    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
