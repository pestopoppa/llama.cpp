// DTREE-STOCH-VERIFY: common_sampler_sample_and_accept_tree must emit tokens with the target sampling law.
//
//   test-dtree-stoch <model.gguf> [threads=4] [walks=100000]
//
// One llama_decode builds a real two-level token tree from the model's own logits:
//   s0 = prompt             -> row r0   (anchor)
//   s1 = prompt c1          -> row r1   c1, c2 = top-2 tokens of the chain at r0
//   s2 = prompt c1 a        -> row r1a  a, b   = top-2 tokens of the chain at r1
//   s3 = prompt c1 b        -> row r1b
//   s4 = prompt c2          -> row r2
//   s5 = prompt c2 e        -> row r2e  e      = top-1 token of the chain at r2
// paths (main first): [c1 a] rows [r0 r1 r1a], [c1 b] rows [r0 r1 r1b], [c2 e] rows [r0 r2 r2e]
// so the walk meets 2 siblings at depth 0 and 2 siblings under c1 at depth 1 (residual without replacement twice).
//   1. law (temp 1, top_k 20, top_p 0.95): Pearson chi-square, p > 1e-5 (SPEC-PIT.1 threshold), of
//        t1 against p(r0); t2 | t1 = c1 against p(r1); t3 | t1 = c1, t2 = a against p(r1a);
//        t2 | t1 = c2 against p(r2). The emitted length must be 1 + [t1 child] + [t2 child].
//   2. reproducibility: two samplers with the same seed produce identical walk streams.
//   3. greedy (temp 0): the walk equals the existing tree path bitwise -- winner by greedy longest match
//      (common_dtree_greedy_accept_len, lowest index on ties) then common_sampler_sample_and_accept_n on its rows --
//      for the fixture paths and for every permutation of path order.
// GGML_HIP_DEVICE_ACCEPT_STOCH=32 is set so the rows are served from the device top-K candidates (the production
// configuration of the stochastic accept); STOCH_TEST_K overrides it (0 = full-row path).
// Exit 0 iff all pass.

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative-tree.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

static double chi2_pvalue(double x, int dof) {   // Wilson-Hilferty upper tail
    if (dof <= 0) {
        return 1.0;
    }
    const double k = dof;
    const double z = (std::cbrt(x / k) - (1.0 - 2.0 / (9.0 * k))) / std::sqrt(2.0 / (9.0 * k));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

// exact post-chain law of a row (one fresh sampler, one draw, read its candidates)
static std::vector<std::pair<llama_token, double>> row_law(llama_model * model, llama_context * ctx,
                                                           const common_params_sampling & sp, int row) {
    common_params_sampling q = sp;
    q.n_probs = 1;   // full-row path; the chain is unchanged
    common_sampler * s = common_sampler_init(model, q);
    common_sampler_sample(s, ctx, row, false);
    const auto * c = common_sampler_get_candidates(s, true);
    std::vector<std::pair<llama_token, double>> out;
    double sum = 0.0;
    for (size_t i = 0; i < c->size; ++i) {
        if (c->data[i].p > 0.0f) {
            out.emplace_back(c->data[i].id, c->data[i].p);
            sum += c->data[i].p;
        }
    }
    for (auto & e : out) {
        e.second /= sum;
    }
    common_sampler_free(s);
    return out;
}

static bool chi_check(const char * name, const std::vector<std::pair<llama_token, double>> & law,
                      const std::map<llama_token, long> & cnt, double * p_out) {
    long n = 0;
    for (auto & e : cnt) {
        n += e.second;
    }
    double chi = 0.0;
    int dof = -1;
    long seen = 0;
    for (auto & e : law) {
        const auto it = cnt.find(e.first);
        const long obs = it == cnt.end() ? 0 : it->second;
        const double ex = n * e.second;
        seen += obs;
        chi += (obs - ex) * (obs - ex) / ex;
        dof++;
    }
    const double p = chi2_pvalue(chi, dof);
    *p_out = p;
    const bool ok = seen == n && (dof < 1 || p > 1e-5);
    printf("  %-22s n=%7ld support=%3zu chi2=%9.2f dof=%3d p=%.4g outside=%ld %s\n", name, n, law.size(), chi, dof, p,
           n - seen, ok ? "ok" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [threads] [walks]\n", argv[0]);
        return 2;
    }
    setenv("GGML_HIP_DEVICE_ACCEPT_STOCH", getenv("STOCH_TEST_K") ? getenv("STOCH_TEST_K") : "32", 1);
    const int nth   = argc > 2 ? atoi(argv[2]) : 4;
    const int walks = argc > 3 ? atoi(argv[3]) : 100000;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "FAIL: model load\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 256; cp.n_ubatch = 256; cp.n_seq_max = 6; cp.kv_unified = false;
    cp.n_threads = nth; cp.n_threads_batch = nth;

    const char * prompt = "Once upon a time, in a small village by the sea, there lived";
    std::vector<llama_token> pt(128);
    pt.resize(std::max(0, llama_tokenize(vocab, prompt, (int) strlen(prompt), pt.data(), (int) pt.size(), true, false)));

    common_params_sampling sp;
    sp.temp = 1.0f; sp.top_k = 20; sp.top_p = 0.95f; sp.min_p = 0.0f; sp.seed = 4321; sp.n_probs = 0;

    // stage A: decode the anchor and the depth-1 prefixes to choose the children, stage B: the full tree.
    // Each stage is its own context so that row indices are those of one decode.
    auto decode_seqs = [&](llama_context * ctx, const std::vector<std::vector<llama_token>> & seqs, std::vector<int> & rows) {
        llama_batch b = llama_batch_init(512, 0, 1);
        rows.clear();
        for (size_t s = 0; s < seqs.size(); ++s) {
            for (size_t i = 0; i < seqs[s].size(); ++i) {
                const int k = b.n_tokens++;
                b.token[k] = seqs[s][i]; b.pos[k] = (llama_pos) i; b.n_seq_id[k] = 1; b.seq_id[k][0] = (llama_seq_id) s;
                b.logits[k] = i + 1 == seqs[s].size();
                if (b.logits[k]) {
                    rows.push_back(k);
                }
            }
        }
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        return rc == 0;
    };
    auto top_tokens = [&](const std::vector<std::pair<llama_token, double>> & law, int n) {
        std::vector<llama_token> t;
        for (int i = 0; i < n && i < (int) law.size(); ++i) {
            t.push_back(law[i].first);
        }
        return t;
    };

    llama_token c1, c2, a, b, e;
    {
        llama_context * ctx = llama_init_from_model(model, cp);
        std::vector<int> rows;
        if (!ctx || !decode_seqs(ctx, { pt }, rows)) {
            fprintf(stderr, "FAIL: stage A decode\n");
            return 1;
        }
        auto t0 = top_tokens(row_law(model, ctx, sp, rows[0]), 2);
        llama_free(ctx);
        if (t0.size() < 2) {
            fprintf(stderr, "FAIL: anchor support < 2\n");
            return 1;
        }
        c1 = t0[0]; c2 = t0[1];
        ctx = llama_init_from_model(model, cp);
        auto p1 = pt; p1.push_back(c1);
        auto p2 = pt; p2.push_back(c2);
        if (!decode_seqs(ctx, { p1, p2 }, rows)) {
            fprintf(stderr, "FAIL: stage A2 decode\n");
            return 1;
        }
        auto t1 = top_tokens(row_law(model, ctx, sp, rows[0]), 2);
        auto t2 = top_tokens(row_law(model, ctx, sp, rows[1]), 1);
        llama_free(ctx);
        if (t1.size() < 2 || t2.empty()) {
            fprintf(stderr, "FAIL: child support too small\n");
            return 1;
        }
        a = t1[0]; b = t1[1]; e = t2[0];
    }

    llama_context * ctx = llama_init_from_model(model, cp);
    std::vector<int> R;
    {
        std::vector<std::vector<llama_token>> seqs(6, pt);
        seqs[1].push_back(c1);
        seqs[2].push_back(c1); seqs[2].push_back(a);
        seqs[3].push_back(c1); seqs[3].push_back(b);
        seqs[4].push_back(c2);
        seqs[5].push_back(c2); seqs[5].push_back(e);
        if (!ctx || !decode_seqs(ctx, seqs, R)) {
            fprintf(stderr, "FAIL: tree decode\n");
            return 1;
        }
    }
    const int r0 = R[0], r1 = R[1], r1a = R[2], r1b = R[3], r2 = R[4], r2e = R[5];
    (void) r1b; (void) r2e;
    const std::vector<std::vector<int>> rows   = { { r0, r1, r1a }, { r0, r1, r1b }, { r0, r2, r2e } };
    const std::vector<llama_tokens>     drafts = { { c1, a }, { c1, b }, { c2, e } };

    printf("tree: c1=%d c2=%d | a=%d b=%d | e=%d ; walks=%d K=%s\n", c1, c2, a, b, e, walks,
           getenv("GGML_HIP_DEVICE_ACCEPT_STOCH"));

    bool ok = true;

    // 1 + 2: law and reproducibility
    {
        const auto L0  = row_law(model, ctx, sp, r0);
        const auto L1  = row_law(model, ctx, sp, r1);
        const auto L1a = row_law(model, ctx, sp, r1a);
        const auto L2  = row_law(model, ctx, sp, r2);

        std::map<llama_token, long> n0, n1, n1a, n2;
        long n_len_bad = 0, n_repro_bad = 0;
        long n_acc[3] = { 0, 0, 0 };
        common_sampler * S  = common_sampler_init(model, sp);
        common_sampler * S2 = common_sampler_init(model, sp);
        for (int w = 0; w < walks; ++w) {
            int32_t win = -1, win2 = -1;
            const auto res  = common_sampler_sample_and_accept_tree(S,  ctx, rows, drafts, &win);
            const auto res2 = common_sampler_sample_and_accept_tree(S2, ctx, rows, drafts, &win2);
            if (res != res2 || win != win2) {
                n_repro_bad++;
            }
            n_acc[std::min<size_t>(res.size() - 1, 2)]++;
            const llama_token t1 = res[0];
            n0[t1]++;
            const bool t1c = t1 == c1 || t1 == c2;
            size_t want = 1;
            if (t1c) {
                want = 2;
                const llama_token t2 = res.size() > 1 ? res[1] : LLAMA_TOKEN_NULL;
                if (t1 == c1) {
                    n1[t2]++;
                    if (t2 == a || t2 == b) {
                        want = 3;
                        if (t2 == a && res.size() > 2) {
                            n1a[res[2]]++;
                        }
                    }
                } else {
                    n2[t2]++;
                    if (t2 == e) {
                        want = 3;
                    }
                }
            }
            if (res.size() != want) {
                n_len_bad++;
            }
        }
        common_sampler_free(S);
        common_sampler_free(S2);

        printf("law (temp %.1f top_k %d top_p %.2f): accepted drafts 0/1/2 = %ld/%ld/%ld\n", sp.temp, sp.top_k, sp.top_p,
               n_acc[0], n_acc[1], n_acc[2]);
        double p;
        ok &= chi_check("t1 ~ p(r0)", L0, n0, &p);
        ok &= chi_check("t2 | c1 ~ p(r1)", L1, n1, &p);
        ok &= chi_check("t3 | c1 a ~ p(r1a)", L1a, n1a, &p);
        ok &= chi_check("t2 | c2 ~ p(r2)", L2, n2, &p);
        printf("  emitted-length violations=%ld  same-seed stream mismatches=%ld\n", n_len_bad, n_repro_bad);
        ok &= n_len_bad == 0 && n_repro_bad == 0;
    }

    // 3: greedy equals the existing tree path, bitwise, for every path order
    {
        common_params_sampling g = sp;
        g.temp = 0.0f;
        std::vector<int> perm = { 0, 1, 2 };
        long n_bad = 0, n_cases = 0;
        do {
            std::vector<std::vector<int>> pr;
            std::vector<llama_tokens>     pd;
            for (int i : perm) {
                pr.push_back(rows[i]);
                pd.push_back(drafts[i]);
            }
            common_sampler * A = common_sampler_init(model, g);
            common_sampler * B = common_sampler_init(model, g);
            int32_t win = -1;
            const auto ra = common_sampler_sample_and_accept_tree(A, ctx, pr, pd, &win);
            int32_t best = 0;
            int32_t best_n = common_dtree_greedy_accept_len(ctx, pr[0], pd[0]);
            for (size_t p = 1; p < pr.size(); ++p) {
                const int32_t n = common_dtree_greedy_accept_len(ctx, pr[p], pd[p]);
                if (n > best_n) {
                    best = (int32_t) p;
                    best_n = n;
                }
            }
            const auto rb = common_sampler_sample_and_accept_n(B, ctx, pr[best], pd[best], false, nullptr);
            n_cases++;
            if (ra != rb || win != best) {
                n_bad++;
                printf("  greedy mismatch perm %d%d%d: tree win %d len %zu, existing win %d len %zu\n",
                       perm[0], perm[1], perm[2], win, ra.size(), best, rb.size());
            }
            common_sampler_free(A);
            common_sampler_free(B);
        } while (std::next_permutation(perm.begin(), perm.end()));
        printf("greedy (temp 0): %ld/%ld path orders bitwise equal to the existing tree path\n", n_cases - n_bad, n_cases);
        ok &= n_bad == 0;
    }

    printf("%s\n", ok ? "PASS" : "FAIL");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
