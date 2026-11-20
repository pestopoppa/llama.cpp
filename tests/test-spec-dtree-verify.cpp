// AK-DTREE-1: greedy token-identity test of the path-row tree verify on a real (hybrid) model.
//
//   test-spec-dtree-verify <model.gguf> [n_gen] [n_threads] [D] [n_paths]
//
// n_paths -1 = SCHEDULE (the varying 2/3-sequence verify of the 27B T9 server row; with DTREE_NGL=99 on gfx90a this is the
// regression test of the AK-ASM-H15 input/output alias race), 0 = mixed 2/3 paths, 2..3 = fixed. D != 6 (or DTREE_ORACLE=1)
// checks every emitted token against a fresh prefill instead of one no-spec run.
//
// Uses an ORACLE drafter (the no-spec greedy continuation) with a deterministic corruption per round, so the
// round outcome is controlled: the main path is wrong at position c, alternative 1 is the oracle (right at c),
// alternative 2 is wrong at c in a different way. Three decoders must emit the same tokens:
//   ref    : no speculation, one token per decode
//   linear : main path only (the linear speculative verify, partial rollback via the recurrent snapshots)
//   tree   : main + 2 alternative paths in ONE decode, each its own seq (common_dtree_fork, which carries the
//            KV and the recurrent state incl. a pending rollback), the longest greedy match wins
//            (common_dtree_promote), then the same partial rollback.
// Also checks that the tree needs fewer rounds than linear (an alternative actually won and was used).
// Intended model: a qwen35 (GDN hybrid, rs rollback) target, e.g. Qwen3.5-0.8B-Q8_0. CPU is enough.

#include "common.h"
#include "llama.h"
#include "speculative-tree.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static int32_t D = 6;        // draft length (argv[4])
static int32_t N_PATHS = 3;  // main + alternatives in tree mode (argv[5], 2..3; 0 = mixed 2/3 per round)

static llama_token argmax_row(llama_context * ctx, int32_t i) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * l = llama_get_logits_ith(ctx, i);
    return (llama_token) std::distance(l, std::max_element(l, l + n_vocab));
}

struct runner {
    llama_model * model;
    llama_context * ctx;
    llama_batch batch;
    int32_t n_vocab;

    // returns the first sampled token after the prompt, pos of that token in *pos
    llama_token prefill(const std::vector<llama_token> & prompt, llama_pos * pos) {
        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch_clear(batch);
        for (size_t i = 0; i < prompt.size(); ++i) {
            common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
        }
        CHECK(llama_decode(ctx, batch) == 0);
        *pos = (llama_pos) prompt.size();
        return argmax_row(ctx, batch.n_tokens - 1);
    }
};

static std::vector<std::vector<float>> g_ref_logits; // [i] = logits after token i of the output (debug)

static std::vector<llama_token> run_ref(runner & r, const std::vector<llama_token> & prompt, int32_t n_gen) {
    llama_pos pos;
    llama_token tok = r.prefill(prompt, &pos);
    std::vector<llama_token> out = { tok };
    g_ref_logits.clear();
    while ((int32_t) out.size() < n_gen + D + 2) {
        common_batch_clear(r.batch);
        common_batch_add(r.batch, tok, pos++, { 0 }, true);
        CHECK(llama_decode(r.ctx, r.batch) == 0);
        if (getenv("DTREE_DEBUG")) {
            const float * l = llama_get_logits_ith(r.ctx, 0);
            g_ref_logits.emplace_back(l, l + r.n_vocab);
        }
        tok = argmax_row(r.ctx, 0);
        out.push_back(tok);
    }
    return out;
}

// ORACLE mode (DTREE_ORACLE=1, or the default for D != 6): instead of token identity with one no-spec run
// (fragile: a near-tie, e.g. a 0.02 logit gap, flips between batch shapes), every round re-derives the drafts from a
// FRESH prefill of prompt + emitted tokens in a second context and checks each emitted token against that fresh
// prefill's logits: an emitted token whose fresh logit is more than ORACLE_MARGIN below the fresh argmax is a
// verify bug, not GEMM noise. This is the fresh-prefill oracle gate of the AK-DTREE rows.
static runner * g_orc = nullptr;
static constexpr float ORACLE_MARGIN = 0.5f;
static int32_t g_n_bad = 0;
static float   g_max_margin = 0.0f;

// greedy continuation of prompt + prefix by n tokens from a fresh prefill; rows[i] = logits predicting token i
static std::vector<llama_token> orc_continue(const std::vector<llama_token> & prompt, const std::vector<llama_token> & prefix,
                                             int32_t n, std::vector<std::vector<float>> & rows) {
    runner & o = *g_orc;
    std::vector<llama_token> all = prompt;
    all.insert(all.end(), prefix.begin(), prefix.end());
    llama_pos pos;
    llama_token tok = o.prefill(all, &pos);
    rows.clear();
    {
        const float * l = llama_get_logits_ith(o.ctx, o.batch.n_tokens - 1);
        rows.emplace_back(l, l + o.n_vocab);
    }
    std::vector<llama_token> res = { tok };
    while ((int32_t) res.size() < n) {
        common_batch_clear(o.batch);
        common_batch_add(o.batch, tok, pos++, { 0 }, true);
        CHECK(llama_decode(o.ctx, o.batch) == 0);
        const float * l = llama_get_logits_ith(o.ctx, 0);
        rows.emplace_back(l, l + o.n_vocab);
        tok = argmax_row(o.ctx, 0);
        res.push_back(tok);
    }
    return res;
}

// mode 0 = linear, 1 = tree. Returns emitted tokens; *n_rounds / *n_alt_wins out.
static std::vector<llama_token> run_spec(runner & r, const std::vector<llama_token> & prompt,
                                         const std::vector<llama_token> & ref, int32_t n_gen, int mode,
                                         int32_t * n_rounds, int32_t * n_alt_wins) {
    llama_pos pos;
    llama_token sampled = r.prefill(prompt, &pos);
    std::vector<llama_token> out = { sampled };
    *n_rounds = 0;
    *n_alt_wins = 0;

    while ((int32_t) out.size() < n_gen) {
        const size_t k = out.size() - 1; // index of `sampled` in ref (valid while out == ref)
        std::vector<llama_token> oracle;
        std::vector<std::vector<float>> orc_rows; // ORACLE mode: fresh logits predicting out[k + 1 + i]
        if (g_orc) {
            // the fresh prefill ends at `sampled`: its first prediction is the token after it
            oracle = orc_continue(prompt, out, D + 1, orc_rows);
            oracle.resize(D);
        } else {
            CHECK(k + D < ref.size());
            oracle.assign(ref.begin() + k + 1, ref.begin() + k + 1 + D);
        }

        // round script (round % 4), c = round % D:
        //   0: main wrong at c, alt 1 = oracle               -> alt 1 wins fully (promote, no rollback)
        //   1: main wrong at c, alt 1 right at c but wrong at c + 2 (when it exists)
        //                                                   -> alt 1 wins partially (promote + partial rollback)
        //   2: main wrong at c, both alternatives wrong at c -> main wins partially (partial rollback)
        //   3: main right, alt 1 wrong at c                  -> main wins fully
        // The fork of the round after 1 and 2 copies a sequence with a PENDING recurrent rollback.
        const int32_t round = *n_rounds;
        const int32_t c = round % D;
        const int32_t kind = round % 4;
        auto corrupt = [&](std::vector<llama_token> & p, int32_t at, int32_t by) {
            p[at] = (p[at] + by) % r.n_vocab;
        };

        std::vector<std::vector<llama_token>> paths;
        paths.push_back(oracle);
        if (kind != 3) {
            corrupt(paths[0], c, 1);
        }
        if (mode == 1) {
            paths.push_back(oracle);
            if (N_PATHS != 2) {
                paths.push_back(oracle);
                corrupt(paths[2], c, 2);
            }
            if (kind == 1 && c + 2 < D) {
                corrupt(paths[1], c + 2, 3);
            }
            if (kind == 2 || kind == 3) {
                corrupt(paths[1], c, 3);
            }
            if (N_PATHS == 0) {
                // MIXED (the server's fixed-width trees): the path count varies per round (2 or 3) and every other
                // block of 4 rounds puts the winning alternative on the LAST sequence, so a round with fewer paths
                // follows a round whose winner was the highest alternative sequence
                if ((round / 4) % 2 == 1) {
                    std::swap(paths[1], paths[2]);
                }
                if (round % 3 == 2) {
                    paths.pop_back();
                }
            }
        }

        if (mode == 1 && N_PATHS < 0) {
            // SCHEDULE (from the 27B T9 server trace): main partial (2 paths) -> the LAST alternative of a 3-path
            // round wins fully -> main full win in a 2-path round -> main full win in a 2-path round
            const int32_t ph = round % 4;
            paths.assign(1, oracle);
            if (ph <= 1) {
                corrupt(paths[0], c, 1);
            }
            paths.push_back(oracle);
            if (ph != 3) {
                corrupt(paths[1], c, 3);
            }
            if (ph == 1) {
                paths.push_back(oracle);
            }
        }

        common_batch_clear(r.batch);
        std::vector<std::vector<int32_t>> rows(paths.size());
        for (size_t p = 0; p < paths.size(); ++p) {
            const llama_seq_id seq = (llama_seq_id) p;
            if (p > 0) {
                common_dtree_fork(r.ctx, 0, seq);
            }
            rows[p].push_back(r.batch.n_tokens);
            common_batch_add(r.batch, sampled, pos, { seq }, true);
            for (int32_t i = 0; i < D; ++i) {
                rows[p].push_back(r.batch.n_tokens);
                common_batch_add(r.batch, paths[p][i], pos + 1 + i, { seq }, true);
            }
        }
        CHECK(llama_decode(r.ctx, r.batch) == 0);

        // every path starts from the same state: the anchor rows (and the shared prefix rows) must agree.
        // This is what fails when a fork does not carry the KV / recurrent state (incl. a pending rollback).
        for (size_t p = 1; p < paths.size(); ++p) {
            int32_t shared = 0;
            while (shared < D && paths[p][shared] == paths[0][shared]) {
                ++shared;
            }
            for (int32_t i = 0; i <= shared; ++i) {
                const float * a = llama_get_logits_ith(r.ctx, rows[0][i]);
                const float * b = llama_get_logits_ith(r.ctx, rows[p][i]);
                float md = 0.0f;
                for (int32_t v = 0; v < r.n_vocab; ++v) {
                    md = std::max(md, std::fabs(a[v] - b[v]));
                }
                // noise between paths (FA accumulation order over differently placed cells) is <= ~0.3 on
                // Qwen3.5-0.8B Q8_0 CPU; a fork that misses a pending recurrent rollback differs by >= 10
                if (md > 2.0f) {
                    fprintf(stderr, "mode %d round %d: path %zu row %d differs from path 0 (max |d| = %g)\n",
                            mode, round, p, i, md);
                    if (i == 0 && k < g_ref_logits.size()) {
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int32_t v = 0; v < r.n_vocab; ++v) {
                            d0 = std::max(d0, std::fabs(a[v] - g_ref_logits[k][v]));
                            d1 = std::max(d1, std::fabs(b[v] - g_ref_logits[k][v]));
                        }
                        fprintf(stderr, "  vs no-spec ref: path0 %g, path%zu %g\n", d0, p, d1);
                    }
                    fprintf(stderr, "  argmax path0 %d, path%zu %d, expected (ref) %d\n",
                            argmax_row(r.ctx, rows[0][i]), p, argmax_row(r.ctx, rows[p][i]),
                            i == 0 ? oracle[0] : paths[0][i]);
                    exit(1);
                }
            }
        }

        if (getenv("DTREE_DEBUG") && k < g_ref_logits.size()) {
            for (size_t p = 0; p < paths.size(); ++p) {
                const float * a = llama_get_logits_ith(r.ctx, rows[p][0]);
                float d = 0.0f;
                for (int32_t v = 0; v < r.n_vocab; ++v) {
                    d = std::max(d, std::fabs(a[v] - g_ref_logits[k][v]));
                }
                fprintf(stderr, "DBG mode %d round %d k %zu path %zu row0 vs ref %g\n", mode, round, k, p, d);
            }
        }

        int32_t best = 0;
        int32_t best_n = common_dtree_greedy_accept_len(r.ctx, rows[0], paths[0]);
        for (size_t p = 1; p < paths.size(); ++p) {
            const int32_t n = common_dtree_greedy_accept_len(r.ctx, rows[p], paths[p]);
            if (n > best_n) {
                best = (int32_t) p;
                best_n = n;
            }
        }
        if (best != 0) {
            common_dtree_promote(r.ctx, best, 0);
            ++*n_alt_wins;
        }
        for (size_t p = 1; p < paths.size(); ++p) {
            if ((int32_t) p != best) {
                common_dtree_drop(r.ctx, (llama_seq_id) p);
            }
        }

        // emit: accepted drafts + bonus from the winner's row best_n
        for (int32_t i = 0; i < best_n; ++i) {
            out.push_back(paths[best][i]);
        }
        const llama_token bonus = argmax_row(r.ctx, rows[best][best_n]);
        out.push_back(bonus);

        // keep sampled + best_n accepted (positions pos .. pos+best_n), roll back the rest
        const llama_pos keep_end = pos + best_n + 1;
        if (best_n < D) {
            CHECK(llama_memory_seq_rm(llama_get_memory(r.ctx), 0, keep_end, -1));
        }
        CHECK(llama_memory_seq_pos_max(llama_get_memory(r.ctx), 0) == keep_end - 1);
        for (llama_seq_id s = 1; s < 3; ++s) {
            CHECK(llama_memory_seq_pos_max(llama_get_memory(r.ctx), s) < 0);
        }

        sampled = bonus;
        pos = keep_end;
        ++*n_rounds;

        if (g_orc) {
            for (int32_t i = 0; i <= best_n; ++i) {
                const auto & L = orc_rows[i];
                const llama_token t = out[k + 1 + i];
                const float m = *std::max_element(L.begin(), L.end()) - L[t];
                g_max_margin = std::max(g_max_margin, m);
                if (m > ORACLE_MARGIN) {
                    ++g_n_bad;
                    fprintf(stderr, "mode %d round %d (kind %d, winner %d, n %d): emitted token %zu = %d is %.3f below the "
                            "fresh-prefill argmax\n", mode, round, kind, best, best_n, k + 1 + i, t, m);
                }
            }
            continue;
        }

        // early exit with a precise report on the first divergence
        for (size_t i = 0; i < out.size() && i < ref.size(); ++i) {
            if (out[i] != ref[i]) {
                fprintf(stderr, "mode %d: divergence at token %zu (round %d, winner %d, n %d): %d vs ref %d\n",
                        mode, i, round, best, best_n, out[i], ref[i]);

                exit(1);
            }
        }
    }
    out.resize(n_gen);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_gen] [n_threads]\n", argv[0]);
        return 2;
    }
    const int32_t n_gen     = argc > 2 ? atoi(argv[2]) : 64;
    const int32_t n_threads = argc > 3 ? atoi(argv[3]) : 8;
    D       = argc > 4 ? atoi(argv[4]) : 6;
    N_PATHS = argc > 5 ? atoi(argv[5]) : 3;
    CHECK(D >= 2 && (N_PATHS <= 0 || (N_PATHS >= 2 && N_PATHS <= COMMON_DTREE_MAX_PATHS)));

    llama_backend_init();

    auto mparams = llama_model_default_params();
    // DTREE_NGL=99 runs the target on the GPU build (the AK-DTREE rows' backend); default CPU
    mparams.n_gpu_layers = getenv("DTREE_NGL") ? atoi(getenv("DTREE_NGL")) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    CHECK(model);

    auto cparams = llama_context_default_params();
    cparams.n_ctx           = 2048;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = COMMON_DTREE_MAX_PATHS;
    cparams.n_rs_seq        = D;
    cparams.kv_unified      = true;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    CHECK(ctx);
    if (llama_n_rs_seq(ctx) < (uint32_t) D) {
        fprintf(stderr, "SKIP: model has no recurrent partial rollback (n_rs_seq=%u)\n", llama_n_rs_seq(ctx));
        return 0;
    }

    runner r { model, ctx, llama_batch_init(512, 0, COMMON_DTREE_MAX_PATHS),
               llama_vocab_n_tokens(llama_model_get_vocab(model)) };

    const std::string text =
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n";
    const auto prompt = common_tokenize(ctx, text, true, false);

    const auto ref = run_ref(r, prompt, n_gen);

    const char * e_orc = getenv("DTREE_ORACLE");
    const bool use_orc = e_orc ? atoi(e_orc) != 0 : D != 6;
    runner orc { model, nullptr, llama_batch_init(2048, 0, 1), r.n_vocab };
    if (use_orc) {
        auto cp = cparams;
        cp.n_seq_max = 1;
        cp.n_batch   = 2048;
        cp.n_ubatch  = 512;
        orc.ctx = llama_init_from_model(model, cp);
        CHECK(orc.ctx);
        g_orc = &orc;
    }

    int32_t rounds_lin = 0, wins_lin = 0, rounds_tree = 0, wins_tree = 0;
    const auto lin  = run_spec(r, prompt, ref, n_gen, 0, &rounds_lin,  &wins_lin);
    const auto tree = run_spec(r, prompt, ref, n_gen, 1, &rounds_tree, &wins_tree);

    const std::vector<llama_token> ref_n(ref.begin(), ref.begin() + n_gen);
    if (use_orc) {
        printf("oracle: %d emitted tokens more than %.2f below the fresh-prefill argmax (max %.3f), D=%d paths=%d\n",
               g_n_bad, ORACLE_MARGIN, g_max_margin, D, N_PATHS);
        CHECK(g_n_bad == 0);
    } else {
        CHECK(lin == ref_n);
        CHECK(tree == ref_n);
    }
    CHECK(wins_lin == 0);
    CHECK(wins_tree > 0);
    CHECK(rounds_tree < rounds_lin);

    printf("test-spec-dtree-verify: OK  n_gen=%d  D=%d  paths=%d  linear rounds=%d  tree rounds=%d (alt wins %d)  %s\n",
           n_gen, D, N_PATHS, rounds_lin, rounds_tree, wins_tree,
           use_orc ? "every emitted token agrees with a fresh prefill" : "token-identical to no-spec greedy");
    if (orc.ctx) {
        llama_free(orc.ctx);
    }
    llama_batch_free(orc.batch);

    llama_batch_free(r.batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
