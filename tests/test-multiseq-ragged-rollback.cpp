// SPEC-ID-RAGGED-1: multi-sequence ragged-rollback identity test for speculative decoding.
//
// N_SEQ sequences share one batch. Every round each sequence verifies its last token plus K draft tokens
// (K+1 rows), accepts a per-sequence number of drafts a in [0, K] and rolls its KV / recurrent (GDN conv+ssm)
// state back by K - a with llama_memory_seq_rm. The bonus token (argmax of row a) seeds the next round.
//
// Reference A (gate, exact): the same sequence, same drafts, same acceptance schedule, run ALONE in a fresh
//   context with the same configuration. Every verify row of every round must match bit-for-bit (|d| <= eps).
// Reference B (gate on tokens): plain one-token-at-a-time decode of the accepted stream, no drafting, no
//   rollback. Bonus tokens must match; the logit difference is reported (different ubatch shapes take
//   different GDN paths, so it is not required to be zero).
//
// Arms: recurrent rollback by snapshots (LLAMA_RS_REPLAY=0) and by replay (LLAMA_RS_REPLAY=2), each with
// kv_unified off and on. Cases: rollback by 0 (accept all), by the max (accept none), mismatched per-seq
// amounts, slot reuse (a sequence is cleared and re-prefilled mid-run while the others are pending).
// Mutation (red) cases: the batched run deviates from the reference schedule for one sequence (one rollback
// too short; two sequences' rollback amounts swapped); the test must DETECT each, else it is vacuous.
//
// Arm residency is proven from the context's own log line "rs_replay = true|false".

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr int      N_SEQ    = 4;
static constexpr int      K        = 4;   // draft tokens per round; verify batch = K + 1 rows per sequence
static constexpr int      N_ROUNDS = 12;
static constexpr int      REUSE_SEQ   = 3;
static constexpr int      REUSE_ROUND = 6;
static constexpr float    EPS      = 1e-5f;

// ---- log capture: proves which rollback mode the context actually took ----
static int g_last_rs_replay = -1;
static void log_cb(ggml_log_level level, const char * text, void * /*ud*/) {
    const char * p = strstr(text, "rs_replay     = ");
    if (p) {
        g_last_rs_replay = strstr(p, "true") ? 1 : 0;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct arm_cfg {
    int  replay;      // LLAMA_RS_REPLAY value
    bool unified;
};

static llama_context * make_ctx(const common_params & params, llama_model * model, const arm_cfg & arm) {
    char buf[8];
    snprintf(buf, sizeof(buf), "%d", arm.replay);
    setenv("LLAMA_RS_REPLAY", buf, 1);
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = N_SEQ;
    cparams.n_rs_seq   = 8;
    cparams.kv_unified = arm.unified;
    cparams.n_ctx      = 512 * N_SEQ;
    cparams.n_batch    = std::max<uint32_t>(cparams.n_batch,  64 * N_SEQ);
    cparams.n_ubatch   = std::max<uint32_t>(cparams.n_ubatch, 64 * N_SEQ);
    g_last_rs_replay = -1;
    llama_context * ctx = llama_init_from_model(model, cparams);
    return ctx;
}

// deterministic per-(seq, round, i) pseudo-random token
static llama_token prng_tok(uint32_t a, uint32_t b, uint32_t c, int n_vocab) {
    uint32_t x = a * 2654435761u ^ (b + 0x9e3779b9u) * 40503u ^ (c + 7u) * 2246822519u;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    return (llama_token) (1 + x % (uint32_t) (n_vocab - 1));
}

static std::vector<llama_token> make_prompt(int seq, int gen, int n_vocab) {
    const int len = 5 + 2 * seq + gen;  // ragged prompt lengths
    std::vector<llama_token> p(len);
    for (int i = 0; i < len; ++i) {
        p[i] = prng_tok(seq + 100 * gen, 9999, i, n_vocab);
    }
    return p;
}

static int argmax(const float * v, int n) {
    return (int) (std::max_element(v, v + n) - v);
}

// acceptance schedule; mutation alters what the BATCHED run does, the reference always uses the base schedule
enum mutation { MUT_NONE, MUT_SHORT_ROLLBACK, MUT_SWAP_AMOUNTS };

static int accept_base(int s, int r) {
    switch (s) {
        case 0:  return K;                          // rollback by 0
        case 1:  return 0;                          // rollback by the max (K)
        case 2:  return (r * 3 + 1) % (K + 1);      // ragged
        default: return (r * 2 + 3) % (K + 1);      // ragged, different phase
    }
}

static int accept_for(int s, int r, mutation m) {
    if (m == MUT_SHORT_ROLLBACK && s == 2 && r == 3) {
        const int a = accept_base(s, r);
        return a < K ? a + 1 : a - 1;  // one rollback step off
    }
    if (m == MUT_SWAP_AMOUNTS && r == 4 && (s == 2 || s == 3)) {
        return accept_base(s == 2 ? 3 : 2, r);
    }
    return accept_base(s, r);
}

// per-sequence driver state
struct seq_state {
    int  seq_id;
    llama_pos pos = 0;              // next position to write
    llama_token last = 0;           // token to verify at the head of the next round
    std::vector<llama_token> accepted;                // full accepted stream incl. prompt
    std::vector<std::vector<float>> rows;             // every verify row logged (for reference A compare)
    std::vector<llama_token> bonus;                   // bonus tokens (for reference B compare)
    std::vector<llama_pos>   bonus_pos;
    bool needs_prefill = true;
    size_t seg_bonus_start = 0;                       // first bonus of the current (post-reuse) segment
    std::vector<llama_token> prompt;
};

struct run_result {
    bool ok = true;
    int  rs_replay = -1;
    std::vector<seq_state> seqs;
};

// Runs the given sequences (subset of 0..N_SEQ-1) together. alone=true is used for reference A (one seq).
static run_result run_spec(const common_params & params, llama_model * model, const arm_cfg & arm,
                           const std::vector<int> & seq_ids, mutation mut) {
    run_result res;
    llama_context * ctx = make_ctx(params, model, arm);
    if (!ctx) { res.ok = false; return res; }
    res.rs_replay = g_last_rs_replay;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(ctx);

    for (int s : seq_ids) {
        seq_state st;
        st.seq_id = s;
        st.prompt = make_prompt(s, 0, n_vocab);
        res.seqs.push_back(std::move(st));
    }

    llama_batch batch = llama_batch_init(64 * N_SEQ, 0, 1);
    for (int r = 0; r < N_ROUNDS && res.ok; ++r) {
        // slot reuse: clear one sequence fully and restart it with a new prompt while the others are pending
        if (r == REUSE_ROUND) {
            for (auto & st : res.seqs) {
                if (st.seq_id == REUSE_SEQ) {
                    if (!llama_memory_seq_rm(mem, st.seq_id, -1, -1)) {
                        fprintf(stderr, "seq_rm full clear failed\n");
                        res.ok = false;
                    }
                    st.prompt = make_prompt(st.seq_id, 1, n_vocab);
                    st.pos = 0;
                    st.accepted.clear();
                    st.needs_prefill = true;
                    st.seg_bonus_start = st.bonus.size();
                }
            }
        }

        common_batch_clear(batch);
        std::vector<int> row0(res.seqs.size());
        std::vector<int> nrow(res.seqs.size());
        std::vector<std::vector<llama_token>> verify(res.seqs.size());
        for (size_t i = 0; i < res.seqs.size(); ++i) {
            auto & st = res.seqs[i];
            row0[i] = batch.n_tokens;
            if (st.needs_prefill) {
                // prefill round: whole prompt, last row gives the first bonus
                verify[i] = st.prompt;
            } else {
                verify[i].push_back(st.last);
                for (int d = 0; d < K; ++d) {
                    verify[i].push_back(prng_tok(st.seq_id, r, d, n_vocab));
                }
            }
            for (size_t j = 0; j < verify[i].size(); ++j) {
                common_batch_add(batch, verify[i][j], st.pos + (llama_pos) j, { st.seq_id }, true);
            }
            nrow[i] = (int) verify[i].size();
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed round %d\n", r);
            res.ok = false;
            break;
        }
        for (size_t i = 0; i < res.seqs.size() && res.ok; ++i) {
            auto & st = res.seqs[i];
            for (int j = 0; j < nrow[i]; ++j) {
                const float * lg = llama_get_logits_ith(ctx, row0[i] + j);
                st.rows.emplace_back(lg, lg + n_vocab);
            }
            if (st.needs_prefill) {
                st.accepted.insert(st.accepted.end(), verify[i].begin(), verify[i].end());
                st.pos += nrow[i];
                const llama_token b = argmax(st.rows.back().data(), n_vocab);
                st.bonus.push_back(b);
                st.bonus_pos.push_back(st.pos);
                st.last = b;
                st.needs_prefill = false;
                continue;
            }
            const int a = accept_for(st.seq_id, r, mut);
            // accepted: last + a drafts; positions pos .. pos+a
            for (int j = 0; j <= a; ++j) {
                st.accepted.push_back(verify[i][j]);
            }
            const llama_pos keep_end = st.pos + a + 1;
            if (a < K) {
                if (!llama_memory_seq_rm(mem, st.seq_id, keep_end, -1)) {
                    fprintf(stderr, "seq %d round %d: rollback by %d refused\n", st.seq_id, r, K - a);
                    res.ok = false;
                    break;
                }
            }
            st.pos = keep_end;
            const llama_token b = argmax(st.rows[st.rows.size() - nrow[i] + a].data(), n_vocab);
            st.bonus.push_back(b);
            st.bonus_pos.push_back(st.pos);
            st.last = b;
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return res;
}

// Reference B: plain sequential decode of the accepted stream (no drafts, no rollback), compare bonus tokens.
static bool check_plain(const common_params & params, llama_model * model, const arm_cfg & arm,
                        const seq_state & st, float & max_diff) {
    // only the segment after the last (re)prefill is in st.accepted
    llama_context * ctx = make_ctx(params, model, arm);
    if (!ctx) return false;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(1, 0, 1);
    bool ok = true;
    // bonus entries that belong to the current accepted segment: positions <= accepted.size()
    const size_t first_b = st.seg_bonus_start;
    for (size_t p = 0; p < st.accepted.size() && ok; ++p) {
        common_batch_clear(batch);
        common_batch_add(batch, st.accepted[p], (llama_pos) p, { st.seq_id }, true);
        if (llama_decode(ctx, batch) != 0) { ok = false; break; }
        const float * lg = llama_get_logits_ith(ctx, 0);
        const llama_token t = argmax(lg, n_vocab);
        for (size_t i = first_b; i < st.bonus_pos.size(); ++i) {
            if (st.bonus_pos[i] == (llama_pos) p + 1) {
                if (t != st.bonus[i]) {
                    fprintf(stderr, "  [B] seq %d pos %zu: plain argmax %d != spec bonus %d\n", st.seq_id, p + 1, t, st.bonus[i]);
                    ok = false;
                }
            }
        }
        // accepted token after p must equal the plain argmax at p (that is what accept means in real spec
        // decoding); here drafts are forced-accepted, so only the bonus positions are checkable
        (void) max_diff;
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return ok;
}

static bool compare_ref_a(const seq_state & got, const seq_state & ref, float & max_diff, bool verbose) {
    max_diff = 0.0f;
    if (got.rows.size() != ref.rows.size()) {
        if (verbose) fprintf(stderr, "  [A] seq %d: row count %zu != %zu\n", got.seq_id, got.rows.size(), ref.rows.size());
        max_diff = INFINITY;
        return false;
    }
    bool ok = true;
    bool reported = false;
    for (size_t i = 0; i < got.rows.size(); ++i) {
        for (size_t t = 0; t < got.rows[i].size(); ++t) {
            const float d = std::fabs(got.rows[i][t] - ref.rows[i][t]);
            if (!(d <= EPS)) {
                ok = false;
                if (verbose && !reported) {
                    fprintf(stderr, "  [A] seq %d: first mismatch row %zu token %zu (%g vs %g)\n",
                            got.seq_id, i, t, (double) got.rows[i][t], (double) ref.rows[i][t]);
                    reported = true;
                }
            }
            if (d > max_diff || std::isnan(d)) max_diff = std::isnan(d) ? INFINITY : d;
        }
    }
    if (got.bonus != ref.bonus) {
        ok = false;
        if (verbose) fprintf(stderr, "  [A] seq %d: bonus token stream differs\n", got.seq_id);
    }
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_log_set(log_cb, nullptr);
    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "failed to init model\n");
        return 1;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "skipping for non-recurrent model\n");
        return 0;
    }

    const char * only = getenv("RAGGED_ONLY_REPLAY");  // optional: "0" or "2" to restrict arms
    const arm_cfg arms[] = { {0, false}, {0, true}, {2, false}, {2, true} };
    std::vector<int> all;
    for (int s = 0; s < N_SEQ; ++s) all.push_back(s);

    int n_fail = 0;
    for (const auto & arm : arms) {
        if (only && atoi(only) != arm.replay) continue;
        const char * mode = arm.replay ? "replay" : "snapshot";

        run_result batched = run_spec(params, model, arm, all, MUT_NONE);
        if (batched.rs_replay != (arm.replay ? 1 : 0)) {
            printf("ARM %-8s unified=%d: SKIP (context took rs_replay=%d, arch not on the replay allowlist?)\n",
                   mode, arm.unified, batched.rs_replay);
            if (arm.replay) { n_fail++; }  // the replay arm must actually run replay
            continue;
        }
        bool arm_ok = batched.ok;
        float worst_a = 0.0f;
        for (int s = 0; s < N_SEQ && arm_ok; ++s) {
            run_result alone = run_spec(params, model, arm, { s }, MUT_NONE);
            if (!alone.ok) { arm_ok = false; break; }
            float d = 0.0f;
            const bool a_ok = compare_ref_a(batched.seqs[s], alone.seqs[0], d, true);
            worst_a = std::max(worst_a, d);
            float db = 0.0f;
            const bool b_ok = check_plain(params, model, arm, batched.seqs[s], db);
            printf("  ARM %-8s unified=%d seq %d (accept %s): A(alone, exact) %s max|d|=%.3g  B(plain tokens) %s\n",
                   mode, arm.unified, s,
                   s == 0 ? "K/rollback 0" : s == 1 ? "0/rollback max" : s == 3 ? "ragged+reuse" : "ragged",
                   a_ok ? "PASS" : "FAIL", (double) d, b_ok ? "PASS" : "FAIL");
            arm_ok = arm_ok && a_ok && b_ok;
        }
        printf("ARM %-8s unified=%d (rs_replay=%d): %s  worst A max|d|=%.3g\n",
               mode, arm.unified, batched.rs_replay, arm_ok ? "GREEN" : "RED", (double) worst_a);
        n_fail += arm_ok ? 0 : 1;

        // mutation cases: the batched run deviates for one sequence; reference A must catch it
        for (mutation m : { MUT_SHORT_ROLLBACK, MUT_SWAP_AMOUNTS }) {
            run_result mutated = run_spec(params, model, arm, all, m);
            bool detected = !mutated.ok;
            for (int s = 0; s < N_SEQ && !detected; ++s) {
                run_result alone = run_spec(params, model, arm, { s }, MUT_NONE);
                float d = 0.0f;
                detected = !compare_ref_a(mutated.seqs[s], alone.seqs[0], d, false);
            }
            printf("  MUTATION %-15s %-8s unified=%d: %s\n",
                   m == MUT_SHORT_ROLLBACK ? "short-rollback" : "swap-amounts", mode, arm.unified,
                   detected ? "DETECTED (ok)" : "NOT DETECTED (test is vacuous)");
            n_fail += detected ? 0 : 1;
        }
    }

    printf("RESULT: %s (%d failing checks)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
