// Rollback guards of llama_memory_recurrent: which partial seq_rm calls must be refused, and that an
// accepted one leaves the exact state. Needs a recurrent/hybrid model; logits are compared against a
// context that never decoded the rolled-back tokens.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

static common_params g_params;
static llama_model * g_model   = nullptr;
static int           g_n_vocab = 0;

constexpr uint32_t  N_RS_SEQ = 8;
constexpr llama_pos N_PREFIX = 16;
constexpr float     EPS      = 1e-3f;

using ctx_t = llama_context;

static ctx_t * make_ctx(uint32_t n_seq_max, bool kv_unified) {
    auto cparams = common_context_params_to_llama(g_params);
    cparams.n_seq_max  = n_seq_max;
    cparams.n_rs_seq   = N_RS_SEQ;
    cparams.n_batch    = 64;
    cparams.n_ubatch   = 64;
    cparams.kv_unified = kv_unified;
    return llama_init_from_model(g_model, cparams);
}

static llama_token tok(llama_seq_id seq, llama_pos pos) {
    return (llama_token) (((uint64_t) (pos + 1) * 7919u + (uint64_t) seq * 104729u) % (uint64_t) g_n_vocab);
}

struct span { llama_seq_id seq; llama_pos p0; llama_pos p1; };

// one llama_batch holding every span; the last token of each span outputs logits
static bool decode(ctx_t * ctx, const std::vector<span> & spans) {
    int32_t n = 0;
    for (const auto & s : spans) { n += s.p1 - s.p0; }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (const auto & s : spans) {
        for (llama_pos p = s.p0; p < s.p1; ++p) {
            common_batch_add(b, tok(s.seq, p), p, { s.seq }, p + 1 == s.p1);
        }
    }
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

static bool rm(ctx_t * ctx, llama_seq_id seq, llama_pos p0) {
    return llama_memory_seq_rm(llama_get_memory(ctx), seq, p0, -1);
}

static float max_diff(ctx_t * a, ctx_t * b) {
    const float * la = llama_get_logits_ith(a, -1);
    const float * lb = llama_get_logits_ith(b, -1);
    if (la == nullptr || lb == nullptr) { return INFINITY; }
    float d = 0.0f;
    for (int i = 0; i < g_n_vocab; ++i) { d = std::max(d, std::fabs(la[i] - lb[i])); }
    return d;
}

#define REQUIRE(cond, ...) do { if (!(cond)) { fprintf(stderr, "    FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); return false; } } while (0)

// a rollback is only honourable when the last ubatch wrote its snapshot; a refusal must not mutate
static bool case_bound() {
    for (const uint32_t n : { 1u, 2u, N_RS_SEQ + 1, N_RS_SEQ + 2 }) {
        const uint32_t  r_max = n <= N_RS_SEQ + 1 ? n - 1 : 0;
        const llama_pos end   = N_PREFIX + (llama_pos) n;
        ctx_t * ctx = make_ctx(1, false);
        REQUIRE(ctx && decode(ctx, {{0, 0, N_PREFIX}}) && decode(ctx, {{0, N_PREFIX, end}}), "decode failed");
        for (uint32_t r = N_RS_SEQ + 1; r > r_max; --r) {
            REQUIRE(!rm(ctx, 0, end - (llama_pos) r), "rollback %u after a %u-token ubatch was accepted", r, n);
            REQUIRE(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == end - 1, "refusal mutated pos_max");
        }
        if (r_max > 0) {
            REQUIRE(rm(ctx, 0, end - (llama_pos) r_max), "rollback %u after a %u-token ubatch was refused", r_max, n);
        }
        llama_free(ctx);
    }
    return true;
}

static bool case_shared() {
    ctx_t * ctx = make_ctx(2, false);
    REQUIRE(ctx && decode(ctx, {{0, 0, N_PREFIX}}) && decode(ctx, {{0, N_PREFIX, N_PREFIX + 4}}), "decode failed");
    llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, -1, -1);
    REQUIRE(!rm(ctx, 0, N_PREFIX + 2), "rollback of a cell shared by two sequences was accepted");
    REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1), "full removal of seq 1 failed");
    REQUIRE(rm(ctx, 0, N_PREFIX + 2), "rollback refused after the cell stopped being shared");
    llama_free(ctx);
    return true;
}

static bool case_pending() {
    ctx_t * ctx = make_ctx(1, false);
    REQUIRE(ctx && decode(ctx, {{0, 0, N_PREFIX}}) && decode(ctx, {{0, N_PREFIX, N_PREFIX + 5}}), "decode failed");
    REQUIRE(rm(ctx, 0, N_PREFIX + 3), "first rollback refused");
    REQUIRE(!rm(ctx, 0, N_PREFIX + 2), "second rollback before any decode was accepted");
    REQUIRE(decode(ctx, {{0, N_PREFIX + 3, N_PREFIX + 7}}), "decode failed");
    REQUIRE(rm(ctx, 0, N_PREFIX + 5), "rollback after a fresh decode was refused");
    llama_free(ctx);
    return true;
}

static bool case_exact() {
    ctx_t * a = make_ctx(1, false);
    ctx_t * b = make_ctx(1, false);
    REQUIRE(a && b && decode(a, {{0, 0, N_PREFIX}}) && decode(a, {{0, N_PREFIX, N_PREFIX + 6}}), "decode failed");
    REQUIRE(rm(a, 0, N_PREFIX + 3), "legal rollback refused");
    REQUIRE(decode(a, {{0, N_PREFIX + 3, N_PREFIX + 4}}), "replay failed");
    REQUIRE(decode(b, {{0, 0, N_PREFIX}}) && decode(b, {{0, N_PREFIX, N_PREFIX + 3}}) &&
            decode(b, {{0, N_PREFIX + 3, N_PREFIX + 4}}), "reference failed");
    const float d = max_diff(a, b);
    fprintf(stderr, "    max |logit diff| = %g\n", d);
    REQUIRE(d <= EPS, "rolled-back state differs from the reference");
    llama_free(a); llama_free(b);
    return true;
}

// X decodes alone, then a later ubatch of other sequences reorders the cells. A rollback of X must be either
// refused or exact: it must not read the snapshots of a row it no longer owns.
static bool displaced(const std::vector<llama_seq_id> & order, const char * what, bool expect_refusal_allowed) {
    ctx_t * a = make_ctx(3, true);
    ctx_t * b = make_ctx(3, true);
    REQUIRE(a && b, "ctx failed");
    for (auto s : order) {
        REQUIRE(decode(a, {{s, 0, N_PREFIX}}), "prefix failed");
    }
    REQUIRE(decode(a, {{0, N_PREFIX, N_PREFIX + 6}}), "X decode failed");
    REQUIRE(decode(a, {{1, N_PREFIX, N_PREFIX + 1}, {2, N_PREFIX, N_PREFIX + 1}}), "other seqs failed");

    REQUIRE(decode(b, {{0, 0, N_PREFIX}}) && decode(b, {{0, N_PREFIX, N_PREFIX + 3}}) &&
            decode(b, {{0, N_PREFIX + 3, N_PREFIX + 4}}), "reference failed");

    if (!rm(a, 0, N_PREFIX + 3)) {
        fprintf(stderr, "    %s: rollback refused\n", what);
        REQUIRE(expect_refusal_allowed, "%s: rollback refused but the cell was not displaced", what);
    } else {
        REQUIRE(decode(a, {{0, N_PREFIX + 3, N_PREFIX + 4}}), "replay failed");
        const float d = max_diff(a, b);
        fprintf(stderr, "    %s: rollback accepted, max |logit diff| = %g\n", what, d);
        REQUIRE(d <= EPS, "%s: accepted rollback gave wrong state", what);
    }
    llama_free(a); llama_free(b);
    return true;
}

static bool case_displaced()     { return displaced({2, 0, 1}, "displaced", true); }
static bool case_not_displaced() { return displaced({0, 1, 2}, "control", false); }

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();
    if (!common_params_parse(argc, argv, g_params, LLAMA_EXAMPLE_COMMON)) { return 1; }
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(g_params);
    g_model = init->model();
    if (g_model == nullptr) { return 1; }
    if (!llama_model_is_recurrent(g_model) && !llama_model_is_hybrid(g_model)) {
        fprintf(stderr, "skipping: non-recurrent model\n");
        return 0;
    }
    g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(g_model));

    const std::vector<std::pair<const char *, std::function<bool()>>> cases = {
        {"bound", case_bound}, {"shared", case_shared}, {"pending", case_pending}, {"exact", case_exact},
        {"control_not_displaced", case_not_displaced}, {"displaced_by_other_seqs", case_displaced},
    };
    int n_fail = 0;
    for (const auto & [name, fn] : cases) {
        const bool ok = fn();
        fprintf(stderr, "CASE %-26s %s\n", name, ok ? "PASS" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
    fprintf(stderr, "RESULT %d failed of %zu\n", n_fail, cases.size());
    return n_fail == 0 ? 0 : 1;
}
