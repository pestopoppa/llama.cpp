// [KPF-17] cross-slot prefix fork: shared attention cells under seq_cp_ext / seq_rm / seq_keep.
//
// One unified KV pool, four sequences (the server's slots). Sequence 0 is the source: it decodes a
// prefix [0, p), a checkpoint of its non-rollbackable state is taken at p (what the server's
// create_checkpoint stores), then it keeps decoding to N. Sequence 1 is forked from it at p:
// attention cells shared zero-copy (LLAMA_MEMORY_SEQ_CP_FLAGS_ATTN_ONLY), recurrent / SWA state
// restored from the checkpoint into sequence 1's OWN rows (PARTIAL_ONLY). The test checks:
//   - cell accounting: seq bits, the `used` count, shared vs private (llama_memory_get_cell_stats)
//   - find_slot never hands a shared cell to new tokens (used grows by exactly the new tokens)
//   - fork == fresh: the forked sequence's suffix logits match a fresh decode of the same tokens
//   - fork == copy: the same state restored by the classic full-state copy path
//   - source invariance: the source's continuation is unchanged by the fork taken from it
//   - purge safety: removing the source drops only its seq bit; the fork's serialized state (its cells'
//     positions and K/V bytes) is byte-identical before and after the purge, and its continuation keeps
//     the same top-1 (logits are not compared bit-exactly there: the new token lands in a freed cell that
//     precedes the suffix cells in memory, so the attention sum runs in a different order than the
//     contiguous reference - a property of any fragmented unified pool, fork or not)
//   - seq_keep over shared cells
// Runs on any model; exact cell counts are asserted only where one token = one cell (no SWA).

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do {                                                   \
    if (!(cond)) {                                                              \
        fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);         \
        fprintf(stderr, __VA_ARGS__);                                           \
        fprintf(stderr, "\n");                                                  \
        g_fail++;                                                               \
    }                                                                           \
} while (0)

constexpr int N_SEQ = 4;

struct stats {
    llama_memory_cell_stats pool = { 0, 0, 0 };
    int64_t priv[N_SEQ] = { 0 };
    int64_t shr [N_SEQ] = { 0 };
    bool ok = false;
};

static stats get_stats(llama_context * ctx) {
    stats s;
    s.ok = llama_memory_get_cell_stats(llama_get_memory(ctx), &s.pool, N_SEQ, s.priv, s.shr);
    return s;
}

static void print_stats(const char * tag, const stats & s) {
    fprintf(stderr, "  [%s] size=%lld used=%lld shared=%lld | priv=(%lld,%lld,%lld,%lld) shr=(%lld,%lld,%lld,%lld)\n", tag,
            (long long) s.pool.n_size, (long long) s.pool.n_used, (long long) s.pool.n_shared,
            (long long) s.priv[0], (long long) s.priv[1], (long long) s.priv[2], (long long) s.priv[3],
            (long long) s.shr[0],  (long long) s.shr[1],  (long long) s.shr[2],  (long long) s.shr[3]);
}

// every used cell is either private to exactly one sequence or shared
static void check_invariant(const char * tag, const stats & s) {
    int64_t sum_priv = 0;
    for (int i = 0; i < N_SEQ; ++i) {
        sum_priv += s.priv[i];
    }
    CHECK(s.pool.n_used == sum_priv + s.pool.n_shared, "%s: used %lld != sum(private) %lld + shared %lld", tag,
          (long long) s.pool.n_used, (long long) sum_priv, (long long) s.pool.n_shared);
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_rs_seq) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = N_SEQ;
    cparams.kv_unified = true;
    cparams.n_ctx      = std::max<uint32_t>(cparams.n_ctx, 1024);
    cparams.n_batch    = std::max<uint32_t>(cparams.n_batch,  128);
    cparams.n_ubatch   = std::max<uint32_t>(cparams.n_ubatch, 128);
    cparams.n_rs_seq   = n_rs_seq;
    return llama_init_from_model(model, cparams);
}

// decode tokens[p0, p1) on seq_id as one batch; logits for the last token
static bool decode_range(llama_context * ctx, const std::vector<llama_token> & tokens, int p0, int p1, llama_seq_id seq_id) {
    llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
    for (int pos = p0; pos < p1; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { seq_id }, pos + 1 == p1);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static std::vector<float> last_logits(llama_context * ctx, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, -1);
    return l ? std::vector<float>(l, l + n_vocab) : std::vector<float>();
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static int argmax(const std::vector<float> & a) {
    return (int) (std::max_element(a.begin(), a.end()) - a.begin());
}

// compare, report bit-identity separately from the tolerance verdict
static void compare(const char * tag, const std::vector<float> & a, const std::vector<float> & b, float eps) {
    const float d = max_abs_diff(a, b);
    const bool  bit = !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    fprintf(stderr, "  %-34s max|d| = %.3g, bit-identical = %s, top1 %d vs %d\n", tag, (double) d, bit ? "yes" : "no",
            a.empty() ? -1 : argmax(a), b.empty() ? -1 : argmax(b));
    CHECK(d <= eps, "%s: max|d| %g > %g", tag, (double) d, (double) eps);
    CHECK(!a.empty() && !b.empty() && argmax(a) == argmax(b), "%s: top-1 differs", tag);
}

static int run(const common_params & params, llama_model * model, uint32_t n_rs_seq) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    const bool needs_ckpt = llama_model_is_recurrent(model) || llama_model_is_hybrid(model) || llama_model_n_swa(model) > 0;
    const bool exact      = llama_model_n_swa(model) == 0; // one token = one attention cell

    fprintf(stderr, "== n_rs_seq = %u, recurrent = %d, hybrid = %d, n_swa = %d\n", n_rs_seq,
            (int) llama_model_is_recurrent(model), (int) llama_model_is_hybrid(model), llama_model_n_swa(model));

    llama_context * ctx  = make_ctx(params, model, n_rs_seq);
    llama_context * ref  = make_ctx(params, model, n_rs_seq); // fresh reference, same shapes
    llama_context * ref2 = make_ctx(params, model, n_rs_seq); // source without any fork
    if (!ctx || !ref || !ref2) {
        fprintf(stderr, "failed to create contexts\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);

    if (llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        // no attention cells: the ATTN_ONLY share must refuse, changing nothing
        CHECK(!llama_memory_seq_cp_ext(mem, 0, 1, 0, 0, LLAMA_MEMORY_SEQ_CP_FLAGS_ATTN_ONLY), "pure recurrent memory accepted ATTN_ONLY");
        CHECK(!get_stats(ctx).ok, "pure recurrent memory reported cell stats");
        llama_free(ctx); llama_free(ref); llama_free(ref2);
        return 0;
    }

    // capability probe: an empty range changes nothing
    CHECK(llama_memory_seq_cp_ext(mem, 0, 1, 0, 0, LLAMA_MEMORY_SEQ_CP_FLAGS_ATTN_ONLY), "probe refused on unified KV");

    const int p = 40;   // fork point
    const int N = 64;   // source length
    const int S = 9;    // fork suffix

    // deterministic tokens; the fork's suffix differs from the source's continuation
    std::vector<llama_token> src_tok(N + 1), dst_tok(p + S + 1);
    for (int i = 0; i <= N; ++i) {
        src_tok[i] = 1 + (i * 7919 + 13) % (n_vocab - 1);
    }
    for (int i = 0; i < (int) dst_tok.size(); ++i) {
        dst_tok[i] = i < p ? src_tok[i] : 1 + (i * 104729 + 7) % (n_vocab - 1);
    }

    // 1. source prefix, checkpoint at p (non-rollbackable state), and a full-state copy for the copy path
    CHECK(decode_range(ctx, src_tok, 0, p, 0), "decode src prefix");
    common_prompt_checkpoint ck_partial;
    ck_partial.update_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    common_prompt_checkpoint ck_full;
    ck_full.update_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);

    // 2. the source keeps going to N (it is "busy" past the fork point)
    CHECK(decode_range(ctx, src_tok, p, N, 0), "decode src rest");

    stats a = get_stats(ctx);
    print_stats("source only", a);
    CHECK(a.ok, "cell stats unsupported");
    check_invariant("source only", a);
    if (exact) {
        CHECK(a.pool.n_used == N && a.pool.n_shared == 0 && a.priv[0] == N, "source-only counts");
    }

    // 3. fork seq 1 from seq 0 at p
    llama_memory_seq_rm(mem, 1, -1, -1);
    CHECK(llama_memory_seq_cp_ext(mem, 0, 1, 0, p, LLAMA_MEMORY_SEQ_CP_FLAGS_ATTN_ONLY), "ATTN_ONLY share refused");
    if (needs_ckpt) {
        ck_partial.load_tgt(ctx, 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    }
    CHECK(llama_memory_seq_pos_max(mem, 1) == p - 1, "fork pos_max %d != %d", llama_memory_seq_pos_max(mem, 1), p - 1);

    stats b = get_stats(ctx);
    print_stats("after fork", b);
    check_invariant("after fork", b);
    if (exact) {
        CHECK(b.pool.n_used == N, "fork allocated cells: used %lld", (long long) b.pool.n_used);
        CHECK(b.pool.n_shared == p && b.shr[0] == p && b.shr[1] == p, "shared counts");
        CHECK(b.priv[0] == N - p && b.priv[1] == 0, "private counts");
    }

    // 4. the fork's suffix: new tokens must land in free cells, never in a shared one
    CHECK(decode_range(ctx, dst_tok, p, p + S, 1), "decode fork suffix");
    const auto l_fork = last_logits(ctx, n_vocab);

    stats c = get_stats(ctx);
    print_stats("fork suffix", c);
    check_invariant("fork suffix", c);
    if (exact) {
        CHECK(c.pool.n_used == N + S, "find_slot reused a cell: used %lld != %d", (long long) c.pool.n_used, N + S);
        CHECK(c.priv[1] == S && c.shr[1] == p && c.pool.n_shared == p, "suffix counts");
    }

    // 5. fork == fresh: the same tokens decoded from scratch, same chunking
    CHECK(decode_range(ref, dst_tok, 0, p, 0), "ref prefix");
    CHECK(decode_range(ref, dst_tok, p, p + S, 0), "ref suffix");
    const auto l_fresh = last_logits(ref, n_vocab);
    compare("fork vs fresh (suffix)", l_fork, l_fresh, 1e-3f);

    // 6. fork == copy: seq 2 restored from the FULL state saved at p (own cells, a real copy)
    ck_full.load_tgt(ctx, 2, LLAMA_STATE_SEQ_FLAGS_NONE);
    CHECK(decode_range(ctx, dst_tok, p, p + S, 2), "decode copy suffix");
    const auto l_copy = last_logits(ctx, n_vocab);
    compare("fork vs full-state copy (suffix)", l_fork, l_copy, 1e-3f);

    // 7. source invariance: the source's next token, with and without a fork taken from it
    CHECK(decode_range(ctx, src_tok, N, N + 1, 0), "src continuation");
    const auto l_src = last_logits(ctx, n_vocab);
    CHECK(decode_range(ref2, src_tok, 0, p, 0), "ref2 prefix");
    CHECK(decode_range(ref2, src_tok, p, N, 0), "ref2 rest");
    CHECK(decode_range(ref2, src_tok, N, N + 1, 0), "ref2 continuation");
    const auto l_src_ref = last_logits(ref2, n_vocab);
    compare("source continuation vs no-fork", l_src, l_src_ref, 1e-3f);

    // 8. purge the source (the server's prompt_clear / [TAG_IDLE_SLOT_CLEAR]): only its bit goes
    llama_memory_seq_rm(mem, 2, -1, -1);
    stats d0 = get_stats(ctx);
    common_prompt_checkpoint st_before;
    st_before.update_tgt(ctx, 1, LLAMA_STATE_SEQ_FLAGS_NONE);
    llama_memory_seq_rm(mem, 0, -1, -1);
    common_prompt_checkpoint st_after;
    st_after.update_tgt(ctx, 1, LLAMA_STATE_SEQ_FLAGS_NONE);
    const bool st_same = st_before.data_tgt == st_after.data_tgt;
    fprintf(stderr, "  %-34s %zu bytes, identical = %s\n", "fork state across src purge", st_after.data_tgt.size(), st_same ? "yes" : "no");
    CHECK(st_same, "the purge of the source changed the fork's serialized state");
    stats d = get_stats(ctx);
    print_stats("source purged", d);
    check_invariant("source purged", d);
    if (exact) {
        CHECK(d.pool.n_used == p + S, "after purge used %lld != %d", (long long) d.pool.n_used, p + S);
        CHECK(d.pool.n_shared == 0 && d.priv[1] == p + S && d.priv[0] == 0, "after purge counts");
        CHECK(d0.pool.n_used - d.pool.n_used == (N + 1 - p), "purge freed %lld cells, expected %d",
              (long long) (d0.pool.n_used - d.pool.n_used), N + 1 - p);
    }
    CHECK(llama_memory_seq_pos_max(mem, 1) == p + S - 1, "fork lost positions after purge");

    // the fork's continuation is unchanged by the purge
    CHECK(decode_range(ctx, dst_tok, p + S, p + S + 1, 1), "fork continuation after purge");
    const auto l_after = last_logits(ctx, n_vocab);
    CHECK(decode_range(ref, dst_tok, p + S, p + S + 1, 0), "ref continuation");
    const auto l_after_ref = last_logits(ref, n_vocab);
    compare("fork continuation after src purge", l_after, l_after_ref, INFINITY); // top-1 only, see header

    // 9. seq_keep over shared cells: fork seq 3 from seq 1, keep only seq 1
    CHECK(llama_memory_seq_cp_ext(mem, 1, 3, 0, p, LLAMA_MEMORY_SEQ_CP_FLAGS_ATTN_ONLY), "second share");
    stats e = get_stats(ctx);
    check_invariant("second fork", e);
    if (exact) {
        CHECK(e.pool.n_used == p + S + 1 && e.pool.n_shared == p && e.shr[3] == p, "second fork counts");
    }
    llama_memory_seq_keep(mem, 1);
    stats f = get_stats(ctx);
    print_stats("seq_keep(1)", f);
    check_invariant("seq_keep", f);
    if (exact) {
        CHECK(f.pool.n_used == p + S + 1 && f.pool.n_shared == 0 && f.priv[3] == 0 && f.shr[3] == 0, "seq_keep counts");
    }

    // 10. full removal frees everything
    llama_memory_seq_rm(mem, -1, -1, -1);
    stats g = get_stats(ctx);
    CHECK(g.pool.n_used == 0 && g.pool.n_shared == 0, "clear left %lld cells", (long long) g.pool.n_used);

    llama_free(ctx);
    llama_free(ref);
    llama_free(ref2);
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    run(params, model, 0);
    if (llama_model_is_recurrent(model) || llama_model_is_hybrid(model)) {
        // the production shape keeps a rollback ring per sequence
        run(params, model, 8);
    }

    if (g_fail) {
        fprintf(stderr, "%s : %d check(s) failed\n", __func__, g_fail);
        return 1;
    }

    fprintf(stderr, "%s : OK\n", __func__);
    return 0;
}
