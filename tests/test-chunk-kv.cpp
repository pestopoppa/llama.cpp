// chunk-KV reuse, phase P1 slice 1: libllama range export / inject, RoPE re-rotation and its self-check.
//
// Run with -m <attention-only gguf> -ctk f32 -ctv f32 (the f32 K cache is what G1 prescribes; a quantized K cache
// re-quantizes on shift and is not an exactness test). With a hybrid / recurrent model the same binary asserts the
// clean REFUSAL instead (return 0 / false, cache and decode untouched), which is the phase-1 contract.
//
// Gates (P0_THRESHOLDS.md, section 2): K/V NMSE <= 1e-6 after a shift with no recompute, logits NMSE <= 1e-5 at the
// suffix, round-trip self-check NMSE <= 1e-6. The measured values are printed so the margin is visible.
#undef NDEBUG
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static double nmse(const std::vector<float> & x, const std::vector<float> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) x[i] - (double) ref[i];
        num += d*d;
        den += (double) ref[i]*(double) ref[i];
    }
    return den > 0.0 ? num/den : num;
}

using tokens_t = std::vector<llama_token>;

static tokens_t rand_tokens(std::mt19937 & rng, int n, int n_vocab) {
    tokens_t t(n);
    for (auto & x : t) {
        x = (llama_token) (rng() % (uint32_t) n_vocab);
    }
    return t;
}

// decode toks at positions pos0.. in seq; returns logits of every token (n x n_vocab) when want_logits
static bool decode(llama_context * ctx, const tokens_t & toks, llama_pos pos0, llama_seq_id seq, int n_vocab,
        std::vector<float> * logits = nullptr) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], pos0 + (llama_pos) i, { seq }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    if (ok && logits) {
        logits->assign((size_t) toks.size()*n_vocab, 0.0f);
        for (size_t i = 0; i < toks.size(); ++i) {
            memcpy(logits->data() + i*n_vocab, llama_get_logits_ith(ctx, (int32_t) i), n_vocab*sizeof(float));
        }
    }
    llama_batch_free(batch);
    return ok;
}

static bool read_kv(llama_context * ctx, llama_seq_id seq, llama_pos p0, llama_pos p1,
        std::vector<float> & k, std::vector<float> & v) {
    size_t n_v = 0;
    const size_t n_k = llama_chunk_kv_read_f32(ctx, seq, p0, p1, nullptr, nullptr, &n_v);
    if (n_k == 0) {
        return false;
    }
    k.assign(n_k, 0.0f);
    v.assign(n_v, 0.0f);
    return llama_chunk_kv_read_f32(ctx, seq, p0, p1, k.data(), v.data(), &n_v) == n_k;
}

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 8;
    cparams.n_ctx     = 8192;
    cparams.n_batch   = 512;
    cparams.n_ubatch  = 512;
    return llama_init_from_model(model, cparams);
}

// the phase-1 contract on memory it cannot serve: refuse everything, leave the context usable
static void test_refusal(const common_params & params, llama_model * model, int n_vocab) {
    llama_context * ctx = make_ctx(params, model);
    CHECK(ctx != nullptr);
    if (!ctx) return;

    char why[256] = {0};
    CHECK(!llama_chunk_kv_supported(ctx, why, sizeof(why)));
    CHECK(why[0] != 0);
    printf("refusal reason: %s\n", why);

    std::mt19937 rng(7);
    const tokens_t toks = rand_tokens(rng, 16, n_vocab);
    CHECK(decode(ctx, toks, 0, 0, n_vocab));

    std::vector<uint8_t> blob(1 << 20);
    CHECK(llama_chunk_kv_export(ctx, nullptr, 0, 0, 0, 16) == 0);
    CHECK(llama_chunk_kv_export(ctx, blob.data(), blob.size(), 0, 0, 16) == 0);
    CHECK(llama_chunk_kv_inject(ctx, blob.data(), blob.size(), 1, 32) == 0);
    double e = -1.0;
    CHECK(!llama_chunk_kv_rope_selfcheck(ctx, 64, &e));
    CHECK(!llama_chunk_kv_apply_shift(ctx) || true); // no pending shift; must not crash

    // the context still decodes after the refusals
    CHECK(decode(ctx, rand_tokens(rng, 4, n_vocab), 16, 0, n_vocab));

    llama_free(ctx);
}

static void test_attention(const common_params & params, llama_model * model, int n_vocab) {
    llama_context * ctx = make_ctx(params, model);
    CHECK(ctx != nullptr);
    if (!ctx) return;

    char why[256] = {0};
    CHECK(llama_chunk_kv_supported(ctx, why, sizeof(why)));

    std::mt19937 rng(1234);

    const int n_chunk  = 24;
    const int n_suffix = 6;

    const tokens_t chunk  = rand_tokens(rng, n_chunk,  n_vocab);
    const tokens_t suffix = rand_tokens(rng, n_suffix, n_vocab);

    // 1. self-check: round trip +d / -d
    for (llama_pos d : { 1, 17, 300, 4096 }) {
        double e = -1.0;
        CHECK(llama_chunk_kv_rope_selfcheck(ctx, d, &e));
        printf("rope round-trip delta=%5d  NMSE=%.3e\n", (int) d, e);
        CHECK(e >= 0.0 && e <= 1e-6);
    }

    // 2. isolated encode at offset 0 in seq 0, export
    std::vector<float> logits_dummy;
    CHECK(decode(ctx, chunk, 0, 0, n_vocab));

    const size_t need = llama_chunk_kv_export(ctx, nullptr, 0, 0, 0, n_chunk);
    CHECK(need > 0);
    std::vector<uint8_t> blob(need);
    CHECK(llama_chunk_kv_export(ctx, blob.data(), blob.size(), 0, 0, n_chunk) == need);

    // export is read-only: the source sequence is intact
    CHECK(llama_memory_seq_pos_min(llama_get_memory(ctx), 0) == 0);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == n_chunk - 1);

    std::vector<float> k_src, v_src;
    CHECK(read_kv(ctx, 0, 0, n_chunk, k_src, v_src));

    // 3. G1 shift exactness for several deltas: inject at delta (seq 1) vs encode in place at delta (seq 2)
    int seq_next = 1;
    for (llama_pos delta : { 1, 17, 300 }) {
        const llama_seq_id s_inj = seq_next++;
        const llama_seq_id s_ref = seq_next++;

        CHECK(llama_chunk_kv_inject(ctx, blob.data(), blob.size(), s_inj, delta) == blob.size());

        // in place, same isolated context (no predecessor), same positions
        CHECK(decode(ctx, chunk, delta, s_ref, n_vocab));

        std::vector<float> k_i, v_i, k_r, v_r;
        CHECK(read_kv(ctx, s_inj, delta, delta + n_chunk, k_i, v_i));
        CHECK(read_kv(ctx, s_ref, delta, delta + n_chunk, k_r, v_r));
        CHECK(k_i.size() == k_r.size() && v_i.size() == v_r.size());

        const double nk = nmse(k_i, k_r);
        const double nv = nmse(v_i, v_r);

        // V is position-free: it must be a bit-exact copy of the source
        CHECK(v_i == v_src);

        // continuation: decode the suffix on both and compare logits
        std::vector<float> l_i, l_r;
        CHECK(decode(ctx, suffix, delta + n_chunk, s_inj, n_vocab, &l_i));
        CHECK(decode(ctx, suffix, delta + n_chunk, s_ref, n_vocab, &l_r));
        const double nl = nmse(l_i, l_r);

        printf("G1 delta=%4d  K NMSE=%.3e  V NMSE=%.3e  logits NMSE=%.3e\n", (int) delta, nk, nv, nl);
        CHECK(nk <= 1e-6);
        CHECK(nv <= 1e-6);
        CHECK(nl <= 1e-5);

        llama_memory_seq_rm(llama_get_memory(ctx), s_inj, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx), s_ref, -1, -1);
        seq_next = 1;
    }

    // 4. delta = 0 is an exact copy (no rotation is queued), also for a range that does not start at 0
    {
        const llama_seq_id s = 7;
        CHECK(llama_chunk_kv_inject(ctx, blob.data(), blob.size(), s, 0) == blob.size());
        std::vector<float> k, v;
        CHECK(read_kv(ctx, s, 0, n_chunk, k, v));
        CHECK(k == k_src);
        CHECK(v == v_src);
        llama_memory_seq_rm(llama_get_memory(ctx), s, -1, -1);

        // [8, 24) of seq 0 re-placed at +100: positions 108..123, V identical to the source's V[8..24)
        const size_t need2 = llama_chunk_kv_export(ctx, nullptr, 0, 0, 8, 24);
        CHECK(need2 > 0 && need2 < need);
        std::vector<uint8_t> blob2(need2);
        CHECK(llama_chunk_kv_export(ctx, blob2.data(), blob2.size(), 0, 8, 24) == need2);
        CHECK(llama_chunk_kv_inject(ctx, blob2.data(), blob2.size(), s, 100) == blob2.size());
        std::vector<float> k2, v2, ks, vs;
        CHECK(read_kv(ctx, s, 108, 124, k2, v2));
        CHECK(read_kv(ctx, 0, 8, 24, ks, vs));
        CHECK(v2 == vs);
        CHECK(llama_memory_seq_pos_min(llama_get_memory(ctx), s) == 108);
        CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), s) == 123);
        llama_memory_seq_rm(llama_get_memory(ctx), s, -1, -1);
    }

    // 5. refusals leave the cache untouched
    {
        const llama_seq_id s = 7;
        CHECK(llama_chunk_kv_inject(ctx, blob.data(), blob.size(), s, 50) == blob.size());
        // overlapping range in the same sequence
        CHECK(llama_chunk_kv_inject(ctx, blob.data(), blob.size(), s, 60) == 0);
        CHECK(llama_memory_seq_pos_min(llama_get_memory(ctx), s) == 50);
        CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), s) == 50 + n_chunk - 1);
        // an adjacent, non-overlapping second chunk is accepted (multi-chunk assembly)
        CHECK(llama_chunk_kv_inject(ctx, blob.data(), blob.size(), s, 50 + n_chunk) == blob.size());
        CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), s) == 50 + 2*n_chunk - 1);
        // negative start
        CHECK(llama_chunk_kv_inject(ctx, blob.data(), blob.size(), 6, -5) == 0);
        // corrupt magic
        std::vector<uint8_t> bad = blob;
        bad[0] ^= 0xff;
        CHECK(llama_chunk_kv_inject(ctx, bad.data(), bad.size(), 6, 5) == 0);
        // truncated blob
        CHECK(llama_chunk_kv_inject(ctx, blob.data(), blob.size() / 2, 6, 5) == 0);
        CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 6) < 0);
        // a hole in the source range refuses the export
        llama_memory_seq_rm(llama_get_memory(ctx), 0, 12, 13);
        CHECK(llama_chunk_kv_export(ctx, nullptr, 0, 0, 8, 24) == 0);
        CHECK(llama_chunk_kv_export(ctx, nullptr, 0, 0, 0, 12) > 0);
        llama_memory_seq_rm(llama_get_memory(ctx), s, -1, -1);
    }

    llama_free(ctx);
}

// OFF means identical: a context that exported and was refused an inject decodes bit-for-bit like one that never
// touched the API
static void test_off_identical(const common_params & params, llama_model * model, int n_vocab) {
    std::mt19937 rng(99);
    const tokens_t a = rand_tokens(rng, 20, n_vocab);
    const tokens_t b = rand_tokens(rng, 8,  n_vocab);

    llama_context * c1 = make_ctx(params, model);
    llama_context * c2 = make_ctx(params, model);
    CHECK(c1 && c2);
    if (!c1 || !c2) return;

    std::vector<float> l1, l2;
    CHECK(decode(c1, a, 0, 0, n_vocab));
    CHECK(decode(c2, a, 0, 0, n_vocab));

    std::vector<uint8_t> blob(1 << 22);
    const size_t n = llama_chunk_kv_export(c2, blob.data(), blob.size(), 0, 0, 20);
    (void) n;
    CHECK(llama_chunk_kv_inject(c2, blob.data(), 12, 1, 5) == 0); // truncated -> refused

    CHECK(decode(c1, b, 20, 0, n_vocab, &l1));
    CHECK(decode(c2, b, 20, 0, n_vocab, &l2));
    CHECK(l1.size() == l2.size() && memcmp(l1.data(), l2.data(), l1.size()*sizeof(float)) == 0);

    llama_free(c1);
    llama_free(c2);
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

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const bool attention_only = !llama_model_is_recurrent(model) && !llama_model_is_hybrid(model);

    if (attention_only) {
        test_attention(params, model, n_vocab);
    } else {
        test_refusal(params, model, n_vocab);
    }
    test_off_identical(params, model, n_vocab);

    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("OK (%s)\n", attention_only ? "attention-only" : "refusal");
    return 0;
}
