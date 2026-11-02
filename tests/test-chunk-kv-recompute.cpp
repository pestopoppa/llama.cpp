// chunk-KV reuse, phase P1, KPF-62.4: sparse recompute by explicit-position batches, gate G0 (P0_THRESHOLDS section 1).
//
// Run with -m <attention-only gguf> -ctk f32 -ctv f32 [-t N]. With a hybrid / recurrent model the binary asserts the clean
// REFUSAL of llama_chunk_kv_decode_at instead.
//
// Prompt layout (token positions):  [prefix P][chunk A][gap G][chunk B][suffix S]   (sizes from CKV_N_CHUNK, default 96)
// Every arm runs in its own sequence (kv_unified = false, so each sequence has its own KV stream with an identical cell
// layout) and is decoded with the SAME batch partition: segment by segment, each segment split at n_batch from its start.
//
//   ref        plain llama_decode, segment by segment                                      (the champion path)
//   g0a        store bypassed, every token through server_chunk::decode_range with holes = true, i.e.
//              llama_chunk_kv_decode_at                                                     GATE: bit-identical to ref
//   g0a-inj    chunks encoded in isolation (scratch sequence, positions 0..n-1), exported, injected at their offset, then
//              r = 100%: every chunk token removed and recomputed by explicit-position batches
//                                                                                           GATE: bit-identical to ref
//   head       inject, recompute the first W tokens of each chunk (selector v0, CKV_WINDOW, default 16)
//              GATE: the unselected chunk cells are bit-identical before and after the recompute; the recomputed cells'
//              layer-0 K equals ref's (NMSE <= 1e-6: layer 0 depends only on token and position). Suffix logits vs ref
//              are REPORTED (that is the G3 approximation, not a G0 property).
//   none       inject, no recompute (r = 0, the naive baseline): REPORTED.
// Plus the refusal paths of llama_chunk_kv_decode_at (occupied positions, non-consecutive positions, two sequences) and
// the fact that plain llama_decode refuses a hole fill (which is why the call exists).
#undef NDEBUG
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"

#include "server-chunk-recompute.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

using tokens_t = std::vector<llama_token>;

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

static double nmse(const float * x, const float * ref, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) x[i] - (double) ref[i];
        num += d*d;
        den += (double) ref[i]*(double) ref[i];
    }
    return den > 0.0 ? num/den : num;
}

static double nmse(const std::vector<float> & x, const std::vector<float> & ref) {
    if (x.size() != ref.size() || ref.empty()) {
        return INFINITY;
    }
    return nmse(x.data(), ref.data(), ref.size());
}

static bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0);
}

// mean over rows of KL(ref || x), natural log, full vocabulary, f64
static double mean_kl(const std::vector<float> & x, const std::vector<float> & ref, int n_vocab) {
    if (x.size() != ref.size() || ref.empty()) {
        return INFINITY;
    }
    const size_t n_rows = ref.size() / n_vocab;
    double sum = 0.0;
    for (size_t r = 0; r < n_rows; ++r) {
        const float * a = ref.data() + r*n_vocab;
        const float * b = x.data()   + r*n_vocab;
        double ma = -INFINITY, mb = -INFINITY;
        for (int v = 0; v < n_vocab; ++v) { ma = std::max(ma, (double) a[v]); mb = std::max(mb, (double) b[v]); }
        double za = 0.0, zb = 0.0;
        for (int v = 0; v < n_vocab; ++v) { za += exp((double) a[v] - ma); zb += exp((double) b[v] - mb); }
        const double la = ma + log(za), lb = mb + log(zb);
        double kl = 0.0;
        for (int v = 0; v < n_vocab; ++v) {
            const double lpa = (double) a[v] - la;
            const double lpb = (double) b[v] - lb;
            kl += exp(lpa) * (lpa - lpb);
        }
        sum += kl;
    }
    return sum / (double) n_rows;
}

static tokens_t rand_tokens(std::mt19937 & rng, int n, int n_vocab) {
    tokens_t t(n);
    for (auto & x : t) {
        x = (llama_token) (rng() % (uint32_t) n_vocab);
    }
    return t;
}

static bool read_kv(llama_context * ctx, llama_seq_id seq, llama_pos p0, llama_pos p1, std::vector<float> & k, std::vector<float> & v) {
    size_t n_v = 0;
    const size_t n_k = llama_chunk_kv_read_f32(ctx, seq, p0, p1, nullptr, nullptr, &n_v);
    if (n_k == 0) {
        return false;
    }
    k.assign(n_k, 0.0f);
    v.assign(n_v, 0.0f);
    return llama_chunk_kv_read_f32(ctx, seq, p0, p1, k.data(), v.data(), &n_v) == n_k;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_ctx_seq, int n_batch) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = 8;
    cparams.n_ctx      = 8*n_ctx_seq;
    cparams.n_batch    = n_batch;
    cparams.n_ubatch   = n_batch;
    cparams.kv_unified = false;
    return llama_init_from_model(model, cparams);
}

struct layout {
    std::vector<server_chunk::span> chunks;
    int32_t E = 0; // end of the last chunk = first suffix position
    int32_t N = 0; // prompt length
};

// decode the prompt region [0, E) of seq in the ref partition (segment by segment), holes selects the API
static bool decode_region_plain(llama_context * ctx, const tokens_t & prompt, const layout & L, llama_seq_id seq, int n_batch, bool holes) {
    for (const auto & sg : server_chunk::make_segments(L.chunks, 0, L.E)) {
        if (server_chunk::decode_range(ctx, prompt.data(), sg.p0, sg.p1, seq, n_batch, holes && sg.is_chunk) != 0) {
            return false;
        }
    }
    return true;
}

static std::vector<float> suffix_logits(llama_context * ctx, const tokens_t & prompt, const layout & L, llama_seq_id seq, int n_batch, int n_vocab, bool * ok) {
    std::vector<float> out;
    *ok = server_chunk::decode_range(ctx, prompt.data(), L.E, L.N, seq, n_batch, false, &out, n_vocab) == 0;
    return out;
}

static void test_refusal(const common_params & params, llama_model * model, int n_vocab) {
    llama_context * ctx = make_ctx(params, model, 512, 64);
    CHECK(ctx != nullptr);
    if (!ctx) return;

    std::mt19937 rng(7);
    const tokens_t toks = rand_tokens(rng, 36, n_vocab);
    CHECK(server_chunk::decode_range(ctx, toks.data(), 0, 32, 0, 64, false) == 0);

    // hybrid / recurrent memory: the sparse decode is refused, the context keeps decoding
    CHECK(server_chunk::decode_range(ctx, toks.data(), 0, 4, 1, 64, true) == -1);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 1) < 0);
    CHECK(server_chunk::decode_range(ctx, toks.data(), 32, 36, 0, 64, false) == 0);
    llama_free(ctx);
    printf("VERDICT G0 refusal (non-attention memory): llama_chunk_kv_decode_at refused, context usable\n");
}

static void test_recompute(const common_params & params, llama_model * model, int n_vocab) {
    const int n_chunk = std::min(512, std::max(24, env_int("CKV_N_CHUNK", 96)));
    const int n_batch = std::max(8,  env_int("CKV_N_BATCH", 64));
    const int window  = std::max(1,  env_int("CKV_WINDOW", 16));
    const int P = 40, G = 24, S = 8;

    layout L;
    L.chunks = { { P, n_chunk }, { P + n_chunk + G, n_chunk } };
    L.E = P + 2*n_chunk + G;
    L.N = L.E + S;

    uint32_t n_ctx_seq = 512;
    while (n_ctx_seq < (uint32_t) (L.N + 64)) {
        n_ctx_seq *= 2;
    }

    llama_context * ctx = make_ctx(params, model, n_ctx_seq, n_batch);
    CHECK(ctx != nullptr);
    if (!ctx) return;
    CHECK(llama_chunk_kv_supported(ctx, nullptr, 0));

    printf("layout: prefix %d, chunk A [%d,%d), gap %d, chunk B [%d,%d), suffix %d; n_batch %d, window %d\n", P,
            L.chunks[0].offset, L.chunks[0].offset + n_chunk, G, L.chunks[1].offset, L.chunks[1].offset + n_chunk, S, n_batch, window);

    std::mt19937 rng(4242);
    const tokens_t prompt = rand_tokens(rng, L.N, n_vocab);

    const llama_seq_id s_ref = 0, s_g0a = 1, s_inj = 2, s_head = 3, s_none = 4, s_ref2 = 5, s_all = 6, s_scr = 7;
    auto * mem = llama_get_memory(ctx);

    // ref: the champion path, segment partition
    bool ok = decode_region_plain(ctx, prompt, L, s_ref, n_batch, false);
    CHECK(ok);
    const std::vector<float> l_ref = suffix_logits(ctx, prompt, L, s_ref, n_batch, n_vocab, &ok);
    CHECK(ok);
    std::vector<float> k_ref, v_ref;
    CHECK(read_kv(ctx, s_ref, 0, L.E, k_ref, v_ref));

    // G0a: same partition, chunk segments through llama_chunk_kv_decode_at
    CHECK(decode_region_plain(ctx, prompt, L, s_g0a, n_batch, true));
    const std::vector<float> l_g0a = suffix_logits(ctx, prompt, L, s_g0a, n_batch, n_vocab, &ok);
    CHECK(ok);
    std::vector<float> k_g0a, v_g0a;
    CHECK(read_kv(ctx, s_g0a, 0, L.E, k_g0a, v_g0a));
    const bool g0a = bit_equal(l_g0a, l_ref) && bit_equal(k_g0a, k_ref) && bit_equal(v_g0a, v_ref);
    printf("G0a     store bypassed, r=100%% via decode_at: logits %s, K/V %s (logits NMSE=%.3e)\n",
            bit_equal(l_g0a, l_ref) ? "bit-identical" : "DIFFER", bit_equal(k_g0a, k_ref) && bit_equal(v_g0a, v_ref) ? "bit-identical" : "DIFFER",
            nmse(l_g0a, l_ref));
    CHECK(g0a);

    // G0a-aligned: the server's mode "all" (server_chunk::assemble, n_batch-aligned region, then the plain loop) against a
    // plain prefill of the whole prompt from 0 in n_batch pieces (the server's plain partition): K/V of every position
    // must be bit-identical, so every later logit is too.
    bool g0a_al = false;
    {
        CHECK(server_chunk::decode_range(ctx, prompt.data(), 0, L.N, s_ref2, n_batch, false) == 0);
        server_chunk::assemble_stats st;
        int32_t p_done = -1;
        auto no_blob = [](int32_t) { return server_chunk::blob_ref{}; };
        CHECK(server_chunk::assemble(ctx, prompt.data(), L.chunks, 0, L.E, s_all, n_batch, server_chunk::SELECT_ALL, 0, no_blob, st, &p_done) == 0);
        CHECK(p_done == server_chunk::aligned_end(0, L.E, n_batch));
        CHECK(server_chunk::decode_range(ctx, prompt.data(), p_done, L.N, s_all, n_batch, false) == 0);
        std::vector<float> k_a, v_a, k_b, v_b;
        CHECK(read_kv(ctx, s_ref2, 0, L.N, k_a, v_a));
        CHECK(read_kv(ctx, s_all,  0, L.N, k_b, v_b));
        g0a_al = bit_equal(k_a, k_b) && bit_equal(v_a, v_b) && !k_a.empty();
        printf("G0a-al  mode all (aligned region %d of %d tokens by decode_at, rest plain) vs plain prefill: K/V %s\n",
                p_done, L.E, g0a_al ? "bit-identical" : "DIFFER");
        CHECK(g0a_al);
        llama_memory_seq_rm(mem, s_ref2, -1, -1);
        llama_memory_seq_rm(mem, s_all, -1, -1);
    }

    // isolated encodes of the chunks (scratch sequence, positions 0..n-1), exported
    std::vector<std::vector<uint8_t>> blobs;
    for (const auto & sp : L.chunks) {
        CHECK(server_chunk::decode_range(ctx, prompt.data() + sp.offset, 0, sp.n_tokens, s_scr, n_batch, false) == 0);
        const size_t need = llama_chunk_kv_export(ctx, nullptr, 0, s_scr, 0, sp.n_tokens);
        CHECK(need > 0);
        std::vector<uint8_t> blob(need);
        CHECK(llama_chunk_kv_export(ctx, blob.data(), blob.size(), s_scr, 0, sp.n_tokens) == need);
        blobs.push_back(std::move(blob));
        llama_memory_seq_rm(mem, s_scr, -1, -1);
    }
    auto get_blob = [&](int32_t i) { return server_chunk::blob_ref{ blobs[i].data(), blobs[i].size() }; };

    // G0a-inj: inject, then r = 100% by sparse recompute, in server order (each chunk recomputed before what follows it)
    bool g0a_inj = false;
    {
        server_chunk::assemble_stats st;
        int32_t p_done = -1;
        CHECK(server_chunk::assemble(ctx, prompt.data(), L.chunks, 0, L.E, s_inj, n_batch, server_chunk::SELECT_ALL_INJECT, 0, get_blob, st, &p_done) == 0);
        CHECK(p_done == L.E);
        CHECK(st.n_injected == 2*n_chunk && st.n_recomputed == 2*n_chunk && st.n_gap == P + G);
        const std::vector<float> l_inj = suffix_logits(ctx, prompt, L, s_inj, n_batch, n_vocab, &ok);
        CHECK(ok);
        std::vector<float> k_inj, v_inj;
        CHECK(read_kv(ctx, s_inj, 0, L.E, k_inj, v_inj));
        g0a_inj = bit_equal(l_inj, l_ref) && bit_equal(k_inj, k_ref) && bit_equal(v_inj, v_ref);
        printf("G0a-inj inject + r=100%% sparse recompute: logits %s, K/V %s (logits NMSE=%.3e, K NMSE=%.3e)\n",
                bit_equal(l_inj, l_ref) ? "bit-identical" : "DIFFER", bit_equal(k_inj, k_ref) && bit_equal(v_inj, v_ref) ? "bit-identical" : "DIFFER",
                nmse(l_inj, l_ref), nmse(k_inj, k_ref));
        CHECK(g0a_inj);
    }

    // head window, by hand in server order so each recompute can be inspected: the unselected cells are untouched by the
    // recompute and the recomputed cells' layer-0 K equals ref's
    const int n_layer = llama_model_n_layer(model);
    bool head_ok = true;
    for (const auto & sg : server_chunk::make_segments(L.chunks, 0, L.E)) {
        if (!sg.is_chunk) {
            CHECK(server_chunk::decode_range(ctx, prompt.data(), sg.p0, sg.p1, s_head, n_batch, false) == 0);
            continue;
        }
        const auto & sp = L.chunks[sg.i_chunk];
        CHECK(llama_chunk_kv_inject(ctx, blobs[sg.i_chunk].data(), blobs[sg.i_chunk].size(), s_head, sp.offset) == blobs[sg.i_chunk].size());
        const auto runs = server_chunk::select_runs(sp, server_chunk::SELECT_HEAD, window);
        CHECK(runs.size() == 1 && runs[0].p0 == sp.offset && runs[0].n() == std::min(window, sp.n_tokens));
        if (runs.size() != 1) {
            head_ok = false;
            continue;
        }
        const int32_t u0 = sp.offset + runs[0].n(), u1 = sp.offset + sp.n_tokens;

        std::vector<float> k_before, v_before;
        const bool have_unsel = u1 > u0;
        if (have_unsel) {
            CHECK(read_kv(ctx, s_head, u0, u1, k_before, v_before)); // applies the pending K re-rotation first
        }
        CHECK(server_chunk::recompute_runs(ctx, prompt.data(), runs, s_head, n_batch) == 0);
        if (have_unsel) {
            std::vector<float> k_after, v_after;
            CHECK(read_kv(ctx, s_head, u0, u1, k_after, v_after));
            const bool same = bit_equal(k_after, k_before) && bit_equal(v_after, v_before);
            printf("head    chunk@%d: unselected [%d,%d) cells %s by the recompute\n", sp.offset, u0, u1, same ? "untouched" : "CHANGED");
            CHECK(same);
            head_ok = head_ok && same;
        }

        std::vector<float> k_rc, v_rc, k_rf, v_rf;
        CHECK(read_kv(ctx, s_head, runs[0].p0, runs[0].p1, k_rc, v_rc));
        CHECK(read_kv(ctx, s_ref,  runs[0].p0, runs[0].p1, k_rf, v_rf));
        if (!k_rc.empty() && k_rc.size() == k_rf.size()) {
            const size_t per = k_rc.size() / n_layer;
            const double e0 = nmse(k_rc.data(), k_rf.data(), per);
            printf("head    chunk@%d: recomputed [%d,%d) layer-0 K NMSE vs ref = %.3e (gate <= 1e-6); all layers K NMSE = %.3e\n",
                    sp.offset, runs[0].p0, runs[0].p1, e0, nmse(k_rc, k_rf));
            CHECK(e0 <= 1e-6);
            head_ok = head_ok && e0 <= 1e-6;
        } else {
            head_ok = false;
        }
    }
    const std::vector<float> l_head = suffix_logits(ctx, prompt, L, s_head, n_batch, n_vocab, &ok);
    CHECK(ok);

    // naive: shift only (r = 0)
    {
        server_chunk::assemble_stats st;
        int32_t p_done = -1;
        CHECK(server_chunk::assemble(ctx, prompt.data(), L.chunks, 0, L.E, s_none, n_batch, server_chunk::SELECT_NONE, 0, get_blob, st, &p_done) == 0);
        CHECK(p_done == L.E && st.n_recomputed == 0);
    }
    const std::vector<float> l_none = suffix_logits(ctx, prompt, L, s_none, n_batch, n_vocab, &ok);
    CHECK(ok);

    printf("report  suffix vs ref: head W=%d mean KL=%.3e logits NMSE=%.3e | none (r=0) mean KL=%.3e logits NMSE=%.3e\n",
            window, mean_kl(l_head, l_ref, n_vocab), nmse(l_head, l_ref), mean_kl(l_none, l_ref, n_vocab), nmse(l_none, l_ref));

    // refusals of the sparse decode, cache unchanged
    {
        const llama_pos pmax = llama_memory_seq_pos_max(mem, s_none);
        // positions already present
        CHECK(server_chunk::decode_range(ctx, prompt.data(), P, P + 4, s_none, n_batch, true) == -1);
        llama_memory_seq_rm(mem, s_none, P, P + 4);
        llama_batch b = llama_batch_init(4, 0, 2);
        // non-consecutive positions
        for (int i = 0; i < 2; ++i) {
            b.token[i] = prompt[P + 2*i]; b.pos[i] = P + 2*i; b.n_seq_id[i] = 1; b.seq_id[i][0] = s_none; b.logits[i] = 0;
        }
        b.n_tokens = 2;
        CHECK(llama_chunk_kv_decode_at(ctx, b) == -1);
        // two sequences
        for (int i = 0; i < 2; ++i) {
            b.token[i] = prompt[P + i]; b.pos[i] = P + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = i == 0 ? s_none : s_head; b.logits[i] = 0;
        }
        CHECK(llama_chunk_kv_decode_at(ctx, b) == -1);
        llama_batch_free(b);
        CHECK(llama_memory_seq_pos_max(mem, s_none) == pmax);

        // plain llama_decode refuses to fill the hole (Y = X + 1, or X < Y for M-RoPE): the reason the call exists
        CHECK(server_chunk::decode_range(ctx, prompt.data(), P, P + 4, s_none, n_batch, false) != 0);
        // and the sparse decode fills it
        CHECK(server_chunk::decode_range(ctx, prompt.data(), P, P + 4, s_none, n_batch, true) == 0);
        CHECK(llama_memory_seq_pos_max(mem, s_none) == pmax);
    }

    printf("VERDICT G0a (r=100%%, store bypassed, explicit-position batches == champion path, bit-for-bit): %s\n", g0a ? "PASS" : "FAIL");
    printf("VERDICT G0a-al (mode all, n_batch-aligned, == plain prefill partition, K/V bit-for-bit): %s\n", g0a_al ? "PASS" : "FAIL");
    printf("VERDICT G0a-inj (inject + r=100%% sparse recompute == champion path, bit-for-bit): %s\n", g0a_inj ? "PASS" : "FAIL");
    printf("VERDICT G0-head (unselected cells untouched, recomputed layer-0 K == ref): %s\n", head_ok ? "PASS" : "FAIL");

    llama_free(ctx);
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
        test_recompute(params, model, n_vocab);
    } else {
        test_refusal(params, model, n_vocab);
    }

    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("OK (%s)\n", attention_only ? "attention-only" : "refusal");
    return 0;
}
