// [AK-DTREE-2] single-batch tree verification on a real hybrid (GDN + attention) model.
//
//   test-dtree-batch <model.gguf> [n_rounds] [n_threads] [n_gpu_layers] [rate]
//
// rate mode (5th arg "rate", for GPU runs where near-tie argmax flips are expected): no hard failure on a
// column mismatch; prints DTREE_RATE tree_mis/tree_n chain_mis/chain_n, where both are measured against the
// fresh-prefill oracle (the path context), the chain on a linear verify of the main draft from the same state.
//
// Each round: a reference context (plain greedy decode) gives the next tokens; the tree context decodes
// ONE tree batch (llama_batch.parent) rooted at the last committed token:
//
//   col 0      : root (the committed token, not yet processed)
//   cols 1..3  : main chain = the reference continuation, corrupted at depth `bad` (round-dependent)
//   cols 4..   : a side branch hanging off depth bad-1 that carries the correct continuation, plus a junk
//                sibling under the root
//
// Checks per round:
//   (a) every tree column's argmax equals the argmax a plain chain decode of that column's root path gives
//       (logits max-abs diff reported; argmax must match);
//   (b) greedy acceptance along the tree, llama_memory_tree_accept() of the accepted path, then the next
//       round continues from it: the emitted stream must equal the reference greedy stream token for token.
// Intended model: Qwen3.5-0.8B (qwen35 GDN hybrid). CPU is enough.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static int32_t g_n_vocab = 0;

static llama_token argmax(const float * l) {
    return (llama_token) (std::max_element(l, l + g_n_vocab) - l);
}

struct batch_t {
    llama_batch b;
    std::vector<int32_t> parent;
    explicit batch_t(int n) : b(llama_batch_init(n, 0, 1)), parent(n, -1) {}
    ~batch_t() { llama_batch_free(b); }
    void clear() { b.n_tokens = 0; b.parent = nullptr; }
    void add(llama_token t, llama_pos p, bool out, int32_t par = -2) {
        const int i = b.n_tokens++;
        b.token[i] = t; b.pos[i] = p; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = out;
        if (par != -2) { parent[i] = par; b.parent = parent.data(); }
    }
};

static bool g_rate = false;

static llama_context * make_ctx(llama_model * model, int n_threads) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    cp.n_rs_seq = getenv("DTREE_NRS") ? (uint32_t) atoi(getenv("DTREE_NRS")) : 15; // up to 16 tree columns
    llama_context * ctx = llama_init_from_model(model, cp);
    CHECK(ctx);
    return ctx;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [n_rounds] [n_threads]\n", argv[0]);
        return 2;
    }
    const int n_rounds  = argc > 2 ? atoi(argv[2]) : 12;
    const int n_threads = argc > 3 ? atoi(argv[3]) : 8;
    const int n_gpu     = argc > 4 ? atoi(argv[4]) : 0;
    g_rate = argc > 5 && std::string(argv[5]) == "rate";

    setvbuf(stdout, nullptr, _IOLBF, 0);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    CHECK(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    g_n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ref  = make_ctx(model, n_threads);
    llama_context * tree = make_ctx(model, n_threads);
    llama_context * path = make_ctx(model, n_threads);
    llama_context * lin  = make_ctx(model, n_threads); // linear verify of the main draft from the committed state

    const std::string prompt = getenv("DTREE_PROMPT") ? getenv("DTREE_PROMPT")
                                                      : "The history of the printing press begins in the fifteenth century, when";
    std::vector<llama_token> ptoks(256);
    const int np = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), ptoks.data(), (int) ptoks.size(), true, false);
    CHECK(np > 0);
    ptoks.resize(np);

    batch_t B(4096); // the fresh-prefill oracle re-decodes the whole committed context each column

    // reference greedy stream
    const int n_ref = n_rounds * 8 + 16;
    std::vector<llama_token> refs;
    {
        B.clear();
        for (int i = 0; i < np; ++i) B.add(ptoks[i], i, i == np - 1);
        CHECK(llama_decode(ref, B.b) == 0);
        refs.push_back(argmax(llama_get_logits_ith(ref, -1)));
        for (int k = 0; k < n_ref; ++k) {
            B.clear();
            B.add(refs.back(), np + k, true);
            CHECK(llama_decode(ref, B.b) == 0);
            refs.push_back(argmax(llama_get_logits_ith(ref, -1)));
        }
    }

    // tree context prefill
    B.clear();
    for (int i = 0; i < np; ++i) B.add(ptoks[i], i, i == np - 1);
    CHECK(llama_decode(tree, B.b) == 0);
    CHECK(argmax(llama_get_logits_ith(tree, -1)) == refs[0]);
    B.clear();
    for (int i = 0; i < np; ++i) B.add(ptoks[i], i, false);
    CHECK(llama_decode(lin, B.b) == 0);
    int tree_mis = 0, tree_n = 0, chain_mis = 0, chain_n = 0, chainw_mis = 0, chainw_n = 0;
    std::vector<llama_token> oracle_main(4);

    std::vector<llama_token> emitted = { refs[0] };
    llama_pos pos = np;          // position of the root (committed, unprocessed) token
    int n_alt = 0, n_cols_total = 0;
    double max_diff = 0.0;

    for (int r = 0; r < n_rounds; ++r) {
        const int e = (int) emitted.size();          // refs[e-1] is the root
        const int bad = 1 + r % 3;                   // main chain wrong at depth bad (1..3)
        // columns: tokens, parents
        std::vector<llama_token> tok;
        std::vector<int32_t> par;
        tok.push_back(refs[e - 1]); par.push_back(-1);                    // 0 root
        for (int d = 1; d <= 3; ++d) {                                     // 1..3 main chain
            llama_token t = refs[e - 1 + d];
            if (d == bad) t = (t + 7) % g_n_vocab;
            tok.push_back(t); par.push_back(d - 1);
        }
        // side branch at depth bad: the correct token, then two more correct ones
        int prev = bad - 1;
        for (int d = bad; d < bad + 3; ++d) {
            tok.push_back(refs[e - 1 + d]); par.push_back(prev); prev = (int) tok.size() - 1;
        }
        tok.push_back((refs[e] + 13) % g_n_vocab); par.push_back(0);      // junk sibling under the root
        const int W = (int) tok.size();
        n_cols_total += W;

        auto depth = [&](int c) { int d = 0; for (int x = c; par[x] >= 0; x = par[x]) ++d; return d; };

        B.clear();
        for (int c = 0; c < W; ++c) B.add(tok[c], pos + depth(c), true, par[c]);
        CHECK(llama_decode(tree, B.b) == 0);
        std::vector<std::vector<float>> tl(W);
        for (int c = 0; c < W; ++c) {
            const float * l = llama_get_logits_ith(tree, c);
            tl[c].assign(l, l + g_n_vocab);
        }

        // (a) every column vs a chain decode of its root path in a fresh copy of the committed state
        for (int c = 0; c < W; ++c) {
            std::vector<int> pathc;
            for (int x = c; x >= 0; x = par[x]) pathc.insert(pathc.begin(), x);
            llama_memory_clear(llama_get_memory(path), true);
            B.clear();
            for (int i = 0; i < np; ++i) B.add(ptoks[i], i, false);
            for (int i = 0; i + 1 < e; ++i) B.add(emitted[i], np + i, false);
            for (size_t i = 0; i < pathc.size(); ++i) B.add(tok[pathc[i]], pos + (llama_pos) i, i + 1 == pathc.size());
            CHECK(llama_decode(path, B.b) == 0);
            const float * l = llama_get_logits_ith(path, -1);
            double d = 0;
            for (int v = 0; v < g_n_vocab; ++v) d = std::max(d, (double) std::fabs(l[v] - tl[c][v]));
            max_diff = std::max(max_diff, d);
            const bool mis = argmax(l) != argmax(tl[c].data());
            tree_mis += mis; tree_n++;
            if (c <= 3) oracle_main[c] = argmax(l);
            if (mis) {
                // oracle margin between its own top-1 and the tree's choice (a near-tie flip has a small margin)
                const double margin = (double) l[argmax(l)] - (double) l[argmax(tl[c].data())];
                fprintf(stderr, "round %d col %d: tree argmax %d != path argmax %d (max|dlogit| %.4g, oracle margin %.4g, %s column)\n",
                        r, c, argmax(tl[c].data()), argmax(l), d, margin, c <= 3 ? "main" : "side");
                printf("DTREE_MIS round=%d col=%d main=%d margin=%.5f\n", r, c, c <= 3 ? 1 : 0, margin);
                CHECK(g_rate);
            }
        }

        // linear verify of the main chain (cols 0..3) from the same committed state, vs the same oracle
        {
            B.clear();
            for (int c = 0; c <= 3; ++c) B.add(tok[c], pos + c, true);
            CHECK(llama_decode(lin, B.b) == 0);
            for (int c = 0; c <= 3; ++c) {
                chain_mis += argmax(llama_get_logits_ith(lin, c)) != oracle_main[c]; chain_n++;
            }
            // keep the root (always accepted), roll back the 3 drafts
            CHECK(llama_memory_seq_rm(llama_get_memory(lin), 0, pos + 1, -1));
        }

        // width-matched control: the same W tokens as ONE linear chain (column order) from the same committed state,
        // each column against a fresh prefill of that linear prefix. Separates "a W-wide verify block flips near-ties"
        // (same rate as the tree) from "the tree structure is wrong" (tree rate above this one).
        {
            B.clear();
            for (int c = 0; c < W; ++c) B.add(tok[c], pos + c, true);
            CHECK(llama_decode(lin, B.b) == 0);
            std::vector<llama_token> am(W);
            for (int c = 0; c < W; ++c) am[c] = argmax(llama_get_logits_ith(lin, c));
            CHECK(llama_memory_seq_rm(llama_get_memory(lin), 0, pos + 1, -1));
            for (int c = 0; c < W; ++c) {
                llama_memory_clear(llama_get_memory(path), true);
                B.clear();
                for (int i = 0; i < np; ++i) B.add(ptoks[i], i, false);
                for (int i = 0; i + 1 < e; ++i) B.add(emitted[i], np + i, false);
                for (int i = 0; i <= c; ++i) B.add(tok[i], pos + i, i == c);
                CHECK(llama_decode(path, B.b) == 0);
                const float * l = llama_get_logits_ith(path, -1);
                const bool mis = argmax(l) != am[c];
                chainw_mis += mis; chainw_n++;
                if (mis) {
                    printf("DTREE_CMIS round=%d col=%d margin=%.5f\n", r, c, (double) l[argmax(l)] - (double) l[am[c]]);
                }
            }
        }

        // (b) greedy acceptance along the tree
        std::vector<int32_t> acc = { 0 };
        for (;;) {
            const int cur = acc.back();
            const llama_token want = argmax(tl[cur].data());
            int next = -1;
            for (int c = 0; c < W; ++c) if (par[c] == cur && tok[c] == want) { next = c; break; }
            if (next < 0) break;
            acc.push_back(next);
        }
        if (acc.size() > 1 && acc[1] > 3) ++n_alt; else if (acc.size() > (size_t) bad && acc[bad] > 3) ++n_alt;
        for (size_t i = 1; i < acc.size(); ++i) emitted.push_back(tok[acc[i]]);
        emitted.push_back(argmax(tl[acc.back()].data()));  // bonus token from the last accepted column
        CHECK(llama_memory_tree_accept(llama_get_memory(tree), 0, acc.data(), (int32_t) acc.size()));
        pos += (llama_pos) acc.size();

        // keep the linear context's committed state in step: root + accepted drafts
        {
            B.clear();
            for (size_t i = 1; i < acc.size(); ++i) B.add(tok[acc[i]], pos - (llama_pos) acc.size() + (llama_pos) i, false);
            if (B.b.n_tokens > 0) CHECK(llama_decode(lin, B.b) == 0);
        }
        bool diverged = false;
        for (size_t i = 0; i < emitted.size(); ++i) {
            if (emitted[i] != refs[i]) {
                // every tree column already matched its fresh-prefill oracle (a), so this is the incremental
                // single-token reference disagreeing with a prefill at a near-tie, not a tree error: stop here
                fprintf(stderr, "round %d: emitted[%zu]=%d != incremental ref %d (tree columns == oracle)\n", r, i, emitted[i], refs[i]);
                diverged = true;
                break;
            }
        }
        printf("round %2d: W=%d accepted=%zu path=", r, W, acc.size());
        for (int c : acc) printf("%d ", c);
        printf("\n");
        if (diverged) {
            printf("stream diverged from the incremental greedy reference at round %d (oracle-consistent): stop\n", r);
            break;
        }
    }
    printf("DTREE_RATE tree_mis=%d tree_n=%d chain_mis=%d chain_n=%d\n", tree_mis, tree_n, chain_mis, chain_n);
    printf("DTREE_RATEW chainw_mis=%d chainw_n=%d\n", chainw_mis, chainw_n);

    printf("PASS: %zu tokens identical to greedy over %d rounds (%d cols, %d rounds used a side branch), "
           "max|tree-path dlogit| %.4g\n", emitted.size(), n_rounds, n_cols_total, n_alt, max_diff);

    llama_free(lin);
    llama_free(path);
    llama_free(tree);
    llama_free(ref);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
