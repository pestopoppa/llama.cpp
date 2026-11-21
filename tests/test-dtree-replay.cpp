// [DTREE-REPLAY] tree verify + GDN input-log replay vs tree verify + snapshot slots on a real hybrid model.
//
//   test-dtree-replay <model.gguf> [n_threads] [W]
//
// Two contexts on the same model: SNAP (LLAMA_RS_REPLAY unset: one recurrent snapshot per tree column) and REP
// (LLAMA_RS_REPLAY=2: the tree columns are logged as pending rows; tree_accept() keeps the accepted path and the
// next ubatch gathers and replays it). For every tree shape and every accepted path (each leaf choice c: side
// branches, partial paths, the full path; accept depths 1..W on the chain tree), from the same committed state:
//
//   tree verify 1 (W columns) -> tree_accept(path to c) -> tree verify 2 (replays the path) -> tree_accept(path2)
//   -> a 3-token chain decode (replays path2) -> a 1-token decode
//
// Every logits row of every step must be BITWISE identical between SNAP and REP. Intended model: Qwen3.5-0.8B
// (qwen35 GDN hybrid). Run on the CPU (HIP_VISIBLE_DEVICES=-1).

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

static llama_context * make_ctx(llama_model * model, int n_threads, int W, bool replay) {
    if (replay) {
        setenv("LLAMA_RS_REPLAY", "2", 1);
    } else {
        unsetenv("LLAMA_RS_REPLAY");
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    cp.n_rs_seq = (uint32_t) (W - 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    CHECK(ctx);
    unsetenv("LLAMA_RS_REPLAY");
    return ctx;
}

static int g_rows = 0, g_steps = 0;

// decode the same batch on both contexts; every output row must be bitwise identical
static void decode_both(llama_context * a, llama_context * b, batch_t & B, const char * what) {
    CHECK(llama_decode(a, B.b) == 0);
    CHECK(llama_decode(b, B.b) == 0);
    int k = 0;
    for (int i = 0; i < B.b.n_tokens; ++i) {
        if (!B.b.logits[i]) continue;
        const float * la = llama_get_logits_ith(a, i);
        const float * lb = llama_get_logits_ith(b, i);
        if (memcmp(la, lb, sizeof(float)*g_n_vocab) != 0) {
            float d = 0;
            for (int v = 0; v < g_n_vocab; ++v) d = std::max(d, std::abs(la[v] - lb[v]));
            fprintf(stderr, "FAIL: %s row %d differs (max|d| %g)\n", what, i, d);
            exit(1);
        }
        ++k;
    }
    g_rows += k;
    g_steps++;
}

// commit the accepted path on both contexts. A chain-shaped tree is decoded as a plain chain ubatch (llama_batch
// drops all-chain parents), so tree_accept() refuses it on both and the path (a prefix) is committed by seq_rm, as
// the server does.
static int g_seqrm = 0;
static void accept_both(llama_context * a, llama_context * b, const std::vector<int32_t> & p, llama_pos root) {
    const bool ta = llama_memory_tree_accept(llama_get_memory(a), 0, p.data(), (int32_t) p.size());
    const bool tb = llama_memory_tree_accept(llama_get_memory(b), 0, p.data(), (int32_t) p.size());
    CHECK(ta == tb);
    if (!ta) {
        for (size_t i = 0; i < p.size(); ++i) CHECK(p[i] == (int32_t) i);
        CHECK(llama_memory_seq_rm(llama_get_memory(a), 0, root + (llama_pos) p.size(), -1));
        CHECK(llama_memory_seq_rm(llama_get_memory(b), 0, root + (llama_pos) p.size(), -1));
        g_seqrm++;
    }
}

static std::vector<int32_t> path_to(const std::vector<int32_t> & par, int c) {
    std::vector<int32_t> p;
    for (int x = c; x >= 0; x = par[x]) p.insert(p.begin(), x);
    return p;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [n_threads] [W]\n", argv[0]);
        return 2;
    }
    const int n_threads = argc > 2 ? atoi(argv[2]) : 8;
    const int W         = argc > 3 ? atoi(argv[3]) : 12;

    setvbuf(stdout, nullptr, _IOLBF, 0);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    CHECK(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    g_n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * snap = make_ctx(model, n_threads, W, false);
    llama_context * rep  = make_ctx(model, n_threads, W, true);

    const std::string prompt = "The history of the printing press begins in the fifteenth century, when";
    std::vector<llama_token> ptoks(256);
    const int np = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), ptoks.data(), (int) ptoks.size(), true, false);
    CHECK(np > 0);
    ptoks.resize(np);

    // tree tokens: arbitrary but fixed (identity does not need greedy-correct drafts)
    std::vector<llama_token> tokA(W), tokB(W);
    for (int c = 0; c < W; ++c) { tokA[c] = (ptoks[c % np] + 101*c) % g_n_vocab; tokB[c] = (ptoks[(c + 3) % np] + 57*c + 9) % g_n_vocab; }

    // shapes (parent of each column; column 0 is the root)
    std::vector<std::pair<std::string, std::vector<int32_t>>> shapes;
    {
        std::vector<int32_t> chain(W); for (int t = 0; t < W; ++t) chain[t] = t - 1;
        shapes.push_back({ "chain", chain });
        // a deep chain plus one sibling under the root: a real tree, so tree_accept() commits paths of depth 1..W-1
        std::vector<int32_t> chainx = chain; chainx[W - 1] = 0;
        shapes.push_back({ "chainx", chainx });
        std::vector<int32_t> side = { -1, 0, 1, 2, 0, 4, 5, 0 };
        while ((int) side.size() < W) side.push_back((int32_t) side.size() - 1);
        side.resize(W);
        shapes.push_back({ "side", side });
        std::vector<int32_t> mixed(W, -1);
        for (int t = 1; t < W; ++t) mixed[t] = (t % 4 == 0) ? (t / 3) : t - 1;
        shapes.push_back({ "mixed", mixed });
    }

    batch_t B(512);
    int n_paths = 0;
    std::vector<int> depth_seen(W + 1, 0);
    for (const auto & [name, par] : shapes) {
        for (int c = 0; c < W; ++c) {
            // the same committed state on both: prompt prefill (nothing pending afterwards: longer than n_rs_seq + 1)
            llama_memory_clear(llama_get_memory(snap), true);
            llama_memory_clear(llama_get_memory(rep),  true);
            B.clear();
            for (int i = 0; i < np; ++i) B.add(ptoks[i], i, i == np - 1);
            decode_both(snap, rep, B, "prefill");
            llama_pos pos = np;

            const auto depth = [&](int x) { int d = 0; for (; par[x] >= 0; x = par[x]) ++d; return d; };

            // tree verify 1, accept the path to leaf c
            B.clear();
            for (int t = 0; t < W; ++t) B.add(tokA[t], pos + depth(t), true, par[t]);
            decode_both(snap, rep, B, "tree1");
            const std::vector<int32_t> p1 = path_to(par, c);
            depth_seen[p1.size()]++;
            accept_both(snap, rep, p1, pos);
            pos += (llama_pos) p1.size();

            // tree verify 2 replays path 1; accept a different path
            B.clear();
            for (int t = 0; t < W; ++t) B.add(tokB[t], pos + depth(t), true, par[t]);
            decode_both(snap, rep, B, "tree2");
            const std::vector<int32_t> p2 = path_to(par, (W - 1 - c + W) % W);
            accept_both(snap, rep, p2, pos);
            pos += (llama_pos) p2.size();

            // a chain ubatch replays path 2, then a single token replays the chain's pending tokens
            B.clear();
            for (int i = 0; i < 3; ++i) B.add(tokA[(c + i) % W], pos + i, true);
            decode_both(snap, rep, B, "chain3");
            pos += 3;
            B.clear();
            B.add(tokB[c], pos, true);
            decode_both(snap, rep, B, "single");

            printf("%-6s W=%d path1=[", name.c_str(), W);
            for (int x : p1) printf(" %d", x);
            printf(" ] path2=[");
            for (int x : p2) printf(" %d", x);
            printf(" ]: all logits bit-identical\n");
            ++n_paths;
        }
    }
    printf("accept depths covered:");
    for (int m = 1; m <= W; ++m) if (depth_seen[m]) printf(" %d", m);
    printf("\n(chain-shaped trees committed by seq_rm: %d)", g_seqrm);
    printf("\nPASS: tree+replay == tree+snapshot bitwise over %d accepted paths (%d decode steps, %d logits rows)\n",
           n_paths, g_steps, g_rows);

    llama_free(rep);
    llama_free(snap);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
