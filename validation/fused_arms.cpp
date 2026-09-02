// INF-70 Axis A measurement harness.
//
// One model load, one process, one window: a graph-path context plus one fused
// context per configuration arm. Each decode step runs the graph arm first (the
// control) and then every fused arm, timing each and reporting the per-step
// max-abs logit difference and NMSE against the graph arm.
//
// Arms are selected by env at decode time, which is exactly how the fused path
// reads its switches (once per token, in fused_decode):
//   C : GGML_FUSED_MM_LEGACY=1 GGML_FUSED_ARENA_OFF=1   -- pre-A1/A2 behaviour
//   B : GGML_FUSED_MM_LEGACY=1                          -- +A2 arena
//   A : (none)                                          -- +A1 batched mul_mat
//
// usage: fused_arms <model> <prompt> [steps]   (env VAL_THREADS, default 1)
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <chrono>

using clk = std::chrono::steady_clock;

struct Arm {
    const char * name;
    bool mm_legacy;
    bool arena_off;
    llama_context * ctx = nullptr;
    double ms_sum = 0.0;
    int    n = 0;
};

static void set_arm_env(const Arm & a) {
    setenv("GGML_FUSED_DECODE", "1", 1);
    unsetenv("GGML_FUSED_DECODE_OFF");
    if (a.mm_legacy) setenv("GGML_FUSED_MM_LEGACY", "1", 1); else unsetenv("GGML_FUSED_MM_LEGACY");
    if (a.arena_off) setenv("GGML_FUSED_ARENA_OFF", "1", 1); else unsetenv("GGML_FUSED_ARENA_OFF");
}
static void set_graph_env() {
    unsetenv("GGML_FUSED_DECODE");
    setenv("GGML_FUSED_DECODE_OFF", "1", 1);
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model prompt [steps]\n", argv[0]); return 1; }
    const int steps = argc > 3 ? atoi(argv[3]) : 8;
    const int nthr  = getenv("VAL_THREADS") ? atoi(getenv("VAL_THREADS")) : 48;

    set_graph_env();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_threads = nthr; cp.n_threads_batch = nthr;

    llama_context * ctx_g = llama_init_from_model(model, cp);
    // Order matters: the pre-A1 arm restores the per-row reads (including the
    // misread of the repacked IQ4_NL hc loras) and is the one most likely to
    // abort, so it runs LAST — a crash there still leaves A and B measured.
    std::vector<Arm> arms = {
        { "A_batched",  false, false },
        { "B_arena",    true,  false },
        { "C_pre_A1A2", true,  true  },
    };
    if (const char * only = getenv("VAL_ONLY_ARM")) {
        std::vector<Arm> keep;
        for (auto & a : arms) if (strcmp(a.name, only) == 0) keep.push_back(a);
        arms = keep;
    }
    for (auto & a : arms) a.ctx = llama_init_from_model(model, cp);
    if (!ctx_g) { fprintf(stderr, "ctx failed\n"); return 1; }
    for (auto & a : arms) if (!a.ctx) { fprintf(stderr, "ctx %s failed\n", a.name); return 1; }
    fprintf(stderr, "threads=%d steps=%d arms=%zu\n", nthr, steps, arms.size());

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks;
    {
        int n = llama_tokenize(vocab, argv[2], (int) strlen(argv[2]), nullptr, 0, true, false);
        if (n < 0) n = -n;
        toks.resize(n);
        llama_tokenize(vocab, argv[2], (int) strlen(argv[2]), toks.data(), n, true, false);
    }
    const int np = (int) toks.size();
    fprintf(stderr, "prompt tokens: %d vocab: %d\n", np, n_vocab);

    llama_batch batch = llama_batch_init(np, 0, 1);
    for (int i = 0; i < np; i++) {
        batch.token[i] = toks[i]; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
    }
    batch.n_tokens = np;

    // the prompt batch goes through the GRAPH on every context (n_tokens > 1
    // never takes the fused hook, but keep the env explicit)
    set_graph_env();
    if (llama_decode(ctx_g, batch) != 0) { fprintf(stderr, "g batch failed\n"); return 1; }
    for (auto & a : arms) if (llama_decode(a.ctx, batch) != 0) { fprintf(stderr, "%s batch failed\n", a.name); return 1; }

    std::vector<float> lg(n_vocab);
    memcpy(lg.data(), llama_get_logits_ith(ctx_g, np - 1), n_vocab * sizeof(float));

    llama_batch sb = llama_batch_init(1, 0, 1);
    double g_ms_sum = 0.0;
    for (int s = 1; s <= steps; s++) {
        const int tok = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        sb.token[0] = tok; sb.pos[0] = np - 1 + s; sb.n_seq_id[0] = 1; sb.seq_id[0][0] = 0; sb.n_tokens = 1;

        set_graph_env();
        const auto t0 = clk::now();
        if (llama_decode(ctx_g, sb) != 0) { fprintf(stderr, "g step %d failed\n", s); return 1; }
        const double g_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        g_ms_sum += g_ms;
        memcpy(lg.data(), llama_get_logits_ith(ctx_g, 0), n_vocab * sizeof(float));
        const int tg = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        fprintf(stderr, "step %2d tok=%-7d GRAPH  %9.1f ms  greedy=%d\n", s, tok, g_ms, tg);

        for (auto & a : arms) {
            set_arm_env(a);
            const auto ta = clk::now();
            if (llama_decode(a.ctx, sb) != 0) { fprintf(stderr, "%s step %d failed\n", a.name, s); return 1; }
            const double ms = std::chrono::duration<double, std::milli>(clk::now() - ta).count();
            a.ms_sum += ms; a.n++;
            const float * lf = llama_get_logits_ith(a.ctx, 0);
            double md = 0.0, sq = 0.0;
            for (int i = 0; i < n_vocab; i++) {
                const double d = (double) lg[i] - (double) lf[i];
                md = std::max(md, std::fabs(d)); sq += d * d;
            }
            const int tf = (int) (std::max_element(lf, lf + n_vocab) - lf);
            fprintf(stderr, "step %2d          %-11s %9.1f ms  x%.2f  max_abs=%.3e nmse=%.3e greedy=%d %s\n",
                    s, a.name, ms, ms / g_ms, md, sq / n_vocab, tf, tf == tg ? "" : "<- DIVERGE");
            fflush(stderr);
        }
    }
    set_graph_env();
    fprintf(stderr, "\nSUMMARY threads=%d steps=%d: GRAPH mean %.1f ms/token\n", nthr, steps, g_ms_sum / steps);
    for (auto & a : arms) {
        fprintf(stderr, "SUMMARY   %-11s mean %9.1f ms/token  x%.2f vs graph\n",
                a.name, a.ms_sum / a.n, (a.ms_sum / a.n) / (g_ms_sum / steps));
    }

    llama_batch_free(batch); llama_batch_free(sb);
    for (auto & a : arms) llama_free(a.ctx);
    llama_free(ctx_g); llama_model_free(model); llama_backend_free();
    return 0;
}
