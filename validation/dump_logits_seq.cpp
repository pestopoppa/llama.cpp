// INF-64 validation harness: fused-vs-graph single-token decode logit diff.
// One model, two contexts; both decode the same prompt batch (graph path),
// then K single-token decodes. ctx_g runs with GGML_FUSED_DECODE_OFF=1
// (graph path), ctx_f with the env unset (fused fast path). Greedy tokens
// from the graph ctx feed both contexts. Per-step max-abs-diff + NMSE.
// Optionally compares the batch-decode logits against a reference file
// (argv[3], the logits-ref.bin captured pre-fusion on the same tree).
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <chrono>

static int greedy_argmax(const float * logits, int n) {
    return (int) (std::max_element(logits, logits + n) - logits);
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model prompt [ref.bin] [steps]\n", argv[0]); return 1; }
    const int steps = argc > 4 ? atoi(argv[4]) : 8;
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = false;
    llama_model * model = llama_load_model_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    const int val_threads = getenv("VAL_THREADS") ? atoi(getenv("VAL_THREADS")) : 48;
    cp.n_ctx = 512; cp.n_threads = val_threads; cp.n_threads_batch = val_threads;
    fprintf(stderr, "val threads: %d\n", val_threads);
    llama_context * ctx_g = llama_new_context_with_model(model, cp);
    llama_context * ctx_f = llama_new_context_with_model(model, cp);
    if (!ctx_g || !ctx_f) { fprintf(stderr, "ctx failed\n"); return 1; }

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
    fprintf(stderr, "prompt tokens: %d, vocab: %d\n", np, n_vocab);

    llama_batch batch = llama_batch_init(np, 0, 1);
    for (int i = 0; i < np; i++) {
        batch.token[i] = toks[i]; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
    }
    batch.n_tokens = np;

    // step 0: prompt batch through the graph on BOTH contexts (sanity)
    unsetenv("GGML_FUSED_DECODE");
    if (llama_decode(ctx_g, batch) != 0) { fprintf(stderr, "g batch decode failed\n"); return 1; }
    if (llama_decode(ctx_f, batch) != 0) { fprintf(stderr, "f batch decode failed\n"); return 1; }
    const float * lg = llama_get_logits_ith(ctx_g, np - 1);
    const float * lf = llama_get_logits_ith(ctx_f, np - 1);
    double md = 0.0, sq = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        md = std::max(md, (double) std::fabs(lg[i] - lf[i]));
        sq += (double) (lg[i] - lf[i]) * (lg[i] - lf[i]);
    }
    fprintf(stderr, "step %3d (batch): max_abs=%.3e nmse=%.3e\n", 0, md, sq / n_vocab);

    if (argc > 3) {
        FILE * f = fopen(argv[3], "rb");
        if (f) {
            std::vector<float> ref(n_vocab);
            size_t rd = fread(ref.data(), sizeof(float), n_vocab, f);
            fclose(f);
            if (rd == (size_t) n_vocab) {
                double rmd = 0.0, rsq = 0.0;
                for (int i = 0; i < n_vocab; i++) {
                    rmd = std::max(rmd, (double) std::fabs(lg[i] - ref[i]));
                    rsq += (double) (lg[i] - ref[i]) * (lg[i] - ref[i]);
                }
                fprintf(stderr, "batch vs ref-file: max_abs=%.3e nmse=%.3e\n", rmd, rsq / n_vocab);
            } else {
                fprintf(stderr, "ref file short read (%zu/%d)\n", rd, n_vocab);
            }
        }
    }

    // single-token steps: graph ctx drives the greedy token stream
    const bool ext_off = getenv("GGML_FUSED_DECODE_OFF") != NULL;
    llama_batch sb = llama_batch_init(1, 0, 1);
    for (int s = 1; s <= steps; s++) {
        const int tok = greedy_argmax(lg, n_vocab);
        sb.token[0] = tok; sb.pos[0] = np - 1 + s; sb.n_seq_id[0] = 1; sb.seq_id[0][0] = 0; sb.n_tokens = 1;

        // graph arm: hard-off + opt-in cleared (works with both the pre-A4
        // opt-out hook and the A4 opt-in hook)
        using clk = std::chrono::steady_clock;
        if (!ext_off) { setenv("GGML_FUSED_DECODE_OFF", "1", 1); unsetenv("GGML_FUSED_DECODE"); }
        const auto tg0 = clk::now();
        if (llama_decode(ctx_g, sb) != 0) { fprintf(stderr, "g step %d failed\n", s); return 1; }
        const double g_ms = std::chrono::duration<double, std::milli>(clk::now() - tg0).count();
        // fused arm: opt-in set, hard-off cleared
        if (!ext_off) { unsetenv("GGML_FUSED_DECODE_OFF"); setenv("GGML_FUSED_DECODE", "1", 1); }
        const auto tf0 = clk::now();
        if (llama_decode(ctx_f, sb) != 0) { fprintf(stderr, "f step %d failed\n", s); return 1; }
        const double f_ms = std::chrono::duration<double, std::milli>(clk::now() - tf0).count();
        if (!ext_off) unsetenv("GGML_FUSED_DECODE");
        fprintf(stderr, "TIMING step %3d: graph=%.1f ms fused=%.1f ms ratio=%.2f\n", s, g_ms, f_ms, f_ms / g_ms);

        lg = llama_get_logits_ith(ctx_g, 0);
        lf = llama_get_logits_ith(ctx_f, 0);
        md = 0.0; sq = 0.0;
        for (int i = 0; i < n_vocab; i++) {
            md = std::max(md, (double) std::fabs(lg[i] - lf[i]));
            sq += (double) (lg[i] - lf[i]) * (lg[i] - lf[i]);
        }
        const int tg = greedy_argmax(lg, n_vocab);
        const int tf = greedy_argmax(lf, n_vocab);
        fprintf(stderr, "step %3d: tok=%d max_abs=%.3e nmse=%.3e greedy g=%d f=%d %s\n",
                s, tok, md, sq / n_vocab, tg, tf, tg == tf ? "" : "<- DIVERGE");
    }

    llama_batch_free(batch); llama_batch_free(sb);
    llama_free(ctx_g); llama_free(ctx_f); llama_free_model(model); llama_backend_free();
    return 0;
}
