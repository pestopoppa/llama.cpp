#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model prompt [outfile]\n", argv[0]); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_load_model_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_threads = 48; cp.n_threads_batch = 48;
    llama_context * ctx = llama_new_context_with_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // proper two-pass tokenize
    int n = llama_tokenize(vocab, argv[2], (int) strlen(argv[2]), nullptr, 0, true, false);
    if (n < 0) n = -n; // two-pass: negative = required buffer size
    if (n <= 0) { fprintf(stderr, "tokenize failed: %d\n", n); return 1; }
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, argv[2], (int) strlen(argv[2]), toks.data(), n, true, false);
    llama_batch batch = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        batch.token[i] = toks[i]; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
    }
    batch.n_tokens = toks.size();
    int rc = llama_decode(ctx, batch);
    if (rc != 0) { fprintf(stderr, "decode rc=%d\n", rc); return 1; }
    float * logits = llama_get_logits_ith(ctx, toks.size() - 1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (argc > 3) {
        FILE * f = fopen(argv[3], "wb");
        if (!f) { fprintf(stderr, "outfile open failed\n"); return 1; }
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        printf("wrote %d logits to %s\n", n_vocab, argv[3]);
    } else {
        double sum = 0, sumsq = 0; float mx = -1e30f;
        for (int i = 0; i < n_vocab; i++) { sum += logits[i]; sumsq += (double)logits[i]*logits[i]; if (logits[i] > mx) mx = logits[i]; }
        printf("n_tok=%d n_vocab=%d sum=%.4f sumsq=%.4f max=%.4f first8:", (int)toks.size(), n_vocab, sum, sumsq, mx);
        for (int i = 0; i < 8; i++) printf(" %.4f", logits[i]);
        printf("\n");
    }
    llama_batch_free(batch);
    llama_free(ctx); llama_free_model(model); llama_backend_free();
    return 0;
}
