// test-rs-replay-model.cpp: [RS-REPLAY-Q4X-FIX] graph-level identity of GDN input-log replay against snapshot rollback on a real
// model (all of its recurrent rows: delta-net state, conv history and, for qwen4exp, the PLE conv history), single sequence, CPU.
//
// usage: test-rs-replay-model <model.gguf> [n_rs_seq=4]
// Two contexts on one model: A = snapshots (LLAMA_RS_REPLAY unset), B = replay (LLAMA_RS_REPLAY=2, forced). For every accept count
// a in 0..T (T = n_rs_seq + 1): decode the same prompt, a T-token verify batch, seq_rm back to (prompt + a), then the same 3 single
// tokens and one more T-token verify; the logits of every decoded token must be BITWISE identical between A and B.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0, llama_seq_id sq = 0) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i] = toks[i]; b.pos[i] = p0 + (llama_pos) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = sq; b.logits[i] = 1;
    }
    b.n_tokens = (int) toks.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static llama_context * make_ctx(llama_model * m, uint32_t n_rs_seq, bool replay) {
    if (replay) { setenv("LLAMA_RS_REPLAY", "2", 1); } else { unsetenv("LLAMA_RS_REPLAY"); }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 4; cp.kv_unified = false; cp.n_rs_seq = n_rs_seq;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    llama_context * c = llama_init_from_model(m, cp);
    unsetenv("LLAMA_RS_REPLAY");
    return c;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_rs_seq]\n", argv[0]); return 2; }
    const uint32_t n_rs_seq = argc > 2 ? (uint32_t) atoi(argv[2]) : 4;
    const int T = (int) n_rs_seq + 1;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));

    llama_context * A = make_ctx(m, n_rs_seq, false);
    llama_context * B = make_ctx(m, n_rs_seq, true);
    if (!A || !B) { fprintf(stderr, "context init failed\n"); return 1; }

    // fixed pseudo-random token ids (no tokenizer dependence)
    std::vector<llama_token> prompt, verify, tail, verify2;
    uint32_t x = 12345;
    auto nxt = [&]() { x = x * 1103515245u + 12345u; return (llama_token) (1000 + (x >> 8) % 30000); };
    for (int i = 0; i < 24; i++) prompt.push_back(nxt());
    for (int i = 0; i < T; i++) verify.push_back(nxt());
    for (int i = 0; i < 3; i++) tail.push_back(nxt());
    for (int i = 0; i < T; i++) verify2.push_back(nxt());

    int fails = 0;
    for (int a = 1; a <= T; a++) {
        int step_fail = 0;
        auto cmp = [&](const char * what, int n) {
            for (int i = 0; i < n; i++) {
                const float * la = llama_get_logits_ith(A, i), * lb = llama_get_logits_ith(B, i);
                if (memcmp(la, lb, n_vocab * sizeof(float)) != 0) {
                    int nd = 0; float md = 0;
                    for (int v = 0; v < n_vocab; v++) { float d = la[v] - lb[v]; if (d != 0) { nd++; if (d < 0) d = -d; if (d > md) md = d; } }
                    printf("  accept=%d %s token %d: DIFFERS (%d logits, max |d| %.3g)\n", a, what, i, nd, md);
                    step_fail++;
                    return;
                }
            }
        };
        for (llama_context * c : { A, B }) { llama_memory_clear(llama_get_memory(c), true); }
        bool ok = true;
        for (llama_context * c : { A, B }) ok &= decode(c, prompt, 0);
        cmp("prompt", (int) prompt.size());
        for (llama_context * c : { A, B }) ok &= decode(c, verify, (llama_pos) prompt.size());
        cmp("verify", T);
        const llama_pos keep_to = (llama_pos) prompt.size() + a;
        bool rmA = llama_memory_seq_rm(llama_get_memory(A), 0, keep_to, -1);
        bool rmB = llama_memory_seq_rm(llama_get_memory(B), 0, keep_to, -1);
        if (rmA != rmB) { printf("  accept=%d seq_rm A=%d B=%d\n", a, rmA, rmB); step_fail++; }
        llama_pos p = keep_to;
        for (llama_token t : tail) {
            for (llama_context * c : { A, B }) ok &= decode(c, { t }, p);
            cmp("tail", 1);
            p++;
        }
        for (llama_context * c : { A, B }) ok &= decode(c, verify2, p);
        cmp("verify2", T);
        if (!ok) { printf("  accept=%d decode failed\n", a); step_fail++; }
        // the served pattern: rollback, then IMMEDIATELY the next verify (keep ubatch that replays the kept pending tokens), twice
        {
            for (llama_context * c : { A, B }) { llama_memory_clear(llama_get_memory(c), true); }
            for (llama_context * c : { A, B }) ok &= decode(c, prompt, 0);
            for (llama_context * c : { A, B }) ok &= decode(c, verify, (llama_pos) prompt.size());
            llama_pos q = (llama_pos) prompt.size() + std::max(a, 1);
            for (llama_context * c : { A, B }) llama_memory_seq_rm(llama_get_memory(c), 0, q, -1);
            for (llama_context * c : { A, B }) ok &= decode(c, verify2, q);
            cmp("rb-verify2", T);
            const int b2 = (a * 3 + 1) % T + 1;
            q += b2;
            for (llama_context * c : { A, B }) llama_memory_seq_rm(llama_get_memory(c), 0, q, -1);
            for (llama_context * c : { A, B }) ok &= decode(c, verify, q);
            cmp("rb-verify3", T);
            for (llama_context * c : { A, B }) ok &= decode(c, { tail[0] }, q + T);
            cmp("rb-tail", 1);
        }
        // served multi-slot server, single active slot: seqs 0..3 hold earlier requests, then seq (a % 4) is cleared and decoded
        // alone with rollbacks; its tail cell need not be cell s+min, so find_slot swaps cells
        {
            for (llama_context * c : { A, B }) { llama_memory_clear(llama_get_memory(c), true); }
            for (llama_seq_id sq = 0; sq < 4; sq++) {
                for (llama_context * c : { A, B }) ok &= decode(c, prompt, 0, sq);
                for (llama_context * c : { A, B }) ok &= decode(c, verify, (llama_pos) prompt.size(), sq);
                for (llama_context * c : { A, B }) llama_memory_seq_rm(llama_get_memory(c), 0 + sq, (llama_pos) prompt.size() + 1 + sq % 3, -1);
            }
            const llama_seq_id sq = (llama_seq_id) (a % 4);
            for (llama_context * c : { A, B }) llama_memory_seq_rm(llama_get_memory(c), sq, -1, -1);
            for (llama_context * c : { A, B }) ok &= decode(c, verify2, 0, sq);
            cmp("ms-prompt", T);
            llama_pos q = T;
            for (int r = 0; r < 4; r++) {
                for (llama_context * c : { A, B }) ok &= decode(c, r % 2 ? verify : verify2, q, sq);
                cmp("ms-verify", T);
                const int acc = (a + r) % T + 1;
                q += acc;
                bool ra = llama_memory_seq_rm(llama_get_memory(A), sq, q, -1);
                bool rb = llama_memory_seq_rm(llama_get_memory(B), sq, q, -1);
                if (ra != rb) { printf("  accept=%d ms seq_rm A=%d B=%d\n", a, ra, rb); step_fail++; }
                if (!ra) { break; }
            }
            for (llama_context * c : { A, B }) ok &= decode(c, { tail[1] }, q, sq);
            cmp("ms-tail", 1);
        }
        printf("accept=%d (seq_rm %s): %s\n", a, rmA ? "ok" : "refused", step_fail ? "DIFFERS" : "bit-identical");
        fails += step_fail;
    }
    llama_free(A); llama_free(B); llama_model_free(m); llama_backend_free();
    printf(fails ? "FAIL (%d)\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
