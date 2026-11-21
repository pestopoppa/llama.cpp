// C17-FIX-LAZY regression: the deferred (lazy) logits download of AK-GAP-1-SAMPLE (LLAMA_DEVICE_ACCEPT, default
// ON since C17) must survive a multi-ubatch decode in which every output lands in the first ubatch.
// C17 @ ed5d94969 aborted with "tensor read out of bounds" (ggml-backend.cpp) in logits_lazy_flush: the lazy
// pointer outlived the next ubatch's graph rebuild. Reproducer = the CH-33 non-causal short case
// (n_seq_max=3, kv_unified=false -> split_equal, logits for seq 0 only) plus a causal multi-slot case
// (seq 0 asks for its logits, seqs 1-2 are output-less prompt chunks, as in a llama-server np>1 batch).
//
//   test-device-accept-multiseq <model.gguf> [threads=4] [ngl=0]
//
// The process leaves LLAMA_DEVICE_ACCEPT at its default (ON). For each case it decodes the multi-seq batch, reads
// every requested logits row (forcing the deferred download) and compares it to the same sequence decoded alone
// in a fresh context: max cosine distance must be <= 1e-3 (the CH-33 threshold). Exit 0 iff both cases pass.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * T  = "Explain why a shared key value cache in a language model server requires careful isolation "
                         "between requests that arrive at the same time from different users.";
static const char * SH = "Cache isolation.";

static std::vector<llama_token> tok(const llama_vocab * v, const char * s) {
    std::vector<llama_token> t(512);
    const int n = llama_tokenize(v, s, (int) strlen(s), t.data(), (int) t.size(), true, false);
    t.resize(n > 0 ? n : 0);
    return t;
}

// decode seqs (seq 0 outputs all its tokens) in one llama_decode; return seq-0 logits rows
static bool run(llama_model * model, int nth, bool causal, const std::vector<std::vector<llama_token>> & seqs,
                std::vector<std::vector<float>> & out) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1536; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 3; cp.kv_unified = false;
    cp.embeddings = false; cp.n_threads = nth; cp.n_threads_batch = nth;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return false;
    }
    llama_set_causal_attn(ctx, causal);
    llama_batch b = llama_batch_init(1536, 0, 3);
    std::vector<int> idx;
    for (size_t s = 0; s < seqs.size(); ++s) {
        for (size_t i = 0; i < seqs[s].size(); ++i) {
            const int k = b.n_tokens++;
            b.token[k] = seqs[s][i]; b.pos[k] = (llama_pos) i; b.n_seq_id[k] = 1; b.seq_id[k][0] = (llama_seq_id) s;
            b.logits[k] = s == 0;
            if (s == 0) {
                idx.push_back(k);
            }
        }
    }
    const int rc = llama_decode(ctx, b);
    bool ok = rc == 0;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int k : idx) {
        if (!ok) {
            break;
        }
        const float * l = llama_get_logits_ith(ctx, k);
        if (!l) {
            ok = false;
            break;
        }
        out.emplace_back(l, l + nv);
    }
    llama_batch_free(b);
    llama_free(ctx);
    return ok;
}

static double cosd(const std::vector<float> & a, const std::vector<float> & b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += (double) a[i] * b[i]; aa += (double) a[i] * a[i]; bb += (double) b[i] * b[i];
    }
    return 1.0 - ab / (std::sqrt(aa) * std::sqrt(bb) + 1e-30);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [threads] [ngl]\n", argv[0]);
        return 2;
    }
    const int nth = argc > 2 ? atoi(argv[2]) : 4;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = argc > 3 ? atoi(argv[3]) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 3;
    }
    const llama_vocab * v = llama_model_get_vocab(model);
    const auto t = tok(v, T), sh = tok(v, SH);
    int fails = 0;
    for (int causal = 0; causal <= 1; ++causal) {
        std::vector<std::vector<float>> solo, multi;
        const bool ok_s = run(model, nth, causal, {t}, solo);
        const bool ok_m = run(model, nth, causal, {t, sh, sh}, multi);
        double mx = -1;
        if (ok_s && ok_m && solo.size() == multi.size()) {
            for (size_t i = 0; i < solo.size(); ++i) {
                mx = std::fmax(mx, cosd(solo[i], multi[i]));
            }
        }
        const bool pass = ok_s && ok_m && mx >= 0 && mx <= 1e-3;
        fprintf(stdout, "case %s: solo_ok=%d multi_ok=%d rows=%zu max_cosd=%.3g %s\n", causal ? "causal" : "noncausal",
                ok_s, ok_m, multi.size(), mx, pass ? "PASS" : "FAIL");
        fails += !pass;
    }
    llama_model_free(model);
    llama_backend_free();
    return fails ? 1 : 0;
}
