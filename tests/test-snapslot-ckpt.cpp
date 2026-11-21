// [KPF-64-SNAPSLOT-COPY] restore-equivalence test for the snapshot-slot n-4 checkpoint.
//
// Cut path (oracle, what llama-server does today): decode the prompt up to N-4, take a PARTIAL_ONLY checkpoint.
// Copy path: decode all N tokens, llama_memory_seq_snap_begin(4), take the PARTIAL_ONLY checkpoint, _snap_end.
// Restore each checkpoint into a context that holds the full attention KV of the prompt, trim the attention KV to
// N-4, replay the last 4 prompt tokens plus a teacher-forced continuation, and compare the next-token
// distributions of the two restores (KLD, top-1). Also checks that snap_begin/_end leaves the live state
// bit-identical, and that the guard refuses when the last ubatch did not write slot 4.
//
// Not byte-identical by design: slot 4 comes from the per-token tail of the chunked scan, the cut path's state is a
// chunked scan that ends at N-4. Gate: top-1 agreement == 100% and mean KLD <= KLD_MAX (default 1e-3, override with
// SNAPSLOT_KLD_MAX).

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_ubatch) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch, (uint32_t) 1024);
    cparams.n_ubatch  = n_ubatch;
    return llama_init_from_model(model, cparams);
}

// decode tokens[p0, p1) at positions p0.., logits for every row when all_logits
static bool decode_range(llama_context * ctx, const std::vector<llama_token> & tokens, int p0, int p1, bool all_logits) {
    llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
    for (int p = p0; p < p1; ++p) {
        common_batch_add(batch, tokens[p], p, { 0 }, all_logits || p + 1 == p1);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static void softmax(const float * x, int n, std::vector<double> & p) {
    p.resize(n);
    double mx = x[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, (double) x[i]);
    double s = 0;
    for (int i = 0; i < n; ++i) { p[i] = std::exp((double) x[i] - mx); s += p[i]; }
    for (int i = 0; i < n; ++i) p[i] /= s;
}

static int argmax(const float * x, int n) {
    return (int) (std::max_element(x, x + n) - x);
}

// logits of rows [0, n) after replaying tokens[N-4, N+M) on top of a restored checkpoint
static bool restore_and_replay(llama_context * ctx, const common_prompt_checkpoint & ck, const std::vector<llama_token> & all,
                               int N, int n_vocab, std::vector<std::vector<float>> & out) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);
    // full prompt first, so the attention KV of 0..N-1 exists (as on a server slot), then the recurrent checkpoint
    if (!decode_range(ctx, all, 0, N, false)) return false;
    if (!ck.load_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) return false;
    if (!llama_memory_seq_rm(mem, 0, N - 4, -1)) {
        fprintf(stderr, "seq_rm(N-4) after restore failed\n");
        return false;
    }
    if (llama_memory_seq_pos_max(mem, 0) != N - 5) {
        fprintf(stderr, "restored pos_max = %d, want %d\n", llama_memory_seq_pos_max(mem, 0), N - 5);
        return false;
    }
    const int n_rows = (int) all.size() - (N - 4);
    if (!decode_range(ctx, all, N - 4, (int) all.size(), true)) return false;
    out.assign(n_rows, std::vector<float>(n_vocab));
    for (int i = 0; i < n_rows; ++i) {
        const float * l = llama_get_logits_ith(ctx, i);
        if (!l) return false;
        std::memcpy(out[i].data(), l, sizeof(float) * n_vocab);
    }
    return true;
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
        fprintf(stderr, "failed to init model\n");
        return 1;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "skipping for non-recurrent model\n");
        return 0;
    }
    const double kld_max = getenv("SNAPSLOT_KLD_MAX") ? atof(getenv("SNAPSLOT_KLD_MAX")) : 1e-3;

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    int n_fail = 0;
    // (prompt text repeats, n_ubatch): single-ubatch, multi-ubatch with split_equal tail, small ubatch
    const struct { int reps; uint32_t n_ubatch; } cases[] = { {2, 512}, {6, 64}, {6, 16} };
    for (const auto & cs : cases) {
        llama_context * ctx_cut  = make_ctx(params, model, cs.n_ubatch);
        llama_context * ctx_snap = make_ctx(params, model, cs.n_ubatch);
        llama_context * ctx_ref  = make_ctx(params, model, cs.n_ubatch);
        if (!ctx_cut || !ctx_snap || !ctx_ref) { fprintf(stderr, "ctx init failed\n"); return 1; }
        if (llama_n_rs_seq(ctx_cut) < 4) { fprintf(stderr, "skipping: n_rs_seq < 4\n"); return 0; }

        std::string text;
        for (int r = 0; r < cs.reps; ++r) {
            text += "In the beginning the recurrent state was a summary of everything that came before it, and the "
                    "checkpoint had to be taken four tokens before the end of the prompt. ";
        }
        std::vector<llama_token> prompt = common_tokenize(ctx_cut, text, true);
        const std::vector<llama_token> cont = common_tokenize(ctx_cut, " The snapshot slot holds that state already, so no cut is needed.", false);
        const int N = (int) prompt.size();
        std::vector<llama_token> all = prompt;
        all.insert(all.end(), cont.begin(), cont.end());

        // cut path checkpoint
        common_prompt_checkpoint ck_cut;
        if (!decode_range(ctx_cut, prompt, 0, N - 4, false)) { fprintf(stderr, "cut decode failed\n"); return 1; }
        ck_cut.update_tgt(ctx_cut, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        // copy path checkpoint
        common_prompt_checkpoint ck_snap;
        llama_memory_t mem_snap = llama_get_memory(ctx_snap);
        if (!decode_range(ctx_snap, prompt, 0, N, false)) { fprintf(stderr, "snap decode failed\n"); return 1; }
        if (!llama_memory_seq_snap_begin(mem_snap, 0, 4)) {
            fprintf(stderr, "FAIL [N=%d ub=%u] snap_begin(4) refused on a %d-token final ubatch\n", N, cs.n_ubatch, N);
            n_fail++;
            continue;
        }
        ck_snap.update_tgt(ctx_snap, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (llama_memory_seq_snap_begin(mem_snap, 0, 4)) {
            fprintf(stderr, "FAIL nested snap_begin accepted\n");
            n_fail++;
        }
        llama_memory_seq_snap_end(mem_snap, 0);

        // the live state is untouched: next-token logits after one more token equal a reference context's
        {
            if (!decode_range(ctx_ref, prompt, 0, N, false)) return 1;
            std::vector<llama_token> probe = all;
            if (!decode_range(ctx_snap, probe, N, N + 1, false) || !decode_range(ctx_ref, probe, N, N + 1, false)) return 1;
            const float * a = llama_get_logits_ith(ctx_snap, -1);
            const float * b = llama_get_logits_ith(ctx_ref, -1);
            const bool same = std::memcmp(a, b, sizeof(float) * n_vocab) == 0;
            printf("[N=%d ub=%u] live state after snap_begin/end bit-identical: %s\n", N, cs.n_ubatch, same ? "yes" : "NO");
            n_fail += !same;
        }

        // blob comparison (informational)
        {
            const auto & a = ck_cut.data_tgt;
            const auto & b = ck_snap.data_tgt;
            size_t n_diff = 0;
            if (a.size() == b.size()) {
                for (size_t i = 0; i < a.size(); ++i) n_diff += a[i] != b[i];
            }
            printf("[N=%d ub=%u] checkpoint blob: cut %zu B, copy %zu B, differing bytes %zu\n", N, cs.n_ubatch, a.size(), b.size(), n_diff);
            if (a.size() != b.size()) { fprintf(stderr, "FAIL blob size mismatch\n"); n_fail++; }
        }

        // restore equivalence, plus an A/A run of the cut path (restore floor)
        std::vector<std::vector<float>> L_cut, L_cut2, L_snap;
        if (!restore_and_replay(ctx_ref, ck_cut,  all, N, n_vocab, L_cut) ||
            !restore_and_replay(ctx_ref, ck_cut,  all, N, n_vocab, L_cut2) ||
            !restore_and_replay(ctx_ref, ck_snap, all, N, n_vocab, L_snap)) {
            fprintf(stderr, "FAIL [N=%d ub=%u] restore/replay failed\n", N, cs.n_ubatch);
            n_fail++;
            continue;
        }
        auto compare = [&](const std::vector<std::vector<float>> & X, const std::vector<std::vector<float>> & Y,
                           double & kld_mean, double & kld_max_seen, int & top1_agree) {
            std::vector<double> p, q;
            kld_mean = 0; kld_max_seen = 0; top1_agree = 0;
            for (size_t r = 0; r < X.size(); ++r) {
                softmax(X[r].data(), n_vocab, p);
                softmax(Y[r].data(), n_vocab, q);
                double k = 0;
                for (int i = 0; i < n_vocab; ++i) if (p[i] > 0) k += p[i] * (std::log(p[i]) - std::log(std::max(q[i], 1e-300)));
                kld_mean += k; kld_max_seen = std::max(kld_max_seen, k);
                top1_agree += argmax(X[r].data(), n_vocab) == argmax(Y[r].data(), n_vocab);
            }
            kld_mean /= X.size();
        };
        double km, kx; int t1;
        compare(L_cut, L_cut2, km, kx, t1);
        printf("[N=%d ub=%u] A/A cut-vs-cut: rows %zu, KLD mean %.3e max %.3e, top-1 %d/%zu\n", N, cs.n_ubatch, L_cut.size(), km, kx, t1, L_cut.size());
        compare(L_cut, L_snap, km, kx, t1);
        const bool pass = t1 == (int) L_cut.size() && km <= kld_max;
        printf("[N=%d ub=%u] cut-vs-copy:     rows %zu, KLD mean %.3e max %.3e, top-1 %d/%zu -> %s (KLD_MAX %.1e)\n",
               N, cs.n_ubatch, L_cut.size(), km, kx, t1, L_cut.size(), pass ? "PASS" : "FAIL", kld_max);
        n_fail += !pass;

        // guard: a final ubatch of 3 tokens did not write slot 4 -> refuse
        {
            llama_memory_clear(llama_get_memory(ctx_ref), true);
            if (!decode_range(ctx_ref, prompt, 0, N - 3, false) || !decode_range(ctx_ref, prompt, N - 3, N, false)) return 1;
            const bool refused = !llama_memory_seq_snap_begin(llama_get_memory(ctx_ref), 0, 4);
            printf("[N=%d ub=%u] guard refuses slot 4 after a 3-token ubatch: %s\n", N, cs.n_ubatch, refused ? "yes" : "NO");
            n_fail += !refused;
            if (!refused) llama_memory_seq_snap_end(llama_get_memory(ctx_ref), 0);
        }

        llama_free(ctx_cut);
        llama_free(ctx_snap);
        llama_free(ctx_ref);
    }

    printf("test-snapslot-ckpt: %s (%d failures)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
