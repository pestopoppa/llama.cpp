// Multi-sequence ubatch regression test (np > 1 serving shape) for hybrid recurrent targets.
//
//   test-multiseq-ubatch <model.gguf> [n_steps] [n_threads]      (MSU_NGL=99 runs the target on the GPU)
//
// Three independent sequences share one context (unified KV, n_seq_max 3) and are teacher-forced along ONE fixed
// token stream. Every step decodes 1..5 tokens for a SUBSET of the sequences that changes from step to step:
//   * schedule A ("np2")   : always seqs {0, 1}, 5 tokens each              -> plain 2-slot serving
//   * schedule B ("mixed") : {0,1,2}, {0,1}, {0}, {1,2}, {0,2}, {0,1,2}, ... -> n_seqs changes between ubatches
// Each decoded row's logits are compared with a single-sequence reference (seq 0 alone, same tokens, same chunking
// as the step). A correct build agrees to backend noise (<= ~0.5 max |logit| on Qwen3.5-0.8B Q8_0); a corrupted
// recurrent / conv state gives several units. Origin: the AK-ASM-H15 fused conv kernel wrote silu/l2 outputs over
// its own inputs when the allocator overlapped them, which it did on 2- and 3-sequence ubatches (AK-DTREE T9).

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static constexpr float TOL = 1.5f;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_steps] [n_threads]\n", argv[0]);
        return 2;
    }
    const int n_steps   = argc > 2 ? atoi(argv[2]) : 48;
    const int n_threads = argc > 3 ? atoi(argv[3]) : 8;

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = getenv("MSU_NGL") ? atoi(getenv("MSU_NGL")) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    CHECK(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    auto cparams = llama_context_default_params();
    cparams.n_ctx           = 4096;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = 3;
    cparams.kv_unified      = true;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    CHECK(ctx);
    llama_batch batch = llama_batch_init(512, 0, 3);

    const std::string text = "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n";
    const auto prompt = common_tokenize(ctx, text, true, false);

    // the fixed token stream: greedy continuation of seq 0 alone
    std::vector<llama_token> stream = prompt;
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch_clear(batch);
        for (size_t i = 0; i < prompt.size(); ++i) {
            common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
        }
        CHECK(llama_decode(ctx, batch) == 0);
        for (int k = 0; k < n_steps * 5 + 8; ++k) {
            const float * l = llama_get_logits_ith(ctx, -1);
            const llama_token t = (llama_token) (std::max_element(l, l + n_vocab) - l);
            stream.push_back(t);
            common_batch_clear(batch);
            common_batch_add(batch, t, (llama_pos) stream.size() - 1, { 0 }, true);
            CHECK(llama_decode(ctx, batch) == 0);
        }
    }

    // reference logits per stream position (single sequence, chunks of `chunk` tokens), keyed by chunk layout
    auto run = [&](const std::vector<std::vector<int>> & sched, const std::vector<int> & chunk,
                   std::vector<std::vector<std::vector<float>>> & out /* [seq][pos] */) {
        llama_memory_clear(llama_get_memory(ctx), true);
        out.assign(3, std::vector<std::vector<float>>(stream.size()));
        std::vector<size_t> pos(3, prompt.size());
        // prefill every sequence used by the schedule (one ubatch per seq so prefill shapes are single-seq)
        for (int s = 0; s < 3; ++s) {
            common_batch_clear(batch);
            for (size_t i = 0; i < prompt.size(); ++i) {
                common_batch_add(batch, prompt[i], (llama_pos) i, { s }, false);
            }
            CHECK(llama_decode(ctx, batch) == 0);
        }
        for (size_t st = 0; st < sched.size(); ++st) {
            const int n = chunk[st % chunk.size()];
            common_batch_clear(batch);
            std::vector<std::pair<int, size_t>> rows; // (seq, stream pos)
            for (int s : sched[st]) {
                for (int k = 0; k < n; ++k) {
                    rows.push_back({ s, pos[s] });
                    common_batch_add(batch, stream[pos[s]], (llama_pos) pos[s], { s }, true);
                    pos[s]++;
                }
            }
            CHECK(llama_decode(ctx, batch) == 0);
            for (size_t r = 0; r < rows.size(); ++r) {
                const float * l = llama_get_logits_ith(ctx, (int) r);
                out[rows[r].first][rows[r].second].assign(l, l + n_vocab);
            }
        }
    };

    int n_fail = 0;
    auto compare = [&](const char * name, const std::vector<std::vector<int>> & sched, const std::vector<int> & chunk) {
        // reference: each sequence's own step sequence replayed ALONE (same chunk sizes at the same steps)
        std::vector<std::vector<std::vector<float>>> got, ref_all(3);
        run(sched, chunk, got);
        float worst = 0.0f;
        for (int s = 0; s < 3; ++s) {
            std::vector<std::vector<int>> solo;
            std::vector<int> solo_chunk;
            for (size_t st = 0; st < sched.size(); ++st) {
                if (std::find(sched[st].begin(), sched[st].end(), s) != sched[st].end()) {
                    solo.push_back({ s });
                    solo_chunk.push_back(chunk[st % chunk.size()]);
                }
            }
            if (solo.empty()) {
                continue;
            }
            std::vector<std::vector<std::vector<float>>> ref;
            run(solo, solo_chunk, ref);
            for (size_t p = prompt.size(); p < stream.size(); ++p) {
                if (got[s][p].empty() || ref[s][p].empty()) {
                    continue;
                }
                float md = 0.0f;
                for (int v = 0; v < n_vocab; ++v) {
                    md = std::max(md, std::fabs(got[s][p][v] - ref[s][p][v]));
                }
                worst = std::max(worst, md);
                if (md > TOL) {
                    if (n_fail < 8) {
                        fprintf(stderr, "%s: seq %d pos %zu differs from its single-sequence replay (max |d| = %.3f)\n", name, s, p, md);
                    }
                    ++n_fail;
                }
            }
        }
        printf("%s: worst max |logit diff| vs single-sequence replay = %.3f (tol %.2f)\n", name, worst, TOL);
    };

    std::vector<std::vector<int>> np2, mixed;
    const std::vector<std::vector<int>> cyc = { { 0, 1, 2 }, { 0, 1 }, { 0 }, { 1, 2 }, { 0, 2 }, { 0, 1, 2 }, { 2 }, { 0, 1 } };
    for (int st = 0; st < n_steps; ++st) {
        np2.push_back({ 0, 1 });
        mixed.push_back(cyc[st % cyc.size()]);
    }
    compare("np2", np2, { 5 });
    compare("np2-chunks", np2, { 5, 1, 3, 1 });
    compare("mixed", mixed, { 5 });
    compare("mixed-chunks", mixed, { 5, 1, 3, 2 });

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    CHECK(n_fail == 0);
    printf("test-multiseq-ubatch: OK\n");
    return 0;
}
