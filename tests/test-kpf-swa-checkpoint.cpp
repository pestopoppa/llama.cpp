// [KPF-17.4] SWA checkpoint range correctness ([TAG_CHECKPOINTS_FIX_POS_MIN]).
//
// On an SWA target (Gemma-class) save a PARTIAL_ONLY checkpoint at position c (what create_checkpoint stores), keep
// decoding to N, restore the checkpoint IN PLACE (what pre_decode does: seq_rm(pos_max+1, -1) + set_data_ext), re-decode
// [c, N) and compare next-token + logits with a run that never used a checkpoint. Also records, for every checkpoint,
// the pos_min the server would record (cache pos_min before the save) against the pos_min the saved data actually
// restores into an empty sequence, and prints one "KPF174 ..." line per case (the TODO says these are not the same).
//
// Window-shortfall case: the restored sequence is truncated to [c - n_swa/2, c), i.e. it provably no longer covers the
// window [c - n_swa, c). The acceptance gate must use the REAL cache pos_min: it then rejects and the slot falls back to
// a full re-process (verified equal to the no-checkpoint run). With KPF_SWA_MUTATE=use_recorded_range the gate trusts the
// RECORDED pos_min (the value create_checkpoint stores, 0) and continues from the truncated cache: that must FAIL.
//
// Usage: test-kpf-swa-checkpoint -m <swa model.gguf>   (exits 0 with SKIP on a non-SWA model)

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do {                                                   \
    if (!(cond)) {                                                              \
        fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);         \
        fprintf(stderr, __VA_ARGS__);                                           \
        fprintf(stderr, "\n");                                                  \
        g_fail++;                                                               \
    }                                                                           \
} while (0)

static bool decode_chunked(llama_context * ctx, const std::vector<llama_token> & t, int p0, int p1, llama_seq_id seq, int chunk = 128) {
    for (int a = p0; a < p1; a += chunk) {
        const int b = std::min(p1, a + chunk);
        llama_batch batch = llama_batch_init(b - a, 0, 1);
        for (int pos = a; pos < b; ++pos) {
            common_batch_add(batch, t[pos], pos, { seq }, pos + 1 == b);
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        if (!ok) {
            return false;
        }
    }
    return true;
}

static std::vector<float> last_logits(llama_context * ctx, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, -1);
    return l ? std::vector<float>(l, l + n_vocab) : std::vector<float>();
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static int argmax(const std::vector<float> & a) {
    return a.empty() ? -1 : (int) (std::max_element(a.begin(), a.end()) - a.begin());
}

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = 2;
    cparams.kv_unified = true;
    cparams.n_ctx      = 4096;
    cparams.n_batch    = 512;
    cparams.n_ubatch   = 128;
    return llama_init_from_model(model, cparams);
}

static void compare(const char * tag, const std::vector<float> & a, const std::vector<float> & b, float eps) {
    const float d = max_abs_diff(a, b);
    fprintf(stderr, "  %-44s max|d| = %.3g, top1 %d vs %d\n", tag, (double) d, argmax(a), argmax(b));
    CHECK(d <= eps, "%s: max|d| %g > %g", tag, (double) d, (double) eps);
    CHECK(argmax(a) == argmax(b), "%s: top-1 differs", tag);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
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

    const int n_swa = llama_model_n_swa(model);
    if (n_swa <= 0) {
        fprintf(stderr, "SKIP: not an SWA model\n");
        return 0;
    }
    const char * mut_env = getenv("KPF_SWA_MUTATE");
    const bool   mut_recorded = mut_env && strstr(mut_env, "use_recorded_range") != nullptr;
    fprintf(stderr, "n_swa = %d, mutate = %s\n", n_swa, mut_recorded ? "use_recorded_range" : "none");

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    std::string text;
    while ((int) text.size() < 8 * (2 * n_swa + 400)) {
        text += "The lighthouse keeper kept a careful log. Every evening he wrote down the wind, the height of the waves, "
                "the ships that passed, and the colour of the sky at sunset. Over thirty years the log grew to forty volumes. ";
    }
    std::vector<llama_token> toks = common_tokenize(vocab, text, true, false);
    const int N = 2 * n_swa + 300;
    if ((int) toks.size() < N) {
        fprintf(stderr, "prompt too short (%zu < %d)\n", toks.size(), N);
        return 1;
    }
    toks.resize(N);

    llama_context * ref = make_ctx(params, model);
    CHECK(decode_chunked(ref, toks, 0, N, 0), "reference decode");
    const std::vector<float> l_ref = last_logits(ref, n_vocab);

    for (const int c : { n_swa + 100, 2 * n_swa }) {
        fprintf(stderr, "== checkpoint at c = %d (N = %d)\n", c, N);
        llama_context * ctx = make_ctx(params, model);
        llama_memory_t mem  = llama_get_memory(ctx);

        CHECK(decode_chunked(ctx, toks, 0, c, 0), "prefix decode");
        const llama_pos rec_min = llama_memory_seq_pos_min(mem, 0); // what create_checkpoint records
        const llama_pos rec_max = llama_memory_seq_pos_max(mem, 0);
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        std::vector<uint8_t> ck(sz);
        CHECK(llama_state_seq_get_data_ext(ctx, ck.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == sz, "save");

        // what the saved bytes really restore into an EMPTY sequence
        llama_memory_seq_rm(mem, 1, -1, -1);
        llama_state_seq_set_data_ext(ctx, ck.data(), sz, 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const llama_pos real_min = llama_memory_seq_pos_min(mem, 1);
        const llama_pos real_max = llama_memory_seq_pos_max(mem, 1);
        llama_memory_seq_rm(mem, 1, -1, -1);
        const bool covers = real_min >= 0 && real_min <= std::max(0, c - n_swa);
        fprintf(stderr, "KPF174 c=%d n_swa=%d recorded=[%d,%d] real_restored=[%d,%d] window_start=%d covers_window=%s%s\n", c, n_swa,
                rec_min, rec_max, real_min, real_max, std::max(0, c - n_swa), covers ? "yes" : "no",
                (real_min != rec_min) ? " RECORDED_RANGE_DIFFERS" : "");

        // continue to N without any checkpoint use, then restore in place and re-decode
        CHECK(decode_chunked(ctx, toks, c, N, 0), "continuation");
        compare("continuation == no-checkpoint run", last_logits(ctx, n_vocab), l_ref, 2e-3f);

        llama_memory_seq_rm(mem, 0, rec_max + 1, -1);
        CHECK(llama_state_seq_set_data_ext(ctx, ck.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == sz, "restore");
        CHECK(decode_chunked(ctx, toks, c, N, 0), "re-decode after restore");
        compare("restore + re-decode == no-checkpoint run", last_logits(ctx, n_vocab), l_ref, 2e-3f);

        // window shortfall: the cache provably lacks [c - n_swa, c - n_swa/2 ...)
        llama_memory_seq_rm(mem, 0, rec_max + 1, -1);
        llama_state_seq_set_data_ext(ctx, ck.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        llama_memory_seq_rm(mem, 0, 0, c - n_swa / 2);
        const llama_pos short_min = llama_memory_seq_pos_min(mem, 0);
        const llama_pos thold     = std::max(0, c - n_swa);
        const llama_pos gate_pos  = mut_recorded ? rec_min : short_min; // the mutation trusts the recorded value
        const bool accept = gate_pos < thold || gate_pos == 0;
        fprintf(stderr, "  shortfall: short_min=%d recorded=%d thold=%d -> gate %s\n", short_min, rec_min, thold, accept ? "ACCEPT" : "REJECT");
        if (accept) {
            CHECK(decode_chunked(ctx, toks, c, N, 0), "re-decode from truncated cache");
        } else {
            llama_memory_seq_rm(mem, 0, -1, -1);
            CHECK(decode_chunked(ctx, toks, 0, N, 0), "full re-process after reject");
        }
        compare("shortfall path == no-checkpoint run", last_logits(ctx, n_vocab), l_ref, 2e-3f);

        llama_free(ctx);
    }
    llama_free(ref);

    if (g_fail) {
        fprintf(stderr, "test-kpf-swa-checkpoint : %d check(s) failed\n", g_fail);
        return 1;
    }
    fprintf(stderr, "test-kpf-swa-checkpoint : OK\n");
    return 0;
}
