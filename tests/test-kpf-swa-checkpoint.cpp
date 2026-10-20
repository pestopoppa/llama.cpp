// [KPF-17.4] SWA checkpoint range correctness ([TAG_CHECKPOINTS_FIX_POS_MIN]).
//
// On an SWA target (Gemma-class) save a PARTIAL_ONLY checkpoint at position c (what create_checkpoint stores), keep
// decoding to N, restore the checkpoint IN PLACE (what pre_decode does: seq_rm(pos_max+1, -1) + set_data_ext), re-decode
// [c, N) and compare next-token + logits with a run that never used a checkpoint. Also records, for every checkpoint,
// the pos_min the server would record (cache pos_min before the save) against the pos_min the saved data actually
// restores into an empty sequence, and prints one "KPF174 ..." line per case (the TODO says these are not the same).
//
// ORACLE. Logit-level bit-exactness across different float evaluation orders is not available on CPU: moving the
// ubatch boundaries alone moves the last logits by max|d| 0.32 on gemma-3-1b Q8_0 and 0.47 on the NON-SWA
// Qwen2.5-0.5B f16 (same top-1; 2026-10-20, this binary with -m on that model prints the floor and SKIPs), and moving
// only the KV cell placement moves them by 0.33 on gemma. Job 23 of 2026-10-20 gated these perturbations at eps 2e-3
// against a 0..N run and failed on them, including "continuation", which never uses a checkpoint. So: two
// no-checkpoint perturbations are measured per case ("boundary floor": [0,c)+[c,N) vs a 0..N run; "placement floor":
// the same decode after an unrelated seq-1 restore/remove), the restore is proven exact at the STATE level (the
// restored sequence re-serializes byte-identical to the checkpoint), and at the logit level it must keep top-1 and stay
// within FLOOR_MULT x the measured floor of the matched control (same boundaries). The shortfall mutation moves the
// logits by >5x the floor (4.6 and 3.8 vs eps 0.66 on gemma-3-1b).
//
// COVERAGE. The saved SWA data must cover the window needed to RESUME at the checkpoint's own end (pre_decode resumes at
// pos_max, i.e. needs [pos_max - n_swa + 1, pos_max - 1]); that is asserted. The RECORDED pos_min (the live cache pos_min,
// which still counts lazily-unpruned masked cells) is <= the real one, so a gate fed the recorded value is only ever more
// permissive, never less; it is safe exactly because the data always covers its own resume window. The server reads the
// recorded pos_min only in that acceptance heuristic (pre_decode, and find_fork_source which mirrors it), never as
// evidence that LIVE cells exist: pre_decode decides whether to restore at all from the live cache pos_min. The
// window-shortfall state below (a live cache missing part of the window, accepted on the recorded value) is therefore
// the mutation's construction, not a reachable server state. What the lie can cost is reuse, not correctness: once the
// SWA cache has pruned to its window (context longer than the SWA cache), recorded == real == pos_max - n_swa + 1, and
// upstream's gate (pos_min < pos_next - n_swa) then rejects every checkpoint at pos_max >= pos_next - 1, i.e. exactly a
// fork junction: no fork / full re-process, never wrong output. Fixing that gate is upstream's TAG_CHECKPOINTS_FIX_POS_MIN
// (take it through CH-18).
//
// Window-shortfall case: the restored sequence is truncated to [c - n_swa/2, c), i.e. it provably no longer covers the
// window [c - n_swa, c). The acceptance gate must use the REAL cache pos_min: it then rejects and the slot falls back to
// a full re-process (verified equal to the no-checkpoint run). With KPF_SWA_MUTATE=use_recorded_range the gate trusts the
// RECORDED pos_min (the value create_checkpoint stores, 0) and continues from the truncated cache: that must FAIL.
//
// Usage: test-kpf-swa-checkpoint -m <swa model.gguf>   (on a non-SWA model: prints the batch-boundary floor, then SKIP)

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

// eps < 0: measurement only (top-1 still required)
static float compare(const char * tag, const std::vector<float> & a, const std::vector<float> & b, float eps) {
    const float d = max_abs_diff(a, b);
    fprintf(stderr, "  %-44s max|d| = %.3g, top1 %d vs %d%s\n", tag, (double) d, argmax(a), argmax(b),
            eps < 0.0f ? " (floor measurement)" : "");
    if (eps >= 0.0f) {
        CHECK(d <= eps, "%s: max|d| %g > %g", tag, (double) d, (double) eps);
    }
    CHECK(argmax(a) == argmax(b), "%s: top-1 differs", tag);
    return d;
}

static constexpr float FLOOR_MULT = 2.0f;  // restore vs matched control may move as much as 2 no-checkpoint perturbations
static constexpr float EPS_MIN    = 2e-3f; // when both measured floors are 0 (an order-insensitive model/backend)

static std::vector<uint8_t> save_partial(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> v(llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    v.resize(llama_state_seq_get_data_ext(ctx, v.data(), v.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    return v;
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

    const int n_swa_model = llama_model_n_swa(model);
    // on a non-SWA model only the batch-boundary floor is measured (same N and c as an n_swa = 512 model), then SKIP
    const int n_swa = n_swa_model > 0 ? n_swa_model : 512;
    const char * mut_env = getenv("KPF_SWA_MUTATE");
    const bool   mut_recorded = mut_env && strstr(mut_env, "use_recorded_range") != nullptr;
    fprintf(stderr, "n_swa = %d%s, mutate = %s\n", n_swa, n_swa_model > 0 ? "" : " (non-SWA model: floor only)",
            mut_recorded ? "use_recorded_range" : "none");

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

    struct ckpt_case {
        int c;
        std::vector<float> l_ctl; // [0, c) + [c, N), no checkpoint, fresh context
        std::vector<float> l_restore;
        std::vector<float> l_short;
        bool short_accept = false;
    };
    std::vector<ckpt_case> cases;
    float floor = 0.0f; // max no-checkpoint perturbation over every case

    for (const int c : { n_swa + 100, 2 * n_swa }) {
        fprintf(stderr, "== checkpoint at c = %d (N = %d)\n", c, N);
        ckpt_case cc;
        cc.c = c;

        // the matched no-checkpoint control: same ubatch boundaries as every path below ([0, c) then [c, N))
        {
            llama_context * ctl = make_ctx(params, model);
            CHECK(decode_chunked(ctl, toks, 0, c, 0) && decode_chunked(ctl, toks, c, N, 0), "control decode");
            cc.l_ctl = last_logits(ctl, n_vocab);
            llama_free(ctl);
        }
        floor = std::max(floor, compare("boundary floor: ctl vs 0..N run", cc.l_ctl, l_ref, -1.0f));
        if (n_swa_model <= 0) {
            continue;
        }

        llama_context * ctx = make_ctx(params, model);
        llama_memory_t mem  = llama_get_memory(ctx);

        CHECK(decode_chunked(ctx, toks, 0, c, 0), "prefix decode");
        const llama_pos rec_min = llama_memory_seq_pos_min(mem, 0); // what create_checkpoint records
        const llama_pos rec_max = llama_memory_seq_pos_max(mem, 0);
        const std::vector<uint8_t> ck = save_partial(ctx, 0);
        const size_t sz = ck.size();
        CHECK(sz > 0, "save");

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
        // the saved data covers the window needed to resume at its own end (pre_decode re-decodes pos_max)
        CHECK(covers, "saved SWA data [%d, %d] does not cover the resume window from %d", real_min, real_max, std::max(0, c - n_swa));
        CHECK(real_max == rec_max, "saved SWA data ends at %d, the cache at %d", real_max, rec_max);
        // the recorded value can only be the more permissive one
        CHECK(rec_min <= real_min, "recorded pos_min %d > real %d", rec_min, real_min);

        // continue to N without any checkpoint use: differs from ctl only by the KV cell placement the seq-1 probe left
        CHECK(decode_chunked(ctx, toks, c, N, 0), "continuation");
        floor = std::max(floor, compare("placement floor: continuation vs ctl", last_logits(ctx, n_vocab), cc.l_ctl, -1.0f));

        // restore in place (pre_decode: seq_rm(pos_max + 1, -1) + PARTIAL_ONLY load): the state must be the checkpoint
        llama_memory_seq_rm(mem, 0, rec_max + 1, -1);
        CHECK(llama_state_seq_set_data_ext(ctx, ck.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == sz, "restore");
        CHECK(llama_memory_seq_pos_max(mem, 0) == rec_max, "restored pos_max %d != %d", llama_memory_seq_pos_max(mem, 0), rec_max);
        CHECK(save_partial(ctx, 0) == ck, "restored state does not re-serialize byte-identical to the checkpoint");
        CHECK(decode_chunked(ctx, toks, c, N, 0), "re-decode after restore");
        cc.l_restore = last_logits(ctx, n_vocab);

        // window shortfall: the cache provably lacks [c - n_swa, c - n_swa/2 ...)
        llama_memory_seq_rm(mem, 0, rec_max + 1, -1);
        llama_state_seq_set_data_ext(ctx, ck.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        llama_memory_seq_rm(mem, 0, 0, c - n_swa / 2);
        const llama_pos short_min = llama_memory_seq_pos_min(mem, 0);
        const llama_pos thold     = std::max(0, c - n_swa);
        const llama_pos gate_pos  = mut_recorded ? rec_min : short_min; // the mutation trusts the recorded value
        cc.short_accept = gate_pos < thold || gate_pos == 0;
        fprintf(stderr, "  shortfall: short_min=%d recorded=%d thold=%d -> gate %s\n", short_min, rec_min, thold, cc.short_accept ? "ACCEPT" : "REJECT");
        if (cc.short_accept) {
            CHECK(decode_chunked(ctx, toks, c, N, 0), "re-decode from truncated cache");
        } else {
            llama_memory_seq_rm(mem, 0, -1, -1);
            CHECK(decode_chunked(ctx, toks, 0, N, 0), "full re-process after reject");
        }
        cc.l_short = last_logits(ctx, n_vocab);

        llama_free(ctx);
        cases.push_back(std::move(cc));
    }

    // gate every checkpoint path against its matched control at FLOOR_MULT x the largest measured perturbation
    const float eps = floor > 0.0f ? FLOOR_MULT * floor : EPS_MIN;
    fprintf(stderr, "== no-checkpoint perturbation floor = %.3g -> eps = %.3g\n", (double) floor, (double) eps);
    for (const auto & cc : cases) {
        fprintf(stderr, "-- c = %d\n", cc.c);
        compare("restore + re-decode == matched control", cc.l_restore, cc.l_ctl, eps);
        // accept continues with the [0, c) + [c, N) boundaries (ctl); reject re-processes like the 0..N run (bit-exact)
        compare("shortfall path == no-checkpoint run", cc.l_short, cc.short_accept ? cc.l_ctl : l_ref, cc.short_accept ? eps : 0.0f);
    }
    llama_free(ref);

    if (n_swa_model <= 0) {
        fprintf(stderr, "SKIP: not an SWA model (batch-boundary floor printed above)\n");
        return g_fail ? 1 : 0;
    }
    if (g_fail) {
        fprintf(stderr, "test-kpf-swa-checkpoint : %d check(s) failed\n", g_fail);
        return 1;
    }
    fprintf(stderr, "test-kpf-swa-checkpoint : OK\n");
    return 0;
}
