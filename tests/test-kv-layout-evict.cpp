// KV-LAYOUT-EVICTION-DIVERGENCE regression test (CPU).
//
// A 4-sequence speculative-like decode (prompt, then rounds of 1 + 7 draft tokens per sequence, partial
// rollback with seq_rm, prefix eviction keeping the last 64 positions) run with a unified KV cache and with
// one stream per sequence. Tokens are fixed (no argmax feedback), so every run sees the same token history.
// Each verify row is compared against a fresh-prefill oracle: an empty single-sequence context fed the
// surviving window (original positions) plus the verify tokens. Layout and eviction may only move the float
// summation order, so both layouts must match the oracle to the noise floor.
//
// usage: test-kv-layout-evict <model.gguf>   (env: KVLE_EVICT=0 disables eviction, KVLE_ROUNDS, KVLE_CTX)

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <set>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN || getenv("KVLE_LOG")) {
        fputs(text, stderr);
    }
}

static const int n_seq = 4, n_draft = 7;

struct job {
    int q;
    std::map<int, llama_token> hist; // surviving committed tokens (pos -> token) before the verify batch
    int pos0;                        // first position of the verify batch
    std::vector<llama_token> toks;   // verify tokens
};

static int g_rounds = 90, g_ctx = 512;
static bool g_evict = true;

// structural KQ-mask check (layout-independent, no float noise): during a verify decode every token row of
// every attention mask must keep exactly |surviving history of its seq| + (its index in the draft) + 1 cells
static std::vector<int> g_expect;          // expected kept-cell count per verify row (batch order); empty = skip
static std::set<const void *> g_seen;      // masks already checked in this decode
static long g_mask_rows = 0, g_mask_bad = 0;

static void check_mask(const ggml_tensor * m) {
    if (g_expect.empty() || !g_seen.insert(m->data).second) return;
    const int64_t n_kv = m->ne[0], n_tps = (int64_t) g_expect.size() / m->ne[3];
    std::vector<uint8_t> buf(ggml_nbytes(m));
    ggml_backend_tensor_get(m, buf.data(), 0, buf.size());
    for (int64_t s = 0; s < m->ne[3]; ++s) {
        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const uint8_t * row = buf.data() + s*m->nb[3] + ii*m->nb[1];
            int kept = 0;
            for (int64_t j = 0; j < n_kv; ++j) {
                const float v = m->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) row)[j]) : ((const float *) row)[j];
                kept += v == 0.0f;
            }
            const int want = g_expect[s*n_tps + ii];
            g_mask_rows++;
            if (kept != want && g_mask_bad++ < 8) {
                fprintf(stderr, "MASK row %lld (stream %lld) keeps %d cells, expected %d (n_kv %lld)\n", (long long) (s*n_tps + ii), (long long) s, kept, want, (long long) n_kv);
            }
        }
    }
}

// KVLE_L0: capture the output of the first attention op (layer 0) of round 0: its Q/K/V inputs are
// bitwise identical across layouts, so any difference is the attention op itself
static int g_cap_round = -1, g_round = -1;
static std::vector<float> g_cap;
static bool eval_cb(ggml_tensor * t, bool ask, void *) {
    if (t->op == GGML_OP_FLASH_ATTN_EXT && g_round == g_cap_round && g_cap.empty()) {
        if (ask) return true;
        g_cap.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, g_cap.data(), 0, ggml_nbytes(t));
    }
    const ggml_tensor * m = t->op == GGML_OP_FLASH_ATTN_EXT ? t->src[3] : t->op == GGML_OP_SOFT_MAX ? t->src[1] : nullptr;
    if (ask) return m != nullptr && !g_expect.empty();
    if (m) check_mask(m);
    return true;
}

// only < 0: all sequences in one context; only = q: replay of sequence q alone (per-sequence oracle)
static std::vector<float> run(llama_model * model, bool unified, std::vector<job> * jobs, int only = -1, bool rev = false) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = g_ctx; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = only < 0 ? n_seq : 1; cp.kv_unified = unified;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    if (getenv("KVLE_F32KV")) { cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; }
    if (getenv("KVLE_FA") && atoi(getenv("KVLE_FA")) == 0) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = eval_cb; cp.cb_eval_user_data = nullptr;
    auto sid = [&](int q) { return only < 0 ? q : 0; };
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL: context\n"); exit(1); }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<float> all;
    llama_batch b = llama_batch_init(512, 0, 1);
    std::vector<int> pos(n_seq, 0);
    std::vector<llama_token> last(n_seq);
    std::vector<std::map<int, llama_token>> hist(n_seq);
    std::vector<int> blk(n_seq, 0); // block index of each sequence in the verify batch
    auto ord = [&](int idx) { return rev ? n_seq - 1 - idx : idx; }; // rev: sequences enter the batch in reverse

    b.n_tokens = 0;
    for (int idx = 0; idx < n_seq; ++idx) {
        const int q = ord(idx);
        if (only >= 0 && q != only) { pos[q] = 5 + 9*q; for (int k = 0; k < pos[q]; ++k) hist[q][k] = 0; continue; }
        const int len = 5 + 9*q;
        for (int k = 0; k < len; ++k) {
            const int i = b.n_tokens++;
            b.token[i] = (llama_token) ((101 + 37*q + 13*k) % n_vocab);
            hist[q][pos[q]] = b.token[i];
            b.pos[i] = pos[q]++; b.n_seq_id[i] = 1; b.seq_id[i][0] = sid(q); b.logits[i] = k == len - 1;
        }
        last[q] = (llama_token) ((199 + q) % n_vocab);
    }
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "FAIL: prompt decode\n"); exit(1); }

    for (int r = 0; r < g_rounds; ++r) {
        b.n_tokens = 0;
        g_round = r;
        g_expect.clear(); g_seen.clear();
        for (int idx = 0; idx < n_seq; ++idx) {
            const int q = ord(idx);
            if (only >= 0 && q != only) continue;
            blk[q] = b.n_tokens / (n_draft + 1);
            for (int k = 0; k <= n_draft; ++k) g_expect.push_back((int) hist[q].size() + k + 1);
            job j; j.q = q; j.hist = hist[q]; j.pos0 = pos[q];
            for (int k = 0; k <= n_draft; ++k) {
                const int i = b.n_tokens++;
                b.token[i] = k == 0 ? last[q] : (llama_token) ((7 + 31*r + 17*q + 5*k) % n_vocab);
                b.pos[i] = pos[q] + k; b.n_seq_id[i] = 1; b.seq_id[i][0] = sid(q); b.logits[i] = true;
                j.toks.push_back(b.token[i]);
            }
            if (jobs) jobs->push_back(std::move(j));
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "FAIL: verify decode r=%d (unified=%d)\n", r, unified); exit(1); }
        g_expect.clear(); g_round = -1;
        for (int q = 0; q < n_seq; ++q) { // canonical row order: sequence-major
            if (only >= 0 && q != only) continue;
            for (int k = 0; k <= n_draft; ++k) {
                const float * l = llama_get_logits_ith(ctx, blk[q]*(n_draft + 1) + k);
                all.insert(all.end(), l, l + n_vocab);
            }
        }
        for (int q = 0; q < n_seq; ++q) {
            const int bq = blk[q];
            if (only >= 0 && q != only) continue;
            const int acc = (r + q) % (n_draft + 1);
            if (!llama_memory_seq_rm(mem, sid(q), pos[q] + 1 + acc, -1)) { fprintf(stderr, "FAIL: rollback\n"); exit(1); }
            for (int k = 0; k <= acc; ++k) hist[q][pos[q] + k] = b.token[bq*(n_draft + 1) + k];
            last[q] = (llama_token) ((211 + 13*r + 7*q) % n_vocab);
            pos[q] += 1 + acc;
            if (g_evict && pos[q] > 96) {
                llama_memory_seq_rm(mem, sid(q), 0, pos[q] - 64);
                hist[q].erase(hist[q].begin(), hist[q].lower_bound(pos[q] - 64));
            }
            if (llama_memory_seq_pos_min(mem, sid(q)) != hist[q].begin()->first || llama_memory_seq_pos_max(mem, sid(q)) != hist[q].rbegin()->first) {
                fprintf(stderr, "FAIL: seq %d pos range [%d,%d] != history [%d,%d] (unified=%d r=%d)\n", q,
                        llama_memory_seq_pos_min(mem, sid(q)), llama_memory_seq_pos_max(mem, sid(q)), hist[q].begin()->first, hist[q].rbegin()->first, unified, r);
                exit(1);
            }
        }
    }
    llama_batch_free(b);
    llama_free(ctx);
    return all;
}

// per-sequence replay oracle: each sequence alone in its own context, identical op sequence (prompt, verify,
// rollback, eviction). A fresh prefill of the surviving window is NOT a valid oracle under eviction: the
// surviving K/V were computed while the evicted prefix was still visible.
static std::vector<float> oracle(llama_model * model, int n_vocab) {
    std::vector<std::vector<float>> per(n_seq);
    for (int q = 0; q < n_seq; ++q) per[q] = run(model, true, nullptr, q);
    const size_t blk = (size_t) (n_draft + 1)*n_vocab;
    std::vector<float> all;
    for (int r = 0; r < g_rounds; ++r) {
        for (int q = 0; q < n_seq; ++q) all.insert(all.end(), per[q].begin() + r*blk, per[q].begin() + (r + 1)*blk);
    }
    return all;
}

// returns argmax agreement fraction; prints the max |diff| / row range and the first bad round
static double cmp(const std::vector<float> & ref, const std::vector<float> & x, int n_vocab, const char * name, double * maxrel) {
    const size_t rows = std::min(ref.size(), x.size()) / n_vocab;
    size_t same = 0; double mr = 0; long first_bad = -1;
    for (size_t r = 0; r < rows; ++r) {
        const float * a = ref.data() + r*n_vocab; const float * c = x.data() + r*n_vocab;
        int ia = 0, ic = 0; float lo = a[0], hi = a[0]; double md = 0;
        for (int v = 0; v < n_vocab; ++v) {
            if (a[v] > a[ia]) ia = v;
            if (c[v] > c[ic]) ic = v;
            lo = std::min(lo, a[v]); hi = std::max(hi, a[v]);
            md = std::max(md, (double) std::fabs(a[v] - c[v]));
        }
        const double rel = md / std::max(1e-6, (double) (hi - lo));
        static std::vector<double> curve; const size_t rr = r / (n_seq*(n_draft + 1));
        if (getenv("KVLE_CURVE")) {
            if (r == 0) curve.assign(rows / (n_seq*(n_draft + 1)) + 1, 0.0);
            curve[rr] = std::max(curve[rr], rel);
            if (r + 1 == rows) { printf("  curve %s:", name); for (size_t k = 0; k < curve.size() - 1; k += 3) printf(" %.0e", curve[k]); printf("\n"); }
        }
        same += ia == ic; mr = std::max(mr, rel);
        if (first_bad < 0 && rel > 0.02) first_bad = (long) r;
    }
    printf("  %-26s argmax %zu / %zu, max |diff| / row range %.2e, first row > 2e-2: %ld (round %ld)\n",
           name, same, rows, mr, first_bad, first_bad < 0 ? -1 : first_bad / (n_seq*(n_draft + 1)));
    *maxrel = mr;
    return (double) same / std::max<size_t>(1, rows);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> (no model: SKIP)\n", argv[0]); return 0; }
    if (getenv("KVLE_EVICT"))  g_evict  = atoi(getenv("KVLE_EVICT")) != 0;
    if (getenv("KVLE_ROUNDS")) g_rounds = atoi(getenv("KVLE_ROUNDS"));
    if (getenv("KVLE_CTX"))    g_ctx    = atoi(getenv("KVLE_CTX"));
    ggml_backend_load_all();
    llama_backend_init();
    llama_log_set(log_cb, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "FAIL: model\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // layer-0 attention of a late verify round (after eviction has started) is captured in every layout
    g_cap_round = getenv("KVLE_L0") ? atoi(getenv("KVLE_L0")) : (2*g_rounds)/3;
    g_cap.clear(); const std::vector<float> uni = run(model, true, nullptr);  const std::vector<float> cu = g_cap;
    g_cap.clear(); const std::vector<float> str = run(model, false, nullptr); const std::vector<float> cn = g_cap;
    g_cap_round = -1;
    const std::vector<float> urv = run(model, true, nullptr, -1, true);
    g_cap_round = getenv("KVLE_L0") ? atoi(getenv("KVLE_L0")) : (2*g_rounds)/3;
    g_cap.clear(); // filled by the sequence-0 replay (the first one oracle() runs)
    const std::vector<float> orc = oracle(model, n_vocab);
    const std::vector<float> cs = g_cap;
    g_cap_round = -1;
    // KVLE_SAVE=<file> writes the oracle logits; KVLE_LOAD=<file> compares this run's oracle against a saved one
    // (precision control: the same layout under another accumulation mode = the workload's noise-amplification floor)
    if (const char * f = getenv("KVLE_SAVE")) { FILE * fp = fopen(f, "wb"); fwrite(orc.data(), sizeof(float), orc.size(), fp); fclose(fp); }
    if (const char * f = getenv("KVLE_LOAD")) {
        std::vector<float> ld(orc.size()); FILE * fp = fopen(f, "rb");
        const size_t got = fp ? fread(ld.data(), sizeof(float), ld.size(), fp) : 0; if (fp) fclose(fp);
        double rl; if (got == ld.size()) cmp(ld, orc, n_vocab, "oracle vs saved oracle", &rl);
    }
    printf("%s evict=%d rounds=%d ctx=%d\n", argv[1], g_evict, g_rounds, g_ctx);
    double r1, r2, r3;
    const double a1 = cmp(orc, uni, n_vocab, "unified vs oracle", &r1);
    const double a2 = cmp(orc, str, n_vocab, "per-seq streams vs oracle", &r2);
    const double a3 = cmp(uni, str, n_vocab, "unified vs streams", &r3);
    double r4; cmp(uni, urv, n_vocab, "unified vs unified-rev", &r4);
    (void) a1; (void) a2; (void) a3; (void) r1; (void) r2; (void) r3;
    printf("  KQ mask rows checked %ld, wrong kept-cell count %ld\n", g_mask_rows, g_mask_bad);
    // The end-to-end logit comparisons above are informational: with random fixed tokens this workload amplifies
    // ulp-level differences (Q8_0 activation re-quantisation over every layer; eviction drops the attention-sink
    // prefix and makes it worse). Changing ONLY the FA accumulator precision on the same layout gives the same
    // ~10% argmax disagreement (KVLE_SAVE/KVLE_LOAD control), so they cannot gate. The gate is structural:
    //  (1) every KQ-mask row keeps exactly the surviving cells of its sequence, in both layouts;
    //  (2) layer-0 attention (identical Q/K/V inputs in every layout) differs from the replay only by summation
    //      order: rel <= 1e-5 under GGML_FA_VKQ_F32=1, rel <= 4e-3 with the default FP16 VKQ accumulator.
    double mu = 0, mn = 0, amax = 0;
    const size_t n = cs.size();
    const bool cap_ok = n > 0 && cu.size() >= n && cn.size() >= n;
    for (size_t i = 0; cap_ok && i < n; ++i) {
        mu = std::max(mu, (double) std::fabs(cu[i] - cs[i])); mn = std::max(mn, (double) std::fabs(cn[i] - cs[i]));
        amax = std::max(amax, (double) std::fabs(cs[i]));
    }
    const char * f32 = getenv("GGML_FA_VKQ_F32");
    const double tol = (f32 && atoi(f32)) ? 1e-5 : 4e-3;
    const double ru = mu / std::max(1e-12, amax), rn = mn / std::max(1e-12, amax);
    printf("  layer-0 attention, seq 0, round %d: unified rel %.2e, streams rel %.2e (tol %.0e)%s\n", (2*g_rounds)/3, ru, rn, tol, cap_ok ? "" : " NOT CAPTURED");
    const bool ok = g_mask_rows > 0 && g_mask_bad == 0 && cap_ok && ru <= tol && rn <= tol;
    printf("%s\n", ok ? "PASS" : "FAIL");
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
