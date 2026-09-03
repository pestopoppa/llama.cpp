// INF-70 GDN-ROWEXACT reproducer: is a multi-token (and multi-sequence) forward row-exact
// against single-token decode, and if not, which graph node differs first?
//
// Modes:
//   batch : feed n tokens (after an identically-fed prefix) as ONE batch vs n single-token decodes;
//           capture every graph node in both, slice the batch node at each token row, compare bytes.
//   multi : feed S sequences x n tokens in one batch (n_seqs = S ubatch) vs each sequence alone as
//           one n-token batch; compare per-sequence rows.
// Every comparison is bitwise; max |diff| is reported for F32/F16 nodes.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

struct rec {
    std::string name;
    ggml_op     op;
    ggml_type   type;
    int64_t     ne[4];
    size_t      nb[4];
    std::vector<uint8_t> data; // span starting at t->data (strides preserved)
};

struct capture {
    bool enabled = false;
    std::vector<rec> recs;
};

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    if (ask) return true;
    auto * c = (capture *) ud;
    if (!c->enabled) return true;
    rec r;
    r.name = t->name;
    r.op   = t->op;
    r.type = t->type;
    for (int i = 0; i < 4; i++) { r.ne[i] = t->ne[i]; r.nb[i] = t->nb[i]; }
    const size_t n = ggml_nbytes(t);
    r.data.resize(n);
    ggml_backend_tensor_get(t, r.data.data(), 0, n);
    c->recs.push_back(std::move(r));
    return true;
}

// materialize the elements of a rec restricted to fixed indices on some dims (-1 = all)
// output order: i0 fastest, then i1, i2, i3
static bool materialize(const rec & r, const int64_t fix[4], std::vector<uint8_t> & out, int64_t & n_elem) {
    const size_t ts = ggml_type_size(r.type);
    if (ggml_blck_size(r.type) != 1) return false;
    int64_t lo[4], hi[4];
    n_elem = 1;
    for (int d = 0; d < 4; d++) {
        if (fix[d] >= 0) { lo[d] = fix[d]; hi[d] = fix[d] + 1; } else { lo[d] = 0; hi[d] = r.ne[d]; }
        n_elem *= (hi[d] - lo[d]);
    }
    out.resize(n_elem * ts);
    size_t o = 0;
    for (int64_t i3 = lo[3]; i3 < hi[3]; i3++)
    for (int64_t i2 = lo[2]; i2 < hi[2]; i2++)
    for (int64_t i1 = lo[1]; i1 < hi[1]; i1++) {
        const size_t base = i3*r.nb[3] + i2*r.nb[2] + i1*r.nb[1] + lo[0]*r.nb[0];
        const size_t len  = (hi[0]-lo[0]) * ts;
        if (r.nb[0] == ts) {
            if (base + len > r.data.size()) return false;
            memcpy(out.data() + o, r.data.data() + base, len);
            o += len;
        } else {
            for (int64_t i0 = lo[0]; i0 < hi[0]; i0++) {
                const size_t off = base + (i0-lo[0])*r.nb[0];
                if (off + ts > r.data.size()) return false;
                memcpy(out.data() + o, r.data.data() + off, ts);
                o += ts;
            }
        }
    }
    return true;
}

// choose the slice of batch rec `r` that corresponds to token index i (0..N*S-1) where the ubatch
// is n_seq_tokens=N, n_seqs=S. returns false if no token dimension can be identified.
static bool token_slice(const rec & r, int64_t N, int64_t S, int64_t i, int64_t fix[4]) {
    const int64_t T = N*S;
    const int64_t s = i / N, t = i % N;
    for (int d = 0; d < 4; d++) fix[d] = -1;
    if (r.ne[3] == T && T > 1)                         { fix[3] = i; return true; }
    if (r.ne[2] == N && r.ne[3] == S && S > 1)         { fix[2] = t; fix[3] = s; return true; }
    if (r.ne[2] == T && r.ne[3] == 1 && T > 1)         { fix[2] = i; return true; }
    if (r.ne[1] == N && r.ne[2] == S && r.ne[3] == 1 && S > 1) { fix[1] = t; fix[2] = s; return true; }
    if (r.ne[1] == T && r.ne[2] == 1 && r.ne[3] == 1 && T > 1) { fix[1] = i; return true; }
    return false;
}

struct diff_stat { bool ok; bool shape_skip; int64_t n_elem; int64_t n_diff; double max_abs; int64_t first_idx; };

static diff_stat compare(const rec & a, const int64_t fixa[4], const rec & b, const int64_t fixb[4]) {
    diff_stat d{true, false, 0, 0, 0.0, -1};
    std::vector<uint8_t> ma, mb; int64_t na = 0, nb = 0;
    if (a.type != b.type || !materialize(a, fixa, ma, na) || !materialize(b, fixb, mb, nb) || na != nb) {
        d.shape_skip = true; return d;
    }
    d.n_elem = na;
    if (memcmp(ma.data(), mb.data(), ma.size()) == 0) return d;
    d.ok = false;
    const size_t ts = ggml_type_size(a.type);
    for (int64_t k = 0; k < na; k++) {
        if (memcmp(ma.data()+k*ts, mb.data()+k*ts, ts) != 0) {
            d.n_diff++;
            if (d.first_idx < 0) d.first_idx = k;
            double va = 0, vb = 0;
            if (a.type == GGML_TYPE_F32) { va = ((float*)ma.data())[k]; vb = ((float*)mb.data())[k]; }
            else if (a.type == GGML_TYPE_F16) { va = ggml_fp16_to_fp32(((ggml_fp16_t*)ma.data())[k]); vb = ggml_fp16_to_fp32(((ggml_fp16_t*)mb.data())[k]); }
            d.max_abs = std::max(d.max_abs, std::fabs(va - vb));
        }
    }
    return d;
}

static std::string ne_str(const rec & r) {
    char buf[96]; snprintf(buf, sizeof buf, "[%lld,%lld,%lld,%lld]", (long long)r.ne[0], (long long)r.ne[1], (long long)r.ne[2], (long long)r.ne[3]); return buf;
}

// ---- runs -------------------------------------------------------------------------------------

struct opts {
    std::string model;
    int n_threads = 48;
    int n = 3;
    int prefix = 8;
    int n_seq_max = 4;
    int seqs = 2;
    int unified = -1;      // -1 = default
    int fa = -1;           // -1 = auto, 0 off, 1 on
    int rs_seq = 0;
    int show = 12;
    std::string mode = "batch";
    int prompt_a = 0, prompt_b = 1;
};

static const char * PROMPTS[] = {
    "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is",
    "Write a short Python function that reverses a string. Here is one clean implementation:",
    "Explain in two sentences why the sky is blue. The short physical explanation is that",
    "In 1969 the first humans landed on the Moon. The mission was called Apollo",
};

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const char * text, int want) {
    std::vector<llama_token> toks(512);
    int n = llama_tokenize(vocab, text, strlen(text), toks.data(), toks.size(), /*add_special*/ true, /*parse_special*/ false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); exit(1); }
    toks.resize(n);
    if ((int) toks.size() < want) { fprintf(stderr, "prompt has only %d tokens, need %d\n", n, want); exit(1); }
    toks.resize(want);
    return toks;
}

static llama_context * make_ctx(llama_model * model, const opts & o, capture * cap) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 512;
    cp.n_ubatch   = 512;
    cp.n_seq_max  = o.n_seq_max;
    cp.n_threads  = o.n_threads;
    cp.n_threads_batch = o.n_threads;
    cp.no_perf    = true;
    cp.n_rs_seq   = o.rs_seq;
    if (o.unified >= 0) cp.kv_unified = o.unified != 0;
    if (o.fa == 0) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (o.fa == 1) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); exit(1); }
    return ctx;
}

static void decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq, bool logits) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens = 1; b.token[0] = tok; b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = seq; b.logits[0] = logits;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
    llama_batch_free(b);
}

struct run_result {
    std::vector<std::vector<rec>> per_token;   // for single-step runs: one node list per token
    std::vector<rec> batch;                     // for batched runs: one node list
    std::vector<std::vector<float>> logits;     // per token
    std::vector<uint8_t> state;                 // seq 0 state after the run
    int n_seq_tokens = 1, n_seqs = 1;
};

static std::vector<float> get_logits(llama_context * ctx, int i, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, i);
    return std::vector<float>(l, l + n_vocab);
}

static std::vector<uint8_t> get_state(llama_context * ctx, llama_seq_id seq) {
    size_t sz = llama_state_seq_get_size(ctx, seq);
    std::vector<uint8_t> s(sz);
    llama_state_seq_get_data(ctx, s.data(), sz, seq);
    return s;
}

// reference: prefix one-by-one, then the n test tokens one-by-one (captured)
static run_result run_single(llama_model * model, const opts & o, const std::vector<llama_token> & toks, int n_vocab) {
    capture cap; run_result rr;
    llama_context * ctx = make_ctx(model, o, &cap);
    for (int i = 0; i < o.prefix; i++) decode_one(ctx, toks[i], i, 0, false);
    for (int t = 0; t < o.n; t++) {
        cap.recs.clear(); cap.enabled = true;
        decode_one(ctx, toks[o.prefix + t], o.prefix + t, 0, true);
        cap.enabled = false;
        rr.per_token.push_back(std::move(cap.recs));
        rr.logits.push_back(get_logits(ctx, 0, n_vocab));
    }
    rr.state = get_state(ctx, 0);
    llama_free(ctx);
    return rr;
}

// batched: prefix one-by-one per seq, then S seqs x n tokens in ONE decode call
static run_result run_batched(llama_model * model, const opts & o, const std::vector<std::vector<llama_token>> & toks, int n_vocab) {
    capture cap; run_result rr;
    const int S = toks.size();
    llama_context * ctx = make_ctx(model, o, &cap);
    for (int i = 0; i < o.prefix; i++) for (int s = 0; s < S; s++) decode_one(ctx, toks[s][i], i, s, false);
    llama_batch b = llama_batch_init(S * o.n, 0, 1);
    b.n_tokens = 0;
    for (int s = 0; s < S; s++) for (int t = 0; t < o.n; t++) {
        int k = b.n_tokens++;
        b.token[k] = toks[s][o.prefix + t]; b.pos[k] = o.prefix + t; b.n_seq_id[k] = 1; b.seq_id[k][0] = s; b.logits[k] = true;
    }
    cap.recs.clear(); cap.enabled = true;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "batched decode failed\n"); exit(1); }
    cap.enabled = false;
    rr.batch = std::move(cap.recs);
    for (int k = 0; k < b.n_tokens; k++) rr.logits.push_back(get_logits(ctx, k, n_vocab));
    llama_batch_free(b);
    rr.state = get_state(ctx, 0);
    rr.n_seq_tokens = o.n; rr.n_seqs = S;
    llama_free(ctx);
    return rr;
}

static void cmp_logits(const std::vector<float> & a, const std::vector<float> & b, const char * label) {
    int64_t nd = 0; double mx = 0; int am_a = 0, am_b = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) { nd++; mx = std::max(mx, (double) std::fabs(a[i]-b[i])); }
        if (a[i] > a[am_a]) am_a = i;
        if (b[i] > b[am_b]) am_b = i;
    }
    printf("  logits %s: %s n_diff=%lld/%zu max|d|=%.3e argmax %d%s%d\n", label, nd == 0 ? "IDENTICAL" : "DIFFER",
           (long long) nd, a.size(), mx, am_a, am_a == am_b ? "==" : "!=", am_b);
}

static void cmp_state(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, const char * label) {
    if (a.size() != b.size()) { printf("  state %s: SIZE DIFFERS %zu vs %zu\n", label, a.size(), b.size()); return; }
    size_t nd = 0; for (size_t i = 0; i < a.size(); i++) nd += a[i] != b[i];
    printf("  state %s (%zu bytes): %s (%zu bytes differ)\n", label, a.size(), nd == 0 ? "IDENTICAL" : "DIFFER", nd);
}

// compare the batch node list (rows of token i) against a reference node list whose tensors are for
// exactly that token (single-step run), or, when ref_N/ref_S > 1, against rows of another batch.
static void cmp_nodes(const std::vector<rec> & batch, int N, int S, int i,
                      const std::vector<rec> & ref, int refN, int refS, int refi, int show, const char * label) {
    std::map<std::string, std::vector<size_t>> ref_idx;
    for (size_t k = 0; k < ref.size(); k++) ref_idx[ref[k].name].push_back(k);
    std::map<std::string, size_t> seen;
    int64_t n_cmp = 0, n_bad = 0, n_skip = 0, n_unmatched = 0, n_shown = 0;
    std::string first_bad;
    for (size_t k = 0; k < batch.size(); k++) {
        const rec & rb = batch[k];
        size_t occ = seen[rb.name]++;
        auto it = ref_idx.find(rb.name);
        if (it == ref_idx.end() || occ >= it->second.size()) { n_unmatched++; continue; }
        const rec & rr = ref[it->second[occ]];
        int64_t fb[4], fr[4];
        if (!token_slice(rb, N, S, i, fb)) { n_skip++; continue; }
        if (refN * refS > 1) { if (!token_slice(rr, refN, refS, refi, fr)) { n_skip++; continue; } }
        else { for (int d = 0; d < 4; d++) fr[d] = -1; }
        diff_stat d = compare(rb, fb, rr, fr);
        if (d.shape_skip) { n_skip++; continue; }
        n_cmp++;
        if (!d.ok) {
            n_bad++;
            if (first_bad.empty()) first_bad = rb.name;
            if (n_shown < show) {
                n_shown++;
                printf("    #%-5zu %-36s %-14s %-22s %-4s diff %lld/%lld max|d|=%.3e first@%lld\n",
                       k, rb.name.c_str(), ggml_op_name(rb.op), ne_str(rb).c_str(), ggml_type_name(rb.type),
                       (long long) d.n_diff, (long long) d.n_elem, d.max_abs, (long long) d.first_idx);
            }
        }
    }
    printf("  nodes %s: compared %lld, DIFFER %lld, first=%s, shape-skip %lld, unmatched-name %lld (batch %zu nodes, ref %zu nodes)\n",
           label, (long long) n_cmp, (long long) n_bad, first_bad.empty() ? "-" : first_bad.c_str(), (long long) n_skip, (long long) n_unmatched, batch.size(), ref.size());
}

int main(int argc, char ** argv) {
    opts o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if      (a == "-m")          o.model = next();
        else if (a == "-t")          o.n_threads = atoi(next().c_str());
        else if (a == "-n")          o.n = atoi(next().c_str());
        else if (a == "--prefix")    o.prefix = atoi(next().c_str());
        else if (a == "--mode")      o.mode = next();
        else if (a == "--seqs")      o.seqs = atoi(next().c_str());
        else if (a == "--n-seq-max") o.n_seq_max = atoi(next().c_str());
        else if (a == "--unified")   o.unified = atoi(next().c_str());
        else if (a == "--fa")        o.fa = atoi(next().c_str());
        else if (a == "--rs-seq")    o.rs_seq = atoi(next().c_str());
        else if (a == "--show")      o.show = atoi(next().c_str());
        else if (a == "--prompt")    o.prompt_a = atoi(next().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (o.model.empty()) { fprintf(stderr, "usage: -m model [-t 48] [-n 3] [--prefix 8] [--mode batch|multi] [--seqs 2] [--n-seq-max 4] [--unified 0|1] [--fa 0|1] [--rs-seq K]\n"); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.use_mmap = false;
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int want = o.prefix + o.n;
    printf("rowexact: mode=%s n=%d prefix=%d seqs=%d n_seq_max=%d unified=%d fa=%d rs_seq=%d threads=%d\n",
           o.mode.c_str(), o.n, o.prefix, o.seqs, o.n_seq_max, o.unified, o.fa, o.rs_seq, o.n_threads);

    if (o.mode == "batch") {
        auto toks = tokenize(vocab, PROMPTS[o.prompt_a % 4], want);
        printf("tokens:"); for (auto t : toks) printf(" %d", t); printf("\n");
        printf("== reference: %d single-token decodes\n", o.n); fflush(stdout);
        run_result ref = run_single(model, o, toks, n_vocab);
        printf("== test: one %d-token batch\n", o.n); fflush(stdout);
        run_result bat = run_batched(model, o, {toks}, n_vocab);
        for (int t = 0; t < o.n; t++) {
            printf("-- token row %d (pos %d)\n", t, o.prefix + t);
            cmp_nodes(bat.batch, o.n, 1, t, ref.per_token[t], 1, 1, 0, o.show, "batch-row vs single");
            cmp_logits(bat.logits[t], ref.logits[t], "batch-row vs single");
        }
        cmp_state(bat.state, ref.state, "seq0 after batch vs after singles");
    } else if (o.mode == "multi") {
        std::vector<std::vector<llama_token>> toks;
        for (int s = 0; s < o.seqs; s++) toks.push_back(tokenize(vocab, PROMPTS[(o.prompt_a + s) % 4], want));
        std::vector<run_result> refs;
        for (int s = 0; s < o.seqs; s++) {
            printf("== reference seq %d: alone, one %d-token batch\n", s, o.n); fflush(stdout);
            refs.push_back(run_batched(model, o, {toks[s]}, n_vocab));
        }
        printf("== reference seq 0: alone, %d single-token decodes\n", o.n); fflush(stdout);
        run_result ref_single = run_single(model, o, toks[0], n_vocab);
        printf("== test: %d seqs x %d tokens in one batch\n", o.seqs, o.n); fflush(stdout);
        run_result mul = run_batched(model, o, toks, n_vocab);
        for (int s = 0; s < o.seqs; s++) for (int t = 0; t < o.n; t++) {
            const int i = s * o.n + t;
            printf("-- seq %d token row %d (pos %d)\n", s, t, o.prefix + t);
            cmp_nodes(mul.batch, o.n, o.seqs, i, refs[s].batch, o.n, 1, t, o.show, "multi-row vs alone-batch-row");
            cmp_logits(mul.logits[i], refs[s].logits[t], "multi-row vs alone-batch-row");
            if (s == 0) cmp_logits(mul.logits[i], ref_single.logits[t], "multi-row vs single");
        }
        cmp_state(mul.state, refs[0].state, "seq0 after multi vs after alone-batch");
        cmp_state(mul.state, ref_single.state, "seq0 after multi vs after singles");
    } else {
        fprintf(stderr, "unknown mode\n"); return 1;
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
