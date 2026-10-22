// llama-weight-handoff: NEW-13 zero-training prefill/decode weight-set handoff probe
// (handoffs/active/iqk-iquant-enablement.md, NEW-13; intake-2467, intake-2485).
//
// Two GGUFs of the SAME architecture are loaded side by side (mmap), one llama_context each.
// For every passage of n_prompt + n_cont tokens:
//   - the prefill weight set processes the first n_prompt-1 prompt tokens;
//   - the sequence state (attention KV + indexer KV + recurrent conv/SSM/PLE state, i.e. everything
//     llama_state_seq_get_data serialises) is copied into the decode context with
//     llama_state_seq_get_data / llama_state_seq_set_data;
//   - the decode weight set teacher-forces the last prompt token plus the first n_cont-1
//     continuation tokens, so ALL n_cont scored predictions (of continuation tokens 0..n_cont-1)
//     come from the decode weights;
//   - per-token KLD(ref || arm) over the full vocabulary, argmax agreement and target NLL are written.
//
// Arms (P = --model-p, D = --model-d):  A = D prefill, D decode   B = P prefill -> D decode
//                                       C = P prefill, P decode   D = D prefill -> P decode
// Reference (--ref): "P" = the P-only run (arm C IS the reference, KLD_C == 0 by construction), or
//                    "R" = a third GGUF (--model-r) run the same way as arm C.
// The tool never decides which reference is right; the caller pins it (NEW-13 PREDECLARED.json).
//
// Safety: refuses unless GGML_FUSED_DECODE_OFF=1 (v10's fused qwen4exp single-token decode aborts on
// the GSQ GGUF, NEW-10 RCA) unless --allow-fused; refuses on any architecture / hparam / vocab /
// state-size mismatch between the weight sets, on a short or failed state read, and on a self-handoff
// round trip that does not reproduce the direct run (--selftest).
//
// Only llama.h is used, so the binary runs against the production store's libllama/libggml
// (LD_LIBRARY_PATH) built from the same commit.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

// ---------------------------------------------------------------------------------------------- util

[[noreturn]] static void die(int rc, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "weight-handoff: REFUSE: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(rc);
}

static void logf_(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "weight-handoff: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += (char) c;
        }
    }
    return o;
}

static std::string fmt_g(double v) {
    char b[48];
    snprintf(b, sizeof(b), "%.7g", v);
    return b;
}

// exit codes (the job script maps them)
enum {
    RC_OK        = 0,
    RC_STOPPED   = 10, // stopped between passages (stop file / deadline); resumable
    RC_USAGE     = 64,
    RC_ENV       = 65, // fused decode not disabled / IQK not set
    RC_LOAD      = 66, // model or context failed
    RC_COMPAT    = 67, // architecture / hparams / vocab mismatch between weight sets
    RC_STATE     = 68, // state copy failed, short read, size or position mismatch
    RC_SELFTEST  = 69, // self-handoff round trip does not reproduce the direct run
    RC_INPUT     = 70, // passages file malformed
    RC_NONFINITE = 71, // non-finite logits
    RC_IO        = 72,
};

// ---------------------------------------------------------------------------------------------- args

struct args_t {
    std::string model_p, model_d, model_r;
    std::string ref = "";          // "P" or "R"
    std::string passages, out, compat_out;
    std::string arms = "A,B,C,D";
    int first = 0, count = -1;
    int n_ctx = 0, n_batch = 2048, n_ubatch = 512;
    int threads = 48, threads_batch = 96;
    int tf_batch = 1;
    int selftest = 4;
    double selftest_tol = 1e-4;
    bool allow_fused = false, allow_no_iqk = false;
    bool check_only = false;
    std::string stop_file;          // checked between passages
    long long deadline = 0;         // unix epoch seconds; no passage is started that would not finish before it
    bool debug_corrupt_selftest = false; // non-vacuity check of the self-test: zero the tail of the restored state
    // generation (coherence) mode
    bool gen = false;
    std::string gen_arm, prompt_file, gen_out, gen_meta;
    int n_gen = 512;
};

static void usage() {
    fprintf(stderr,
        "usage: llama-weight-handoff --model-p P.gguf --model-d D.gguf --ref P|R [--model-r R.gguf]\n"
        "         (--passages FILE.bin --out OUT.jsonl [--arms A,B,C,D] [--first N --count N] [--selftest K]\n"
        "          [--selftest-tol X] [--tf-batch 1] [--stop-file F] [--deadline EPOCH] | --check-only | --gen --gen-arm A|B|C|D --prompt-file F\n"
        "          --gen-out TXT --gen-meta JSON [--n-gen 512])\n"
        "         [--compat-out JSON] [--n-ctx N] [--n-batch 2048] [--n-ubatch 512] [-t 48] [-tb 96]\n"
        "         [--allow-fused] [--allow-no-iqk]\n");
}

static args_t parse_args(int argc, char ** argv) {
    args_t a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&](void) -> std::string {
            if (i + 1 >= argc) { usage(); die(RC_USAGE, "missing value for %s", k.c_str()); }
            return argv[++i];
        };
        if      (k == "--model-p")       a.model_p = val();
        else if (k == "--model-d")       a.model_d = val();
        else if (k == "--model-r")       a.model_r = val();
        else if (k == "--ref")           a.ref = val();
        else if (k == "--passages")      a.passages = val();
        else if (k == "--out")           a.out = val();
        else if (k == "--compat-out")    a.compat_out = val();
        else if (k == "--arms")          a.arms = val();
        else if (k == "--first")         a.first = std::stoi(val());
        else if (k == "--count")         a.count = std::stoi(val());
        else if (k == "--n-ctx")         a.n_ctx = std::stoi(val());
        else if (k == "--n-batch")       a.n_batch = std::stoi(val());
        else if (k == "--n-ubatch")      a.n_ubatch = std::stoi(val());
        else if (k == "-t" || k == "--threads")        a.threads = std::stoi(val());
        else if (k == "-tb" || k == "--threads-batch") a.threads_batch = std::stoi(val());
        else if (k == "--tf-batch")      a.tf_batch = std::stoi(val());
        else if (k == "--selftest")      a.selftest = std::stoi(val());
        else if (k == "--selftest-tol")  a.selftest_tol = std::stod(val());
        else if (k == "--allow-fused")   a.allow_fused = true;
        else if (k == "--allow-no-iqk")  a.allow_no_iqk = true;
        else if (k == "--check-only")    a.check_only = true;
        else if (k == "--stop-file")     a.stop_file = val();
        else if (k == "--deadline")      a.deadline = std::stoll(val());
        else if (k == "--debug-corrupt-selftest") a.debug_corrupt_selftest = true;
        else if (k == "--gen")           a.gen = true;
        else if (k == "--gen-arm")       a.gen_arm = val();
        else if (k == "--prompt-file")   a.prompt_file = val();
        else if (k == "--gen-out")       a.gen_out = val();
        else if (k == "--gen-meta")      a.gen_meta = val();
        else if (k == "--n-gen")         a.n_gen = std::stoi(val());
        else if (k == "-h" || k == "--help") { usage(); exit(0); }
        else { usage(); die(RC_USAGE, "unknown argument %s", k.c_str()); }
    }
    if (a.model_p.empty() || a.model_d.empty()) { usage(); die(RC_USAGE, "--model-p and --model-d are required"); }
    if (a.ref != "P" && a.ref != "R") { usage(); die(RC_USAGE, "--ref must be P or R (the caller pins the reference)"); }
    if (a.ref == "R" && a.model_r.empty()) die(RC_USAGE, "--ref R needs --model-r");
    if (a.ref == "P" && !a.model_r.empty()) die(RC_USAGE, "--ref P takes no --model-r");
    if (a.tf_batch < 1) die(RC_USAGE, "--tf-batch must be >= 1");
    if (!a.check_only && !a.gen && (a.passages.empty() || a.out.empty())) { usage(); die(RC_USAGE, "--passages and --out are required"); }
    if (a.gen && (a.gen_arm.size() != 1 || std::string("ABCD").find(a.gen_arm) == std::string::npos ||
                  a.prompt_file.empty() || a.gen_out.empty() || a.gen_meta.empty())) {
        usage(); die(RC_USAGE, "--gen needs --gen-arm A|B|C|D --prompt-file --gen-out --gen-meta");
    }
    return a;
}

// ---------------------------------------------------------------------------------------------- passages

// NW13 binary: char[4] "NW13", u32 version(1), u32 n_pass, u32 n_prompt, u32 n_cont, then n_pass*(n_prompt+n_cont) i32
struct passages_t {
    uint32_t n_pass = 0, n_prompt = 0, n_cont = 0;
    std::vector<int32_t> ids;
    const int32_t * row(uint32_t i) const { return ids.data() + (size_t) i * (n_prompt + n_cont); }
};

static passages_t read_passages(const std::string & path, int32_t n_vocab) {
    passages_t p;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) die(RC_INPUT, "cannot open passages %s", path.c_str());
    char magic[4];
    uint32_t hdr[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "NW13", 4) != 0) die(RC_INPUT, "bad magic in %s", path.c_str());
    if (fread(hdr, 4, 4, f) != 4) die(RC_INPUT, "short header in %s", path.c_str());
    if (hdr[0] != 1) die(RC_INPUT, "unsupported passages version %u", hdr[0]);
    p.n_pass = hdr[1]; p.n_prompt = hdr[2]; p.n_cont = hdr[3];
    if (p.n_pass == 0 || p.n_prompt < 2 || p.n_cont < 1) die(RC_INPUT, "degenerate passages header");
    const size_t n = (size_t) p.n_pass * (p.n_prompt + p.n_cont);
    p.ids.resize(n);
    if (fread(p.ids.data(), 4, n, f) != n) die(RC_INPUT, "short passages body");
    char extra;
    if (fread(&extra, 1, 1, f) != 0) die(RC_INPUT, "trailing bytes in passages file");
    fclose(f);
    for (size_t i = 0; i < n; ++i) {
        if (p.ids[i] < 0 || p.ids[i] >= n_vocab) die(RC_INPUT, "token id %d out of vocab at %zu", p.ids[i], i);
    }
    return p;
}

// ---------------------------------------------------------------------------------------------- model

static std::string meta_str(const llama_model * m, int i, bool key) {
    std::vector<char> buf(1 << 16);
    int32_t n = key ? llama_model_meta_key_by_index(m, i, buf.data(), buf.size())
                    : llama_model_meta_val_str_by_index(m, i, buf.data(), buf.size());
    if (n < 0) return "";
    if ((size_t) n >= buf.size()) {
        buf.resize(n + 1);
        n = key ? llama_model_meta_key_by_index(m, i, buf.data(), buf.size())
                : llama_model_meta_val_str_by_index(m, i, buf.data(), buf.size());
    }
    return std::string(buf.data(), std::max(0, n));
}

static std::vector<std::pair<std::string, std::string>> meta_all(const llama_model * m) {
    std::vector<std::pair<std::string, std::string>> r;
    const int n = llama_model_meta_count(m);
    for (int i = 0; i < n; ++i) r.emplace_back(meta_str(m, i, true), meta_str(m, i, false));
    return r;
}

static llama_model * load_model(const std::string & path) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap     = true;
    mp.use_mlock    = false;
    llama_model * m = llama_model_load_from_file(path.c_str(), mp);
    if (!m) die(RC_LOAD, "failed to load %s", path.c_str());
    return m;
}

static llama_context * make_ctx(llama_model * m, const args_t & a, uint32_t n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = a.n_batch;
    cp.n_ubatch        = a.n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = a.threads;
    cp.n_threads_batch = a.threads_batch;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // identical on every context: V layout is part of the state
    cp.type_k          = GGML_TYPE_F16;                 // the Q38FN recipe forbids KV quantisation
    cp.type_v          = GGML_TYPE_F16;
    cp.offload_kqv     = false;
    cp.no_perf         = true;
    llama_context * c = llama_init_from_model(m, cp);
    if (!c) die(RC_LOAD, "failed to create context (n_ctx %u)", n_ctx);
    return c;
}

// The handoff is only defined between weight sets whose memory layouts are identical. Everything that shapes the
// serialised sequence state lives in the "<arch>." metadata (layer count, layer types, head counts, GDN conv/state
// dims, PLE conv kernel and n-gram size, indexer heads, compress ratios, hyper-connection count) plus the vocab
// (PLE hashes token ids). Quantisation keys (general.*, quantize.*, split.*) are allowed to differ.
static std::string check_compat(const llama_model * mp, const llama_model * md, const char * name_d) {
    std::ostringstream js;
    std::vector<std::string> diffs;
    auto ival = [&](const char * what, int64_t x, int64_t y) {
        js << "\"" << what << "\":[" << x << "," << y << "],";
        if (x != y) diffs.push_back(std::string(what) + " " + std::to_string(x) + " != " + std::to_string(y));
    };
    ival("n_layer", llama_model_n_layer(mp), llama_model_n_layer(md));
    ival("n_layer_nextn", llama_model_n_layer_nextn(mp), llama_model_n_layer_nextn(md));
    ival("n_embd", llama_model_n_embd(mp), llama_model_n_embd(md));
    ival("n_head", llama_model_n_head(mp), llama_model_n_head(md));
    ival("n_head_kv", llama_model_n_head_kv(mp), llama_model_n_head_kv(md));
    ival("n_swa", llama_model_n_swa(mp), llama_model_n_swa(md));
    ival("n_ctx_train", llama_model_n_ctx_train(mp), llama_model_n_ctx_train(md));
    ival("is_hybrid", llama_model_is_hybrid(mp), llama_model_is_hybrid(md));
    ival("is_recurrent", llama_model_is_recurrent(mp), llama_model_is_recurrent(md));
    const llama_vocab * vp = llama_model_get_vocab(mp);
    const llama_vocab * vd = llama_model_get_vocab(md);
    ival("n_vocab", llama_vocab_n_tokens(vp), llama_vocab_n_tokens(vd));

    auto kp = meta_all(mp), kd = meta_all(md);
    std::string arch;
    for (auto & kv : kp) if (kv.first == "general.architecture") arch = kv.second;
    std::string arch_d;
    for (auto & kv : kd) if (kv.first == "general.architecture") arch_d = kv.second;
    js << "\"arch\":[\"" << json_escape(arch) << "\",\"" << json_escape(arch_d) << "\"],";
    if (arch.empty() || arch != arch_d) diffs.push_back("general.architecture " + arch + " != " + arch_d);

    auto shaping = [&](const std::string & k) {
        if (!arch.empty() && k.rfind(arch + ".", 0) == 0) return true;
        static const char * tok_keys[] = {"tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.eos_token_id",
                                          "tokenizer.ggml.bos_token_id", "tokenizer.ggml.padding_token_id",
                                          "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"};
        for (auto * t : tok_keys) if (k == t) return true;
        return false;
    };
    int n_cmp = 0;
    std::set<std::string> seen;
    for (auto & kv : kp) {
        if (!shaping(kv.first)) continue;
        seen.insert(kv.first);
        ++n_cmp;
        bool found = false;
        for (auto & kv2 : kd) if (kv2.first == kv.first) {
            found = true;
            if (kv2.second != kv.second) diffs.push_back("meta " + kv.first + ": '" + kv.second.substr(0, 80) + "' != '" + kv2.second.substr(0, 80) + "'");
        }
        if (!found) diffs.push_back("meta " + kv.first + " missing in " + name_d);
    }
    for (auto & kv : kd) if (shaping(kv.first) && !seen.count(kv.first)) diffs.push_back("meta " + kv.first + " missing in P");
    js << "\"meta_keys_compared\":" << n_cmp << ",";

    int64_t vdiff = 0;
    const int nv = std::min(llama_vocab_n_tokens(vp), llama_vocab_n_tokens(vd));
    for (int t = 0; t < nv; ++t) {
        const char * a = llama_vocab_get_text(vp, t);
        const char * b = llama_vocab_get_text(vd, t);
        if ((a == nullptr) != (b == nullptr) || (a && strcmp(a, b) != 0)) ++vdiff;
    }
    if (vdiff) diffs.push_back("vocab text differs at " + std::to_string(vdiff) + " ids");
    js << "\"vocab_text_diffs\":" << vdiff << ",";
    js << "\"diffs\":[";
    for (size_t i = 0; i < diffs.size(); ++i) js << (i ? "," : "") << "\"" << json_escape(diffs[i]) << "\"";
    js << "],\"ok\":" << (diffs.empty() ? "true" : "false");
    return js.str();
}

// ---------------------------------------------------------------------------------------------- decode helpers

static void decode_or_die(llama_context * c, llama_batch & b, const char * what) {
    const int32_t rc = llama_decode(c, b);
    if (rc != 0) die(RC_LOAD, "llama_decode failed (%d) during %s", rc, what);
}

// process ids[0..n) at positions pos0.. with no scored outputs except the last token (kept for a finite check)
static void prefill(llama_context * c, const int32_t * ids, int n, int n_batch, const char * what) {
    llama_batch b = llama_batch_init(n_batch, 0, 1);
    for (int s = 0; s < n; s += n_batch) {
        const int m = std::min(n_batch, n - s);
        b.n_tokens = m;
        for (int j = 0; j < m; ++j) {
            b.token[j] = ids[s + j];
            b.pos[j] = s + j;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = (s + j == n - 1);
        }
        decode_or_die(c, b, what);
    }
    llama_batch_free(b);
}

// teacher-force ids[0..n) starting at position pos0; on_logits(k, logits) is called for k = 0..n-1 in order
template <class F>
static void teacher_force(llama_context * c, const int32_t * ids, int n, int pos0, int chunk, const char * what, F on_logits) {
    llama_batch b = llama_batch_init(std::max(chunk, 1), 0, 1);
    for (int s = 0; s < n; s += chunk) {
        const int m = std::min(chunk, n - s);
        b.n_tokens = m;
        for (int j = 0; j < m; ++j) {
            b.token[j] = ids[s + j];
            b.pos[j] = pos0 + s + j;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = 1;
        }
        decode_or_die(c, b, what);
        for (int j = 0; j < m; ++j) {
            const float * lg = llama_get_logits_ith(c, j);
            if (!lg) die(RC_LOAD, "no logits at %d during %s", s + j, what);
            on_logits(s + j, lg);
        }
    }
    llama_batch_free(b);
}

static void log_softmax(const float * lg, int nv, std::vector<double> & out, bool & finite) {
    out.resize(nv);
    double mx = -INFINITY;
    for (int v = 0; v < nv; ++v) {
        if (!std::isfinite(lg[v])) finite = false;
        mx = std::max(mx, (double) lg[v]);
    }
    double s = 0.0;
    for (int v = 0; v < nv; ++v) s += std::exp((double) lg[v] - mx);
    const double lse = mx + std::log(s);
    for (int v = 0; v < nv; ++v) out[v] = (double) lg[v] - lse;
}

static int argmax_d(const double * x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[b]) b = i;
    return b;
}

struct state_buf_t {
    std::vector<uint8_t> data;
    size_t size = 0;
    llama_pos p0 = -1, p1 = -1;   // the source memory's own seq_pos_min/max when the state was taken
};

// pos_max: the last prefilled position. The source must report it as its max; its min is whatever the memory type
// reports (a hybrid memory reports max(attn min, recurrent min), i.e. the recurrent cell's single position).
static void state_get(llama_context * c, state_buf_t & sb, llama_pos pos_max, const char * what) {
    llama_memory_t mem = llama_get_memory(c);
    sb.p0 = llama_memory_seq_pos_min(mem, 0);
    sb.p1 = llama_memory_seq_pos_max(mem, 0);
    if (sb.p1 != pos_max) die(RC_STATE, "%s: source holds positions up to %d, expected %d", what, sb.p1, pos_max);
    const size_t n = llama_state_seq_get_size(c, 0);
    if (n == 0) die(RC_STATE, "%s: llama_state_seq_get_size returned 0", what);
    sb.data.resize(n);
    const size_t got = llama_state_seq_get_data(c, sb.data.data(), n, 0);
    if (got != n) die(RC_STATE, "%s: get_data wrote %zu of %zu bytes", what, got, n);
    sb.size = n;
}

// clear dst, load the state, require the whole buffer to be consumed and the restored position range to equal the source's
static void state_set(llama_context * dst, const state_buf_t & sb, const char * what) {
    llama_memory_t mem = llama_get_memory(dst);
    llama_memory_clear(mem, true);
    const size_t rd = llama_state_seq_set_data(dst, sb.data.data(), sb.size, 0);
    if (rd == 0) die(RC_STATE, "%s: llama_state_seq_set_data failed (incompatible state layout)", what);
    if (rd != sb.size) die(RC_STATE, "%s: set_data consumed %zu of %zu bytes", what, rd, sb.size);
    const llama_pos p0 = llama_memory_seq_pos_min(mem, 0), p1 = llama_memory_seq_pos_max(mem, 0);
    if (p0 != sb.p0 || p1 != sb.p1) die(RC_STATE, "%s: restored positions [%d, %d], source had [%d, %d]", what, p0, p1, sb.p0, sb.p1);
}

// ---------------------------------------------------------------------------------------------- per-arm scoring

struct arm_rec_t {
    std::vector<double> kld, nll;
    std::string top1;   // '1' = argmax agrees with the reference
    bool finite = true;
    double ms = 0.0;
};

static std::string arm_json(const arm_rec_t & r) {
    std::string s = "{\"kld\":[";
    for (size_t i = 0; i < r.kld.size(); ++i) { if (i) s += ","; s += fmt_g(r.kld[i]); }
    s += "],\"nll\":[";
    for (size_t i = 0; i < r.nll.size(); ++i) { if (i) s += ","; s += fmt_g(r.nll[i]); }
    s += "],\"top1\":\"" + r.top1 + "\",\"finite\":" + (r.finite ? "true" : "false") + ",\"tf_ms\":" + fmt_g(r.ms) + "}";
    return s;
}

// ---------------------------------------------------------------------------------------------- main

static void check_env(const args_t & a) {
    const char * fo = getenv("GGML_FUSED_DECODE_OFF");
    if (!a.allow_fused && !(fo && strcmp(fo, "1") == 0)) {
        die(RC_ENV, "GGML_FUSED_DECODE_OFF=1 is required (v10 fused qwen4exp decode aborts on the GSQ GGUF; registered Q38FN serving env)");
    }
    const char * iq = getenv("GGML_IQK");
    if (!a.allow_no_iqk && !(iq && strcmp(iq, "1") == 0)) die(RC_ENV, "GGML_IQK=1 is required (codified CPU recipe)");
}

static std::set<int> read_done(const std::string & path) {
    std::set<int> done;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line.back() != '}') continue;                 // torn tail of a killed run
        if (line.find("\"complete\":true") == std::string::npos) continue;
        const size_t k = line.find("\"passage\":");
        if (k == std::string::npos) continue;
        done.insert(atoi(line.c_str() + k + 10));
    }
    return done;
}

int main(int argc, char ** argv) {
    args_t a = parse_args(argc, argv);
    check_env(a);

    llama_backend_init();
    logf_("system: %s", llama_print_system_info());
    logf_("env: GGML_IQK=%s GGML_FUSED_DECODE_OFF=%s OMP_NUM_THREADS=%s", getenv("GGML_IQK") ? getenv("GGML_IQK") : "",
          getenv("GGML_FUSED_DECODE_OFF") ? getenv("GGML_FUSED_DECODE_OFF") : "", getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "");

    double t0 = now_ms();
    llama_model * mP = load_model(a.model_p);
    llama_model * mD = load_model(a.model_d);
    llama_model * mR = a.ref == "R" ? load_model(a.model_r) : nullptr;
    logf_("models loaded in %.1f s", (now_ms() - t0) / 1000.0);

    std::string compat = "{\"P_vs_D\":{" + check_compat(mP, mD, "D") + "}";
    const bool ok_pd = compat.find("\"ok\":true") != std::string::npos;
    bool ok_pr = true;
    if (mR) {
        const std::string cr = check_compat(mP, mR, "R");
        ok_pr = cr.find("\"ok\":true") != std::string::npos;
        compat += ",\"P_vs_R\":{" + cr + "}";
    }
    compat += "}";
    if (!a.compat_out.empty()) {
        FILE * f = fopen(a.compat_out.c_str(), "w");
        if (!f) die(RC_IO, "cannot write %s", a.compat_out.c_str());
        fprintf(f, "%s\n", compat.c_str());
        fclose(f);
    }
    logf_("compat: %s", compat.c_str());
    if (!ok_pd) die(RC_COMPAT, "P and D weight sets are not handoff-compatible (see compat report)");
    if (!ok_pr) die(RC_COMPAT, "P and the reference are not the same architecture/vocab (see compat report)");

    const llama_vocab * vocab = llama_model_get_vocab(mP);
    const int nv = llama_vocab_n_tokens(vocab);

    // ------------------------------------------------------------------ generation (coherence) mode
    if (a.gen) {
        std::ifstream pf(a.prompt_file, std::ios::binary);
        if (!pf) die(RC_INPUT, "cannot read %s", a.prompt_file.c_str());
        std::string text((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(),
                               llama_vocab_get_add_bos(vocab), false);
        if (n < 2) die(RC_INPUT, "prompt tokenised to %d tokens", n);
        toks.resize(n);
        const char arm = a.gen_arm[0];
        const bool pre_p = (arm == 'B' || arm == 'C');
        const bool dec_p = (arm == 'C' || arm == 'D');
        const uint32_t n_ctx = a.n_ctx > 0 ? a.n_ctx : (uint32_t) (n + a.n_gen + 64);
        llama_context * cPre = make_ctx(pre_p ? mP : mD, a, n_ctx);
        llama_context * cDec = (pre_p == dec_p) ? cPre : make_ctx(dec_p ? mP : mD, a, n_ctx);
        double tp = now_ms();
        prefill(cPre, toks.data(), n - 1, a.n_batch, "gen prefill");
        tp = now_ms() - tp;
        size_t sbytes = 0;
        if (cDec != cPre) {
            state_buf_t sb;
            state_get(cPre, sb, n - 2, "gen handoff");
            state_set(cDec, sb, "gen handoff");
            sbytes = sb.size;
        }
        std::string out;
        int gen = 0;
        bool stopped = false;
        llama_token cur = toks[n - 1];
        double td = now_ms();
        llama_batch b = llama_batch_init(1, 0, 1);
        for (; gen < a.n_gen; ++gen) {
            b.n_tokens = 1; b.token[0] = cur; b.pos[0] = n - 1 + gen; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            decode_or_die(cDec, b, "gen decode");
            const float * lg = llama_get_logits_ith(cDec, 0);
            int best = 0;
            for (int v = 1; v < nv; ++v) if (lg[v] > lg[best]) best = v;
            if (!std::isfinite(lg[best])) die(RC_NONFINITE, "non-finite logits at generated token %d", gen);
            if (llama_vocab_is_eog(vocab, best)) { stopped = true; break; }
            char piece[512];
            const int k = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, false);
            if (k > 0) out.append(piece, k);
            cur = best;
        }
        td = now_ms() - td;
        llama_batch_free(b);
        FILE * fo = fopen(a.gen_out.c_str(), "wb");
        if (!fo) die(RC_IO, "cannot write %s", a.gen_out.c_str());
        fwrite(out.data(), 1, out.size(), fo);
        fclose(fo);
        FILE * fm = fopen(a.gen_meta.c_str(), "w");
        if (!fm) die(RC_IO, "cannot write %s", a.gen_meta.c_str());
        fprintf(fm, "{\"schema\":\"epyc.new13.gen.v1\",\"arm\":\"%c\",\"prompt_tokens\":%d,\"gen_tokens\":%d,\"finish\":\"%s\","
                    "\"state_bytes\":%zu,\"prefill_ms\":%s,\"decode_ms\":%s}\n",
                arm, n, gen, stopped ? "stop" : "length", sbytes, fmt_g(tp).c_str(), fmt_g(td).c_str());
        fclose(fm);
        logf_("gen arm %c: prompt %d tokens, generated %d (%s)", arm, n, gen, stopped ? "stop" : "length");
        if (cDec != cPre) llama_free(cDec);
        llama_free(cPre);
        llama_model_free(mP); llama_model_free(mD); if (mR) llama_model_free(mR);
        llama_backend_free();
        return RC_OK;
    }

    if (a.check_only) {
        // contexts are created too, so a context/memory construction failure also surfaces here
        llama_context * cP = make_ctx(mP, a, a.n_ctx > 0 ? a.n_ctx : 4096);
        llama_context * cD = make_ctx(mD, a, a.n_ctx > 0 ? a.n_ctx : 4096);
        logf_("check-only: compatible; contexts created");
        llama_free(cP); llama_free(cD);
        llama_model_free(mP); llama_model_free(mD); if (mR) llama_model_free(mR);
        llama_backend_free();
        return RC_OK;
    }

    // ------------------------------------------------------------------ KLD mode
    passages_t ps = read_passages(a.passages, nv);
    const int NP = (int) ps.n_prompt, NC = (int) ps.n_cont;
    const uint32_t n_ctx = a.n_ctx > 0 ? (uint32_t) a.n_ctx : (uint32_t) (NP + NC + 64);
    if ((int) n_ctx < NP + NC) die(RC_USAGE, "--n-ctx %u < passage length %d", n_ctx, NP + NC);
    const int first = a.first;
    const int last = a.count < 0 ? (int) ps.n_pass : std::min((int) ps.n_pass, first + a.count);
    const bool want[4] = {a.arms.find('A') != std::string::npos, a.arms.find('B') != std::string::npos,
                          a.arms.find('C') != std::string::npos, a.arms.find('D') != std::string::npos};

    llama_context * cP = make_ctx(mP, a, n_ctx);
    llama_context * cD = make_ctx(mD, a, n_ctx);
    llama_context * cR = mR ? make_ctx(mR, a, n_ctx) : nullptr;

    std::set<int> done = read_done(a.out);
    logf_("passages %u x (%d + %d); range [%d, %d); %zu already complete in %s; ref=%s tf_batch=%d",
          ps.n_pass, NP, NC, first, last, done.size(), a.out.c_str(), a.ref.c_str(), a.tf_batch);

    const int fd = open(a.out.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) die(RC_IO, "cannot open %s", a.out.c_str());

    // reference distribution per scored position, stored as float probabilities p_r (rounded once) plus
    // negent[k] = sum_v p_r * ln p_r computed from the SAME rounded p_r, so KLD(ref||ref) is exactly 0 and
    // KLD(ref||arm) = negent[k] - sum_v p_r * ln p_arm in double.
    std::vector<float> ref_p((size_t) NC * nv);
    std::vector<double> ref_negent(NC);
    std::vector<int> ref_top1(NC);
    std::vector<double> ref_nll(NC);
    std::vector<float> keepA, keepC;               // only for self-test passages
    std::vector<double> lp;
    state_buf_t SP, SD, tmp;
    int n_selftested = 0;
    double t_passages_ms = 0.0;
    int n_passages_timed = 0;
    bool stopped = false;

    for (int pi = first; pi < last; ++pi) {
        if (done.count(pi)) continue;
        if (!a.stop_file.empty() && access(a.stop_file.c_str(), F_OK) == 0) {
            logf_("stop file %s present: stopping before passage %d", a.stop_file.c_str(), pi);
            stopped = true;
            break;
        }
        if (a.deadline > 0) {
            // first passage of a run also pays the page-in of both weight sets: assume 10 min until one is timed
            const double est_s = n_passages_timed ? 1.3 * t_passages_ms / n_passages_timed / 1000.0 : 600.0;
            if ((double) time(nullptr) + est_s > (double) a.deadline) {
                logf_("deadline: %.0f s left, a passage needs ~%.0f s: stopping before passage %d",
                      (double) a.deadline - (double) time(nullptr), est_s, pi);
                stopped = true;
                break;
            }
        }
        const double t_passage0 = now_ms();
        const int32_t * row = ps.row(pi);
        const int32_t * tf_in = row + (NP - 1);    // last prompt token + first NC-1 continuation tokens
        const int32_t * target = row + NP;         // the NC continuation tokens being predicted
        const llama_pos pos_max = NP - 2;          // prefill covers positions 0 .. NP-2
        const bool st = n_selftested < a.selftest;
        double ms_pre_p = 0, ms_pre_d = 0, ms_pre_r = 0;
        bool finite_all = true;
        arm_rec_t rec[4];                          // A B C D

        // score one arm's logits against the reference; also stores the reference if 'store_ref'
        auto scorer = [&](arm_rec_t & r, bool store_ref, std::vector<float> * keep) {
            r.kld.assign(NC, 0.0); r.nll.assign(NC, 0.0); r.top1.assign(NC, '0');
            if (keep) keep->resize((size_t) NC * nv);
            return [&, store_ref, keep](int k, const float * lg) {
                bool fin = true;
                log_softmax(lg, nv, lp, fin);
                if (!fin) { r.finite = false; finite_all = false; }
                float * pr = ref_p.data() + (size_t) k * nv;
                if (store_ref) {
                    double ne = 0.0;
                    for (int v = 0; v < nv; ++v) {
                        pr[v] = (float) std::exp(lp[v]);
                        ne += (double) pr[v] * lp[v];
                    }
                    ref_negent[k] = ne;
                    ref_top1[k] = argmax_d(lp.data(), nv);
                    ref_nll[k] = -lp[target[k]];
                }
                if (keep) { float * kk = keep->data() + (size_t) k * nv; for (int v = 0; v < nv; ++v) kk[v] = (float) lp[v]; }
                double dot = 0.0;
                for (int v = 0; v < nv; ++v) dot += (double) pr[v] * lp[v];
                r.kld[k] = ref_negent[k] - dot;
                r.nll[k] = -lp[target[k]];
                r.top1[k] = (argmax_d(lp.data(), nv) == ref_top1[k]) ? '1' : '0';
            };
        };
        auto run_tf = [&](llama_context * c, arm_rec_t & r, bool store_ref, std::vector<float> * keep, const char * what) {
            double t = now_ms();
            teacher_force(c, tf_in, NC, NP - 1, a.tf_batch, what, scorer(r, store_ref, keep));
            r.ms = now_ms() - t;
        };

        // reference R (if a third weight set): prefill + teacher-force exactly like arm C
        if (cR) {
            llama_memory_clear(llama_get_memory(cR), true);
            double t = now_ms(); prefill(cR, row, NP - 1, a.n_batch, "R prefill"); ms_pre_r = now_ms() - t;
            arm_rec_t rr;
            run_tf(cR, rr, true, nullptr, "R teacher-force");
            if (!rr.finite) die(RC_NONFINITE, "non-finite reference logits at passage %d", pi);
        }

        // 1. P prefill, keep its state
        llama_memory_clear(llama_get_memory(cP), true);
        { double t = now_ms(); prefill(cP, row, NP - 1, a.n_batch, "P prefill"); ms_pre_p = now_ms() - t; }
        state_get(cP, SP, pos_max, "P state");
        // 2. arm C (P decodes its own prefill); with --ref P it IS the reference
        if (want[2] || !cR) run_tf(cP, rec[2], !cR, st ? &keepC : nullptr, "C teacher-force");
        if (!cR && !rec[2].finite) die(RC_NONFINITE, "non-finite reference (P) logits at passage %d", pi);
        // 3. arm B: P's state into D's context, D decodes
        if (want[1]) { state_set(cD, SP, "B handoff P->D"); run_tf(cD, rec[1], false, nullptr, "B teacher-force"); }
        // 4. D prefill, keep its state; the two serialised states must have identical size (same layout)
        llama_memory_clear(llama_get_memory(cD), true);
        { double t = now_ms(); prefill(cD, row, NP - 1, a.n_batch, "D prefill"); ms_pre_d = now_ms() - t; }
        state_get(cD, SD, pos_max, "D state");
        if (SD.size != SP.size || SD.p0 != SP.p0 || SD.p1 != SP.p1) die(RC_STATE, "passage %d: states differ (P %zu bytes [%d,%d], D %zu bytes [%d,%d]): layouts are not identical", pi, SP.size, SP.p0, SP.p1, SD.size, SD.p0, SD.p1);
        // 5. arm A (D decodes its own prefill)
        if (want[0]) run_tf(cD, rec[0], false, st ? &keepA : nullptr, "A teacher-force");
        // 6. arm D (reverse): D's state into P's context, P decodes
        if (want[3]) { state_set(cP, SD, "D handoff D->P"); run_tf(cP, rec[3], false, nullptr, "D teacher-force"); }

        // 7. self-test: a clear + restore round trip inside the same context must reproduce the direct run.
        //    This is what proves the serialised state is COMPLETE for this architecture (KV, indexer KV, GDN conv/SSM,
        //    PLE conv history, the cells' token ids for the PLE n-gram hash); a dropped component shows up as KLD >> tol.
        std::string st_json = "null";
        if (st && want[0] && want[2]) {
            auto roundtrip_max = [&](llama_context * c, const state_buf_t & sb0, const std::vector<float> & keep, const char * what) {
                state_buf_t corrupt;
                if (a.debug_corrupt_selftest) {
                    // zero 1% of the serialised bytes in the middle of the buffer (inside a tensor payload, not a header);
                    // the self-test MUST then fail, proving it can see a state component that is not carried over
                    corrupt = sb0;
                    std::fill(corrupt.data.begin() + corrupt.size / 2, corrupt.data.begin() + corrupt.size / 2 + corrupt.size / 100, 0);
                }
                const state_buf_t & sb = a.debug_corrupt_selftest ? corrupt : sb0;
                state_set(c, sb, what);
                double mx = 0.0;
                int top1_diff = 0;
                teacher_force(c, tf_in, NC, NP - 1, a.tf_batch, what, [&](int k, const float * lg) {
                    bool fin = true;
                    log_softmax(lg, nv, lp, fin);
                    const float * kk = keep.data() + (size_t) k * nv;
                    double kl = 0.0;
                    int kb = 0;
                    for (int v = 0; v < nv; ++v) { kl += std::exp((double) kk[v]) * ((double) kk[v] - lp[v]); if (kk[v] > kk[kb]) kb = v; }
                    if (!fin) kl = INFINITY;
                    mx = std::max(mx, std::fabs(kl));
                    if (argmax_d(lp.data(), nv) != kb) ++top1_diff;
                });
                return std::make_pair(mx, top1_diff);
            };
            auto rp = roundtrip_max(cP, SP, keepC, "selftest P->P");
            auto rd = roundtrip_max(cD, SD, keepA, "selftest D->D");
            st_json = "{\"P_roundtrip_max_kld\":" + fmt_g(rp.first) + ",\"P_roundtrip_top1_diff\":" + std::to_string(rp.second) +
                      ",\"D_roundtrip_max_kld\":" + fmt_g(rd.first) + ",\"D_roundtrip_top1_diff\":" + std::to_string(rd.second) +
                      ",\"tol\":" + fmt_g(a.selftest_tol) + "}";
            logf_("passage %d selftest: P round trip max KLD %.3g (top1 diff %d), D round trip max KLD %.3g (top1 diff %d)",
                  pi, rp.first, rp.second, rd.first, rd.second);
            if (!(rp.first <= a.selftest_tol) || !(rd.first <= a.selftest_tol)) {
                die(RC_SELFTEST, "passage %d: state round trip does not reproduce the direct run (P %.3g, D %.3g > tol %.3g): "
                                 "the serialised state is incomplete for this architecture", pi, rp.first, rd.first, a.selftest_tol);
            }
            ++n_selftested;
        }

        // record
        std::string line = "{\"schema\":\"epyc.new13.passage.v1\",\"passage\":" + std::to_string(pi) +
            ",\"n_prompt\":" + std::to_string(NP) + ",\"n_cont\":" + std::to_string(NC) + ",\"ref\":\"" + a.ref + "\"" +
            ",\"tf_batch\":" + std::to_string(a.tf_batch) + ",\"state_bytes\":" + std::to_string(SP.size) +
            ",\"t_ms\":{\"prefill_P\":" + fmt_g(ms_pre_p) + ",\"prefill_D\":" + fmt_g(ms_pre_d) + ",\"prefill_R\":" + fmt_g(ms_pre_r) + "}" +
            ",\"ref_nll\":[";
        for (int k = 0; k < NC; ++k) { if (k) line += ","; line += fmt_g(ref_nll[k]); }
        line += "],\"arms\":{";
        const char * names[4] = {"A", "B", "C", "D"};
        bool firstarm = true;
        for (int i = 0; i < 4; ++i) {
            if (!(want[i] || (i == 2 && !cR))) continue;
            line += std::string(firstarm ? "" : ",") + "\"" + names[i] + "\":" + arm_json(rec[i]);
            firstarm = false;
        }
        line += "},\"selftest\":" + st_json + ",\"finite\":" + (finite_all ? "true" : "false") + ",\"complete\":true}\n";
        if (write(fd, line.data(), line.size()) != (ssize_t) line.size()) die(RC_IO, "short write to %s", a.out.c_str());
        fsync(fd);

        double mA = 0, mB = 0, mC = 0, mDr = 0;
        for (int k = 0; k < NC; ++k) { mA += rec[0].kld[k]; mB += rec[1].kld[k]; mC += rec[2].kld[k]; mDr += rec[3].kld[k]; }
        logf_("passage %d done: mean KLD A %.4f B %.4f C %.4f D %.4f | prefill P %.1fs D %.1fs | tf A %.1fs B %.1fs C %.1fs D %.1fs",
              pi, mA / NC, mB / NC, mC / NC, mDr / NC, ms_pre_p / 1000, ms_pre_d / 1000,
              rec[0].ms / 1000, rec[1].ms / 1000, rec[2].ms / 1000, rec[3].ms / 1000);
        if (!finite_all) die(RC_NONFINITE, "passage %d: non-finite logits (recorded)", pi);
        t_passages_ms += now_ms() - t_passage0;
        ++n_passages_timed;
    }
    close(fd);

    if (cR) llama_free(cR);
    llama_free(cD); llama_free(cP);
    if (mR) llama_model_free(mR);
    llama_model_free(mD); llama_model_free(mP);
    llama_backend_free();
    if (stopped) {
        logf_("stopped early after %d passage(s) this run (resumable)", n_passages_timed);
        return RC_STOPPED;
    }
    logf_("all requested passages complete (%d this run, mean %.1f s/passage)", n_passages_timed,
          n_passages_timed ? t_passages_ms / n_passages_timed / 1000.0 : 0.0);
    return RC_OK;
}
