// llama-moe-trace: per-token MoE routing tap (INF-72 RT-1 / RT-2 / RT-2.1 / RT-3).
//
// Taps the router's top-k expert selection ("ffn_moe_topk-<layer>") of every MoE layer for every
// token through the backend-scheduler eval callback, with the model weights untouched, and writes
// the epyc.moe.routing_trace.v1 on-disk contract (see moe-trace-core.h and
// scripts/kernel_rnd/moe_routing_tap/CPP_CHANGE_SPEC.md in epyc-inference-research):
//
//   <MOE_TRACE_OUT>/shard-<first_window>/routing_trace.json          manifest (schema-valid)
//   <MOE_TRACE_OUT>/shard-<first_window>/routing_trace.rows.bin      one 16-byte row per token
//   <MOE_TRACE_OUT>/shard-<first_window>/routing_trace.ids.i16       int16 [rows, n_layers, top_k]
//   <MOE_TRACE_OUT>/shard-<first_window>/routing_trace.eor.json      EOR / IR_t rider + chance baselines
//   <MOE_TRACE_OUT>/shard-<first_window>/routing_trace.provenance.json
//   (SRP only) routing_trace.argsort.i16                             int16 [rows, n_layers, n_experts]
//
// Port of github.com/Shriniwas410/cacheable-by-design @ 03366ef1 llama-moe-trace/moe-trace.cpp
// (MIT, Copyright (c) 2026 Shriniwas Ramesh Suram). Differences, each one a verified adjustment:
//   1. int16 expert ids, not int8 (256/512-expert fleet); n_experts / top_k read PER LAYER from the graph
//   2. -b N with the per-window llama_memory_clear gives [samples, N, top_k] by reshape; ubatch is forced
//      equal to batch so that one llama_decode is exactly one forward pass
//   3. SRP (MOE_TRACE_SRP=1) additionally taps "ffn_moe_argsort-<layer>" (src/llama-graph.cpp, cb of
//      selected_experts->src[0]); SCH needs only the top-k the tap always emits
//   4. any arch whose MoE goes through build_moe_ffn() is covered (qwen35moe, qwen3next, qwen4exp, ...);
//      the fused single-token decode of qwen4exp bypasses the graph entirely, so it is forced OFF
//      (GGML_FUSED_DECODE_OFF=1) and a forward with a missing layer is a hard error, never a short trace
//   5. RT-3: the router tensor is a strided VIEW; it is read row by row at i * nb[1] and every
//      prefill forward is checked for the stride-hazard signature (moe_trace::hazard_check)
//   6. RT-2.1 columns: phase (prefill/decode), step, spec_block; optional greedy decode leg
//   7. RT-2 rider: EOR / IR_t with K/N chance baselines, written as routing_trace.eor.json
//
// Usage (the corpus comes from -f exactly like llama-perplexity; run it with the store's libs and
// prove linkage first - see INF-72-RT1/trace_job.sh):
//   MOE_TRACE_OUT=dir MOE_TRACE_DOMAIN=prose llama-moe-trace -m model.gguf -f corpus.txt
//       -c 1024 -b 512 -t 16 -tb 16 -ngl 0 --no-mmap -fa on
//
// Environment:
//   MOE_TRACE_OUT            output directory (required)
//   MOE_TRACE_DOMAIN         label stored in the manifest (default "unspecified")
//   MOE_TRACE_START_SAMPLE   first window of the token stream to trace (default 0) - resume point
//   MOE_TRACE_MAX_SAMPLES    stop after this many windows (default 0 = to the end of the corpus)
//   MOE_TRACE_SHARD_SAMPLES  windows per shard directory (default 64; 0 = one shard)
//   MOE_TRACE_DECODE_TOKENS  greedy decode tokens appended to every window (default 0)
//   MOE_TRACE_SRP            1 = also write the full argsort order (large)
//   MOE_TRACE_KEEP_TAIL      1 = also trace a final partial window of >= 8 tokens (default: full windows only)
//   MOE_TRACE_STOPFILE       stop cleanly at the next window boundary when this path exists
//   MOE_TRACE_DEADLINE       epoch seconds; start no window after it
//   MOE_TRACE_SELFTEST_ONLY  1 = run the no-model selftest and exit 0/4
//
// Exit codes: 0 done (corpus/budget exhausted)  10 stopped early (resumable)  1 error
//             2 tap incomplete (a layer missing, wrong row count: a short trace is refused)
//             3 stride-hazard signature detected  4 selftest failed

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include "moe-trace-tap.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <set>

using namespace moe_trace;

static volatile std::sig_atomic_t g_signal = 0;
static void on_signal(int) { g_signal = 1; }

// ---------------------------------------------------------------------------------------------

static std::string now_iso() {
    std::time_t t = std::time(nullptr); char b[40]; std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t)); return b;
}
static std::string env_or(const char * k, const char * d) { const char * v = std::getenv(k); return (v && *v) ? v : d; }
static long env_long(const char * k, long d) { const char * v = std::getenv(k); return (v && *v) ? std::atol(v) : d; }
static bool file_exists(const std::string & p) { std::error_code ec; return std::filesystem::exists(p, ec); }

static bool write_text(const std::string & path, const std::string & body) {
    const std::string tmp = path + ".tmp";
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(body.data(), 1, body.size(), f) == body.size() && std::fflush(f) == 0 && std::fclose(f) == 0;
    return ok && std::rename(tmp.c_str(), path.c_str()) == 0;
}

struct driver {
    // configuration
    std::string out_dir, domain, model_path, arch, argv_joined, corpus_sha, tokens_sha;
    long start_window = 0, max_samples = 0, shard_samples = 64, decode_tokens = 0;
    bool srp = false, keep_tail = false;
    int  chunk = 0;
    long deadline = 0;
    std::string stopfile;
    size_t corpus_tokens = 0;

    // state
    tap_state st;
    std::vector<layer_info> layers;
    bool layers_known = false;
    int  max_k = 0;
    run_writer rw;
    eor_tracker eor;
    std::string shard_dir, started;
    long shard_first_window = 0;
    long shard_windows = 0;
    guard_stats gs;
    bool stride_reported = false;
    std::string stop_reason = "corpus-exhausted";
    std::vector<std::pair<long, uint64_t>> shards_done;   // (first window, rows)

    bool stop_requested(std::string & why) const {
        if (g_signal) { why = "signal"; return true; }
        if (!stopfile.empty() && file_exists(stopfile)) { why = "stopfile"; return true; }
        if (deadline > 0 && (long) std::time(nullptr) >= deadline) { why = "deadline"; return true; }
        return false;
    }

    // turn the taps of the forward that just ran into rows. n = rows expected.
    int flush_forward(uint32_t sample_id, uint32_t pos0, int phase, uint32_t step, int n, long window) {
        if (!st.fatal.empty()) { LOG_ERR("moe-trace: %s\n", st.fatal.c_str()); return st.fatal_rc; }
        if (!st.stride_note.empty() && !stride_reported) { LOG_INF("moe-trace: %s\n", st.stride_note.c_str()); stride_reported = true; }
        const bool was_known = layers_known;
        assembled_forward af;
        std::string msg;
        const int arc = assemble_forward(st.cur, n, phase == PHASE_PREFILL, srp, layers, layers_known, af, gs, msg);
        if (arc != RC_DONE) {
            LOG_ERR("moe-trace: window %ld step %u: %s\n", window, step, msg.c_str());
            if (arc == RC_HAZARD) write_failed(msg + " (window " + std::to_string(window) + ")");
            return arc;
        }
        if (!was_known) {
            max_k = max_top_k(layers);
            eor.init(layers);
            std::string map;
            for (size_t i = 0; i < layers.size(); i++) map += (i ? "," : "") + std::to_string(layers[i].il);
            LOG_INF("moe-trace: %zu MoE layers tapped (il = %s); n_experts[0]=%d top_k[0]=%d chance=%.4f\n",
                    layers.size(), map.c_str(), layers[0].n_experts, layers[0].top_k, (double) layers[0].top_k / layers[0].n_experts);
        }
        if (!rw.is_open() && !open_shard(window)) return RC_ERROR;
        std::string err;
        if (!rw.append_forward(sample_id, domain, pos0, phase, step, n, af.ids.data(), srp ? af.order.data() : nullptr, err)) { LOG_ERR("moe-trace: %s\n", err.c_str()); return RC_ERROR; }
        eor.add_forward(phase, af.ids.data(), n);
        st.cur.clear();
        return RC_DONE;
    }

    // leaves an unmistakable marker next to the shards; the in-flight shard is never given a manifest
    void write_failed(const std::string & why) const {
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        write_text(out_dir + "/FAILED", why + "\n");
    }

    bool open_shard(long window) {
        shard_first_window = window; shard_windows = 0;
        char nm[64]; std::snprintf(nm, sizeof nm, "/shard-%06ld", window);
        shard_dir = out_dir + nm;
        run_meta m;
        m.model = std::filesystem::path(model_path).filename().string();
        m.arch = arch; m.domain = domain; m.batch_size = chunk; m.layers = layers; m.srp = srp;
        std::string err;
        std::error_code ec;
        for (const char * f : {"routing_trace.json", "routing_trace.eor.json", "routing_trace.provenance.json"})
            std::filesystem::remove(shard_dir + "/" + f, ec);   // a re-opened shard is never half-valid
        if (!rw.open(shard_dir, m, err)) { LOG_ERR("moe-trace: %s\n", err.c_str()); return false; }
        eor.init(layers);
        started = now_iso();
        return true;
    }

    int close_shard() {
        if (!rw.is_open()) return RC_DONE;
        std::string manifest, err;
        const uint64_t rows = rw.rows();
        if (!rw.close(manifest, err)) { LOG_ERR("moe-trace: cannot close shard %s: %s\n", shard_dir.c_str(), err.c_str()); return RC_ERROR; }
        write_text(shard_dir + "/routing_trace.eor.json", eor.to_json());
        std::string il_map;
        for (size_t i = 0; i < layers.size(); i++) il_map += (i ? "," : "") + std::to_string(layers[i].il);
        std::string p = "{\n  \"schema\": \"epyc.moe.routing_trace.provenance.v1\",\n  \"tool\": \"llama-moe-trace\",\n  \"tool_format\": \"moe-trace-1\",\n";
        p += "  \"llama_build_info\": \"" + json_escape(llama_build_info()) + "\",\n";
        p += "  \"arch\": \"" + json_escape(arch) + "\",\n  \"model_path\": \"" + json_escape(model_path) + "\",\n";
        p += "  \"domain\": \"" + json_escape(domain) + "\",\n  \"batch_size\": " + std::to_string(chunk) + ",\n";
        p += "  \"first_window\": " + std::to_string(shard_first_window) + ",\n  \"n_windows\": " + std::to_string(shard_windows) + ",\n";
        p += "  \"rows\": " + std::to_string(rows) + ",\n  \"decode_tokens_per_window\": " + std::to_string(decode_tokens) + ",\n";
        p += "  \"srp\": " + std::string(srp ? "true" : "false") + ",\n";
        if (srp) p += "  \"argsort_sha256\": \"" + rw.srp_sha() + "\",\n  \"argsort_bytes\": " + std::to_string(rw.srp_bytes()) + ",\n  \"argsort_shape\": \"[rows, n_layers, max n_experts] int16, descending score, -1 pad\",\n";
        p += "  \"corpus_sha256\": \"" + corpus_sha + "\",\n  \"corpus_tokens\": " + std::to_string(corpus_tokens) + ",\n  \"tokens_sha256\": \"" + tokens_sha + "\",\n";
        p += "  \"layer_il\": [" + il_map + "],\n";
        p += "  \"hazard_guard\": {\"checks\": " + std::to_string(gs.checks) + ", \"sub_chance_checks\": " + std::to_string(gs.sub_chance)
           + ", \"cv_min\": " + (gs.checks ? std::to_string(gs.cv_min) : "null") + ", \"cv_max\": " + (gs.checks ? std::to_string(gs.cv_max) : "null") + ", \"fatal\": 0},\n";
        p += "  \"env\": {\"GGML_FUSED_DECODE_OFF\": \"" + env_or("GGML_FUSED_DECODE_OFF", "") + "\", \"GGML_IQK\": \"" + env_or("GGML_IQK", "") + "\"},\n";
        p += "  \"argv\": \"" + json_escape(argv_joined) + "\",\n  \"started\": \"" + started + "\",\n  \"finished\": \"" + now_iso() + "\"\n}\n";
        write_text(shard_dir + "/routing_trace.provenance.json", p);
        LOG_INF("moe-trace: shard %s closed: %llu rows, %zu samples, rows_sha256 %s\n", shard_dir.c_str(), (unsigned long long) rows, rw.n_samples(), manifest.substr(manifest.find("rows_sha256") + 15, 16).c_str());
        shards_done.push_back({shard_first_window, rows});
        shard_dir.clear();
        return RC_DONE;
    }
};

int main(int argc, char ** argv) {
    // RT-3 selftest first: no model needed, microseconds. A reader or guard that does not work must stop the run before any compute.
    const std::string st_err = selftest();
    if (!st_err.empty()) {
        std::fprintf(stderr, "moe-trace: SELFTEST FAILED: %s\n", st_err.c_str());
        return RC_SELFTEST;
    }
    if (env_long("MOE_TRACE_SELFTEST_ONLY", 0) == 1) {
        std::fprintf(stdout, "moe-trace: selftest ok (stride-safe reader, stride-hazard guard, sha256, int16 ids)\n");
        return RC_DONE;
    }

    driver d;
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return RC_ERROR;

    d.out_dir = env_or("MOE_TRACE_OUT", "");
    if (d.out_dir.empty()) { LOG_ERR("MOE_TRACE_OUT (output directory) is required\n"); return RC_ERROR; }
    d.domain        = env_or("MOE_TRACE_DOMAIN", "unspecified");
    d.start_window  = env_long("MOE_TRACE_START_SAMPLE", 0);
    d.max_samples   = env_long("MOE_TRACE_MAX_SAMPLES", 0);
    d.shard_samples = env_long("MOE_TRACE_SHARD_SAMPLES", 64);
    d.decode_tokens = env_long("MOE_TRACE_DECODE_TOKENS", 0);
    d.srp           = env_long("MOE_TRACE_SRP", 0) == 1;
    d.keep_tail     = env_long("MOE_TRACE_KEEP_TAIL", 0) == 1;
    d.deadline      = env_long("MOE_TRACE_DEADLINE", 0);
    d.stopfile      = env_or("MOE_TRACE_STOPFILE", "");
    d.st.srp        = d.srp;
    for (int i = 0; i < argc; i++) { d.argv_joined += (i ? " " : ""); d.argv_joined += argv[i]; }
    if (params.prompt.empty()) { LOG_ERR("no corpus: pass -f corpus.txt\n"); return RC_ERROR; }
    if (d.start_window < 0 || d.max_samples < 0 || d.shard_samples < 0 || d.decode_tokens < 0) { LOG_ERR("negative MOE_TRACE_* value\n"); return RC_ERROR; }

    // the tap needs the graph path and one forward per llama_decode
    setenv("GGML_FUSED_DECODE_OFF", "1", 1);
    if (params.n_ubatch != params.n_batch) {
        LOG_WRN("moe-trace: forcing -ub %d -> -b %d so one llama_decode is exactly one forward pass\n", params.n_ubatch, params.n_batch);
        params.n_ubatch = params.n_batch;
    }
    d.chunk = params.n_batch;
    if (d.chunk < 8 || d.chunk > 65535) { LOG_ERR("-b %d outside 8..65535\n", d.chunk); return RC_ERROR; }
    if (params.n_ctx <= 0) {
        // 0 means "the model's trained context" (262144 for Q38FN): a multi-hundred-GiB KV the tap does not need
        params.n_ctx = d.chunk + (int) d.decode_tokens;
        LOG_WRN("moe-trace: no -c given; using -c %d (= -b + MOE_TRACE_DECODE_TOKENS)\n", params.n_ctx);
    }
    if ((long) params.n_ctx < d.chunk + d.decode_tokens) {
        LOG_ERR("-c %d is smaller than -b %d + MOE_TRACE_DECODE_TOKENS %ld\n", params.n_ctx, d.chunk, d.decode_tokens);
        return RC_ERROR;
    }
    d.model_path = params.model.path;

    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = moe_trace_cb;
    params.cb_eval_user_data = &d.st;
    params.warmup            = false;
    params.fit_params        = false;   // a CPU-only tool with explicit -c / -ngl: nothing to fit, and no extra load pass

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) { LOG_ERR("failed to init model/context\n"); return RC_ERROR; }
    if (llama_n_ctx(ctx) < (uint32_t) (d.chunk + d.decode_tokens)) { LOG_ERR("context %u < b + decode tokens\n", llama_n_ctx(ctx)); return RC_ERROR; }

    char buf[256];
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof buf) > 0) d.arch = buf;
    if (d.arch.empty()) d.arch = "unknown";
    if (d.arch != "unknown") {
        const std::string key = d.arch + ".expert_count";
        if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof buf) > 0) d.st.model_n_expert_fallback = std::atoi(buf);
    }
    LOG_INF("moe-trace: arch=%s chunk=%d decode_tokens=%ld srp=%d shard=%ld start=%ld max=%ld\n",
            d.arch.c_str(), d.chunk, d.decode_tokens, (int) d.srp, d.shard_samples, d.start_window, d.max_samples);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), false);
    d.corpus_tokens = tokens.size();
    d.corpus_sha = sha256::of(params.prompt.data(), params.prompt.size());
    d.tokens_sha = sha256::of(tokens.data(), tokens.size() * sizeof(llama_token));

    const long full_windows = (long) (tokens.size() / (size_t) d.chunk);
    const size_t tail = tokens.size() % (size_t) d.chunk;
    long total_windows = full_windows + ((d.keep_tail && tail >= 8) ? 1 : 0);
    long end_window = total_windows;
    if (d.max_samples > 0) end_window = std::min(end_window, d.start_window + d.max_samples);
    if (d.start_window >= end_window) {
        LOG_INF("moe-trace: nothing to do (start window %ld >= end %ld of %ld)\n", d.start_window, end_window, total_windows);
        return RC_DONE;
    }
    if (end_window < total_windows) d.stop_reason = "max-samples-reached";
    LOG_INF("moe-trace: %zu tokens = %ld full windows of %d (+%zu tail); tracing windows [%ld, %ld)\n",
            tokens.size(), full_windows, d.chunk, tail, d.start_window, end_window);

    // logits for every position: otherwise llama.cpp prunes the last layer's FFN to the final token and truncates its trace
    llama_batch batch = llama_batch_init(d.chunk, 0, 1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const auto t0 = std::chrono::steady_clock::now();
    long done_windows = 0; uint64_t tok_traced = 0;
    int rc = RC_DONE;

    for (long w = d.start_window; w < end_window && rc == RC_DONE; w++) {
        std::string why;
        if (d.stop_requested(why)) { d.stop_reason = why; rc = RC_STOPPED; break; }

        if (d.shard_samples > 0 && d.rw.is_open() && d.shard_windows >= d.shard_samples) {
            rc = d.close_shard();
            if (rc != RC_DONE) break;
        }

        const size_t s0 = (size_t) w * d.chunk;
        const int n = (int) std::min((size_t) d.chunk, tokens.size() - s0);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i] = tokens[s0 + i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        d.eor.reset_sample();
        d.st.cur.clear();
        if (llama_decode(ctx, batch)) { LOG_ERR("moe-trace: llama_decode failed at window %ld\n", w); rc = d.st.fatal.empty() ? RC_ERROR : d.st.fatal_rc; if (!d.st.fatal.empty()) LOG_ERR("moe-trace: %s\n", d.st.fatal.c_str()); break; }

        const uint32_t sample_id = (uint32_t) (d.rw.is_open() ? d.rw.n_samples() : 0);
        rc = d.flush_forward(sample_id, 0, PHASE_PREFILL, 0, n, w);
        if (rc != RC_DONE) break;
        tok_traced += (uint64_t) n;

        // optional greedy decode leg: phase=decode, spec_block=1, one row per step
        if (d.decode_tokens > 0) {
            const float * lg = llama_get_logits_ith(ctx, n - 1);
            for (long j = 0; j < d.decode_tokens; j++) {
                if (!lg) { LOG_ERR("moe-trace: no logits to continue decoding at window %ld\n", w); rc = RC_ERROR; break; }
                llama_token best = 0; float bv = lg[0];
                for (int v = 1; v < n_vocab; v++) if (lg[v] > bv) { bv = lg[v]; best = v; }
                if (llama_vocab_is_eog(vocab, best)) break;
                batch.n_tokens = 1; batch.token[0] = best; batch.pos[0] = n + (int) j;
                batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
                d.st.cur.clear();
                if (llama_decode(ctx, batch)) { LOG_ERR("moe-trace: decode step %ld failed at window %ld\n", j, w); rc = d.st.fatal.empty() ? RC_ERROR : d.st.fatal_rc; if (!d.st.fatal.empty()) LOG_ERR("moe-trace: %s\n", d.st.fatal.c_str()); break; }
                rc = d.flush_forward(sample_id, (uint32_t) (n + j), PHASE_DECODE, (uint32_t) (j + 1), 1, w);
                if (rc != RC_DONE) break;
                tok_traced++;
                lg = llama_get_logits_ith(ctx, 0);
            }
            if (rc != RC_DONE) break;
        }

        d.shard_windows++; done_windows++;
        if (done_windows % 8 == 0 || w + 1 == end_window) {
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            LOG_INF("moe-trace: window %ld/%ld  %llu tokens traced  %.1f tok/s\n", w + 1, end_window, (unsigned long long) tok_traced, el > 0 ? tok_traced / el : 0.0);
        }
    }

    llama_batch_free(batch);
    if (rc == RC_DONE || rc == RC_STOPPED) {
        // a finished or cleanly stopped run still yields a valid, closed shard
        const int rc_close = d.close_shard();
        if (rc_close != RC_DONE) rc = rc_close;
    } else {
        // a failed run: the in-flight shard is left WITHOUT routing_trace.json (removed at open), so no consumer can mistake it for a trace
        if (d.st.fatal.size()) d.write_failed(d.st.fatal);
        else if (!std::filesystem::exists(d.out_dir + "/FAILED")) d.write_failed("llama-moe-trace failed rc=" + std::to_string(rc));
    }
    LOG_INF("moe-trace: exit rc=%d reason=%s shards=%zu windows=%ld\n", rc, rc == RC_DONE ? d.stop_reason.c_str() : rc == RC_STOPPED ? d.stop_reason.c_str() : "failure (see above)", d.shards_done.size(), done_windows);
    llama_backend_free();
    return rc;
}
