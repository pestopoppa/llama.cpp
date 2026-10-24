// chunk-KV reuse, phase P1, KPF-62.6: per-position data for the G3 approximation gate (P0_THRESHOLDS section 4).
//
// This binary only MEASURES. Pooling, percentiles, buckets, the pre-registration check (corpus manifest sha256 against
// P0_THRESHOLDS section 8) and the verdicts are scripts/chunk-kv/g3_gate.py, so the gate logic lives in one place.
//
//   test-chunk-kv-g3 -m MODEL -ctk f32 -ctv f32 [-t N] [-b N -ub N] --g3-corpus FILE.jsonl --g3-out DIR
//                    [--g3-ids FILE] [--g3-modes head,none,all-inject] [--g3-window 64] [--g3-n-cont 64]
//                    [--g3-store-mib 4096] [--g3-max-requests N]
//   test-chunk-kv-g3 -m TINY_MODEL -ctk f32 -ctv f32 --g3-self-test          (synthetic corpus, harness sanity, ctest)
//
// Corpus line (JSONL), one request per line, either form:
//   {"id": "...", "tokens": [ids...], "chunks": [{"offset": p, "n_tokens": n}, ...]}
//   {"id": "...", "segments": [{"text": "...", "chunk": true|false}, ...]}     (tokenized per segment, BOS on the first)
//
// Per request (all arms in one context, own KV stream per sequence, same batch partition in every arm):
//   reference  (seq 0) full recompute: the prompt prefilled segment by segment, then the greedy continuation of
//              N_cont tokens generated one token at a time. Scored rows = the logits at every suffix position (after the
//              last chunk) plus the logits after each of the N_cont continuation tokens (P0_THRESHOLDS amendment of
//              2026-10-21: suffix plus 64 teacher-forced positions of the reference's own greedy continuation).
//   reuse arm  (seq 1, one per mode) chunks from the content-keyed store (a miss = isolated encode at 0..n-1 in seq 2,
//              exported, stored), assembled by server_chunk::assemble (the server's code), then the suffix and the SAME
//              continuation tokens teacher-forced one at a time.
//   modes: head (selector v0: head window W), none (r = 0, the naive baseline), all-inject (r = 100% after inject: must
//          reproduce the reference bit-for-bit, KL exactly 0; a per-request sanity check of the harness itself).
// Outputs in --g3-out: positions.jsonl (id, mode, row, pos, kl, top1, nmse), requests.jsonl, meta.json.
#undef NDEBUG
#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"

#include "server-chunk-recompute.h"
#include "server-chunk-store.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json     = nlohmann::ordered_json;
using tokens_t = std::vector<llama_token>;

struct g3_args {
    std::string corpus;
    std::string out;
    std::string ids;
    std::vector<server_chunk::select_mode> modes = { server_chunk::SELECT_HEAD, server_chunk::SELECT_NONE, server_chunk::SELECT_ALL_INJECT };
    int32_t window       = 64;
    int32_t n_cont       = 64;
    int64_t store_mib    = 4096;
    int32_t max_requests = 0;
    bool    self_test    = false;
};

struct request {
    std::string id;
    tokens_t toks;
    std::vector<server_chunk::span> chunks;
};

struct row_stats {
    double kl   = 0.0;
    int    top1 = 0;
    double nmse = 0.0;
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int argmax(const float * x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[best]) { // strict: ties resolve to the lowest id
            best = i;
        }
    }
    return best;
}

static double logsumexp(const float * x, int n) {
    double m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, (double) x[i]);
    double z = 0.0;
    for (int i = 0; i < n; ++i) z += exp((double) x[i] - m);
    return m + log(z);
}

// KL(ref || x) in nats over the full vocabulary, top-1 agreement, logits NMSE; f64 accumulation
static row_stats compare_row(const float * ref, double lse_ref, const float * x, int n_vocab) {
    row_stats st;
    const double lse_x = logsumexp(x, n_vocab);
    double kl = 0.0, num = 0.0, den = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        const double lpa = (double) ref[v] - lse_ref;
        const double lpb = (double) x[v]   - lse_x;
        kl += exp(lpa) * (lpa - lpb);
        const double d = (double) x[v] - (double) ref[v];
        num += d*d;
        den += (double) ref[v] * (double) ref[v];
    }
    st.kl   = std::max(0.0, kl); // rounding can give -1e-17 for identical rows
    st.top1 = argmax(ref, n_vocab) == argmax(x, n_vocab) ? 1 : 0;
    st.nmse = den > 0.0 ? num/den : num;
    return st;
}

static bool split_args(int argc, char ** argv, g3_args & a, std::vector<char *> & rest) {
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](std::string & dst) {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", k.c_str()); return false; }
            dst = argv[++i];
            return true;
        };
        std::string v;
        if (k == "--g3-corpus")          { if (!need(a.corpus)) return false; }
        else if (k == "--g3-out")        { if (!need(a.out)) return false; }
        else if (k == "--g3-ids")        { if (!need(a.ids)) return false; }
        else if (k == "--g3-window")     { if (!need(v)) return false; a.window = atoi(v.c_str()); }
        else if (k == "--g3-n-cont")     { if (!need(v)) return false; a.n_cont = atoi(v.c_str()); }
        else if (k == "--g3-store-mib")  { if (!need(v)) return false; a.store_mib = atoll(v.c_str()); }
        else if (k == "--g3-max-requests") { if (!need(v)) return false; a.max_requests = atoi(v.c_str()); }
        else if (k == "--g3-self-test")  { a.self_test = true; }
        else if (k == "--g3-modes") {
            if (!need(v)) return false;
            a.modes.clear();
            std::stringstream ss(v);
            std::string m;
            while (std::getline(ss, m, ',')) {
                server_chunk::select_mode sm;
                if (!server_chunk::parse_select_mode(m, sm) || sm == server_chunk::SELECT_ALL) {
                    fprintf(stderr, "bad --g3-modes entry '%s' (head, none, all-inject)\n", m.c_str());
                    return false;
                }
                a.modes.push_back(sm);
            }
        } else {
            rest.push_back(argv[i]);
        }
    }
    return true;
}

static bool parse_request(const json & j, const llama_vocab * vocab, request & r, std::string & why) {
    r.id = j.value("id", std::string());
    if (r.id.empty()) { why = "missing id"; return false; }
    if (j.contains("tokens")) {
        for (const auto & t : j.at("tokens")) {
            r.toks.push_back(t.get<llama_token>());
        }
        for (const auto & c : j.value("chunks", json::array())) {
            r.chunks.push_back({ c.at("offset").get<int32_t>(), c.at("n_tokens").get<int32_t>() });
        }
    } else if (j.contains("segments")) {
        bool first = true;
        for (const auto & sg : j.at("segments")) {
            const std::string text = sg.value("text", std::string());
            const tokens_t t = common_tokenize(vocab, text, first, false);
            first = false;
            if (sg.value("chunk", false) && !t.empty()) {
                r.chunks.push_back({ (int32_t) r.toks.size(), (int32_t) t.size() });
            }
            r.toks.insert(r.toks.end(), t.begin(), t.end());
        }
    } else {
        why = "neither tokens nor segments";
        return false;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (auto t : r.toks) {
        if (t < 0 || t >= n_vocab) { why = "token id out of range"; return false; }
    }
    if (r.chunks.empty()) { why = "no chunks"; return false; }
    // at least one suffix token after the last chunk (it carries the first scored row)
    return server_chunk::validate_spans(r.chunks, 0, (int32_t) r.toks.size() - 1, why);
}

static std::vector<request> synthetic_corpus(int n_vocab) {
    std::mt19937 rng(62);
    auto rnd = [&](int n) { tokens_t t(n); for (auto & x : t) x = (llama_token) (rng() % (uint32_t) n_vocab); return t; };
    // a small chunk library; each chunk appears in several requests at different offsets and after different predecessors
    std::vector<tokens_t> lib = { rnd(48), rnd(80), rnd(33), rnd(64) };
    const std::vector<std::vector<int>> plans = { {0, 1}, {1, 0, 2}, {2, 3}, {3, 1, 0} };
    std::vector<request> out;
    for (size_t i = 0; i < plans.size(); ++i) {
        request r;
        r.id = "selftest-" + std::to_string(i);
        const tokens_t pre = rnd(10 + 7*(int) i);
        r.toks.insert(r.toks.end(), pre.begin(), pre.end());
        for (int c : plans[i]) {
            const tokens_t glue = rnd(5 + (int) i);
            r.toks.insert(r.toks.end(), glue.begin(), glue.end());
            r.chunks.push_back({ (int32_t) r.toks.size(), (int32_t) lib[c].size() });
            r.toks.insert(r.toks.end(), lib[c].begin(), lib[c].end());
        }
        const tokens_t suf = rnd(12);
        r.toks.insert(r.toks.end(), suf.begin(), suf.end());
        out.push_back(std::move(r));
    }
    return out;
}

// prefill [0, E) segment by segment (the partition every arm shares), then the suffix with logits
static bool ref_prefill(llama_context * ctx, const request & r, llama_seq_id seq, int32_t n_batch, int n_vocab, std::vector<float> & rows) {
    const int32_t E = r.chunks.back().offset + r.chunks.back().n_tokens;
    for (const auto & sg : server_chunk::make_segments(r.chunks, 0, E)) {
        if (server_chunk::decode_range(ctx, r.toks.data(), sg.p0, sg.p1, seq, n_batch, false) != 0) {
            return false;
        }
    }
    return server_chunk::decode_range(ctx, r.toks.data(), E, (int32_t) r.toks.size(), seq, n_batch, false, &rows, n_vocab) == 0;
}

// feed cont[k] at pos N + k one token at a time, appending the logits rows; when generate, cont is produced greedily
static bool continuation(llama_context * ctx, tokens_t & cont, int32_t n_cont, int32_t N, llama_seq_id seq, int n_vocab,
        std::vector<float> & rows, bool generate) {
    if (generate) {
        cont.clear();
        cont.push_back(argmax(rows.data() + rows.size() - n_vocab, n_vocab));
    }
    llama_batch one = llama_batch_init(1, 0, 1);
    for (int32_t k = 0; k < n_cont; ++k) {
        one.n_tokens     = 1;
        one.token[0]     = cont[k];
        one.pos[0]       = N + k;
        one.n_seq_id[0]  = 1;
        one.seq_id[0][0] = seq;
        one.logits[0]    = 1;
        const float * row = llama_decode(ctx, one) == 0 ? llama_get_logits_ith(ctx, 0) : nullptr;
        if (row == nullptr) {
            llama_batch_free(one);
            return false;
        }
        rows.insert(rows.end(), row, row + n_vocab);
        if (generate && k + 1 < n_cont) {
            cont.push_back(argmax(rows.data() + rows.size() - n_vocab, n_vocab));
        }
    }
    llama_batch_free(one);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    g3_args a;
    std::vector<char *> rest;
    if (!split_args(argc, argv, a, rest)) {
        return 2;
    }
    if (!a.self_test && (a.corpus.empty() || a.out.empty())) {
        fprintf(stderr, "need --g3-corpus FILE and --g3-out DIR (or --g3-self-test)\n");
        return 2;
    }
    if (a.self_test) {
        a.n_cont = std::min(a.n_cont, 8);
        a.window = std::min(a.window, 16);
        a.modes  = { server_chunk::SELECT_ALL_INJECT, server_chunk::SELECT_HEAD, server_chunk::SELECT_NONE };
    }

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 2;
    }
    ggml_backend_load_all();

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model %s\n", params.model.path.c_str());
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // corpus
    std::vector<request> reqs;
    if (a.self_test) {
        reqs = synthetic_corpus(n_vocab);
    } else {
        std::set<std::string> want;
        if (!a.ids.empty()) {
            std::ifstream fi(a.ids);
            std::string line;
            while (std::getline(fi, line)) {
                if (!line.empty()) want.insert(line);
            }
        }
        std::ifstream f(a.corpus);
        if (!f) {
            fprintf(stderr, "cannot read corpus %s\n", a.corpus.c_str());
            return 2;
        }
        std::string line;
        int n_line = 0;
        while (std::getline(f, line)) {
            ++n_line;
            if (line.empty()) continue;
            request r;
            std::string why;
            json j;
            try {
                j = json::parse(line);
            } catch (const std::exception & e) {
                fprintf(stderr, "corpus line %d: %s\n", n_line, e.what());
                return 2;
            }
            try {
                if (!parse_request(j, vocab, r, why)) {
                    fprintf(stderr, "corpus line %d: %s\n", n_line, why.c_str());
                    return 2;
                }
            } catch (const std::exception & e) {
                fprintf(stderr, "corpus line %d: %s\n", n_line, e.what());
                return 2;
            }
            if (!want.empty() && !want.count(r.id)) continue;
            reqs.push_back(std::move(r));
            if (a.max_requests > 0 && (int) reqs.size() >= a.max_requests) break;
        }
    }
    if (reqs.empty()) {
        fprintf(stderr, "no requests selected\n");
        return 2;
    }

    size_t max_len = 0;
    for (const auto & r : reqs) max_len = std::max(max_len, r.toks.size());
    uint32_t n_ctx_seq = 256;
    while (n_ctx_seq < max_len + (size_t) a.n_cont + 8) n_ctx_seq *= 2;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = 3;
    cparams.n_ctx      = 3*n_ctx_seq;
    cparams.kv_unified = false;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create the context (n_ctx %u)\n", 3*n_ctx_seq);
        return 2;
    }
    {
        char why[256] = {0};
        if (!llama_chunk_kv_supported(ctx, why, sizeof(why))) {
            fprintf(stderr, "REFUSED: chunk-KV not supported on this model: %s\n", why);
            return 4;
        }
    }
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    auto * mem = llama_get_memory(ctx);
    const llama_seq_id s_ref = 0, s_arm = 1, s_scr = 2;

    // the store key must cover everything that changes a blob's bytes (server-chunk-store.h)
    const std::string fp = params.model.path + "|" + llama_build_info() + "|k=" + ggml_type_name(cparams.type_k) +
        "|v=" + ggml_type_name(cparams.type_v) + "|fa=" + std::to_string((int) cparams.flash_attn_type) + "|CKV1";
    server_chunk::store store((size_t) std::max<int64_t>(1, a.store_mib) * 1024 * 1024);

    FILE * f_pos = nullptr;
    FILE * f_req = nullptr;
    if (!a.out.empty()) {
        f_pos = fopen((a.out + "/positions.jsonl").c_str(), "w");
        f_req = fopen((a.out + "/requests.jsonl").c_str(), "w");
        if (!f_pos || !f_req) {
            fprintf(stderr, "cannot write into %s (create it first)\n", a.out.c_str());
            return 2;
        }
    }

    int n_fail = 0;
    double sanity_kl_max = 0.0;
    int    sanity_top1_miss = 0;
    double sanity_nmse_max = 0.0;
    double none_kl_sum = 0.0, head_kl_sum = 0.0;
    size_t none_rows = 0, head_rows = 0;

    for (size_t ri = 0; ri < reqs.size(); ++ri) {
        const request & r = reqs[ri];
        const int32_t N = (int32_t) r.toks.size();
        const int32_t E = r.chunks.back().offset + r.chunks.back().n_tokens;
        int32_t n_chunk_tokens = 0;
        for (const auto & c : r.chunks) n_chunk_tokens += c.n_tokens;

        // reference
        llama_memory_seq_rm(mem, s_ref, -1, -1);
        const double t0 = now_ms();
        std::vector<float> ref_rows;
        tokens_t cont;
        bool ok = ref_prefill(ctx, r, s_ref, n_batch, n_vocab, ref_rows) && continuation(ctx, cont, a.n_cont, N, s_ref, n_vocab, ref_rows, true);
        const double t_ref = now_ms() - t0;
        if (!ok) {
            fprintf(stderr, "request %s: reference decode failed\n", r.id.c_str());
            return 3;
        }
        llama_memory_seq_rm(mem, s_ref, -1, -1);
        const size_t n_rows = ref_rows.size() / n_vocab;
        std::vector<double> lse(n_rows);
        for (size_t k = 0; k < n_rows; ++k) lse[k] = logsumexp(ref_rows.data() + k*n_vocab, n_vocab);

        for (auto mode : a.modes) {
            llama_memory_seq_rm(mem, s_arm, -1, -1);
            const double t1 = now_ms();
            uint64_t hits0 = store.get_stats().hits, miss0 = store.get_stats().misses;
            std::vector<std::vector<uint8_t>> local(r.chunks.size()); // when the store cannot hold a blob

            auto get_blob = [&](int32_t i) -> server_chunk::blob_ref {
                const auto & sp = r.chunks[i];
                const server_chunk::tokens ct(r.toks.begin() + sp.offset, r.toks.begin() + sp.offset + sp.n_tokens);
                if (auto * e = store.get(fp, ct)) {
                    return { e->blob.data(), e->blob.size() };
                }
                // miss: isolated encode at 0..n-1 in the scratch sequence
                llama_memory_seq_rm(mem, s_scr, -1, -1);
                if (server_chunk::decode_range(ctx, ct.data(), 0, sp.n_tokens, s_scr, n_batch, false) != 0) {
                    return {};
                }
                const size_t need = llama_chunk_kv_export(ctx, nullptr, 0, s_scr, 0, sp.n_tokens);
                std::vector<uint8_t> blob(need);
                const bool exp_ok = need > 0 && llama_chunk_kv_export(ctx, blob.data(), blob.size(), s_scr, 0, sp.n_tokens) == need;
                llama_memory_seq_rm(mem, s_scr, -1, -1);
                if (!exp_ok) {
                    return {};
                }
                local[i] = blob;
                store.put(fp, ct, std::move(blob)); // a refused put (budget) only costs the reuse, never correctness
                return { local[i].data(), local[i].size() };
            };

            server_chunk::assemble_stats st;
            int32_t p_done = -1;
            const int32_t ret = server_chunk::assemble(ctx, r.toks.data(), r.chunks, 0, E, s_arm, n_batch, mode, a.window, get_blob, st, &p_done);
            if (ret != 0 || p_done != E || st.n_no_blob != 0) {
                fprintf(stderr, "request %s mode %s: assembly failed (ret %d, done %d of %d, no-blob %d)\n", r.id.c_str(),
                        server_chunk::select_mode_name(mode), ret, p_done, E, st.n_no_blob);
                return 3;
            }
            const double t_asm = now_ms() - t1;
            std::vector<float> rows;
            ok = server_chunk::decode_range(ctx, r.toks.data(), E, N, s_arm, n_batch, false, &rows, n_vocab) == 0 &&
                 continuation(ctx, cont, a.n_cont, N, s_arm, n_vocab, rows, false);
            const double t_arm = now_ms() - t1;
            if (!ok || rows.size() != ref_rows.size()) {
                fprintf(stderr, "request %s mode %s: decode failed\n", r.id.c_str(), server_chunk::select_mode_name(mode));
                return 3;
            }
            llama_memory_seq_rm(mem, s_arm, -1, -1);

            double kl_sum = 0.0;
            int top1_sum = 0;
            for (size_t k = 0; k < n_rows; ++k) {
                const row_stats rs = compare_row(ref_rows.data() + k*n_vocab, lse[k], rows.data() + k*n_vocab, n_vocab);
                kl_sum += rs.kl;
                top1_sum += rs.top1;
                const int32_t pos = k < (size_t) (N - E) ? E + (int32_t) k : N + (int32_t) (k - (N - E));
                if (f_pos) {
                    fprintf(f_pos, "{\"id\":%s,\"mode\":\"%s\",\"row\":%zu,\"pos\":%d,\"kl\":%.9e,\"top1\":%d,\"nmse\":%.9e}\n",
                            json(r.id).dump().c_str(), server_chunk::select_mode_name(mode), k, pos, rs.kl, rs.top1, rs.nmse);
                }
                if (mode == server_chunk::SELECT_ALL_INJECT) {
                    sanity_kl_max   = std::max(sanity_kl_max, rs.kl);
                    sanity_nmse_max = std::max(sanity_nmse_max, rs.nmse);
                    sanity_top1_miss += 1 - rs.top1;
                } else if (mode == server_chunk::SELECT_NONE) {
                    none_kl_sum += rs.kl; none_rows++;
                } else if (mode == server_chunk::SELECT_HEAD) {
                    head_kl_sum += rs.kl; head_rows++;
                }
            }
            const bool exact = mode == server_chunk::SELECT_ALL_INJECT && rows.size() == ref_rows.size() &&
                memcmp(rows.data(), ref_rows.data(), rows.size()*sizeof(float)) == 0;

            json jr = {
                {"id", r.id}, {"mode", server_chunk::select_mode_name(mode)}, {"window", mode == server_chunk::SELECT_HEAD ? a.window : 0},
                {"n_tokens", N}, {"n_chunks", (int) r.chunks.size()}, {"n_chunk_tokens", n_chunk_tokens},
                {"n_suffix", N - E}, {"n_cont", a.n_cont}, {"n_rows", (int) n_rows},
                {"n_injected", st.n_injected}, {"n_recomputed", st.n_recomputed},
                {"r", n_chunk_tokens > 0 ? (double) st.n_recomputed / n_chunk_tokens : 0.0},
                {"mean_kl", kl_sum / n_rows}, {"top1", (double) top1_sum / n_rows},
                {"store_hits", store.get_stats().hits - hits0}, {"store_misses", store.get_stats().misses - miss0},
                {"t_ref_ms", t_ref}, {"t_assemble_ms", t_asm}, {"t_arm_ms", t_arm},
            };
            if (mode == server_chunk::SELECT_ALL_INJECT) {
                jr["bit_identical_to_ref"] = exact;
                if (!exact) {
                    fprintf(stderr, "SANITY FAIL request %s: all-inject rows differ from the reference\n", r.id.c_str());
                    n_fail++;
                }
            }
            if (f_req) {
                fprintf(f_req, "%s\n", jr.dump().c_str());
                fflush(f_req);
            }
            printf("[%zu/%zu] %s %-10s r=%.3f mean KL=%.4e top1=%.4f hits=%llu (%.0f ms)\n", ri + 1, reqs.size(), r.id.c_str(),
                    server_chunk::select_mode_name(mode), jr["r"].get<double>(), kl_sum / n_rows, (double) top1_sum / n_rows,
                    (unsigned long long) (store.get_stats().hits - hits0), t_arm);
        }
        if (f_pos) fflush(f_pos);
    }

    if (!a.out.empty()) {
        json meta = {
            {"tool", "test-chunk-kv-g3"}, {"build_info", llama_build_info()}, {"model", params.model.path},
            {"n_vocab", n_vocab}, {"n_requests", (int) reqs.size()}, {"corpus", a.corpus}, {"ids", a.ids},
            {"window", a.window}, {"n_cont", a.n_cont}, {"n_batch", n_batch}, {"n_ubatch", (int) llama_n_ubatch(ctx)},
            {"n_threads", params.cpuparams.n_threads}, {"type_k", ggml_type_name(cparams.type_k)}, {"type_v", ggml_type_name(cparams.type_v)},
            {"selector", "head-window-v0 (KPF-62.3 selector not landed)"},
            {"scored_rows", "logits at every suffix position after the last chunk, plus after each of n_cont teacher-forced tokens of the reference greedy continuation"},
            {"sanity_all_inject_bit_identical", n_fail == 0},
        };
        json modes = json::array();
        for (auto m : a.modes) modes.push_back(server_chunk::select_mode_name(m));
        meta["modes"] = modes;
        std::ofstream(a.out + "/meta.json") << meta.dump(2) << "\n";
    }
    if (f_pos) fclose(f_pos);
    if (f_req) fclose(f_req);

    printf("sanity (all-inject vs reference): max KL=%.3e max NMSE=%.3e top1 misses=%d -> %s\n", sanity_kl_max, sanity_nmse_max,
            sanity_top1_miss, n_fail == 0 ? "bit-identical" : "DIFFER");
    if (none_rows && head_rows) {
        printf("pooled mean KL: head W=%d %.4e | none %.4e | naive/selector ratio %.2f\n", a.window, head_kl_sum / head_rows,
                none_kl_sum / none_rows, head_kl_sum > 0 ? (none_kl_sum / none_rows) / (head_kl_sum / head_rows) : INFINITY);
    }
    if (a.self_test) {
        // the library chunks recur across requests: the content-keyed store must have served hits
        if (store.get_stats().hits == 0) {
            fprintf(stderr, "SELF-TEST FAIL: no store hits across requests\n");
            n_fail++;
        }
    }

    llama_free(ctx);
    llama_model_free(model);

    if (n_fail) {
        fprintf(stderr, "%d failure(s)\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
