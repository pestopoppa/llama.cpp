#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Actual NextN/MTP context tests for GLM5Next.  This intentionally uses the
// staging llama-ext API used by common/speculative.cpp; it does not synthesize
// k-pool membership or infer it from token count.

struct ctx_deleter { void operator()(llama_context * p) const { llama_free(p); } };
using ctx_ptr = std::unique_ptr<llama_context, ctx_deleter>;

struct mtp_pair {
    ctx_ptr tgt;
    ctx_ptr dft;
    std::vector<float> pending_h;
};

struct eval_result {
    std::vector<float> logits;
    std::vector<int32_t> selection;
    size_t selection_width = 0;
};

static int env_int(const char * name, int fallback, int minimum) {
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char * end = nullptr; errno = 0; long value = std::strtol(raw, &end, 10);
    if (errno || !end || *end || value < minimum || value > 10000000) {
        std::fprintf(stderr, "invalid %s=%s\n", name, raw); std::exit(2);
    }
    return (int) value;
}

static double env_real(const char * name, double fallback) {
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char * end = nullptr; errno = 0; double value = std::strtod(raw, &end);
    if (errno || !end || *end || !std::isfinite(value) || value < 0) {
        std::fprintf(stderr, "invalid %s=%s\n", name, raw); std::exit(2);
    }
    return value;
}

static std::vector<llama_token> deterministic_tokens(int n_vocab, int n, int salt = 0) {
    if (n_vocab <= 0) {
        throw std::runtime_error("model has no vocabulary entries");
    }
    std::vector<llama_token> out((size_t) n);
    for (int i = 0; i < n; ++i) {
        out[(size_t) i] = n_vocab > 1 ? 1 + (i + salt) % (n_vocab - 1) : 0;
    }
    return out;
}

static mtp_pair make_pair(const common_params & params, llama_model * model,
                          uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
                          int32_t n_rs_seq = -1) {
    auto cp_tgt = common_context_params_to_llama(params);
    cp_tgt.n_ctx = n_ctx;
    cp_tgt.n_batch = n_batch;
    cp_tgt.n_ubatch = n_ubatch;
    cp_tgt.n_seq_max = 1;
    cp_tgt.n_outputs_max = n_batch;
    if (n_rs_seq >= 0) {
        cp_tgt.n_rs_seq = (uint32_t) n_rs_seq;
    }
    cp_tgt.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    cp_tgt.ctx_other = nullptr;
    mtp_pair result;
    result.tgt.reset(llama_init_from_model(model, cp_tgt));
    if (!result.tgt) throw std::runtime_error("cannot create target context");
    llama_set_embeddings_nextn(result.tgt.get(), true, false);

    auto cp_dft = cp_tgt;
    cp_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp_dft.ctx_other = result.tgt.get();
    result.dft.reset(llama_init_from_model(model, cp_dft));
    if (!result.dft) throw std::runtime_error("cannot create MTP context");
    llama_set_embeddings_nextn(result.dft.get(), true, true);
    if (!llama_set_mtp_dsa_index_share(result.dft.get(), true)) {
        throw std::runtime_error("model/context refused MTP DSA index sharing");
    }
    result.pending_h.assign((size_t) llama_model_n_embd_out(model), 0.0f);
    return result;
}

static llama_batch make_batch(const std::vector<llama_token> & tokens,
                              int begin, int count, int n_embd,
                              std::vector<llama_token> * token_storage) {
    llama_batch batch = llama_batch_init(count, n_embd, 1);
    if (n_embd > 0) {
        token_storage->assign(tokens.begin() + begin, tokens.begin() + begin + count);
        batch.token = token_storage->data();
    }
    for (int j = 0; j < count; ++j) {
        common_batch_add(batch, tokens[(size_t) begin + j], begin + j, {0}, j + 1 == count);
    }
    return batch;
}

static void free_mtp_batch(llama_batch & batch) {
    // llama_batch_init(n, n_embd, ...) did not allocate token storage.
    batch.token = nullptr;
    llama_batch_free(batch);
}

static eval_result process_tokens(mtp_pair & pair, llama_model * model,
                                  const std::vector<llama_token> & tokens,
                                  int begin, int count, bool expect_selection = true) {
    llama_batch tgt = make_batch(tokens, begin, count, 0, nullptr);
    if (llama_decode(pair.tgt.get(), tgt) != 0) {
        llama_batch_free(tgt); throw std::runtime_error("target decode failed");
    }

    const int n_embd = llama_model_n_embd_out(model);
    std::vector<llama_token> token_storage;
    llama_batch dft = make_batch(tokens, begin, count, n_embd, &token_storage);
    for (int j = 0; j < count; ++j) {
        const float * src = j == 0 ? pair.pending_h.data()
                                  : llama_get_embeddings_nextn_ith(pair.tgt.get(), j - 1);
        if (!src) {
            free_mtp_batch(dft); llama_batch_free(tgt);
            throw std::runtime_error("target did not expose NextN embeddings");
        }
        std::memcpy(dft.embd + (size_t) j*n_embd, src, (size_t) n_embd*sizeof(float));
    }
    const float * last_h = llama_get_embeddings_nextn_ith(pair.tgt.get(), count - 1);
    if (!last_h) {
        free_mtp_batch(dft); llama_batch_free(tgt);
        throw std::runtime_error("target did not expose final NextN embedding");
    }
    pair.pending_h.assign(last_h, last_h + n_embd);

    if (llama_decode(pair.dft.get(), dft) != 0) {
        free_mtp_batch(dft); llama_batch_free(tgt);
        throw std::runtime_error("MTP decode failed");
    }

    eval_result result;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(pair.dft.get(), -1);
    if (!logits) {
        free_mtp_batch(dft); llama_batch_free(tgt);
        throw std::runtime_error("MTP context did not expose logits");
    }
    result.logits.assign(logits, logits + n_vocab);

    size_t n_selection = 0;
    const int32_t * selection = llama_get_mtp_dsa_selection(pair.dft.get(), &n_selection);
    if (!expect_selection) {
        if (selection != nullptr || n_selection != 0) {
            free_mtp_batch(dft); llama_batch_free(tgt);
            throw std::runtime_error("multi-ubatch MTP decode exposed a DSA selection");
        }
        free_mtp_batch(dft);
        llama_batch_free(tgt);
        return result;
    }
    if (!selection || n_selection == 0 || n_selection % (size_t) count != 0) {
        free_mtp_batch(dft); llama_batch_free(tgt);
        throw std::runtime_error("MTP graph did not expose a rectangular DSA selection");
    }
    result.selection_width = n_selection/(size_t) count;
    const int32_t * last = selection + (size_t) (count - 1)*result.selection_width;
    result.selection.assign(last, last + result.selection_width);

    free_mtp_batch(dft);
    llama_batch_free(tgt);
    return result;
}

static eval_result decode_mtp_one(llama_context * dft, llama_model * model,
                                  llama_token token, llama_pos pos,
                                  const std::vector<float> & h) {
    const int n_embd = llama_model_n_embd_out(model);
    llama_batch batch = llama_batch_init(1, n_embd, 1);
    std::vector<llama_token> storage { token };
    batch.token = storage.data();
    common_batch_add(batch, token, pos, {0}, true);
    std::memcpy(batch.embd, h.data(), (size_t) n_embd*sizeof(float));
    if (llama_decode(dft, batch) != 0) {
        free_mtp_batch(batch); throw std::runtime_error("MTP continuation decode failed");
    }
    eval_result result;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(dft, -1);
    if (!logits) { free_mtp_batch(batch); throw std::runtime_error("missing MTP continuation logits"); }
    result.logits.assign(logits, logits + n_vocab);
    size_t n = 0;
    const int32_t * sel = llama_get_mtp_dsa_selection(dft, &n);
    if (!sel || n == 0) { free_mtp_batch(batch); throw std::runtime_error("missing continuation selection"); }
    result.selection_width = n;
    result.selection.assign(sel, sel + n);
    free_mtp_batch(batch);
    return result;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return std::numeric_limits<double>::infinity();
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<double>::infinity();
        result = std::max(result, std::fabs((double) a[i] - b[i]));
    }
    return result;
}

static llama_token argmax(const std::vector<float> & x) {
    return (llama_token) (std::max_element(x.begin(), x.end()) - x.begin());
}

static void write_ints(std::ostream & out, const std::vector<int32_t> & xs) {
    out << '[';
    for (size_t i = 0; i < xs.size(); ++i) { if (i) out << ','; out << xs[i]; }
    out << ']';
}

static int run_restore(const common_params & params, llama_model * model,
                       int n_vocab, double tolerance, std::ostream & out) {
    const int prefix = env_int("GLM53_TEST_PREFIX", 12, 4);
    const int n_ctx = env_int("GLM53_TEST_N_CTX", 128, prefix + 2);
    auto ref = make_pair(params, model, n_ctx, n_ctx, n_ctx);
    auto replay = make_pair(params, model, n_ctx, n_ctx, n_ctx);
    auto used = make_pair(params, model, n_ctx, n_ctx, n_ctx);
    const auto tokens = deterministic_tokens(n_vocab, prefix + 1);
    // The dirty destination is deliberately longer so its staged selection
    // contains positions which are invalid after restoring the shorter prefix.
    const auto noise = deterministic_tokens(n_vocab, prefix + 3, 17);
    const auto ref_pre = process_tokens(ref, model, tokens, 0, prefix);
    const auto replay_pre = process_tokens(replay, model, tokens, 0, prefix);
    const auto used_pre = process_tokens(used, model, noise, 0, prefix + 3);
    if (ref_pre.selection.empty() || replay_pre.selection.empty() || used_pre.selection.empty()) {
        throw std::runtime_error("precondition: MTP selection was not populated");
    }
    if (used_pre.selection == ref_pre.selection) {
        throw std::runtime_error("precondition: dirty MTP selection unexpectedly equals source selection");
    }

    // Stage the actual dirty graph output as the reusable selection.  The full
    // restore below must invalidate it; otherwise the next graph can reuse
    // positions belonging only to the longer dirty prefix.
    if (!llama_set_mtp_dsa_selection(used.dft.get(), used_pre.selection.data(), used_pre.selection.size())) {
        throw std::runtime_error("cannot stage dirty MTP DSA selection");
    }
    llama_set_mtp_dsa_selection(ref.dft.get(), nullptr, 0);
    llama_set_mtp_dsa_selection(replay.dft.get(), nullptr, 0);

    common_prompt_checkpoint ckpt;
    ckpt.update_dft(ref.dft.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    ckpt.load_dft(used.dft.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    // pending_h belongs to common speculative state rather than llama_context;
    // use the same explicit NextN input to isolate context restore correctness.
    used.pending_h = ref.pending_h;

    const auto rr = decode_mtp_one(ref.dft.get(), model, tokens[(size_t) prefix], prefix, ref.pending_h);
    const auto rp = decode_mtp_one(replay.dft.get(), model, tokens[(size_t) prefix], prefix, replay.pending_h);
    const auto ru = decode_mtp_one(used.dft.get(), model, tokens[(size_t) prefix], prefix, used.pending_h);
    const double source_diff = max_abs_diff(rr.logits, rp.logits);
    const double used_diff = max_abs_diff(ru.logits, rp.logits);
    if (!std::isfinite(source_diff) || !std::isfinite(used_diff)) {
        throw std::runtime_error("nonfinite MTP continuation logits");
    }
    const bool pass = source_diff <= tolerance && used_diff <= tolerance &&
                      argmax(rr.logits) == argmax(rp.logits) && argmax(ru.logits) == argmax(rp.logits) &&
                      rr.selection == rp.selection && ru.selection == rp.selection;
    out << "{\"schema\":\"epyc.glm53.used_mtp_restore_case.v1\",\"prefix_tokens\":" << prefix
        << ",\"mtp_selection_populated\":true,\"dirty_mtp_selection_populated\":true"
        << ",\"restore_flags\":0,\"continuation_uses_actual_nextn_graph\":true"
        << ",\"source_replay_max_abs_logit_diff\":" << source_diff
        << ",\"used_replay_max_abs_logit_diff\":" << used_diff
        << ",\"declared_logit_tolerance\":" << tolerance
        << ",\"source_argmax\":" << argmax(rr.logits)
        << ",\"replay_argmax\":" << argmax(rp.logits)
        << ",\"used_argmax\":" << argmax(ru.logits)
        << ",\"selection_width\":" << rp.selection_width
        << ",\"selection_equal\":" << (rr.selection == rp.selection ? "true" : "false")
        << ",\"used_selection_equal\":" << (ru.selection == rp.selection ? "true" : "false")
        << ",\"verdict\":\"" << (pass ? "PASS" : "FAIL") << "\"}\n";
    return pass ? 0 : 3;
}

static eval_result run_pool_variant(const common_params & params, llama_model * model,
                                    const std::vector<llama_token> & tokens,
                                    uint32_t ubatch) {
    const uint32_t n = (uint32_t) tokens.size();
    // This gate tests ordinary prefill chunking. Recurrent speculative
    // rollback tails impose a separate minimum ubatch and are covered by the
    // restore test, so disable them here before exercising tiny chunks.
    auto pair = make_pair(params, model, n + 2, n, ubatch, 0);
    eval_result result;
    for (uint32_t begin = 0; begin < n; begin += ubatch) {
        result = process_tokens(pair, model, tokens, (int) begin, (int) std::min(ubatch, n - begin));
    }
    return result;
}

static int run_export(const common_params & params, llama_model * model,
                      int n_vocab, double tolerance, std::ostream & out) {
    const uint32_t kpool = model->hparams.indexer_kpool;
    const uint32_t topk = model->hparams.indexer_top_k;
    if (!model->hparams.indexer_kpool_select_tail || topk <= 64*kpool) {
        std::fprintf(stderr, "REFUSE: export test requires topk > 64*kpool and tail selection\n");
        return 2;
    }

    const uint32_t n = 64*kpool + 1;
    const uint32_t ubatch = std::max<uint32_t>(kpool, n/2);
    const auto tokens = deterministic_tokens(n_vocab, (int) n + 1);
    auto split = make_pair(params, model, n + 4, n + 1, ubatch, 0);
    auto full = make_pair(params, model, n + 4, n + 1, n + 1, 0);

    const auto prefill_split = process_tokens(split, model, tokens, 0, (int) n, false);
    const auto prefill_full = process_tokens(full, model, tokens, 0, (int) n);
    const auto next_split = decode_mtp_one(split.dft.get(), model, tokens[n], (llama_pos) n, split.pending_h);
    const auto next_full = decode_mtp_one(full.dft.get(), model, tokens[n], (llama_pos) n, full.pending_h);
    const double prefill_diff = max_abs_diff(prefill_split.logits, prefill_full.logits);
    const double next_diff = max_abs_diff(next_split.logits, next_full.logits);
    const bool pass = prefill_diff <= tolerance && next_diff <= tolerance &&
                      argmax(next_split.logits) == argmax(next_full.logits) &&
                      next_split.selection == next_full.selection &&
                      next_split.selection_width > 0;
    out << "{\"schema\":\"epyc.glm53.mtp_dsa_export.v1\",\"prefill_tokens\":" << n
        << ",\"n_ubatch\":" << ubatch
        << ",\"prefill_selection_null\":true"
        << ",\"next_selection_width\":" << next_split.selection_width
        << ",\"next_selection_equal\":" << (next_split.selection == next_full.selection ? "true" : "false")
        << ",\"prefill_max_abs_logit_diff\":" << prefill_diff
        << ",\"next_max_abs_logit_diff\":" << next_diff
        << ",\"declared_logit_tolerance\":" << tolerance
        << ",\"verdict\":\"" << (pass ? "PASS" : "FAIL") << "\"}\n";
    return pass ? 0 : 3;
}

static int run_pool(const common_params & params, llama_model * model,
                    int n_vocab, double tolerance, std::ostream & out) {
    const int kpool = env_int("GLM53_TEST_KPOOL", 4, 2);
    const int topk = env_int("GLM53_TEST_TOPK", 8, kpool);
    const int chunk = env_int("GLM53_TEST_CHUNK", topk == 8 ? 3 : 257, 1);
    if ((int) model->hparams.indexer_kpool != kpool || (int) model->hparams.indexer_top_k != topk) {
        std::fprintf(stderr, "REFUSE: model kpool/topk=%u/%u, requested=%d/%d\n",
                     model->hparams.indexer_kpool, model->hparams.indexer_top_k, kpool, topk);
        return 2;
    }
    if (!model->hparams.indexer_kpool_select_tail || topk % kpool) {
        std::fprintf(stderr, "REFUSE: incompatible k-pool metadata\n"); return 2;
    }
    const std::vector<int> lengths {topk - 1, topk, topk + 1,
                                    topk + kpool - 1, topk + kpool, topk + kpool + 1};
    bool all_pass = true;
    for (int n : lengths) {
        const auto tokens = deterministic_tokens(n_vocab, n);
        const auto full = run_pool_variant(params, model, tokens, (uint32_t) n);
        const auto chunked = run_pool_variant(params, model, tokens, (uint32_t) std::min(n, chunk));
        if (full.selection_width != (size_t) topk + kpool - 1 ||
                chunked.selection_width != full.selection_width) {
            throw std::runtime_error("unexpected actual selection width");
        }
        auto split = [topk](const std::vector<int32_t> & selection, bool tail) {
            const size_t begin = tail ? (size_t) topk : 0;
            const size_t end = tail ? selection.size() : (size_t) topk;
            std::vector<int32_t> result;
            for (size_t i = begin; i < end; ++i) if (selection[i] >= 0) result.push_back(selection[i]);
            return result;
        };
        const auto fm = split(full.selection, false), ft = split(full.selection, true);
        const auto cm = split(chunked.selection, false), ct = split(chunked.selection, true);
        auto fm_canon = fm, ft_canon = ft, cm_canon = cm, ct_canon = ct;
        std::sort(fm_canon.begin(), fm_canon.end());
        std::sort(ft_canon.begin(), ft_canon.end());
        std::sort(cm_canon.begin(), cm_canon.end());
        std::sort(ct_canon.begin(), ct_canon.end());
        const int expected = std::min(n - n % kpool, topk) + n % kpool;
        const std::vector<int32_t> expected_tail((size_t) (n % kpool));
        std::vector<int32_t> tail_positions(expected_tail.size());
        for (size_t i = 0; i < tail_positions.size(); ++i) tail_positions[i] = n - n % kpool + (int) i;
        auto aligned = [kpool](std::vector<int32_t> members) {
            std::sort(members.begin(), members.end());
            if (std::adjacent_find(members.begin(), members.end()) != members.end()) return false;
            for (size_t i = 0; i < members.size(); i += (size_t) kpool) {
                if (i + kpool > members.size() || members[i] % kpool) return false;
                for (int j = 1; j < kpool; ++j) if (members[i + j] != members[i] + j) return false;
            }
            return true;
        };
        const double diff = max_abs_diff(full.logits, chunked.logits);
        const bool pass = (int) (fm.size() + ft.size()) == expected &&
                          (int) (cm.size() + ct.size()) == expected &&
                          ft_canon == tail_positions && ct_canon == tail_positions &&
                          aligned(fm) && aligned(cm) &&
                          fm_canon == cm_canon && ft_canon == ct_canon &&
                          argmax(full.logits) == argmax(chunked.logits) && diff <= tolerance;
        if (!std::isfinite(diff)) {
            throw std::runtime_error("nonfinite full/chunked logits");
        }
        all_pass &= pass;
        out << "{\"schema\":\"epyc.glm53.kpool_runtime_pair.v1\",\"prefill_tokens\":" << n
            << ",\"full_ubatch\":" << n << ",\"chunked_ubatch\":" << std::min(n, chunk)
            << ",\"kpool\":" << kpool << ",\"topk\":" << topk
            << ",\"expected_valid_selection\":" << expected << ",\"full_members\":";
        write_ints(out, fm); out << ",\"full_tail\":"; write_ints(out, ft);
        out << ",\"chunked_members\":"; write_ints(out, cm);
        out << ",\"chunked_tail\":"; write_ints(out, ct);
        out << ",\"actual_selection_set_equal\":"
            << (fm_canon == cm_canon && ft_canon == ct_canon ? "true" : "false")
            << ",\"next_token_full\":" << argmax(full.logits)
            << ",\"next_token_chunked\":" << argmax(chunked.logits)
            << ",\"max_abs_logit_diff\":" << diff
            << ",\"declared_logit_tolerance\":" << tolerance
            << ",\"verdict\":\"" << (pass ? "PASS" : "FAIL") << "\"}\n";
    }
    return all_pass ? 0 : 3;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    ggml_backend_load_all();
    auto init = common_init_from_params(params);
    llama_model * model = init ? init->model() : nullptr;
    if (!model) return 1;
    char arch[64] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0 ||
            (std::string(arch) != "glm5next" && std::string(arch) != "glm5-next")) {
        std::fprintf(stderr, "REFUSE: model is not a GLM5Next alias\n"); return 2;
    }
    if (llama_model_n_layer_nextn(model) < 1) {
        std::fprintf(stderr, "REFUSE: model has no NextN layer\n"); return 2;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const double tolerance = env_real("GLM53_TEST_LOGIT_TOLERANCE", 1e-5);
    const char * path = std::getenv("GLM53_TEST_JSONL");
    if (!path || !*path) { std::fprintf(stderr, "GLM53_TEST_JSONL is required\n"); return 2; }
    std::ofstream out(path, std::ios::trunc);
    if (!out) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    out << std::setprecision(17);
    try {
        const std::string mode = std::getenv("GLM53_TEST_MODE") ? std::getenv("GLM53_TEST_MODE") : "restore";
        if (mode == "restore") return run_restore(params, model, n_vocab, tolerance, out);
        if (mode == "pool") return run_pool(params, model, n_vocab, tolerance, out);
        if (mode == "export") return run_export(params, model, n_vocab, tolerance, out);
        std::fprintf(stderr, "REFUSE: GLM53_TEST_MODE must be restore, pool, or export\n"); return 2;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "REFUSE: %s\n", exc.what()); return 2;
    }
}
