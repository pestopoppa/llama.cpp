// Q38FN-GEMV2 DRAFTVOCAB: MTP draft head on a fixed top-K vocabulary (see llama-mtp-vocab-ops.h).
// Env LLAMA_MTP_DRAFT_VOCAB=<token id list> (default unset = off); LLAMA_MTP_DRAFT_VOCAB_K=<n> (default 32768).
#include "llama-mtp-vocab.h"

#include "llama-impl.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

struct vocab_state {
    llama_mtp_vocab v = {};
    bool ok = false;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

vocab_state g_vs;
std::once_flag g_vs_once;

bool head_is_plain_host(const ggml_tensor * w, const char ** bn_out) {
    const char * bn = w->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(w->buffer)) : "";
    *bn_out = bn;
    return w->buffer && ggml_backend_buffer_is_host(w->buffer) && w->extra == nullptr && !ggml_is_transposed(w) &&
           ggml_is_contiguous(w) && std::strstr(bn, "REPACK") == nullptr && std::strstr(bn, "AMX") == nullptr;
}

void vocab_build(const ggml_tensor * w) {
    const char * path = getenv("LLAMA_MTP_DRAFT_VOCAB");
    if (!path || !*path) {
        return;
    }
    const char * bn = "";
    if (!head_is_plain_host(w, &bn)) {
        fprintf(stderr, "[mtp-vocab] LLAMA_MTP_DRAFT_VOCAB ignored: head in buffer '%s' is not a plain host tensor\n", bn);
        return;
    }
    int64_t kmax = 32768;
    if (const char * ks = getenv("LLAMA_MTP_DRAFT_VOCAB_K")) {
        kmax = std::max<int64_t>(1, std::atoll(ks));
    }
    const int64_t V = w->ne[1];
    std::vector<int32_t> ids;
    FILE * fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[mtp-vocab] LLAMA_MTP_DRAFT_VOCAB ignored: cannot read %s\n", path);
        return;
    }
    long long t;
    while ((int64_t) ids.size() < kmax && fscanf(fp, "%lld", &t) == 1) {
        if (t >= 0 && t < V) {
            ids.push_back((int32_t) t);
        }
    }
    fclose(fp);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.empty()) {
        fprintf(stderr, "[mtp-vocab] LLAMA_MTP_DRAFT_VOCAB ignored: %s has no ids in [0, %lld)\n", path, (long long) V);
        return;
    }
    const int64_t K = (int64_t) ids.size();
    ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ws = ggml_new_tensor_2d(ctx, w->type, w->ne[0], K);
    ggml_tensor * ix = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K);
    ggml_set_name(ws, "mtp_vocab.w_sub");
    ggml_set_name(ix, "mtp_vocab.idx");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (!buf) {
        fprintf(stderr, "[mtp-vocab] LLAMA_MTP_DRAFT_VOCAB ignored: buffer allocation failed\n");
        ggml_free(ctx);
        return;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t rs = ggml_row_size(w->type, w->ne[0]);
    for (int64_t j = 0; j < K; ++j) {
        std::memcpy((char *) ws->data + j * ws->nb[1], (const char *) w->data + (int64_t) ids[j] * w->nb[1], rs);
    }
    std::memcpy(ix->data, ids.data(), K * sizeof(int32_t));
    g_vs.ctx = ctx;
    g_vs.buf = buf;
    g_vs.v = { ws, ix };
    g_vs.ok = true;
    // stderr, not only LLAMA_LOG: llama-server drops library INFO lines, and the A/B rows prove the knob from this line
    fprintf(stderr, "[mtp-vocab] INFO: LLAMA_MTP_DRAFT_VOCAB=%s ACTIVE: K %lld of vocab %lld, head %s (%.1f MB sub-head)\n",
            path, (long long) K, (long long) V, ggml_type_name(w->type), (double) ggml_nbytes(ws) / 1e6);
    LLAMA_LOG_INFO("%s: LLAMA_MTP_DRAFT_VOCAB: %s K %lld\n", __func__, path, (long long) K);
}

} // namespace

const llama_mtp_vocab * llama_mtp_vocab_get(const ggml_tensor * head_w) {
    if (!head_w) {
        return nullptr;
    }
    std::call_once(g_vs_once, vocab_build, head_w);
    return g_vs.ok ? &g_vs.v : nullptr;
}
