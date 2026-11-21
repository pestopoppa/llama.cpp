// DRAFT-HEAD-SVD-CPU: sidecar loader for the two-stage truncated-SVD MTP draft head (see llama-mtp-svd-ops.h).
// Env LLAMA_MTP_DRAFT_HEAD_SVD=<factor gguf> (default unset = off); LLAMA_MTP_DRAFT_HEAD_SVD_C=<n> overrides the file's C.
#include "llama-mtp-svd.h"

#include "llama-impl.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

struct svd_state {
    llama_mtp_svd f = {};
    bool ok = false;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

svd_state g_svd;
std::once_flag g_svd_once;

void svd_load() {
    const char * path = getenv("LLAMA_MTP_DRAFT_HEAD_SVD");
    if (!path || !*path) {
        return;
    }
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) {
        LLAMA_LOG_ERROR("%s: LLAMA_MTP_DRAFT_HEAD_SVD ignored: cannot read %s; draft head stays full\n", __func__, path);
        return;
    }
    ggml_tensor * v = ggml_get_tensor(meta, "mtp_svd.v");
    ggml_tensor * a = ggml_get_tensor(meta, "mtp_svd.a");
    const int64_t kc = gguf_find_key(g, "mtp_svd.c");
    if (!v || !a || kc < 0 || v->type != GGML_TYPE_F32 || v->ne[1] != a->ne[0]) {
        LLAMA_LOG_ERROR("%s: LLAMA_MTP_DRAFT_HEAD_SVD: %s lacks mtp_svd.v (F32 [d,r]) / mtp_svd.a ([r,V]) / mtp_svd.c\n", __func__, path);
        gguf_free(g);
        ggml_free(meta);
        return;
    }
    int64_t C = gguf_get_val_u32(g, kc);
    if (const char * cs = getenv("LLAMA_MTP_DRAFT_HEAD_SVD_C")) {
        C = std::atoll(cs);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(meta, ggml_backend_cpu_buffer_type());
    if (!buf) {
        LLAMA_LOG_ERROR("%s: LLAMA_MTP_DRAFT_HEAD_SVD: buffer allocation failed\n", __func__);
        gguf_free(g);
        ggml_free(meta);
        return;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    FILE * fp = fopen(path, "rb");
    bool rd = fp != nullptr;
    for (int64_t i = 0; rd && i < gguf_get_n_tensors(g); ++i) {
        ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(g, i));
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
        rd = t && fseek(fp, (long) off, SEEK_SET) == 0 && fread(t->data, 1, ggml_nbytes(t), fp) == ggml_nbytes(t);
    }
    if (fp) {
        fclose(fp);
    }
    gguf_free(g);
    if (!rd) {
        LLAMA_LOG_ERROR("%s: LLAMA_MTP_DRAFT_HEAD_SVD: short read of %s\n", __func__, path);
        ggml_backend_buffer_free(buf);
        ggml_free(meta);
        return;
    }
    g_svd.ctx = meta;
    g_svd.buf = buf;
    g_svd.f = { v, a, C };
    g_svd.ok = true;
    // stderr, not only LLAMA_LOG: llama-server drops library INFO lines, and the A/B rows prove the knob from this line
    fprintf(stderr, "[mtp-svd] ACTIVE: LLAMA_MTP_DRAFT_HEAD_SVD: %s rank %lld, d %lld, vocab %lld, A %s, C %lld\n", path,
            (long long) v->ne[1], (long long) v->ne[0], (long long) a->ne[1], ggml_type_name(a->type), (long long) C);
    LLAMA_LOG_INFO("%s: LLAMA_MTP_DRAFT_HEAD_SVD: %s rank %lld, C %lld\n", __func__, path, (long long) v->ne[1], (long long) C);
}

} // namespace

const llama_mtp_svd * llama_mtp_svd_get(const ggml_tensor * head_w) {
    std::call_once(g_svd_once, svd_load);
    if (!g_svd.ok || !head_w) {
        return nullptr;
    }
    static std::once_flag warn_once;
    const bool shape_ok = g_svd.f.v->ne[0] == head_w->ne[0] && g_svd.f.a->ne[1] == head_w->ne[1];
    // the gather copies W rows verbatim: W must be a plain host tensor (no repacked/extra buffer layout)
    const char * bn = head_w->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(head_w->buffer)) : "";
    const bool layout_ok = head_w->buffer && ggml_backend_buffer_is_host(head_w->buffer) && head_w->extra == nullptr &&
                           !ggml_is_transposed(head_w) && ggml_is_contiguous(head_w) && std::strstr(bn, "REPACK") == nullptr &&
                           std::strstr(bn, "AMX") == nullptr;
    if (!shape_ok || !layout_ok) {
        std::call_once(warn_once, [&] {
            fprintf(stderr, "[mtp-svd] LLAMA_MTP_DRAFT_HEAD_SVD ignored: head [%lld, %lld] in buffer '%s' (factors d %lld vocab %lld)\n",
                    (long long) head_w->ne[0], (long long) head_w->ne[1], bn,
                    (long long) g_svd.f.v->ne[0], (long long) g_svd.f.a->ne[1]);
        });
        return nullptr;
    }
    return &g_svd.f;
}
