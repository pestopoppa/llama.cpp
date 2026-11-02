#include "llama-kv-rows-reference.h"

#include <cmath>
#include <limits>
#include <unordered_set>

bool llama_kv_rows_reference_graph(
        ggml_context * ctx, const llama_kv_rows_plan & plan,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        const std::vector<ggml_fp16_t> & full_mask, float scale,
        std::vector<ggml_tensor *> & out) {
    out.clear();
    // Restrict the reference to one KV head, one stream and equal dimensions;
    // all other shapes continue through the existing full-mask graph.
    if (!ctx || !q || !k || !v || !plan.usable() || !std::isfinite(scale) ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        q->ne[2] != 1 || q->ne[3] != 1 || k->ne[2] != 1 || k->ne[3] != 1 ||
        v->ne[2] != 1 || v->ne[3] != 1 || k->ne[1] != plan.n_kv || v->ne[1] != plan.n_kv ||
        q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0] || q->ne[1] <= 0 ||
        q->ne[1] > UINT32_MAX || !ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v)) return false;
    const uint32_t n_tokens = static_cast<uint32_t>(q->ne[1]);
    std::vector<ggml_fp16_t> compact;
    const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    if (!llama_kv_rows_compact_mask(plan, n_tokens, full_mask, masked, compact)) return false;
    // A repeated row changes softmax weights. Reject it rather than silently
    // making a malformed plan appear valid. Unknown/padding is represented -1.
    for (uint32_t s = 0; s < plan.n_slices; ++s) {
        std::unordered_set<int32_t> seen;
        for (uint32_t r = 0; r < plan.padded_rows; ++r) {
            int32_t row = plan.rows[size_t(s) * plan.padded_rows + r];
            if (row >= 0 && !seen.insert(row).second) return false;
        }
    }
    const uint32_t per_slice = n_tokens / plan.n_slices;
    for (uint32_t s = 0; s < plan.n_slices; ++s) {
        std::unordered_set<int32_t> present;
        for (uint32_t r = 0; r < plan.padded_rows; ++r) {
            const auto row = plan.rows[size_t(s) * plan.padded_rows + r];
            if (row >= 0) present.insert(row);
        }
        for (uint32_t t = 0; t < per_slice; ++t) {
            bool any = false;
            const size_t token = size_t(s) * per_slice + t;
            for (uint32_t j = 0; j < plan.n_kv; ++j) {
                float m = ggml_fp16_to_fp32(full_mask[token * plan.n_kv + j]);
                if (std::isfinite(m)) {
                    if (!present.count(static_cast<int32_t>(j))) return false;
                    any = true;
                } else if (!(std::isinf(m) && m < 0)) return false;
            }
            // Existing kernels differ on fully masked rows; fall back.
            if (!any) return false;
        }
    }
    std::vector<ggml_tensor *> outputs;
    outputs.reserve(plan.n_slices);
    for (uint32_t s = 0; s < plan.n_slices; ++s) {
        auto * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, plan.padded_rows);
        auto * ids = static_cast<int32_t *>(rows->data);
        if (!ids) return false;
        for (uint32_t r = 0; r < plan.padded_rows; ++r) {
            const auto row = plan.rows[size_t(s) * plan.padded_rows + r];
            ids[r] = row < 0 ? 0 : row; // padding's compact mask is always -inf
        }
        auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, plan.padded_rows, per_slice);
        if (!mask->data) return false;
        std::copy_n(compact.data() + size_t(s) * per_slice * plan.padded_rows,
                    size_t(per_slice) * plan.padded_rows, static_cast<ggml_fp16_t *>(mask->data));
        auto * qs = ggml_view_4d(ctx, q, q->ne[0], per_slice, 1, 1, q->nb[1], q->nb[2], q->nb[3],
                                 size_t(s) * per_slice * q->nb[1]);
        auto * ks = ggml_get_rows(ctx, k, rows);
        auto * vs = ggml_get_rows(ctx, v, rows);
        outputs.push_back(ggml_flash_attn_ext(ctx, qs, ks, vs, mask, scale, 0.0f, 0.0f));
    }
    out.swap(outputs);
    return true;
}
