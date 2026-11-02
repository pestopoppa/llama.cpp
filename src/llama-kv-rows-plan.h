#pragma once

#include "llama-batch.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <unordered_set>

enum class llama_kv_rows_sharing { absent, present, unknown };

// Conservative shape preflight for a future kv_rows graph. Return zero to use
// the existing full-mask attention path. The caller must separately establish
// that no cache cell is shared between sequence owners before passing absent.
// This function does not enable kv_rows or alter the attention graph.
inline uint32_t llama_kv_rows_slices(
        const llama_ubatch & batch, bool backend_supported,
        uint32_t n_stream, llama_kv_rows_sharing sharing) {
    if (!backend_supported || n_stream != 1 || sharing != llama_kv_rows_sharing::absent ||
        batch.n_tokens == 0 || batch.n_seqs_unq == 0 ||
        !batch.n_seq_id || !batch.seq_id) {
        return 0;
    }

    // Verify the actual token owners, not just batch shape flags. In particular,
    // a shared token cannot be represented by disjoint per-sequence rows.
    for (uint32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || !batch.seq_id[i] ||
            batch.seq_id[i][0] < 0 || batch.seq_id[i][0] >= LLAMA_MAX_SEQ) {
            return 0;
        }
    }

    uint32_t slices = 0;
    if (batch.n_seqs_unq == 1) {
        slices = 1;
    } else if (batch.equal_seqs() && batch.n_seqs == batch.n_seqs_unq) {
        slices = batch.n_seqs;
    } else if (batch.n_tokens == batch.n_seqs_unq) {
        slices = batch.n_tokens;
    } else {
        return 0;
    }

    if (slices == 0 || batch.n_tokens % slices != 0) {
        return 0;
    }
    const uint32_t tokens_per_slice = batch.n_tokens / slices;
    std::unordered_set<llama_seq_id> owners;
    for (uint32_t s = 0; s < slices; ++s) {
        const auto id = batch.seq_id[s * tokens_per_slice][0];
        if (!owners.insert(id).second) {
            return 0;
        }
        for (uint32_t i = 1; i < tokens_per_slice; ++i) {
            if (batch.seq_id[s * tokens_per_slice + i][0] != id) {
                return 0;
            }
        }
    }
    return owners.size() == batch.n_seqs_unq ? slices : 0;
}

// A stable graph input layout: row indices are physical cache cells, grouped
// by slice, with -1 padding. An empty plan always means use the full mask.
// Recompute this plan for every ubatch; the graph may be reused only when
// n_kv, n_slices and padded_rows match, while its row input is refreshed.
struct llama_kv_rows_plan {
    uint32_t n_kv = 0;
    uint32_t n_slices = 0;
    uint32_t padded_rows = 0;
    std::vector<int32_t> rows;

    bool usable() const { return n_kv != 0 && n_slices != 0 && padded_rows != 0 && padded_rows <= n_kv; }
    bool same_graph_shape(const llama_kv_rows_plan & other) const {
        return usable() && other.usable() && n_kv == other.n_kv &&
               n_slices == other.n_slices && padded_rows == other.padded_rows;
    }
};

// Cells supplies size(), is_empty(i), seq_count(i), seq_has(i,id). Only
// single-owner cells may enter the sparse graph; shared or unknown ownership
// causes complete fallback rather than a partial/incorrect row map.
template<typename Cells>
llama_kv_rows_plan llama_kv_rows_make_plan(const llama_ubatch & batch, const Cells & cells,
        uint32_t n_kv, bool backend_supported, uint32_t n_stream) {
    llama_kv_rows_plan result;
    if (n_kv == 0 || n_kv > cells.size() || n_kv > uint32_t(std::numeric_limits<int32_t>::max())) return result;
    const uint32_t slices = llama_kv_rows_slices(batch, backend_supported, n_stream,
                                                 llama_kv_rows_sharing::absent);
    if (!slices) return result;
    const uint32_t tokens_per_slice = batch.n_tokens / slices;
    std::vector<std::vector<int32_t>> per_slice(slices);
    for (uint32_t i = 0; i < n_kv; ++i) {
        if (cells.is_empty(i)) continue;
        if (cells.seq_count(i) != 1) return {};
        for (uint32_t s = 0; s < slices; ++s) {
            const auto id = batch.seq_id[s*tokens_per_slice][0];
            if (id < 0 || id >= LLAMA_MAX_SEQ) return {};
            if (cells.seq_has(i, id)) per_slice[s].push_back(int32_t(i));
        }
    }
    size_t max_rows = 0;
    for (const auto & rows : per_slice) max_rows = std::max(max_rows, rows.size());
    if (!max_rows) return result;
    // Match the cache's 256-cell attention padding without exceeding n_kv.
    const size_t padded = std::min<size_t>(((max_rows + 255) / 256) * 256, n_kv);
    if (padded > std::numeric_limits<uint32_t>::max() || padded > SIZE_MAX / slices) return result;
    result.n_kv = n_kv;
    result.n_slices = slices;
    result.padded_rows = uint32_t(padded);
    result.rows.assign(padded * slices, -1);
    for (uint32_t s = 0; s < slices; ++s) {
        std::copy(per_slice[s].begin(), per_slice[s].end(), result.rows.begin() + s*padded);
    }
    return result;
}

// Build a compact mask in [slice, token-in-slice, row] order from the
// existing full [token, physical-cell] mask. A graph consumer must retain the
// full mask fallback and refresh both inputs on every execution. Padding is
// always supplied by the caller (negative infinity for floating masks).
// Returns false without exposing partial data on inconsistent dimensions.
template<typename T>
bool llama_kv_rows_compact_mask(const llama_kv_rows_plan & plan, uint32_t n_tokens,
                                const std::vector<T> & full_mask, T masked,
                                std::vector<T> & out) {
    out.clear();
    if (!plan.usable() || n_tokens == 0 || n_tokens % plan.n_slices != 0 ||
        size_t(n_tokens) > SIZE_MAX / plan.n_kv ||
        full_mask.size() != size_t(n_tokens) * plan.n_kv ||
        size_t(plan.padded_rows) > SIZE_MAX / plan.n_slices ||
        plan.rows.size() != size_t(plan.padded_rows) * plan.n_slices ||
        size_t(n_tokens) > SIZE_MAX / plan.padded_rows) return false;
    std::vector<T> compact(size_t(n_tokens) * plan.padded_rows, masked);
    const auto per_slice = n_tokens / plan.n_slices;
    for (uint32_t s = 0; s < plan.n_slices; ++s) {
        for (uint32_t r = 0; r < plan.padded_rows; ++r) {
            const auto physical = plan.rows[size_t(s) * plan.padded_rows + r];
            if (physical < -1 || physical >= int64_t(plan.n_kv)) return false;
            if (physical == -1) continue;
            for (uint32_t t = 0; t < per_slice; ++t) {
                const size_t token = size_t(s) * per_slice + t;
                compact[token * plan.padded_rows + r] = full_mask[token * plan.n_kv + physical];
            }
        }
    }
    out.swap(compact);
    return true;
}
