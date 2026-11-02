#pragma once

#include "llama-batch.h"

#include <cstdint>
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
        if (batch.n_seq_id[i] != 1 || !batch.seq_id[i]) {
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
