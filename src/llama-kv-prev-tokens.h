#pragma once

// Q38FN-CPU-HOST-DEPTH-1: the history collection step of llama_kv_cache::get_prev_tokens(), shared with its unit test.
//
// The reference path walks EVERY used cell (a std::set) per ubatch: O(context) host work per decode step outside the graph
// (perf, ak-q38fn-cpu-hostdepth-perf-20261125: 0.07 ms/step at d1k -> 3.73 ms/step at d32k on Q38FN CPU).
// The fast path (LLAMA_KV_PREV_FAST=1, default off) walks the used cells in reverse and stops once every (seq, pos) key the
// lookup will ask for is present. If any key is absent it walks everything, so the collected values the lookup can observe are
// identical to the reference: reverse + first-insert-wins == forward + last-write-wins for hist, and >= in reverse == the first
// strict > in forward order for `below`.

#include "llama-kv-cells.h"

#include <array>
#include <bitset>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

inline uint64_t llama_kv_prev_key(llama_seq_id seq_id, llama_pos pos) {
    return ((uint64_t) seq_id << 32) | (uint32_t) pos;
}

// needed == nullptr: reference (full forward) walk. Otherwise the fast walk with early stop on `needed`.
inline void llama_kv_prev_collect(
        const std::vector<llama_kv_cells> & v_cells,
        uint32_t n_stream,
        const std::bitset<LLAMA_MAX_SEQ> & seqs,
        llama_pos w0,
        llama_pos p_max,
        const std::unordered_set<uint64_t> * needed,
        std::unordered_map<uint64_t, llama_token> & hist,
        std::array<std::pair<llama_pos, llama_token>, LLAMA_MAX_SEQ> & below) {
    below.fill({ -1, LLAMA_TOKEN_NULL });

    if (needed == nullptr) {
        for (uint32_t s = 0; s < n_stream; ++s) {
            // p_max inclusive: an embd token looks up cells at its own (shared) position
            v_cells[s].for_each_token_in(seqs, 0, p_max + 1,
                [&](llama_seq_id seq_id, llama_pos pos, llama_token tok) {
                    if (pos >= w0) {
                        hist[llama_kv_prev_key(seq_id, pos)] = tok;
                    } else if (pos > below[seq_id].first) {
                        below[seq_id] = { pos, tok };
                    }
                });
        }
        return;
    }

    size_t remaining = needed->size();
    if (remaining == 0) {
        return;
    }

    // streams are visited in reverse too, so "last stream, last cell wins" of the reference is "first visit wins" here
    for (uint32_t s = n_stream; s-- > 0;) {
        const bool done = v_cells[s].for_each_token_in_rev(seqs, 0, p_max + 1,
            [&](llama_seq_id seq_id, llama_pos pos, llama_token tok) {
                if (pos >= w0) {
                    const uint64_t k = llama_kv_prev_key(seq_id, pos);
                    if (hist.emplace(k, tok).second && needed->count(k)) {
                        return --remaining == 0;
                    }
                } else if (pos >= below[seq_id].first && pos >= 0) {
                    below[seq_id] = { pos, tok };
                }
                return false;
            });
        if (done) {
            return;
        }
    }
}
