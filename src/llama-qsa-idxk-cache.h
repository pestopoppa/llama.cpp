#pragma once

// [TAG_QSA_IDXK_CACHE] LLAMA_QSA_IDXK_CACHE=1 (default off): keep the qwen4exp QSA indexer's pooled,
// normed and roped block keys across decode steps, so a step pools+norms+ropes only the blocks that
// are new or changed instead of all n_blocks.
//
// What is cached: the POST-rope row of each block. The rope position of block b is b*ratio (see
// set_input_qsa), a pure function of the block index, so caching after the rope is as exact as caching
// before it and also saves the rope. The cached row is only ever produced by the same graph ops the
// full recompute runs (get_rows -> slice sum -> scale -> rms_norm*w -> rope), all of which are row
// independent on the CPU backend, so a row computed from identical inputs is bit-identical.
//
// The only job of this header is the identical-inputs guarantee. A block's row depends on the raw
// indexer keys of its ratio cells and on its index. Validity rule, per (ratio, stream):
//   - a block is cacheable only when it is complete (every one of its ratio positions has a cell) and
//     every one of its positions is below the lowest position the current ubatch writes;
//   - a cached block stays valid only while (a) its ratio cells and their positions are unchanged and
//     (b) no later ubatch or seq_rm touched a position <= its last position (`on_write`). (b) is what
//     catches an MTP rollback or a prompt-cache suffix re-decode that rewrites a cell at the SAME
//     index and position with a different token;
//   - validity is a prefix: blocks [0, n_valid) are valid, so a hole invalidates everything after it;
//   - anything else (seq_add/seq_div/seq_cp/seq_keep/clear/state_read) resets the stream.
// This header is pure host logic with no ggml dependency so tests/test-qsa-idxk-cache.cpp can drive it.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <vector>

// map cells to blocks exactly as llama_memory_hybrid_idx_context::set_input_qsa does
//   cell_pos  [n_kv]           position of each cell, < 0 for an empty cell
//   blk_cells [r*n_blocks]     out: cell at slot (b, p%r), 0 where no cell
//   filled    [n_blocks]       out: number of cells landing in block b
// returns false when a cell position runs past the block window (the caller then falls back)
static inline bool llama_qsa_idxk_map(uint32_t r, uint32_t n_blocks, const llama_pos * cell_pos, uint32_t n_kv,
                                      int32_t * blk_cells, int32_t * filled) {
    std::fill(blk_cells, blk_cells + (size_t) r*n_blocks, 0);
    std::fill(filled,    filled    + n_blocks,            0);

    bool ok = true;
    for (uint32_t j = 0; j < n_kv; ++j) {
        const llama_pos p = cell_pos[j];
        if (p < 0) {
            continue;
        }
        const uint32_t b = (uint32_t) p / r;
        if (b >= n_blocks) {
            ok = false;
            continue;
        }
        blk_cells[(size_t) b*r + (uint32_t) p % r] = (int32_t) j;
        filled[b]++;
    }
    return ok;
}

struct llama_qsa_idxk_validity {
    uint32_t r       = 0;
    uint32_t n_valid = 0;            // blocks [0, n_valid) hold a valid cached row
    std::vector<int32_t> cells;      // [r*n_valid] the cell of every slot when the row was cached

    void reset() {
        n_valid = 0;
        cells.clear();
    }

    // a ubatch or a seq_rm writes/removes positions >= p_min: drop every block whose last position >= p_min
    void on_write(llama_pos p_min) {
        if (p_min < 0) {
            p_min = 0;
        }
        const uint32_t keep = (uint32_t) p_min / r;   // block b ends at b*r + r - 1 < p_min  <=>  b < p_min/r
        if (keep < n_valid) {
            n_valid = keep;
            cells.resize((size_t) r*n_valid);
        }
    }

    // leading blocks whose cached row may be reused against the current layout
    uint32_t n_reusable(const int32_t * blk_cells, const int32_t * filled, uint32_t n_blocks) const {
        const uint32_t n = std::min(n_valid, n_blocks);
        for (uint32_t b = 0; b < n; ++b) {
            if (filled[b] != (int32_t) r) {
                return b;
            }
            for (uint32_t i = 0; i < r; ++i) {
                if (blk_cells[(size_t) b*r + i] != cells[(size_t) b*r + i]) {
                    return b;
                }
            }
        }
        return n;
    }

    // leading blocks cacheable after a ubatch whose lowest position (in this stream) is ub_min_pos
    static uint32_t n_cacheable(uint32_t r, const int32_t * filled, uint32_t n_blocks, llama_pos ub_min_pos) {
        const uint32_t lim = ub_min_pos < 0 ? 0 : std::min<uint32_t>(n_blocks, (uint32_t) ub_min_pos / r);
        uint32_t b = 0;
        while (b < lim && filled[b] == (int32_t) r) {
            ++b;
        }
        return b;
    }

    // the step computed successfully: rows [0, c) were valid and reused, rows [c, n_blocks) were just
    // recomputed from the current cells and written, so every cacheable leading block is now valid
    void commit(const int32_t * blk_cells, const int32_t * filled, uint32_t n_blocks, llama_pos ub_min_pos) {
        n_valid = n_cacheable(r, filled, n_blocks, ub_min_pos);
        cells.assign(blk_cells, blk_cells + (size_t) r*n_valid);
    }
};

// first recomputed block for a step: never past the reusable prefix, never the last block (so at least one
// row is recomputed and no tensor is empty), rounded down to `gran` so graph reuse survives most steps
static inline uint32_t llama_qsa_idxk_split(uint32_t n_reusable_min, uint32_t n_blocks, uint32_t gran) {
    if (n_blocks == 0) {
        return 0;
    }
    uint32_t c = std::min(n_reusable_min, n_blocks - 1);
    if (gran > 1) {
        c = c/gran*gran;
    }
    return c;
}
