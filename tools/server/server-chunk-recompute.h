#pragma once

// Chunk-KV reuse phase P1, KPF-62.4: assemble a prompt from injected chunks, then SPARSE RECOMPUTE selected tokens by
// explicit-position batches. Shared by the server (`kv_chunks`, the branch beside the --cache-reuse block), the
// libllama-level G0 test (tests/test-chunk-kv-recompute.cpp) and the G3 harness (tests/test-chunk-kv-g3.cpp), so the
// code that is gated is the code that serves.
//
// A prompt region is cut into segments in position order: GAPS (ordinary tokens, exact prefill) and CHUNKS (declared
// spans whose K/V comes from an isolated encode at positions 0..n-1, injected at the span offset; K is re-rotated by the
// in-tree K-shift graph). After a chunk is injected, its SELECTED tokens are recomputed: the cells of a selected run are
// removed (llama_memory_seq_rm) and the run is decoded again at its real positions with llama_chunk_kv_decode_at, runs in
// increasing position order, one contiguous run per decode (split at n_batch). Causal masking is by position, so a
// recomputed token attends the true prefix, the fresh recomputed tokens before it and the stale (shifted) cells of the
// unselected tokens before it; its own fresh K/V replaces the stale cells it had.
//
// Recompute modes:
//   none  r = 0: shift only, the naive baseline of the G3 gate
//   head  the first `window` tokens of every chunk (selector v0; KPF-62.3 adds check-layer deviation and aux scoring)
//   all   r = 100%: every chunk token recomputed and the store bypassed, so nothing is injected. The region is decoded by
//         explicit-position batches aligned to the plain prefill's n_batch partition, which makes the request
//         bit-identical to the same request without `kv_chunks` (gate G0a, P0_THRESHOLDS section 1).
//
// Header only; depends on llama.h and the experimental llama-ext.h only.

#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace server_chunk {

struct span {
    int32_t offset   = 0; // first token position of the chunk in the prompt
    int32_t n_tokens = 0;
};

enum select_mode {
    SELECT_NONE       = 0,
    SELECT_HEAD       = 1,
    SELECT_ALL        = 2, // r = 100%, store bypassed (G0a)
    SELECT_ALL_INJECT = 3, // diagnostic: inject, then recompute every chunk token (G0a-inj, harness sanity arm)
};

inline const char * select_mode_name(select_mode m) {
    switch (m) {
        case SELECT_NONE:       return "none";
        case SELECT_HEAD:       return "head";
        case SELECT_ALL:        return "all";
        case SELECT_ALL_INJECT: return "all-inject";
    }
    return "?";
}

inline bool parse_select_mode(const std::string & s, select_mode & out) {
    if (s == "none")       { out = SELECT_NONE;       return true; }
    if (s == "head")       { out = SELECT_HEAD;       return true; }
    if (s == "all")        { out = SELECT_ALL;        return true; }
    if (s == "all-inject") { out = SELECT_ALL_INJECT; return true; }
    return false;
}

// half-open position range
struct run {
    int32_t p0 = 0;
    int32_t p1 = 0;

    int32_t n() const { return p1 - p0; }
};

// sort by offset, then require: n_tokens >= 1, offset >= p_min, end <= p_max, no overlap. On false `why` says which.
inline bool validate_spans(std::vector<span> & spans, int32_t p_min, int32_t p_max, std::string & why) {
    std::sort(spans.begin(), spans.end(), [](const span & a, const span & b) { return a.offset < b.offset; });
    int32_t prev_end = p_min;
    for (const auto & sp : spans) {
        if (sp.n_tokens < 1) {
            why = "chunk with n_tokens < 1";
            return false;
        }
        if (sp.offset < p_min) {
            why = "chunk starts before the reusable prefix ends";
            return false;
        }
        if (sp.offset < prev_end) {
            why = "chunks overlap";
            return false;
        }
        if ((int64_t) sp.offset + sp.n_tokens > (int64_t) p_max) {
            why = "chunk runs past the last token that may be reused";
            return false;
        }
        prev_end = sp.offset + sp.n_tokens;
    }
    return true;
}

// the runs of one chunk that are recomputed, in absolute positions, sorted and disjoint
inline std::vector<run> select_runs(const span & sp, select_mode mode, int32_t window) {
    std::vector<run> out;
    switch (mode) {
        case SELECT_NONE:
            break;
        case SELECT_ALL:
        case SELECT_ALL_INJECT:
            out.push_back({ sp.offset, sp.offset + sp.n_tokens });
            break;
        case SELECT_HEAD:
            if (window > 0) {
                out.push_back({ sp.offset, sp.offset + std::min(window, sp.n_tokens) });
            }
            break;
    }
    return out;
}

inline int32_t runs_tokens(const std::vector<run> & runs) {
    int32_t n = 0;
    for (const auto & r : runs) {
        n += r.n();
    }
    return n;
}

// a region [p_begin, p_end) cut into gaps and chunks, in position order (spans validated and sorted)
struct segment {
    bool    is_chunk = false;
    int32_t p0       = 0;
    int32_t p1       = 0;
    int32_t i_chunk  = -1; // index into the spans when is_chunk
};

inline std::vector<segment> make_segments(const std::vector<span> & spans, int32_t p_begin, int32_t p_end) {
    std::vector<segment> out;
    int32_t p = p_begin;
    for (size_t i = 0; i < spans.size(); ++i) {
        const auto & sp = spans[i];
        if (sp.offset >= p_end) {
            break;
        }
        if (sp.offset > p) {
            out.push_back({ false, p, sp.offset, -1 });
        }
        out.push_back({ true, sp.offset, sp.offset + sp.n_tokens, (int32_t) i });
        p = sp.offset + sp.n_tokens;
    }
    if (p < p_end) {
        out.push_back({ false, p, p_end, -1 });
    }
    return out;
}

// Decode toks[p0, p1) (toks is indexed by absolute position) at positions p0.. in seq, in batches of at most n_batch.
// holes = false: llama_decode (the range must continue the sequence). holes = true: llama_chunk_kv_decode_at (the range
// may fill a hole). out_logits: when given, logits are requested for every token and appended row by row (n_vocab each).
// Returns 0 on success, otherwise the failing decode's result (or -1).
inline int32_t decode_range(llama_context * ctx, const llama_token * toks, int32_t p0, int32_t p1, llama_seq_id seq,
        int32_t n_batch, bool holes, std::vector<float> * out_logits = nullptr, int32_t n_vocab = 0) {
    if (p1 <= p0) {
        return 0;
    }
    n_batch = std::max(1, std::min(n_batch, (int32_t) llama_n_batch(ctx)));

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int32_t ret = 0;
    for (int32_t b0 = p0; b0 < p1 && ret == 0; b0 += n_batch) {
        const int32_t b1 = std::min(p1, b0 + n_batch);
        batch.n_tokens = 0;
        for (int32_t p = b0; p < b1; ++p) {
            const int32_t i = batch.n_tokens++;
            batch.token[i]     = toks[p];
            batch.pos[i]       = p;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = seq;
            batch.logits[i]    = out_logits != nullptr;
        }
        ret = holes ? llama_chunk_kv_decode_at(ctx, batch) : llama_decode(ctx, batch);
        if (ret == 0 && out_logits) {
            const size_t base = out_logits->size();
            out_logits->resize(base + (size_t) (b1 - b0) * n_vocab);
            for (int32_t i = 0; i < b1 - b0; ++i) {
                const float * row = llama_get_logits_ith(ctx, i);
                if (row == nullptr) {
                    ret = -1;
                    break;
                }
                memcpy(out_logits->data() + base + (size_t) i * n_vocab, row, (size_t) n_vocab * sizeof(float));
            }
        }
    }
    llama_batch_free(batch);
    return ret;
}

// Sparse recompute: for each run in increasing position order, remove its cells from seq and decode it again at its
// real positions (a hole fill). Returns 0 on success, otherwise the failing decode's result (or -1).
inline int32_t recompute_runs(llama_context * ctx, const llama_token * toks, const std::vector<run> & runs, llama_seq_id seq,
        int32_t n_batch) {
    std::vector<run> sorted = runs;
    std::sort(sorted.begin(), sorted.end(), [](const run & a, const run & b) { return a.p0 < b.p0; });
    for (const auto & r : sorted) {
        if (r.n() <= 0) {
            continue;
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx), seq, r.p0, r.p1)) {
            return -1;
        }
        const int32_t ret = decode_range(ctx, toks, r.p0, r.p1, seq, n_batch, true);
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
}

// G0a alignment for mode "all": the largest end <= p_end such that [p_begin, end) is a whole number of n_batch batches,
// i.e. exactly the batches the plain prefill loop makes from p_begin. The rest is left to that loop.
inline int32_t aligned_end(int32_t p_begin, int32_t p_end, int32_t n_batch) {
    if (n_batch <= 0 || p_end <= p_begin) {
        return p_begin;
    }
    return p_begin + ((p_end - p_begin) / n_batch) * n_batch;
}

struct blob_ref {
    const uint8_t * data = nullptr;
    size_t          size = 0;
};

struct assemble_stats {
    int32_t n_gap        = 0; // tokens prefilled exactly (gaps, and chunks that had no blob)
    int32_t n_injected   = 0; // chunk tokens placed from a blob
    int32_t n_recomputed = 0; // chunk tokens recomputed by sparse decode (mode all: every token of the aligned region)
    int32_t n_no_blob    = 0; // chunks served by exact prefill because no blob was available
};

// Assemble seq over [p_begin, p_end) in position order. The sequence must hold exactly [0, p_begin) (nothing at or past
// p_begin). spans: validated and sorted, inside [p_begin, p_end). get_blob(i) -> blob_ref for chunk i ({nullptr, 0}: no
// blob, the chunk is prefilled exactly). On success *p_done is the first position NOT assembled (== p_end, except in mode
// all, where it is the end of the n_batch-aligned region and the caller's plain prefill continues from there).
// Returns 0 on success, otherwise a decode result or -1 (the caller must then drop the sequence from p_begin).
template <typename F>
inline int32_t assemble(llama_context * ctx, const llama_token * toks, const std::vector<span> & spans, int32_t p_begin,
        int32_t p_end, llama_seq_id seq, int32_t n_batch, select_mode mode, int32_t window, F && get_blob,
        assemble_stats & st, int32_t * p_done) {
    *p_done = p_begin;

    if (mode == SELECT_ALL) {
        const int32_t end = aligned_end(p_begin, p_end, n_batch);
        const int32_t ret = decode_range(ctx, toks, p_begin, end, seq, n_batch, true);
        if (ret != 0) {
            return ret;
        }
        st.n_recomputed += end - p_begin;
        *p_done = end;
        return 0;
    }

    for (const auto & sg : make_segments(spans, p_begin, p_end)) {
        if (!sg.is_chunk) {
            const int32_t ret = decode_range(ctx, toks, sg.p0, sg.p1, seq, n_batch, false);
            if (ret != 0) {
                return ret;
            }
            st.n_gap += sg.p1 - sg.p0;
            *p_done = sg.p1;
            continue;
        }

        const blob_ref b = get_blob(sg.i_chunk);
        if (b.data == nullptr || b.size == 0) {
            const int32_t ret = decode_range(ctx, toks, sg.p0, sg.p1, seq, n_batch, false);
            if (ret != 0) {
                return ret;
            }
            st.n_gap += sg.p1 - sg.p0;
            st.n_no_blob++;
            *p_done = sg.p1;
            continue;
        }

        // the blob holds positions 0..n-1 of an isolated encode: place it at the chunk offset
        if (llama_chunk_kv_inject(ctx, b.data, b.size, seq, sg.p0) != b.size) {
            return -1;
        }
        if (llama_memory_seq_pos_max(llama_get_memory(ctx), seq) != sg.p1 - 1) {
            return -1; // blob length does not match the declared chunk
        }
        st.n_injected += sg.p1 - sg.p0;

        const auto runs = select_runs(spans[sg.i_chunk], mode, window);
        const int32_t ret = recompute_runs(ctx, toks, runs, seq, n_batch);
        if (ret != 0) {
            return ret;
        }
        st.n_recomputed += runs_tokens(runs);
        *p_done = sg.p1;
    }
    return 0;
}

} // namespace server_chunk
