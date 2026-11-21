#pragma once

// Expected Attention KV Cache Compression
//
// Scores KV cache entries by predicted future attention importance using
// the Expected Attention algorithm (NVIDIA KVPress). Supports multi-layer
// aggregation, GQA, hybrid SSM memory, and quantized KV types.
//
// Reference: KVPress (github.com/NVIDIA/kvpress), ExpectedAttentionPress

#include "llama.h"

#include <vector>

struct llama_kv_compress_params {
    float compression_ratio = 0.50f;  // fraction of KV entries to REMOVE
    int   n_future          = 128;    // future positions for RoPE averaging
    int   n_sink            = 4;      // sink tokens to protect (never evicted)
    int   n_recent          = 0;      // trailing positions to protect (never evicted); 0 = legacy behaviour. Without it the
                                      // last prompt position can be evicted and the next decode fails the "consecutive
                                      // positions" check (seen with a 5x ratio on a 3K prompt).
    bool  use_covariance    = true;   // include covariance term (more accurate, slower)

    // EA-POOL-SCREEN (host-side score post-processing before the eviction threshold; all default off):
    int      pool_kernel = 1;         // >1: max-pool the EA scores over a centred sequence window of this many positions
    int      chunk_size  = 0;         // >0: ChunkEA, score = mean EA score of fixed chunks (from n_sink), evict whole chunks
    int      rand_mode   = 0;         // 1: random per-token scores, 2: random per-span scores (control arms; ignores EA scores)
    int      rand_span   = 64;        // span length for rand_mode 2
    uint64_t rand_seed   = 20261121;  // fixed seed for the random controls

    // If true, shift remaining positions to close gaps after eviction via seq_add.
    // WARNING: Only use when the caller manages token positions directly (e.g., test
    // binaries). Do NOT use from llama-server — the server's prompt cache tracks
    // positions independently and will desync. Default: false (leave gaps).
    bool  compact_positions = false;

    // Layer-adaptive scoring weights. If empty, uniform weighting (1/n_layers).
    // Length must equal the number of attention layers in the KV cache.
    // Higher weight = that layer's scores contribute more to the final ranking.
    // Autopilot control surface: learn per-role weight vectors via NumericSwarm.
    //
    // Example: deep layers matter more for code → weights = [0.5, 0.5, ..., 2.0, 2.0]
    //          early layers matter more for chat → weights = [2.0, 2.0, ..., 0.5, 0.5]
    std::vector<float> layer_weights;
};

// Score all KV positions for a sequence across all attention layers.
// Returns per-position importance scores (higher = more important).
// Sink tokens receive max_score + 1.0 (never evicted).
// Thread-safe: reads KV cache data without modification.
std::vector<float> llama_kv_compress_score(
        llama_context * ctx,
        llama_seq_id    seq_id,
        const llama_kv_compress_params & params);

// Attention Matching L4 "lower bound" selector (uniform subsample + constant beta, NO NNLS/OLS refit):
// keep the first `keep_first` and last `keep_last` positions plus an evenly strided sample of the middle until
// `keep_ratio` of the sequence is kept, evict the rest (gapped, via llama_memory_seq_rm) and set `beta` on every
// kept position through llama_memory_set_beta (consumed only with LLAMA_KV_AM_BETA=1).
// Returns the number of evicted entries, or -1 on error; *n_beta_set (optional) = positions that took a beta.
int llama_kv_compress_uniform_l4(
        llama_context * ctx,
        llama_seq_id    seq_id,
        float           keep_ratio,
        int             keep_first,
        int             keep_last,
        float           beta,
        int           * n_beta_set);

// Score and evict low-importance KV entries in one call.
// Evicts the bottom `compression_ratio` fraction of non-sink entries.
// Returns number of entries evicted, or -1 on error.
int llama_kv_compress_evict(
        llama_context * ctx,
        llama_seq_id    seq_id,
        const llama_kv_compress_params & params);
