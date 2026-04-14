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
    bool  use_covariance    = true;   // include covariance term (more accurate, slower)

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

// Score and evict low-importance KV entries in one call.
// Evicts the bottom `compression_ratio` fraction of non-sink entries.
// Returns number of entries evicted, or -1 on error.
int llama_kv_compress_evict(
        llama_context * ctx,
        llama_seq_id    seq_id,
        const llama_kv_compress_params & params);
