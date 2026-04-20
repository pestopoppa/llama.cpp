#pragma once

// TIDE (Token-level Intelligent Dynamic Exit) — runtime router evaluation
//
// Loaded at server startup when --tide-router <path> is specified.
// Called after each slot decode to adaptively set n_layer_exit for the next token.
//
// Flow:
//   1. Server starts with tide_router loaded from .tide.bin
//   2. First N tokens decode at full layers (warmup)
//   3. After each decode, extract embedding (hidden state at exit layer)
//   4. Evaluate router MLP: Linear(n_embd, 128) -> ReLU -> Linear(128, 1) -> Sigmoid
//   5. If sigmoid > threshold: keep/reduce n_layer_exit
//   6. If sigmoid < threshold: increase n_layer_exit (needs more computation)

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdint>

struct tide_router_weights {
    int32_t n_embd;        // input dimension (model hidden size)
    int32_t n_hidden;      // router hidden dimension (typically 128)
    int32_t n_checkpoints; // number of checkpoint layers with routers
    std::vector<int32_t> checkpoint_layers; // which layers have routers

    // Weights for each checkpoint router: w1[n_hidden x n_embd], b1[n_hidden], w2[1 x n_hidden], b2[1]
    struct router_mlp {
        std::vector<float> w1; // [n_hidden * n_embd]
        std::vector<float> b1; // [n_hidden]
        std::vector<float> w2; // [n_hidden]
        float b2;
    };
    std::vector<router_mlp> routers; // one per checkpoint
};

struct tide_state {
    tide_router_weights weights;
    float threshold;          // confidence threshold for early exit (e.g., 0.9)
    int32_t warmup_tokens;    // decode this many at full layers before enabling TIDE
    int32_t n_layer_total;    // total model layers

    // Per-slot runtime state
    int32_t current_exit_layer; // current n_layer_exit for this slot (0 = full)
    int32_t tokens_decoded;     // count for warmup tracking
    int32_t consecutive_exit;   // how many consecutive tokens router said "exit"
    int32_t consecutive_full;   // how many consecutive tokens router said "need more"

    bool loaded = false;
};

// Evaluate a single router MLP on a hidden state vector
// Returns sigmoid output (0-1, higher = more converged)
inline float tide_evaluate_router(
        const tide_router_weights::router_mlp & router,
        const float * hidden_state,
        int32_t n_embd,
        int32_t n_hidden) {

    // Layer 1: w1 * x + b1, then ReLU
    std::vector<float> h(n_hidden);
    for (int i = 0; i < n_hidden; i++) {
        float sum = router.b1[i];
        for (int j = 0; j < n_embd; j++) {
            sum += router.w1[i * n_embd + j] * hidden_state[j];
        }
        h[i] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    // Layer 2: w2 * h + b2, then sigmoid
    float logit = router.b2;
    for (int i = 0; i < n_hidden; i++) {
        logit += router.w2[i] * h[i];
    }

    return 1.0f / (1.0f + std::exp(-logit)); // sigmoid
}

// Find the best (earliest) exit layer where the router is confident
// Returns 0 if no router fires (use all layers)
inline int32_t tide_find_exit_layer(
        const tide_state & state,
        const float * embedding,  // hidden state from current decode
        int32_t current_layer) {   // which layer this embedding is from

    if (!state.loaded || state.tokens_decoded < state.warmup_tokens) {
        return 0; // warmup: use all layers
    }

    // Find which checkpoint router corresponds to current_layer
    int ckpt_idx = -1;
    for (int i = 0; i < (int)state.weights.checkpoint_layers.size(); i++) {
        if (state.weights.checkpoint_layers[i] == current_layer) {
            ckpt_idx = i;
            break;
        }
    }

    if (ckpt_idx < 0) {
        return 0; // no router for this layer
    }

    float confidence = tide_evaluate_router(
        state.weights.routers[ckpt_idx],
        embedding,
        state.weights.n_embd,
        state.weights.n_hidden
    );

    if (confidence > state.threshold) {
        return current_layer; // exit here
    }

    return 0; // need more layers
}

// Update TIDE state after a decode step
// Returns the n_layer_exit to use for the NEXT decode
inline int32_t tide_update(
        tide_state & state,
        const float * embedding,   // hidden state from this decode (at current exit layer)
        int32_t exit_layer_used) { // which layer was used (0 = full model)

    state.tokens_decoded++;

    if (!state.loaded || state.tokens_decoded < state.warmup_tokens) {
        return 0; // still warming up, use full layers
    }

    int32_t actual_layer = (exit_layer_used > 0) ? exit_layer_used : state.n_layer_total;

    // Evaluate router at the current exit point
    int32_t suggested_exit = tide_find_exit_layer(state, embedding, actual_layer);

    if (suggested_exit > 0) {
        state.consecutive_exit++;
        state.consecutive_full = 0;

        // After 3 consecutive "exit" signals, reduce layers
        if (state.consecutive_exit >= 3) {
            state.current_exit_layer = suggested_exit;
            return suggested_exit;
        }
    } else {
        state.consecutive_full++;
        state.consecutive_exit = 0;

        // After 2 consecutive "need more" signals, restore full layers
        if (state.consecutive_full >= 2) {
            state.current_exit_layer = 0;
            return 0;
        }
    }

    // Maintain current state (hysteresis)
    return state.current_exit_layer;
}

// Load router weights from .tide.bin file (PyTorch format via simple binary)
// Returns true if loaded successfully
inline bool tide_load_router(tide_state & state, const std::string & path, int32_t n_layer, float threshold, int32_t warmup) {
    // TODO: implement proper loading from PyTorch .tide.bin format
    // For now, this is a placeholder that enables the infrastructure test
    // The actual loading will parse the torch.save() format from calibrate_tide_router.py

    state.threshold = threshold;
    state.warmup_tokens = warmup;
    state.n_layer_total = n_layer;
    state.current_exit_layer = 0;
    state.tokens_decoded = 0;
    state.consecutive_exit = 0;
    state.consecutive_full = 0;

    // Check if file exists
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "TIDE: router file not found: %s\n", path.c_str());
        return false;
    }
    fclose(f);

    fprintf(stderr, "TIDE: router loaded from %s (threshold=%.2f, warmup=%d tokens)\n",
            path.c_str(), threshold, warmup);
    state.loaded = true;
    return true;
}
