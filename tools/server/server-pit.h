#pragma once
// SPEC-PIT.1: per-token masses for a randomized probability integral transform (PIT).
//
// For a sampled token x with post-chain probability p_x, the PIT value
//   u = less + U * tie,   U ~ Uniform(0,1)
// is Uniform(0,1) exactly when x was drawn from the post-chain distribution. "less" is the mass of
// candidates strictly less probable than x, "tie" the mass of candidates with exactly p_x (x included).
// Both are normalised by the total candidate mass, so the result does not depend on whether the
// sampler left the candidate array normalised. Computed over the FULL post-chain candidate array, not
// just the top n_probs, so it stays exact when the support (top_p = 1, top_k = 0) is the whole vocab.
//
// Returns false (and leaves the outputs at -1) when x is not among the candidates or has p == 0, which
// can only happen if the sampled token is outside the target support -- a gate failure by itself.

#include "llama.h"

#include <cstddef>

static inline bool server_pit_masses(const llama_token_data * cands, size_t n, llama_token tok, float & less, float & tie) {
    less = -1.0f;
    tie  = -1.0f;
    float px = -1.0f;
    for (size_t i = 0; i < n; ++i) {
        if (cands[i].id == tok) {
            px = cands[i].p;
            break;
        }
    }
    if (!(px > 0.0f)) {
        return false;
    }
    double total = 0.0, l = 0.0, t = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float p = cands[i].p;
        if (!(p > 0.0f)) {
            continue;
        }
        total += p;
        if (p < px) {
            l += p;
        } else if (p == px) {
            t += p;
        }
    }
    if (!(total > 0.0)) {
        return false;
    }
    less = (float) (l / total);
    tie  = (float) (t / total);
    return true;
}
