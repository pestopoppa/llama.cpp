#pragma once

#include "ggml.h"

#include <cstdint>

// The fused decode path has no output tensor of its own: it writes one row of
// logits into the previous graph's t_logits. That tensor is only usable if it
// holds at least one row. A previous ubatch with no outputs builds it with
// ne[1] == 0 (ggml_get_rows over an empty out_ids), so it has 0 bytes and the
// write and the later extraction would run out of bounds.
inline bool llama_fused_decode_prev_logits_ok(const ggml_tensor * t_logits, int64_t n_vocab) {
    return t_logits != nullptr && t_logits->data != nullptr &&
           t_logits->ne[0] == n_vocab && t_logits->ne[1] >= 1;
}
