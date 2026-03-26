#pragma once

#include "ggml.h"

// Fast Walsh-Hadamard Transform (WHT) for KV cache smoothing.
//
// Applying WHT before quantization distributes outlier magnitudes across all
// dimensions, reducing quantization error.  Because WHT is self-inverse
// (H·H = I when normalized by 1/sqrt(n)), the same transform applied to Q
// and the attention output cancels out exactly.
//
// Reference: ExLlamaV2 Q-cache evaluation demonstrates that Hadamard-smoothed
// Q4 KV matches f16 quality, closing the ~0.2 PPL gap of naive Q4.

// In-place Fast Walsh-Hadamard Transform, normalized by 1/sqrt(n).
// n MUST be a power of 2.
void fwht_inplace(float * data, int n);

// ggml_map_custom1 callback: applies WHT independently to each row (dim0)
// of the input tensor.  Rows are partitioned across threads.
void ggml_hadamard_custom_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * src,
        int ith, int nth, void * userdata);
